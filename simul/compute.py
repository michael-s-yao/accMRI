"""
Simulated compute cluster for accelerated MRI data processing and control.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import matplotlib.pyplot as plt
from multiprocessing import Queue
import numpy as np
import time
import torch
from typing import Optional
from scanner import KSpaceData


class ComputeCluster:
    def __init__(
        self,
        shape: tuple = (128, 128,),
        num_coils: int = 1,
        num_lines: Optional[int] = None,
        no_data_latency: int = 1,
        min_recon_latency: Optional[int] = None,
        seed: Optional[int] = None,
        savefig: bool = True,
        debug: bool = False
    ):
        """
        Args:
            shape: 2D size of the output phantom image. The image coordinate
                system is always bounded by (-1, -1) and (1, 1).
            num_coils: number of coils, default 1.
            num_lines: total number of kspace lines to generate. If None, then
                the upper bound is set to 1e12.
            no_data_latency: if no kspace lines are available, the
                program waits the specified number of seconds before querying
                again.
            min_recon_latency: optional minimum number of seconds to spend on
                the reconstruction step.
            seed: optional random seed.
            savefig: whether to save the output reconstructions.
            debug: whether to print debugging statements or not.
        """
        self.rng = np.random.RandomState(seed)
        self.savefig = savefig
        if self.savefig:
            plt.figure()
        self.debug = debug
        if num_lines is not None and int(num_lines) > 0:
            self.num_lines = min(num_lines, 1e12)
        else:
            self.num_lines = 1e12

        self.acquired_lines = []
        self.no_data_latency = no_data_latency
        self.min_recon_latency = min_recon_latency
        h, w = shape
        self.shape = (1, num_coils, h, w, 2)

        self.reconstructor = None  # TODO

        self.kspace = torch.zeros(self.shape)

    def reconstruct(self, pipeline: Queue, requests: Queue) -> None:
        """
        Reconstructs the currently available data and inputs new requests for
        future data acquisition.
        Input:
            pipeline: the data pipeline of acquired kspace lines.
            requests: the kspace column index queue.
        Returns:
            None. Requested kspace data is added to the end of  the pipeline
            queue. Data is of size BCHW2.
        """
        num_iter = 0
        while num_iter < self.num_lines:
            if num_iter > 0:
                # No point in continuing if no data available.
                if not pipeline.empty():
                    kspace_data = pipeline.get(block=False)
                else:
                    if self.debug:
                        print(
                            "No data available yet, reconstructor " +
                            "waiting on data..."
                        )
                    time.sleep(self.no_data_latency)
                    continue
            else:
                kspace_data = KSpaceData(
                    torch.zeros(self.shape),
                    torch.zeros(self.shape[-2]),
                    []
                )

            t0 = time.time()
            # Convert mask from 1D to 2D (BHW).
            mask = torch.unsqueeze(torch.cat(
                (torch.unsqueeze(kspace_data.mask, 0),) * self.shape[-3],
                dim=0
            ), dim=0)
            self.kspace = torch.where(
                kspace_data.kspace != 0.0,
                kspace_data.kspace,
                self.kspace
            )
            self.acquired_lines += list(kspace_data.line_idxs)
            reconstruction, next_sample = self.reconstructor(self.kspace, mask)
            # Convert mask from 2D to 1D.
            # Assume that we're only working with a batch size of 1.
            # Compress to a single column dimension.
            next_sample = torch.sum(next_sample, dim=(0, 1))
            next_lines = []
            for col in range(next_sample.size()[-1]):
                if next_sample[col] > 0.0:
                    next_lines.append(col)
            dt = time.time() - t0
            if (self.min_recon_latency is not None and
                    dt < self.min_recon_latency):
                time.sleep(max(self.min_recon_latency - dt, 0.001))
            if self.debug:
                print(
                    "Iteration " + str(num_iter) + ": Reconstruction step " +
                    "took a total of " + str(max(dt, self.min_recon_latency)) +
                    " seconds.",
                    flush=True
                )
            # If no lines indicated to sample, then just sample random lines.
            if len(next_lines) == 0:
                print(
                    "Iteration " + str(num_iter) + ": No lines requested " +
                    "by reconstructor, choosing random lines to sample."
                )
                num_lines_to_sample = 8
                next_lines = list(self.rng.choice(
                    self.shape[-2], size=num_lines_to_sample
                ))

            requests.put(next_lines)

            if self.savefig:
                plt.gca()
                plt.cla()
                plt.imshow(
                    reconstruction[0].detach().cpu().numpy(), cmap="gray"
                )
                plt.savefig("output" + str(num_iter) + ".png")

            num_iter += 1
