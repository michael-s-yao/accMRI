"""
Simulated MRI scanner for kspace data acquisition.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
from multiprocessing import Queue
import numpy as np
import sys
import time
import torch
from typing import NamedTuple, Optional, Sequence, Union

sys.path.append("..")
from common.utils.math import fft2c


class KSpaceData(NamedTuple):
    kspace: torch.Tensor
    mask: torch.Tensor
    line_idxs: int


class MRIScanner:
    def __init__(
        self,
        shape: tuple = (128, 128,),
        ellipses: Optional[Union[np.ndarray, torch.Tensor]] = None,
        modified_sl: bool = True,
        num_lines: Optional[int] = None,
        data_acq_mean: float = 10,
        data_acq_std: float = 2,
        no_requests_latency: int = 1,
        seed: Optional[int] = None,
        debug: bool = False,
    ):
        """
        Args:
            shape: 2D size of the output phantom image. The image coordinate
                system is always bounded by (-1, -1) and (1, 1).
            ellipses: a custom set of ellipses to use. If none, defaults to
                Shepp-Logan. This argument is an Nx6 array for N ellipses,
                where each ellipse is represented as a row [I, a, b, x0, y0,
                phi].
                    I: additive intensity of the ellipse.
                    a: length of the major axis.
                    b: length of the minor axis.
                    x0: horizontal offset of the center of the ellipse.
                    y0: vertical offset of the center of the ellipse.
                    phi: counterclockwise rotation of the ellipse in degrees,
                        measured as the angle between the horizontal axis and
                        te ellipse major axis.
            modified_sl: whether to use modified Shepp-Logan. Ignored if
                ellipses is not None.
            num_lines: total number of kspace lines to generate. If None, then
                the upper bound is set to 1e12.
            data_acq_mean: average length of time per single kspace line
                acquisition.
            data_acq_std: standard deviation for length of time per single
                kspace line acquisition. Duration is sampled from a normal
                N(data_acq_mean, data_acq_std) distribution (in seconds).
            no_requests_latency: if no sampling requests are queued, the
                program waits the specified number of seconds before querying
                the requests queue again.
            seed: optional random seed.
            debug: whether to print debugging statements or not.
        """
        if seed is not None:
            self.seed = seed
        else:
            self.seed = int(time.time())
        self.rng = np.random.RandomState(self.seed)

        if len(shape) != 2:
            raise ValueError(
                f"Output phantom shape must be 2D, got {len(shape)}"
            )
        h, w = shape
        self.shape = (1, 1, h, w, 2)
        if num_lines is not None and int(num_lines) > 0:
            self.num_lines = min(num_lines, 1e12)
        else:
            self.num_lines = 1e12
        self.MAX_PIXEL = 255.0
        if ellipses is not None:
            if ellipses.shape[-1] != 6 or ellipses.ndim != 2:
                raise ValueError(
                    f"Invalid ellipses shape {ellipses.shape}, expected Nx6."
                )
            self.ellipses = self.ellipses
        elif modified_sl:
            self.ellipses = self.mod_shepp_logan_default()
        else:
            self.ellipses = self.shepp_logan_default()

        Ilim = [-0.05, 0.05]
        alim = [-0.05, 0.05]
        blim = [-0.05, 0.05]
        x0lim = [-0.08, 0.08]
        y0lim = [-0.08, 0.08]
        philim = [-8, 8]
        limits = np.array([
            Ilim, alim, blim, x0lim, y0lim, philim
        ])

        self.deformations = self.rng.random_sample(
            (self.ellipses.shape[0], self.ellipses.shape[-1],)
        )
        for col in range(self.deformations.shape[-1]):
            self.deformations[..., col] = (
                limits[col, 1] - limits[col, 0]
            ) * self.deformations[..., col] + limits[col, 0]

        self.data_acq_mean = data_acq_mean
        self.data_acq_std = data_acq_std
        self.no_requests_latency = no_requests_latency
        self.debug = debug

    def sample(self, pipeline: Queue, requests: Queue) -> None:
        """
        Return the next requested kspace line.
        Input:
            pipeline: the data pipeline to queue the acquired kspace line.
            requests: the kspace column index queue.
        Returns:
            None. Requested kspace data is added to the end of the pipeline
            queue. Data is of size BCHW2.
        """
        # TODO: Data is currently of size BHW2 -> multicoil support needs
        # to be added. Support for fastMRI dataset also needs to be added.
        kspace = fft2c(self.phantom())
        num_iter = 0
        while num_iter < self.num_lines:
            # Retrieve next line of kspace data to sample.
            if not requests.empty():
                line_idxs = requests.get(block=False)
                if not isinstance(line_idxs, list):
                    line_idxs = [line_idxs]
            else:
                if self.debug:
                    print(
                        "No requests for data in queue, generator waiting " +
                        str(self.no_requests_latency) +
                        " seconds before trying again...",
                        flush=True
                    )
                time.sleep(self.no_requests_latency)
                continue

            # Sample the requested data and place it into the queue.
            data = torch.zeros(self.shape, dtype=torch.float32)
            for col in line_idxs:
                data[..., col, :] = kspace[..., col, :]
            mask = self.mask(line_idxs)
            data_acq_latency = np.sum(self.rng.normal(
                self.data_acq_mean, self.data_acq_std, size=(len(line_idxs),)
            ))
            time.sleep(data_acq_latency)
            if self.debug:
                print(
                    "Iteration " + str(num_iter) + ": Sampling line " +
                    "number(s) " + str(line_idxs) + ", took a total of " +
                    str(data_acq_latency) + " seconds.",
                    flush=True
                )
            pipeline.put(KSpaceData(data, mask, line_idxs))

            num_iter += 1

    def mask(self, line_idxs: Sequence) -> torch.Tensor:
        """
        Generates a 1D sampling mask with the columns specified by line_idxs
        unmasked.
        Input:
            line_idxs: A sequence of kspace lines that were sampled.
        Returns:
            A 1D sampling mask with length equal to the width of the shape of
                the specified kspace.
        """
        kspace_mask = torch.zeros(self.shape[-2], dtype=torch.float32)
        for col in line_idxs:
            kspace_mask[col % self.shape[-2]] = 1.0
        return kspace_mask

    def phantom(self) -> torch.Tensor:
        """
        Generates the phantom image data.
        Input:
            None.
        Returns:
            The single coil complex phantom image. Assumes the imaginary
            component is zero for simplicity. Return shape is 11HW2.
        """
        _, _, h, w, _ = self.shape
        phantom = np.zeros((h, w,))
        jv, iv = np.meshgrid(
            np.linspace(-1, 1, h),
            np.linspace(-1, 1, w)
        )
        for i in range(self.ellipses.shape[0]):
            intensity = self.ellipses[i, 0] + self.deformations[i, 0]
            a2 = np.square(self.ellipses[i, 1] + self.deformations[i, 1])
            b2 = np.square(self.ellipses[i, 2] + self.deformations[i, 2])
            x0 = self.ellipses[i, 3] + self.deformations[i, 3]
            y0 = self.ellipses[i, 4] + self.deformations[i, 4]
            phi = (self.ellipses[i, 5] + self.deformations[i, 5]) * np.pi / 180

            x, y = jv - x0, iv - y0
            locs = np.square((x * np.cos(phi)) + (y * np.sin(phi))) / a2 + (
                np.square((y * np.cos(phi)) - (x * np.sin(phi))) / b2
            ) <= 1

            phantom[locs] += intensity

        phantom = torch.unsqueeze(torch.from_numpy(np.clip(
            phantom * self.MAX_PIXEL, 0, self.MAX_PIXEL
        )), dim=-1)
        return torch.unsqueeze(torch.unsqueeze(torch.cat(
            (phantom, torch.ones_like(phantom),), dim=-1
        ), dim=0), dim=0)

    def shepp_logan_default(self) -> np.ndarray:
        """
        Returns ellipse parameters for the default Shepp-Logan phantom.
        Input:
            None.
        Returns:
            The ellipse parameters for the Shepp-Logan toy phantom.

        Shepp LA and Logan BF. (1974). Reconstructing interior head tissue
        from X-ray transmissions. IEEE Transactions on Nuclear Science.
        """
        return np.array([
            [   2,   .69,   .92,    0,      0,   0],
            [-.98, .6624, .8740,    0, -.0184,   0],
            [-.02, .1100, .3100,  .22,      0, -18],
            [-.02, .1600, .4100, -.22,      0,  18],
            [ .01, .2100, .2500,    0,    .35,   0],
            [ .01, .0460, .0460,    0,     .1,   0],
            [ .02, .0460, .0460,    0,    -.1,   0],
            [ .01, .0460, .0230, -.08,  -.605,   0],
            [ .01, .0230, .0230,    0,  -.606,   0],
            [ .01, .0230, .0460,  .06,  -.605,   0],
        ])

    def mod_shepp_logan_default(self) -> np.ndarray:
        """
        Returns ellipse parameters for the modified Shepp-Logan phantom,
        adjusted to improve constrast.
        Input:
            None.
        Returns:
            The ellipse parameters for the modified Shepp-Logan phantom.

        Toft P. (1996). The radon transform - theory and implementation,
        PhD thesis, Department of Mathematical Modelling, Technical
        University of Denmark.
        """
        return np.array([
            [   1,   .69,   .92,    0,      0,   0],
            [-.80, .6624, .8740,    0, -.0184,   0],
            [-.20, .1100, .3100,  .22,      0, -18],
            [-.20, .1600, .4100, -.22,      0,  18],
            [ .10, .2100, .2500,    0,    .35,   0],
            [ .10, .0460, .0460,    0,     .1,   0],
            [ .10, .0460, .0460,    0,    -.1,   0],
            [ .10, .0460, .0230, -.08,  -.605,   0],
            [ .10, .0230, .0230,    0,  -.606,   0],
            [ .10, .0230, .0460,  .06,  -.605,   0],
        ])
