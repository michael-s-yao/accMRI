"""
Defines the data loader for training the k-space discriminator model.
Portions of this code were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch
from torchvision.transforms import CenterCrop
import xml.etree.ElementTree as etree
from torch.utils.data import Dataset
from typing import (
    Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union
)

sys.path.append("..")
from helper.utils import transforms as T
from helper.utils.math import ifft2c, rss_complex


class DiscriminatorSample(NamedTuple):
    ref_kspace: torch.Tensor
    distorted_kspace: torch.Tensor
    theta: float
    dx: float
    dy: float
    sampled_mask: torch.Tensor
    acquiring_mask: torch.Tensor
    metadata: dict
    fn: str
    slice_idx: int


class DiscriminatorDataset(Dataset):
    """Dataset for training k-space discriminator models."""

    def __init__(
        self,
        data_path: str,
        transform: Callable,
        seed: Optional[int] = None,
        fast_dev_run: bool = False,
        num_gpus: int = 0
    ):
        """
        Args:
            data_path: a string path to the input dataset (or data file).
            transform: an image transform module.
            seed: optional random seed for determining order of dataset.
            fast_dev_run: whether we are running a test fast_dev_run.
            num_gpus: number of GPUs used for training.
        """
        super().__init__()
        self.transform = transform
        self.fast_dev_run = fast_dev_run

        if os.path.isdir(data_path):
            data = os.listdir(data_path)
            fns = [
                f for f in data if os.path.isfile(os.path.join(data_path, f))
            ]
        elif os.path.isfile(data_path):
            fns = [data_path]
        else:
            raise ValueError(f"{data_path} is not a valid data path.")
        self.data_path = data_path
        self.fns = fns
        if self.fast_dev_run:
            self.fns = self.fns[:16]
        self.data = []
        for fn in sorted(self.fns):
            if not fn.endswith(".h5"):
                continue
            metadata, num_slices = self.slice_metadata(fn)
            for slice_idx in range(num_slices):
                self.data += [(fn, slice_idx, metadata)]
        self.rng = np.random.RandomState(seed)
        self.rng.shuffle(self.data)

        if num_gpus > 1:
            even_length = (len(self.data) // num_gpus) * num_gpus
            self.data = self.data[:even_length]

    def __len__(self) -> int:
        """
        Returns the number of samples in our dataset. Required for
        torch.utils.data.Dataset subclass implementation.
        Input:
            None.
        Returns:
            Number of samples in our dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> DiscriminatorSample:
        """
        Loads and returns a sample from the dataset at the given index.
        Required for torch.utils.data.Dataset subclass implementation.
        Input:
            idx: index of desired dataset.
        Returns:
            A VarNetSample object (for training and validation datasets) or
            a TestSample object (for test datasets).
        """
        fn, slice_idx, metadata = self.data[idx]
        with h5py.File(os.path.join(self.data_path, fn), "r") as hf:
            kspace = T.to_tensor(hf["kspace"][slice_idx])
            # Add a coil dimension for single coil data.
            if kspace.ndim < 4:
                kspace = torch.unsqueeze(kspace, dim=0)
            # Update metadata with any additional data associated with file.
            metadata.update(dict(hf.attrs))

        return self.transform(kspace, metadata, fn, slice_idx)

    def slice_metadata(self, fn: str) -> Tuple[Dict, int]:
        """
        Retrieves metadata associated with a particular input file. This
        function is equivalent to the _retrieve_metadata() function defined
        for the SliceDataset class in the fastMRI repo.
        Input:
            fn: input data file name.
        Returns:
            padding_left, padding_right, encoding_size, recon_size.
        """
        with h5py.File(os.path.join(self.data_path, fn), "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            recon = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, recon + ["x"])),
                int(et_query(et_root, recon + ["y"])),
                int(et_query(et_root, recon + ["z"])),
            )

            limits = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, limits + ["center"]))
            enc_limits_max = int(et_query(et_root, limits + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def debug_plots(
        self,
        idx: Optional[int] = None,
        center_crop: Optional[Union[torch.Size, tuple]] = None,
        savefig: bool = False,
        seed: Optional[int] = None
    ) -> None:
        """
        Generate debug plots to visualize the dataset.
        Input:
            idx: training sample item index to visualize. If None, then a
                random item is chosen out of the dataset to plot.
            center_crop: an optional tuple of shape HW to crop the images to.
            savefig: whether or not to save the debug plot.
            seed: optional random seed.
        Returns:
            None.
        """
        if seed is None or seed < 0:
            seed = int(time.time())
        rng = np.random.RandomState(seed)
        if idx is None or not isinstance(idx, int):
            idx = rng.randint(0, self.__len__())
        idx = idx % self.__len__()

        # Center crop operation.
        if center_crop is None:
            center_crop = (320, 320)
        crop = CenterCrop(
            (min(320, center_crop[0]), min(320, center_crop[-1]))
        )

        item = self.__getitem__(idx)
        fig, axs = plt.subplots(2, 2, figsize=(5, 8))
        eps = torch.finfo(torch.float32).eps
        axs[0, 0].imshow(
            torch.log(rss_complex(item.ref_kspace) + eps), cmap="gray"
        )
        axs[0, 0].set_title("Reference kspace")
        axs[0, 1].imshow(
            torch.log(rss_complex(item.distorted_kspace) + eps), cmap="gray"
        )
        axs[0, 1].set_title("Distorted kspace")
        axs[1, 0].imshow(
            crop(rss_complex(ifft2c(item.ref_kspace))), cmap="gray"
        )
        axs[1, 0].set_title("Reference Target")
        axs[1, 0].axis("off")
        axs[1, 1].imshow(
            crop(rss_complex(ifft2c(item.distorted_kspace))), cmap="gray"
        )
        axs[1, 1].set_title("Distorted Target")
        axs[1, 1].axis("off")

        fig.tight_layout()
        if savefig:
            plt.savefig(
                f"debug_discriminator_{seed}.png",
                dpi=600,
                format="png",
                bbox_inches="tight",
                transparent=True
            )
        else:
            plt.show()


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function. This can be used to query an XML document via
    ElementTree. It uses qlist for nested queries. This function was taken from
    the fastmri.data.mri_data module in the fastMRI repo.
    Input:
        root: root of the XML to search through.
        qlist: A list of strings for nested searches.
        namespace: XML namespace to prepend query.
    Returns:
        The returned data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s += f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Value not found.")

    return str(value.text)
