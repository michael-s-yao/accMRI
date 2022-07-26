"""
Defines the data loader for training the k-space discriminator model.
Portions of this code were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import h5py
from fastmri.data.mri_data import et_query
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import time
import torch
from torchvision.transforms import CenterCrop
import xml.etree.ElementTree as etree
from torch.utils.data import Dataset
from typing import (
    Callable, Dict, NamedTuple, Optional, Tuple, Union
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
    uncompressed_ref_kspace: torch.Tensor
    uncompressed_distorted_kspace: torch.Tensor
    target: torch.Tensor
    max_value: float


class DiscriminatorDataset(Dataset):
    """Dataset for training k-space discriminator models."""

    def __init__(
        self,
        data_path: str,
        transform: Callable,
        seed: Optional[int] = None,
        fast_dev_run: bool = False,
        num_gpus: int = 0,
        dataset_cache_file: Optional[str] = None,
        is_mlp: bool = True,
        split: Optional[str] = None,
        center_crop: Optional[Tuple[int]] = (-1, -1)
    ):
        """
        Args:
            data_path: a string path to the input dataset (or data file).
            transform: an image transform module.
            seed: optional random seed for determining order of dataset.
            fast_dev_run: whether we are running a test fast_dev_run.
            num_gpus: number of GPUs used for training.
            dataset_cache_file: optional dataset cache file to use for faster
                load times.
            is_mlp: specify whether discriminator is an MLP network.
            split: specify whether dataset is for validation or testing. Only
                required if is_mlp is True.
            center_crop: kspace center crop dimensions. Default no center crop.
        """
        super().__init__()
        self.transform = transform
        self.fast_dev_run = fast_dev_run
        self.rng = np.random.RandomState(seed)
        self.data_path = data_path
        self.dataset_cache_file = dataset_cache_file
        self.center_crop = center_crop
        if self.dataset_cache_file is None:
            if "multicoil" in data_path.lower():
                cache_path = "multicoil_"
            elif "singlecoil" in data_path.lower():
                cache_path = "singlecoil_"
            else:
                cache_path = "coil_"
            if "knee" in data_path.lower():
                cache_path += "knee_"
            elif "brain" in data_path.lower():
                cache_path += "brain_"
            else:
                cache_path += "anatomy_"
            cache_path += "cache.pkl"
            self.dataset_cache_file = os.path.join(
                os.environ.get("AMLT_OUTPUT_DIR", "./"),
                cache_path
            )

        if os.path.isfile(self.dataset_cache_file):
            with open(self.dataset_cache_file, "rb") as f:
                cache = pickle.load(f)
        else:
            cache = {}
        self.data = cache.get(os.path.basename(self.data_path), [])
        self.fns = None
        if len(self.data) > 0:
            print(f"Using data cache file {self.dataset_cache_file}.")
        else:
            if os.path.isdir(self.data_path):
                data = os.listdir(self.data_path)
                fns = []
                for f in data:
                    if os.path.isfile(os.path.join(self.data_path, f)):
                        fns.append(f)
            elif os.path.isfile(self.data_path):
                fns = [self.data_path]
            else:
                raise ValueError(f"{self.data_path} is not a valid data path.")
            self.fns = fns
            if self.fast_dev_run:
                self.fns = self.fns[:16]

            for fn in sorted(self.fns):
                if not fn.endswith(".h5"):
                    continue
                metadata, num_slices = self.slice_metadata(fn)
                for slice_idx in range(num_slices):
                    self.data += [(fn, slice_idx, metadata)]

            if not self.fast_dev_run and os.path.isdir(self.data_path):
                cache[os.path.basename(self.data_path)] = self.data
                with open(self.dataset_cache_file, "w+b") as f:
                    pickle.dump(cache, f)
                print(
                    "Saved dataset cache to",
                    f"{os.path.abspath(self.dataset_cache_file)}."
                )
        # For MLP training, split the validation dataset into half for
        # validation and half for testing.
        mid = len(self.data) // 2
        if is_mlp and split is not None and split.lower() == "val":
            self.data[:mid]
        elif is_mlp and split is not None:
            self.data[mid:]
        self.rng.shuffle(self.data)

        if num_gpus > 1:
            even_length = (len(self.data) // num_gpus) * num_gpus
            self.data = self.data[:even_length]

        self.recons_key = "reconstruction_esc"
        if "multicoil" in self.data_path:
            self.recons_key = "reconstruction_rss"

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
            A DiscriminatorSample object.
        """
        fn, slice_idx, metadata = self.data[idx]
        with h5py.File(os.path.join(self.data_path, fn), "r") as hf:
            kspace = T.to_tensor(hf["kspace"][slice_idx])
            # Add a coil dimension for single coil data.
            if kspace.ndim < 4:
                kspace = torch.unsqueeze(kspace, dim=0)
            if self.center_crop[0] > 0 and self.center_crop[-1] > 0:
                kspace = torch.permute(kspace, dims=(0, -1, 1, 2))
                kspace = T.center_crop(kspace, self.center_crop)
                kspace = torch.permute(kspace, dims=(0, 2, 3, 1))

            targ = None
            if self.recons_key in hf:
                targ = hf[self.recons_key][slice_idx]
            targ = torch.unsqueeze(T.to_tensor(targ), dim=0)

            # Update metadata with any additional data associated with file.
            metadata.update(dict(hf.attrs))

        return self.transform(kspace, metadata, fn, slice_idx, targ)

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
