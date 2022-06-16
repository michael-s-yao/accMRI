"""
Defines the data loader for training accelerated MRI reconstruction models.
Portions of this code were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import xml.etree.ElementTree as etree
from torch.utils.data import Dataset
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

sys.path.append("..")
from helper.utils.math import complex_abs, ifft2c
import helper.utils.transforms as T


class PolicySample(NamedTuple):
    ref_kspace: Optional[torch.Tensor]
    acquired_mask: torch.Tensor
    action_buffer: torch.Tensor
    target: torch.Tensor
    metadata: dict
    fn: str
    slice_idx: int
    crop_size: Tuple[int, int]


class Policy(Dataset):
    """Dataset for training accelerated MRI reconstruction models."""

    def __init__(
        self,
        data_path: str,
        reconstructor: Callable,
        seed: Optional[int] = None
    ):
        """
        Args:
            data_path: a string path to the multicoil dataset.
            transform: an optional callable object that pre-processes the raw
                data into the appropriate form.
            seed: optional random seed for determining order of dataset.
        """
        super().__init__()

        data = os.listdir(data_path)
        fns = [f for f in data if os.path.isfile(os.path.join(data_path, f))]
        self.data_path = data_path
        self.fns = fns
        self.data = []
        for fn in sorted(self.fns):
            if not fn.endswith(".h5"):
                continue
            metadata, num_slices = self.slice_metadata(fn)
            for slice_idx in range(num_slices):
                self.data += [(fn, slice_idx, metadata)]
        self.rng = np.random.RandomState(seed)
        self.rng.shuffle(self.data)

        # Reconstruction target, per fastMRI dataset documentation.
        self.recons_key = "reconstruction_rss"

        self.reconstructor = reconstructor

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

    def __getitem__(self, idx: int) -> PolicySample:
        """
        Loads and returns a sample from the dataset at the given index.
        Required for torch.utils.data.Dataset subclass implementation.
        Input:
            idx: index of desired dataset.
        Returns:
            A ReconstructorSample object.
        """
        fn, slice_idx, metadata = self.data[idx]
        with h5py.File(os.path.join(self.data_path, fn), "r") as hf:
            kspace = T.to_tensor(hf["kspace"][slice_idx])

            mask = None
            if "mask" in hf:
                mask = torch.from_numpy(np.asarray(hf["mask"]))

            targ = None
            if self.recons_key in hf:
                targ = hf[self.recons_key][slice_idx]

            # Update metadata with any additional data associated with file.
            metadata.update(dict(hf.attrs))

        return self.transform(kspace, mask, targ, metadata, fn, slice_idx)

    def slice_metadata(self, fn: str) -> Tuple[Dict, int]:
        """
        Retrieves metadata associated with a particular input file. This
        function is equivalent to the _retrieve_metadata() function defined
        for the SliceDataset class in the fastMRI repo.
        Input:
            fn: input data file name.
        Returns:
            padding_left, padding_right, encoding_size, recon_size, num_slices.
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
