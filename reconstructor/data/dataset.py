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
import pickle
import sys
import torch
import xml.etree.ElementTree as etree
from torch.utils.data import Dataset
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

sys.path.append("..")
from helper.utils.math import complex_abs, ifft2c
import helper.utils.transforms as T


class ReconstructorSample(NamedTuple):
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    center_mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    metadata: dict
    fn: str
    slice_idx: int
    max_value: float
    crop_size: Tuple[int, int]


class ReconstructorDataset(Dataset):
    """Dataset for training accelerated MRI reconstruction models."""

    def __init__(
        self,
        data_path: str,
        transform: Callable,
        seed: Optional[int] = None,
        fast_dev_run: bool = False,
        multicoil: bool = False,
        num_gpus: int = 0,
        dataset_cache_file: Optional[str] = None
    ):
        """
        Args:
            data_path: a string path to the input dataset (or data file).
            transform: an optional callable object that pre-processes the raw
                data into the appropriate form.
            seed: optional random seed for determining order of dataset.
            fast_dev_run: whether we are running a test fast_dev_run.
            multicoil: whether we are using multicoil data or not.
            num_gpus: number of GPUs used for training.
            dataset_cache_file: optional dataset cache file to use for faster
                load times.
        """
        super().__init__()
        self.transform = transform
        self.fast_dev_run = fast_dev_run
        self.rng = np.random.RandomState(seed)
        self.data_path = data_path
        self.dataset_cache_file = dataset_cache_file
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
                with open(cache_path, "w+b") as f:
                    pickle.dump(cache, f)
                print(f"Saved dataset cache to {os.path.abspath(cache_path)}.")
        self.rng.shuffle(self.data)

        if num_gpus > 1:
            even_length = (len(self.data) // num_gpus) * num_gpus
            self.data = self.data[:even_length]

        # Reconstruction target, per fastMRI dataset documentation.
        self.recons_key = "reconstruction_esc"
        if multicoil:
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

    def __getitem__(self, idx: int) -> ReconstructorSample:
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

    def visualize(
        self,
        idx: int,
        key: str = "kspace",
        coils: Union[list, int] = None,
        cmap: str = None,
    ) -> None:
        """
        Visualize data for a particular slice of a scan.
        Input:
            idx: index of desired dataset.
            key: what to visualize, one of ["kspace", "mask", "target",
                "imgspace"].
            coils: which coil(s) to visualize. If not provided, we visualize
                the combined image if key == "imgspace", otherwise visualizes
                all coils.
            cmap: optional color map.
        Returns:
            None.
        """
        item = self.__getitem__(idx)

        data = item.masked_kspace
        if key == "mask":
            if item.mask is None:
                return
            data = item.mask
        elif key == "target":
            if item.target is None:
                return
            data = item.target
        elif key not in ["kspace", "mask", "target", "imgspace"]:
            raise RuntimeError("Unrecognized key.")

        if key == "target":
            coils = [-1]
        if coils is None or len(coils) == 0:
            if key == "kspace":
                coils = range(data.size()[0])
            elif key in ["mask", "imgspace"]:
                coils = [-1]

        if isinstance(coils, int):
            coils = [coils]

        if cmap is None and key in ["target", "imgspace"]:
            cmap = "gray"
        fig, ax = plt.subplots(1, len(coils), sharey="row")
        if len(coils) == 1:
            ax = np.array([ax])
        ax_count = 0
        eps = np.finfo(np.float32).eps
        title = str(idx) + ", " + str(key)
        if coils != [-1]:
            title += ", coils=" + str(coils)
        for coil in coils:
            if key == "kspace":
                ax[ax_count].imshow(
                    torch.log(complex_abs(data[coil]) + eps),
                    cmap=cmap
                )
            elif key == "target":
                ax[ax_count].imshow(data, cmap=cmap)
            elif key == "mask":
                ax[ax_count].imshow(data, cmap=cmap)
            else:
                ax[ax_count].imshow(
                    torch.sqrt(
                        complex_abs(ifft2c(data[[coil], :, :, :])).sum(0)
                    ),
                    cmap=cmap,
                )
            ax_count += 1
        fig.suptitle(title)
        plt.show()

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
