"""
Standalone script for kspace exploration via autocorrelation and auto-
covariance analysis. Portions of this code were taken from the fastMRI
repository at https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time
from typing import Dict, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as etree


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


def slice_metadata(fn: Union[Path, str]) -> Tuple[Dict, int]:
    """
    Retrieves metadata associated with a particular input file. This
    function is equivalent to the _retrieve_metadata() function defined
    for the SliceDataset class in the fastMRI repo.
    Input:
        fn: input data file path.
    Returns:
        Relevant metadata and total number of slices in the dataset.
    """
    with h5py.File(fn, "r") as hf:
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


def center_crop(img: np.ndarray, shape: Tuple[int]):
    """
    Center crop last two dimensions of img matrix to specified shape.
    Input:
        img: input matrix to crop.
        shape: 2D shape to crop to.
    Return:
        img with the last two dimensions center cropped to specified shape.
    """
    y, x = img.shape
    cX, cY = shape
    if cX > x or cX < 1 or cY > y or cY < 1:
        raise ValueError(
            "Invalid crop size {shape} given image shape {img.shape}."
        )
    startX, startY = (x // 2) - (cX // 2), (y // 2) - (cY // 2)
    endX, endY = startX + cX, startY + cY
    return img[..., startY:endY, startX:endX]


def X(
    path: Union[Path, str],
    crop_size: Tuple[int] = (128, 128),
    max_n: Optional[int] = -1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Import kspace data from path directory into a single X input matrix.
    Input:
        path: a directory of kspace data in .h5 format.
        crop_size: center crop for kspace data of shape HW.
        max_n: optional maximum number of slices to import. Default no max.
        seed: optional random seed.
    Returns:
        A matrix of shape (HW)N, where HW is flattened length of input slices
        and N is the total number of imported slices.
    """
    data = []
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".h5"):
            continue
        metadata, num_slices = slice_metadata(os.path.join(path, fn))
        for slice_idx in range(num_slices):
            data.append((fn, slice_idx, metadata))

    np.random.RandomState(seed).shuffle(data)
    if max_n > 0 and len(data) > max_n:
        data = data[:max_n]

    h, w = crop_size
    x = np.zeros((h * w, len(data)), dtype=np.complex64)
    for i in range(len(data)):
        fn, slice_idx, _ = data[i]
        with h5py.File(os.path.join(path, fn), "r") as hf:
            kspace = hf["kspace"][slice_idx]
            x[:, i] = np.reshape(center_crop(kspace, crop_size), (h * w,))
    return x


def autocorr(X: np.ndarray) -> np.ndarray:
    """
    Compute the autocorrelation matrix given an input feature matrix.
    Input:
        X: input feature matrix of shape MN, where M is the number of
            features and N is the total number of samples.
    Returns:
        RXX = (XX.T) / (N - 1), an unbiased estimate of the autocorrelation.
    """
    _, n = X.shape
    return np.divide(X @ np.conj(X.T), n - 1)


def autocovar(X: np.ndarray) -> np.ndarray:
    """
    Compute the covariance matrix given an input feature matrix.
    Input:
        X: input feature matrix of shape MN, where M is the number of
            features and N is the total number of samples.
    Returns:
        KXX = (XX.T) / (N - 1), an unbiased estimate of the auto-covariance.
    """
    _, n = X.shape
    mu = np.mean(X, axis=-1)[..., np.newaxis]
    KXX = autocorr(X) - ((n / (n - 1)) * (mu @ np.conj(mu.T)))
    # Computational error may result in nonreal values along the diagonal, so
    # explicitly make the covariance matrix Hermitian positive-semidefinite.
    return KXX - 1j * np.diagflat(np.diag(KXX)).imag


def heatmap(
    auto_matrix: np.ndarray,
    col_idx: int,
    crop_size: Tuple[int],
    dim_reduction_op: str = "mean"
) -> np.ndarray:
    """
    Compute the covariance matrix map for a particular kspace line.
    Input:
        auto_matrix: MxM matrix, where M is the number of features.
        col_idx: kspace line index to plot the covariance heat map with
            respect to.
        crop_size: original crop size (H, W) of initial features.
        dim_reduction_op: operation to reduce covariance matrix dimensions from
            each kspace point to kspace lines. One of (`max`, `mean`,
            `median`).
    Returns:
        Covariance matrix map for the specified kspace line of shape HW.
    """
    h, w = crop_size
    col_idx = col_idx % w
    heatmap = auto_matrix[
        np.where(np.arange(auto_matrix.shape[0]) % w == col_idx)[0], :
    ]
    heatmap = np.reshape(heatmap, (heatmap.shape[0], h, w))
    heatmap = np.sum(heatmap, axis=1)
    if dim_reduction_op.lower() == "max":
        heatmap = np.max(heatmap, axis=0)
    elif dim_reduction_op.lower() in ["mean", "avg"]:
        heatmap = np.mean(heatmap, axis=0)
    elif dim_reduction_op.lower() in ["median"]:
        heatmap = np.median(heatmap, axis=0)
    else:
        raise ValueError(
            f"Unrecognized dimension reduction operation {dim_reduction_op}."
        )
    return np.stack((heatmap,) * h, axis=0)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fastMRI data explorer")

    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="A directory of kspace data in .h5 format."
    )
    parser.add_argument(
        "--center_crop",
        type=int,
        nargs=2,
        default=(128, 128),
        required=True,
        help=""
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=-1,
        help="Optional maximum number of slices to import. Default no max."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(time.time()),
        help="Optional random seed. Default is seconds since epoch."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="./autocorr",
        help="Directory to save autocorrelation maps. Default ./autocorr."
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=None,
        help="matplotlib color map for plotting. Default `viridis`."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()

    RXX = autocorr(X(args.data_path, args.center_crop, args.max_n, args.seed))
    if args.savepath is not None:
        for i in range(args.center_crop[-1]):
            plt.gca()
            plt.imshow(
                np.abs(heatmap(RXX, i, crop_size=args.center_crop)),
                cmap=args.cmap
            )
            plt.colorbar()
            plt.savefig(
                os.path.join(args.savepath, f"{i}_mag.png"),
                bbox_inches="tight"
            )
            plt.close()
