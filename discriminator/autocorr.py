"""
Standalone script for kspace exploration via autocorrelation and auto-
covariance analysis. Portions of this code were taken from the fastMRI
repository at https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import argparse
from fastmri.data.mri_data import et_query
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
import time
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Union
import xml.etree.ElementTree as etree

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


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
    cache_path: Optional[str] = None,
    cache_key: Optional[str] = "",
) -> np.ndarray:
    """
    Import kspace data from path directory into a single X input matrix.
    Input:
        path: a directory of kspace data in .h5 format.
        crop_size: center crop for kspace data of shape HW.
        max_n: optional maximum number of slices to import. Default no max.
        seed: optional random seed.
        cache_path: optional path to dataset cache file to use.
        cache_key: optional key to specify which list to import from cache
            file.
    Returns:
        A matrix of shape (HW)N, where HW is flattened length of input slices
        and N is the total number of imported slices.
    """
    data = []
    if cache_path is not None and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        data = cache.get(cache_key, [])
    if len(data) > 0:
        print(f"Using cache file {cache_path}.")
        good_data = []
        for item in data:
            good_data.append(
                (os.path.join(path, item[0].stem + ".h5"),) + item[1:]
            )
        data = good_data
    else:
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
        help="kspace center crop dimensions. Default 128 x 128."
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
        help="Directory to save autocorrelation maps and pickled dataset."
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=None,
        help="matplotlib color map for plotting. Default `viridis`."
    )
    parser.add_argument(
        "--plotval",
        type=str,
        choices=("magnitude", "phase", "both"),
        default="both",
        help="Plot either the magnitude and/or phase of generated heatmaps."
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Optional dataset cache file."
    )
    parser.add_argument(
        "--cache_key",
        type=str,
        default=None,
        help="Optional cache key specification. Default is data_path value."
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="Flag to save autocorrelation map plots."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    scale_factor = int(1e10)
    cache_key = args.cache_key
    if cache_key is None:
        cache_key = args.data_path

    RXX = autocorr(
        X(
            args.data_path,
            args.center_crop,
            args.max_n,
            args.seed,
            args.cache_path,
            cache_key
        )
    )
    if args.savepath.lower() != "none":
        if not os.path.isdir(args.savepath):
            os.mkdir(args.savepath)
        correlation_maps = []
        for i in tqdm(range(args.center_crop[-1])):
            correlation_maps.append(
                np.abs(
                    heatmap(RXX, i, crop_size=args.center_crop)[0, :]
                ).tolist()
            )
            if not args.make_plots:
                continue
            plt.gca()
            if args.plotval.lower() in ["phase", "both"]:
                plt.imshow(
                    scale_factor * np.angle(
                        heatmap(RXX, i, crop_size=args.center_crop)
                    ),
                    cmap=args.cmap
                )
            if args.plotval.lower() in ["magnitude", "both"]:
                plt.imshow(
                    scale_factor * np.abs(
                        heatmap(RXX, i, crop_size=args.center_crop)
                    ),
                    cmap=args.cmap
                )
            plt.colorbar(orientation="horizontal", pad=0.01, fraction=0.047)
            plt.axis("off")
            if args.plotval.lower() == "phase":
                plt.savefig(
                    os.path.join(args.savepath, f"{i}_phase.png"),
                    bbox_inches="tight",
                    dpi=600
                )
            else:
                plt.savefig(
                    os.path.join(args.savepath, f"{i}_mag.png"),
                    bbox_inches="tight",
                    dpi=600
                )
            plt.close()

            correlation_maps_path = os.path.abspath(
                os.path.join(args.savepath, "correlation_maps.pkl")
            )
        with open(correlation_maps_path, "w+b") as f:
            pickle.dump(correlation_maps, f)
            print(
                "Saved ground truth correlation maps to",
                correlation_maps_path
            )
