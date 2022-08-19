"""
Image reconstruction evaluation metrics. Portions of this code were taken
from the fastMRI repository at https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import pathlib
from typing import Dict, Optional

sys.path.append("..")
from reconstructor.models.loss import structural_similarity


def ssim(
    target: torch.Tensor,
    pred: torch.Tensor,
    max_val: Optional[float] = None
) -> torch.Tensor:
    """
    Compute the structural similarity index metric (SSIM) between the target
    and prediction.
    Input:
        target: target ground truth tensor.
        pred: predicted tensor.
        max_val: dynamic range of the pixel values.
    Returns:
        SSIM(target, pred)
    """
    if target.size() != pred.size():
        raise ValueError(
            f"Target {target.size()} and pred {pred.size()} dims do not match."
        )

    max_val = torch.max(target) if max_val is None else max_val

    ssim = 0
    for slice_num in range(target.size()[0]):
        ssim = ssim + structural_similarity(
            target[slice_num, ...].unsqueeze(0),
            pred[slice_num, ...].unsqueeze(0),
            data_range=max_val,
        )

    return torch.tensor([ssim / target.size()[0]])


def save_reconstructions(
    outputs: Dict[str, np.ndarray],
    save_path: pathlib.Path,
    save_png: bool = False,
    max_count: int = 16
):
    """
    Save reconstruction images.
    Input:
        outputs: a dictionary mapping input filenames to corresponding
            reconstructions.
        save_path: pathlib.Path object to the output directory where the
            reconstructions should be saved.
        save_png: whether to save pngs of the outputs as well.
        max_count: maximum number of reconstructions to save. Default 16.
    Returns:
        None.
    """
    save_path.mkdir(exist_ok=True, parents=True)
    if save_png:
        (save_path / "images").mkdir(exist_ok=True, parents=True)

    count = 0
    for fname, recons in outputs.items():
        if count > max_count:
            break
        if fname.endswith(".h5"):
            fname = fname[:-3]
        hf = h5py.File(save_path / (fname + ".h5"), "w")
        hf.create_dataset("reconstruction", data=recons)

        if not save_png:
            continue
        (save_path / "images" / fname).mkdir(exist_ok=True, parents=True)
        for i in range(recons.shape[0]):
            plt.imsave(
                save_path / "images" / fname / (str(i) + ".png"),
                recons[i],
                cmap="gray"
            )
        count += 1
