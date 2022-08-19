"""
Baseline, non-learning-based discriminator algorithm for k-space corruption
detection based on cross-correlation between lines of k-space.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import numpy as np
import pickle
from pathlib import Path
from scipy.signal import correlate
from typing import Sequence, Union


class BaselineDiscriminator:
    def __init__(
        self, gt_correlation_map: Union[str, Path], dim_reduction: str = "mean"
    ):
        """
        Args:
            gt_correlation_map: filepath to ground truth k-space correlation
                map to use to normalize values from discriminator (generated
                by autocorr.py).
            dim_reduction: function type to reduce correlation value dimension
                to a single value per kspace line. One of `mean`, `max`, `sum`.
        """
        with open(gt_correlation_map, "rb") as f:
            self.gt_correlation_map = np.array(pickle.load(f))
        if dim_reduction.lower() == "mean":
            self.dim_reduction = np.mean
        elif dim_reduction.lower() == "max":
            self.dim_reduction = np.max
        elif dim_reduction.lower() == "sum":
            self.dim_reduction = np.sum
        else:
            raise ValueError(
                f"Unrecognized dimension reduction operator {dim_reduction}."
            )

    def __call__(
        self, kspace: np.ndarray, hf_idx: int, acs_idxs: Sequence[int]
    ) -> float:
        """
        Calculates the normalized signal cross-correlation of a high-frequency
        line averaged over all of the ACS lines.
        Input:
            kspace: input complex kspace array of shape CHW2.
            hf_idx: index of high-frequency kspace line to calculate the
                discriminator score for.
            acs_idx: list of indices of the ACS lines.
        Returns:
            Estimated discriminator score for the hf_idx line.
        """
        hf_signal = np.squeeze(
            kspace[..., hf_idx, 0] + (1j * kspace[..., hf_idx, 1])
        )
        score = 0
        for acs_loc in acs_idxs:
            lf_signal = np.squeeze(
                kspace[..., acs_loc, 0] + (1j * kspace[..., acs_loc, 1])
            )
            score += np.divide(
                self.dim_reduction(np.abs(correlate(hf_signal, lf_signal))),
                self.gt_correlation_map[hf_idx, acs_loc]
            )
        return score / float(len(acs_idxs))
