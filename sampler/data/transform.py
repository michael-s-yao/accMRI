"""
Defines the data transformer for accelerated MRI active kspace sampling.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import numpy as np
import sys
import torch
from typing import Optional, Sequence, Tuple, Union
from data.dataset import PolicySample

sys.path.append("..")
import helper.utils.transforms as T
from reconstructor.data.transform import ReconstructorDataTransform


class PolicyDataTransform(ReconstructorDataTransform):
    """
    Data transformer for accelerated MRI active kspace sampling. Randomly masks
    data according to specifications if no mask is provided.
    """

    def __init__(
        self,
        center_fractions: Optional[Sequence[float]] = [0.08, 0.04, 0.00],
        center_crop: Optional[Union[torch.Size, tuple]] = None,
        min_lines_acquired: Optional[int] = 0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            center_crop: an optional tuple of shape HW to crop the input kspace
                size to.
            min_lines_acquired: minimum number of total lines acquired.
            seed: optional random seed.
        """
        super().__init__(
            center_fractions, center_crop, min_lines_acquired, seed
        )

    def __call__(
        self,
        kspace: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[np.ndarray] = None,
        metadata: dict = None,
        fn: str = None,
        slice_idx: int = -1,
    ) -> Policy:
        """
        Mask and process kspace data for active MRI acquisition.
        Input:
            kspace: input kspace data of shape CHW2.
            mask: mask associated with dataset (should be None for training and
                validation datasets).
            target: the target image (if applicable).
            metadata: metadata associated with dataset.
            fn: name of the original data file.
            slice_idx: index of slice from original dataset.
        Returns:
            A PolicySample object.
        """
        data = super()(kspace, mask, target, metadata, fn, slice_idx)
        # TODO: Incorporate when working on random delays.
        action_buffer = torch.tensor([])

        return PolicySample(
            kspace,
            data.mask,
            action_buffer,
            data.target,
            metadata,
            data.fn,
            data.slice_idx,
            data.crop_size,
        )
