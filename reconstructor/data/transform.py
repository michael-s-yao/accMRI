"""
Defines the data transformer for accelerated MRI reconstruction.
Portions of this code were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import numpy as np
import torch
from typing import Optional, Sequence, Tuple, Union
from data.dataset import ReconstructorSample

from tools import transforms as T


class ReconstructorDataTransform:
    """
    Data transformer for accelerated MRI reconstruction models. Randomly masks
    data according to specifications if no mask is provided.
    """

    def __init__(
        self,
        center_fractions: Optional[Sequence[float]] = [0.08, 0.04, 0.00],
        center_crop: Optional[Union[torch.Size, tuple]] = None,
        fixed_acceleration: Optional[int] = None,
        min_lines_acquired: Optional[int] = 16,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            center_crop: an optional tuple of shape HW to crop the final
                reconstruction to.
            fixed_acceleration: optional fixed acceleration factor. Default to
                variable acceleration factor (ie a random number of kspace
                between min_lines_acquired and W).
            min_lines_acquired: minimum number of total lines acquired.
            seed: optional random seed.
        """
        self.center_fractions = center_fractions
        self.center_crop = center_crop
        self.fixed_acceleration = fixed_acceleration
        self.min_lines_acquired = min_lines_acquired
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        kspace: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[np.ndarray] = None,
        metadata: dict = None,
        fn: str = None,
        slice_idx: int = -1,
    ) -> ReconstructorSample:
        """
        Mask and preprocess kspace data.
        Input:
            kspace: input kspace data of shape CHW2.
            mask: mask associated with dataset (should be None for training and
                validation datasets).
            target: the target image (if applicable).
            metadata: metadata associated with dataset.
            fn: name of the original data file.
            slice_idx: index of slice from original dataset.
        Returns:
            A ReconstructorSample object.
        """
        # Add a coil dimension to singlecoil data.
        if kspace.ndim < 4:
            kspace = torch.unsqueeze(kspace, dim=0)

        crop_size = (metadata["recon_size"][0], metadata["recon_size"][1])
        if self.center_crop is not None:
            crop_size = (
                min(crop_size[0], self.center_crop[0]),
                min(crop_size[1], self.center_crop[1]),
            )

        if target is not None:
            target = T.to_tensor(target)
            max_value = metadata["max"]
        else:
            target = torch.tensor(0)
            max_value = 0.0

        # Handle training and validation data.
        if mask is None:
            center_frac = self.choose_parameters()
            _, _, w, _ = kspace.size()
            mask, center_mask = self.global_mask(w, center_frac)
            masked_kspace = T.apply_mask(kspace, mask)
            num_low_frequencies = round(w * center_frac)
        else:
            masked_kspace = kspace.clone()
            mask = self.fit_mask(
                mask.type(torch.float32), masked_kspace.size()
            )
            center_mask, num_low_frequencies = self.construct_center_mask(mask)

        return ReconstructorSample(
            masked_kspace,
            mask,
            center_mask,
            num_low_frequencies,
            target,
            metadata,
            fn,
            slice_idx,
            max_value,
            crop_size,
        )

    def choose_parameters(self) -> float:
        """
        Select acceleration and center fraction parameters.
        Input:
            None.
        Returns:
            center_fractions: fraction of low-frequency columns to be retained.
        """
        return self.rng.choice(self.center_fractions)

    def center_mask(self, width: int, center_frac: float) -> torch.Tensor:
        """
        Build center mask based on the center fraction.
        Input:
            width: width of the input kspace data.
            center_frac: fraction of low-frequency data to be retained.
        Returns:
            mask: mask of shape W with only the specified center unmasked.
        """
        num_low_freqs = round(width * center_frac)
        mask = torch.zeros(width, dtype=torch.float32)
        pad = (width - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = 1.0
        return mask

    def construct_center_mask(
        self, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Estimate the center mask from an input mask. This function assumes that
        the input mask is a 1D line mask.
        Input:
            mask: an input mask of shape W.
        Returns:
            center_mask: the estimated center mask of shape W.
            num_low_frequencies: number of low-frequency lines in the center
                mask.
        """
        w = mask.size()[0]
        left = 1 + (w // 2) - torch.argmin(
            torch.squeeze(
                torch.flip(
                    torch.unsqueeze(mask[:w // 2], dim=0), dims=(0, 1)
                ),
                dim=0
            )
        )
        right = (w // 2) + torch.argmin(mask[w // 2:])
        center_mask = torch.zeros_like(mask, dtype=torch.float32).type_as(mask)
        center_mask[left:right] = 1.0
        return center_mask, torch.sum(center_mask)

    def global_mask(
        self, width: int, center_frac: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build mask for both center and non-central acceleration lines/points.
        Input:
            width: width of the input kspace data.
            center_frac: fraction of low-frequency data to be retained.
        Returns:
            mask: mask of shape W.
            center_mask: center mask of shape W with only the specified center
                unmasked.
        """
        center_mask = self.center_mask(width, center_frac)
        mask = torch.clone(center_mask)
        if self.fixed_acceleration is not None:
            acc_factor = self.rng.choice(self.fixed_acceleration)
            num_high_freqs = int(
                int(width / acc_factor) - torch.sum(
                    center_mask
                ).item()
            )
            num_high_freqs = max(
                0, min(width - torch.sum(center_mask).item(), num_high_freqs)
            )
        else:
            num_high_freqs = self.rng.randint(
                max(self.min_lines_acquired - torch.sum(center_mask), 0),
                width - torch.sum(center_mask),
            )
        idxs = self.rng.choice(
            torch.squeeze(
                torch.nonzero(center_mask < 1.0, as_tuple=False), dim=-1
            ),
            num_high_freqs,
            replace=False,
        )
        mask[idxs] = 1.0

        return mask, center_mask

    def fit_mask(
        self, mask: torch.Tensor, kspace_size: Union[torch.Size, tuple]
    ) -> torch.Tensor:
        """
        Crops (or pads) a mask to fit the input kspace size.
        Input:
            mask: 1D mask of shape W'.
            kspace_size: reference kspace size (CHW2).
        Returns:
            mask: 1D mask cropped (or padded) to shape W.
        """
        _, _, w, _ = kspace_size
        wp = mask.size()[0]

        # Handle cropping case.
        if wp > w:
            offset = (wp - w) // 2
            return mask[offset:offset + w]
        # Handle padding case.
        elif wp < w:
            offset = w - wp
            left = torch.zeros(offset // 2, dtype=mask.dtype)
            right = torch.zeros(offset // 2 + (offset % 2), dtype=mask.dtype)
            return torch.cat((left, mask, right))
        return mask
