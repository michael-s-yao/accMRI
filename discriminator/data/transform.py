"""
Defines the data transformer for training and validating k-space discriminator
models. Portions of this code were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import numpy as np
import sys
import torch
import torchvision.transforms
from typing import Optional, Sequence, Tuple
from data.dataset import DiscriminatorSample

sys.path.append("..")
from common.utils.math import fft2c, ifft2c


class DiscriminatorDataTransform:
    """
    Data transformer for training and validating TIVarNet models. Randomly
    masks data according to specifications.
    """

    def __init__(
        self,
        max_rotation: float = 15.0,
        max_x: float = 0.1,
        max_y: float = 0.1,
        p_transform: float = 0.5,
        p_spike: float = 0.5,
        max_spikes: int = 10,
        p_rf_cont: float = 0.5,
        max_rf_cont: int = 10,
        min_lines_acquired: int = 8,
        max_lines_acquiring: int = 12,
        seed: Optional[int] = None,
    ):
        """
        Args:
            max_rotation: maximum angle of rotation about the center of the
                image in degrees. The image will be rotated between
                -max_rotation and max_rotation degrees. Defaults to 15 degrees.
            max_x: maximum absolute fraction for horizontal image translation.
                Defaults to 0.1.
            max_y: maximum absolute fraction for vertical image translation.
                Defaults to 0.1.
            p_transform: probability of applying the affine transform or any
                of the kspace dirtying operations. Defaults to 0.5.
            p_spike: probability of a kspace spike.
            max_spikes: upper cutoff for the maximum number of kspace spikes.
            p_rf_cont: probability of an RF contamination.
            max_rf_cont: upper cutoff for the maximum number of RF
                contaminations.
            min_lines_acquired: minimum number of lines previously acquired.
            max_lines_acquiring: maximum number of lines in a single
                acquisition.
            seed: optional random seed.
        """
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        torch.manual_seed(self.seed)
        max_rotation = abs(max_rotation % 180)
        max_x = max(min(max_x, 1.0), 0.0)
        max_y = max(min(max_y, 1.0), 0.0)
        self.transform = torchvision.transforms.RandomApply(
            torch.nn.ModuleList([
                torchvision.transforms.RandomAffine(
                    degrees=max_rotation, translate=(max_x, max_y)
                )
            ]),
            p=p_transform
        )
        self.p = p_transform
        self.p_spike = p_spike
        self.max_spikes = max_spikes
        self.p_rf_cont = p_rf_cont
        self.max_rf_cont = max_rf_cont
        self.min_lines_acquired = min_lines_acquired
        self.max_lines_acquiring = max_lines_acquiring

    def __call__(
        self,
        kspace: torch.Tensor,
        metadata: dict,
        fn: str,
        slice_idx: int,
    ) -> DiscriminatorSample:
        """
        Generate a kspace mask and apply it on the training kspace data.
        Input:
            kspace: input multi-coil kspace data of shape CHW2.
            metadata: metadata associated with dataset.
            fn: name of the original data file.
            slice_idx: index of slice from original dataset.
        Returns:
            A DiscriminatorSample object.
        """
        _, h, w, _ = kspace.size()
        lines_acquired = self.rng.randint(self.min_lines_acquired, w)
        lines_acquiring = self.rng.randint(1, self.max_lines_acquiring)

        # 1. Construct the acquired and acquiring kspace data masks.
        sampled_mask, acquiring_mask = self.masks(
            kspace.size(), lines_acquired, lines_acquiring
        )
        # Convert from mask to indices.
        lines_acquired = np.where(sampled_mask > 0)[0]
        lines_acquiring = np.where(acquiring_mask > 0)[0]

        cimg = ifft2c(kspace)
        # 2. Apply affine transform.
        distorted_target = self.transform(torch.permute(cimg, (0, -1, 1, 2)))
        distorted_target = torch.permute(distorted_target, (0, 2, 3, 1))
        distorted_kspace = fft2c(distorted_target)

        # 3. Apply kspace spiking.
        if self.rng.uniform() < self.p:
            ud_locs = np.where(self.rng.uniform(size=h) < self.p_spike)[0]
            lr_locs = lines_acquiring[
                np.where(
                    self.rng.uniform(
                        size=lines_acquiring.shape[0]
                    ) < self.p_spike
                )[0]
            ].squeeze()
            ud, lr = np.meshgrid(ud_locs, lr_locs, indexing="ij")
            locs = np.concatenate((ud[..., None], lr[..., None]), axis=-1)
            locs = locs.reshape(-1, locs.shape[-1])
            # Only select some of the specified locations.
            max_spikes = self.rng.randint(1, self.max_spikes + 1)
            if locs.shape[0] > max_spikes:
                self.rng.shuffle(locs)
                locs = locs[:max_spikes, :]
            distorted_kspace = self.add_kspace_spikes(
                distorted_kspace, locs
            )

        # 4. Apply RF contamination.
        if self.rng.uniform() < self.p:
            vert_locs = lines_acquiring[
                np.where(
                    self.rng.uniform(
                        size=lines_acquiring.shape[0]
                    ) < self.p_rf_cont
                )[0]
            ]
            vert_locs = np.vstack((vert_locs, np.ones_like(vert_locs))).T
            hori_locs = np.where(self.rng.uniform(size=h) < self.p_rf_cont)[0]
            hori_locs = np.vstack((hori_locs, np.zeros_like(hori_locs))).T
            locs = np.concatenate((vert_locs, hori_locs))
            # Only select some of the specified locations.
            max_rf_cont = self.rng.randint(1, self.max_spikes + 1)
            if locs.shape[0] > max_rf_cont:
                self.rng.shuffle(locs)
                locs = locs[:max_rf_cont, :]
            distorted_kspace = self.add_rf_contamination(
                distorted_kspace, locs
            )

        return DiscriminatorSample(
            kspace,
            distorted_kspace,
            sampled_mask.type(torch.float32),
            acquiring_mask.type(torch.float32),
            metadata,
            fn,
            slice_idx
        )

    def masks(
        self,
        kspace_shape: Sequence,
        lines_acquired: int,
        lines_acquiring: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build masks of previously acquired lines and lines in current
        acquisition.
        Input:
            kspace_shape: shape of the kspace data (ie CHW2).
            lines_acquired: number of lines that have already been acquired.
            lines_acquiring: number of lines to acquire in next request.
        Returns:
            acquired_mask: mask of shape W of the lines already acquired.
            acquiring_mask: mask of shape W of the next lines to acquire.
        """
        # Cannot have acquired more lines than the total number of lines
        # available. Furthermore, in our implementation we permit the
        # possibility of reacquiring the same lines, since this may happen
        # in practice.
        _, _, w, _ = kspace_shape
        lines_acquired = max(min(lines_acquired, w), 0)
        lines_acquiring = max(min(lines_acquiring, w), 0)

        # Get the index of the acquired and acquiring lines.
        lines_acquired = self.rng.choice(
            np.arange(0, w, dtype=int), size=lines_acquired, replace=False
        )
        lines_acquiring = self.rng.choice(
            np.arange(0, w, dtype=int), size=lines_acquiring, replace=False
        )

        # Convert from indices to actual 1D masks.
        acquired_mask = np.where(
            np.isin(
                np.arange(0, w, dtype=int),
                lines_acquired,
                assume_unique=True,
            ),
            1.0,
            0.0
        )
        acquiring_mask = np.where(
            np.isin(
                np.arange(0, w, dtype=int),
                lines_acquiring,
                assume_unique=True,
            ),
            1.0,
            0.0
        )

        return (
            torch.from_numpy(acquired_mask), torch.from_numpy(acquiring_mask)
        )

    def add_kspace_spikes(
        self, kspace: torch.Tensor, locs: np.ndarray
    ) -> torch.Tensor:
        """
        Adds spikes at the specified locations to kspace. Portions of this code
        was adapted from https://github.com/birogeri/kspace-explorer.
        Input:
            kspace: complex kspace data of shape CHW2.
            locs: an Nx2 matrix of coordinates for the spikes (row, column).
        Returns:
            The complex kspace data with the shapes added.
        """
        mag = torch.sqrt(torch.sum(torch.square(kspace), dim=-1))
        spike_intensity = torch.max(mag)
        scale_factor = self.rng.uniform(1.0, 2.0, size=locs.shape[0])
        spike_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=locs.shape[0])
        for i, (r, c) in enumerate(locs):
            intensity = spike_intensity * scale_factor[i]
            phase = spike_phase[i]
            re = intensity * np.cos(phase)
            im = intensity * np.sin(phase)
            kspace[:, r, c, 0] = re
            kspace[:, r, c, 1] = im
        return kspace

    def add_rf_contamination(
        self, kspace: torch.Tensor, locs: np.ndarray
    ) -> torch.Tensor:
        """
        Adds RF contamination at the specified locations to kspace.
        Input:
            kspace: complex kspace data of shape CHW2.
            locs: an Nx2 matrix of coordinates for the spikes (idx, {0, 1}).
        Returns:
            The complex kspace data with the shapes added.
        """
        spike_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=locs.shape[0])
        # RF contamination magnitude was determined empirically.
        intensities = self.rng.uniform(1e-5, 3e-5, size=locs.shape[0])
        for i, (idx, dim) in enumerate(locs):
            intensity, phase = intensities[i], spike_phase[i]
            re = intensity * np.cos(phase)
            im = intensity * np.sin(phase)
            if dim == 0:
                kspace[:, idx, :, 0] = re
                kspace[:, idx, :, 1] = im
            elif dim == 1:
                kspace[:, :, idx, 0] = re
                kspace[:, :, idx, 1] = im
            else:
                raise ValueError(f"dim {dim} must be one of 0 or 1.")
        return kspace
