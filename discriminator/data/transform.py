"""
Defines the data transformer for training and validating k-space discriminator
models. Portions of this code were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import numpy as np
from fastmri.fftc import fft2c_new, ifft2c_new
import torch
import torchvision.transforms
from typing import Optional, Sequence, Tuple
from data.dataset import DiscriminatorSample

from tools import transforms as T


class DiscriminatorDataTransform:
    """
    Data transformer for training and validating discriminator models. Randomly
    masks data according to specifications.
    """

    def __init__(
        self,
        coil_compression: int = -1,
        rotation: Sequence[float] = [20.0, 50.0],
        dx: Sequence[float] = [0.0, 0.0],
        dy: Sequence[float] = [0.0, 0.0],
        p_transform: float = 0.5,
        p_spike: float = 0.5,
        max_spikes: int = 10,
        p_rf_cont: float = 0.5,
        max_rf_cont: int = 10,
        min_lines_acquired: int = 8,
        max_lines_acquiring: int = 12,
        seed: Optional[int] = None,
        is_mlp: bool = True,
        center_frac: Optional[float] = 0.04,
        center_crop: Optional[Sequence[int]] = None,
        inference: bool = False,
        accuracy_by_line: bool = False
    ):
        """
        Args:
            coil_compression: whether or not to compress coil dimension from
                15 (for fastMRI multicoil data) to 4 using SVD. Only
                applicable for multicoil data.
            rotation: range of rotation magnitude about the center of the
                image in degrees. The image will be rotated between
                -theta and +theta degrees, where rotation[0] < abs(theta) <
                rotation[-1]. Defaults to [5.0, 15.0] degrees.
            dx: range of absolute fraction for horizontal image translation.
                Defaults to [0.05, 0.1].
            dy: range of absolute fraction for vertical image translation.
                Defaults to [0.05, 0.1].
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
            is_mlp: specify whether discriminator is an MLP network.
            center_frac: fraction of low-frequency columns to be retained.
                Only required is is_mlp is True.
            center_crop: kspace crop size. Default no cropping.
            inference: whether or not we are running inference.
            accuracy_by_line: special flag to turn on testing discriminator
                accuracy by kspace line distance from the center. Only applies
                wheen running inference.
        """
        self.seed = seed
        self.is_mlp = is_mlp
        self.center_frac = center_frac
        self.center_crop = center_crop
        self.rng = np.random.RandomState(self.seed)
        torch.manual_seed(self.seed)

        self.coil_compression = coil_compression
        self.rotation = sorted(
            (abs(rotation[0] % 180), abs(rotation[-1] % 180))
        )
        if self.rotation[0] == self.rotation[-1]:
            self.rotation[0] = -1.0 * self.rotation[0]
        self.dx = sorted(
            (max(min(dx[0], 1.0), 0.0), max(min(dx[-1], 1.0), 0.0))
        )
        if self.dx[0] == self.dx[-1]:
            self.dx[0] = -1.0 * self.dx[0]
        self.dy = sorted(
            (max(min(dy[0], 1.0), 0.0), max(min(dy[-1], 1.0), 0.0))
        )
        if self.dy[0] == self.dy[-1]:
            self.dy[0] = -1.0 * self.dy[0]
        self.p = p_transform
        self.p_spike = p_spike
        self.max_spikes = max_spikes
        self.p_rf_cont = p_rf_cont
        self.max_rf_cont = max_rf_cont
        self.min_lines_acquired = min_lines_acquired
        self.max_lines_acquiring = max_lines_acquiring
        self.inference = inference
        self.accuracy_by_line = accuracy_by_line and self.inference

    def __call__(
        self,
        kspace: torch.Tensor,
        metadata: dict,
        fn: str,
        slice_idx: int,
        targ: Optional[torch.Tensor] = None,
    ) -> DiscriminatorSample:
        """
        Generate a kspace mask and apply it on the training kspace data.
        Input:
            kspace: input kspace data of shape CHW2.
            metadata: metadata associated with dataset.
            fn: name of the original data file.
            slice_idx: index of slice from original dataset.
            targ: optional target image. Used only for inference.
        Returns:
            A DiscriminatorSample object.
        """
        # Center crop the kspace if necessary.
        if self.center_crop is not None:
            kspace = torch.permute(kspace, dims=(0, -1, 1, 2))
            kspace = T.center_crop(kspace, self.center_crop)
            kspace = torch.permute(kspace, dims=(0, 2, 3, 1))
        c, h, w, _ = kspace.size()
        lines_acquired = self.rng.randint(self.min_lines_acquired, w)
        lines_acquiring = self.rng.randint(1, self.max_lines_acquiring)

        # 1. Construct the acquired and acquiring kspace data masks.
        sampled_mask, acquiring_mask = self.masks(
            kspace.size(), lines_acquired, lines_acquiring
        )
        # Convert from mask to indices.
        lines_acquired = np.where(sampled_mask > 0)[0]
        lines_acquiring = np.where(acquiring_mask > 0)[0]

        cimg = ifft2c_new(kspace)
        # 2. Apply affine transform.
        no_affine_transform = self.rotation == (0.0, 0.0)
        no_affine_transform = no_affine_transform and self.dx == (0.0, 0.0)
        no_affine_transform = no_affine_transform and self.dy == (0.0, 0.0)
        if no_affine_transform:
            theta, deltax, deltay = 0.0, 0.0, 0.0
            distorted_target = cimg.clone()
        elif self.rng.uniform() < self.p:
            theta = self.rng.uniform(self.rotation[0], self.rotation[-1])
            if self.rng.uniform() < 0.5:
                theta = -1.0 * theta
            deltax = self.rng.uniform(self.dx[0], self.dx[-1])
            if self.rng.uniform() < 0.5:
                deltax = -1.0 * deltax
            deltay = self.rng.uniform(self.dy[0], self.dy[-1])
            if self.rng.uniform() < 0.5:
                deltay = -1.0 * deltay
            distorted_target = torchvision.transforms.functional.affine(
                torch.permute(cimg, (0, -1, 1, 2)),
                angle=theta,
                translate=[deltax, deltay],
                scale=1.0,
                shear=0.0
            )
            distorted_target = torch.permute(distorted_target, (0, 2, 3, 1))
        else:
            theta, deltax, deltay = 0.0, 0.0, 0.0
            distorted_target = cimg.clone()

        distorted_kspace = fft2c_new(distorted_target)

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

        compressed_kspace = kspace.clone()
        compressed_distorted = distorted_kspace.clone()
        # Apply coil compression if specified.
        if c > 1 and self.coil_compression > 0 and self.coil_compression < c:
            # Convert to numpy.
            compressed_kspace = compressed_kspace.detach().cpu().numpy()
            compressed_distorted = compressed_distorted.detach().cpu().numpy()

            compressed_kspace = compressed_kspace[..., 0] + (
                1j * compressed_kspace[..., -1]
            )
            compressed_distorted = compressed_distorted[..., 0] + (
                1j * compressed_distorted[..., -1]
            )

            compressed_kspace = np.reshape(
                compressed_kspace, (compressed_kspace.shape[0], -1)
            )
            compressed_distorted = np.reshape(
                compressed_distorted, (compressed_distorted.shape[0], -1)
            )
            u, _, _ = np.linalg.svd(
                compressed_kspace, compute_uv=True, full_matrices=False
            )
            # Multiply kspace by the complex conjugate of the first
            # self.coil_compression left vectors.
            compressed_kspace = np.reshape(
                np.array(np.matmul(
                    np.matrix(u[:, :self.coil_compression].copy()).H,
                    compressed_kspace
                )),
                (self.coil_compression, h, w)
            )
            compressed_distorted = np.reshape(
                np.array(np.matmul(
                    np.matrix(u[:, :self.coil_compression].copy()).H,
                    compressed_distorted
                )),
                (self.coil_compression, h, w)
            )
            compressed_kspace = T.to_tensor(compressed_kspace)
            compressed_distorted = T.to_tensor(compressed_distorted)

        if targ is not None:
            max_value = metadata["max"]
        else:
            max_value = 0.0

        return DiscriminatorSample(
            compressed_kspace,
            compressed_distorted,
            theta,
            deltax,
            deltay,
            sampled_mask.type(torch.float32),
            acquiring_mask.type(torch.float32),
            metadata,
            fn,
            slice_idx,
            kspace,
            distorted_kspace,
            max_value
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
        # For training MLP discriminators, can just treat the center lines
        # as ground truth and everything else as distorted.
        if self.is_mlp:
            num_center_lines = int(w * self.center_frac)
            l_center = (w - num_center_lines) // 2
            r_center = l_center + num_center_lines
            lines_acquired = np.arange(l_center, r_center)
            lines_acquiring = np.concatenate(
                (np.arange(0, l_center), np.arange(r_center, w),)
            )
            if self.accuracy_by_line:
                pass
            elif not self.inference:
                lines_acquiring = np.array([self.rng.choice(lines_acquiring)])
            else:
                num_lines_acquiring = self.rng.randint(
                    1, self.max_lines_acquiring
                )
                num_lines_acquired = max(
                    self.rng.randint(
                        self.min_lines_acquired, w - num_lines_acquiring
                    ),
                    r_center - l_center
                )
                num_hf_lines_acquired = int(
                    num_lines_acquired - (r_center - l_center)
                )
                hf_lines = self.rng.choice(
                    lines_acquiring,
                    size=(num_hf_lines_acquired + num_lines_acquiring),
                    replace=False
                )
                lines_acquiring = hf_lines[:num_lines_acquiring]
                lines_acquired = np.concatenate(
                    (lines_acquired, hf_lines[num_lines_acquiring:],)
                )
        else:
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
