"""
Dataset implementation for Shepp-Logan phantom reconstruction. Portions of this
code, including certain choices for default parameters, were taken from the
ISMRMRD Python Tools repository at
https://github.com/ismrmrd/ismrmrd-python-tools, as well as the 3D Shepp-Logan
phantom toolkit from Matthias Schabel at
https://mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom.

Author(s):
    Michael Yao
    Michael Hansen

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
from fastmri.fftc import fft2c_new
from fastmri.math import complex_mul
import numpy as np
import phantominator as P
from scipy.ndimage.filters import gaussian_filter
import torch
from typing import Optional, Sequence, Tuple, Union

from data.dataset import ReconstructorSample
from tools import transforms as T


class SheppLoganDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_samples: int,
        num_slices: Optional[int] = 40,
        slice_size: Union[torch.Size, tuple] = (256, 256),
        num_coils: int = 4,
        center_fractions: Sequence[float] = [0.08, 0.04],
        fixed_accelerations: Optional[Sequence[float]] = [4.0, 8.0],
        max_distance_distortion: float = 0.1,
        max_angle_distortion: float = 0.2,
        max_mr_param_distortion: float = 0.01,
        z_lim: float = 0.75,
        field_strengths: Sequence[float] = [1.5, 3.0],
        min_lines_acquired: int = 16,
        seed: Optional[int] = None
    ):
        """
        Args:
            num_samples: size of the dataset.
            num_slices: number of slices per dataset. Default 40.
            slice_size: size of each 2D slice of shape HW.
            num_coils: number of coils to simulate.
            center_fractions: fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of the numbers is
                chosen uniformly each time.
            fixed_accelerations: optional fixed acceleration factor. Default to
                variable acceleration factor (ie a random number of kspace
                between min_lines_acquired and W).
            max_distance_distortion: maximum distance displacement for phantom.
            max_angle_distortion: maximum angle rotation for phantom (radians).
            max_mr_param_distortion: maximum change in MR parameters to
                determine image intensity values.
            z_lim: boundary of allowed z values for slices. Default 0.75.
            field_strengths: B0 field strengths to choose from. Default 1.5T
                and 3.0T.
            min_lines_acquired: minimum number of total lines acquired.
            seed: optional random seed.
        """
        self.transform = None
        self.rng = np.random.RandomState(seed)
        self.num_samples = num_samples
        self.num_coils = num_coils
        self.shape = tuple(slice_size) + (num_slices,)
        self.center_fractions = center_fractions
        self.fixed_accelerations = fixed_accelerations
        self.min_lines_acquired = min(
            max(min_lines_acquired, 0), self.shape[1]
        )

        self.ellipses = P.mr_ellipsoid_parameters()
        self.max_distance_distortion = abs(max_distance_distortion)
        self.max_angle_distortion = abs(max_angle_distortion)
        self.max_mr_param_distortion = abs(max_mr_param_distortion)
        self.B0s = field_strengths
        self.zlims = (-abs(z_lim), abs(z_lim))

        self.target, self.simulated = None, None

    def mr_ellipsoid_distortions(self) -> np.ndarray:
        """
        Generates a random distortion matrix to subtly distort the standard
        Shepp-Logan ellipses.
        Input:
            None.
        Returns:
            Distortion matrix with the same shape as self.ellipses.
        """
        distance_distortions = self.rng.uniform(
            -self.max_distance_distortion,
            self.max_distance_distortion,
        ) * np.ones((self.ellipses.shape[0], 6), dtype=np.float32)
        angle_distortions = self.rng.uniform(
            -self.max_angle_distortion,
            self.max_angle_distortion,
        ) * np.ones((self.ellipses.shape[0], 1), dtype=np.float32)
        mr_param_distortions = self.rng.uniform(
            -self.max_mr_param_distortion,
            self.max_mr_param_distortion,
            (self.ellipses.shape[0], 6)
        )
        return np.concatenate(
            (distance_distortions, angle_distortions, mr_param_distortions,),
            axis=-1
        )

    def mr_phantom(
        self, ellipse_distortions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generates a random Shepp-Logan phantom with MR tissue parameters.
        Randomly chooses to return either proton density, T1, T2, or T2*
        simulation.
        Input:
            ellipse_distortions: optional matrix of ellipse distortions
                generated by self.mr_ellipsoid_distortions(). Default no
                distortions.
        Returns:
            Simulated phantom data of shape NHW, where N is the number of
            slices and HW are the 2D slice dimensions.
        """
        mr_scans = P.mr_shepp_logan(
            self.shape,
            np.add(self.ellipses, ellipse_distortions),
            B0=self.rng.choice(self.B0s),
            T2star=(self.rng.uniform() > 0.5),
            zlims=self.zlims,
        )
        return np.transpose(
            mr_scans[self.rng.choice(len(mr_scans))], axes=(-1, 0, 1)
        )

    def __len__(self) -> int:
        """
        Returns the number of samples in our dataset.
        Input:
            None.
        Returns:
            Number of samples in our dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> ReconstructorSample:
        """
        Loads and returns a sample from the dataset at the given index.
        Input:
            idx: index of desired dataset.
        Returns:
            A ReconstructorSample object.
        """
        if self.target is None:
            ellipse_distortions = self.mr_ellipsoid_distortions()
            self.target = self.mr_phantom(ellipse_distortions)
            # Add Gaussian blur to target.
            self.target = gaussian_filter(self.target, sigma=1)

            self.simulated = self.target.copy()
            phase_map = np.zeros(self.shape, dtype=np.float32)
            background_mask = np.ones(self.shape, dtype=bool)
            self.simulated = np.transpose(self.simulated, axes=(1, 2, 0))

            # Add global phase rolling.
            phase_roll = np.zeros(self.shape, dtype=np.complex64)
            for i in range(self.simulated.shape[-1]):
                phase_roll[..., i] = self._random_phase_roll()
            phase_roll = np.where(
                np.logical_not(background_mask), phase_roll, 1.0 + 0.0j
            )
            phase_map = np.add(phase_map, phase_roll)

            # Add uncorrelated noise to the simulation.
            noise_frac = 0.01
            mag_noise = self.rng.uniform(0.0, 1.0, size=self.simulated.shape)
            mag_noise = noise_frac * np.max(np.abs(self.simulated)) * mag_noise
            phase_noise = self.rng.uniform(
                0.0, 2.0 * np.pi, size=self.simulated.shape
            )
            noise_matrix = mag_noise * np.exp(1j * phase_noise)
            self.simulated = np.add(self.simulated, noise_matrix)

            self.target = T.to_tensor(self.target)
            self.simulated = T.to_tensor(
                np.transpose(
                    np.multiply(self.simulated, phase_map), axes=(-1, 0, 1)
                )
            )

        slice_idx = self.rng.randint(0, self.target.shape[0])
        target = self.target[slice_idx, ...]
        simulated = self.simulated[slice_idx, ...]

        dtheta = self.rng.uniform(np.pi / self.num_coils)
        sens_maps = self._sens_maps(
            rotation=self.rng.uniform(-dtheta, dtheta), normalize=False
        )
        kspace = self.sens_expand(simulated, sens_maps)
        mask, center_mask = self._mask()
        num_low_frequencies = torch.sum(center_mask).item()
        masked_kspace = T.apply_mask(kspace, mask)

        if self.target.shape[0] == 1:
            self.target = None
            self.simulated = None
        else:
            self.target = torch.cat((
                self.target[:slice_idx], self.target[(slice_idx + 1):]
            ), dim=0)
            self.simulated = torch.cat((
                self.simulated[:slice_idx], self.simulated[(slice_idx + 1):]
            ), dim=0)

        return ReconstructorSample(
            masked_kspace.type(torch.float32),
            mask.type(torch.float32),
            center_mask.type(torch.float32),
            num_low_frequencies,
            torch.abs(target).type(torch.float32),
            {},  # No metadata associated with phantom simulation.
            "shepp_logan_simulated.h5",  # Dummy filename.
            slice_idx,
            float(torch.max(torch.abs(target)).item()),
            self.shape[:-1]
        )

    def _phase_roll(
        self,
        rotation: float = 0.0,
        center: Tuple[float, float] = (0.0, 0.0),
        roll: float = np.pi
    ) -> np.ndarray:
        """
        Generates a specified phase roll.
        Input:
            rotation: rotation of phase roll axis relative to horizontal.
            center: center of phase roll in (x, y) coordinates.
            roll: roll scaling factor.
        Returns:
            The phase roll specified by the input parameters.
        """
        x, y = np.meshgrid(
            np.arange(self.shape[1]), np.arange(self.shape[0]), indexing="xy"
        )
        x, y = (2 * x / self.shape[-1]) - 1, (2 * y / self.shape[0]) - 1
        x, y = x - center[-1], y - center[0]
        rotx = (x * np.cos(rotation)) - (y * np.sin(rotation)) + center[0]
        roty = (x * np.sin(rotation)) + (y * np.cos(rotation)) + center[-1]
        return np.cos(rotx * roll) + (1j * np.sin(roty * roll))

    def _random_phase_roll(self) -> np.ndarray:
        """
        Generates a random phase roll.
        Input:
            None.
        Returns:
            Random phase roll.
        """
        rotation = (self.rng.random() - 0.5) * np.pi
        center = (self.rng.random() - 0.5, self.rng.random() - 0.5)
        roll = np.pi * self.rng.random() / 16.0
        return self._phase_roll(rotation, center, roll)

    def sens_expand(
        self, x: torch.Tensor, sens_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the expand operator that computes the images seen by each coil
        from the image and sensitivity maps.
        Input:
            x: input image.
            sens_maps: computed sensitivity maps.
        Returns:
            Individual coil images.
        """
        return fft2c_new(complex_mul(x, sens_maps))

    def _sens_maps(
        self,
        relative_radius: float = 3.0,
        rotation: float = 0.0,
        normalize: bool = False,
        max_noise: float = 0.01
    ) -> torch.Tensor:
        """
        Generate birdcage coil sensitivities.
        Input:
            relative_radius: coil distance from the center of the image in
                image coordinates (-1.0 to +1.0).
            rotation: rotation of the position of the coils in radians.
            normalize: whether to normalize the sensitivity maps by the RSS.
            max_noise: maximum magnitude of random coil sensitivity noise to
                add to the sensitivity maps.
        Returns:
            Generated complex sensitivity maps.
        """
        maps = np.zeros(
            (self.num_coils, self.shape[0], self.shape[1]),
            dtype=np.complex64
        )
        dtheta = 2 * np.pi / self.num_coils
        for c in range(self.num_coils):
            # Calculate coil position in Cartesian coordinates.
            coil_x = relative_radius * np.cos((c * dtheta) + rotation)
            coil_y = relative_radius * np.sin((c * dtheta) + rotation)
            coil_phase = -c * dtheta

            for y in range(self.shape[0]):
                # Convert y index values to coil coordinate system.
                y_co = (((2 / self.shape[0]) * y) - 1) - coil_y
                for x in range(0, self.shape[1]):
                    # Convert x index values to coil coordinate system.
                    x_co = (((2 / self.shape[1]) * x) - 1) - coil_x
                    # Calculate distance of (x, y) point from coil center.
                    r_co = np.sqrt((x_co * x_co) + (y_co * y_co))
                    # Calculate the phase.
                    phi = np.arctan2(x_co, -y_co) + coil_phase
                    # Generate random noise.
                    r_eps = self.rng.uniform(0, max_noise)
                    theta_eps = self.rng.uniform(0, 2 * np.pi)
                    eps = np.sqrt(r_eps) * np.exp(1j * theta_eps)
                    # Estimate the coil sensitivity.
                    maps[c, y, x] = ((1 / r_co) * np.exp(1j * phi)) + eps

        if normalize:
            rss = np.sqrt(np.sum(np.abs(maps) * np.abs(maps), dim=0))
            maps = np.divide(
                maps, rss[np.newaxis, ...]
            )

        return T.to_tensor(maps)

    def _mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build mask for both center and non-central acceleration lines/points.
        Input:
            None.
        Returns:
            mask: mask of shape self.shape[-1].
            center_mask: center mask of shape self.shape[-1] with only the
                specified center unmasked.
        """
        center_frac = self.rng.choice(self.center_fractions)
        # Build the center mask.
        num_low_freqs = round(self.shape[1] * center_frac)
        center_mask = torch.zeros(self.shape[1], dtype=torch.float32)
        left = (self.shape[1] - num_low_freqs + 1) // 2
        right = left + num_low_freqs
        center_mask[left:right] = 1.0
        num_low_freqs = int(torch.sum(center_mask).item())

        mask = torch.clone(center_mask)
        if self.fixed_accelerations is not None:
            acc_factor = self.rng.choice(self.fixed_accelerations)
            total_lines = int(round(self.shape[1] / acc_factor))
        else:
            total_lines = self.rng.randint(
                self.min_lines_acquired, self.shape[1]
            )
        num_high_freqs = max(total_lines - num_low_freqs, 0)
        idxs = self.rng.choice(
            torch.squeeze(
                torch.nonzero(center_mask < 1.0, as_tuple=False), dim=-1
            ),
            num_high_freqs,
            replace=False,
        )
        mask[idxs] = 1.0

        return mask, center_mask
