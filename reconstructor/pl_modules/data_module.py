"""
Data module to train, validate, and test image reconstruction models.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import os
from torch.utils.data import DataLoader
from pathlib import Path
import pytorch_lightning as pl
import sys
import torch
from typing import Optional, Union
from data.dataset import ReconstructorDataset, ReconstructorSample
from data.transform import ReconstructorDataTransform

sys.path.append("..")
import helper.utils.transforms as T


class DataModule(pl.LightningDataModule):
    """Data module for data sets as input to image reconstruction models."""

    def __init__(
        self,
        data_path: Union[Path, str],
        train_dir: str = "multicoil_train",
        val_dir: str = "multicoil_val",
        test_dir: str = "multicoil_test",
        batch_size: int = 1,
        center_crop: Union[torch.Size, tuple] = (640, 368,),
        fixed_acceleration: Optional[int] = None,
        num_workers: int = 4,
        seed: Optional[int] = None,
        fast_dev_run: bool = False,
    ):
        """
        Args:
            data_path: a string path to the reconstruction dataset.
            train_dir: a string path to the training subdirectory.
            val_dir: a string path to the validation subdirectory.
            test_dir: a string path to the test set subdirectory.
            batch_size: batch size.
            center_crop: an optional tuple of shape HW to crop the input kspace
                size to.
            fixed_acceleration: optional fixed acceleration factor. Default to
                variable acceleration factor (ie a random number of kspace
                between min_lines_acquired and W).
            num_workers: number of workers for PyTorch dataloaders.
            seed: optional random seed for determining order of dataset.
            fast_dev_run: whether we are running a test fast_dev_run.
        """
        super().__init__()

        self.data_path = str(data_path)
        self.root_path = Path(self.data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # If all of train_dir, val_dir, and test_dir are None, then we are
        # predicting reconstructions.
        if train_dir is None and val_dir is None and test_dir is None:
            multicoil = "multicoil" in data_path.lower()
            self.predict = ReconstructorDataset(
                self.data_path,
                ReconstructorDataTransform(
                    center_crop=center_crop,
                    seed=seed,
                ),
                seed=seed,
                multicoil=multicoil
            )
            return

        multicoil = "multicoil" in train_dir.lower()
        self.train = ReconstructorDataset(
            str(os.path.join(self.data_path, train_dir)),
            ReconstructorDataTransform(
                center_fractions=[0.08, 0.04],
                center_crop=center_crop,
                fixed_acceleration=fixed_acceleration,
                min_lines_acquired=16,
                seed=seed
            ),
            seed=seed,
            fast_dev_run=fast_dev_run,
            multicoil=multicoil
        )
        self.val = ReconstructorDataset(
            str(os.path.join(self.data_path, val_dir)),
            ReconstructorDataTransform(
                center_fractions=[0.08, 0.04],
                center_crop=center_crop,
                fixed_acceleration=fixed_acceleration,
                min_lines_acquired=16,
                seed=seed,
            ),
            seed=seed,
            fast_dev_run=fast_dev_run,
            multicoil=multicoil
        )
        self.test = ReconstructorDataset(
            str(os.path.join(self.data_path, test_dir)),
            ReconstructorDataTransform(
                center_crop=center_crop,
                seed=seed,
            ),
            seed=seed,
            fast_dev_run=fast_dev_run,
            multicoil=multicoil
        )
        self.predict = None

    def train_dataloader(self):
        """Returns the training dataloader."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        """Returns the validation dataloader."""
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Return the test set dataloader."""
        test_collate_fn = collate_fn if self.batch_size > 1 else None
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=test_collate_fn,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        """Return the prediction set dataloader."""
        if self.predict is None:
            return None
        return DataLoader(self.predict, num_workers=self.num_workers)


def collate_fn(data):
    """Custom function to combine data specifically for test batches."""
    kspace, mask, metadata, fn, slice_idx, max_value, crop_size = zip(*data)
    # Determine minimum kspace size.
    min_size = None
    for ks in kspace:
        if min_size is None:
            min_size = tuple(ks.size()[1:-1])
            continue

        if ks.size()[1] < min_size[0]:
            min_size = (ks.size()[1], min_size[-1],)
        if ks.size()[-2] < min_size[-1]:
            min_size = (min_size[0], ks.size()[-2],)

    kspace_resized = []
    for ks in kspace:
        c, h, w, d = ks.size()
        ks = ks.reshape((d * c, h, w,))
        ks = T.center_crop(ks, min_size)
        dc, h, w = ks.size()
        ks = ks.reshape((dc // d, h, w, d,))
        kspace_resized.append(ks.unsqueeze(0))
    kspace = torch.cat(kspace_resized, dim=0)

    mask_resized = []
    for m in mask:
        mask_resized.append(T.center_crop(m, min_size).unsqueeze(0))
    mask = torch.cat(mask_resized, dim=0)

    return ReconstructorSample(
        kspace, mask, metadata, fn, slice_idx, max_value, crop_size
    )
