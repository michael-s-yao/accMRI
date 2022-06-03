"""
Data module to train, validate, and test a kspace discriminator.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import sys
import torch
from typing import Optional
from data.dataset import DiscriminatorDataset, DiscriminatorSample
from data.transform import DiscriminatorDataTransform

sys.path.append("..")
import common.utils.transforms as T


class DataModule(pl.LightningDataModule):
    """Data module for kspace discriminator data sets."""

    def __init__(
        self,
        data_path: str,
        train_dir: str = "singlecoil_train",
        val_dir: str = "singlecoil_val",
        test_dir: str = "singlecoil_test",
        batch_size: int = 1,
        num_workers: int = 4,
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
        seed: Optional[int] = None
    ):
        """
        Args:
            data_path: a string path to the kspace dataset.
            train_dir: a string path to the training set subdirectory.
            val_dir: a string path to the validation set subdirectory.
            test_dir: a string path to the test set subdirectory.
            batch_size: batch size.
            num_workers: number of workers for PyTorch dataloaders.
            max_rotation: maximum angle of rotation about the center of the
                image in degrees. The image will be rotated between
                -max_rotation and max_rotation degrees. Defaults to 15 degrees.
            max_x: maximum absolute fraction for horizontal image translation.
                Defaults to 0.1.
            max_y: maximum absolute fraction for vertical image translation.
                Defaults to 0.1.
            p_transform: probability of applying the affine transform or any
                of the kspace dirtying operations. Defaults to 0.5.
            p_spike: probability of a kspace spike. Defaults to 0.5.
            max_spikes: upper cutoff for the maximum number of kspace spikes.
            p_rf_cont: probability of an RF contamination. Defauls to 0.5.
            max_rf_cont: upper cutoff for the maximum number of RF
                contaminations.
            min_lines_acquired: minimum number of lines previously acquired.
            max_lines_acquiring: maximum number of lines in a single
                acquisition.
            seed: optional random seed.
        """
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.collate_fn = None
        if self.batch_size > 1:
            self.collate_fn = collate_fn
        self.num_workers = num_workers

        datasets = {}
        keys = [train_dir, val_dir, test_dir]
        for key in keys:
            datasets[key] = DiscriminatorDataset(
                str(os.path.join(self.data_path, key)),
                DiscriminatorDataTransform(
                    max_rotation=max_rotation,
                    max_x=max_x,
                    max_y=max_y,
                    p_transform=p_transform,
                    p_spike=p_spike,
                    max_spikes=max_spikes,
                    p_rf_cont=p_rf_cont,
                    max_rf_cont=max_rf_cont,
                    min_lines_acquired=min_lines_acquired,
                    max_lines_acquiring=max_lines_acquiring,
                    seed=seed,
                ),
                seed=seed
            )
        self.train = datasets[train_dir]
        self.val = datasets[val_dir]
        self.test = datasets[test_dir]

    def train_dataloader(self):
        """Returns the training dataloader."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        """Returns the validation dataloader."""
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Return the test set dataloader."""
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )


def collate_fn(data):
    """Custom function to combine data specifically for test batches."""
    ref_k, dis_k, smp_m, acq_m, metadata, fn, slice_idx = zip(*data)
    # Determine minimum kspace size.
    min_size = None
    for ks in ref_k:
        if min_size is None:
            min_size = tuple(ks.size()[1:-1])
            continue

        if ks.size()[1] < min_size[0]:
            min_size = (ks.size()[1], min_size[-1],)
        if ks.size()[-2] < min_size[-1]:
            min_size = (min_size[0], ks.size()[-2],)

    kspace_resized = []
    for ks in ref_k:
        c, h, w, d = ks.size()
        ks = ks.reshape((d * c, h, w,))
        ks = T.center_crop(ks, min_size)
        dc, h, w = ks.size()
        ks = ks.reshape((dc // d, h, w, d,))
        kspace_resized.append(ks.unsqueeze(0))
    ref_k = torch.cat(kspace_resized, dim=0)

    kspace_resized = []
    for ks in dis_k:
        c, h, w, d = ks.size()
        ks = ks.reshape((d * c, h, w,))
        ks = T.center_crop(ks, min_size)
        dc, h, w = ks.size()
        ks = ks.reshape((dc // d, h, w, d,))
        kspace_resized.append(ks.unsqueeze(0))
    dis_k = torch.cat(kspace_resized, dim=0)

    mask_resized = []
    for m in smp_m:
        dx = m.size()[0] - min_size[-1]
        if dx % 2:
            dx = dx // 2
            if m.size()[0] % 2:
                m = m[dx + 1:m.size()[0] - dx]
            elif min_size[-1] % 2:
                m = m[dx:m.size()[0] - dx - 1]
        else:
            dx = dx // 2
            m = m[dx:m.size()[0] - dx]
        mask_resized.append(T.center_crop(m, min_size).unsqueeze(0))
    smp_m = torch.cat(mask_resized, dim=0)

    mask_resized = []
    for m in acq_m:
        dx = m.size()[0] - min_size[-1]
        if dx % 2:
            dx = dx // 2
            if m.size()[0] % 2:
                m = m[dx + 1:m.size()[0] - dx]
            elif min_size[-1] % 2:
                m = m[dx:m.size()[0] - dx - 1]
        else:
            dx = dx // 2
            m = m[dx:m.size()[0] - dx]
        mask_resized.append(T.center_crop(m, min_size).unsqueeze(0))
    acq_m = torch.cat(mask_resized, dim=0)

    return DiscriminatorSample(
        ref_k, dis_k, smp_m, acq_m, metadata, fn, slice_idx
    )
