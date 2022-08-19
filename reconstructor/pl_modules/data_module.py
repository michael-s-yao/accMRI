"""
Data module to train, validate, and test image reconstruction models.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import os
from torch.utils.data import DataLoader
from pathlib import Path
import pytorch_lightning as pl
import torch
from typing import Optional, Sequence, Union
from data.dataset import ReconstructorDataset
from data.sldataset import SheppLoganDataset
from data.transform import ReconstructorDataTransform

from tools.multinode import worker_init_fn


class DataModule(pl.LightningDataModule):
    """Data module for data sets as input to image reconstruction models."""

    def __init__(
        self,
        data_path: Union[Path, str] = "",
        cache_path: Optional[str] = None,
        train_dir: str = "multicoil_train",
        val_dir: str = "multicoil_val",
        test_dir: str = "multicoil_test",
        batch_size: int = 1,
        center_crop: Union[torch.Size, tuple] = (640, 368,),
        fixed_acceleration: Optional[Sequence[float]] = None,
        num_workers: int = 4,
        seed: Optional[int] = None,
        fast_dev_run: bool = False,
        num_gpus: int = 0,
        tl: bool = False,
        num_slices_per_phantom: Optional[int] = 40,
        num_coils: Optional[int] = 4,
        distributed_sampler: bool = False,
    ):
        """
        Args:
            data_path: a string path to the reconstruction dataset.
            cache_path: optional dataset cache file to use for faster dataset
                load times.
            train_dir: a string path to the training subdirectory.
            val_dir: a string path to the validation subdirectory.
            test_dir: a string path to the test set subdirectory.
            batch_size: batch size.
            center_crop: an optional tuple of shape HW to crop the input kspace
                size to.
            fixed_acceleration: optional fixed acceleration factor. Default to
                variable acceleration factor (ie a random number of kspace
                between min_lines_acquired and W will be acquired).
            num_workers: number of workers for PyTorch dataloaders.
            seed: optional random seed for determining order of dataset.
            fast_dev_run: whether we are running a test fast_dev_run.
            num_gpus: number of GPUs in use.
            tl: specify transfer learning pre-training using Shepp-Logan
                phantoms.
            num_slices_per_phantom: if using phantom data, specify number of
                slices to take per 3D simulated phantom.
            num_coils: if using phantom data, specify number of coils to
                simulate.
            distributed_sampler: whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        self.data_path = str(data_path)
        self.root_path = Path(self.data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        # Sizes for transfer learning set arbitrarily to be the same as the
        # number of slices in the original fastMRI dataset.
        self.fastMRI_size = {
            "train": 70_748, "val": 21_842, "test": 8_852
        }
        self.seed = seed
        self.fast_dev_run = fast_dev_run
        self.num_gpus = num_gpus
        self.cache_path = cache_path

        # If all of train_dir, val_dir, and test_dir are None, then we are
        # predicting reconstructions.
        if train_dir is None and val_dir is None and test_dir is None:
            if tl:
                self.predict = SheppLoganDataset(
                    num_samples=self.fastMRI_size["test"],
                    num_slices=num_slices_per_phantom,
                    num_coils=num_coils,
                    center_fractions=[0.08, 0.04],
                    fixed_accelerations=fixed_acceleration,
                    slice_size=center_crop,
                    min_lines_acquired=16,
                    seed=(seed + sum([ord(c) for c in "predict"]))
                )
                return
            multicoil = "multicoil" in data_path.lower()
            self.predict = ReconstructorDataset(
                self.data_path,
                ReconstructorDataTransform(
                    center_fractions=[0.08, 0.04],
                    center_crop=center_crop,
                    fixed_acceleration=fixed_acceleration,
                    seed=seed,
                ),
                seed=seed,
                multicoil=multicoil
            )
            return

        multicoil = "multicoil" in data_path.lower()
        multicoil = multicoil or "multicoil" in train_dir.lower()
        if tl:
            self.train_transform = None
            self.train = SheppLoganDataset(
                num_samples=self.fastMRI_size["train"],
                num_slices=num_slices_per_phantom,
                num_coils=num_coils,
                center_fractions=[0.08, 0.04],
                fixed_accelerations=fixed_acceleration,
                slice_size=center_crop,
                min_lines_acquired=16,
                seed=(seed + sum([ord(c) for c in "train"]))
            )
        else:
            self.train_transform = ReconstructorDataTransform(
                center_fractions=[0.08, 0.04],
                center_crop=center_crop,
                fixed_acceleration=fixed_acceleration,
                min_lines_acquired=16,
                seed=seed
            )
            self.train_dir = train_dir
            self.train = ReconstructorDataset(
                str(os.path.join(self.data_path, train_dir)),
                self.train_transform,
                seed=seed,
                fast_dev_run=fast_dev_run,
                multicoil=multicoil,
                num_gpus=num_gpus,
                dataset_cache_file=cache_path
            )
        if tl:
            self.val_transform = None
            self.val = SheppLoganDataset(
                num_samples=self.fastMRI_size["val"],
                num_slices=num_slices_per_phantom,
                num_coils=num_coils,
                center_fractions=[0.08, 0.04],
                fixed_accelerations=fixed_acceleration,
                slice_size=center_crop,
                min_lines_acquired=16,
                seed=(seed + sum([ord(c) for c in "val"]))
            )
        else:
            self.val_transform = ReconstructorDataTransform(
                center_fractions=[0.08, 0.04],
                center_crop=center_crop,
                fixed_acceleration=fixed_acceleration,
                min_lines_acquired=16,
                seed=seed,
            )
            self.val_dir = val_dir
            self.val = ReconstructorDataset(
                str(os.path.join(self.data_path, val_dir)),
                self.val_transform,
                seed=seed,
                fast_dev_run=fast_dev_run,
                multicoil=multicoil,
                num_gpus=num_gpus,
                dataset_cache_file=cache_path
            )
        if tl:
            self.test_transform = None
            self.test = SheppLoganDataset(
                num_samples=self.fastMRI_size["test"],
                num_slices=num_slices_per_phantom,
                num_coils=num_coils,
                center_fractions=[0.08, 0.04],
                fixed_accelerations=fixed_acceleration,
                slice_size=center_crop,
                min_lines_acquired=16,
                seed=(seed + sum([ord(c) for c in "test"]))
            )
        else:
            self.test_transform = ReconstructorDataTransform(
                center_crop=center_crop, seed=seed
            )
            self.test_dir = test_dir
            self.test = ReconstructorDataset(
                str(os.path.join(self.data_path, test_dir)),
                self.test_transform,
                seed=seed,
                fast_dev_run=fast_dev_run,
                multicoil=multicoil,
                num_gpus=num_gpus,
                dataset_cache_file=cache_path
            )
        self.predict = None

        self.train_sampler, self.val_sampler = None, None
        if self.distributed_sampler:
            self.train_sampler = torch.utils.data.DistributedSampler(
                self.train
            )
            self.val_sampler = torch.utils.data.DistributedSampler(self.val)

    def train_dataloader(self):
        """Returns the training dataloader."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=self.train_sampler,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        """Returns the validation dataloader."""
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.val_sampler,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self):
        """Return the test set dataloader."""
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def predict_dataloader(self):
        """Return the prediction set dataloader."""
        if self.predict is None:
            return None
        return DataLoader(self.predict, num_workers=self.num_workers)

    def prepare_data(self):
        if self.cache_path is None or not os.path.isfile(self.cache_path):
            return
        multicoil = "multicoil" in self.train_dir.lower()
        if not isinstance(self.train, SheppLoganDataset):
            _ = ReconstructorDataset(
                str(os.path.join(self.data_path, self.train_dir)),
                self.train_transform,
                seed=self.seed,
                fast_dev_run=self.fast_dev_run,
                multicoil=multicoil,
                num_gpus=self.num_gpus,
                dataset_cache_file=self.cache_path
            )
        if not isinstance(self.val, SheppLoganDataset):
            _ = ReconstructorDataset(
                str(os.path.join(self.data_path, self.val_dir)),
                self.val_transform,
                seed=self.seed,
                fast_dev_run=self.fast_dev_run,
                multicoil=multicoil,
                num_gpus=self.num_gpus,
                dataset_cache_file=self.cache_path
            )
        if not isinstance(self.test, SheppLoganDataset):
            _ = ReconstructorDataset(
                str(os.path.join(self.data_path, self.test_dir)),
                self.test_transform,
                seed=self.seed,
                fast_dev_run=self.fast_dev_run,
                multicoil=multicoil,
                num_gpus=self.num_gpus,
                dataset_cache_file=self.cache_path
            )
