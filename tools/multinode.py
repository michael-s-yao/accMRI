"""
Define utility function(s) for multinode model training.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
from math import ceil
import numpy as np
import torch


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    slices_per_worker = int(
        ceil(len(worker_info.dataset) / float(worker_info.num_workers))
    )
    worker_info.dataset.start = worker_id * slices_per_worker
    worker_info.dataset.end = min(
        worker_info.dataset.start + slices_per_worker, len(worker_info.dataset)
    )
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True
    seed = worker_info.seed

    num_workers = worker_info.num_workers
    if worker_info.dataset.transform is not None:
        if is_ddp:
            seed += torch.distributed.get_rank() * num_workers
        seed = seed % ((2 ** 32) - 1)
        worker_info.dataset.transform.rng = np.random.RandomState(seed)
    return
