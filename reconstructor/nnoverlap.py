"""
Standalone script for neural network overlap for different tasks based on
Fisher Information Matrix analysis.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import pickle
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Sequence, Union

sys.path.append("..")
from data.dataset import ReconstructorDataset
from data.transform import ReconstructorDataTransform
from models.loss import structural_similarity
from pl_modules.reconstructor_module import ReconstructorModule
from tools import transforms as T


class FIM:
    """Diagonal Fisher Information Matrix representation of neural networks."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        fim_cache_path: Optional[Union[Path, str]] = None,
        device: str = "cuda",
        normalize: bool = True
    ):
        """
        Args:
            model: neural network model (torch.nn.Module).
            dataloader: training DataLoader.
            fim_cache_path: if specified, Fisher information matrix will be
                read from (or written to) this file.
            device: PyTorch device. Default `cuda`.
            normalize: whether to normalize the sparse matrix to unit trace.
        """
        self.dataloader = dataloader
        self.fim_cache_path = fim_cache_path
        self.device = device

        self.diag_fim = {}
        if fim_cache_path is not None and os.path.isfile(fim_cache_path):
            with open(fim_cache_path, "rb") as f:
                self.diag_fim = pickle.load(f)
            print(f"Using FIM Cache File {os.path.abspath(fim_cache_path)}")

        self.model_params = {
            n: p.clone().detach()
            for n, p in model.named_parameters() if p.requires_grad
        }
        for name, param in deepcopy(self.model_params).items():
            param.data.zero_()
            self.diag_fim[name] = param.data.to(device)

        self.model = deepcopy(model).to(device)
        self.model.eval()
        for item in tqdm(dataloader, desc="FIM Construction"):
            self.model.zero_grad()
            output = self.model(
                item.masked_kspace.to(device),
                item.mask.to(device),
                item.center_mask.to(device)
            )

            loss = torch.log(
                1 - structural_similarity(
                    T.center_crop(output.cpu(), item.crop_size),
                    T.center_crop(item.target.cpu(), item.crop_size),
                    data_range=item.max_value
                )
            )
            loss.backward()

            for name, param in self.model.named_parameters():
                self.diag_fim[name] += (
                    param.grad.data * param.grad.data / len(dataloader)
                )

        if normalize:
            tr = FIM.trace(self)
            self.diag_fim = {
                name: torch.divide(param, tr)
                for name, param in self.diag_fim.items()
            }

        if fim_cache_path is not None:
            with open(fim_cache_path, "w+b") as f:
                diag_fim_cpu = {
                    name: param.cpu() for name, param in self.diag_fim.items()
                }
                pickle.dump(diag_fim_cpu, f)
                print(
                    "Saved FIM Cache File to",
                    f"{os.path.abspath(fim_cache_path)}"
                )

    @staticmethod
    def trace(diag_fim: Union[FIM, dict]) -> float:
        """
        Computes the trace of a Fisher Information Matrix representation.
        Input:
            diag_fim: a diagonal sparse FIM representation.
        Returns:
            Trace of the FIM representation.
        """
        if isinstance(diag_fim, FIM):
            return sum([torch.sum(x) for x in diag_fim.diag_fim.values()])
        return sum([torch.sum(x) for x in diag_fim.values()])

    @staticmethod
    def sqrt(diag_fim: Union[FIM, dict]) -> dict:
        """
        Computes the square root of a Fisher Information Matrix representation.
        It is assumed that the matrix only has nonzero values along its
        diagonal.
        Input:
            diag_fim: a diagonal sparse FIM representation.
        Returns:
            Square root of the FIM representation.
        """
        if isinstance(diag_fim, FIM):
            diag_fim = diag_fim.diag_fim
        return {
            name: torch.sqrt(diag_fim[name]) for name in diag_fim.keys()
        }

    def __matmul__(self, other: Union[FIM, dict]) -> dict:
        """
        Computes the product of the sparse FIM with another sparse FIM. It is
        assumed that both matrices only have nonzero values along their
        respective diagonals.
        Input:
            other: a diagonal sparse FIM representation.
        Returns:
            The matrix product self @ other.
        """
        if self.model_params.keys() != other.model_params.keys():
            raise ValueError("Keys of matrix representations do not match!")
        return {
            name: self.diag_fim[name] * other.diag_fim[name]
            for name in self.model_params.keys()
        }


def frechet_overlap(
    models: Sequence[nn.Module],
    dataloaders: Sequence[DataLoader],
    fim_cache_paths: Optional[Sequence[Union[Path, str]]] = None,
    overlap_save_path: Optional[Union[Path, str]] = None,
    device: str = "cuda"
):
    """
    Computes the Fisher Overlap between two neural networks.
    Input:
        models: list of the 2 neural network models (torch.nn.Module).
        dataloader: list of the 2 training DataLoaders.
        fim_cache_paths: if specified, Fisher information matrices will be
            read from (or written to) these files.
        overlap_save_path: optional path to write the overlap matrix to.
        device: PyTorch device. Default `cuda`.
    Returns:
        The Fisher Overlap score between the two neural networks.

    Citation:
    Kirkpatrick J, Pascanu R et al. Overcoming catastrophic forgetting in
    neural networks. Proc Natl Acad Sci 114(13): 3521-6, 2017. doi: 10.1073/
    pnas.1611835114
    """
    modelA, modelB = models[0].to(device), models[-1].to(device)
    dataloaderA, dataloaderB = dataloaders
    if fim_cache_paths is None:
        fimA_cache_path, fimB_cache_path = None, None
    else:
        fimA_cache_path, fimB_cache_path = fim_cache_paths

    F_A = FIM(
        modelA, dataloaderA, fimA_cache_path, device=device, normalize=True
    )
    F_B = FIM(
        modelB, dataloaderB, fimB_cache_path, device=device, normalize=True
    )

    Omega = FIM.sqrt(F_A @ F_B)
    if overlap_save_path is not None:
        with open(overlap_save_path, "w+b") as f:
            pickle.dump(Omega, f)
    return FIM.trace(Omega)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analytical Tool for Neural Network Fisher Overlap"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=("cuda", "cpu"),
        default="cuda",
        help="Whether to use GPU or CPU. GPU is used by default if available."
    )
    parser.add_argument(
        "--model_checkpoints",
        type=str,
        nargs=2,
        default=[
            "./ckpts/multicoil_knee.ckpt", "./ckpts/multicoil_knee.ckpt"
        ],
        help="Reconstructor model checkpoints to compare."
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        nargs=2,
        required=True,
        help="Reconstructor model datasets."
    )
    parser.add_argument(
        "--data_cache_paths",
        type=str,
        nargs=2,
        default=["./dataset_cache.pkl", "./dataset_cache.pkl"],
        help="Optional dataset cache paths for faster load times."
    )
    parser.add_argument(
        "--fim_cache_paths",
        type=str,
        nargs=2,
        default=["./knee_fim_cache.pkl", "./brain_fim_cache.pkl"],
        help="FIM cache paths."
    )
    parser.add_argument(
        "--overlap_savepath",
        type=str,
        default=None,
        help="Optional path to save the overlap matrix to. Default not saved."
    )
    parser.add_argument(
        "--center_crop",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Image center crop. Default 256 by 256."
    )
    fixed_acceleration_help = "Fixed acceleration factor(s) to use. "
    fixed_acceleration_help += "Default variable acceleration."
    parser.add_argument(
        "--fixed_acceleration",
        type=float,
        nargs="+",
        default=None,
        help=fixed_acceleration_help
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed. Default 42."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()

    torch.manual_seed(args.seed)
    if args.device != "cuda" or not torch.cuda.is_available():
        num_gpus = 0
    else:
        num_gpus = int(torch.cuda.device_count())
    transform = ReconstructorDataTransform(
        center_fractions=[0.08, 0.04],
        center_crop=args.center_crop,
        fixed_acceleration=args.fixed_acceleration,
        seed=args.seed
    )

    models = [
        ReconstructorModule.load_from_checkpoint(args.model_checkpoints[i])
        for i in range(len(args.model_checkpoints))
    ]
    dataloaders = [
        DataLoader(
            ReconstructorDataset(
                args.data_paths[i],
                transform=transform,
                seed=args.seed,
                multicoil=bool("multicoil" in args.data_paths[i]),
                num_gpus=num_gpus,
                dataset_cache_file=args.data_cache_paths[i]
            )
        )
        for i in range(len(models))
    ]
    Omega = frechet_overlap(
        models,
        dataloaders,
        args.fim_cache_paths,
        args.overlap_savepath,
        args.device
    )

    print(f"Frechet Overlap: {Omega}")
