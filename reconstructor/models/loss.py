"""
SSIM and EWC Loss modules. Portions of this code, including certain choices for
default model parameters, were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Authors(s):
    Michael Yao

Licensed under the MIT License.
"""
from copy import deepcopy
import os
from pathlib import Path
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from typing import Optional, Union

from helper.utils import transforms as T


class SSIMLoss(nn.Module):
    """SSIM (structural similarity index measure) loss module."""

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer(
            "w", torch.ones((1, 1, win_size, win_size)) / win_size ** 2
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.w = self.w.to(device)
        # Normalization factor when calculating variance or covariance below.
        self.cov_norm = (win_size ** 2) / ((win_size ** 2) - 1)

    def forward(
        self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates SSIM(X, Y). If there are multiple batches, calculates the
        mean(SSIM(X, Y)) over the entire batch. Returns 1 - SSIM in order
        to minimize loss.
        Input:
            X: input image.
            Y: input image.
            data_range: dynamic range of the pixel values.
        Returns:
            1 - SSIM(X, Y).
        """
        ssims = []
        for i, (x, y,) in enumerate(zip(X, Y)):
            x, y = torch.unsqueeze(x, dim=0), torch.unsqueeze(y, dim=0)
            ux = F.conv2d(x, self.w)  # E(X).
            uy = F.conv2d(y, self.w)  # E(Y).
            uxx = F.conv2d(x * x, self.w)  # E(X ** 2).
            uxy = F.conv2d(x * y, self.w)  # E(X * Y).
            uyy = F.conv2d(y * y, self.w)  # E(Y ** 2).
            vx = self.cov_norm * (uxx - (ux * ux))  # Var(X).
            vy = self.cov_norm * (uyy - (uy * uy))  # Var(Y).
            vxy = self.cov_norm * (uxy - (ux * uy))  # CoVar(X, Y).

            # Dynamic range of pixel values.
            L = data_range[i, None, None, None]
            C1 = (self.k1 * L) ** 2  # (k1 * L) ** 2.
            C2 = (self.k2 * L) ** 2  # (k1 * L) ** 2.
            N1, N2, D1, D2 = (
                (2 * ux * uy) + C1,
                (2 * vxy) + C2,
                (ux * ux) + (uy * uy) + C1,
                vx + vy + C2,
            )
            D = D1 * D2
            S = (N1 * N2) / D
            ssims.append(S.mean())

        return 1 - (sum(ssims) / len(ssims))


def structural_similarity(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: torch.Tensor,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
    device: str = None,
) -> float:
    """
    Calculates SSIM(X, Y). If there are multiple batches, calculates the
    mean(SSIM(X, Y)) over the entire batch.
    Input:
        X: input image of dimensions 1HW.
        Y: input image 1HW.
        data_range: dynamic range of the pixel values.
        win_size: window size for SSIM calculation.
        k1: k1 parameter for SSIM calculation.
        k2: k2 parameter for SSIM calculation.
        device: device specification. Uses CUDA if available by default.
    Returns:
        SSIM(X, Y).
    """
    if X.size() != Y.size() or len(X.size()) != 3:
        raise ValueError(f"Unexpected input sizes {X.size()} and {Y.size()}")

    w = torch.ones((1, 1, win_size, win_size)) / win_size ** 2
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    w = w.to(device)

    # Normalization factor when calculating variance or covariance below.
    cov_norm = (win_size ** 2) / ((win_size ** 2) - 1)

    X = torch.unsqueeze(X.to(device), dim=0)
    Y = torch.unsqueeze(Y.to(device), dim=0)
    ux = F.conv2d(X, w)  # E(X).
    uy = F.conv2d(Y, w)  # E(Y).
    uxx = F.conv2d(X * X, w)  # E(X ** 2).
    uxy = F.conv2d(X * Y, w)  # E(X * Y).
    uyy = F.conv2d(Y * Y, w)  # E(Y ** 2).
    vx = cov_norm * (uxx - (ux * ux))  # Var(X).
    vy = cov_norm * (uyy - (uy * uy))  # Var(Y).
    vxy = cov_norm * (uxy - (ux * uy))  # CoVar(X, Y).

    L = data_range.to(device)  # Dynamic range of pixel values.
    C1 = (k1 * L) ** 2  # (k1 * L) ** 2.
    C2 = (k2 * L) ** 2  # (k1 * L) ** 2.
    N1, N2, D1, D2 = (
        (2 * ux * uy) + C1,
        (2 * vxy) + C2,
        (ux * ux) + (uy * uy) + C1,
        vx + vy + C2,
    )
    D = D1 * D2
    S = (N1 * N2) / D

    return S.mean()


class EWCLoss(nn.Module):
    """
    EWC (elastic weight consolidation) loss module based on @moskomule's
    implementation at https://github.com/moskomule/ewc.pytorch.

    Citation:
    Kirkpatrick J et al. (2017). Overcoming catastrophic forgetting in neural
    networks. https://arxiv.org/pdf/1612.00796.pdf
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        lambda_: float,
        fim_cache_path: Optional[Union[Path, str]] = None,
    ):
        """
        Args:
            model: neural network model (torch.nn.Module).
            dataloader: training DataLoader.
            lambda_: relative loss scaling term.
            fim_cache_path: if specified, Fisher information matrix will be
                read from (or written to) this file.
        """
        super().__init__()

        self.model = model
        self.dataloader = dataloader
        self.lambda_ = lambda_
        self.fim_cache_path = fim_cache_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.means = {}
        self.diag_fim = self.compute_fischer()

        for name, param in deepcopy(self.init_params).items():
            self.means[name] = param.data

    def compute_fischer(self):
        """
        Computes the diagonal Fisher Information Matrix of the neural network.
        Input:
            None.
        Returns:
            Diagonal FIM representation of the neural network.
        """
        diag_fim = {}
        valid_path = self.fim_cache_path is not None
        valid_path = valid_path and os.path.isfile(self.fim_cache_path)
        if valid_path:
            with open(self.fim_cache_path, "rb") as f:
                diag_fim = pickle.load(f)
            print(
                "Using FIM Cache File",
                f"{os.path.abspath(self.fim_cache_path)}"
            )
            return diag_fim

        for name, param in deepcopy(self.init_params).items():
            param.data.zero_()
            diag_fim[name] = param.data

        self.model.eval()
        for item in tqdm(self.dataloader, desc="FIM Construction"):
            self.model.zero_grad()
            output = self.model(
                item.masked_kspace, item.mask, item.center_mask
            )

            loss = 1 - structural_similarity(
                T.center_crop(output, item.crop_size),
                T.center_crop(item.target, item.crop_size),
                data_range=item.max_value
            )
            loss.backward()

            for name, param in self.model.named_parameters():
                diag_fim[name] += (param.grad.data ** 2) / len(self.dataloader)
        self.model.zero_grad()
        self.model.train()

        if self.fim_cache_path is not None:
            with open(self.fim_cache_path, "w+b") as f:
                pickle.dump(diag_fim, f)
                print(
                    "Saved FIM Cache File to",
                    f"{os.path.abspath(self.fim_cache_path)}"
                )

        return diag_fim

    def forward(self, model: nn.Module):
        """
        Computes the elastic weight consolidation loss term.
        Input:
            model: neural network with updated weights.
        Returns:
            Elastic weight consolidation (EWC) loss term.
        """
        loss = 0.0
        for name, param in model.named_parameters():
            loss += torch.sum(
                self.diag_fim[name].to(self.device) * torch.square(
                    param.to(self.device) - self.means[name].to(self.device)
                )
            )
        return (self.lambda_ / 2.0) * loss
