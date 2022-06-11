"""
SSIM Loss module. Portions of this code, including certain choices for default
model parameters, were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Authors(s):
    Michael Yao

Licensed under the MIT License.
"""
import torch
from torch import nn
from torch.nn import functional as F


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
        # Iterate over the batch.
        for i, (x, y,) in enumerate(zip(X, Y)):
            ux = F.conv2d(x, self.w)  # E(X).
            uy = F.conv2d(y, self.w)  # E(Y).
            uxx = F.conv2d(x * x, self.w)  # E(X ** 2).
            uxy = F.conv2d(x * y, self.w)  # E(X * Y).
            uyy = F.conv2d(y * y, self.w)  # E(Y ** 2).
            vx = self.cov_norm * (uxx - (ux * ux))  # Var(X).
            vy = self.cov_norm * (uyy - (uy * uy))  # Var(Y).
            vxy = self.cov_norm * (uxy - (ux * uy))  # CoVar(X, Y).

            L = data_range[i]  # Dynamic range of pixel values.
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
    k2: float = 0.03
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
    Returns:
        SSIM(X, Y).
    """
    if X.size() != Y.size() or len(X.size()) != 3:
        raise ValueError(f"Unexpected input sizes {X.size()} and {Y.size()}")

    w = torch.ones((1, 1, win_size, win_size)) / win_size ** 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = w.to(device)

    # Normalization factor when calculating variance or covariance below.
    cov_norm = (win_size ** 2) / ((win_size ** 2) - 1)

    ux = F.conv2d(X, w)  # E(X).
    uy = F.conv2d(Y, w)  # E(Y).
    uxx = F.conv2d(X * X, w)  # E(X ** 2).
    uxy = F.conv2d(X * Y, w)  # E(X * Y).
    uyy = F.conv2d(Y * Y, w)  # E(Y ** 2).
    vx = cov_norm * (uxx - (ux * ux))  # Var(X).
    vy = cov_norm * (uyy - (uy * uy))  # Var(Y).
    vxy = cov_norm * (uxy - (ux * uy))  # CoVar(X, Y).

    L = data_range  # Dynamic range of pixel values.
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
