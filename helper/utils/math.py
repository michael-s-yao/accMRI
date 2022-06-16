"""
Complex tensor manipulation functions from the fastMRI repository. Portions of
this code were taken from the fastMRI repository at https://github.com/faceboo
kresearch/fastMRI. Herein, all references to "complex valued tensors" are real
arrays with the last dimension being the real representation of the complex
components.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import torch
from typing import Optional, List


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued tensor.
    Input:
        x: a complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        The absolute value of x.
    """
    if x.size()[-1] != 2:
        raise ValueError("Tensor does not have separate complex dimension.")

    return torch.sqrt(torch.sum(x * x, dim=-1))


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication of two input tensors.
    Input:
        x: a complex valued tensor, where the size of the final dimension
            should be 2.
        y: a complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        The complex product of x and y.
    """
    if x.size()[-1] != 2 or y.size()[-1] != 2:
        raise ValueError("Both tensors must have separate complex dimensions.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the complex conjugate of a complex valued tensor.
    Args:
        x: A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        The complex conjugate of x.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def rss(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) along the coil dimension.
    Args:
        x: the input tensor.
        dim: the coil dimension along which to apply the RSS transform.
    Returns:
        RSS(x).
    """
    return torch.sqrt((x * x).sum(dim))


def rss_complex(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for a complex tensor along
    the coil dimension.
    Args:
        x: the input complex valued tensor, where the size of the final
            dimension should be 2.
        dim: the coil dimension along which to apply the RSS transform.
    Returns:
        RSS(x).
    """
    abs_x = complex_abs(x)
    return torch.sqrt(torch.sum(abs_x * abs_x, dim=dim))


def fft2c(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2D fast fourier transform.
    Input:
        x: a complex valued tensor, where the size of the final dimension
            should be 2.
        norm: normalization mode (see `torch.fft.fft`).
    Returns:
        The fast fourier transform of x.
    """
    if x.size()[-1] != 2:
        raise ValueError("Tensor does not have separate complex dimension.")
    x = ifftshift(x, dim=[-3, -2])
    x = torch.view_as_real(
        torch.fft.fftn(torch.view_as_complex(x), dim=(-2, -1), norm=norm)
    )
    x = fftshift(x, dim=[-3, -2])

    return x


def ifft2c(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2D inverse fast fourier transform.
    Input:
        x: a complex valued tensor, where the size of the final dimension
            should be 2.
        norm: normalization mode (see `torch.fft.ifft`).
    Returns:
        The inverse fast fourier transform of x.
    """
    if x.size()[-1] != 2:
        raise ValueError("Tensor does not have separate complex dimension.")

    x = ifftshift(x, dim=[-3, -2])
    x = torch.view_as_real(
        torch.fft.ifftn(torch.view_as_complex(x), dim=(-2, -1), norm=norm)
    )
    x = fftshift(x, dim=[-3, -2])

    return x


def fftshift(
    x: torch.Tensor, dim: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Shift the zero-frequency component to the center of the spectrum.
    Input:
        x: an input tensor.
        dim: which dimension(s) to fftshift.
    Returns:
        An fftshifted version of x.
    """
    if dim is None:
        dim = []
        for i in range(x.dim()):
            dim.append(i)

    shift = []
    for dim_num in dim:
        shift.append(x.size()[dim_num] // 2)

    return roll(x, shift, dim)


def ifftshift(
    x: torch.Tensor, dim: Optional[List[int]] = None
) -> torch.Tensor:
    """
    The inverse of fftshift().
    Input:
        x: an input tensor.
        dim: which dimension(s) to ifftshift.
    Returns:
        An ifftshifted version of x.
    """
    if dim is None:
        dim = []
        for i in range(x.dim()):
            dim.append(i)

    shift = []
    for dim_num in dim:
        shift.append((x.size()[dim_num] + 1) // 2)

    return roll(x, shift, dim)


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Roll tensor elements along a given axis.
    Input:
        x: an input tensor.
        shift: amount to roll.
        dim: which dimension to roll.
    Returns:
        Rolled version of x.
    """
    shift = shift % x.size()[dim]
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(x: torch.Tensor, shift: List[int], dim: List[int]) -> torch.Tensor:
    """
    Roll tensor elements along a given set of axes.
    Input:
        x: an input tensor.
        shift: amount to roll for each dimension provided.
        dim: which dimensions to roll.
    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim).")

    for s, d in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x
