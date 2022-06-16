"""
Complex tensor manipulation functions from the fastMRI repository. Portions of
this code were taken from the fastMRI repository at https://github.com/faceboo
kresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import numpy as np
import torch
import torch.nn.functional as F
from math import ceil, floor
from typing import Tuple, Union


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and
    imaginary parts are stacked along the last dimension.
    Input:
        x: input numpy array.
    Returns:
        PyTorch representation of x.
    """
    if np.iscomplexobj(x):
        x = np.stack((x.real, x.imag), axis=-1)

    return torch.from_numpy(x)


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.
    The minimum is taken over `dim=-1` and `dim=-2`. If x is smaller than y at
    `dim=-1` and y is smaller than x at `dim=-2`, then the returned dimension
    will be a mixture of the two.
    Input:
        x: the first image.
        y: the second image.
    Returns:
        Tuple of tensors x and y, each cropped to the minimum size.
    """
    smallest_height = min(x.size()[-2], y.size()[-2])
    smallest_width = min(x.size()[-1], y.size()[-1])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def center_crop(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.
    Input:
        x: the input tensor to be center cropped. It should have at least
            2 dimensions and the cropping is applied along the last two
            dimensions.
        shape: the output shape.
    Returns:
        The center cropped image.
    """
    if x.size()[-2] == shape[0] and x.size()[-1] == shape[1]:
        return x

    if x.size()[-2] > shape[0]:
        h_from = floor((x.size()[-2] - shape[0]) / 2)
        h_to = h_from + shape[0]
        x = x[..., h_from:h_to, :]
    elif x.size()[-2] < shape[0]:
        h_left = ceil((shape[0] - x.size()[-2]) // 2)
        h_right = floor((shape[0] - x.size()[-2]) // 2)
        x = F.pad(x, (0, 0, h_left, h_right,))
    if x.size()[-1] > shape[1]:
        w_from = floor((x.size()[-1] - shape[1]) / 2)
        w_to = w_from + shape[1]
        x = x[..., w_from:w_to]
    elif x.size()[-1] < shape[1]:
        w_left = ceil((shape[1] - x.size()[-1]) // 2)
        w_right = floor((shape[1] - x.size()[-1]) // 2)
        x = F.pad(x, (w_left, w_right,))

    return x


def apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a 1D mask to an input tensor.
    Input:
        x: the input tensor of shape BHW or HW or BCHW2 or CHW2.
        mask: the input 1D tensor of shape BW or W.
    Returns:
        The masked input tensor with the same shape as the input tensor x.
    """
    return x * extend_dims(mask, x.size())


def extend_dims(
    x: torch.Tensor, ref_size: Union[torch.Size, tuple]
) -> torch.Tensor:
    """
    Extends an input tensor to the desired shape. Written specifically for 1D
    sampling masks to extend them to match a reference kspace dimension.
    Input:
        x: the input tensor (assumed to be of shape BW or W).
        ref_size: reference tensor size (assumed to be BHW or HW or BCHW2 or
            CHW2).
    Returns:
        x extended to the reference size.
    """
    if len(ref_size) == 2 and x.dim() == 1:
        h, w = ref_size
        assert tuple(x.size()) == (w,), (
            f"Incompatible input {x.size()} and ref {ref_size} dimensions."
        )
        return torch.vstack((x,) * h)
    elif len(ref_size) == 3 and x.dim() == 2:
        b, h, w = ref_size
        assert tuple(x.size()) == (b, w), (
            f"Incompatible input {x.size()} and ref {ref_size} dimensions."
        )
        return torch.cat((torch.unsqueeze(x, dim=1),) * h, dim=1)
    elif len(ref_size) == 4 and x.dim() == 1:
        c, h, w, d = ref_size
        assert tuple(x.size()) == (w,), (
            f"Incompatible input {x.size()} and ref {ref_size} dimensions."
        )
        return torch.cat((
            torch.unsqueeze(
                torch.cat(
                    (torch.unsqueeze(torch.vstack((x,) * h), dim=0),) * c,
                    dim=0
                ),
                dim=-1
            ),
        ) * d, dim=-1)
    elif len(ref_size) == 5 and x.dim() == 2:
        b, c, h, w, d = ref_size
        assert tuple(x.size()) == (b, w), (
            f"Incompatible input {x.size()} and ref {ref_size} dimensions."
        )
        return torch.cat((
            torch.unsqueeze(
                torch.cat((
                    torch.unsqueeze(
                        torch.cat((
                            torch.unsqueeze(x, dim=1),
                        ) * h, dim=1), dim=1
                    ),) * c, dim=1
                ), dim=-1
            ),
        ) * d, dim=-1)
    else:
        raise AssertionError(
            f"Incompatible input {x.size()} and ref {ref_size} dimensions."
        )


def scale_max_value(
    x: torch.Tensor, max_val: Union[torch.Tensor, float]
) -> torch.Tensor:
    """
    Scales x to have a maximum value of max_val.
    Input:
        x: input tensor.
        max_val: maximum value to scale x to.
    """
    if isinstance(max_val, float):
        max_val = abs(max_val)
    else:
        max_val = torch.abs(max_val)
    return torch.divide(max_val * x, torch.max(x)).type(x.dtype)
