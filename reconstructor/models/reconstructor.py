"""
Implementation of the image reconstruction module.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import sys
import torch
from torch import nn
from typing import Optional
from models.unet import UNet
from models.varnet import VarNet
from models.sens import SensitivityModel

sys.path.append("..")
from helper.utils.math import (
    complex_mul, ifft2c, complex_conj, rss, complex_abs
)


class Reconstructor(nn.Module):
    """An image reconstruction module for accelerated MRI."""

    def __init__(
        self,
        is_multicoil: bool = False,
        model: str = "varnet",
        sens_chans: int = 8,
        sens_pools: int = 4,
        sens_drop_prob: float = 0.0,
        mask_center: bool = True,
        **kwargs
    ):
        """
        Args:
            is_multicoil: whether we are using multicoil data or not.
            model: reconstructor model. One of [`varnet`, `unet`].
            sens_chans: number of channels for sensitivity map UNet.
            sens_pools: number of down- and up- sampling layers for sensitivity
                map UNet.
            sens_drop_prob: dropout probability for sensitivity map UNet.
            mask_center: whether to mask center of kspace during sensitivity
                map calculation.
        """
        super().__init__()

        self.is_multicoil = is_multicoil

        if model.lower() == "varnet":
            self.model = VarNet(is_multicoil=self.is_multicoil, **kwargs)
        elif model.lower() == "unet":
            self.model = UNet(**kwargs)
        else:
            raise ValueError(f"Unrecognized model {model}.")

        # Include a sensitivity estimation module if multicoil input.
        self.sens_net = None
        if self.is_multicoil:
            self.sens_net = SensitivityModel(
                init_conv_chans=sens_chans,
                num_pool_layers=sens_pools,
                drop_prob=sens_drop_prob,
                mask_center=mask_center,
            )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        center_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reconstructs the image from kspace and estimates coil sensitivity maps
        (if data is multicoil).
        Input:
            masked_kspace: currently acquired kspace data.
            mask: acquisition mask of acquired kspace data.
            center_mask: mask of acquired center low-frequency kspace data.
        Returns:
            Estimated image reconstruction of estimated full kspace data.
        """

        # Calculate coil sensitivity maps.
        sens_maps = None
        if self.is_multicoil:
            sens_maps = self.sens_net(masked_kspace, mask, center_mask)

        if isinstance(self.model, VarNet):
            return self.model(masked_kspace, mask, center_mask, sens_maps)
        elif isinstance(self.model, UNet):
            # Start with zero-filled reconstruction.
            if self.is_multicoil:
                image = self.sens_reduce(masked_kspace, sens_maps)
            else:
                image = ifft2c(masked_kspace)
            cimg = self.chan_complex_to_last_dim(
                self.model(self.complex_to_chan_dim(image))
            )
            return rss(complex_abs(cimg), dim=1)
        else:
            raise ValueError(f"Unrecognized model type {type(self.model)}.")

    def sens_reduce(
        self, x: torch.Tensor, sens_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the reduce operator that combines the individual coil images.
        This function is the same as the `VarNetBlock.sens_reduce()` function
        in `models/varnet.py`.
        Input:
            x: input of individual coil kspace data.
            sens_maps: computed sensitivity maps.
        Returns:
            Combined image.
        """
        return complex_mul(
            ifft2c(x), complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes a tensor representing complex data of shape BCHW2 into
        B(2C)HW.
        Input:
            x: input tensor of shape BCHW2.
        Returns:
            Output tensor of shape B(2C)HW.
        """
        b, c, h, w, d = x.size()
        if d != 2:
            raise RuntimeError(f"Unexpected dimensions {x.size()}")
        return torch.reshape(
            torch.permute(x, (0, 4, 1, 2, 3,)), (b, 2 * c, h, w,)
        )

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes a tensor  representing complex data of shape B(2C)HW into
        BCHW2.
        Input:
            x: input tensor of shape B(2C)HW.
        Returns:
            Output tensor of shape BCHW2 (contiguous in memory).
        """
        b, c2, h, w = x.size()
        if c2 % 2 != 0:
            raise RuntimeError(f"Unexpected dimensions {x.size()}")
        return torch.permute(
            x.view(b, 2, c2 // 2, h, w), (0, 2, 3, 4, 1)
        ).contiguous()
