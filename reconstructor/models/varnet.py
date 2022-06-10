"""
Implementation for variational network image reconstructor. Our model
applies a combination of soft data consistency with the input model as a
regularizer. Portions of this code, including certain choices for default
model parameters, were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import sys
import torch
from torch import nn
from typing import Optional
from models.unet import NormUNet

sys.path.append("..")
from common.utils.math import (
    complex_mul, fft2c, ifft2c, complex_conj, rss, complex_abs
)
import common.utils.transforms as T


class VarNet(nn.Module):
    """
    A end-to-end variational network model for accelerated MRI reconstruction
    and coil sensitivity estimation.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
        is_multicoil: bool = False
    ):
        """
        Args:
            num_cascades: number of cascades (ie layers) for the variational
                network.
            chans: number of channels for the regularization NormUNet's.
            pools: number of down- and up- sampling layers for the
                regulatization NormUNet.
            is_multicoil: whether or not the input data is multicoil.
        """
        super().__init__()

        self.cascades = nn.ModuleList(
            [
                VarNetBlock(
                    NormUNet(
                        init_conv_chans=chans,
                        num_pool_layers=pools
                    ),
                    is_multicoil=is_multicoil
                ) for _ in range(num_cascades)
            ]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        center_mask: Optional[torch.Tensor] = None,
        sens_maps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimates coil sensitivity maps and samples addition kspace data, and
        then estimates the rest of kspace and returns the final estimated
        image reconstruction.
        Input:
            masked_kspace: currently acquired kspace data.
            mask: acquisition mask of acquired kspace data.
            center_mask: mask of acquired center low-frequency kspace data.
            sens_maps: estimated sensitivity maps.
        Returns:
            Estimated image reconstruction of estimated full kspace data.
        """
        # Calculate all of kspace using acquired kspace data.
        pred_kspace = masked_kspace.clone()
        for cascade in self.cascades:
            pred_kspace = cascade(
                pred_kspace, masked_kspace, mask, sens_maps
            )

        return rss(complex_abs(ifft2c(pred_kspace)), dim=1)


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer.
    """

    def __init__(self, model: nn.Module, is_multicoil: bool = False):
        """
        Args:
            model: module for the regularization component of the variational
                network.
            is_multicoil: whether or not the input data is multicoil.
        """
        super().__init__()

        self.model = model
        self.is_multicoil = is_multicoil
        self.dc_weight = nn.Parameter(torch.ones(1))

    @staticmethod
    def sens_expand(
        x: torch.Tensor, sens_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the expand operator that computes the images seen by each coil
        from the image and sensitivity maps.
        Input:
            x: input image.
            sens_maps: computed sensitivity maps.
        Returns:
            Individual coil images.
        """
        return fft2c(complex_mul(x, sens_maps))

    @staticmethod
    def sens_reduce(
        x: torch.Tensor, sens_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the reduce operator that combines the individual coil images.
        This function is the same as the `Reconstructor.sens_reduce()` function
        in `models/reconstructor.py`.
        Input:
            x: input of individual coil kspace data.
            sens_maps: computed sensitivity maps.
        Returns:
            Combined image.
        """
        return complex_mul(
            ifft2c(x), complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)

    def forward(
        self,
        pred_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input:
            pred_kspace: current prediction of full kspace data.
            ref_kspace: ground truth of all acquired masked kspace data.
            mask: acquisition mask of acquired kspace data.
            sens_maps: computed sensitivity maps (irrelevant for single coil).
        Returns:
            Estimated reconstruction of masked_kspace data.
        """
        zeros = torch.zeros_like(pred_kspace).to(pred_kspace)
        soft_dc = self.dc_weight * torch.where(
            T.extend_dims(mask, ref_kspace.size()) > 0.0,
            pred_kspace - ref_kspace,
            zeros
        )
        if self.is_multicoil:
            model_term = VarNet.sens_expand(
                self.model(VarNet.sens_reduce(pred_kspace, sens_maps)),
                sens_maps
            )
        else:
            model_term = self.model(pred_kspace)

        return pred_kspace - soft_dc - model_term
