"""
Implementation for a kspace discriminator model that learns kspace resampling
operations from input kspace data.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import torch
from torch import nn
from models.unet import NormUNet, UNet


class Discriminator(nn.Module):
    """A kspace disciminator (potentially normalized) UNet model."""

    def __init__(
        self,
        num_coils: int = 1,
        chans: int = 18,
        pools: int = 4,
        model: str = "unet"
    ):
        """
        Args:
            num_coils: number of coils (ie coil dimension).
            chans: number of channels for the discriminator NormUNet.
            pools: number of down- and up- sampling layers for the
                regulatization NormUNet.
            model: model for discriminator, one of ["normunet", "unet"].
        """
        super().__init__()

        # One factor of 2 comes from the real and imaginary axis.
        # The other factor comes from feeding in both the acquired and
        # acquiring kspace data.
        in_chans = 2 * 2 * num_coils
        if model.lower() == "unet":
            self.unet = UNet(
                in_chans=in_chans,
                out_chans=1,
                init_conv_chans=chans,
                num_pool_layers=pools
            )
        elif model.lower() == "normunet":
            self.unet = NormUNet(
                in_chans=in_chans,
                out_chans=1,
                init_conv_chans=chans,
                num_pool_layers=pools
            )
        else:
            raise ValueError(f"Unrecognized discriminator model {model}.")

    def forward(
        self,
        acquired_kspace: torch.Tensor,
        acquiring_kspace: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generates a heat map of kspace fidelity.
        Input:
            acquired_kspace: previously acquired masked kspace data of
                shape BCWH2.
            acquiring_kspace: currenty masked kspace data acquisition of
                unknown fidelity of shape BCHW2.
        Returns:
            Heat map of kspace fidelity of shape BHW. All values are between
            0.0 and 1.0. Values closer to 1.0 have lower fidelity.
        """
        features = torch.cat([acquired_kspace, acquiring_kspace], dim=1)
        return torch.sigmoid(torch.squeeze(self.unet(features), dim=1))
