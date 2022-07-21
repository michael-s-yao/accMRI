"""
Implementation for a kspace discriminator model that learns kspace resampling
operations from input kspace data.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import torch
from torch import nn
from typing import Optional

from models.mlp import MLP
from models.conv import ConvBlock
from models.unet import NormUNet, UNet
from helper.utils import math as M


class Discriminator(nn.Module):
    """A kspace disciminator model."""

    def __init__(
        self,
        num_coils: Optional[int] = 1,
        chans: Optional[int] = 18,
        pools: Optional[int] = 4,
        in_chans: Optional[int] = None,
        out_chans: Optional[int] = None,
        model: str = "cnn"
    ):
        """
        Args:
            num_coils: number of coils (ie coil dimension).
            chans: number of intermediate channels for the discriminator.
            pools: number of down- and up- sampling layers for UNet
                architectures, or the number of total layers for MLP.
            in_chans: number of input channels into the MLP (required for MLP).
            out_chans: number of output channels from the MLP (required for
                MLP).
            model: model for discriminator, one of [`normunet`, `unet`, `mlp`,
                `cnn`].
        """
        super().__init__()

        if model.lower() == "mlp":
            self.feature_learning = None
            self.model = MLP(
                inchans=in_chans,
                outchans=out_chans,
                chans=chans,
                numlayers=pools
            )
            return
        elif model.lower() == "cnn":
            num_conv_layers = 2
            self.feature_learning = ConvBlock(
                in_chans=1,
                out_chans=1,
                pool_kernel=3,
                num_conv_layers=num_conv_layers
            )
            in_chans = in_chans // 2
            for _ in range(num_conv_layers):
                in_chans = in_chans // 2
            self.model = MLP(
                inchans=in_chans,
                outchans=out_chans,
                chans=chans,
                numlayers=pools
            )
            return
        # The factor of 2 comes from feeding in both the acquired and
        # acquiring kspace data.
        in_chans = 2 * num_coils
        if model.lower() == "unet":
            self.model = UNet(
                in_chans=in_chans,
                out_chans=1,
                init_conv_chans=chans,
                num_pool_layers=pools
            )
        elif model.lower() == "normunet":
            self.model = NormUNet(
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
        acquired_mask: Optional[torch.Tensor] = None,
        acquiring_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates a heat map of kspace fidelity.
        Input:
            acquired_kspace: previously acquired masked kspace data of
                shape BCWH2.
            acquiring_kspace: currently masked kspace data acquisition of
                unknown fidelity of shape BCHW2.
            acquired_mask: acquired data mask of shape BW. Used only by MLP.
            acquiring_mask: current step acquisition data mask of shape BW.
                Used only by MLP.
        Returns:
            Heat map of kspace fidelity of shape BHW. All values are between
            0.0 and 1.0. Values closer to 1.0 have lower fidelity.
        """
        if isinstance(self.model, MLP):
            # Estimate center mask from acquired_mask.
            batch_idxs, idxs = torch.nonzero(acquiring_mask, as_tuple=True)
            mid = acquired_mask.size()[-1] // 2
            heatmap = torch.zeros_like(acquiring_mask).type(torch.float32)
            heatmap = torch.unsqueeze(heatmap, dim=-1)
            for b, i in zip(batch_idxs, idxs):
                right = torch.nonzero(
                    acquired_mask[b, mid:] < 1.0, as_tuple=True
                )[0][0] + mid
                left = torch.nonzero(
                    torch.flip(acquired_mask[b, :mid], dims=[0]) < 1.0,
                    as_tuple=True
                )[0][0] + 1
                for center_idx in range(left, right):
                    acquiring_features = torch.flatten(
                        M.complex_abs(acquiring_kspace[b, :, :, i, :])
                    )
                    acquired_features = torch.flatten(
                        M.complex_abs(acquired_kspace[b, :, :, center_idx, :])
                    )
                    if self.feature_learning is not None:
                        features = torch.unsqueeze(
                            torch.cat((
                                torch.unsqueeze(acquiring_features, dim=-1),
                                torch.unsqueeze(acquired_features, dim=-1)
                            ), dim=-1), dim=0
                        )
                        features = self.feature_learning(features)
                    else:
                        features = torch.cat(
                            (acquiring_features, acquired_features,)
                        )
                    heatmap[b, i] += self.model(features, batch_dim=False)
                heatmap[b, i] = torch.divide(heatmap[b, i], right - left)
            heatmap = torch.squeeze(heatmap, dim=-1)

            h = acquired_kspace.size()[2]
            return torch.cat((torch.unsqueeze(heatmap, dim=1),) * h, dim=1)
        else:
            features = torch.cat([acquired_kspace, acquiring_kspace], dim=1)
            return torch.squeeze(self.model(features), dim=1)
