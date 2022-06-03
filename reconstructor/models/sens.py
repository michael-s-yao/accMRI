"""
Model for learning coil sensitivity maps from kspace data. This model applies
an IFFT to multi-coil kspace data and then a normalized UNet to the coil images
to estimate coil sensitivities. Portions of this code, including certain
choices for default model parameters, were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Authors(s):
    Michael Yao

Licensed under the MIT License.

Sriram A, Zbontar J, Murrell T, Defazio A, Zitnick CL, ... Johnson P. (2020).
End-to-End Variational Networks for Accelerated MRI Reconstruction. MICCAI
64-73.
"""
import numpy as np
import sys
import torch
from torch import nn
from typing import Optional, Tuple
from models.unet import NormUNet

sys.path.append("..")
from common.utils.math import rss_complex, ifft2c
import common.utils.transforms as T


class SensitivityModel(nn.Module):
    """Model for learning coil sensitivity maps from kspace data."""

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        init_conv_chans: int = 8,
        num_pool_layers: int = 4,
        drop_prob: Optional[float] = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            in_chans: number of channels in the input to the UNet model.
            out_chans: number of channels in the output to the UNet model.
            init_conv_chans: number of output channels of the first convolution
                layer.
            num_pool_layers: number of down- and up- sampling layers.
            drop_prob: dropout probability.
            mask_center: whether to only use center of kspace for sensitivity
                map calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUNet(
            in_chans,
            out_chans,
            init_conv_chans,
            num_pool_layers,
            drop_prob=drop_prob
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Reshapes a tensor representing complex data of shape BCHW2 into
        (BC)1HW2.
        Input:
            x: input tensor of shape BCHW2.
        Returns:
            Output tensor of shape (BC)1HW2.
        """
        if len(x.size()) == 5:
            b, c, h, w, d = x.size()
        elif len(x.size()) == 4:
            c, h, w, d = x.size()
            b = 1
        else:
            raise RuntimeError(f"Unexpected dimensions {x.size()}")
        return x.view(b * c, 1, h, w, d), b

    def batch_chans_to_chan_dim(
        self, x: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Reshapes a tensor representing complex data of shape (BC)1HW2 into
        BCHW2.
        Input:
            x: input tensor of shape (BC)1HW2.
        Returns:
            Output tensor of shape BCHW2.
        """
        bc, n, h, w, d = x.size()
        if d != 2 or n != 1 or bc % batch_size != 0:
            raise RuntimeError(f"Unexpected dimensions {x.size()}")
        return x.view(batch_size, bc // batch_size, h, w, d)

    def divide_rss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor by the root of the sum of squares.
        Input:
            x: input tensor of shape BCHW2.
        Returns:
            Normalized input tensor.
        """
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def construct_center_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Estimate the center mask from an input mask. This function assumes
        that the input mask is a 1D line mask.
        Input:
            mask: an input mask of shape BW.
        Returns:
            The estimated center masks of shape BW.
        """
        b, w = mask.size()
        left = (w // 2) - torch.argmin(mask[:, :w // 2].flip(1), dim=1) + 1
        right = (w // 2) + torch.argmin(mask[:, w // 2:], dim=1)

        center_masks = []
        for i in range(b):
            m = np.zeros(w, dtype=np.float32)
            for j in range(left[i], right[i]):
                m[j] = 1.0
            center_masks.append(m)

        return torch.from_numpy(np.array(center_masks)).type(torch.int32)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        center_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            if center_mask is not None:
                kspace = T.apply_mask(masked_kspace, center_mask)
            else:
                kspace = T.apply_mask(
                    masked_kspace, self.construct_center_mask(mask)
                )
        else:
            kspace = T.apply_mask(masked_kspace, mask)

        # Convert to image space.
        images, batches = self.chans_to_batch_dim(ifft2c(kspace))

        # Estimate sensitivities.
        return self.divide_rss(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )
