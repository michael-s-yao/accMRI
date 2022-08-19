"""
Implementations for UNet and NormUNet. Portions of this code, including certain
choices for default model parameters, were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import torch
from math import floor, ceil
from torch import nn
from torch.nn import functional as F
from typing import List, Optional, Tuple

from models.conv import ConvBlock, TransposeConvBlock


class UNet(nn.Module):
    """PyTorch implementation of a generic UNet model."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        init_conv_chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: Optional[float] = 0.0,
    ):
        """
        Args:
            in_chans: number of channels in the input to the UNet model.
            out_chans: number of channels in the output to the UNet model.
            init_conv_chans: number of output channels of the first
                convolution layer.
            num_pool_layers: number of down- and up- sampling layers.
            drop_prob: dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.init_conv_chans = init_conv_chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList(
            [
                ConvBlock(
                    self.in_chans,
                    self.init_conv_chans,
                    drop_prob,
                    num_conv_layers=2
                )
            ]
        )
        ch = self.init_conv_chans
        for _ in range(self.num_pool_layers - 1):
            self.down_sample_layers.append(
                ConvBlock(ch, 2 * ch, drop_prob, num_conv_layers=2)
            )
            ch *= 2
        self.conv = ConvBlock(ch, 2 * ch, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(self.num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(2 * ch, ch))
            self.up_conv.append(
                ConvBlock(2 * ch, ch, drop_prob, num_conv_layers=2)
            )
            ch //= 2
        self.up_transpose_conv.append(TransposeConvBlock(2 * ch, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(2 * ch, ch, drop_prob, num_conv_layers=1),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            features: input tensor of shape BCHW2 (C is in_chans).
        Returns:
            Output tensor of shape BCHW (C is out_chans).
        """
        b, c, h, w, d = features.size()
        features = torch.reshape(features, (b, d * c, h, w))

        stack = []  # Bottom [0, 1, 2, ..., n-1] Top
        output = features

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect padding on the right/bottom if needed to handle odd input
            # dimensions.
            padding = [0, 0, 0, 0]
            if output.size()[-1] != downsample_layer.size()[-1]:
                padding[1] = 1  # Right padding
            if output.size()[-2] != downsample_layer.size()[-2]:
                padding[-1] = 1  # Bottom padding
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            # Note: Changed the order of downsample_layer and output relative
            # to fastMRI implementation.
            output = torch.cat([downsample_layer, output], dim=1)
            output = conv(output)

        return output


class NormUNet(nn.Module):
    """
    PyTorch implementation of a generic UNet model, with the added caveat of
    applying normalization to the input before feeding it into the UNet. This
    should keep the values more numerically stable during training.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        init_conv_chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: Optional[float] = 0.0,
    ):
        """
        Args:
            in_chans: number of channels in the input to the UNet model.
            out_chans: number of channels in the output to the UNet model.
            init_conv_chans: number of output channels of the first
                convolution layer.
            num_pool_layers: number of down- and up- sampling layers.
            drop_prob: dropout probability.
        """
        super().__init__()

        self.num_pool_layers = num_pool_layers
        self.unet = UNet(
            in_chans=in_chans,
            out_chans=out_chans,
            init_conv_chans=init_conv_chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
        )

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
        Reshapes a tensor representing complex data of shape B(2C)HW into
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

    def norm(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalizes all the real and imaginary components of the input tensor.
        Input:
            x: input tensor of shape BCHW2.
        Returns:
            norm_x: output tensor of shape BCHW2.
            mean: mean of both real and imaginary components of each batch.
            std: STD of both real and imaginary components of each batch.
        """
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """
        Unnormalizes all the real and imaginary components of the input tensor.
        Input:
            x: normalized input tensor of shape BCHW2.
            mean: mean of both real and imaginary components of each batch.
            std: STD of both real and imaginary componetns of each batch.
        Returns:
            Unnormalized output tensor of shape BCHW2.
        """
        return (x * std) + mean

    def pad(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int], List[int], int, int]:
        _, _, h, w = x.size()
        """
        Pads input tensor so that it can be fed through a UNet with
        self.num_pool_layers downsampling layers.
        Input:
            x: input tensor of shape B(2C)HW.
        Returns:
            pad_x: output tensor of shape B(2C)H*W*, where H* and W* are the
                smallest multiples of 2**self.num_pool_layers that are greater
                than H and W, respectively.
            h_pad: applied padding in H dimension.
            w_pad: applied padding in W dimension.
            h_ceil: H*.
            w_ceil: W*.
        """
        # Find the smallest multiple of 2**self.num_pool_layers that is
        # greater than or equal to h and w dimensions.
        h_ceil = ((h - 1) | (2 ** self.num_pool_layers)) + 1
        w_ceil = ((w - 1) | (2 ** self.num_pool_layers)) + 1

        # Calculate the padding necessary to get H and W dimensions to
        # multiple of 2**self.num_pool_layers.
        h_pad = [floor((h_ceil - h) / 2), ceil((h_ceil - h) / 2)]
        w_pad = [floor((w_ceil - w) / 2), ceil((w_ceil - w) / 2)]

        # Apply padding.
        x = F.pad(x, w_pad + h_pad)
        return x, h_pad, w_pad, h_ceil, w_ceil

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_ceil: int,
        w_ceil: int,
    ) -> torch.Tensor:
        """
        Unpads input tensor.
        Input:
            x: padding tensor of shape B(2C)H*W*.
            h_pad: previously applied padding in H dimension.
            w_pad: previously applied padding in W dimension.
            h_ceil: H*.
            w_ceil: W*.
        Returns:
            Unpadded output tensor of original shape B(2C)HW prior to padding.
        """
        return x[..., h_pad[0]:h_ceil - h_pad[1], w_pad[0]:w_ceil - w_pad[1]]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            features: input tensor of shape BCHW2 (C is in_chans).
        Returns:
            Output tensor of shape BCHW2 (C is out_chans).
        """
        # Normalize input.
        features, mean, std = self.norm(features)
        features = self.complex_to_chan_dim(features)
        features, h_pad, w_pad, h_ceil, w_ceil = self.pad(features)

        # Feed into UNet.
        features = self.unet(features)

        # Restore the original unpadded shape.
        features = self.unpad(features, h_pad, w_pad, h_ceil, w_ceil)
        features = torch.squeeze(features, dim=1)

        return features
