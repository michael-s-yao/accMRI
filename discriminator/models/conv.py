"""
Implementations for convolutional blocks and transpose convolutional
blocks. Portions of this code, including certain choices for default
model parameters, were taken from the fastMRI repository at
https://github.com/facebookresearch/fastMRI.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import torch
from collections import OrderedDict
from torch import nn
from typing import Optional


class ConvBlock(nn.Module):
    """
    A convolutional block with a customizeable number of convolution layers,
    each followed by instance normalization, LeakyReLU activation, and dropout.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        drop_prob: float = 0.0,
        num_conv_layers: int = 2,
        pool_kernel: Optional[int] = -1,
    ):
        """
        Args:
            in_chans: number of input channels.
            out_chans: number of output channels.
            drop_prob: dropout probability.
            num_conv_layers: number of convolution layers.
            pool_kernel: optional max pooling kernel size. Default no pooling.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.num_conv_layers = num_conv_layers
        self.pool_kernel = pool_kernel

        sequence = []
        for i in range(num_conv_layers):
            if i == 0:
                in_conv = self.in_chans
            else:
                in_conv = self.out_chans
            sequence.append(
                (
                    "convolution" + str(i),
                    nn.Conv2d(
                        in_conv,
                        self.out_chans,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    )
                )
            )
            sequence.append(
                ("normalization" + str(i), nn.InstanceNorm2d(self.out_chans),)
            )
            if self.pool_kernel > -1:
                sequence.append(
                    (
                        "pooling" + str(i),
                        nn.AvgPool2d(
                            kernel_size=(self.pool_kernel, 1),
                            padding=(1, 0),
                            stride=2
                        )
                    )
                )
            sequence.append(
                (
                    "activation" + str(i),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )
            sequence.append(
                ("dropout" + str(i), nn.Dropout2d(self.drop_prob))
            )
        self.layers = nn.Sequential(OrderedDict(sequence))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            features: input tensor of shape NCHW (C is in_chans).
        Returns:
            Output tensor of shape NCHW (C is out_chans).
        """
        return self.layers(features)


class TransposeConvBlock(nn.Module):
    """
    A transpose convolutional block with a customizeable number of tranpose
    convolution layers, each followed by instance normalization and LeakyReLU
    activation.
    """

    def __init__(
        self, in_chans: int, out_chans: int, num_conv_layers: Optional[int] = 1
    ):
        """
        Args:
            in_chans: number of input channels.
            out_chans: number of output channels.
            num_conv_layers: number of transpose convolution layers.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_conv_layers = num_conv_layers

        sequence = []
        for i in range(num_conv_layers):
            sequence.append(
                (
                    "transpose_convolution" + str(i),
                    nn.ConvTranspose2d(
                        self.in_chans,
                        self.out_chans,
                        kernel_size=2,
                        stride=2,
                        bias=False,
                    ),
                )
            )
            sequence.append(
                (
                    "normalization" + str(i),
                    nn.InstanceNorm2d(self.out_chans)
                )
            )
            sequence.append(
                (
                    "activation" + str(i),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )
        self.layers = nn.Sequential(OrderedDict(sequence))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            features: input tensor of shape NCHW (C is in_chans).
        Returns:
            Output tensor of shape NCHW (C is out_chans).
        """
        return self.layers(features)
