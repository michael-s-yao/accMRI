"""
Multilayer perceptron (MLP) implementation.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
from collections import OrderedDict
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self, inchans: int, outchans: int, chans: int = 128, numlayers: int = 8
    ):
        """
        Arguments:
            inchans: number of input channels for the first layer.
            outchans: number of output channels from the last layer.
            chans: number of channels in the intermediate layers.
            numlayers: total number of layers.
        """
        super().__init__()
        self.layers = [
            ("linear0", nn.Linear(inchans, chans),),
            ("activation0", nn.LeakyReLU(),)
        ]
        for i in range(1, numlayers - 1):
            self.layers.append((
                "linear" + str(i), nn.Linear(chans, chans),
            ))
            self.layers.append((
                "activation" + str(i),
                nn.LeakyReLU(),
            ))
        self.layers.append(
            ("linear" + str(numlayers - 1), nn.Linear(chans, outchans),)
        )
        self.layers = nn.Sequential(OrderedDict(self.layers))

    def forward(self, x: torch.Tensor, batch_dim: bool = True) -> torch.Tensor:
        """
        Forward pass through the multilayer perceptron.
        Input:
            x: input tensor with a flattened size of inchans.
            batch_dim: specify whether a batch dimension is present.
        Returns:
            Output tensor with a size of outchans.
        """
        return self.layers(torch.flatten(x, start_dim=int(batch_dim)))
