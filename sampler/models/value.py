"""
Value network implementations. Portions of this code, including
certain choices for default model parameters, were taken from the
from the active-mri-acquisition repository at
https://github.com/facebookresearch/active-mri-acquisition.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
from collections import OrderedDict
import torch
from torch import nn
from typing import Union


class MLPNetwork(nn.Module):
    """
    Generic symmetric MLP (used as value network for the Double DQN Model).
    """

    def __init__(
        self,
        in_chans: int,
        chans: int = 32,
        num_layers: int = 4,
    ):
        """
        Arguments:
            in_chans: number of input channels for the first layer and
                output channels for the last layer.
            chans: number of the channels in the intermediate layers.
            num_layers: total number of layers.
        """
        super().__init__()
        self.layers = [
            ("linear0", nn.Linear(in_chans, chans),),
            ("activation0", nn.Softplus(beta=1, threshold=20),)
        ]
        for i in range(1, num_layers - 1):
            self.layers.append(("linear" + str(i), nn.Linear(chans, chans),))
            self.layers.append((
                "activation" + str(i),
                nn.Softplus(beta=1, threshold=20),
            ))
        self.layers.append(
            ("linear" + str(num_layers - 1), nn.Linear(chans, in_chans),)
        )
        self.layers = nn.Sequential(OrderedDict(self.layers))

    def forward(
        self, kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the multilayer perceptron.
        Input:
            kspace: input kspace data tensor of size BCHW2.
            mask: input tensor of size BW.
        Returns:
            Output value tensor with the same size as the input tensor.
        """
        return torch.reshape(
            self.layers(torch.flatten(mask, start_dim=1)), mask.size()
        )


class ConvNetwork(nn.Module):
    """
    Generic CNN (used as value network for the Double DQN Model).

    Portions of this code was built based off of the evaluator network design
    described in Zhang Z et al. (2019). Reducing uncertainty in undersampled
    MRI reconstruction with active acquisition. IEEE CVPR Proceedings.
    """

    def __init__(
        self,
        kspace_shape: Union[torch.Size, tuple],
        chans: int = 256,
        num_pools: int = 4,
        drop_prob: float = 0.0
    ):
        """
        Args:
            kspace_shape: Shape of kspace as BCHW2.
            chans: number of filters used in convolutions. Default 256.
            num_pools: number of convolutional layers. Default 4.
            drop_prob: dropout probability.
        """
        super().__init__()
        b, c, h, w, d = kspace_shape
        self.kspace_shape = kspace_shape
        self.drop_prob = drop_prob

        sequence = [
            (
                "convolution0",
                nn.Conv2d(
                    (d * w) + 1,
                    chans,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            ("normalization0", nn.InstanceNorm2d(chans),),
            ("activation0", nn.LeakyReLU(negative_slope=0.2, inplace=True),),
            ("dropout0", nn.Dropout2d(self.drop_prob),)
        ]
        in_chans = chans
        for i in range(1, num_pools):
            out_chans = in_chans
            if i < num_pools - 1:
                out_chans = in_chans * 2

            sequence.append(
                (
                    "convolution" + str(i),
                    nn.Conv2d(
                        in_chans,
                        out_chans,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
            )
            sequence.append(
                ("normalization" + str(i), nn.InstanceNorm2d(out_chans),)
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

            in_chans = out_chans

        kernel_size_width = w // 2 ** num_pools
        kernel_size_height = h // 2 ** num_pools

        sequence.append(
            ("pool", nn.AvgPool2d((kernel_size_height, kernel_size_width,)))
        )
        sequence.append(
            (
                "final",
                nn.Conv2d(in_chans, w, kernel_size=1, strid=1, padding=0),
            )
        )

        self.layers = nn.Sequential(OrderedDict(sequence))

    def forward(
        self, kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the convolutional value network to compute
        scores for each kspace column.
        Input:
            kspace: input kspace data tensor of size BCHW2.
            mask: acquisition mask of shape BW.
        Returns:
            Value tensor of shape BW.
        """
        # Output is of shape BWCH.
        output = self.layers(
            self._complex_to_width_dim(self._embed_mask(kspace, mask))
        )

        return torch.sum(output, dim=(-1, -2))

    def _embed_mask(
        self, kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Embeds a kspace mask into the kspace data tensor.
        Input:
            kspace: kspace data tensor of shape BCHW2.
            mask: acquisition mask of shape BW.
        Returns:
            Embedded mask and kspace tensor of shape B(C+1)HW2.
        """
        b, c, h, w, d = kspace.size()
        mask = torch.cat((torch.unsqueeze(mask, dim=-1),) * d, dim=-1)
        mask = torch.cat((torch.unsqueeze(mask, dim=1),) * h, dim=1)
        return torch.cat((kspace, torch.unsqueeze(mask, dim=1),), dim=1)

    def _complex_to_width_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes a tensor representing complex data of shape BCHW2 into
        B(2W)CH.
        Input:
            x: input tensor of shape BCHW2.
        Returns:
            Output tensor of shape B(2W)CH.
        """
        b, c, h, w, d = x.size()
        if d != 2:
            raise RuntimeError(f"Unexpected dimensions {x.size()}")
        return torch.reshape(
            torch.permute(x, (0, 4, 3, 1, 2,)), (b, 2 * w, c, h,)
        )
