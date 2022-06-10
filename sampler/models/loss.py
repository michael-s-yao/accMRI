import torch
from torch import nn


class DDQNLoss(nn.Module):


    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: threshold to change between L1 and L2 loss.
        """
        super().__init__()

        self.loss = nn.SmoothL1Loss(beta=beta)

    def forward(self):
        pass
