"""
Implementation of the Double DQN value network. Portions of this code,
including certain choices for default model parameters, were taken
from the active-mri-acquisition repository at
https://github.com/facebookresearch/active-mri-acquisition.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import numpy as np
import torch
from torch import nn
from typing import Optional, Sequence
from models.memory import ReplayMemory
from models.mlp import MLP


class DDQN(nn.Module):
    """Double DQN value network implementation."""

    def __init__(
        self,
        in_chans: int,
        memory: Optional[ReplayMemory] = None,
        chans: int = 32,
        num_layers: int = 4,
        seed: Optional[int] = None,
    ):
        """
        Args:
            in_chans: number of input channels to the value network.
            memory: optional replay memory to sample transitions from.
            chans: number of the channels in the intermediate layers.
            num_layers: total number of layers.
        """
        super().__init__()

        self.memory = memory
        self.value_network = MLP(in_chans, chans, num_layers)

        self.rng = np.random.RandomState(seed)

    def add_experience(
        self,
        prev_observations: torch.Tensor,
        action: torch.Tensor,
        next_observations: torch.Tensor,
        reward: float,
        done: bool
    ) -> None:
        """
        TODO
        """
        self.memory.push(
            prev_observations, action, next_observations, reward, done
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        action_buffer: torch.Tensor,
        acquired_mask: torch.Tensor,
        threshmin: float = 0.5
    ) -> torch.Tensor:
        # Predict action values.
        action_values = self.value_network(acquired_mask)
        # TODO

    def get_action(
        self,
        # TODO
        acquired_mask: torch.Tensor,
        eps_thresh: float = 0.0,
        q_thresh: float = 0.0,
        max_lines: int = 8
    ) -> Sequence[int]:
        """
        Returns an action sampled from an epsilon-greedy policy. With
        probability eps_thresh, we sample a random set of kspace columns
        (ignoring active columns). Otherwise, we return the columns with the
        highest estimated Q-values for the observation.
        Input:
            acquired_mask: previous acquisition mask of shape BW.
            eps_thresh: the probability of sampling a random action instead of
                using a greedy action.
            q_thresh: threshold for q values below which we choose not to
                sample.
            max_lines: maximum number of lines to sample per request.
        Returns:
            A tensor of kspace column indices to sample next of shape
                B(max_lines).
        """
        if self.rng.rand() < eps_thresh:
            next_lines = None
            for mask in acquired_mask.unbind():
                unsampled = torch.squeeze(
                    torch.nonzero(mask < 1.0, as_tuple=False)
                )
                unsampled = unsampled[torch.randperm(unsampled.size()[0])]
                if unsampled.size()[-1] > max_lines:
                    unsampled = unsampled[:max_lines]
                elif unsampled.size()[-1] < max_lines:
                    padding = -1.0 * torch.ones(
                        max_lines - unsampled.size()[-1], dtype=unsampled.dtype
                    )
                    unsampled = torch.cat((unsampled, padding))
                unsampled = torch.unsqueeze(unsampled, dim=0)

                if next_lines is None:
                    next_lines = unsampled
                else:
                    next_lines = torch.cat((next_lines, unsampled), dim=0)

            return next_lines
