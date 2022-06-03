"""
Implementation for kspace sampling policies, including random, low-to-high
frequency, greedy oracle, and RL-based active acquisition.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import abc
import numpy as np
import torch
from torch import nn
from typing import Optional, Sequence
from models.memory import ReplayMemory
from models.value import MLP


class MetaPolicy(nn.Module):
    """Generic policy parent class."""

    def __init__(self, reconstructor: nn.Module, max_lines: int = 8):
        """
        Args:
            reconstructor: image reconstruction module.
            max_lines: maximum number of lines to sample per request.
        """
        super().__init__()

        self.reconstructor = reconstructor
        self.max_lines = max_lines

    @abc.classmethod
    @abc.abstractmethod
    def forward(
        self,
        masked_kspace: torch.Tensor,
        action_buffer: torch.Tensor,
        acquired_mask: torch.Tensor,
        threshmin: float = 0.5,
    ) -> torch.Tensor:
        """
        Determines the next set of unsampled kspace lines to sample.
        Input:
            masked_kspace: current acquired kspace data of shape BCHW2.
            action_buffer: buffer of K most recently queued kspace line
                requests of shape K(max_lines). If a request had less than
                max_lines lines requested, the request is padded with -1's to
                to max_lines columns.
            acquired_mask: current mask of acquired kspace data of shape BW.
            threshmin: cutoff for heatmap values to sample.
        Returns:
            next_lines: the next lines to sample of shape max_lines.
        """
        pass


class RandomPolicy(MetaPolicy):
    """Samples a random set of unsampled lines from kspace."""

    def __init__(self, reconstructor: nn.Module, max_lines: int = 8):
        """
        Args:
            reconstructor: image reconstruction module.
            max_lines: maximum number of lines to sample per request.
        """
        super().__init__(reconstructor, max_lines)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        action_buffer: torch.Tensor,
        acquired_mask: torch.Tensor,
        threshmin: float = 0.5,
    ) -> torch.Tensor:
        next_lines = None
        for b in range(acquired_mask.size()[0]):
            sampled = torch.squeeze(
                torch.nonzero(acquired_mask[b, :] > 0.0, as_tuple=False)
            )
            heatmap = torch.rand(acquired_mask.size()[-1])
            heatmap[sampled] = 0.0

            num_lines = min(
                acquired_mask.size()[-1] - sampled.size()[0], self.max_lines
            )
            nxt = torch.argsort(heatmap, descending=True)[:num_lines]

            # Add padding.
            if num_lines < self.max_lines:
                padding = -1.0 * torch.ones(
                    self.max_lines - num_lines, dtype=next_lines.dtype
                )
                nxt = torch.cat((nxt, padding,))

            nxt = torch.unsqueeze(nxt, dim=0)
            if next_lines is None:
                next_lines = nxt
            else:
                next_lines = torch.cat((next_lines, nxt), dim=0)

        return next_lines


class LowToHighPolicy(MetaPolicy):
    """Samples kspace lines from lowest to highest frequency."""

    def __init__(self, reconstructor: nn.Module, max_lines: int = 8):
        """
        Args:
            reconstructor: image reconstruction module.
            max_lines: maximum number of lines to sample per request.
        """
        super().__init__(reconstructor, max_lines)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        action_buffer: torch.Tensor,
        acquired_mask: torch.Tensor,
        threshmin: float = 0.5,
    ) -> torch.Tensor:
        next_lines = None
        for b in range(acquired_mask.size()[0]):
            unsampled = torch.squeeze(
                torch.nonzero(acquired_mask[b, :] < 1.0, as_tuple=False)
            )
            next_lines = []
            m = acquired_mask.size()[-1] // 2
            next_lines = unsampled[torch.argsort(torch.abs(unsampled - m))]

            num_lines = min(unsampled.size()[0], self.max_lines)
            nxt = next_lines[:num_lines]

            # Add padding.
            if num_lines < self.max_lines:
                padding = -1.0 * torch.ones(
                    self.max_lines - num_lines, dtype=next_lines.dtype
                )
                nxt = torch.cat((nxt, padding,))

            nxt = torch.unsqueeze(nxt, dim=0)
            if next_lines is None:
                next_lines = nxt
            else:
                next_lines = torch.cat((next_lines, nxt), dim=0)

        return next_lines


class GreedyOraclePolicy(MetaPolicy):
    """Returns the kspace columns leading to the best reconstruction score."""

    def __init__(self, reconstructor: nn.Module, max_lines: int = 8):
        """
        Args:
            reconstructor: image reconstruction module.
            max_lines: maximum number of lines to sample per request.
        """
        super().__init__(reconstructor, max_lines)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        action_buffer: torch.Tensor,
        acquired_mask: torch.Tensor,
        threshmin: float = 0.5,
        ref_kspace: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        next_lines = None
        for b in range(acquired_mask.size()[0]):
            unsampled = torch.squeeze(
                torch.nonzero(acquired_mask[b, :] < 1.0, as_tuple=False)
            )

            for new_idx in unsampled.unbind():
                acquired_mask[new_idx] = 1.0

                # TODO
                nxt = None
                # Reset acquired_mask.
                acquired_mask[new_idx] = 0.0

            nxt = torch.unsqueeze(nxt, dim=0)
            if next_lines is None:
                next_lines = nxt
            else:
                next_lines = torch.cat((next_lines, nxt), dim=0)

        return next_lines


class DDQNPolicy(MetaPolicy):
    """Active MRI acquisition using a Double DQN RL network."""

    def __init__(
        self,
        reconstructor: nn.Module,
        in_chans: int,
        max_lines: int = 8,
        memory: Optional[ReplayMemory] = None,
        chans: int = 32,
        num_layers: int = 4,
        seed: Optional[int] = None
    ):
        """
        Args:
            reconstructor: image reconstruction module.
            in_chans: number of input channels to the value network.
            max_lines: maximum number of lines to sample per request.
            memory: optional replay memory to sample transitions from.
                May be None if this is a target network.
            chans: number of the channels in the intermediate layers.
            num_layers: total number of layers.
        """
        super().__init__(reconstructor, max_lines)

        self.memory = memory
        self.value_network = MLP(in_chans, chans, num_layers)

        self.rng = np.random.RandomState(seed)

    def add_experience(
        self,
        masked_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        action: torch.Tensor,
        next_kspace: torch.Tensor,
        reward: float,
        done: bool
    ) -> None:
        """
        Adds a transition experience to the replay buffer.
        Input:
            masked_kspace: zero-filled acquired kspace data of shape CHW2.
            acquired_mask: mask of currently acquired data of shape W.
            action: the sequence of kspace lines that was sampled according
                to some sampling policy based on the input observation of
                shape BW.
            next_kspace: zero-filled acquired kspace data after acquisition
                step of shape CHW2.
            reward: the reward associated with choosing to make the particular
                next_observation observation next.
            done: whether or not the acquisition is done.
        Returns:
            None. The transition is recorded into the replay buffer.
        """
        return self.memory.push(
            masked_kspace,
            acquired_mask,
            action,
            next_kspace,
            reward,
            done
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        action_buffer: torch.Tensor,
        acquired_mask: torch.Tensor,
        threshmin: float = 0.5
    ) -> torch.Tensor:

        # Get best reconstruction so far.
        xhat = self.reconstructor(masked_kspace, acquired_mask)

        # Predict action values from de-aliased reconstruction
        # and acquisition mask.
        action_values = self.value_network(xhat, acquired_mask)

        return self.value_network(

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
