"""
Defines the replay buffer for active MRI acquisition using a Double DQN.
Portions of this code, including certain choices for default parameters,
were taken from the active-mri-acquisition repository at
https://github.com/facebookresearch/active-mri-acquisition.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import numpy as np
import os
import torch
from typing import NamedTuple, Optional, Sequence, Union


class MemorySample(NamedTuple):
    masked_kspace: torch.Tensor
    acquired_mask: torch.Tensor
    action: torch.Tensor
    next_kspace: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class ReplayMemory:
    """Replay buffer for active MRI acquisition using a Double DQN."""

    def __init__(
        self,
        obs_shape: Union[tuple, torch.Size] = None,
        capacity: int = 1000,
        batch_size: int = 2,
        burn_in: int = 100,
        max_lines: int = 8,
        seed: Optional[int] = None,
        filepath: Optional[str] = None
    ):
        """
        Args:
            obs_shape: the shape of the tensors representing kspace (BCHW2).
            capacity: how many transitions can be stored. After capcity is
                reached, early transitions are overwritten in FIFO fashion.
            batch_size: batch_size returned by the replay buffer.
            burn_in: threshold for replay buffer size below which the memory
                will return None. Used for burn-in period before training.
            max_lines: the maximum number of lines that can be acquired per
                acquisition step.
            seed: optional random seed.
            filepath: optional .pt file to read from. If provided, the other
                inputs are ignored and parameters are automatically set from
                the .pt file.
        """
        if filepath is not None:
            self.load(filepath)
            return

        self.capacity = capacity
        self.batch_size = batch_size
        self.burn_in = burn_in
        self.max_lines = max_lines
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        _, c, h, w, d = obs_shape
        self.kspace_data = -1.0 * torch.ones(
            (capacity, c, h, w, d), dtype=torch.float32
        )
        self.mask_data = -1.0 * torch.ones(
            (capacity, w), dtype=torch.float32
        )
        self.actions = -1.0 * torch.ones(
            (capacity, max_lines), dtype=torch.float32
        )
        self.next_kspace_data = -1.0 * torch.ones(
            (capacity, c, h, w, d), dtype=torch.float32
        )
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

        self.count = 0

    def push(
        self,
        masked_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        action: torch.Tensor,
        next_kspace: torch.Tensor,
        reward: float,
        done: bool
    ) -> None:
        """
        Records a transition into the replay buffer.
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
        position = self.count % self.capacity
        self.kspace_data[position] = masked_kspace.clone()
        self.mask_data[position] = acquired_mask.clone()

        action = torch.squeeze(action)
        if action.size()[0] > self.max_lines:
            action = action[:self.max_lines]
        elif action.size()[0] < self.max_lines:
            action = torch.cat((
                action,
                -1.0 * torch.ones(
                    self.max_lines - action.size()[0], dtype=torch.float32
                )
            ))
        self.actions[position] = action

        self.next_kspace_data[position] = next_kspace.clone()

        self.rewards[position] = torch.tensor(
            [reward], dtype=torch.float32
        )
        self.dones[position] = torch.tensor([done], dtype=torch.bool)

        self.count += 1

    def sample(self) -> Optional[MemorySample]:
        """
        Samples a batch of transitions from the replay buffer.
        Input:
            None.
        Returns:
            A MemorySample object, or none if the number of entries is less
                than `self.burn_in`.
        """
        if self.count < self.burn_in:
            return None
        indices = self.rng.choice(
            min(self.count - 1, self.capacity), self.batch_size
        )
        return MemorySample(
            self.kspace_data[indices],
            self.mask_data[indices],
            self.actions[indices],
            self.next_kspace_data[indices],
            self.rewards[indices],
            self.dones[indices]
        )

    def save(self, filepath: str = "buffer.pt") -> Sequence[str]:
        """
        Saves the replay buffer to the specified filepath.
        Input:
            filepath: relative filepath to save to.
        Returns:
            A list of the output data keys.
        """
        if os.path.splitext(filepath)[-1] != ".pt":
            raise ValueError(
                f"Filepath {filepath} must be of file type .pt."
            )

        params = torch.Tensor([
            self.count,
            self.capacity,
            self.batch_size,
            self.burn_in,
            self.max_lines,
        ], dtype=torch.int32)
        if self.seed is not None:
            params = torch.cat(
                (params, torch.tensor([self.seed], dtype=torch.int32))
            )
        else:
            params = torch.cat((params, torch.tensor([float("nan")])))

        data = {
            "kspace_data": self.kspace_data,
            "mask_data": self.next_kspace_data,
            "actions": self.actions,
            "next_kspace_data": self.next_kspace_data,
            "rewards": self.rewards,
            "dones": self.dones,
            "params": params
        }
        torch.save(data, filepath)

        return list(data.keys())

    def load(self, filepath: str = "buffer.pt") -> None:
        """
        Loads the replay buffer from the specified filepath.
        Input:
            filepath: relative filepath to read from (must be a .pt file).
        Returns:
            None.
        """
        if os.path.splitext(filepath)[-1] != ".pt":
            raise ValueError(
                f"Filepath {filepath} must be of file type .pt."
            )

        data = torch.load(filepath)

        params = data["params"]
        self.count = int(params[0])
        self.capacity = int(params[1])
        self.batch_size = int(params[2])
        self.burn_in = int(params[3])
        self.max_lines = int(params[4])
        if torch.isnan(params[5]):
            self.seed = None
        else:
            self.seed = params[5]
        self.rng = np.random.RandomState(self.seed)

        self.kspace_data = data["kspace_data"]
        self.mask_data = data["mask_data"]
        self.actions = data["actions"]
        self.next_kspace_data = data["next_kspace_data"]
        self.rewards = data["rewards"]
        self.dones = data["dones"]

        return
