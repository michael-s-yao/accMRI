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


class BufferSample(NamedTuple):
    ref_kspace: torch.Tensor
    acquired_mask: torch.Tensor
    action_buffer: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    sens_maps: torch.Tensor


class ReplayBuffer:
    """Replay buffer for active MRI acquisition using a Double DQN."""

    def __init__(
        self,
        obs_shape: Union[tuple, torch.Size],
        action_buffer_size: int,
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
            action_buffer_size: the number of actions that can be stored.
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
        self.is_multicoil = c > 1
        if self.is_multicoil:
            self.sens_maps_data = -1.0 * torch.ones(
                (capacity, c, h, w, d), dtype=torch.float32
            )
        self.mask_data = -1.0 * torch.ones(
            (capacity, w), dtype=torch.float32
        )
        self.actions = -1.0 * torch.ones(
            (capacity, action_buffer_size, w), dtype=torch.float32
        )
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

        self.count = 0

    def push(
        self,
        ref_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        action_buffer: torch.Tensor,
        reward: float,
        done: bool,
        sens_maps: Optional[torch.Tensor] = None
    ) -> None:
        """
        Records a transition into the replay buffer.
        Input:
            ref_kspace: fully-sampled kspace data of shape CHW2.
            acquired_mask: mask of currently acquired data of shape W.
            action_buffer: the mask of kspace lines that will be sampled next
                according to some sampling policy based on the input
                observation of shape KW (K is the capacity of the buffer).
            reward: the reward associated with choosing to make the particular
                next_observation observation next.
            done: whether or not the acquisition is done.
            sens_maps: sensitivity maps (required if data is multicoil).
        Returns:
            None. The transition is recorded into the replay buffer.
        """
        position = self.count % self.capacity
        if self.is_multicoil:
            if sens_maps is None:
                raise ValueError(
                    "Sensitivity maps are a required for multicoil data."
                )
            self.sens_maps_data[position] = sens_maps.clone()
        self.kspace_data[position] = ref_kspace.clone()
        self.mask_data[position] = acquired_mask.clone()
        self.actions[position] = action_buffer.clone()
        self.rewards[position] = torch.tensor(
            [reward], dtype=torch.float32
        )
        self.dones[position] = torch.tensor([done], dtype=torch.bool)

        self.count += 1

    def sample(self) -> Optional[BufferSample]:
        """
        Samples a batch of transitions from the replay buffer.
        Input:
            None.
        Returns:
            A BufferSample object, or none if the number of entries is less
                than `self.burn_in`.
        """
        if self.count < self.burn_in:
            return None
        indices = self.rng.choice(
            min(self.count - 1, self.capacity), self.batch_size
        )
        return BufferSample(
            self.kspace_data[indices],
            self.mask_data[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
            self.sens_maps_data[indices],
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
            "rewards": self.rewards,
            "dones": self.dones,
            "sens_maps_data": self.sens_maps_data,
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
        self.rewards = data["rewards"]
        self.dones = data["dones"]
        self.sens_maps_data = data["sens_maps_data"]

        return
