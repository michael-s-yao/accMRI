"""
Implementation for kspace sampling policies, including random, low-to-high
frequency, greedy oracle, and RL-based active acquisition.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import abc
import numpy as np
import sys
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union
from models.buffer import ReplayBuffer
from models.value import MLP

sys.path.append("..")
import common.utils.math as M
import common.utils.transforms as T
from reconstructor.models.varnet import VarNet as V
from reconstructor.models.reconstructor import Reconstructor


class MetaPolicy:
    """Generic policy parent class."""

    def __init__(self, max_lines: int = 8):
        """
        Args:
            max_lines: maximum number of lines to sample per request.
        """
        self.max_lines = max_lines

    @abc.classmethod
    @abc.abstractmethod
    def get_action(
        self,
        est_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        action_buffer: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Determines the next set of kspace lines to sample.
        Input:
            est_kspace: estimated kspace data of shape BCHW2.
            acquired_mask: current mask of acquired kspace data of shape BW.
            action_buffer: buffer of K most recently queued kspace line
                requests of shape K(max_lines). If a request had less than
                max_lines lines requested, the request is padded with -1's to
                to max_lines columns.
            target: target image of shape BH'W'. Used and required only for the
                GreedyOraclePolicy.
            sens_maps: estimated sensitivity maps. Required if input est_kspace
                is multicoil data. Used and required only for the
                GreedyOraclePolicy and DDQNPolicy if data is multicoil.
        Returns:
            next_lines: the next lines to sample of shape B(max_lines).
        """
        pass


class RandomPolicy(MetaPolicy):
    """Samples a random set of unsampled lines from kspace."""

    def __init__(self, max_lines: int = 8, **kwargs):
        super().__init__(max_lines)

    def get_action(
        self,
        est_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        action_buffer: torch.Tensor,
        **kwargs
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

    def __init__(self, max_lines: int = 8, **kwargs):
        super().__init__(max_lines)

    def get_action(
        self,
        est_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        action_buffer: torch.Tensor,
        **kwargs
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

    def __init__(self,reconstructor: Callable, max_lines: int = 8, **kwargs):
        super().__init__(max_lines)

        self.reconstructor = reconstructor

    def get_action(
        self,
        est_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        action_buffer: torch.Tensor,
        target: torch.Tensor,
        sens_maps: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        next_lines = None
        for b in range(acquired_mask.size()[0]):
            unsampled = torch.squeeze(
                torch.nonzero(acquired_mask[b, :] < 1.0, as_tuple=False)
            )

            dmetric = []
            for idx in torch.arange(acquired_mask.size()[-1]):
                if acquired_mask[b, idx] == 1.0:
                    dmetric.append(0.0)
                    continue
                acquired_mask[b, idx] = 1.0

                masked_kspace = 
                # Potentially need to pass in target kspace as well.

                # Reset the acquired mask.
                acquired_mask[b, idx] = 0.0

            nxt = torch.unsqueeze(nxt, dim=0)
            if next_lines is None:
                next_lines = nxt
            else:
                next_lines = torch.cat((next_lines, nxt), dim=0)

        return next_lines


class DDQNPolicy(nn.Module, MetaPolicy):
    """Double DQN value network implementation."""

    def __init__(
        self,
        max_lines: int = 8,
        in_chans: int = -1,
        memory: Optional[ReplayBuffer] = None,
        chans: int = 32,
        num_layers: int = 4,
        seed: Optional[int] = None,
        reconstruction_size: Union[torch.Size, tuple] = (320, 320),
        eps_thresh: float = 0.0,
        q_thresh: float = 0.0,
    ):
        """
        Args:
            max_lines: maximum number of lines to sample per request.
            in_chans: number of input channels to the value network.
            memory: optional replay memory to sample transitions from.
            chans: number of the channels in the intermediate layers.
            num_layers: total number of layers.
            seed: random optional seed.
            reconstruction_size: image reconstruction crop size.
            eps_thresh: the probability of sampling a random action instead of
                using a greedy action.
            q_thresh: threshold for q values below which we choose not to
                sample.
        """
        if in_chans < 1:
            raise ValueError("in_chans is a required argument.")

        nn.Module.__init__(self)
        MetaPolicy.__init__(self, max_lines)

        self.memory = memory
        self.value_network = MLP(in_chans, chans, num_layers)

        self.rng = np.random.RandomState(seed)
        self.reconstruction_size = reconstruction_size
        self.eps_thresh = eps_thresh
        self.q_thresh = q_thresh

        self.random_policy = RandomPolicy(max_lines)

        self.reconstructor = Reconstructor()  # TODO

        # TODO
        self.lr = -1
        self.weight_decay = -1.0
        self.optimizer = self.configure_optimizers()

    def add_experience(
        self,
        ref_kspace: torch.Tensor
        acquired_mask: torch.Tensor,
        action_buffer: torch.Tensor,
        reward: float,
        done: bool
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
        return self.memory.push(
            ref_kspace,
            acquired_mask,
            action_buffer,
            next_kspace,
            reward,
            done,
            sens_maps
        )

    def forward(
        self,
        est_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        action_buffer: torch.Tensor,
        sens_maps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predicts action values.
        Input:
            est_kspace: estimated kspace data of shape BCHW2.
            acquired_mask: previous acquisition mask of shape BW.
            action_buffer: buffer of actions that have been queried but have
                not been used to return additional kspace data yet.
            sens_maps: estimated sensitivity maps. Required if input est_kspace
                is multicoil data.
        Returns:
            A matrix of predicted action values of shape BW.
        """
        b, c, h, w, d = est_kspace.size()
        if c > 1 and sens_maps is None:
            raise ValueError(
                "Sensitivity maps must be provided if using multicoil data."
            )
        if c > 1:
            cimg = V.sens_reduce(est_kspace, sens_maps)
        else:
            cimg = M.ifft2c(est_kspace)
        xhat = T.center_crop(
            M.rss(M.complex_abs(cimg)),
            shape=self.reconstruction_size
        )
        xhat = torch.flatten(xhat, start_dim=1)
        features = torch.cat((xhat, acquired_mask), dim=1)

        return self.value_network(features)

    def get_action(
        self,
        est_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        action_buffer: torch.Tensor,
        sens_maps: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Returns the next action from an epsilon-greedy policy. With
        probability self.eps_thresh, we sample a random set of kspace columns
        (ignoring active columns). Otherwise, we return the columns with the
        highest estimated Q-values for the observation.
        Input:
            est_kspace: estimated kspace data of shape BCHW2.
            acquired_mask: previous acquisition mask of shape BW.
            action_buffer: buffer of actions that have been queried but have
                not been used to return additional kspace data yet.
            sens_maps: estimated sensitivity maps. Required if input est_kspace
                is multicoil data.
        Returns:
            next_lines: the next lines to sample of shape B(max_lines).
        """
        if self.rng.rand() < self.eps_thresh:
            return self.random_policy.get_action(
                est_kspace, acquired_mask, action_buffer
            )

        with torch.no_grad():
            self.model.eval()
            q_values = self(
                est_kspace, acquired_mask, action_buffer, sens_maps
            )
        next_lines = None
        for q in q_values.unbind():
            # Threshold by self.q_thresh.
            idxs = torch.squeeze(
                torch.nonzero(q > self.q_thresh, as_tuple=False)
            )
            actions = q[idxs]

            # Sample by selecting the lines with the highest q values.
            actions = torch.argsort(q_values, descending=True)
            if actions.size()[-1] > self.max_lines:
                actions = actions[:self.max_lines]
            elif actions.size()[-1] < self.max_lines:
                padding = -1.0 * torch.ones(
                    self.max_lines - actions.size()[-1], dtype=actions.dtype
                )
                actions = torch.cat((actions, padding))
            actions = torch.unsqueeze(actions, dim=0)

            if next_lines is None:
                next_lines = actions
            else:
                next_lines = torch.cat((next_lines, actions), dim=0)

        return next_lines

    def update_parameters(
        self, target_nn: nn.Module
    ) -> Optional[Dict[str, Any]]:
        """
        Updates the current DQN using the input target_nn, which should be the
        other DQN in the DDQN.
        Input:
            target_nn: the other DQN used to reduce overestimation of Q-values.
        Returns:
            loss: loss value.
            grad_norm: norm of the gradient.
            q_values_mean: mean of the q_values.
            q_values_std: standard deviation of the q_values.
        """
        # Set to training mode.
        self.model.train()

        # Get a batch of data from the replay buffer.
        batch = self.memory.sample()
        # Handle burn in.
        if batch is None:
            return None

        not_done_mask = torch.squeeze(torch.logical_not(batch.dones))

        # Compute Q-values and get the next best action according to the
        # online network.
        q_values = self.forward(
            batch.ref_kspace,
            batch.acquired_mask,
            batch.action_buffer,
            batch.sens_maps
        )

        if self.gamma == 0.0:
            target_values = batch.rewards
        else:
            # TODO: kspace must be returned from the reconstructor. It
            # currently returns the reconstructed image.
            est_kspace = self.reconstructor(
                T.apply_mask(batch.ref_kspace, batch.acquired_mask),
                batch.acquired_mask
            )
            with torch.no_grad():
                # Q1 is trained to select the best actions. Q2 is used to
                # evaluate the actions.

                # Calculate Q1[s(i+1), a(i+1)].
                all_q_values_next = self.forward(
                    est_kspace,
                    batch.acquired_mask,
                    batch.action_buffer,
                    batch.sens_maps
                )
                target_values = torch.zeros(batch.acquired_mask.size()[0])
                if torch.any(not_done_mask).item() != 0:
                    # Calculate max_a(i) Q1[s(i), a(i)].
                    best_actions = torch.argsort(
                        all_q_values_next, dim=-1, descending=True
                    )
                    best_actions = best_actions[:, :self.max_lines]

                    acquiring_mask = batch.acquired_mask.clone()
                    acquiring_mask[best_actions] = 1.0
                    new_est_kspace = self.reconstructor(
                        T.apply_mask(batch.ref_kspace, acquiring_mask),
                        acquiring_mask
                    )
                    # TODO: Other actions in the action buffer are not
                    # currently incorporated into calculation.
                    action_buffer = torch.cat((
                        torch.unsqueeze(best_actions, dim=1),
                        batch.action_buffer,
                    ), dim=1)
                    # Ensure that the action buffer is not over capacity.
                    if action_buffer.size()[1] > self.memory.capacity:
                        action_buffer = action_buffer[
                            :, :self.memory.capacity, :
                        ]

                    # Use the optimal actions selected using Q1 and feed into
                    # Q2 as the evaluation reference.
                    target_values[not_done_mask] = (
                        target_nn.forward(
                            est_kspace,
                            acquiring_mask,
                            action_buffer,
                            batch.sens_maps
                        ).gather(
                            1, torch.unsqueeze(best_actions, dim=1)
                        )[not_done_mask]
                    )
                target_values = self.gamma * target_values + batch.rewards

        loss = F.smooth_l1_loss(q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()

        # Compute the total gradient norm and then clip the gradients to 1.
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.parameters())):
            grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)

        self.optimizer.step()

        torch.cuda.empty_cache()

        return {
            "loss": loss,
            "grad_norm": grad_norm,
            "q_values_mean": q_values.detach().mean().cpu().numpy(),
            "q_values_std": q_values.detach().std().cpu().numpy()
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
