"""
Policy wrapper for the kspace sampling module.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import os
import policy as P
import torch
from typing import Optional, Union


class Sampler:
    "Module for kspace line sampling."

    def __init__(
        self,
        policy: str = "active",
        max_lines: Optional[int] = 8,
        checkpoint: Optional[str] = None
    ):
        """
        Args:
            policy: one of [`random`, `low_to_high`, `greedy`, `active`].
            max_lines: maximum number of lines to sample per request.
            checkpoint: path to the .pt file storing the weights of the trained
                model. Must be specified if policy is `active`.
        """
        if policy.lower() == "random":
            self.policy = P.RandomPolicy(max_lines=max_lines)
        elif policy.lower() == "low_to_high":
            self.policy = P.LowToHighPolicy(max_lines=max_lines)
        elif policy.lower() == "greedy":
            self.policy = P.GreedyOraclePolicy(max_lines=max_lines)
        elif policy.lower() == "active":
            if checkpoint is None:
                raise ValueError(
                    "Trained model weights must be provided for active policy."
                )
            if not os.path.isfile(checkpoint):
                raise ValueError(
                    "Trained model weights file {checkpoint} does not exist."
                )
            self.policy = P.DDQNPolicy(max_lines=max_lines)
            self.policy.load_state_dict(torch.load(checkpoint))
            self.policy.eps_thresh = 0.0
            self.policy.eval()
        else:
            policies = "`random`, `low_to_high`, `greedy`, `active`"
            raise ValueError(
                "Unrecognized policy {policy}, expected one of {policies}."
            )

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
            The next lines to sample of shape B(max_lines).
        """
        return self.policy.get_action(
            est_kspace, acquired_mask, action_buffer, **kwargs
        )
