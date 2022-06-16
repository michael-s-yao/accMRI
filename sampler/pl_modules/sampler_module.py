"""
PyTorch Lightning module for the accelerated MRI kspace sampling model.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
from collections import defaultdict
import pathlib
import numpy as np
import sys
import time
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Optional, Union
from torchmetrics.metric import Metric
from models.buffer import ReplayBuffer
from models.policy import DDQNPolicy
from models.loss import DDQNLoss

sys.path.append("..")
from helper.utils.evaluate import mse, ssim, save_reconstructions
import helper.utils.transforms as T


class SamplerModule(pl.LightningModule):
    """PyTorch Lightning module for accelerated MRI kspace sampling models."""

    def __init__(
        self,
        kspace_size: Union[torch.Size, tuple],
        max_lines: int = 8,
        in_chans: int = -1,
        chans: int = 32,
        num_layers: int = 4,
        seed: Optional[int] = None,
        reconstruction_size: Union[torch.Size, tuple] = (320, 320),
        eps_thresh: float = 0.0,
        q_thresh: float = 0.0,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        num_log_images: int = 16
    ):
        """
        Args:
            max_lines: maximum number of lines to sample per request.
            in_chans: number of input channels to the value network.
            chans: number of the channels in the intermediate layers.
            num_layers: total number of layers.
            seed: random optional seed.
            reconstruction_size: image reconstruction crop size.
            eps_thresh: the probability of sampling a random action instead of
                using a greedy action.
            q_thresh: threshold for q values below which we choose not to
                sample.
            lr: learning rate.
            lr_step_size: learning rate step size.
            lr_gamma: learning rate gamma decay.
            weight_decay: parameter for penalizing weights norm.
            num_log_images: number of images to log.
        """
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.num_log_images = num_log_images
        self.val_log_indices = None
        self.time = str(int(time.time()))
        self.model_path = "model_" + self.time + ".pt"

        self.memory = ReplayBuffer(obs_shape=kspace_size, seed=seed)
        self.training_policy = DDQNPolicy(
            max_lines=max_lines,
            in_chans=in_chans,
            memory=self.memory,
            chans=chans,
            num_layers=num_layers,
            seed=seed,
            reconstruction_size=reconstruction_size,
            eps_thresh=eps_thresh,
            q_thresh=q_thresh
        )
        self.test_policy = Sampler(policy="active", checkpoint=self.model_path)

        self.loss = DDQNLoss()

    def forward(
        self,
        est_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        action_buffer: torch.Tensor,
        sens_maps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
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
        return self.test_policy(
            est_kspace, acquired_mask, action_buffer, sens_maps
        )

    def training_step(self, batch, batch_idx):
        # Simulate N reconstruction steps.
        for _ in self.num_train_steps:
            while # TODO

        q_values = self.training_policy(
            batch.est_kspace,
            batch.acquired_mask,
            batch.action_buffer,
            batch.sens_maps
        )

        target, output = T.center_crop_to_smallest(batch.target, output)
        output = T.scale_max_value(output, batch.max_value)

        loss = self.loss(
            output.unsqueeze(1),
            target.unsqueeze(1),
            data_range=batch.max_value
        )

        self.log("train_loss", loss)

        return loss

    def on_train_end(self):
        # Save the model parameters for future inference.
        torch.save(self.training_policy.state_dict(), self.model_path)

    def validation_step(self, batch, batch_idx):
        q_values = self.training_policy(
            batch.est_kspace,
            batch.acquired_mask,
            batch.action_buffer,
            batch.sens_maps
        )

        target, output = T.center_crop_to_smallest(batch.target, output)
        output = T.scale_max_value(output, batch.max_value)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fn,
            "slice_num": batch.slice_idx,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1),
                target.unsqueeze(1),
                data_range=batch.max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        q_values = self.training_policy(
            batch.est_kspace,
            batch.acquired_mask,
            batch.action_buffer,
            batch.sens_maps
        )

        return {
            "fname": batch.fn,
            "slice_num": batch.slice_idx,
            "est_q_values": q_values,
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]
