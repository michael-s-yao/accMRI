"""
PyTorch Lightning module for the kspace discriminator model.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import sys
import torch
import torch.nn as nn
from typing import Optional, Tuple

from torchmetrics.metric import Metric
from models.discriminator import Discriminator

sys.path.append("..")
import helper.utils.transforms as T


class DiscriminatorModule(pl.LightningModule):
    """PyTorch Lightning module for kspace discriminator models."""

    def __init__(
        self,
        num_coils: int = 1,
        chans: int = 18,
        pools: int = 4,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        model: str = "cnn",
        center_crop: Optional[Tuple[int]] = (-1, -1)
    ):
        """
        Args:
            num_coils: number of coils (ie coil dimension).
            chans: number of channels for discriminator NormUNet.
            pools: number of down- and up- sampling layers for
                regularization NormUNet.
            lr: learning rate.
            lr_step_size: learning rate step size.
            lr_gamma: learning rate gamma decay.
            weight_decay: parameter for penalizing weights norm.
            model: model for discriminator, one of [`normunet`, `unet`, `mlp`,
                `cnn`].
            center_crop: kspace center crop dimensions. Default no center crop.
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_coils = num_coils
        self.chans = chans
        self.pools = pools
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        # Default fastMRI dataset dimension.
        self.h = 640
        if center_crop[0] > 0:
            self.h = center_crop[0]

        # The factor of 2 comes from both acquired and acquiring kspace
        # line samples.
        in_chans = 2 * self.h * self.num_coils
        self.model = Discriminator(
            num_coils=self.num_coils,
            chans=self.chans,
            pools=self.pools,
            model=model,
            in_chans=in_chans,
            out_chans=1,
        )

        self.loss = nn.BCEWithLogitsLoss()
        self.ValLoss = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

    def forward(
        self,
        acquired_kspace: torch.Tensor,
        acquiring_kspace: torch.Tensor,
        acquired_mask: torch.Tensor,
        acquiring_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates a heat map of kspace fidelity.
        Input:
            acquired_kspace: previously acquired masked kspace data of
                shape BCWH2.
            acquiring_kspace: currenty masked kspace data acquisition of
                unknown fidelity of shape BCHW2.
            acquired_mask: acquired data mask of shape BW.
            acquiring_mask: current step acquisition data mask of shape BW.
        Returns:
            Heat map of kspace fidelity of shape BHW. All values are between
            0.0 and 1.0. Values closer to 1.0 have lower fidelity.
        """
        return self.model(
            acquired_kspace, acquiring_kspace, acquired_mask, acquiring_mask
        )

    def training_step(self, batch, batch_idx):
        acquired = T.apply_mask(batch.ref_kspace, batch.sampled_mask)
        acquiring = T.apply_mask(batch.distorted_kspace, batch.acquiring_mask)
        heatmap = self(
            acquired, acquiring, batch.sampled_mask, batch.acquiring_mask
        )
        ground_truth = 1.0 - torch.isclose(
            batch.ref_kspace, batch.distorted_kspace
        ).type(torch.float32)
        # Sum over the coil and real/imaginary axes.
        ground_truth = torch.clamp(
            torch.sum(ground_truth, dim=[1, -1]), 0.0, 1.0
        )
        idxs = torch.nonzero(batch.acquiring_mask > 0.0, as_tuple=True)

        loss = self.loss(
            torch.mean(heatmap[idxs[0], :, idxs[1]].T, dim=0),
            torch.mean(ground_truth[idxs[0], :, idxs[1]].T, dim=0)
        )
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        acquired = T.apply_mask(batch.ref_kspace, batch.sampled_mask)
        acquiring = T.apply_mask(batch.distorted_kspace, batch.acquiring_mask)
        heatmap = self(
            acquired, acquiring, batch.sampled_mask, batch.acquiring_mask
        )
        ground_truth = 1.0 - torch.isclose(
            batch.ref_kspace, batch.distorted_kspace
        ).type(torch.float32)
        # Sum over the coil and real/imaginary axes.
        ground_truth = torch.clamp(
            torch.sum(ground_truth, dim=[1, -1]), 0.0, 1.0
        )
        idxs = torch.nonzero(batch.acquiring_mask > 0.0, as_tuple=True)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fn,
            "slice_num": batch.slice_idx,
            "heatmap": T.apply_mask(heatmap, batch.acquiring_mask),
            "val_loss": self.loss(
                torch.mean(heatmap[idxs[0], :, idxs[1]].T, dim=0),
                torch.mean(ground_truth[idxs[0], :, idxs[1]].T, dim=0)
            )
        }

    def validation_epoch_end(self, val_logs):
        # Aggregate metrics.
        losses = []
        count = 0
        for val_log in val_logs:
            count += 1
            losses.append(val_log["val_loss"].view(-1))

        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )
        self.log(
            "validation_loss", val_loss / tot_slice_examples, prog_bar=True
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        acquired = T.apply_mask(batch.ref_kspace, batch.sampled_mask)
        acquiring = T.apply_mask(batch.distorted_kspace, batch.acquiring_mask)
        heatmap = self(
            acquired, acquiring, batch.sampled_mask, batch.acquiring_mask
        )
        ground_truth = 1.0 - torch.isclose(
            batch.ref_kspace, batch.distorted_kspace
        ).type(torch.float32)
        # Sum over the coil and real/imaginary axes.
        ground_truth = torch.clamp(
            torch.sum(ground_truth, dim=[1, -1]), 0.0, 1.0
        )
        idxs = torch.nonzero(batch.acquiring_mask > 0.0, as_tuple=True)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fn,
            "slice_num": batch.slice_idx,
            "heatmap": T.apply_mask(heatmap, batch.acquiring_mask),
            "loss": self.loss(
                heatmap[idxs[0], :, idxs[1]].T,
                ground_truth[idxs[0], :, idxs[1]].T
            )
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    def log_image(
        self, name: str, image: torch.Tensor, cmap: str = "gray"
    ) -> None:
        plt.gca()
        plt.cla()
        plt.imshow(image, cmap=cmap)
        plt.title(f"{name}, step {self.global_step}")
        plt.axis("off")
        if not os.path.isdir("log"):
            os.mkdir("log")
        plt.savefig(
            os.path.join("log", f"{name}_{self.global_step}.png"),
            dpi=300,
            bbox_inches="tight",
            transparent=True
        )


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "quantity", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, batch: torch.Tensor):
        self.quantity += batch

    def compute(self):
        return self.quantity
