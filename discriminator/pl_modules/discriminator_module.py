"""
PyTorch Lightning module for the kspace discriminator model.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
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

        self.unet = Discriminator(
            num_coils=self.num_coils,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = nn.BCELoss()

    def forward(
        self,
        acquired_kspace: torch.Tensor,
        acquiring_kspace: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generates a heat map of kspace fidelity.
        Input:
            acquired_kspace: previously acquired masked kspace data of
                shape BCWH2.
            acquiring_kspace: currenty masked kspace data acquisition of
                unknown fidelity of shape BCHW2.
        Returns:
            Heat map of kspace fidelity of shape BHW. All values are between
            0.0 and 1.0. Values closer to 1.0 have lower fidelity.
        """
        return self.unet(acquired_kspace, acquiring_kspace)

    def training_step(self, batch, batch_idx):
        acquired = T.apply_mask(batch.ref_kspace, batch.sampled_mask)
        acquiring = T.apply_mask(batch.distorted_kspace, batch.acquiring_mask)
        heatmap = self(acquired, acquiring)
        ground_truth = 1.0 - torch.isclose(
            batch.ref_kspace, batch.distorted_kspace
        ).type(torch.float32)
        # Sum over the coil and real/imaginary axes.
        ground_truth = torch.clamp(
            torch.sum(ground_truth, dim=[1, -1]), 0.0, 1.0
        )
        idxs = torch.nonzero(batch.acquiring_mask > 0.0, as_tuple=True)

        loss = self.loss(
            heatmap[idxs[0], :, idxs[1]].T,
            ground_truth[idxs[0], :, idxs[1]].T
        )
        self.log("train_loss", loss)

        if batch_idx % 100 == 0 + 100:
            self.log_image(
                "heatmap", (255.0 * heatmap[0]).type(torch.uint8), cmap="magma"
            )
            self.log_image("gt", (ground_truth[0] * 255.0).type(torch.uint8))
            b, h, w = heatmap.size()
            mask = torch.cat(
                (torch.unsqueeze(batch.acquiring_mask[0], dim=0),) * h, dim=0
            )
            self.log_image("acquiringmask", (255.0 * mask).type(torch.uint8))
            self.log_image(
                "acquiredkspacemasked",
                torch.sum(acquired[0], dim=(0, -1)),
                cmap="gray"
            )
            self.log_image(
                "acquiringkspacemasked",
                torch.sum(acquiring[0], dim=(0, -1)),
                cmap="gray"
            )
            self.log_image(
                "acquiredkspace",
                torch.sum(batch.ref_kspace[0], dim=(0, -1)),
                cmap="gray"
            )
            self.log_image(
                "acquiring",
                torch.sum(batch.distorted_kspace[0], dim=(0, -1)),
                cmap="gray"
            )

        return loss

    def validation_step(self, batch, batch_idx):
        acquired = T.apply_mask(batch.ref_kspace, batch.sampled_mask)
        acquiring = T.apply_mask(batch.distorted_kspace, batch.acquiring_mask)
        heatmap = self(acquired, acquiring)
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
                heatmap[idxs[0], :, idxs[1]].T,
                ground_truth[idxs[0], :, idxs[1]].T
            )
        }

    def test_step(self, batch, batch_idx):
        acquired = T.apply_mask(batch.ref_kspace, batch.sampled_mask)
        acquiring = T.apply_mask(batch.distorted_kspace, batch.acquiring_mask)
        heatmap = self(acquired, acquiring)
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
