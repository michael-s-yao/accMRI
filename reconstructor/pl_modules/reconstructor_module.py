"""
PyTorch Lightning module for the accelerated MRI reconstruction model.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
from collections import defaultdict
import pathlib
import numpy as np
import sys
import torch
import pytorch_lightning as pl
from typing import Optional, Union
from torchmetrics.metric import Metric
from models.reconstructor import Reconstructor
from models.loss import SSIMLoss

sys.path.append("..")
from common.utils.evaluate import mse, ssim, save_reconstructions
import common.utils.transforms as T


class ReconstructorModule(pl.LightningModule):
    """PyTorch Lightning module for accelerated MRI reconstruction models."""

    def __init__(
        self,
        kspace_size: Union[torch.Size, tuple] = (1, 15, 640, 372, 2,),
        model: str = "varnet",
        chans: int = 18,
        pools: int = 4,
        cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        sens_drop_prob: float = 0.0,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        num_log_images: int = 16
    ):
        """
        Args:
            kspace_size: dimensions of input kspace data.
            model: reconstructor model. One of ["varnet", "unet"].
            chans: number of channels for regularization NormUNet.
            pools: number of down- and up- sampling layers for
                regularization NormUNet.
            cascades: number of cascades (only for VarNet reconstructor).
            sens_chans: number of channels for sensitivity map UNet.
            sens_pools: number of down- and up- sampling layers for
                sensitivity map UNet.
            sens_drop_prob: dropout probability for sensitivity map UNet.
            lr: learning rate.
            lr_step_size: learning rate step size.
            lr_gamma: learning rate gamma decay.
            weight_decay: parameter for penalizing weights norm.
            num_log_images: number of images to log.
        """
        super().__init__()
        self.save_hyperparameters()

        self.kspace_size = kspace_size
        self.chans = chans
        self.pools = pools
        self.cascades = cascades
        self.sens_chans = sens_chans
        self.sens_pools = sens_pools
        self.sens_drop_prob = sens_drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

        self.reconstructor = Reconstructor(
            model=model,
            kspace_size=self.kspace_size,
            num_cascades=self.cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            sens_drop_prob=self.sens_drop_prob,
        )

        self.loss = SSIMLoss()

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        center_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Input:
            masked_kspace: currently acquired kspace data.
            mask: acquisition mask of acquired kspace data.
            center_mask: mask of acquired center low-frequency kspace data.
        Returns:
            Estimated reconstruction of masked_kspace data.
        """
        return self.reconstructor(masked_kspace, mask, center_mask)

    def training_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.mask, batch.center_mask)

        target, output = T.center_crop_to_smallest(batch.target, output)
        output = T.scale_max_value(output, batch.max_value)

        loss = self.loss(
            output.unsqueeze(1),
            target.unsqueeze(1),
            data_range=batch.max_value
        )

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.mask, batch.center_mask)

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

    def validation_step_end(self, val_logs):
        if len(val_logs["output"].size()) == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif len(val_logs["output"].size()) != 3:
            raise RuntimeError("Unexpected output size from validation step.")

        if len(val_logs["target"].size()) == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif len(val_logs["target"].size()) != 3:
            raise RuntimeError("Unexpected target size from validation step.")

        if self.val_log_indices is None:
            self.val_log_indices = list(np.random.permutation(
                len(self.trainer.val_dataloaders[0]))[:self.num_log_images]
            )

        # Log a few images to TensorBoard.
        batch_idxs = val_logs["batch_idx"]
        if isinstance(batch_idxs, int):
            batch_idxs = [batch_idxs]
        for i, batch_idx in enumerate(batch_idxs):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"

                # Log target image.
                target = val_logs["target"][i].unsqueeze(0)
                target = target / torch.max(target)
                self.log_image(f"{key}/target", target)

                # Log output reconstruction image.
                output = val_logs["output"][i].unsqueeze(0)
                output = output / torch.max(output)
                self.log_image(f"{key}/reconstruction", output)

                # Log error between reconstruction and ground truth.
                error = torch.abs(target - output)
                error = error / torch.max(error)
                self.log_image(f"{key}/error", error)

        # Compute evaluation metrics.
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = {}
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = val_logs["slice_num"][i]
            max_val = val_logs["max_value"][i]
            output = val_logs["output"][i]
            target = val_logs["target"][i]
            # Compute and save MSE.
            mse_vals[fname][slice_num] = torch.tensor(
                mse(target, output)
            ).view(1)
            # Compute and save target norm.
            target_norms[fname][slice_num] = torch.tensor(
                mse(target, torch.zeros_like(target))
            ).view(1)
            # Compute and save SSIM.
            ssim_vals[fname][slice_num] = torch.Tensor(
                ssim(
                    target.unsqueeze(0),
                    output.unsqueeze(0),
                    max_val=max_val
                )
            ).view(1)
            # Save maximum value.
            max_vals[fname] = max_val

        return {
            "val_loss": val_logs["val_loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals
        }

    def log_image(self, name: str, image: torch.Tensor) -> None:
        if torch.max(image) <= 1.0:
            image = image * 255.0
        self.logger.experiment.add_image(
            name, image, global_step=self.global_step
        )

    def test_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.mask)

        # Check for FLAIR 203 (always assuming square image reconstructions).
        if self.kspace_size[0] > 1:
            min_crop_size = batch.crop_size[0]
            for crop_size in batch.crop_size:
                if crop_size[-1] < min_crop_size[-1]:
                    min_crop_size = crop_size
            if output.size()[-1] < min_crop_size[-1]:
                crop_size = (output.size()[-1], output.size()[-1])
            else:
                crop_size = min_crop_size
        else:
            if output.size()[-1] < batch.crop_size[-1]:
                crop_size = (output.size()[-1], output.size()[-1])
            else:
                crop_size = batch.crop_size

        output = T.center_crop(output, crop_size)
        return {
            "fname": batch.fn,
            "slice_num": batch.slice_idx,
            "output": output.cpu().detach().numpy(),
        }

    def predict_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.mask)

        # Check for FLAIR 203 (always assuming square image reconstructions).
        if self.kspace_size[0] > 1:
            min_crop_size = batch.crop_size[0]
            for crop_size in batch.crop_size:
                if crop_size[-1] < min_crop_size[-1]:
                    min_crop_size = crop_size
            if output.size()[-1] < min_crop_size[-1]:
                crop_size = (output.size()[-1], output.size()[-1])
            else:
                crop_size = min_crop_size
        else:
            if output.size()[-1] < batch.crop_size[-1]:
                crop_size = (output.size()[-1], output.size()[-1])
            else:
                crop_size = batch.crop_size

        output = T.center_crop(output, crop_size)
        if batch.target is not None and batch.target.ndim > 1:
            if batch.target.ndim < 3:
                target = torch.unsqueeze(batch.target, dim=0)
            else:
                target = batch.target
            if output.ndim < 3:
                output = torch.unsqueeze(output, dim=0)
            ssim = 1 - self.loss(
                output.unsqueeze(0), target.unsqueeze(0), batch.max_value
            )
        else:
            ssim = -1
        return {
            "fname": batch.fn,
            "slice_num": batch.slice_idx,
            "output": output.cpu().detach().numpy() * 255.0,
            "ssim": ssim
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    def validation_epoch_end(self, val_logs):
        # Aggregate metrics.
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # Apply means across image volumes.
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples += 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] += mse_val / target_norm
            metrics["psnr"] += (20 * torch.log10(
                max_vals[fname].type(mse_val.dtype).to(mse_val.device)
            )) - (10 * torch.log10(mse_val))
            metrics["ssim"] += torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log(
            "validation_loss", val_loss / tot_slice_examples, prog_bar=True
        )
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)

    def test_epoch_end(self, test_logs):
        outputs = defaultdict(dict)

        for log in test_logs:
            if isinstance(log["slice_num"], int):
                log["slice_num"] = [log["slice_num"]]
            if isinstance(log["fname"], str):
                log["fname"] = [log["fname"]]
            for i, (fname, slice_num,) in enumerate(
                zip(log["fname"], log["slice_num"])
            ):
                outputs[fname][int(slice_num)] = log["output"][i]

        # Stack all of the slices for each file.
        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )

        # Pull the default_root_dir if we have a trainer, otherwise save to
        # cwd.
        if hasattr(self, "trainer"):
            save_path = (
                pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
            )
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"

        save_reconstructions(outputs, save_path, save_png=True)


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
