"""
Main driver program for accelerated MRI reconstruction.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
from args import Main
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import sys
import time
import torch

sys.path.append("..")
from pl_modules.data_module import DataModule
from pl_modules.reconstructor_module import ReconstructorModule


def main():
    args = Main.build_args()

    seed = args.seed
    pl.seed_everything(seed=seed)

    coil_prefix = "singlecoil_"
    if args.multicoil:
        coil_prefix = "multicoil_"
    train_dir = coil_prefix + "train"
    val_dir = coil_prefix + "val"
    test_dir = coil_prefix + "test"

    data_path = args.data_path
    if data_path is None or len(data_path) == 0:
        data_path = os.environ.get("AMLT_DATA_DIR", "./")

    if args.num_nodes > 1:
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    datamodule = DataModule(
        data_path,
        cache_path=args.cache_path,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        center_crop=args.center_crop,
        fixed_acceleration=args.fixed_acceleration,
        seed=seed,
        fast_dev_run=args.fast_dev_run,
        num_gpus=args.num_gpus,
        tl=args.tl,
        num_coils=args.num_coils,
        distributed_sampler=args.use_distributed_sampler
    )
    ewc_dataloader = None
    if args.ewc > 0.0:
        ewc_dataloader = datamodule.train_dataloader()
    model = ReconstructorModule(
        model=args.model,
        is_multicoil=args.multicoil,
        chans=args.chans,
        pools=args.pools,
        cascades=args.cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        sens_drop_prob=args.sens_drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        num_log_images=args.num_log_images,
        save_reconstructions=args.save_reconstructions,
        ewc=args.ewc,
        ewc_dataloader=ewc_dataloader,
        ewc_state_dict=args.ewc_state_dict,
        FIM_cache_path=args.fim_cache_path
    )
    log_every_n_steps = 1 if args.fast_dev_run else 50
    start = str(int(time.time()))
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        dirpath=os.environ.get("AMLT_OUTPUT_DIR", None),
        filename=("reconstructor-" + start + "-{epoch}-{validation_loss}"),
        monitor="validation_loss",
        save_last=True
    )
    # Will use GPUs automatically if available.
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback],
        default_root_dir=os.environ.get("AMLT_OUTPUT_DIR", None),
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=False),
        replace_sampler_ddp=(not args.use_distributed_sampler),
        auto_select_gpus=True,
    )
    if args.mode.lower() in ("both", "train"):
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    if args.mode.lower() in ("both", "test"):
        trainer.test(model, dataloaders=datamodule)


if __name__ == "__main__":
    main()
