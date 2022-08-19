"""
Main driver program for k-space discriminator.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
from args import Main
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import time

sys.path.append("..")
from pl_modules.data_module import DataModule
from pl_modules.discriminator_module import DiscriminatorModule


def main():
    args = Main.build_args()

    seed = args.seed
    if seed is None or seed < 0:
        seed = int(time.time())
    pl.seed_everything(seed=seed)

    coil_prefix = "singlecoil_"
    num_coils = 1
    if args.multicoil:
        coil_prefix = "multicoil_"
        num_coils = 4 if args.coil_compression else 15
    is_mlp = args.model.lower() in ["mlp", "cnn"]
    train_dir = coil_prefix + "train"
    val_dir = coil_prefix + "val"
    if is_mlp:
        test_dir = coil_prefix + "val"
    else:
        test_dir = coil_prefix + "test"

    data_path = args.data_path
    if data_path is None or len(data_path) == 0:
        data_path = os.environ["AMLT_DATA_DIR"]

    datamodule = DataModule(
        data_path,
        cache_path=args.cache_path,
        coil_compression=args.coil_compression,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=1,
        num_workers=args.num_workers,
        rotation=args.rotation,
        dx=args.x_range,
        dy=args.y_range,
        p_transform=args.p_transform,
        p_spike=args.p_spike,
        max_spikes=args.max_spikes,
        p_rf_cont=args.p_rf_cont,
        max_rf_cont=args.max_rf_cont,
        min_lines_acquired=args.min_lines_acquired,
        max_lines_acquiring=args.max_lines_acquiring,
        seed=seed,
        fast_dev_run=args.fast_dev_run,
        num_gpus=args.num_gpus,
        is_mlp=is_mlp,
        center_crop=args.center_crop
    )
    model = DiscriminatorModule(
        num_coils=num_coils,
        chans=args.chans,
        pools=args.pools,
        model=args.model,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        center_crop=args.center_crop
    )
    log_every_n_steps = 1 if args.fast_dev_run else 50
    start = str(int(time.time()))
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        dirpath=os.environ.get("AMLT_OUTPUT_DIR", None),
        filename=("discriminator-" + start + "-{epoch}-{validation_loss}"),
        monitor="validation_loss",
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto",
        auto_select_gpus=True,
    )

    if args.mode.lower() in ("both", "train"):
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    if args.mode.lower() in ("both", "test"):
        trainer.test(model, dataloaders=datamodule)


if __name__ == "__main__":
    main()
