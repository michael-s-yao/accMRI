"""
Main driver program for accelerated MRI reconstruction.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
from args import Main
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import time
from pl_modules.data_module import DataModule
from pl_modules.reconstructor_module import ReconstructorModule


def main():
    args = Main.build_args()

    seed = args.seed
    if seed is None or seed < 0:
        seed = int(time.time())
    pl.seed_everything(seed=seed)

    coil_prefix = "singlecoil_"
    if args.multicoil:
        coil_prefix = "multicoil_"
    train_dir = coil_prefix + "train"
    val_dir = coil_prefix + "val"
    test_dir = coil_prefix + "test"

    data_path = args.data_path
    if data_path is None or len(data_path) == 0:
        data_path = os.environ["AMLT_DATA_DIR"]

    datamodule = DataModule(
        data_path,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        center_crop=args.center_crop,
        fixed_acceleration=args.fixed_acceleration,
        seed=seed,
        fast_dev_run=args.fast_dev_run
    )
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
        save_reconstructions=args.save_reconstructions
    )
    log_every_n_steps = 1 if args.fast_dev_run else 50
    start = str(int(time.time()))
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=5,
        filename=("reconstructor-" + start + "-{epoch}-{validation_loss}"),
        monitor="validation_loss"
    )
    # Will use GPUs automatically if available.
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto"
    )
    if args.mode.lower() in ("both", "train"):
        trainer.fit(model, datamodule=datamodule)
    if args.mode.lower() in ("both", "test"):
        trainer.test(model, dataloaders=datamodule)


if __name__ == "__main__":
    main()
