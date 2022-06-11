"""
Main driver program for accelerated MRI reconstruction.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
from args import Main 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import torch
from pl_modules.data_module import DataModule
from pl_modules.reconstructor_module import ReconstructorModule


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
        num_coils = 15
    train_dir = coil_prefix + "train"
    val_dir = coil_prefix + "val"
    test_dir = coil_prefix + "test"

    datamodule = DataModule(
        args.data_path,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        center_crop=args.center_crop,
        seed=seed,
        fast_dev_run=args.fast_dev_run
    )
    h, w = args.center_crop
    kspace_size = (args.batch_size, num_coils, h, w, 2)

    model = ReconstructorModule(
        kspace_size,
        model=args.model,
        chans=args.chans,
        pools=args.pools,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        sens_drop_prob=args.sens_drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        num_log_images=args.num_log_images
    )
    log_every_n_steps = 1 if args.fast_dev_run else 50
    start = str(int(time.time()))
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=5,
        filename=("reconstructor-" + start + "-{epoch}-{validation_loss}"),
        monitor="validation_loss"
    )
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps=log_every_n_steps,
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            devices=1
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps=log_every_n_steps,
            callbacks=[checkpoint_callback],
        )
    if args.mode.lower() in ("both", "train"):
        trainer.fit(model, datamodule=datamodule)
    if args.mode.lower() in ("both", "test"):
        trainer.test(model, dataloaders=datamodule)


if __name__ == "__main__":
    main()
