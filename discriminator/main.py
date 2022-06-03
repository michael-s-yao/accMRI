"""
Main driver program for k-space discriminator.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import pytorch_lightning as pl
import time
from args import build_args
from pl_modules.data_module import DataModule
from pl_modules.discriminator_module import DiscriminatorModule


def main():
    args = build_args()

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
        max_rotation=args.max_rotation,
        max_x=args.max_translation[0],
        max_y=args.max_translation[1],
        p_transform=args.p_transform,
        p_spike=args.p_spike,
        max_spikes=args.max_spikes,
        p_rf_cont=args.p_rf_cont,
        max_rf_cont=args.max_rf_cont,
        min_lines_acquired=args.min_lines_acquired,
        max_lines_acquiring=args.max_lines_acquiring,
        seed=seed
    )
    model = DiscriminatorModule(
        num_coils=num_coils,
        chans=args.chans,
        pools=args.pools,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay
    )
    log_every_n_steps = 1 if args.fast_dev_run else 50
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=log_every_n_steps
    )

    if args.mode.lower() in ("both", "train"):
        trainer.fit(model, datamodule=datamodule)
    if args.mode.lower() in ("both", "test"):
        trainer.test(model, dataloaders=datamodule)


if __name__ == "__main__":
    main()
