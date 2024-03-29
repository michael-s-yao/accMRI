"""
CLI-friendly argument parser for accelerated MRI reconstructors.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import argparse
import os
import time


class Main:
    @staticmethod
    def build_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="MRI Image Reconstructor")

        parser.add_argument(
            "--data_path",
            type=str,
            default="",
            help="Data path. Defaults to the configured Amulet data directory."
        )
        parser.add_argument(
            "--cache_path",
            type=str,
            default=None,
            help="Optional dataset cache file to use for faster load times."
        )
        parser.add_argument(
            "--model",
            type=str,
            default="varnet",
            choices=("varnet", "unet"),
            help="Reconstructor model. Default VarNet."
        )
        # Per Yin T, Wu Z, et al, we can optionally choose to crop our
        # reconstruction to a smaller size, like (128, 128,) for instance.
        parser.add_argument(
            "--center_crop",
            type=int,
            default=[256, 256],
            nargs=2,
            help="Image crop size. Default (256, 256)."
        )
        parser.add_argument(
            "--fixed_acceleration",
            type=float,
            nargs="+",
            default=None,
            help="Optional fixed acceleration factor for training."
        )
        parser.add_argument(
            "--multicoil",
            action="store_true",
            help="Explicitly specify whether to use multicoil data."
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=int(time.time()),
            help="Optional random seed. Default to seconds since epoch."
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Batch size. Default 1."
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers. Default 4."
        )
        parser.add_argument(
            "--use_distributed_sampler",
            action="store_true",
            help="Specify whether to use distributed sampler."
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Learning rate. Default 0.001."
        )
        parser.add_argument(
            "--lr_step_size",
            type=int,
            default=40,
            help="Learning rate step size. Default 40."
        )
        parser.add_argument(
            "--lr_gamma",
            type=float,
            default=0.1,
            help="Learning rate decay rate. Default 0.1."
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="Weight decay. Default 0.0."
        )
        parser.add_argument(
            "--max_epochs",
            type=int,
            default=50,
            help="Maximum number of training epochs. Default 50."
        )
        parser.add_argument(
            "--fast_dev_run",
            action="store_true",
            help="Runs a quick unit test."
        )
        parser.add_argument(
            "--chans",
            type=int,
            default=18,
            help="Number of channels for regularization network. Default 18."
        )
        parser.add_argument(
            "--pools",
            type=int,
            default=4,
            help="Number of UNet down- and up- sampling layers. Default 4."
        )
        parser.add_argument(
            "--cascades",
            type=int,
            default=8,
            help="Number of VarNet cascades. Default 8."
        )
        parser.add_argument(
            "--sens_chans",
            type=int,
            default=8,
            help="Number of channels for sensitivity map UNet. Default 8."
        )
        parser.add_argument(
            "--sens_pools",
            type=int,
            default=4,
            help="Number of pool layers for sensitivity map UNet. Default 4."
        )
        parser.add_argument(
            "--sens_drop_prob",
            type=float,
            default=0.0,
            help="Dropout probability for sensitivity map UNet. Default 0.0."
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="both",
            choices=("train", "test", "both"),
            help="One of ['train', 'test', 'both']. Default both."
        )
        parser.add_argument(
            "--num_log_images",
            type=int,
            default=16,
            help="Number of images to log. Default 16."
        )
        parser.add_argument(
            "--save_reconstructions",
            action="store_true",
            help="Save image reconstructions from test dataset."
        )
        parser.add_argument(
            "--num_gpus",
            type=int,
            default=0,
            help="Number of GPUs in use. Default CPU only."
        )
        parser.add_argument(
            "--num_nodes",
            type=int,
            default=1,
            help="Number of nodes in use. Default 1."
        )
        parser.add_argument(
            "--ckpt_path",
            type=str,
            default=None,
            help="Optional path to checkpoint to resume training from."
        )
        parser.add_argument(
            "--tl",
            action="store_true",
            help="Transfer learning pre-training using Shepp-Logan phantoms."
        )
        parser.add_argument(
            "--num_coils",
            type=int,
            default=15,
            help="Number of coils to simulate for Shepp-Logan phantoms."
        )
        ewc_help = "Elastic weight consolidation (EWC) loss scaling. "
        ewc_help += "Default no EWC."
        parser.add_argument("--ewc", type=float, default=0.0, help=ewc_help)
        ewc_statedict_help = "Path to checkpoint file that contains the "
        ewc_statedict_help += "state dict from the first learning task. "
        ewc_statedict_help += "Used for learning multiple anatomies."
        parser.add_argument(
            "--ewc_state_dict", type=str, default=None, help=ewc_statedict_help
        )
        fim_path_help = "Optional path to FIM cache path for computing the "
        fim_path_help += "EWC regularization loss term when --ewc > 0. If "
        fim_path_help += "not provided, the diagonal FIM will be constructed "
        fim_path_help += "at run time."
        parser.add_argument(
            "--fim_cache_path", type=str, default=None, help=fim_path_help
        )

        return parser.parse_args()


class Inference:
    @staticmethod
    def build_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="MRI Image Reconstructor")

        parser.add_argument(
            "--data_path",
            type=str,
            default=os.path.join(
                os.environ.get("AMLT_DATA_DIR", "/mnt/fastmri"),
                "multicoil_val"
            ),
            help="A file or folder of undersampled kspace data."
        )
        parser.add_argument(
            "--cache_path",
            type=str,
            default="./dataset_cache.pkl",
            help="Optional dataset cache file to use for faster load times."
        )
        # Per Yin T, Wu Z, et al, we can optionally choose to crop our
        # reconstruction to a smaller size, like (128, 128,) for instance.
        parser.add_argument(
            "--center_crop",
            type=int,
            default=[256, 256],
            nargs=2,
            help="Image crop size. Default (256, 256)."
        )
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Model checkpoint file. Default zero-filled reconstructor."
        )
        parser.add_argument(
            "--simulated",
            action="store_true",
            help="Run inference on simulated phantom slices."
        )
        fixed_acceleration_help = "Acceleration factor(s) to use for "
        fixed_acceleration_help += "inference. Only applies for non-test "
        fixed_acceleration_help += "datasets. Default variable acceleration."
        parser.add_argument(
            "--fixed_acceleration",
            type=float,
            nargs="+",
            default=None,
            help=fixed_acceleration_help
        )
        parser.add_argument(
            "--save_path",
            type=str,
            default=None,
            help="Path to save image reconstructions to. Default None."
        )
        parser.add_argument(
            "--use_deterministic",
            action="store_true",
            help="Specify whether to use deterministic algorithms for `torch`."
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Optional random seed. Default 42."
        )
        parser.add_argument(
            "--enable_progress_bar",
            action="store_true",
            help="Enable TQDM Progress Bar."
        )
        parser.add_argument(
            "--num_gpus",
            type=int,
            default=0,
            help="Number of GPUs to use. Default no GPUs."
        )

        return parser.parse_args()
