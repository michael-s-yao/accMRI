"""
CLI-friendly argument parser for accelerated MRI reconstructors.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import argparse


class Main:
    @staticmethod
    def build_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "data_path",
            type=str,
            help="A folder containing train, val, and test data subdirectories."
        )
        parser.add_argument(
            "--model",
            type=str,
            default="varnet",
            choices=("varnet", "unet"),
            help="Reconstructor model. Default VarNet."
        )
        # Per Yin T, Wu Z, et al, we can optionally choose to crop kspace data to
        # a smaller size, like (128, 128,), to speed up the pipeline. Otherwise,
        # keep it at (640, 368,).
        parser.add_argument(
            "--center_crop",
            type=int,
            default=[640, 368],
            nargs=2,
            help="kspace crop size. Default (640, 368)."
        )
        parser.add_argument(
            "--multicoil",
            action="store_true",
            help="Explicitly specify whether to use multicoil data."
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
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
            "--lr",
            type=float,
            default=0.0003,
            help="Learning rate. Default 0.0003."
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
            help="Weight decay. Default 0.1."
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
            default=12,
            help="Number of VarNet cascades. Default 12."
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
            help="Number of pooling layers for sensitivity map UNet. Default 4."
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
            help="Operation mode. One of ['train', 'test', 'both']. Default both."
        )
        parser.add_argument(
            "--num_log_images",
            type=int,
            default=16,
            help="Number of images to log. Default 16."
        )

        return parser.parse_args()


class Inference:
    @staticmethod
    def build_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "data_path",
            type=str,
            help="A file or folder of undersampled kspace data."
        )
        parser.add_argument(
            "--lightning_logs_path",
            type=str,
            default="./lightning_logs",
            help="A path to the reconstructor lightning logs."
        )
        parser.add_argument(
            "--model_option",
            type=int,
            default="-1",
            help="Reconstructor option choice (avoids interactive input)."
        )

        return parser.parse_args()