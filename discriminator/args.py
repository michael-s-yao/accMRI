"""
CLI-friendly argument parser for kspace discriminator.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright Microsoft Research 2022.
"""
import argparse
import os


class Main:
    @staticmethod
    def build_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="k-space Discriminator")

        parser.add_argument(
            "--data_path",
            type=str,
            default="./",
            help="Folder path to train, val, and test data subdirectories."
        )
        parser.add_argument(
            "--cache_path",
            type=str,
            default=None,
            help="Optional dataset cache file to use for faster load times."
        )
        # While we implemented our models to be compatible with multi-coil
        # input data, we did not explore their performance in our work.
        parser.add_argument(
            "--coil_compression",
            type=int,
            default=-1,
            help="Number of SVD coils to compress multicoil data to."
        )
        parser.add_argument(
            "--multicoil",
            action="store_true",
            help="Explicitly specify whether to use multicoil data."
        )
        # While we implemented other model architectures, such as MLP and
        # U-Net variants, we did not find much success with them.
        model_choices = (
            "NormUNet", "normunet", "UNet", "unet", "MLP", "mlp", "CNN", "cnn"
        )
        model_help = "Model for discriminator, one of "
        model_help += "[`normunet`, `unet`, `mlp`, `cnn`]."
        parser.add_argument(
            "--model",
            type=str,
            choices=model_choices,
            default="cnn",
            help=model_help
        )
        parser.add_argument(
            "--rotation",
            type=float,
            default=[20.0, 50.0],
            nargs=2,
            help="Rotation magnitude range (in degrees). Default 20 to 50 deg."
        )
        x_range_help = "Horizontal translation magnitude range. "
        x_range_help += "Default no translation."
        parser.add_argument(
            "--x_range",
            type=float,
            default=[0.0, 0.0],
            nargs=2,
            help=x_range_help
        )
        y_range_help = "Vertical translation magnitude range. "
        y_range_help += "Default no translation."
        parser.add_argument(
            "--y_range",
            type=float,
            default=[0.0, 0.0],
            nargs=2,
            help=y_range_help
        )
        parser.add_argument(
            "--p_transform",
            type=float,
            default=0.5,
            help="Probability of a kspace dirtying operation. Default 0.5."
        )
        parser.add_argument(
            "--p_spike",
            type=float,
            default=0.5,
            help="Probability of a kspace spike. Default 0.5."
        )
        parser.add_argument(
            "--max_spikes",
            type=int,
            default=10,
            help="Maximum number of kspace spikes. Default 10."
        )
        parser.add_argument(
            "--p_rf_cont",
            type=float,
            default=0.5,
            help="Probability of an RF contamination. Default 0.5."
        )
        parser.add_argument(
            "--max_rf_cont",
            type=int,
            default=10,
            help="Maximum number of RF contaminations. Default 10."
        )
        parser.add_argument(
            "--min_lines_acquired",
            type=int,
            default=16,
            help="Minimum number of lines previously acquired. Default 16."
        )
        parser.add_argument(
            "--max_lines_acquiring",
            type=int,
            default=16,
            help="Maximum number of lines in a single acquisition. Default 16."
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Optional random seed. Default to seconds since epoch."
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
            default=0.001,
            help="Learning rate. Default 0.001."
        )
        parser.add_argument(
            "--lr_step_size",
            type=int,
            default=50,
            help="Learning rate step size. Default 50."
        )
        parser.add_argument(
            "--lr_gamma",
            type=float,
            default=1.0,
            help="Learning rate decay rate. Default no decay."
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
            default=128,
            help="Number of channels in first hidden layer. Default 128."
        )
        parser.add_argument(
            "--pools",
            type=int,
            default=2,
            help="Number of UNet down- and up- sampling layers. Default 2."
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="both",
            choices=("both", "train", "test"),
            help="Operation mode. Default to both train and test the model."
        )
        parser.add_argument(
            "--num_gpus",
            type=int,
            default=0,
            help="Number of GPUs in use."
        )
        parser.add_argument(
            "--ckpt_path",
            type=str,
            default=None,
            help="Optional path to checkpoint to resume training from."
        )
        # A center crop of 128x128 was used in our experiments.
        parser.add_argument(
            "--center_crop",
            type=int,
            nargs=2,
            default=(-1, -1),
            help="kspace center crop dimensions. Default no center crop."
        )

        return parser.parse_args()


class Inference:
    @staticmethod
    def build_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="k-space Discriminator")

        parser.add_argument(
            "--data_path",
            type=str,
            required=True,
            help="A file or folder of undersampled acquired kspace data."
        )
        parser.add_argument(
            "--coil_compression",
            type=int,
            default=4,
            help="Number of SVD coils to compress multicoil data to."
        )
        parser.add_argument(
            "--cache_path",
            type=str,
            default="./multicoil_knee_cache.pkl",
            help="Optional dataset cache file to use for faster load times."
        )
        threshmin_help = "Threshold for heatmap values, above which acquired "
        threshmin_help += "kspace measurements are considered dirty."
        parser.add_argument(
            "--threshmin", type=float, default=0.1, help=threshmin_help
        )
        model_help = "Path to model checkpoint to use for inference. If not "
        model_help += "specified, then no kspace processing is applied. If "
        model_help += "`model` == 'baseline', then baseline signal "
        model_help += "cross-correlation-based method is used."
        parser.add_argument("--model", type=str, default=None, help=model_help)
        reconstructor_help = "Path to reconstructor checkpoint to use for "
        reconstructor_help += "reconstruction. If not specified, then "
        reconstructor_help += "zero-filled image reconstruction is performed."
        parser.add_argument(
            "--reconstructor", type=str, default=None, help=reconstructor_help
        )
        gt_correlation_map_help = "Path to ground truth correlation maps "
        gt_correlation_map_help += "stored as a pickle file. Required if "
        gt_correlation_map_help += "using `model` == 'baseline'."
        parser.add_argument(
            "--gt_correlation_map",
            type=str,
            default=None,
            help=gt_correlation_map_help
        )
        parser.add_argument(
            "--save_path",
            type=str,
            default=os.environ.get("AMLT_OUTPUT_DIR", "./"),
            help="Save path for generated outputs."
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Optional random seed. Defaults to seconds since epoch."
        )
        parser.add_argument(
            "--center_crop",
            type=int,
            nargs=2,
            default=(256, 256),
            help="Final image crop size. Default (256, 256)."
        )
        parser.add_argument(
            "--rotation",
            type=float,
            default=[20.0, 50.0],
            nargs=2,
            help="Rotation magnitude range (in degrees). Default 20 to 50 deg."
        )
        parser.add_argument(
            "--center_frac",
            type=float,
            default=0.04,
            help="Center fraction to define ACS lines. Default 0.04."
        )
        cpu_help = "Specify whether to run everything on the CPU, even if "
        cpu_help += "GPU(s) is available."
        parser.add_argument("--cpu", action="store_true", help=cpu_help)
        parser.add_argument(
            "--use_deterministic",
            action="store_true",
            help="Specify whether to use deterministic algorithms for `torch`."
        )
        parser.add_argument(
            "--accuracy_by_line",
            action="store_true",
            help="Optional flag to save accuracy scores per kspace line."
        )

        return parser.parse_args()
