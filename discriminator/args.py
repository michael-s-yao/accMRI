"""
CLI-friendly argument parser for kspace discriminator.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import argparse


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="k-space discriminator.")
    parser.add_argument(
        "data_path",
        type=str,
        help="A folder containing train, val, and test data subdirectories."
    )
    parser.add_argument(
        "--multicoil",
        action="store_true",
        help="Explicitly specify whether to use multicoil data."
    )
    parser.add_argument(
        "--max_rotation",
        type=float,
        default=15.0,
        help="Maximum absolute angle of rotation (in degrees). Default 15 deg."
    )
    parser.add_argument(
        "--max_translation",
        type=float,
        default=[0.1, 0.1],
        nargs=2,
        help="Max absolute fraction for image translation. Default (0.1, 0.1)."
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
        default=8,
        help="Minimum number of lines previously acquired. Default 8."
    )
    parser.add_argument(
        "--max_lines_acquiring",
        type=int,
        default=12,
        help="Maximum number of lines in a single acquisition. Default 12."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Default to seconds since epoch."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size. Default 1."
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
        help="Number of channels for discriminator UNet. Default 18."
    )
    parser.add_argument(
        "--pools",
        type=int,
        default=4,
        help="Number of UNet down- and up- sampling layers. Default 4"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=("both", "train", "test"),
        help="Operation mode. Default to both train and test the model."
    )

    return parser.parse_args()
