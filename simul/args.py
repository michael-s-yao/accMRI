"""
CLI-friendly argument parser for accelerated MRI simulation environment.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import argparse


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accelerated MRI sandbox.")
    parser.add_argument(
        "shape",
        type=int,
        default=[128, 128],
        nargs=2,
        help="kspace data height and weight. Default (128, 128)."
    )
    parser.add_argument(
        "--num_coils",
        type=int,
        default=1,
        choices=(1, 15),
        help="Number of coils. Default 1"
    )
    parser.add_argument(
        "--modified_sl",
        action="store_true",
        help="Whether to use modified Shepp-Logan phantoms."
    )
    parser.add_argument(
        "--init_delay",
        type=float,
        default=1.0,
        help="TODO"
    )
    parser.add_argument(
        "--num_lines",
        type=int,
        default=None,
        help="Total number of kspace lines to generate. Default 1e12."
    )
    parser.add_argument(
        "--data_acq_mean",
        type=float,
        default=1.0,
        help="Average number of seconds per single kspace line acquisition."
    )
    parser.add_argument(
        "--data_acq_std",
        type=float,
        default=0.1,
        help="Std dev of a single kspace line acquisition. Default 0.1."
    )
    parser.add_argument(
        "--no_requests_latency",
        type=float,
        default=1,
        help="Number of seconds to wait if no sampling requests are queued."
    )
    parser.add_argument(
        "--no_data_latency",
        type=float,
        default=1.0,
        help="Number of seconds to wait if no kspace data is available."
    )
    parser.add_argument(
        "--min_recon_latency",
        type=float,
        default=-1.0,
        help="Optional minimum number of seconds for the reconstruction step."
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        help="Whether to save the output reconstructions."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Default to seconds since epoch."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to print debugging statements or not."
    )

    return parser.parse_args()
