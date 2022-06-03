"""
Main driver program for accelerated MRI simulation environment.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
from multiprocessing import Process, Queue
import time
from compute import ComputeCluster
from scanner import MRIScanner
from args import build_args


def main():
    args = build_args()

    seed = args.seed
    if seed is None or seed < 0:
        seed = int(time.time())

    data_pipeline = Queue()
    kspace_queries = Queue()
    g = MRIScanner(
        args.shape,
        modified_sl=args.modified_sl,
        num_lines=args.num_lines,
        data_acq_mean=args.data_acq_mean,
        data_acq_std=args.data_acq_std,
        no_requests_latency=args.no_requests_latency,
        seed=seed,
        debug=args.debug
    )
    r = ComputeCluster(
        args.shape,
        num_coils=args.num_coils,
        num_lines=args.num_lines,
        no_data_latency=args.no_data_latency,
        min_recon_latency=args.min_recon_latency,
        seed=seed,
        savefig=args.savefig,
        debug=args.debug
    )

    generator = Process(target=g.sample, args=(data_pipeline, kspace_queries,))
    reconstructor = Process(
        target=r.reconstruct, args=(data_pipeline, kspace_queries,)
    )

    generator.start()
    time.sleep(args.init_delay)
    reconstructor.start()

    generator.join()
    reconstructor.join()


if __name__ == "__main__":
    main()
