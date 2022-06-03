"""
Toy model for two-agent simulation.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import argparse
from multiprocessing import Process, Queue, Value
from queue import Empty
import time
import random


def maker(q: Queue, status: Value, thresh: int = 40) -> None:
    """
    Process that inserts a random integer into a queue.
    Input:
        q: the queue to insert a random integer into.
        status: the current global state (ie sum of all taken integers).
        thresh: threshold for status above which the process will terminate.
    Returns:
        None.
    """
    while status.value < thresh:
        n = random.randint(0, 10)
        print(f"maker generating {n}", flush=True)
        q.put(n)
        time.sleep(random.randint(0, 5))


def taker(q: Queue, status: Value, thresh: int = 40) -> None:
    """
    Process that retrieves a value from a queue and updates the global state.
    Input:
        q: the queue to insert a random integer into.
        status: the current global state (ie sum of all taken integers).
        thresh: threshold for status above which the process will terminate.
    Returns:
        None.
    """
    while status.value < thresh:
        try:
            n = q.get(block=False)
            print(
                f"taker taking {n}, total is now {status.value + n}",
                flush=True
            )
            status.value += n
        except Empty:
            print("Queue is currently waiting, taker waiting...", flush=True)
        time.sleep(random.randint(0, 5))


def build_args() -> argparse.Namespace:
    """
    CLI-friendly argument parser for toy model.
    Input:
        None.
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Toy two-agent model.")
    parser.add_argument(
        "--thresh",
        type=int,
        default=30,
        help="Threshold above which the processes will terminate."
    )

    return parser.parse_args()


def main():
    q = Queue()
    status = Value("d", 0.0)
    thresh = 30

    m = Process(target=maker, args=(q, status, thresh,))
    t = Process(target=taker, args=(q, status, thresh,))

    m.start()
    t.start()

    m.join()
    t.join()


if __name__ == "__main__":
    main()
