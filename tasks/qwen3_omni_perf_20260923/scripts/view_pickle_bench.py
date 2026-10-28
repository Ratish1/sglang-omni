"""Sending a CPU view through a pickled header as it is against copying it out first.

For a storage of each size and views filling a fraction of it, times pickle.dumps then
pickle.loads of the view as it is (pickle writes the whole storage) and of view.clone() (the
clone's time included), and prints the medians, the header bytes and which is faster.

usage: python view_pickle_bench.py --storage-bytes 1048576 67108864 --repeats 20
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import time

import torch

FILL_FRACTIONS = (0.01, 0.5, 0.75, 0.9, 0.99, 1.0)


def round_trip_ms(make_payload, repeats: int) -> tuple[float, int]:
    times = []
    header_bytes = 0
    for _ in range(repeats):
        start = time.perf_counter()
        payload = make_payload()
        header = pickle.dumps(payload)
        pickle.loads(header)
        times.append((time.perf_counter() - start) * 1e3)
        header_bytes = len(header)
    return statistics.median(times), header_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-bytes", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    print(
        f"torch threads {torch.get_num_threads()}, pickle protocol {pickle.DEFAULT_PROTOCOL}"
    )
    for storage_bytes in args.storage_bytes:
        storage = torch.randint(0, 255, (storage_bytes,), dtype=torch.uint8)
        for fraction in FILL_FRACTIONS:
            view = storage[: max(int(storage_bytes * fraction), 1)]
            as_is_ms, as_is_bytes = round_trip_ms(lambda: view, args.repeats)
            copied_ms, copied_bytes = round_trip_ms(lambda: view.clone(), args.repeats)
            print(
                f"storage {storage_bytes:>10d} fill {fraction:4.2f}: as is {as_is_ms:9.3f} ms"
                f" ({as_is_bytes} B), copied {copied_ms:9.3f} ms ({copied_bytes} B),"
                f" copied/as is {copied_ms / as_is_ms:5.2f}"
            )


if __name__ == "__main__":
    main()
