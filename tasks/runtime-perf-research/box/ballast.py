"""Hold one card's memory until the driver reports a given free amount, then wait until killed.

Usage: python3 ballast.py --card N --free-mib M --ready-file PATH
"""

import argparse
import signal
import time

import torch

MIB = 1 << 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--free-mib", type=int, required=True)
    parser.add_argument("--ready-file", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    torch.empty(1, device="cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    held = torch.empty(
        free_bytes - args.free_mib * MIB, dtype=torch.uint8, device="cuda"
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    with open(args.ready_file, "w") as file:
        file.write(
            f"held {held.numel() // MIB} MiB, free {free_bytes // MIB} MiB of {total_bytes // MIB} MiB\n"
        )
    signal.signal(signal.SIGTERM, lambda *_: exit(0))
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
