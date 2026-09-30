"""DCGM means of one card over a boot's measured window (window.txt: epoch start and end of the
timed pass, written by run_probe_boot.sh), for clean boots where no trace exists.

A sample's value at ts covers [ts - lag - period, ts - lag] (dcgm_known_answer.py: 100 ms
period, 122 ms lag on the H100 host engine); only windows inside the measured pass count.
SM active over GR active is the share of SMs busy while any kernel runs; SM occupancy over SM
active is the resident warps per busy SM as a share of the maximum.

usage: python3 dcgm_window.py samples.tsv window.txt --gpu 0 [--lag-ms 122] [--period-ms 100]
"""

from __future__ import annotations

import argparse
import collections
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples")
    parser.add_argument("window")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--lag-ms", type=float, default=122.0)
    parser.add_argument("--period-ms", type=float, default=100.0)
    args = parser.parse_args()
    with open(args.window) as handle:
        begin, end = (float(value) * 1e6 for value in handle.read().split()[:2])
    lag, period = args.lag_ms * 1000, args.period_ms * 1000
    values = collections.defaultdict(list)
    with open(args.samples) as handle:
        handle.readline()
        for line in handle:
            ts, card, name, value = line.rstrip("\n").split("\t")
            covered_end = int(ts) - lag
            if (
                int(card) == args.gpu
                and covered_end - period >= begin
                and covered_end <= end
            ):
                values[name].append(float(value))
    if not values:
        raise SystemExit(f"no sample of card {args.gpu} inside the window")
    means = {name: statistics.fmean(samples) for name, samples in values.items()}
    print(
        f"card {args.gpu}, window {(end - begin) / 1e6:.1f} s, "
        f"{len(values['gr_active'])} samples of {args.period_ms:.0f} ms"
    )
    for name in sorted(means):
        ordered = sorted(values[name])
        print(
            f"  {name:<16} mean {100 * means[name]:6.1f} %  p10 "
            f"{100 * ordered[len(ordered) // 10]:6.1f} %  p90 "
            f"{100 * ordered[len(ordered) * 9 // 10]:6.1f} %"
        )
    print(
        f"  SM active / GR active {means['sm_active'] / means['gr_active']:.3f}, "
        f"SM occupancy / SM active {means['sm_occupancy'] / means['sm_active']:.3f}"
    )


if __name__ == "__main__":
    main()
