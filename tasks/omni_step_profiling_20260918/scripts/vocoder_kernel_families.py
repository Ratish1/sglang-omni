"""Kernel families of the vocoder in a serve trace's bench window.

Reads an nsys sqlite export of a serve run with the pipeline_nvtx probe (owners voc.*), and
for the kernels the vocoder owns, eager and graph nodes alike, prints the count and the
device time of each family: cuDNN's layout flips, its fallback kernels, the other
convolution kernels, the SnakeBeta kernels, Inductor's kernels and copy kernels. Kernels
of every other owner are summed in one line.

usage: python vocoder_kernel_families.py REPORT.sqlite --bench-log bench.log
"""

from __future__ import annotations

import argparse
import collections

from pipeline_census import Report

FAMILIES = (
    ("layout flip", ("nchwToNhwc", "nhwcToNchw")),
    ("cudnn fallback", ("implicit_convolve_sgemm", "conv2d_grouped_direct")),
    ("snake beta", ("snake_beta",)),
    ("inductor", ("triton_poi", "triton_per", "triton_red")),
    ("conv", ("xmma", "cutlass", "conv", "fprop", "dgrad", "wgrad", "sm90_", "sm80_")),
    ("copy", ("copy", "Copy", "CatArray", "cat_", "transpose", "permute")),
)


def family_of(name: str) -> str:
    for family, patterns in FAMILIES:
        if any(pattern in name for pattern in patterns):
            return family
        else:
            pass
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--bench-log", required=True)
    args = parser.parse_args()
    report = Report(args.report, args.bench_log)
    window_ms = (report.t1 - report.t0) / 1e6
    counts = collections.Counter()
    times = collections.Counter()
    other_owner_ms = 0.0
    for start, end, owner, name, *_ in report.device:
        if name.startswith("graph "):
            continue
        else:
            pass
        start, end = max(start, report.t0), min(end, report.t1)
        if end <= start:
            continue
        else:
            pass
        if owner.startswith("voc"):
            family = family_of(name)
            counts[family] += 1
            times[family] += (end - start) / 1e6
        else:
            other_owner_ms += (end - start) / 1e6
    print(f"bench window {window_ms:.0f} ms, node mode {report.node_mode}")
    print(f"{'vocoder family':<16}{'kernels':>10}{'ms':>12}")
    for family in [family for family, _ in FAMILIES] + ["other"]:
        print(f"{family:<16}{counts[family]:>10}{times[family]:>12.1f}")
    print(
        f"{'vocoder total':<16}{sum(counts.values()):>10}{sum(times.values()):>12.1f}"
    )
    print(f"{'other owners':<16}{'':>10}{other_owner_ms:>12.1f}")


if __name__ == "__main__":
    main()
