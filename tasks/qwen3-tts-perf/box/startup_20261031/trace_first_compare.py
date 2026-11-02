# Block 77 reduction: each arm's codec_kernels.py runs against main's, over every non-tail key. Per arm: build seconds,
# programs, replay summed per group (the keys main already compiles; the follow-up and window keys main runs eager),
# keys slower or faster than main by more than 1 %, the worst key, kernel launches at the key main compiles, and
# waveforms bit-equal to the first arm named.
# usage: python3 trace_first_compare.py <dir> main=<run>,<run> <arm>=<run>,<run> ...
import json
import statistics
import sys

import torch


def load(directory, spec):
    name, runs = spec.split("=")
    return name, runs.split(","), [json.load(open(f"{directory}/{run}.json")) for run in runs.split(",")]


def main():
    directory = sys.argv[1]
    arms = [load(directory, spec) for spec in sys.argv[2:]]
    main_name, main_runs, main_data = arms[0]
    keys = list(main_data[0]["keys"])
    median = lambda data, key: statistics.median(run["keys"][key]["ms"] for run in data)
    compiled_on_main = [key for key in keys if key.startswith("cold:") or key.split(":")[1].startswith("8x")]
    newly = [key for key in keys if key not in compiled_on_main]
    for name, runs, data in arms:
        sums = (sum(median(data, key) for key in compiled_on_main), sum(median(data, key) for key in newly))
        ratios = {key: median(data, key) / median(main_data, key) - 1 for key in keys}
        slower = {key: round(100 * value, 1) for key, value in ratios.items() if value > 0.01}
        faster = sum(value < -0.01 for value in ratios.values())
        worst = max(compiled_on_main, key=lambda key: ratios[key])
        launches = {key: data[0]["keys"][key]["kernels"].__len__() for key in ("cold:2x2", "warm0:8x2", "warm0:8x4", "warm0:8x8", "cold:1x8")}
        print(
            f"{name}: build {[run['build_s'] for run in data]} programs {data[0]['programs']}; main's keys {sums[0]:.2f} ms, new keys {sums[1]:.2f} ms; "
            f"worst of main's keys {worst} {100 * ratios[worst]:+.2f} %; slower than main by 1 % {slower}; faster by 1 % {faster} keys; launches {launches}"
        )
        per_key = " ".join(f"{key.split(':')[0][0]}{key.split(':')[1]}={100 * ratios[key]:+.1f}" for key in compiled_on_main)
        print(f"   {per_key}")
    reference = torch.load(f"{directory}/{arms[1][1][0]}.pt")
    for name, runs, data in arms:
        waves = torch.load(f"{directory}/{runs[0]}.pt")
        print(f"{name} waveforms bit-equal to {arms[1][0]}: {sum(torch.equal(reference[key], waves[key]) for key in keys)} of {len(keys)}")


if __name__ == "__main__":
    main()
