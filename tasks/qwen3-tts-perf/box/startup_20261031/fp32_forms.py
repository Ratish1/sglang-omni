# Block 68 reduction: each arm's waveform per codec key against the float32 whole decode of the same codes
# (codec_forms.py --fp32-reference), per runner: median and worst relative L2, and on how many keys the arm is farther
# from float32 than the reference arm (the first), plus the replay sum per runner.
# usage: python3 fp32_forms.py <block dir> <arm> [<arm> ...]
import json
import statistics
import sys

import torch


def main():
    directory, arms = sys.argv[1], sys.argv[2:]
    reference = torch.load(f"{directory}/{arms[0]}_ref.pt")
    errors = {}
    for arm in arms:
        waves = torch.load(f"{directory}/{arm}.pt")
        errors[arm] = {
            label: float((waves[label] - truth).norm() / truth.norm()) for label, truth in reference.items()
        }
    runners = sorted({label.split(":")[0] for label in reference})
    for runner in runners:
        labels = [label for label in reference if label.startswith(runner + ":")]
        print(f"== {runner} ({len(labels)} keys): relative L2 to float32, median / worst; keys farther than {arms[0]}")
        for arm in arms:
            values = [errors[arm][label] for label in labels]
            farther = sum(errors[arm][label] > errors[arms[0]][label] * 1.001 for label in labels)
            print(f"   {arm:<18} {statistics.median(values):.4f} / {max(values):.4f}   farther {farther}/{len(labels)}")
    for arm in arms:
        replays = json.load(open(f"{directory}/{arm}.json"))["replays"]
        sums = {}
        for item in replays:
            sums[item["runner"]] = sums.get(item["runner"], 0.0) + item["ms"]
        print(f"replay {arm:<18} " + " ".join(f"{runner} {total:.2f}" for runner, total in sums.items())
              + f" | total {sum(sums.values()):.2f} ms")


if __name__ == "__main__":
    main()
