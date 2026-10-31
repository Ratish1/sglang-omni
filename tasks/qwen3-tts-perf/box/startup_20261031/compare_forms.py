# Block 43 reduction: per arm, the vocoder build and codec precompile seconds cold and warm, the programs compiled,
# each compiled key's replay time against main's, and its waveform against main's (bit-equal or the max difference).
# usage: python3 compare_forms.py <block43 dir> <arm> [<arm> ...]   (the first arm is the reference)
import json
import sys

import torch


def load(directory, arm, phase):
    try:
        with open(f"{directory}/{arm}_{phase}.json") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def main():
    directory, arms = sys.argv[1], sys.argv[2:]
    reference = arms[0]
    reference_warm = load(directory, reference, "warm")
    reference_ms = {(item["runner"], item["frames"], item["batch"]): item["ms"] for item in reference_warm["replays"]}
    reference_waves = torch.load(f"{directory}/{reference}_warm.pt")
    print("arm | cold build s | cold precompile s | warm build s | warm precompile s | programs | replay sum ms (ref) | worst key ratio | bit-equal keys | max abs diff")
    for arm in arms:
        cold, warm = load(directory, arm, "cold"), load(directory, arm, "warm")
        if cold is None or warm is None:
            print(f"{arm} | missing")
            continue
        else:
            pass
        ms = {(item["runner"], item["frames"], item["batch"]): item["ms"] for item in warm["replays"]}
        shared = sorted(set(ms) & set(reference_ms))
        ratios = {key: ms[key] / reference_ms[key] for key in shared}
        worst = max(ratios, key=ratios.get)
        waves = torch.load(f"{directory}/{arm}_warm.pt")
        equal = sum(torch.equal(waves[label], reference_waves[label]) for label in reference_waves if label in waves)
        difference = max(
            (float((waves[label] - reference_waves[label]).abs().max()) for label in reference_waves if label in waves),
            default=float("nan"),
        )
        print(
            f"{arm} | {cold['build_s']} | {cold['precompile_total_s']} | {warm['build_s']} | {warm['precompile_total_s']} | "
            f"{cold['unique_graphs']} | {sum(ms[key] for key in shared):.3f} ({sum(reference_ms[key] for key in shared):.3f}) | "
            f"{worst} {ratios[worst]:.3f} | {equal}/{len(reference_waves)} | {difference:.3g}"
        )
    for arm in arms:
        warm = load(directory, arm, "warm")
        if warm is None:
            continue
        else:
            pass
        print(f"\n== {arm} per key ms (warm run): " + " ".join(f"{item['runner']}:{item['frames']}x{item['batch']}={item['ms']}" for item in warm["replays"]))
        cold = load(directory, arm, "cold")
        print(f"== {arm} cold precompiles: " + " ".join(f"{item['shape']}={item['seconds']}s+{item['new_graphs']}" for item in cold["precompiles"]))
        print(f"== {arm} warm precompiles: " + " ".join(f"{item['shape']}={item['seconds']}s+{item['new_graphs']}" for item in warm["precompiles"]))
        print(f"== {arm} cold captures: " + " ".join(f"{item['mode']}={item['seconds']}s/{item['graphs']}" for item in cold["captures"]))
        print(f"== {arm} warm captures: " + " ".join(f"{item['mode']}={item['seconds']}s/{item['graphs']}" for item in warm["captures"]))


if __name__ == "__main__":
    main()
