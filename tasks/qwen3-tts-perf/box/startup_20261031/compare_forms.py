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
    print("arm | cold build s | cold precompile s | warm build s | warm precompile s | programs | replay sum ms (ref) | worst key ratio | bit-equal keys | worst relative L2")
    for arm in arms:
        cold, warm = load(directory, arm, "cold"), load(directory, arm, "warm")
        if warm is None:
            print(f"{arm} | missing")
            continue
        else:
            pass
        ms = {(item["runner"], item["frames"], item["batch"]): item["ms"] for item in warm["replays"]}
        shared = sorted(set(ms) & set(reference_ms))
        ratios = {key: ms[key] / reference_ms[key] for key in shared}
        worst = max(ratios, key=ratios.get)
        waves = torch.load(f"{directory}/{arm}_warm.pt")
        labels = [label for label in reference_waves if label in waves]
        equal = sum(torch.equal(waves[label], reference_waves[label]) for label in labels)
        relative = max(
            (float((waves[label] - reference_waves[label]).norm() / reference_waves[label].norm()) for label in labels),
            default=float("nan"),
        )
        cold_build, cold_precompile = (cold["build_s"], cold["precompile_total_s"]) if cold else ("-", "-")
        print(
            f"{arm} | {cold_build} | {cold_precompile} | {warm['build_s']} | {warm['precompile_total_s']} | "
            f"{warm['unique_graphs']} | {sum(ms[key] for key in shared):.3f} ({sum(reference_ms[key] for key in shared):.3f}) | "
            f"{worst} {ratios[worst]:.3f} | {equal}/{len(labels)} | {relative:.2e}"
        )
    for arm in arms:
        warm = load(directory, arm, "warm")
        if warm is None:
            continue
        else:
            pass
        print(f"\n== {arm} per key ms (warm run): " + " ".join(f"{item['runner']}:{item['frames']}x{item['batch']}={item['ms']}" for item in warm["replays"]))
        cold = load(directory, arm, "cold")
        for phase, run in (("cold", cold), ("warm", warm)):
            if run is None:
                continue
            else:
                pass
            print(f"== {arm} {phase} precompiles: " + " ".join(f"{item['shape']}={item['seconds']}s+{item['new_graphs']}" for item in run["precompiles"]))
            print(f"== {arm} {phase} captures: " + " ".join(f"{item['mode']}={item['seconds']}s/{item['graphs']}" for item in run["captures"]))


if __name__ == "__main__":
    main()
