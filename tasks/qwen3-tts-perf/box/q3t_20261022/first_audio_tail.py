"""Where a pair's first-audio tail comes from: per arm, the p99 over all requests and over the requests after the
opening wave (the requests the closed-loop client sent without waiting for a slot, all at once), the list positions of
the slowest 1 %, and for those prompts the other arm's first audio.

usage: python3 first_audio_tail.py LABEL [LABEL ...]   (labels under PAIR_ROOT)

per_request keeps the client's submission order, so a request's list position is its send order.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PAIR_ROOT", "/data/q3t_20261022/pairs"))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def arm_requests(label: str, arm: str) -> tuple[list[dict[str, object]], int]:
    results = json.load(open(ROOT / label / arm / "bench" / "speed_results.json"))
    return results["per_request"], int(results["config"]["concurrency"])


def main() -> None:
    for label in sys.argv[1:]:
        arms = {arm: arm_requests(label, arm) for arm in ("a", "b")}
        concurrency = arms["a"][1]
        first_audio_ms = {
            arm: {str(request["id"]): 1000 * float(request["audio_ttfp_s"]) for request in requests}
            for arm, (requests, _) in arms.items()
        }
        print(f"{label} (concurrency {concurrency})")
        for arm, (requests, _) in arms.items():
            other = "b" if arm == "a" else "a"
            values = [1000 * float(request["audio_ttfp_s"]) for request in requests]
            in_opening_wave = [not request["waited_for_slot"] for request in requests]
            after_wave = [value for value, opening in zip(values, in_opening_wave) if not opening]
            slowest = sorted(range(len(values)), key=lambda index: values[index], reverse=True)[: len(values) // 100]
            in_wave = sum(in_opening_wave[index] for index in slowest)
            print(
                f"  {arm}: p99 {percentile(values, 0.99):.1f} ms, after the opening wave {percentile(after_wave, 0.99):.1f}"
                f" ms, p95 {percentile(values, 0.95):.1f} / {percentile(after_wave, 0.95):.1f} ms;"
                f" slowest {len(slowest)}: {in_wave} in the opening wave, positions {sorted(slowest)}"
            )
            print(
                "     those prompts in this arm / the other: "
                + ", ".join(
                    f"{first_audio_ms[arm][str(requests[index]['id'])]:.0f}/"
                    f"{first_audio_ms[other].get(str(requests[index]['id']), float('nan')):.0f}"
                    for index in sorted(slowest)
                )
            )


if __name__ == "__main__":
    main()
