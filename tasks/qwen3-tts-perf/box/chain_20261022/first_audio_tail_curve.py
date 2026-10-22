"""Whether a pair's first-audio tail after the opening wave moved as a whole or at one percentile: per arm the
after-wave quantiles 0.90 to 0.995 and the max, and per quantile the b - a difference with a bootstrap 95 % interval
(2,000 resamples of each arm's after-wave requests, seeded).

usage: PAIR_ROOT=<pairs> python3 first_audio_tail_curve.py LABEL [LABEL ...]
"""

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(os.environ["PAIR_ROOT"])
QUANTILES = (0.90, 0.95, 0.97, 0.98, 0.99, 0.995)
RESAMPLES = 2000


def quantile(ordered: list[float], fraction: float) -> float:
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def after_wave_ms(label: str, arm: str) -> list[float]:
    requests = json.load(open(ROOT / label / arm / "bench" / "speed_results.json"))["per_request"]
    return sorted(1000 * float(request["audio_ttfp_s"]) for request in requests if request["waited_for_slot"])


def main() -> None:
    generator = random.Random(0)
    for label in sys.argv[1:]:
        arms = {arm: after_wave_ms(label, arm) for arm in ("a", "b")}
        print(f"{label}: after-wave requests a {len(arms['a'])}, b {len(arms['b'])}; max {arms['a'][-1]:.1f} / {arms['b'][-1]:.1f} ms")
        resampled = {
            arm: [sorted(generator.choices(values, k=len(values))) for _ in range(RESAMPLES)]
            for arm, values in arms.items()
        }
        for fraction in QUANTILES:
            point_a, point_b = quantile(arms["a"], fraction), quantile(arms["b"], fraction)
            deltas = sorted(
                quantile(resampled["b"][index], fraction) - quantile(resampled["a"][index], fraction)
                for index in range(RESAMPLES)
            )
            print(
                f"  q{fraction:.3f}: {point_a:6.1f} / {point_b:6.1f} ms, b - a {point_b - point_a:+5.1f} "
                f"(95 % {quantile(deltas, 0.025):+5.1f} to {quantile(deltas, 0.975):+5.1f})"
            )


if __name__ == "__main__":
    main()
