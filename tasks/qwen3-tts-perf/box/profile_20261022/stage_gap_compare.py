"""Two trees' per-request first-audio stage gaps pooled over pairs, from pair.py --events: for each arm group (named by
tree commit), every request's gap between consecutive first-audio milestones (first_audio_chain.py's chain) and its
total; per gap the median and mean of each group and the change's minus the base's, with a bootstrap 95 % interval
(2,000 resamples of the requests). Then every prompt's first audio paired by sample id across each pair's two arms
(bench records), pooled, with its mean, median and bootstrap interval.

usage: PAIR_ROOT=<pairs> python3 stage_gap_compare.py BASE_COMMIT CHANGE_COMMIT LABEL [LABEL ...]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

from first_audio_chain import chain_gaps

RESAMPLES = 2000
GAPS = (
    "vocoder.qwen3_tts_vocoder_decode_enqueued -> vocoder.qwen3_tts_vocoder_decode_dispatched",
    "vocoder.qwen3_tts_vocoder_decode_dispatched -> vocoder.qwen3_tts_vocoder_decode_launched",
    "vocoder.qwen3_tts_vocoder_decode_launched -> vocoder.qwen3_tts_vocoder_decode_resolved",
    "TOTAL admission -> first chunk at coordinator",
)


def resampled(values: np.ndarray, statistic, generator: np.random.Generator) -> np.ndarray:
    draws = generator.choice(values, size=(RESAMPLES, values.size), replace=True)
    return statistic(draws, axis=1)


def main() -> None:
    base, change, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
    root = Path(os.environ["PAIR_ROOT"])
    pooled: dict[str, dict[str, list[float]]] = {base: {}, change: {}}
    paired_first_audio: list[float] = []
    for label in labels:
        commits, ttfp = {}, {}
        for arm in ("a", "b"):
            commit = (root / label / arm / "tree.txt").read_text().splitlines()[1][:9]
            commits[arm] = commit
            for name, values in chain_gaps(root / label / arm / "events").items():
                pooled.setdefault(commit, {}).setdefault(name, []).extend(values)
            ttfp[commit] = {
                record["id"]: 1000 * record["audio_ttfp_s"]
                for record in json.loads((root / label / arm / "bench" / "speed_results.json").read_text())["per_request"]
                if record["is_success"] and record["audio_ttfp_s"] is not None
            }
        print(f"{label}: a {commits['a']}, b {commits['b']}")
        if base in ttfp and change in ttfp:
            paired_first_audio.extend(
                ttfp[change][prompt] - ttfp[base][prompt] for prompt in ttfp[base] if prompt in ttfp[change]
            )
        else:
            pass
    generator = np.random.default_rng(0)
    for name in GAPS:
        values_a = np.asarray(pooled[base].get(name, []))
        values_b = np.asarray(pooled[change].get(name, []))
        if values_a.size == 0 or values_b.size == 0:
            print(f"{name}: no requests")
            continue
        else:
            pass
        parts = []
        for label, statistic in (("median", np.median), ("mean", np.mean)):
            deltas = resampled(values_b, statistic, generator) - resampled(values_a, statistic, generator)
            low, high = np.quantile(deltas, (0.025, 0.975))
            parts.append(
                f"{label} {statistic(values_a):.3f} / {statistic(values_b):.3f} ms, change "
                f"{statistic(values_b) - statistic(values_a):+.3f} (95 % {low:+.3f} to {high:+.3f})"
            )
        print(f"{name} ({values_a.size} / {values_b.size} requests): " + "; ".join(parts))
    if paired_first_audio:
        deltas = np.asarray(paired_first_audio)
        low, high = np.quantile(resampled(deltas, np.mean, generator), (0.025, 0.975))
        print(
            f"first audio per prompt, change minus base, {deltas.size} prompt pairs: mean {deltas.mean():+.3f} ms "
            f"(95 % {low:+.3f} to {high:+.3f}), median {np.median(deltas):+.3f} ms"
        )
    else:
        pass


if __name__ == "__main__":
    main()
