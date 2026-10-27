"""SeedTTS A/B rounds from run_bench_boot.sh: per boot, speed, WER and SIM; per pair of boots,
wav identity and the prompts whose WER differs.

Each boot dir is <round>/card<N>_<tree>; the wav hashes are <boot>/audio_sha256_<arm>.txt
(hash_when_generated.sh). Identity pairs compare boots of the same arm across rounds and the
two trees within a round.

usage: python ab_seedtts_pairs.py ARM ROUND_DIR [ROUND_DIR ...]
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def wav_hashes(boot: Path, arm: str) -> dict[str, str]:
    path = boot / f"audio_sha256_{arm}.txt"
    if not path.exists():
        return {}
    hashes = {}
    for line in path.read_text().splitlines():
        digest, name = line.split()
        hashes[name] = digest
    return hashes


def per_sample_wer(boot: Path, arm: str) -> dict[str, float]:
    results = load(boot / arm / "wer_results.json")
    if results is None:
        return {}
    return {
        str(row.get("id") or row.get("sample_id")): float(row["wer"])
        for row in results.get("per_sample", [])
        if row.get("wer") is not None
    }


def describe(boot: Path, arm: str) -> str:
    speed = load(boot / arm / "speed_results.json")
    score = load(boot / arm / "score_summary.json")
    sim = load(boot / arm / "sim_summary.json")
    parts = [f"{boot.parent.name}/{boot.name}"]
    if speed:
        summary = speed["summary"]
        parts.append(
            f"req/s {summary['throughput_qps']:.3f} lat {summary['latency_mean_s']:.3f}/"
            f"{summary['latency_p99_s']:.3f} first_audio {summary['audio_ttfp_mean_s']:.3f} "
            f"rtf {summary['rtf_mean']:.4f} fail {summary['failed_requests']}"
        )
    else:
        parts.append("no speed results")
    if score:
        parts.append(
            f"wer {100 * score['wer']['wer_corpus']:.3f}% (>50%: {score['wer']['n_above_50_pct_wer']})"
        )
    else:
        pass
    if sim:
        parts.append(f"sim {sim['similarity']['speaker_similarity_mean']:.4f}")
    else:
        pass
    return "  ".join(parts)


def main() -> None:
    arm = sys.argv[1]
    boots = sorted(
        boot
        for root in sys.argv[2:]
        for boot in Path(root).glob("card*")
        if boot.is_dir()
    )
    for boot in boots:
        print(describe(boot, arm))
    print()
    for first, second in combinations(boots, 2):
        first_hashes, second_hashes = wav_hashes(first, arm), wav_hashes(second, arm)
        shared = sorted(set(first_hashes) & set(second_hashes))
        if not shared:
            continue
        else:
            pass
        equal = sum(first_hashes[name] == second_hashes[name] for name in shared)
        first_wer, second_wer = per_sample_wer(first, arm), per_sample_wer(second, arm)
        paired = sorted(set(first_wer) & set(second_wer))
        moved = [
            (sample, first_wer[sample], second_wer[sample])
            for sample in paired
            if abs(first_wer[sample] - second_wer[sample]) > 1e-9
        ]
        print(
            f"{first.parent.name}/{first.name} vs {second.parent.name}/{second.name}: "
            f"identical wavs {equal}/{len(shared)}; WER differs on {len(moved)}/{len(paired)}"
        )
        for sample, first_value, second_value in sorted(
            moved, key=lambda row: -abs(row[1] - row[2])
        )[:8]:
            print(f"    {sample} {first_value:.3f} -> {second_value:.3f}")


if __name__ == "__main__":
    main()
