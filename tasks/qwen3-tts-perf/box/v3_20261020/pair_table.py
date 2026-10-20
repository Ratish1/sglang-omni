"""One row per arm of pair.py runs: tree, card, speed, first audio when streamed, quality, startup and card memory.

usage: python3 pair_table.py LABEL [LABEL ...]   (labels under /data/v3_20261020/pairs)

Columns: label, arm, tree commit, card, startup s, card MiB at health, RTF mean, req/s, latency mean and p99, first
audio p50 and p95 (streaming only), WER corpus over all samples and excluding samples over 50 %, samples over 50 %
(and how many of those ran to the 2048-token cap), max token hits, speaker similarity, and with pair.py --gpu-metrics
the window SM Issue and GR Active over the arm's benchmark.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path("/data/v3_20261020/pairs")
CAP_TOKENS = 2048


def last_float(pattern: str, text: str) -> float | None:
    found = re.findall(pattern, text)
    return float(found[-1]) if found else None


def row(label: str, arm: str) -> str:
    directory = ROOT / label / arm
    tree_lines = (directory / "tree.txt").read_text().splitlines()
    card = tree_lines[0].split()[2]
    commit = tree_lines[1][:9]
    startup = (directory / "startup.txt").read_text().strip() if (directory / "startup.txt").exists() else "-"
    gpu = directory / "gpu_healthy.txt"
    memory = gpu.read_text().splitlines()[int(card)].split(",")[1].strip() if gpu.exists() else "-"
    results = json.load(open(directory / "bench" / "speed_results.json"))
    summary = results["summary"]
    records = [record for record in results["per_request"] if record["is_success"]]
    first_audio = "-"
    if summary.get("audio_ttfp_median_s") is not None:
        first_audio = f"{1e3 * summary['audio_ttfp_median_s']:.1f} / {1e3 * summary['audio_ttfp_p95_s']:.1f}"
    else:
        pass
    wer_all = wer_excl = over = "-"
    log = directory / "score--transcribe-only.log"
    if log.exists():
        text = log.read_text(errors="ignore")
        value = last_float(r"WER \(corpus, micro-avg\):\s+([0-9.]+)", text)
        wer_all = f"{100 * value:.2f} %" if value is not None else "-"
        value = last_float(r"WER corpus \(excl >50%\):\s+([0-9.]+)", text)
        wer_excl = f"{100 * value:.2f} %" if value is not None else "-"
        wer_path = directory / "bench" / "wer_results.json"
        if wer_path.exists():
            tokens = {record["id"]: int(record["completion_tokens"]) for record in records}
            bad = [
                sample for sample in json.load(open(wer_path))["per_sample"]
                if sample["wer"] is not None and float(sample["wer"]) > 0.5
            ]
            capped = sum(tokens.get(sample["id"], 0) >= CAP_TOKENS for sample in bad)
            over = f"{len(bad)} ({capped})"
        else:
            pass
    else:
        pass
    similarity = "-"
    similarity_log = directory / "score--similarity-only.log"
    if similarity_log.exists():
        value = last_float(r"speaker similarity: ([0-9.]+)", similarity_log.read_text(errors="ignore"))
        similarity = f"{value:.4f}" if value is not None else "-"
    else:
        pass
    issue = active = "-"
    metrics = directory / "metrics.txt"
    if metrics.exists():
        text = metrics.read_text(errors="ignore")
        value = last_float(r"SM Issue \[Throughput %\]\s+mean\s+([0-9.]+)", text)
        issue = f"{value:.2f} %" if value is not None else "-"
        value = last_float(r"GR Active \[Throughput %\]\s+mean\s+([0-9.]+)", text)
        active = f"{value:.1f} %" if value is not None else "-"
    else:
        pass
    return (
        f"{label} | {arm} | {commit} | {card} | {startup} | {memory} | {summary['rtf_mean']:.4f} | "
        f"{summary['throughput_qps']:.2f} | {summary['latency_mean_s']:.3f} / {summary['latency_p99_s']:.3f} | "
        f"{first_audio} | {wer_all} | {wer_excl} | {over} | {summary.get('max_token_hits', '-')} | {similarity} | "
        f"{issue} | {active} | {len(records)}/{summary['total_requests']}"
    )


def main() -> None:
    print("label | arm | commit | card | startup s | MiB | RTF | req/s | lat mean / p99 | FA p50 / p95 | WER all | WER excl"
          " | >50 % (capped) | cap hits | SIM | SM Issue | GR Active | ok")
    for label in sys.argv[1:]:
        for arm in ("a", "b"):
            print(row(label, arm))


if __name__ == "__main__":
    main()
