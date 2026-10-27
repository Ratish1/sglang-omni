"""The talker batch sizes each request decoded in, against its quality and its playback, from boots run with the hop
probe and OMNI_HOP_PROBE_TALKER_BATCHES=1 ("req": request id and prompt text; "tb": each decode step's rows and ids).

usage: PYTHONPATH=<tree> python3 talker_batch_report.py <boot dir> [<boot dir> ...]

Per boot, requests are matched to benchmark samples by the probe's first 300 characters of the prompt, rebuilt for
each SeedTTS EN sample with the benchmark's voice-clone template (a prefix shared by two samples is dropped), in pass
order:
c64_warm, the c64_clean repeats in order, c64_events (a text's earliest extra requests, the warmup's, are dropped).
Each request's decode steps give its mean and peak rows and its share of steps above 32 rows. Over the clean
repeats, requests are grouped by outcome: WER above 50 %, a run-on (more than 1 s of audio per completion token),
an estimated playback stall (underrun.py's rule), or none of these; each group's row statistics are printed.
"""

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.dataset.seedtts import load_seedtts_samples

RUN_ON_SECONDS_PER_TOKEN = 1.0
SEEDTTS = "zhaochenyang20/seed-tts-eval-arrow"
PROMPT_PREFIX_CHARS = 300


def sample_by_prompt_prefix() -> dict[str, str]:
    """The probe's text prefix of each SeedTTS EN voice-clone prompt, as benchmarks/tasks/tts.py builds it."""
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for sample in load_seedtts_samples(SEEDTTS, None, split="en"):
        prompt = (
            f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
            f"Now please read the following text out loud in the same voice and style: "
            f"{sample.target_text}"
        )
        by_prefix[prompt[:PROMPT_PREFIX_CHARS]].append(sample.sample_id)
    return {prefix: ids[0] for prefix, ids in by_prefix.items() if len(ids) == 1}


def probe_records(boot: Path) -> tuple[list[dict], list[dict]]:
    requests, steps = [], []
    for path in glob.glob(str(boot / "probe" / "*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
                elif item.get("k") == "req":
                    requests.append(item)
                elif item.get("k") == "tb":
                    steps.append(item)
                else:
                    pass
    return requests, steps


def stall_s(request: dict) -> float:
    gaps = request["inter_chunk_s"] or []
    chunk_duration = request["audio_duration_s"] / (len(gaps) + 1)
    arrival, worst = 0.0, 0.0
    for index, gap in enumerate(gaps, start=1):
        arrival += gap
        worst = max(worst, arrival - index * chunk_duration)
    return worst


def boot_rows(boot: Path, sample_by_prefix: dict[str, str]) -> list[dict]:
    passes = [boot / "c64_warm" / "bench"]
    passes += sorted((boot / "c64_clean" / "bench").glob("rep*"))
    passes += [boot / "c64_events" / "bench"]
    requests, steps = probe_records(boot)
    rows_by_rid: dict[str, list[int]] = defaultdict(list)
    for step in steps:
        for rid in step["rids"]:
            rows_by_rid[rid].append(step["rows"])
    rids_by_sample: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for request in requests:
        sample = sample_by_prefix.get(request.get("text") or "")
        if sample is not None:
            rids_by_sample[sample].append((request["ns"], request["rid"]))
        else:
            pass
    out = []
    for pass_index, pass_dir in enumerate(passes):
        if pass_dir.parent.name != "c64_clean":
            continue
        else:
            pass
        speed = {r["id"]: r for r in json.load(open(pass_dir / "speed_results.json"))["per_request"]}
        wer_path = pass_dir / "wer_results.json"
        wer = (
            {s["id"]: s.get("wer") for s in json.load(open(wer_path))["per_sample"]}
            if wer_path.exists()
            else {}
        )
        for sample, timed_rids in rids_by_sample.items():
            ordered = [rid for _, rid in sorted(timed_rids)]
            if len(ordered) < len(passes):
                continue
            else:
                pass
            rid = ordered[len(ordered) - len(passes) + pass_index]
            request = speed.get(sample)
            rows = rows_by_rid.get(rid, [])
            if request is None or not request["is_success"] or not rows:
                continue
            else:
                pass
            tokens = max(int(request["completion_tokens"] or 1), 1)
            out.append(
                {
                    "boot": boot.name,
                    "pass": pass_dir.name,
                    "sample": sample,
                    "mean_rows": statistics.mean(rows),
                    "peak_rows": max(rows),
                    "share_above_32": sum(r > 32 for r in rows) / len(rows),
                    "bad_wer": (wer.get(sample) or 0.0) > 0.5,
                    "run_on": request["audio_duration_s"] / tokens > RUN_ON_SECONDS_PER_TOKEN,
                    "stall": stall_s(request) > 0,
                }
            )
    return out


def summarize(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {label:28s} none")
        return
    else:
        pass
    print(
        f"  {label:28s} {len(rows):5d} requests, mean rows {statistics.mean(r['mean_rows'] for r in rows):5.1f}, "
        f"peak {statistics.mean(r['peak_rows'] for r in rows):5.1f}, "
        f"steps above 32 rows {100 * statistics.mean(r['share_above_32'] for r in rows):5.1f} %"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("boots", nargs="+")
    args = parser.parse_args()
    sample_by_prefix = sample_by_prompt_prefix()
    for boot in args.boots:
        rows = boot_rows(Path(boot), sample_by_prefix)
        print(f"== {boot}: {len(rows)} matched clean requests")
        summarize("WER above 50 %", [r for r in rows if r["bad_wer"]])
        summarize("run-on", [r for r in rows if r["run_on"]])
        summarize("estimated stall", [r for r in rows if r["stall"]])
        summarize("none of these", [r for r in rows if not (r["bad_wer"] or r["run_on"] or r["stall"])])
        for kind in ("bad_wer", "run_on"):
            for r in rows:
                if r[kind]:
                    print(f"    {kind} {r['pass']} {r['sample']} mean rows {r['mean_rows']:.1f} peak {r['peak_rows']}")
                else:
                    pass


if __name__ == "__main__":
    main()
