"""The talker batch sizes each request decoded in, against its quality and its playback, from boots run with the hop
probe and OMNI_HOP_PROBE_TALKER_BATCHES=1 ("req": request id and prompt text; "tb": each decode step's rows and ids).

usage: python3 talker_batch_report.py <boot dir> [<boot dir> ...]

Per boot, requests are matched to benchmark samples by the sample's target text inside the prompt, in pass order:
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

RUN_ON_SECONDS_PER_TOKEN = 1.0


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


def boot_rows(boot: Path) -> list[dict]:
    passes = [boot / "c64_warm" / "bench"]
    passes += sorted((boot / "c64_clean" / "bench").glob("rep*"))
    passes += [boot / "c64_events" / "bench"]
    samples_by_text: dict[str, list[str]] = defaultdict(list)
    with open(passes[1] / "speed_results.json", encoding="utf-8") as source:
        for request in json.load(source)["per_request"]:
            samples_by_text[request["text"]].append(request["id"])
    requests, steps = probe_records(boot)
    rows_by_rid: dict[str, list[int]] = defaultdict(list)
    for step in steps:
        for rid in step["rids"]:
            rows_by_rid[rid].append(step["rows"])
    rids_by_text: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for request in requests:
        text = request.get("text") or ""
        for target in samples_by_text:
            if text.endswith(target):
                rids_by_text[target].append((request["ns"], request["rid"]))
                break
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
        for text, sample_ids in samples_by_text.items():
            ordered = [rid for _, rid in sorted(rids_by_text.get(text, []))]
            if len(sample_ids) != 1 or len(ordered) < len(passes):
                continue
            else:
                pass
            rid = ordered[len(ordered) - len(passes) + pass_index]
            sample = sample_ids[0]
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
    for boot in args.boots:
        rows = boot_rows(Path(boot))
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
