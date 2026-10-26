"""Read the preprocessing study: per arm, SeedTTS speed and first audio states, CPU, and Video-MME speed and answers.

usage: python3 pre_report.py /data/runtime-perf/pre --arms A,T16,T32,P2T16,P2T32

Boots are <run>/s<arm><suffix> (SeedTTS, c16 and c64 warm, clean, events) and <run>/v<arm><suffix> (Video-MME CI text,
c16 clean = cold pass, c16 warm = hot pass). Every value is the mean over the arm's boots, each boot listed after it.
"""

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

from first_audio_states import CHAIN, TAIL, state_rows

SPEECH_ROWS = (
    ("req/s", "throughput_qps", 1.0),
    ("first audio mean ms", "audio_ttfp_mean_s", 1e3),
    ("first audio p99 ms", "audio_ttfp_p99_s", 1e3),
    ("latency median ms", "latency_median_s", 1e3),
    ("latency p99 ms", "latency_p99_s", 1e3),
    ("inter chunk mean ms", "inter_chunk_mean_s", 1e3),
    ("RTF mean", "rtf_mean", 1.0),
)
STATES = (
    "server first audio",
    "preprocessing queue",
    "preprocessing work",
    "hop to the audio encoder",
)
VIDEO_ROWS = (
    ("req/s", "throughput_qps", 1.0),
    ("latency mean s", "latency_mean_s", 1.0),
    ("latency p95 s", "latency_p95_s", 1.0),
)


def boots_of(run: Path, prefix: str, arm: str) -> list[Path]:
    return sorted(
        Path(path)
        for path in glob.glob(f"{run}/{prefix}{arm}*")
        if Path(path).is_dir()
        and Path(path).name[len(prefix) + len(arm) :] in ("1", "2", "a", "b")
    )


def text(values: list[float], digits: int) -> str:
    return f"{statistics.fmean(values):.{digits}f} ({' / '.join(f'{v:.{digits}f}' for v in values)})"


def thinker_token_interval(boot: Path, cell: str) -> float:
    sent: dict[str, dict[int, float]] = defaultdict(dict)
    for path in glob.glob(f"{boot}/{cell}/events/events_thinker_*.jsonl"):
        with open(path, encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                metadata = record.get("metadata") or {}
                if (
                    record["event_name"] == "stage_stream_chunk_sent"
                    and metadata.get("to_stage") == "talker_ar"
                ):
                    sent[record["request_id"]][metadata["chunk_id"]] = (
                        record["timestamp_ns"] / 1e6
                    )
                else:
                    pass
    gaps = [
        chunks[i] - chunks[i - 1]
        for chunks in sent.values()
        for i in chunks
        if i - 1 in chunks
    ]
    return statistics.median(gaps)


def preprocessing_cores(boot: Path, cell: str) -> float:
    pids = {
        Path(path).stem.rsplit("_", 1)[1]
        for path in glob.glob(f"{boot}/c16_events/events/events_preprocessing_*")
    }
    cpu = json.loads((boot / cell / "thread_cpu.json").read_text(encoding="utf-8"))
    window = json.loads((boot / cell / "window.json").read_text(encoding="utf-8"))
    seconds = sum(
        value["cpu_s"] - cpu["start"].get(key, {"cpu_s": 0.0})["cpu_s"]
        for key, value in cpu["end"].items()
        if key.split("/")[0] in pids
    )
    return seconds / (window["end"] - window["start"])


def speech(run: Path, arm: str) -> None:
    boots = boots_of(run, "s", arm)
    print(f"== SeedTTS arm {arm}: {', '.join(boot.name for boot in boots)}")
    for load in (16, 64):
        for label, key, scale in SPEECH_ROWS:
            values = [
                json.loads(
                    (boot / f"c{load}_clean/bench/speed_results.json").read_text()
                )["summary"][key]
                * scale
                for boot in boots
            ]
            print(
                f"  c{load} {label:22s} {text(values, 3 if key == 'rtf_mean' else 2)}"
            )
        rows = [state_rows(str(boot), f"c{load}_events") for boot in boots]
        for index, label in enumerate(
            ("server first audio", *(name for _, _, name in CHAIN[1:]))
        ):
            if label not in STATES:
                continue
            else:
                pass
            means = [
                statistics.fmean(row[index] for row in boot_rows) for boot_rows in rows
            ]
            tails = []
            for boot_rows in rows:
                ordered = sorted(boot_rows, key=lambda row: row[0])
                tails.append(
                    statistics.fmean(
                        row[index] for row in ordered[int(len(ordered) * (1 - TAIL)) :]
                    )
                )
            print(
                f"  c{load} {label:22s} mean {text(means, 1)}  slowest 5 % {text(tails, 1)}"
            )
        intervals = [thinker_token_interval(boot, f"c{load}_events") for boot in boots]
        cores = [preprocessing_cores(boot, f"c{load}_clean") for boot in boots]
        print(f"  c{load} thinker token interval p50 ms {text(intervals, 1)}")
        print(f"  c{load} preprocessing cores     {text(cores, 1)}")


def video(run: Path, arm: str) -> dict[str, list[dict[str, str]]]:
    boots = boots_of(run, "v", arm)
    print(f"== Video-MME CI arm {arm}: {', '.join(boot.name for boot in boots)}")
    answers: dict[str, list[dict[str, str]]] = {"cold": [], "hot": []}
    for kind, cell in (("cold", "c16_clean"), ("hot", "c16_warm")):
        results = [
            json.loads((boot / cell / "bench/videomme_results.json").read_text())
            for boot in boots
        ]
        for label, key, scale in VIDEO_ROWS:
            print(
                f"  {kind} {label:16s} {text([result['speed'][key] * scale for result in results], 3)}"
            )
        print(
            f"  {kind} accuracy         {text([result['summary']['accuracy'] for result in results], 3)}"
        )
        for result in results:
            answers[kind].append(
                {
                    record["sample_id"]: record["predicted"]
                    for record in result["per_sample"]
                }
            )
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--arms", default="A,T16,T32,P2T16,P2T32")
    args = parser.parse_args()
    run = Path(args.run)
    arms = args.arms.split(",")
    for arm in arms:
        speech(run, arm)
    answers = {arm: video(run, arm) for arm in arms}
    print(
        "== Video-MME predicted answers that differ from the A arm's first cold pass (count of 50)"
    )
    reference = answers[arms[0]]["cold"][0]
    for arm in arms:
        for kind in ("cold", "hot"):
            flips = [
                sum(reference[sample] != passed[sample] for sample in reference)
                for passed in answers[arm][kind]
            ]
            print(f"  {arm} {kind}: {flips}")


if __name__ == "__main__":
    main()
