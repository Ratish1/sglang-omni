"""Runs of exact digital zero in served Qwen3-TTS audio, natural stops against runaways. scan prints, per outcome and
set, how many clips hold a zero run of at least each length and where a runaway's first such run after the opening
starts; cut writes a layout.py serve directory of the runaways (--outcome length) or of the natural stops that hold
such a run (--outcome stop), ended where their first zero run of at least --min-ms after the opening starts; report
prints it against the same clips as served (a scored.jsonl) after layout.py score.

usage: python3 zero_runs.py scan SERVED_DIR
       python3 zero_runs.py cut SERVED_DIR OUT_DIR [--min-ms 1000] [--outcome {length,stop}]
       python3 zero_runs.py report BASE_SCORED_JSONL OUT_DIR
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile

SAMPLE_RATE_HZ = 24000
RUN_MS = (250, 500, 1000, 2000, 3000)


def zero_runs_ms(path: str) -> list[tuple[float, float]]:
    samples, _ = soundfile.read(path, dtype="int16")
    zero = np.concatenate(([0], (samples == 0).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(zero))
    starts, ends = edges[0::2], edges[1::2]
    return [(1000 * start / SAMPLE_RATE_HZ, 1000 * (end - start) / SAMPLE_RATE_HZ) for start, end in zip(starts, ends)]


def load(served: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in (served / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line and "audio" in json.loads(line)
    ]


def scan(served: Path) -> None:
    rows: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    first_long: list[str] = []
    for record in load(served):
        runs = zero_runs_ms(str(record["audio"]))
        outcome = "runaway" if record.get("finish_reason") == "length" else "natural"
        rows[f"{outcome} {record['set']}"].append(runs)
        if outcome == "runaway":
            start = next((f"{start:.0f}" for start, length in runs if start > 0 and length >= 1000), "-")
            first_long.append(f"{record['id']} | {record['input']} | {start}")
        else:
            pass
    print("outcome set | n | clips with a zero run of at least " + " / ".join(f"{ms} ms" for ms in RUN_MS))
    for key in sorted(rows):
        clips = rows[key]
        counts = [sum(any(length >= ms for _, length in runs) for runs in clips) for ms in RUN_MS]
        print(f"{key} | {len(clips)} | " + " / ".join(str(count) for count in counts))
    print("runaway | input | first zero run of at least 1000 ms starts at ms")
    print("\n".join(first_long))


def cut(served: Path, out_dir: Path, min_ms: float, outcome: str) -> None:
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for record in load(served):
        if record.get("finish_reason") != outcome:
            continue
        else:
            pass
        samples, rate = soundfile.read(str(record["audio"]), dtype="float32")
        start = next(
            (start for start, length in zero_runs_ms(str(record["audio"])) if start > 0 and length >= min_ms), None
        )
        if outcome == "stop" and start is None:
            continue
        else:
            pass
        kept = samples if start is None else samples[: int(start * rate / 1000)]
        audio = out_dir / "audio" / Path(str(record["audio"])).name
        soundfile.write(str(audio), kept, rate, subtype="PCM_16")
        rows.append({**record, "audio": str(audio), "zero_start_ms": start, "label": f"{record['label']}_zero"})
    (out_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def report(base_path: Path, out_dir: Path) -> None:
    def scored(path: Path) -> dict[str, dict[str, object]]:
        return {
            record["id"]: record
            for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        }

    zero = scored(out_dir / "scored.jsonl")
    capped = {key: value for key, value in scored(base_path).items() if key in zero}
    print("id | input | zero run starts at ms | served WER: transcript | zero-cut WER: transcript")
    for case_id in sorted(capped, key=lambda key: (capped[key]["set"], key)):
        print(
            f"{case_id} | {capped[case_id]['input']} | {zero[case_id]['zero_start_ms']} | "
            f"{capped[case_id]['wer']:.2f}: {capped[case_id]['transcript']} | "
            f"{zero[case_id]['wer']:.2f}: {zero[case_id]['transcript']}"
        )
    for arm, records in (("served", capped), ("zero", zero)):
        values = [float(record["wer"]) for record in records.values()]
        print(f"{arm}: n {len(values)}, exact {sum(value == 0 for value in values)}, mean WER {np.mean(values):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan")
    scan_parser.add_argument("served", type=Path)
    cut_parser = commands.add_parser("cut")
    cut_parser.add_argument("served", type=Path)
    cut_parser.add_argument("out", type=Path)
    cut_parser.add_argument("--min-ms", type=float, default=1000.0)
    cut_parser.add_argument("--outcome", choices=("length", "stop"), default="length")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("base", type=Path)
    report_parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "scan":
        scan(args.served)
    elif args.command == "cut":
        cut(args.served, args.out, args.min_ms, args.outcome)
    else:
        report(args.base, args.out)


if __name__ == "__main__":
    main()
