"""Each runaway ended where its codec EOS was likeliest, against the same runaway kept to the cap. cut joins
`probe/qwen3-tts-eos-forensics`'s per-step lines to a layout.py serve directory by speech id, prints how often a
threshold on per-step EOS probability is crossed before a natural stop's own EOS against how often a runaway reaches
it, and where each runaway's peak sits against the natural stops of the same target text; then writes two layout.py
serve directories holding the runaways only, OUT_DIR/capped with the audio as served and OUT_DIR/peak with the audio
ended before the peak-EOS step. report prints both after layout.py score, case by case.

usage: python3 eos_trim.py cut SERVED_DIR PROBE_DIR OUT_DIR [--max-new-tokens 300]
       python3 eos_trim.py report OUT_DIR
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile

SAMPLES_PER_FRAME = 1920
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)


def cut(served: Path, probe: Path, out_dir: Path, max_new_tokens: int) -> None:
    trajectories: dict[str, dict[int, float]] = defaultdict(dict)
    for path in sorted(probe.glob("eos_*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            request_id, frame, probability, _, _ = line.split()
            trajectories[request_id][int(frame)] = float(probability)
    records = [json.loads(line) for line in (served / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    natural_frames: dict[str, list[int]] = defaultdict(list)
    natural_before_stop: list[float] = []
    runaways: list[dict[str, object]] = []
    for record in records:
        trajectory = trajectories.get(str(record.get("speech_id")))
        if not trajectory or "frames" not in record:
            continue
        else:
            pass
        frames = sorted(trajectory)
        probabilities = np.array([trajectory[frame] for frame in frames])
        if record.get("finish_reason") == "length" or int(record["frames"]) >= max_new_tokens:
            peak = int(np.argmax(probabilities))
            runaways.append({**record, "peak": float(probabilities[peak]), "peak_frame": frames[peak]})
        else:
            natural_frames[str(record["input"])].append(int(record["frames"]))
            natural_before_stop.append(float(probabilities[:-1].max()) if probabilities.size > 1 else 0.0)
    print(f"natural stops {len(natural_before_stop)}, runaways {len(runaways)}")
    print("threshold | natural stops crossing it before their own EOS | runaways whose peak reaches it")
    for threshold in THRESHOLDS:
        crossed = sum(value >= threshold for value in natural_before_stop)
        reached = sum(float(runaway["peak"]) >= threshold for runaway in runaways)
        print(
            f"{threshold} | {crossed} ({100 * crossed / len(natural_before_stop):.2f} %) | "
            f"{reached} ({100 * reached / len(runaways):.1f} %)"
        )
    print("id | set | input | peak EOS | peak frame | natural frames of the same text (min / median / max, n)")
    for runaway in sorted(runaways, key=lambda item: (str(item["set"]), str(item["id"]))):
        siblings = sorted(natural_frames.get(str(runaway["input"]), []))
        sibling_text = (
            f"{siblings[0]} / {int(np.median(siblings))} / {siblings[-1]}, {len(siblings)}" if siblings else "-"
        )
        print(
            f"{runaway['id']} | {runaway['set']} | {runaway['input']} | {float(runaway['peak']):.3g} | "
            f"{runaway['peak_frame']} | {sibling_text}"
        )
    for arm in ("capped", "peak"):
        (out_dir / arm / "audio").mkdir(parents=True, exist_ok=True)
        rows = []
        for runaway in runaways:
            samples, rate = soundfile.read(str(runaway["audio"]), dtype="float32")
            kept = samples if arm == "capped" else samples[: int(runaway["peak_frame"]) * SAMPLES_PER_FRAME]
            audio = out_dir / arm / "audio" / Path(str(runaway["audio"])).name
            soundfile.write(str(audio), kept, rate, subtype="PCM_16")
            rows.append({**runaway, "audio": str(audio), "label": f"{runaway['label']}_{arm}"})
        (out_dir / arm / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def report(out_dir: Path) -> None:
    arms = {
        arm: {
            record["id"]: record
            for record in (
                json.loads(line)
                for line in (out_dir / arm / "scored.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            )
        }
        for arm in ("capped", "peak")
    }
    print("id | set | input | peak EOS @ frame | capped WER: transcript | peak WER: transcript")
    for case_id in sorted(arms["capped"], key=lambda key: (arms["capped"][key]["set"], key)):
        capped, peak = arms["capped"][case_id], arms["peak"][case_id]
        print(
            f"{case_id} | {capped['set']} | {capped['input']} | {capped['peak']:.3g} @ {capped['peak_frame']} | "
            f"{capped['wer']:.2f}: {capped['transcript']} | {peak['wer']:.2f}: {peak['transcript']}"
        )
    for arm, records in arms.items():
        values = [float(record["wer"]) for record in records.values()]
        exact = sum(value == 0 for value in values)
        print(f"{arm}: n {len(values)}, exact {exact}, mean WER {np.mean(values):.3f}, median {np.median(values):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cut_parser = commands.add_parser("cut")
    cut_parser.add_argument("served", type=Path)
    cut_parser.add_argument("probe", type=Path)
    cut_parser.add_argument("out", type=Path)
    cut_parser.add_argument("--max-new-tokens", type=int, default=300)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "cut":
        cut(args.served, args.probe, args.out, args.max_new_tokens)
    else:
        report(args.out)


if __name__ == "__main__":
    main()
