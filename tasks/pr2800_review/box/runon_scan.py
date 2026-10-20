"""Every request over 20 s of audio in the passes still on the box: the talker run-on candidates, with each prompt's
usual length from the other passes.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/runon_scan.py --out /data/pr2800/<dir>
Walks /data/pr2800/run*/ for speed_results.json files (a pass of a pair_check.py arm or a queue's own pass), reads
every successful per-request record, and keeps the ones whose audio_duration_s exceeds --seconds. For each kept
request: the run, arm directory and pass, the pass's seed and concurrency from its config, the prompt id and text,
the thinker's completion tokens and finish reason, the audio and latency seconds, and the median audio of the same
prompt over every other pass that served it. Copies each kept request's wav into <out>/audio when it still exists.
Writes <out>/runons.json and prints the totals and every kept request.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

DATA = Path("/data/pr2800")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    passes = sorted(path for path in DATA.glob("run*/**/speed_results.json") if args.out not in path.parents)
    durations: dict[str, list[float]] = defaultdict(list)
    kept = []
    scanned = 0
    for path in passes:
        results = json.loads(path.read_text())
        config = results.get("config") or {}
        for row in results["per_request"]:
            if not row.get("is_success"):
                continue
            else:
                pass
            scanned += 1
            durations[row["id"]].append(row["audio_duration_s"])
            if row["audio_duration_s"] > args.seconds:
                kept.append(dict(
                    pass_dir=str(path.parent.relative_to(DATA)), seed=config.get("seed"),
                    concurrency=config.get("max_concurrency"), id=row["id"], text=row["text"],
                    completion_tokens=row["completion_tokens"], finish_reason=row["finish_reason"],
                    audio_s=row["audio_duration_s"], latency_s=row["latency_s"], wav_path=row["wav_path"],
                ))
            else:
                pass
    for record in kept:
        others = list(durations[record["id"]])
        others.remove(record["audio_s"])
        record["usual_audio_s"] = round(statistics.median(others), 3) if others else None
        record["served_elsewhere"] = len(others)
        wav = Path(record["wav_path"])
        if wav.exists():
            target = args.out / "audio" / f"{record['pass_dir'].replace('/', '__')}__{wav.name}"
            shutil.copyfile(wav, target)
            record["copied_wav"] = target.name
        else:
            record["copied_wav"] = None
    (args.out / "runons.json").write_text(json.dumps(dict(passes=len(passes), requests=scanned, seconds=args.seconds,
                                                          runons=kept), indent=1))
    print(f"{len(passes)} passes, {scanned} requests, {len(kept)} over {args.seconds} s of audio")
    for record in kept:
        print(f"  {record['pass_dir']} seed {record['seed']} c{record['concurrency']} {record['id']}: "
              f"{record['audio_s']} s (usual {record['usual_audio_s']} s over {record['served_elsewhere']}), "
              f"thinker {record['completion_tokens']} tokens {record['finish_reason']}, "
              f"wav {'copied' if record['copied_wav'] else 'gone'}: {record['text'][:60]!r}")


if __name__ == "__main__":
    main()
