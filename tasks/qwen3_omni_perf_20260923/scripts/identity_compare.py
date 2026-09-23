"""Per-sample identity of two run_bench_boot.sh dirs (temperature 0, same samples).

For every arm present in both dirs, compares each sample's result record without its
timing fields, and the sha256 of each generated wav under <arm>/audio. Prints, per arm,
the samples compared, the samples whose record differs (with the differing keys of the
first few) and the wavs that differ. Run on the box before the wavs are deleted.

usage: python identity_compare.py RUN_A RUN_B
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

TIMING_KEY = re.compile(
    r"latency|time|ttf|ttft|rtf|duration|throughput|rate|_s$|_ms$|wav_path|audio_path|itl"
)


def records(arm_dir: Path) -> dict[str, dict]:
    for path in sorted(arm_dir.glob("*_results.json")):
        per_sample = json.loads(path.read_text()).get("per_sample")
        if per_sample:
            break
    else:
        return {}
    keyed = {}
    for index, record in enumerate(per_sample):
        sample_id = str(record.get("id", record.get("sample_id", index)))
        keyed[sample_id] = {
            key: value for key, value in record.items() if not TIMING_KEY.search(key)
        }
    return keyed


def wav_digests(arm_dir: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((arm_dir / "audio").glob("*.wav"))
    }


def main(run_a: Path, run_b: Path) -> None:
    arms = sorted(
        path.name
        for path in run_a.iterdir()
        if path.is_dir() and (run_b / path.name).is_dir()
    )
    for arm in arms:
        a_records, b_records = records(run_a / arm), records(run_b / arm)
        shared = sorted(set(a_records) & set(b_records))
        differing = [key for key in shared if a_records[key] != b_records[key]]
        print(f"{arm}: {len(shared)} samples, {len(differing)} records differ")
        for key in differing[:3]:
            keys = sorted(
                field
                for field in set(a_records[key]) | set(b_records[key])
                if a_records[key].get(field) != b_records[key].get(field)
            )
            print(f"  {key}: {keys}")
        a_wavs, b_wavs = wav_digests(run_a / arm), wav_digests(run_b / arm)
        if a_wavs or b_wavs:
            shared_wavs = sorted(set(a_wavs) & set(b_wavs))
            differing_wavs = [
                name for name in shared_wavs if a_wavs[name] != b_wavs[name]
            ]
            print(f"  wavs: {len(shared_wavs)} compared, {len(differing_wavs)} differ")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
