"""Whether two arms served the same audio and the same thinker token count for every prompt of a seeded pass.

Run: python3 byte_identity_read.py <run_dir> <arm_a> <arm_b> <pass> [<pass> ...]
For every round that ran both arms, reads each arm's <pass>/speed_results.json per_request (completion_tokens,
wav_path) and compares each prompt's wav file byte for byte. Prints, per round and pass, the prompts compared, how
many match in bytes and in completion tokens, and every prompt that does not.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def requests(pass_dir: Path) -> dict[str, dict]:
    rows = json.loads((pass_dir / "speed_results.json").read_text())["per_request"]
    return {Path(row["wav_path"]).stem: row for row in rows if row["is_success"]}


def main() -> None:
    run_dir, arm_a, arm_b, pass_names = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
    rounds: dict[int, dict[str, Path]] = defaultdict(dict)
    for directory in sorted(run_dir.iterdir()):
        match = re.fullmatch(r"r(\d+)_card(\d+)_(\w+)", directory.name)
        if match:
            rounds[int(match[1])][match[3]] = directory
        else:
            pass
    for round_index, directories in sorted(rounds.items()):
        if arm_a not in directories or arm_b not in directories:
            continue
        else:
            pass
        for pass_name in pass_names:
            rows_a = requests(directories[arm_a] / pass_name)
            rows_b = requests(directories[arm_b] / pass_name)
            prompts = sorted(set(rows_a) & set(rows_b))
            differing_audio = [
                prompt
                for prompt in prompts
                if Path(rows_a[prompt]["wav_path"]).read_bytes() != Path(rows_b[prompt]["wav_path"]).read_bytes()
            ]
            differing_tokens = [
                prompt for prompt in prompts if rows_a[prompt]["completion_tokens"] != rows_b[prompt]["completion_tokens"]
            ]
            print(
                f"round {round_index} ({directories[arm_a].name}, {directories[arm_b].name}) {pass_name}: "
                f"{len(prompts)} prompts in both ({len(rows_a)} / {len(rows_b)} succeeded); audio identical "
                f"{len(prompts) - len(differing_audio)}, thinker tokens equal {len(prompts) - len(differing_tokens)}"
            )
            for prompt in differing_audio[:20]:
                print(
                    f"  audio differs {prompt}: tokens {rows_a[prompt]['completion_tokens']} / "
                    f"{rows_b[prompt]['completion_tokens']}, audio {rows_a[prompt]['audio_duration_s']} / "
                    f"{rows_b[prompt]['audio_duration_s']} s"
                )


if __name__ == "__main__":
    main()
