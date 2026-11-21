"""Compare prompt_mode_serving.py runs of main and the prompt mode branch, request by request.

Each request is compared with main's request in the same position (same value, stream and input),
since the first streamed requests after a boot can differ from later repeats on both trees. Main
drops non_streaming_mode, so on the branch the absent value and the task's default value must repeat
main's bytes (the default path is unchanged), and the other value must differ from them (the field
reaches the model).

usage: python3 prompt_mode_compare.py MAIN_DIR BRANCH_DIR
"""

import json
import sys
from pathlib import Path

DEFAULT_MODE = {"base": False, "customvoice": True, "voicedesign": True}


def load(directory: Path) -> dict[tuple, dict]:
    records = {}
    for line in (directory / "requests.jsonl").read_text().splitlines():
        record = json.loads(line)
        records[(record["model"], record["value"], record["stream"], record["input"])] = record
    return records


def main() -> None:
    main_records, branch_records = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    is_ok = True
    for model in DEFAULT_MODE:
        for stream in (True, False):
            keys = sorted({key[3] for key in main_records if key[0] == model and key[2] == stream})
            if not keys:
                continue
            else:
                pass
            row = []
            for value in (None, True, False):
                same = [
                    branch_records[(model, value, stream, index)]["sha256"]
                    == main_records[(model, value, stream, index)]["sha256"]
                    for index in keys
                ]
                expected = value is None or value == DEFAULT_MODE[model]
                is_row_ok = all(same) if expected else not any(same)
                is_ok = is_ok and is_row_ok
                row.append(f"{value}: same as main {sum(same)}/{len(same)} ({'ok' if is_row_ok else 'UNEXPECTED'})")
            main_same = all(
                main_records[(model, True, stream, index)]["sha256"] == main_records[(model, False, stream, index)]["sha256"]
                for index in keys
            )
            is_ok = is_ok and main_same
            chunks = [branch_records[(model, value, stream, index)]["chunks"] for value in (None, True, False) for index in keys]
            first_bytes = [
                round(branch_records[(model, value, stream, index)]["first_byte_s"] * 1000, 1)
                for value in (None, True, False)
                for index in keys
            ]
            print(
                f"{model} stream={stream}: {'; '.join(row)}; main ignores the field {main_same}; "
                f"branch chunks {min(chunks)}-{max(chunks)}, first byte {min(first_bytes)}-{max(first_bytes)} ms"
            )
    print("PASS" if is_ok else "FAIL")


if __name__ == "__main__":
    main()
