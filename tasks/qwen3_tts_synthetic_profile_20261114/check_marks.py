"""Resolve every function in the mark lists against the tree on PYTHONPATH; exit 1 if any is missing."""

import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK_LISTS = (HERE / "marks.json", HERE / "marks_capture.json")


def resolve(module_name: str, qualified_name: str) -> bool:
    target = importlib.import_module(module_name)
    for part in qualified_name.split("."):
        if not hasattr(target, part):
            return False
        else:
            target = getattr(target, part)
    return callable(target)


def main() -> int:
    missing = []
    total = 0
    for entry in [entry for path in MARK_LISTS for entry in json.loads(path.read_text())]:
        for function in entry["functions"]:
            name = function if isinstance(function, str) else function["function"]
            total += 1
            if not resolve(entry["module"], name):
                missing.append(f"{entry['module']}:{name}")
            else:
                pass
    print(json.dumps({"marks": total, "missing": missing}))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
