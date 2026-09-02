# SPDX-License-Identifier: Apache-2.0
"""Side by side readout of a full_ab.sh run.

Usage:
    python full_ab_compare.py OUT [--md FILE]

Every directory under OUT that holds both an A and a B subdirectory is one
stage. For each stage the script prints the pytest exit code of both arms,
then every *_results.json found under A matched to the same relative path
under B, with the numeric leaves of the summary, speed, speed_metrics and
wer.summary blocks side by side and the difference B minus A. After the
metrics it prints the scheduler lines pulled from the logs of both arms:
the KV pool sizes at boot and the number of retraction lines.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BLOCKS = ("summary", "speed", "speed_metrics", ("wer", "summary"))
POOL_RE = re.compile(r"KV Cache is allocated.*?#tokens:\s*(\d+)")
RETRACT_RE = re.compile(r"Retract requests|Testing retraction")


def _block(doc: dict, key) -> dict:
    if isinstance(key, tuple):
        cur = doc
        for k in key:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        return cur if isinstance(cur, dict) else {}
    value = doc.get(key, {})
    return value if isinstance(value, dict) else {}


def _leaves(doc: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in BLOCKS:
        label = ".".join(key) if isinstance(key, tuple) else key
        for name, value in _block(doc, key).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                out[f"{label}.{name}"] = float(value)
    return out


def _results(arm_dir: Path) -> dict[str, Path]:
    return {
        str(p.relative_to(arm_dir)): p for p in sorted(arm_dir.rglob("*_results.json"))
    }


def _exit_code(arm_dir: Path) -> str:
    p = arm_dir / "exit_code"
    return p.read_text().strip() if p.exists() else "missing"


def _scheduler_lines(arm_dir: Path) -> tuple[list[int], int]:
    pools: list[int] = []
    retracts = 0
    for log in arm_dir.rglob("*.log"):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        pools.extend(int(m) for m in POOL_RE.findall(text))
        retracts += len(RETRACT_RE.findall(text))
    return pools, retracts


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 100 or value == int(value):
        return f"{value:.0f}" if value == int(value) else f"{value:.1f}"
    return f"{value:.4f}"


def compare_stage(stage: Path, out: list[str]) -> None:
    a_dir, b_dir = stage / "A", stage / "B"
    order = (
        (stage / "order.txt").read_text().strip()
        if (stage / "order.txt").exists()
        else "?"
    )
    out.append(f"\n## {stage.name}  (order {order})")
    out.append(f"pytest exit: A {_exit_code(a_dir)}, B {_exit_code(b_dir)}")
    a_res, b_res = _results(a_dir), _results(b_dir)
    for rel in sorted(set(a_res) | set(b_res)):
        a_doc = json.loads(a_res[rel].read_text()) if rel in a_res else {}
        b_doc = json.loads(b_res[rel].read_text()) if rel in b_res else {}
        a_leaves, b_leaves = _leaves(a_doc), _leaves(b_doc)
        keys = sorted(set(a_leaves) | set(b_leaves))
        if not keys:
            continue
        out.append(f"\n{rel}")
        out.append("| metric | A | B | B - A | B / A |")
        out.append("|---|---|---|---|---|")
        for key in keys:
            a, b = a_leaves.get(key), b_leaves.get(key)
            delta = b - a if a is not None and b is not None else None
            ratio = b / a if a not in (None, 0) and b is not None else None
            out.append(
                f"| {key} | {_fmt(a)} | {_fmt(b)} | {_fmt(delta)} | "
                f"{'-' if ratio is None else f'{ratio:.3f}'} |"
            )
    a_pools, a_retracts = _scheduler_lines(a_dir)
    b_pools, b_retracts = _scheduler_lines(b_dir)
    out.append(
        f"\nKV pools at boot: A {a_pools or 'not found'}, B {b_pools or 'not found'}. "
        f"Retraction lines: A {a_retracts}, B {b_retracts}."
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--md", help="also write the readout to this file")
    args = p.parse_args(argv)
    root = Path(args.out)
    stages = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and (d / "A").is_dir() and (d / "B").is_dir()
    )
    if not stages:
        print(f"no stage directory with both A and B under {root}", file=sys.stderr)
        return 1
    lines = [f"# full A/B readout of {root}"]
    for stage in stages:
        compare_stage(stage, lines)
    text = "\n".join(lines) + "\n"
    print(text)
    if args.md:
        Path(args.md).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
