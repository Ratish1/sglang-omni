"""Reduce ipc_payload_probe records to the step that owns each payload hop's time.

For every serialize and deserialize call: wall and thread CPU time, each step's share, the
residual not inside any timed step, the header size, and for serialize the header's largest
fields. Then the medians per kind and step.

usage: python ipc_probe_report.py <dir of ipc_*.jsonl>
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys


def largest_fields(node: dict, path: str, out: list[tuple[int, str, str]]) -> None:
    pickled = node.get("pickled")
    if isinstance(pickled, int):
        out.append((pickled, path, node.get("type", "?")))
    else:
        pass
    for key, child in (node.get("fields") or {}).items():
        largest_fields(child, f"{path}.{key}", out)
    for index, child in enumerate(node.get("items") or []):
        largest_fields(child, f"{path}[{index}]", out)


def main() -> None:
    records = []
    for path in sorted(glob.glob(os.path.join(sys.argv[1], "ipc_*.jsonl"))):
        with open(path) as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(key=lambda record: record["time"])
    by_step: dict[tuple[str, str], list[float]] = {}
    for record in records:
        steps = record["steps"]
        timed = sum(step["wall_ms"] for step in steps)
        print(
            f"{record['kind']:11s} rid={record.get('request_id', '?')} pid={record['pid']} "
            f"thread={record['thread']} wall={record['wall_ms']:.2f} cpu={record['cpu_ms']:.2f} "
            f"header={record.get('header_bytes')} tensors={record.get('tensor_entry_bytes')} "
            f"residual={record['wall_ms'] - timed:.2f}"
        )
        merged: dict[str, list[float]] = {}
        for step in steps:
            merged.setdefault(step["step"], [0.0, 0.0, 0])
            merged[step["step"]][0] += step["wall_ms"]
            merged[step["step"]][1] += step["cpu_ms"]
            merged[step["step"]][2] += 1
        for name, (wall, cpu, count) in merged.items():
            print(f"    {name:22s} n={count:2d} wall={wall:9.2f} cpu={cpu:9.2f}")
            by_step.setdefault((record["kind"], name), []).append(wall)
        by_step.setdefault((record["kind"], "total"), []).append(record["wall_ms"])
        by_step.setdefault((record["kind"], "residual"), []).append(
            record["wall_ms"] - timed
        )
        if record["kind"] == "serialize" and "data_without_tensors" in record:
            fields: list[tuple[int, str, str]] = []
            largest_fields(record["request"], "request", fields)
            largest_fields(record["data_without_tensors"], "data", fields)
            fields.sort(reverse=True)
            print(
                "    largest pickled: "
                + ", ".join(
                    f"{path}={size} ({kind})" for size, path, kind in fields[:6]
                )
            )
        else:
            pass
    print("\nmedians (ms) by kind and step over calls")
    for (kind, name), values in sorted(by_step.items()):
        print(
            f"  {kind:11s} {name:22s} calls={len(values):3d} "
            f"median={statistics.median(values):9.2f} max={max(values):9.2f}"
        )


if __name__ == "__main__":
    main()
