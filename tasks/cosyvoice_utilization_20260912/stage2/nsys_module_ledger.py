#!/usr/bin/env python3
"""Kernels and CUDA runtime calls per scenario and module, from an Nsight
SQLite export of s4_vocoder_profile.py.

A runtime call belongs to the innermost NVTX range open on its thread when it
starts; a kernel belongs to the runtime call that shares its correlation id (a
replayed graph's kernels share the id of their cudaGraphLaunch). Module paths
are folded over the block index, so transformer_blocks.7.attn.to_q counts under
transformer_blocks.N.attn.to_q. Runs in the container, standard library only.

  nsys export --type sqlite -o vocoder.sqlite vocoder.nsys-rep
  python nsys_module_ledger.py vocoder.sqlite --json ledger.json --md ledger.md
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict

SCENARIO = "scenario:"
INDEX = re.compile(r"\.\d+(?=\.|$)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--json", required=True)
    parser.add_argument("--md", required=True)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    columns = {row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    text = "COALESCE(text, '')" if "text" in columns else "''"
    text_id = "textId" if "textId" in columns else "NULL"
    ranges = defaultdict(list)
    for start, end, label, label_id, thread in db.execute(
        f"SELECT start, end, {text}, {text_id}, globalTid FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL ORDER BY start"
    ):
        ranges[thread].append((start, end, label or strings.get(label_id, "")))

    # correlation id -> (scenario, module, api name, host ns)
    owner: dict[int, tuple[str, str]] = {}
    calls = defaultdict(Counter)
    host_ns = defaultdict(int)
    span = {}
    runtime = db.execute(
        "SELECT start, end, nameId, correlationId, globalTid "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY globalTid, start"
    )
    cursor: dict[int, int] = defaultdict(int)
    stacks: dict[int, list] = defaultdict(list)
    for start, end, name_id, correlation, thread in runtime:
        stack, opened = stacks[thread], ranges[thread]
        while cursor[thread] < len(opened) and opened[cursor[thread]][0] <= start:
            stack.append(opened[cursor[thread]])
            cursor[thread] += 1
        while stack and stack[-1][1] < start:
            stack.pop()
        live = [entry for entry in stack if entry[1] >= start]
        scenario = next(
            (entry[2] for entry in live if entry[2].startswith(SCENARIO)), None
        )
        if scenario is None:
            continue
        span[scenario] = next(e[1] - e[0] for e in live if e[2] == scenario)
        module = next(
            (e[2] for e in reversed(live) if not e[2].startswith(SCENARIO)), "(glue)"
        )
        key = (scenario, INDEX.sub(".N", module))
        owner[correlation] = key
        calls[key][strings[name_id].split("_v")[0]] += 1
        host_ns[key] += end - start

    kernels = defaultdict(Counter)
    kernel_ns = defaultdict(Counter)
    gpu_ns = defaultdict(int)
    names = {
        row[1] for row in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")
    }
    name_column = "shortName" if "shortName" in names else "demangledName"
    for start, end, correlation, name_id in db.execute(
        f"SELECT start, end, correlationId, {name_column} FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        key = owner.get(correlation)
        if key is None:
            continue
        kernels[key][strings[name_id][:80]] += 1
        kernel_ns[key][strings[name_id][:80]] += end - start
        gpu_ns[key] += end - start

    ledger = defaultdict(dict)
    for key in calls:
        scenario, module = key
        ledger[scenario][module] = {
            "runtime_calls": sum(calls[key].values()),
            "runtime_api_ms": host_ns[key] / 1e6,
            "kernels": sum(kernels[key].values()),
            "gpu_ms": gpu_ns[key] / 1e6,
            "apis": dict(calls[key].most_common(6)),
            "top_kernels": dict(kernels[key].most_common(4)),
            "kernel_ms_by_name": {
                name: (ns / 1e6, kernels[key][name])
                for name, ns in kernel_ns[key].most_common(8)
            },
        }
    with open(args.json, "w") as out:
        json.dump(
            {"span_ms": {k: v / 1e6 for k, v in span.items()}, "ledger": ledger},
            out,
            indent=1,
        )

    with open(args.md, "w") as out:
        for scenario in sorted(ledger):
            modules = ledger[scenario]
            total_calls = sum(m["runtime_calls"] for m in modules.values())
            total_gpu = sum(m["gpu_ms"] for m in modules.values())
            out.write(
                f"\n## {scenario}: {span[scenario] / 1e6:.1f} ms wall, {total_calls} "
                f"runtime calls, {sum(m['kernels'] for m in modules.values())} kernels, "
                f"{total_gpu:.1f} ms of kernels\n\n"
                "| module | runtime calls | kernels | kernel ms | top kernel |\n|---|---|---|---|---|\n"
            )
            ranked = sorted(modules.items(), key=lambda kv: -kv[1]["runtime_calls"])
            for module, row in ranked[:40]:
                top = next(iter(row["top_kernels"]), "")
                out.write(
                    f"| {module} | {row['runtime_calls']} | {row['kernels']} | "
                    f"{row['gpu_ms']:.1f} | {top[:60]} |\n"
                )


if __name__ == "__main__":
    main()
