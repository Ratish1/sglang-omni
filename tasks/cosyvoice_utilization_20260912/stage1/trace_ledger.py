"""Profile one call and attribute every GPU activity to the Python range that
launched it.

A call is timed without instrumentation first (median of synchronized repeats),
then run once under torch.profiler with a record_function range around the
call, around every forward of the given modules and around the given functions.
The exported Chrome trace is read back with three rules:

  a device event (any category whose name starts with gpu_ other than
  gpu_user_annotation, or kernel) belongs to the host runtime event with the
  same args.correlation;
  a runtime event belongs to the innermost range on its thread whose
  interval contains it;
  busy time is the union of device intervals, never their sum.

Nothing about the trace schema is assumed beyond ts and dur in microseconds,
tid, cat, name and args.correlation; the category inventory is reported so a
schema change is visible.
"""

from __future__ import annotations

import collections
import contextlib
import gzip
import json
import os
import re
import statistics
import time
from collections.abc import Callable, Iterable

import torch
from torch.profiler import ProfilerActivity, profile, record_function

SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")
BLOCKING_COPY_NAMES = ("cudaMemcpy",)
LAUNCH = re.compile(r"LaunchKernel|GraphLaunch")
POINTWISE = re.compile(
    r"elementwise|vectorized|reduce_kernel|CatArrayBatched|index_elementwise|copy_",
    re.IGNORECASE,
)
INSTANCE_INDEX = re.compile(r"\.\d+(?=\.|$)")


@contextlib.contextmanager
def module_ranges(roots: dict[str, torch.nn.Module]):
    labels: dict[int, str] = {}
    for root_name, root in roots.items():
        for name, module in root.named_modules():
            qualified = f"{root_name}.{name}" if name else root_name
            labels[id(module)] = "mod:" + INSTANCE_INDEX.sub(".*", qualified)
    open_ranges: list[tuple[int, object]] = []

    def enter(module, args):
        label = labels.get(id(module))
        if label is not None:
            context = record_function(label)
            context.__enter__()
            open_ranges.append((id(module), context))

    def leave(module, args, output):
        if id(module) not in labels:
            return
        while open_ranges:
            module_id, context = open_ranges.pop()
            context.__exit__(None, None, None)
            if module_id == id(module):
                break

    pre = torch.nn.modules.module.register_module_forward_pre_hook(enter)
    post = torch.nn.modules.module.register_module_forward_hook(leave)
    try:
        yield
    finally:
        pre.remove()
        post.remove()
        while open_ranges:
            open_ranges.pop()[1].__exit__(None, None, None)


@contextlib.contextmanager
def function_ranges(targets: Iterable[tuple[object, str]]):
    restore = []
    for owner, attribute in targets:
        original = getattr(owner, attribute)
        owned = attribute in vars(owner)
        owner_name = getattr(owner, "__name__", type(owner).__name__)
        label = f"fn:{owner_name.rsplit('.', 1)[-1]}.{attribute}"

        def wrap(function=original, label=label):
            def wrapped(*args, **kwargs):
                with record_function(label):
                    return function(*args, **kwargs)

            return wrapped

        setattr(owner, attribute, wrap())
        restore.append((owner, attribute, original, owned))
    try:
        yield
    finally:
        for owner, attribute, original, owned in reversed(restore):
            if owned:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    current_start = current_end = None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total / 1e3


def parse_trace(path: str, call_label: str, top: int = 15) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        data = json.load(handle)
    events = data["traceEvents"] if isinstance(data, dict) else data
    categories = collections.Counter(str(e.get("cat")) for e in events)
    device_categories = {
        c
        for c in categories
        if c == "kernel" or (c.startswith("gpu_") and c != "gpu_user_annotation")
    }
    calls = [
        e
        for e in events
        if e.get("name") == call_label and e.get("cat") not in device_categories
    ]
    if not calls:
        raise RuntimeError(f"no range named {call_label!r} in {path}")
    call = max(calls, key=lambda e: e.get("dur", 0))
    tid, start, end = call.get("tid"), call["ts"], call["ts"] + call.get("dur", 0)

    def inside(event):
        return (
            event.get("tid") == tid
            and start <= event.get("ts", -1)
            and event["ts"] + event.get("dur", 0) <= end
        )

    ranges = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and inside(e)
        and str(e.get("name", "")).startswith(("mod:", "fn:"))
    ]
    runtime = [e for e in events if e.get("cat") == "cuda_runtime" and inside(e)]
    if runtime and not any("correlation" in e.get("args", {}) for e in runtime):
        raise RuntimeError("cuda_runtime events carry no args.correlation")

    timeline = [
        (e["ts"], 0, -e.get("dur", 0), i, "range") for i, e in enumerate(ranges)
    ]
    timeline += [(e["ts"], 1, 0, i, "runtime") for i, e in enumerate(runtime)]
    timeline.sort()
    owner_of: dict[int, tuple[int, ...]] = {}
    stack: list[int] = []
    for ts, _, _, index, kind in timeline:
        while stack and ranges[stack[-1]]["ts"] + ranges[stack[-1]].get("dur", 0) <= ts:
            stack.pop()
        if kind == "range":
            stack.append(index)
        else:
            owner_of[index] = tuple(stack)

    by_correlation = {
        e["args"]["correlation"]: i
        for i, e in enumerate(runtime)
        if "correlation" in e.get("args", {})
    }
    device = []
    unattributed = 0
    for event in events:
        if event.get("cat") not in device_categories:
            continue
        correlation = event.get("args", {}).get("correlation")
        index = by_correlation.get(correlation)
        if index is None:
            unattributed += 1
            continue
        device.append((event, index))

    def label_of(path_indices: tuple[int, ...]) -> str:
        return ranges[path_indices[-1]]["name"] if path_indices else "(outside ranges)"

    busy_ms = _union_ms([(e["ts"], e["ts"] + e.get("dur", 0)) for e, _ in device])
    window_ms = (
        (
            max(e["ts"] + e.get("dur", 0) for e, _ in device)
            - min(e["ts"] for e, _ in device)
        )
        / 1e3
        if device
        else 0.0
    )

    self_stats = collections.defaultdict(lambda: collections.Counter())
    inclusive_device = collections.Counter()
    kernel_stats = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
    sequences = collections.defaultdict(list)
    for event, index in device:
        path_indices = owner_of.get(index, ())
        label = label_of(path_indices)
        duration_ms = event.get("dur", 0) / 1e3
        stats = self_stats[label]
        stats["device_events"] += 1
        stats["device_us"] += event.get("dur", 0)
        if event.get("cat") == "kernel" and POINTWISE.search(str(event.get("name"))):
            stats["pointwise_kernels"] += 1
            stats["pointwise_us"] += event.get("dur", 0)
        for ancestor in set(ranges[i]["name"] for i in path_indices):
            inclusive_device[ancestor] += event.get("dur", 0)
        name = str(event.get("name"))[:120]
        kernel = kernel_stats[name]
        kernel[0] += 1
        kernel[1] += duration_ms
        kernel[2][label] += 1
        instance = path_indices[-1] if path_indices else -1
        sequences[(label, instance)].append((event["ts"], name))

    syncs = collections.Counter()
    sync_ms = collections.Counter()
    copies = collections.Counter()
    launches = collections.Counter()
    graph_launches = 0
    for index, event in enumerate(runtime):
        label = label_of(owner_of.get(index, ()))
        name = event.get("name", "")
        if name in SYNC_NAMES:
            syncs[label] += 1
            sync_ms[label] += event.get("dur", 0) / 1e3
        elif name in BLOCKING_COPY_NAMES:
            copies[label] += 1
            sync_ms[label] += event.get("dur", 0) / 1e3
        if LAUNCH.search(name):
            launches[label] += 1
            if "GraphLaunch" in name:
                graph_launches += 1

    repeated = collections.defaultdict(lambda: [0, 0])
    for (label, _), kernels in sequences.items():
        signature = tuple(name for _, name in sorted(kernels))
        entry = repeated[(label, signature)]
        entry[0] += 1
        entry[1] = len(signature)
    repeated_rows = sorted(
        (
            {
                "range": label,
                "instances": count,
                "kernels_per_instance": length,
                "launches": count * length,
            }
            for (label, _), (count, length) in repeated.items()
            if count > 1
        ),
        key=lambda row: -row["launches"],
    )[:top]

    labels = set(self_stats) | set(launches)
    range_rows = sorted(
        (
            {
                "range": label,
                "launches": launches[label],
                "device_events": self_stats[label]["device_events"],
                "self_device_ms": self_stats[label]["device_us"] / 1e3,
                "inclusive_device_ms": inclusive_device[label] / 1e3,
                "pointwise_kernels": self_stats[label]["pointwise_kernels"],
                "pointwise_ms": self_stats[label]["pointwise_us"] / 1e3,
                "syncs": syncs[label],
                "blocking_copies": copies[label],
                "blocked_ms": sync_ms[label],
            }
            for label in labels
        ),
        key=lambda row: -row["self_device_ms"],
    )
    kernel_rows = sorted(
        (
            {
                "kernel": name,
                "count": count,
                "device_ms": ms,
                "owners": dict(owners.most_common(3)),
            }
            for name, (count, ms, owners) in kernel_stats.items()
        ),
        key=lambda row: -row["device_ms"],
    )[:top]
    host_ms = call.get("dur", 0) / 1e3
    blocked_ms = sum(sync_ms.values())
    return {
        "trace_categories": dict(categories),
        "host_ms_profiled": host_ms,
        "device_busy_ms": busy_ms,
        "device_window_ms": window_ms,
        "device_busy_share_of_window": busy_ms / window_ms if window_ms else None,
        "device_busy_share_of_host": busy_ms / host_ms if host_ms else None,
        "host_blocked_ms": blocked_ms,
        "launches": sum(launches.values()),
        "graph_launches": graph_launches,
        "device_events": len(device),
        "device_events_without_launch_in_call": unattributed,
        "syncs": sum(syncs.values()),
        "blocking_copies": sum(copies.values()),
        "ranges": range_rows,
        "top_kernels": kernel_rows,
        "repeated_sequences": repeated_rows,
    }


def measure(
    call: Callable[[], object],
    *,
    label: str,
    out_dir: str,
    roots: dict[str, torch.nn.Module] | None = None,
    functions: Iterable[tuple[object, str]] = (),
    warmup: int = 2,
    repeats: int = 5,
    keep_trace: bool = True,
    prepare: Callable[[], None] | None = None,
) -> dict:
    # prepare runs before every run, outside the timed and profiled window: a
    # call that consumes state (a prefill admits and finishes its requests)
    # gets identical state each time.
    def ready():
        if prepare is not None:
            prepare()
        torch.cuda.synchronize()

    for _ in range(warmup):
        ready()
        call()
        torch.cuda.synchronize()
    walls = []
    for _ in range(repeats):
        ready()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - started) * 1e3)
    call_label = f"call:{label}"
    ready()
    with module_ranges(roots or {}), function_ranges(list(functions)):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as profiler:
            with record_function(call_label):
                call()
            torch.cuda.synchronize()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}.trace.json")
    profiler.export_chrome_trace(path)
    ledger = parse_trace(path, call_label)
    if keep_trace:
        with open(path, "rb") as source, gzip.open(f"{path}.gz", "wb") as target:
            target.write(source.read())
    os.remove(path)
    ledger.update(
        {
            "label": label,
            "wall_ms_median": statistics.median(walls),
            "wall_ms_min": min(walls),
            "wall_ms_max": max(walls),
            "repeats": repeats,
        }
    )
    ledger["device_busy_share_of_wall"] = (
        ledger["device_busy_ms"] / ledger["wall_ms_median"]
    )
    return ledger


def render_markdown(ledgers: list[dict], top: int = 10) -> str:
    lines = [
        "| point | wall ms | busy ms | busy / wall | launches | graph launches | syncs + blocking copies | blocked ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ledger in ledgers:
        lines.append(
            f"| {ledger['label']} | {ledger['wall_ms_median']:.1f} | {ledger['device_busy_ms']:.1f} "
            f"| {ledger['device_busy_share_of_wall']:.2f} | {ledger['launches']} | {ledger['graph_launches']} "
            f"| {ledger['syncs'] + ledger['blocking_copies']} | {ledger['host_blocked_ms']:.1f} |"
        )
    for ledger in ledgers:
        lines += [
            "",
            f"### {ledger['label']}",
            "",
            "| range | launches | self device ms | inclusive device ms | pointwise kernels | syncs | blocked ms |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in ledger["ranges"][:top]:
            lines.append(
                f"| {row['range']} | {row['launches']} | {row['self_device_ms']:.1f} | {row['inclusive_device_ms']:.1f} "
                f"| {row['pointwise_kernels']} | {row['syncs'] + row['blocking_copies']} | {row['blocked_ms']:.1f} |"
            )
        lines += ["", "| kernel | count | device ms | owners |", "|---|---|---|---|"]
        for row in ledger["top_kernels"][:top]:
            owners = ", ".join(f"{k} {v}" for k, v in row["owners"].items())
            lines.append(
                f"| `{row['kernel']}` | {row['count']} | {row['device_ms']:.1f} | {owners} |"
            )
        lines += [
            "",
            "| repeated range | instances | kernels each | launches |",
            "|---|---|---|---|",
        ]
        for row in ledger["repeated_sequences"][:top]:
            lines.append(
                f"| {row['range']} | {row['instances']} | {row['kernels_per_instance']} | {row['launches']} |"
            )
    return "\n".join(lines) + "\n"
