"""First audio of a Qwen3-TTS events pair split into consecutive request states, base against head.

Each request's first stamp per (stage, event) is read from <root>/order<k>/<side>/<cell>/events/*.jsonl.
The chain is every (stage, event) both arms record for nearly every request, ordered by base's median offset
from admission and cut at the coordinator's first audio receipt, so the states sum to the server-side first
audio. The warmup and the opening burst (the first admissions of the cell) are dropped. Each order is printed
alone; a state counts only when both orders move it the same way.

usage: python3 first_audio_states.py --root /data/.../events_c64 --cell stream_c64
"""

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

ADMISSION = ("coordinator", "request_admission")
COVERAGE = 0.98
TAIL = 0.05


def first_stamps(directory: Path) -> dict[str, dict[tuple[str, str], int]]:
    first: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for path in glob.glob(str(directory / "events" / "*.jsonl")):
        with open(path) as source:
            for line in source:
                record = json.loads(line)
                if "event_name" not in record or "request_id" not in record:
                    continue
                else:
                    pass
                key = (record["stage"], record["event_name"])
                seen = first[record["request_id"]]
                if key not in seen or record["timestamp_ns"] < seen[key]:
                    seen[key] = record["timestamp_ns"]
                else:
                    pass
    return first


def steady(
    stamps: dict[str, dict[tuple[str, str], int]], skip: int
) -> dict[str, dict[tuple[str, str], int]]:
    admitted = sorted(
        (events[ADMISSION], request_id)
        for request_id, events in stamps.items()
        if ADMISSION in events
    )
    kept = {request_id for _, request_id in admitted[skip:]}
    return {request_id: stamps[request_id] for request_id in kept}


def offsets(
    stamps: dict[str, dict[tuple[str, str], int]]
) -> dict[tuple[str, str], list[float]]:
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for events in stamps.values():
        start = events[ADMISSION]
        for key, stamp in events.items():
            by_key[key].append((stamp - start) / 1e6)
    return by_key


def chain_for(
    base: dict[str, dict[tuple[str, str], int]],
    head: dict[str, dict[tuple[str, str], int]],
    end_event: str,
) -> list[tuple[str, str]]:
    base_offsets = offsets(base)
    head_offsets = offsets(head)
    shared = [
        key
        for key in base_offsets
        if len(base_offsets[key]) >= COVERAGE * len(base)
        and len(head_offsets.get(key, [])) >= COVERAGE * len(head)
    ]
    ordered = sorted(shared, key=lambda key: statistics.median(base_offsets[key]))
    end = next(index for index, key in enumerate(ordered) if key[1] == end_event)
    return ordered[: end + 1]


def state_rows(
    stamps: dict[str, dict[tuple[str, str], int]], chain: list[tuple[str, str]]
) -> list[list[float]]:
    rows = []
    for events in stamps.values():
        values = [events.get(key) for key in chain]
        if any(value is None for value in values):
            continue
        else:
            pass
        states = [(after - before) / 1e6 for before, after in zip(values, values[1:])]
        rows.append([(values[-1] - values[0]) / 1e6, *states])
    return rows


def print_unshared(name: str, mine: dict, other: dict, count: int) -> None:
    for key, values in sorted(mine.items()):
        if key not in other and len(values) >= COVERAGE * count:
            print(
                f"    only {name}: {key[0]}/{key[1]} median {statistics.median(values):.1f} ms"
            )
        else:
            pass


def client_steady_ttfp(directory: Path, skip: int) -> float:
    records = json.loads((directory / "bench/speed_results.json").read_text())[
        "per_request"
    ]
    return 1000 * statistics.fmean(record["audio_ttfp_s"] for record in records[skip:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--cell", default="stream_c64")
    parser.add_argument("--skip", type=int, default=65)
    parser.add_argument("--end-event", default="coordinator_stream_received")
    arguments = parser.parse_args()
    root = Path(arguments.root)
    for order_directory in sorted(root.glob("order*")):
        cells = {
            side: order_directory / side / arguments.cell for side in ("base", "head")
        }
        stamps = {
            side: steady(first_stamps(directory), arguments.skip)
            for side, directory in cells.items()
        }
        chain = chain_for(stamps["base"], stamps["head"], arguments.end_event)
        rows = {side: state_rows(stamps[side], chain) for side in cells}
        print(
            f"{order_directory.name}: base {len(rows['base'])} requests, head {len(rows['head'])}; "
            f"client first audio past the opening burst: base "
            f"{client_steady_ttfp(cells['base'], arguments.skip - 1):.2f} ms, head "
            f"{client_steady_ttfp(cells['head'], arguments.skip - 1):.2f} ms"
        )
        tails = {}
        for side in cells:
            ranked = sorted(rows[side], key=lambda row: row[0])
            tails[side] = ranked[int(len(ranked) * (1 - TAIL)) :]
        print(
            f"  {'state (ms)':58s}{'base p50':>9s}{'head p50':>9s}{'base mean':>10s}{'head mean':>10s}"
            f"{'change':>8s}{'base tail':>10s}{'head tail':>10s}{'change':>8s}"
        )
        labels = ["server first audio"] + [
            f"{before[0]}/{before[1]} -> {after[0]}/{after[1]}"
            for before, after in zip(chain, chain[1:])
        ]
        for index, label in enumerate(labels):
            base_values = [row[index] for row in rows["base"]]
            head_values = [row[index] for row in rows["head"]]
            base_mean = statistics.fmean(base_values)
            head_mean = statistics.fmean(head_values)
            base_tail = statistics.fmean(row[index] for row in tails["base"])
            head_tail = statistics.fmean(row[index] for row in tails["head"])
            print(
                f"  {label[:58]:58s}{statistics.median(base_values):9.2f}{statistics.median(head_values):9.2f}"
                f"{base_mean:10.2f}{head_mean:10.2f}{head_mean - base_mean:+8.2f}"
                f"{base_tail:10.2f}{head_tail:10.2f}{head_tail - base_tail:+8.2f}"
            )
        base_offsets = offsets(stamps["base"])
        head_offsets = offsets(stamps["head"])
        print_unshared("base", base_offsets, head_offsets, len(stamps["base"]))
        print_unshared("head", head_offsets, base_offsets, len(stamps["head"]))


if __name__ == "__main__":
    main()
