"""Where a pair's pooled inter chunk mean moves: per gap position (chunk k to k+1), the request mix (how many gaps each
position contributes), chunk arrival from the request's start paired by request id, and each stream's delivery pace
after its first chunk (audio seconds per wall second). Reads a serve pair's speed_results.json files.
usage: python3 inter_chunk_mechanism_c20261018.py <pair dir> <base head prefix> <arm head prefix> <cell> [<cell> ...]"""

import json
import statistics
import sys
from pathlib import Path


def load_arms(pair_dir: Path, base_head: str, arm_head: str, cell: str) -> dict[str, dict[str, dict]]:
    arms = {}
    for order_dir in sorted(p for p in pair_dir.iterdir() if p.is_dir()):
        for identity in sorted(order_dir.glob("identity_gpu*.txt")):
            head = identity.read_text().split()[0]
            card = identity.stem.removeprefix("identity_")
            name = "base" if head.startswith(base_head) else "arm" if head.startswith(arm_head) else None
            if name is None:
                continue
            records = json.loads((order_dir / card / cell / "speed_results.json").read_text())["per_request"]
            arms[f"{order_dir.name}/{name}"] = {r["id"]: r for r in records if r["is_success"]}
    return arms


def gaps_by_position(records: dict[str, dict]) -> dict[int, list[float]]:
    positions: dict[int, list[float]] = {}
    for record in records.values():
        for index, gap in enumerate(record["inter_chunk_s"] or []):
            positions.setdefault(index + 1, []).append(gap)
    return positions


def arrivals(record: dict) -> list[float]:
    times = [record["audio_ttfp_s"]]
    for gap in record["inter_chunk_s"] or []:
        times.append(times[-1] + gap)
    return times


def pace_after_first(record: dict) -> float | None:
    gaps = record["inter_chunk_s"] or []
    if not gaps:
        return None
    return sum(record["chunk_audio_duration_s"][1:]) / sum(gaps)


def report(arms: dict[str, dict[str, dict]], order: str) -> None:
    base, arm = arms[f"{order}/base"], arms[f"{order}/arm"]
    base_positions, arm_positions = gaps_by_position(base), gaps_by_position(arm)
    base_all = [g for gaps in base_positions.values() for g in gaps]
    arm_all = [g for gaps in arm_positions.values() for g in gaps]
    print(f"order {order}: pooled mean {statistics.mean(base_all):.4f} -> {statistics.mean(arm_all):.4f} s "
          f"({100 * (statistics.mean(arm_all) / statistics.mean(base_all) - 1):+.1f} %), gaps {len(base_all)} -> "
          f"{len(arm_all)}")
    mix_only = sum(len(arm_positions.get(k, [])) * statistics.mean(base_positions[k]) for k in base_positions
                   if k in arm_positions) / len(arm_all)
    print(f"  the arm's mix at the base's per position means: {mix_only:.4f} s "
          f"({100 * (mix_only / statistics.mean(base_all) - 1):+.1f} %)")
    for k in sorted(set(base_positions) | set(arm_positions)):
        b, a = base_positions.get(k, []), arm_positions.get(k, [])
        if len(b) < 5 or len(a) < 5:
            continue
        print(f"  gap {k}->{k + 1}: n {len(b)} -> {len(a)}, mean {statistics.mean(b):.4f} -> {statistics.mean(a):.4f}, "
              f"median {statistics.median(b):.4f} -> {statistics.median(a):.4f} s")
    shared = sorted(set(base) & set(arm))
    same_chunks = [i for i in shared if base[i]["audio_chunk_count"] == arm[i]["audio_chunk_count"]]
    print(f"  paired: {len(shared)} requests, {len(same_chunks)} with the same chunk count")
    for k in range(4):
        deltas = [arrivals(arm[i])[k] - arrivals(base[i])[k] for i in same_chunks if base[i]["audio_chunk_count"] > k]
        if len(deltas) >= 5:
            print(f"  chunk {k + 1} arrival, arm minus base: median {1000 * statistics.median(deltas):+.1f} ms, "
                  f"mean {1000 * statistics.mean(deltas):+.1f} ms (n {len(deltas)})")
    for name, records in (("base", base), ("arm", arm)):
        paces = [p for p in (pace_after_first(records[i]) for i in same_chunks) if p is not None]
        print(f"  {name}: audio s per wall s after the first chunk, median {statistics.median(paces):.2f}, "
              f"mean {statistics.mean(paces):.2f} (n {len(paces)})")


pair_dir, base_head, arm_head = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for cell in sys.argv[4:]:
    print(f"== {pair_dir.name} {cell}")
    arms = load_arms(pair_dir, base_head, arm_head, cell)
    for order in sorted({key.split("/")[0] for key in arms}):
        report(arms, order)
