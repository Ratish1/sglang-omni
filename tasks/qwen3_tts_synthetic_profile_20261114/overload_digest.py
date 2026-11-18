"""One table per rate from a bar_overload_pair.py run: the client's summary next to the API process's
loop load (its main thread's CPU share and the /health round trip), the client's own CPU and
dispatch lag, and the stage threads that hold the most CPU.

usage: python3 overload_digest.py OUT
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TOP_STAGE_THREADS = 4


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    else:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def fmt(value: float | None, scale: float = 1.0, digits: int = 2) -> str:
    return "-" if value is None else f"{value * scale:.{digits}f}"


def process_view(path: Path) -> dict:
    """Per sample: API main thread and process CPU, client CPU, and the /health round trip."""
    api_main, api_total, client_total, client_top, health_s, tracked = [], [], [], [], [], []
    stage_cpu: dict[str, list[float]] = defaultdict(list)
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for sample in samples[1:]:
        threads = sample["threads"]
        api = [t for t in threads if t["role"] == "api"]
        api_main.append(sum(t["cpu"] for t in api if t["tid"] == t["pid"]))
        api_total.append(sum(t["cpu"] for t in api))
        client = [t["cpu"] for t in threads if t["role"] == "client"]
        client_total.append(sum(client))
        client_top.append(max(client, default=0.0))
        for thread in threads:
            if thread["role"] == "stage":
                stage_cpu[thread["name"]].append(thread["cpu"])
            else:
                pass
        if sample["health_s"] is not None:
            health_s.append(sample["health_s"])
        else:
            pass
        tracked.append(sample["health"].get("total_requests", 0))
    count = max(1, len(samples) - 1)
    stage_mean = sorted(((sum(v) / count, name) for name, v in stage_cpu.items()), reverse=True)
    return {
        "api_main_p50": quantile(api_main, 0.5),
        "api_main_p90": quantile(api_main, 0.9),
        "api_total_p50": quantile(api_total, 0.5),
        "client_total_p50": quantile(client_total, 0.5),
        "client_top_p90": quantile(client_top, 0.9),
        "health_p50": quantile(health_s, 0.5),
        "health_p90": quantile(health_s, 0.9),
        "health_max": max(health_s, default=None),
        "tracked_max": max(tracked, default=0),
        "stage_top": ", ".join(f"{name} {cpu:.2f}" for cpu, name in stage_mean[:TOP_STAGE_THREADS]),
    }


def main() -> None:
    out = Path(sys.argv[1])
    for seed_dir in sorted(out.glob("seed*")):
        arms = sorted(p.name for p in seed_dir.iterdir() if p.is_dir())
        rates = sorted({int(p.name[3:]) for arm in arms for p in (seed_dir / arm).glob("rps*") if p.is_dir()})
        for rate in rates:
            print(f"\n{seed_dir.name} {rate} req/s")
            print("| Arm | Succeeded | Underrun requests | Largest stall ms | First byte p50 / p95 ms | Dispatch lag p99 ms "
                  "| API main thread CPU p50 / p90 | API process CPU | /health p50 / p90 / max ms | Client CPU / top thread p90 "
                  "| Tracked max | Top stage threads (cores) |")
            print("|---" * 12 + "|")
            for arm in arms:
                summary_path = seed_dir / arm / f"rps{rate}" / "summary.json"
                procs_path = seed_dir / arm / f"rps{rate}.procs.jsonl"
                if not summary_path.exists():
                    print(f"| {arm} | missing |")
                    continue
                else:
                    pass
                summary = json.loads(summary_path.read_text())
                view = process_view(procs_path) if procs_path.exists() else {}
                tree_line = (seed_dir / arm / "tree.txt").read_text().splitlines()[0]
                label = f"{tree_line.split()[0].rsplit('/', 1)[-1]} {tree_line.partition(' args ')[2]}".strip()
                print(
                    f"| {arm} {label} | {fmt(summary['success_fraction'], 100, 1)} % ({summary['successful_requests']}) "
                    f"| {summary['underrun_requests']} | {fmt(summary['largest_stall_ms'], 1, 1)} "
                    f"| {fmt(summary['ttfb_ms']['p50'], 1, 1)} / {fmt(summary['ttfb_ms']['p95'], 1, 1)} "
                    f"| {fmt(summary['scheduling_lag_ms']['p99'], 1, 2)} "
                    f"| {fmt(view.get('api_main_p50'))} / {fmt(view.get('api_main_p90'))} | {fmt(view.get('api_total_p50'))} "
                    f"| {fmt(view.get('health_p50'), 1000, 1)} / {fmt(view.get('health_p90'), 1000, 1)} / "
                    f"{fmt(view.get('health_max'), 1000, 1)} "
                    f"| {fmt(view.get('client_total_p50'))} / {fmt(view.get('client_top_p90'))} "
                    f"| {view.get('tracked_max', '-')} | {view.get('stage_top', '-')} |"
                )


if __name__ == "__main__":
    main()
