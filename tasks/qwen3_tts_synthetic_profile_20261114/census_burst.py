"""Tracked objects a waiting request holds in the API process. One server on one card with the
default settings and gc_probe: an object census at idle, then BURST streaming requests at once (the
coordinator admits up to its in-flight cap and holds the rest), a second census after HOLD_S while
they wait, and per type the difference divided by the requests that had no audio yet.

usage: python3 census_burst.py --tree TREE --card N --out DIR [--burst 2000] [--hold-s 3]
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from bar_overload_pair import GC_PROBE_DIR
from seedtts_pair import serve, stop
from served_baseline import MODELS, wait_healthy

CENSUS_TIMEOUT_S = 120
REQUEST_TIMEOUT_S = 600
TEXT = "Please confirm the meeting time with the rest of the team before noon tomorrow."
TOP_TYPES = 30


def read_census(path: Path) -> tuple[int, dict[str, int]]:
    lines = path.read_text().splitlines()
    tracked = int(lines[0].split()[2])
    counts = {}
    for line in lines[1:]:
        count, name = line.split("\t")
        counts[name] = int(count)
    return tracked, counts


async def take_census(out: Path, index: int) -> Path:
    (out / f"server.census.{index}").touch()
    deadline = time.monotonic() + CENSUS_TIMEOUT_S
    while time.monotonic() < deadline:
        found = sorted(out.glob(f"server.*.census.{index}.txt"))
        if found:
            return found[0]
        else:
            await asyncio.sleep(0.2)
    raise TimeoutError(f"census {index} not written")


async def stream_one(client: httpx.AsyncClient, url: str, model: str, state: dict[str, int]) -> None:
    body = {"model": model, "input": TEXT, "voice": "Ryan", "language": "English", "response_format": "pcm", "stream": True}
    async with client.stream("POST", f"{url}/v1/audio/speech", json=body) as response:
        is_first = True
        async for _ in response.aiter_bytes():
            if is_first:
                state["with_audio"] += 1
                is_first = False
            else:
                pass
    state["done"] += 1


async def drive(url: str, model: str, out: Path, burst: int, hold_s: float) -> dict:
    idle_path = await take_census(out, 1)
    state = {"with_audio": 0, "done": 0}
    limits = httpx.Limits(max_connections=burst + 8, max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, limits=limits) as client:
        tasks = [asyncio.create_task(stream_one(client, url, model, state)) for _ in range(burst)]
        await asyncio.sleep(hold_s)
        with_audio_at_census = state["with_audio"]
        loaded_path = await take_census(out, 2)
        health = (await client.get(f"{url}/health")).json()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    idle_tracked, idle_counts = read_census(idle_path)
    loaded_tracked, loaded_counts = read_census(loaded_path)
    waiting = burst - with_audio_at_census
    deltas = sorted(
        ((loaded_counts.get(name, 0) - idle_counts.get(name, 0), name) for name in set(idle_counts) | set(loaded_counts)),
        reverse=True,
    )
    return {
        "burst": burst, "with_audio_at_census": with_audio_at_census, "waiting_at_census": waiting,
        "admitted_at_census": health.get("total_requests"),
        "idle_tracked": idle_tracked, "loaded_tracked": loaded_tracked,
        "tracked_per_waiting": (loaded_tracked - idle_tracked) / max(1, waiting),
        "top_types": [
            {"type": name, "delta": delta, "per_waiting": round(delta / max(1, waiting), 2)}
            for delta, name in deltas[:TOP_TYPES]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29000)
    parser.add_argument("--burst", type=int, default=2000)
    parser.add_argument("--hold-s", type=float, default=3.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model, _ = MODELS["customvoice"]
    server = serve(
        args.tree, model, args.card, args.port, args.out / "serve.log",
        {"PYTHONPATH": f"{args.tree}:{GC_PROBE_DIR}", "GC_PROBE_OUT": str(args.out / "server")},
    )
    try:
        wait_healthy(args.port, server)
        result = asyncio.run(drive(f"http://127.0.0.1:{args.port}", model, args.out, args.burst, args.hold_s))
    finally:
        stop(server)
    (args.out / "census_burst.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({key: value for key, value in result.items() if key != "top_types"}), flush=True)
    for row in result["top_types"]:
        print(f"{row['delta']:>9} {row['per_waiting']:>8} {row['type']}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
