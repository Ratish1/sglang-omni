"""How far ahead of playback the second and third chunks arrive, per arm and rate of a bar_overload_pair run.

From Nari's client chunk events: the second chunk's margin is the first chunk's arrival plus its media length
minus the second chunk's arrival, and the third chunk's adds the second chunk's media. Warm-up requests and
requests with fewer than three audio chunks are left out.

usage: python3 opening_margin.py OUT
"""

import collections
import json
import sys
from pathlib import Path


def margins(path: Path) -> list[tuple[float, float, float]]:
    chunks: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    phase: dict[str, str | None] = {}
    with path.open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("type") != "response.body_chunk":
                continue
            else:
                pass
            media_ns = event["data"].get("media_duration_ns", 0)
            if media_ns <= 0:
                continue
            else:
                pass
            chunks[event["request_id"]].append((event["elapsed_ns"], media_ns))
            phase[event["request_id"]] = event.get("phase")
    out = []
    for request_id, items in chunks.items():
        if phase.get(request_id) == "warmup" or len(items) < 3:
            continue
        else:
            pass
        items.sort()
        (first_ns, first_media_ns), (second_ns, second_media_ns), (third_ns, _) = items[:3]
        out.append((
            (first_ns + first_media_ns - second_ns) / 1e6,
            (first_ns + first_media_ns + second_media_ns - third_ns) / 1e6,
            first_media_ns / 1e6,
        ))
    return out


def quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return round(values[min(len(values) - 1, int(fraction * len(values)))], 1)


def main() -> None:
    root = Path(sys.argv[1])
    for events in sorted(root.glob("seed*/*/rps*/events.jsonl")):
        found = margins(events)
        if not found:
            print(f"{events.parent} no requests")
            continue
        else:
            pass
        second = [margin[0] for margin in found]
        third = [margin[1] for margin in found]
        first_media = collections.Counter(round(margin[2]) for margin in found)
        print(
            f"{events.parts[-4]} {events.parts[-2]} {events.parts[-3]}: n {len(found)} | chunk2 margin ms p1/p5/p50 "
            f"{quantile(second, .01)}/{quantile(second, .05)}/{quantile(second, .5)} | chunk3 margin p1/p5/p50 "
            f"{quantile(third, .01)}/{quantile(third, .05)}/{quantile(third, .5)} | first chunk media ms "
            f"{dict(first_media.most_common(3))}"
        )


if __name__ == "__main__":
    main()
