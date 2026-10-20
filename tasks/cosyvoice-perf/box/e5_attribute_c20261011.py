# SPDX-License-Identifier: Apache-2.0
"""E5 attribution: from one allocator snapshot (torch.cuda.memory._dump_snapshot, history recorded from
healthy), the device memory each stream mapped while serving, by the Python code that asked for it.

A segment_map (expandable segments) or segment_alloc entry is the allocator taking new device memory for
an allocation that did not fit what the stream already held; its frames are that allocation's stack. Per
stream: total bytes mapped and unmapped, then the mapped bytes grouped by the innermost frame in
sglang_omni, by the innermost frame in sglang, and by the innermost Python frame, largest first.

usage: python e5_attribute_c20261011.py <snapshot.pickle> [top]
"""

from __future__ import annotations

import collections
import pickle
import sys

GIB = 2**30


def frame_text(frame: dict) -> str:
    return f"{frame['filename'].split('site-packages/')[-1].split('/workspace/')[-1]}:{frame['line']} {frame['name']}"


def innermost(frames: list[dict], needle: str | None) -> str:
    for frame in frames:
        if needle is None or needle in frame["filename"]:
            return frame_text(frame)
    return "(none)"


def main() -> None:
    snapshot = pickle.load(open(sys.argv[1], "rb"))
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    traces = [entry for device in snapshot["device_traces"] for entry in device]
    actions = collections.Counter(entry["action"] for entry in traces)
    print(f"trace entries {len(traces)}: {dict(actions)}")
    grown = [e for e in traces if e["action"] in ("segment_map", "segment_alloc")]
    shrunk = [e for e in traces if e["action"] in ("segment_unmap", "segment_free")]
    streams = sorted({e["stream"] for e in grown + shrunk})
    for stream in streams:
        mapped = [e for e in grown if e["stream"] == stream]
        unmapped = [e for e in shrunk if e["stream"] == stream]
        print(
            f"\nstream {stream}: mapped {sum(e['size'] for e in mapped) / GIB:.3f} GiB in "
            f"{len(mapped)} events, unmapped {sum(e['size'] for e in unmapped) / GIB:.3f} GiB in "
            f"{len(unmapped)} events"
        )
        for label, needle in (("sglang_omni", "sglang_omni"), ("sglang", "/sglang/"), ("python", None)):
            by = collections.Counter()
            for e in mapped:
                by[innermost(e.get("frames", []), needle)] += e["size"]
            print(f"  by innermost {label} frame:")
            for where, size in by.most_common(top):
                print(f"    {size / GIB:7.3f} GiB  {where}")
    for e in traces:
        if e["action"] == "oom":
            print("oom", e.get("size"), innermost(e.get("frames", []), "sglang_omni"))


if __name__ == "__main__":
    main()
