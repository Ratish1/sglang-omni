"""Check that two node traces served the same graph programs.

For every CUDA graph capture (a probe cap range) the kernels its replays ran, one name
per executed node, in node order. Captures are matched by their site and their order at
that site. The capture time op labels of a pass with OMNI_PIPE_LINES must not change a
single graph, so a labeled and an unlabeled report must agree on every matched capture;
any difference names the capture and the kernels that differ.

usage: python graph_program_diff.py LABELED.sqlite UNLABELED.sqlite
"""

from __future__ import annotations

import bisect
import collections
import sqlite3
import sys

PUSH_POP = 59


def programs(path: str) -> dict[tuple[str, int], list[str]]:
    db = sqlite3.connect(path)
    strings = dict(db.execute("select id, value from StringIds"))
    caps = collections.defaultdict(list)
    for start, end, tid, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS where eventType = ?",
        (PUSH_POP,),
    ):
        label = text if text is not None else strings.get(text_id, "")
        if label.startswith("cap ") and end is not None:
            caps[tid].append((start, end, label[4:]))
    for items in caps.values():
        items.sort()
    starts = {tid: [c[0] for c in items] for tid, items in caps.items()}
    original, created = {}, {}
    for start, tid, node, orig in db.execute(
        "select start, globalTid, graphNodeId, originalGraphNodeId from CUDA_GRAPH_NODE_EVENTS"
    ):
        if orig is not None:
            original[node] = orig
        elif node not in created:
            created[node] = (start, tid)
    capture_of = {}
    kernels = {}
    for node, name_id in db.execute(
        "select distinct graphNodeId, demangledName from CUPTI_ACTIVITY_KIND_KERNEL "
        "where graphNodeId is not null"
    ):
        root = node
        while root in original:
            root = original[root]
        origin = created.get(root)
        if origin is None:
            continue
        start, tid = origin
        items = caps.get(tid, [])
        index = bisect.bisect_right(starts.get(tid, []), start) - 1
        if index < 0 or items[index][1] < start:
            continue
        capture_of[node] = (items[index][2], items[index][0])
        kernels[node] = (root, strings.get(name_id, str(name_id)))
    by_capture = collections.defaultdict(list)
    for node, key in capture_of.items():
        by_capture[key].append(kernels[node])
    ordered = collections.defaultdict(list)
    for (site, start), nodes in by_capture.items():
        ordered[site].append((start, [name for _, name in sorted(nodes)]))
    result = {}
    for site, entries in ordered.items():
        for index, (_, names) in enumerate(sorted(entries)):
            result[(site, index)] = names
    return result


def main() -> None:
    a, b = programs(sys.argv[1]), programs(sys.argv[2])
    common = sorted(set(a) & set(b))
    same = sum(1 for key in common if a[key] == b[key])
    print(
        f"captures replayed: labeled {len(a)}, unlabeled {len(b)}, matched {len(common)}, identical {same}"
    )
    for key in common:
        if a[key] != b[key]:
            only_a = collections.Counter(a[key]) - collections.Counter(b[key])
            only_b = collections.Counter(b[key]) - collections.Counter(a[key])
            print(f"DIFF {key[0]} #{key[1]}: {len(a[key])} vs {len(b[key])} nodes")
            for name, count in list(only_a.items())[:5]:
                print(f"   labeled only  {count} x {name[:100]}")
            for name, count in list(only_b.items())[:5]:
                print(f"   unlabeled only {count} x {name[:100]}")
    for key in sorted(set(a) ^ set(b))[:20]:
        print(f"unmatched {'labeled' if key in a else 'unlabeled'}: {key[0]} #{key[1]}")


if __name__ == "__main__":
    main()
