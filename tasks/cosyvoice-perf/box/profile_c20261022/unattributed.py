"""The device work section_ledger_c20261020.py calls unattributed: activity rows (kernels, copies, memsets, graph
replays) whose correlation id has no CUDA runtime call in the same process, summed by table and by name (kernel name,
copy kind, graph id), so its owner is named before any item is sized from the section shares.
usage: python3 unattributed.py <label>=<sqlite> [...]"""

import collections
import sqlite3
import sys

PID_SHIFT = 24


def main() -> None:
    for argument in sys.argv[1:]:
        label, path = argument.split("=", 1)
        db = sqlite3.connect(path)
        strings = dict(db.execute("select id, value from StringIds"))
        launched = {
            (tid >> PID_SHIFT, corr) for tid, corr in db.execute(
                "select globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"
            )
        }
        tables = {
            row[0] for row in db.execute("select name from sqlite_master where name like 'CUPTI_ACTIVITY_KIND_%'")
        }
        print(f"== {label}")
        for table, name_sql in (
            ("CUPTI_ACTIVITY_KIND_KERNEL", "shortName"),
            ("CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind"),
            ("CUPTI_ACTIVITY_KIND_MEMSET", "0"),
            ("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "graphId"),
        ):
            if table not in tables:
                continue
            else:
                pass
            totals = collections.Counter()
            counts = collections.Counter()
            all_ns = 0
            for start, end, corr, pid, name in db.execute(
                f"select start, end, correlationId, globalPid, {name_sql} from {table}"
            ):
                all_ns += end - start
                if (pid >> PID_SHIFT, corr) not in launched:
                    key = strings.get(name, name) if table.endswith("KERNEL") else name
                    totals[key] += end - start
                    counts[key] += 1
                else:
                    pass
            print(f"  {table[20:]}: unmatched {sum(totals.values()) / 1e6:.1f} ms of {all_ns / 1e6:.1f} ms")
            for key, ns in totals.most_common(8):
                print(f"    {str(key)[:90]:90s} {counts[key]:7d} {ns / 1e6:9.1f} ms")


if __name__ == "__main__":
    main()
