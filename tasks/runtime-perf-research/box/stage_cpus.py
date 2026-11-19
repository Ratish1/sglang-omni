"""The CPUs and OpenMP setting each process of a server runs with: OMP_NUM_THREADS from its environment, its affinity
count and its thread count, one line a process, for every process in the server's process group.

usage: python3 stage_cpus.py <server pgid>
"""

import os
import sys
from pathlib import Path


def main() -> None:
    pgid = int(sys.argv[1])
    for entry in sorted(Path("/proc").iterdir(), key=lambda path: path.name):
        if not entry.name.isdigit():
            continue
        else:
            pass
        try:
            if os.getpgid(int(entry.name)) != pgid:
                continue
            else:
                pass
            environ = dict(
                item.split("=", 1)
                for item in (entry / "environ").read_bytes().decode(errors="replace").split("\0")
                if "=" in item
            )
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:120]
            print(
                f"pid {entry.name} omp {environ.get('OMP_NUM_THREADS', '-')} "
                f"cpus {len(os.sched_getaffinity(int(entry.name)))} threads {len(list((entry / 'task').iterdir()))} "
                f"comm {(entry / 'comm').read_text().strip()} cmd {command}",
                flush=True,
            )
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            continue


if __name__ == "__main__":
    main()
