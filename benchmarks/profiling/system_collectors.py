# SPDX-License-Identifier: Apache-2.0
"""Linux host evidence collectors for SGLang-Omni profiling runs."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PERF_BASE_EVENTS = (
    "task-clock",
    "cycles",
    "ref-cycles",
    "instructions",
    "context-switches",
    "cpu-migrations",
    "page-faults",
)


@dataclass(frozen=True)
class CommandCapture:
    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    available: bool = True


def capture_command(argv: Sequence[str], *, timeout_s: float = 20.0) -> CommandCapture:
    executable = shutil.which(argv[0])
    if executable is None:
        return CommandCapture(
            argv=list(argv),
            returncode=None,
            stdout="",
            stderr=f"{argv[0]} is not installed",
            available=False,
        )
    try:
        completed = subprocess.run(
            [executable, *argv[1:]],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandCapture(
            argv=list(argv),
            returncode=124,
            stdout=(
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            ),
            stderr=(
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            + f"\ncommand timed out after {timeout_s}s",
        )
    return CommandCapture(
        argv=list(argv),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_process_environ(pid: int) -> dict[str, str]:
    """Read a bounded, profiling-relevant subset of a process environment."""
    wanted_prefixes = (
        "CUDA",
        "NCCL",
        "OMP",
        "MKL",
        "OPENBLAS",
        "NUMEXPR",
        "TORCH",
        "SGLANG",
        "HF_",
        "PYTORCH",
    )
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key_raw, value_raw = item.split(b"=", 1)
        key = key_raw.decode(errors="replace")
        if key.startswith(wanted_prefixes):
            result[key] = value_raw.decode(errors="replace")
    return result


def _process_tree(root_pid: int) -> list[dict[str, Any]]:
    """Snapshot the root process and all visible descendants from procfs."""
    parent_by_pid: dict[int, int] = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = {}
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key] = value.strip()
            pid = int(fields["Pid"])
            parent_by_pid[pid] = int(fields["PPid"])
        except (OSError, KeyError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent_by_pid.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    result: list[dict[str, Any]] = []
    for pid in sorted(descendants):
        cmdline = _read_text(f"/proc/{pid}/cmdline")
        result.append(
            {
                "pid": pid,
                "ppid": parent_by_pid.get(pid),
                "cmdline": cmdline.replace("\0", " ") if cmdline else None,
                "comm": _read_text(f"/proc/{pid}/comm"),
                "cgroup": _read_text(f"/proc/{pid}/cgroup"),
                "cpus_allowed_list": _status_value(pid, "Cpus_allowed_list"),
                "mems_allowed_list": _status_value(pid, "Mems_allowed_list"),
                "thread_count": len(list(Path(f"/proc/{pid}/task").glob("[0-9]*"))),
            }
        )
    return result


def _status_value(pid: int, key: str) -> str | None:
    try:
        for line in (
            Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def collect_static_manifest(
    *,
    server_pid: int | None,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Capture version, topology, policy, and process facts without mutation."""

    repo = Path(repo_root).resolve()
    commands = {
        "git_revision": ["git", "-C", str(repo), "rev-parse", "HEAD"],
        "git_status": ["git", "-C", str(repo), "status", "--short"],
        "uname": ["uname", "-a"],
        "lscpu": [
            "lscpu",
            "-e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ",
        ],
        "numactl": ["numactl", "--hardware"],
        "ambient_processes": [
            "ps",
            "-eLo",
            "pid,ppid,tid,psr,pcpu,pmem,comm",
            "--sort=-pcpu",
        ],
        "nvidia_topology": ["nvidia-smi", "topo", "-m"],
        "nvidia_smi": [
            "nvidia-smi",
            (
                "--query-gpu=index,name,uuid,pci.bus_id,driver_version,"
                "temperature.gpu,power.limit"
            ),
            "--format=csv,noheader",
        ],
        "python": ["python3", "--version"],
    }
    if server_pid is not None:
        commands.update(
            {
                "process": [
                    "ps",
                    "-o",
                    "pid,ppid,psr,nlwp,pcpu,pmem,etime,command",
                    "-p",
                    str(server_pid),
                ],
                "process_affinity": [
                    "taskset",
                    "-pc",
                    str(server_pid),
                ],
                "process_numa": ["numastat", "-p", str(server_pid)],
            }
        )
    captures = {name: asdict(capture_command(argv)) for name, argv in commands.items()}
    manifest = {
        "captured_at_ns": time.time_ns(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "server_pid": server_pid,
        "commands": captures,
        "cpu_policy": {
            "intel_pstate_status": _read_text(
                "/sys/devices/system/cpu/intel_pstate/status"
            ),
            "boost": _read_text("/sys/devices/system/cpu/cpufreq/boost"),
            "no_turbo": _read_text("/sys/devices/system/cpu/intel_pstate/no_turbo"),
            "smt_control": _read_text("/sys/devices/system/cpu/smt/control"),
            "scaling_driver": _read_text(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"
            ),
            "scaling_governor": _read_text(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            ),
            "energy_performance_preference": _read_text(
                "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"
            ),
        },
        "cgroup": {
            "cpu_max": _read_text("/sys/fs/cgroup/cpu.max"),
            "cpuset_effective": _read_text("/sys/fs/cgroup/cpuset.cpus.effective"),
            "memory_nodes_effective": _read_text(
                "/sys/fs/cgroup/cpuset.mems.effective"
            ),
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    if server_pid is not None:
        manifest["server_process"] = {
            "cmdline": (
                (_read_text(f"/proc/{server_pid}/cmdline") or "").replace("\0", " ")
                or None
            ),
            "environment": _read_process_environ(server_pid),
            "tree": _process_tree(server_pid),
        }
    return manifest


def read_psi(
    root: str | Path = "/proc/pressure",
    *,
    cgroup_layout: bool = False,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"captured_monotonic_ns": time.monotonic_ns()}
    for resource in ("cpu", "memory", "io"):
        path = Path(root) / (f"{resource}.pressure" if cgroup_layout else resource)
        lines: dict[str, dict[str, float | int]] = {}
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            snapshot[resource] = {"error": str(exc)}
            continue
        for raw_line in content.splitlines():
            fields = raw_line.split()
            if not fields:
                continue
            values: dict[str, float | int] = {}
            for field in fields[1:]:
                key, raw_value = field.split("=", 1)
                values[key] = int(raw_value) if key == "total" else float(raw_value)
            lines[fields[0]] = values
        snapshot[resource] = lines
    return snapshot


def psi_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    delta: dict[str, Any] = {
        "window_ns": int(after["captured_monotonic_ns"])
        - int(before["captured_monotonic_ns"])
    }
    for resource in ("cpu", "memory", "io"):
        resource_delta: dict[str, Any] = {}
        before_resource = before.get(resource, {})
        after_resource = after.get(resource, {})
        for scope in ("some", "full"):
            before_scope = before_resource.get(scope)
            after_scope = after_resource.get(scope)
            if not isinstance(before_scope, dict) or not isinstance(after_scope, dict):
                continue
            resource_delta[scope] = {
                "total_us": int(after_scope.get("total", 0))
                - int(before_scope.get("total", 0)),
                "avg10_end": after_scope.get("avg10"),
                "avg60_end": after_scope.get("avg60"),
                "avg300_end": after_scope.get("avg300"),
            }
        delta[resource] = resource_delta
    return delta


def _process_cgroup_root(pid: int) -> Path | None:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0":
            relative = fields[2].lstrip("/")
            return Path("/sys/fs/cgroup") / relative
    return None


def read_process_cgroup_psi(pid: int) -> dict[str, Any]:
    root = _process_cgroup_root(pid)
    if root is None:
        return {
            "captured_monotonic_ns": time.monotonic_ns(),
            "error": f"cannot resolve cgroup v2 root for pid {pid}",
        }
    snapshot = read_psi(root, cgroup_layout=True)
    snapshot["root"] = str(root)
    return snapshot


def _sched_counter(path: Path, name: str) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == name:
                return int(float(value.strip()))
    except (OSError, ValueError):
        pass
    return None


def read_thread_snapshot(pid: int) -> dict[str, Any]:
    """Read one process-wide native-thread CPU and scheduler snapshot."""
    rows: list[dict[str, Any]] = []
    for task_dir in sorted(Path(f"/proc/{pid}/task").glob("[0-9]*")):
        try:
            tid = int(task_dir.name)
            raw_stat = (task_dir / "stat").read_text(encoding="utf-8")
            close = raw_stat.rfind(")")
            comm = raw_stat[raw_stat.find("(") + 1 : close]
            fields = raw_stat[close + 2 :].split()
            schedstat = (task_dir / "schedstat").read_text(encoding="utf-8").split()
            rows.append(
                {
                    "tid": tid,
                    "comm": comm,
                    "state": fields[0],
                    "utime_ticks": int(fields[11]),
                    "stime_ticks": int(fields[12]),
                    "starttime_ticks": int(fields[19]),
                    "processor": int(fields[36]),
                    "runtime_ns": int(schedstat[0]),
                    "runqueue_delay_ns": int(schedstat[1]),
                    "timeslices": int(schedstat[2]),
                    "migrations": _sched_counter(
                        task_dir / "sched",
                        "se.nr_migrations",
                    ),
                }
            )
        except (OSError, ValueError, IndexError):
            continue
    return {
        "monotonic_ns": time.monotonic_ns(),
        "pid": int(pid),
        "threads": rows,
    }


def find_processes_with_thread_comms(
    required_comms: Sequence[str],
    *,
    proc_root: str | Path = "/proc",
) -> list[dict[str, Any]]:
    """Find processes containing every required Linux native-thread name."""
    required = {str(comm) for comm in required_comms if str(comm)}
    if not required:
        raise ValueError("at least one native-thread comm is required")
    root = Path(proc_root)
    candidates: list[dict[str, Any]] = []
    for process_dir in sorted(root.glob("[0-9]*")):
        try:
            pid = int(process_dir.name)
        except ValueError:
            continue
        observed: dict[str, list[int]] = defaultdict(list)
        for comm_path in (process_dir / "task").glob("[0-9]*/comm"):
            try:
                comm = comm_path.read_text(encoding="utf-8").strip()
                tid = int(comm_path.parent.name)
            except (OSError, ValueError):
                continue
            if comm in required:
                observed[comm].append(tid)
        if required <= observed.keys():
            candidates.append(
                {
                    "pid": pid,
                    "required_thread_tids": {
                        comm: sorted(observed[comm]) for comm in sorted(required)
                    },
                    "cmdline": (
                        (_read_text(str(process_dir / "cmdline")) or "").replace(
                            "\0", " "
                        )
                        or None
                    ),
                }
            )
    return candidates


def summarize_thread_snapshot_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Summarize native-thread deltas across one workload window."""

    def identity(row: dict[str, Any]) -> tuple[int, int | None]:
        starttime = row.get("starttime_ticks")
        return (
            int(row["tid"]),
            int(starttime) if starttime is not None else None,
        )

    first = {identity(row): row for row in before.get("threads", [])}
    last = {identity(row): row for row in after.get("threads", [])}
    clock_ticks = float(os.sysconf("SC_CLK_TCK"))
    window_ns = max(
        0,
        int(after["monotonic_ns"]) - int(before["monotonic_ns"]),
    )
    rows: list[dict[str, Any]] = []
    persistent = first.keys() & last.keys()
    started = last.keys() - first.keys()
    exited = first.keys() - last.keys()
    for thread_identity in sorted(
        persistent | started,
        key=lambda value: (
            value[0],
            value[1] if value[1] is not None else -1,
        ),
    ):
        end = last[thread_identity]
        start = first.get(thread_identity)
        lifecycle = "persistent" if start is not None else "started"
        runtime_ns = max(
            0,
            int(end["runtime_ns"]) - (int(start["runtime_ns"]) if start else 0),
        )
        delay_ns = max(
            0,
            int(end["runqueue_delay_ns"])
            - (int(start["runqueue_delay_ns"]) if start else 0),
        )
        cpu_ticks = max(
            0,
            int(end["utime_ticks"])
            + int(end["stime_ticks"])
            - (int(start["utime_ticks"]) + int(start["stime_ticks"]) if start else 0),
        )
        start_migrations = start.get("migrations") if start else 0
        end_migrations = end.get("migrations")
        rows.append(
            {
                "tid": int(end["tid"]),
                "starttime_ticks": end.get("starttime_ticks"),
                "comm": end.get("comm"),
                "lifecycle": lifecycle,
                "cpu_ms": cpu_ticks * 1000.0 / clock_ticks,
                "runtime_ms": runtime_ns / 1e6,
                "runqueue_delay_ms": delay_ns / 1e6,
                "timeslices": max(
                    0,
                    int(end["timeslices"]) - (int(start["timeslices"]) if start else 0),
                ),
                "migrations": (
                    max(0, int(end_migrations) - int(start_migrations))
                    if start_migrations is not None and end_migrations is not None
                    else None
                ),
                "runtime_fraction": runtime_ns / window_ns if window_ns else None,
                "runqueue_delay_fraction": (
                    delay_ns / (runtime_ns + delay_ns) if runtime_ns + delay_ns else 0.0
                ),
                "first_processor": start.get("processor") if start else None,
                "last_processor": end.get("processor"),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["cpu_ms"]),
            int(row["tid"]),
            int(row["starttime_ticks"] or -1),
        )
    )
    return {
        "window_ns": window_ns,
        "threads_before": len(first),
        "threads_after": len(last),
        "threads_observed": len(rows),
        "threads_persistent": len(persistent),
        "threads_started": sorted(tid for tid, _starttime in started),
        "threads_exited": sorted(tid for tid, _starttime in exited),
        "cpu_ms": sum(float(row["cpu_ms"]) for row in rows),
        "runtime_ms": sum(float(row["runtime_ms"]) for row in rows),
        "runqueue_delay_ms": sum(float(row["runqueue_delay_ms"]) for row in rows),
        "migrations": sum(
            int(row["migrations"]) for row in rows if row.get("migrations") is not None
        ),
        "threads": rows,
    }


class ManagedCollector:
    """A signal-terminated collector with explicit startup and finalization."""

    def __init__(
        self,
        *,
        name: str,
        argv: Sequence[str],
        stdout_path: str | Path | None = None,
    ) -> None:
        self.name = name
        self.argv = list(argv)
        self.stdout_path = Path(stdout_path).resolve() if stdout_path else None
        self._process: subprocess.Popen[str] | None = None
        self._stdout_handle: Any = None
        self.started_monotonic_ns: int | None = None
        self.stopped_monotonic_ns: int | None = None

    def start(self) -> None:
        executable = shutil.which(self.argv[0])
        if executable is None:
            raise RuntimeError(f"{self.argv[0]} is not installed")
        if self._process is not None:
            raise RuntimeError(f"collector {self.name} is already running")
        stdout: Any = subprocess.DEVNULL
        if self.stdout_path is not None:
            self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
            self._stdout_handle = self.stdout_path.open("w", encoding="utf-8")
            stdout = self._stdout_handle
        self._process = subprocess.Popen(
            [executable, *self.argv[1:]],
            text=True,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
        self.started_monotonic_ns = time.monotonic_ns()

    def stop(self, *, timeout_s: float = 30.0) -> dict[str, Any]:
        process = self._process
        if process is None:
            raise RuntimeError(f"collector {self.name} was not started")
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        self.stopped_monotonic_ns = time.monotonic_ns()
        if self._stdout_handle is not None:
            self._stdout_handle.flush()
            self._stdout_handle.close()
            self._stdout_handle = None
        result = {
            "name": self.name,
            "argv": self.argv,
            "returncode": process.returncode,
            "started_monotonic_ns": self.started_monotonic_ns,
            "stopped_monotonic_ns": self.stopped_monotonic_ns,
            "output": str(self.stdout_path) if self.stdout_path else None,
        }
        self._process = None
        return result


def _nsys_linux_ids(global_tid: int) -> tuple[int, int]:
    """Decode Nsight's Linux PID/TID fields from a serialized globalTid."""
    unsigned = int(global_tid) & ((1 << 64) - 1)
    return (unsigned // 0x1000000) % 0x1000000, unsigned % 0x1000000


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def _nsys_cpu_hotspots(
    path: Path,
    *,
    global_tids: Sequence[int],
    denominator: int,
    leaf_only: bool,
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in global_tids)
    leaf_filter = "AND calls.stackDepth = 0" if leaf_only else ""
    query = f"""
        SELECT COALESCE(symbols.value, printf('symbol-id:%d', calls.symbol)),
               COALESCE(modules.value, printf('module-id:%d', calls.module)),
               COUNT(*) AS sample_count
        FROM COMPOSITE_EVENTS AS events
        JOIN SAMPLING_CALLCHAINS AS calls ON calls.id = events.id
        LEFT JOIN StringIds AS symbols ON symbols.id = calls.symbol
        LEFT JOIN StringIds AS modules ON modules.id = calls.module
        WHERE events.cpuCycles = 1
          AND events.globalTid IN ({placeholders})
          {leaf_filter}
        GROUP BY calls.symbol, calls.module
        ORDER BY sample_count DESC, symbols.value, modules.value
        LIMIT 25
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(query, list(global_tids)).fetchall()
    finally:
        connection.close()
    return [
        {
            "symbol": str(symbol),
            "module": str(module),
            "sample_count": int(count),
            "occurrences_per_cpu_sample": (
                int(count) / denominator if denominator else None
            ),
        }
        for symbol, module, count in rows
    ]


def inspect_nsys_cpu_sqlite(
    path: str | Path,
    *,
    target_pid: int,
    thread_snapshot_before: dict[str, Any],
    thread_snapshot_after: dict[str, Any],
    required_thread_comms: Sequence[str],
) -> dict[str, Any]:
    """Validate system-wide Nsight CPU evidence for one stable target process.

    Nsight's SQLite schema is versioned.  This reader intentionally relies only
    on the documented SCHED_EVENTS and COMPOSITE_EVENTS contracts and records
    the observed schema so an incompatible exporter fails visibly.
    """
    source = Path(path).resolve()
    errors: list[str] = []
    result: dict[str, Any] = {
        "path": str(source),
        "target_pid": int(target_pid),
        "required_thread_comms": list(required_thread_comms),
        "errors": errors,
    }
    if not source.is_file() or source.stat().st_size == 0:
        errors.append(f"Nsight SQLite export is missing or empty: {source}")
        result["valid"] = False
        return result

    before_rows = {
        (int(row["tid"]), int(row["starttime_ticks"])): row
        for row in thread_snapshot_before.get("threads", [])
        if row.get("starttime_ticks") is not None
    }
    after_rows = {
        (int(row["tid"]), int(row["starttime_ticks"])): row
        for row in thread_snapshot_after.get("threads", [])
        if row.get("starttime_ticks") is not None
    }
    required_identities: dict[str, list[tuple[int, int]]] = {}
    for comm in required_thread_comms:
        identities = sorted(
            identity for identity, row in before_rows.items() if row.get("comm") == comm
        )
        required_identities[comm] = identities
        if not identities:
            errors.append(
                f"required native thread comm {comm!r} was absent before capture"
            )
            continue
        persistent = [identity for identity in identities if identity in after_rows]
        if not persistent:
            errors.append(
                f"required native thread comm {comm!r} had no persistent identity"
            )
        renamed = [
            identity
            for identity in persistent
            if after_rows[identity].get("comm") != comm
        ]
        if renamed:
            errors.append(
                f"required native thread comm {comm!r} changed identity/name: "
                f"{[identity[0] for identity in renamed]}"
            )

    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        result["tables"] = sorted(tables)
        required_columns = {
            "SCHED_EVENTS": {"globalTid", "start", "isSchedIn"},
            "COMPOSITE_EVENTS": {"globalTid", "start", "cpuCycles"},
            "SAMPLING_CALLCHAINS": {
                "id",
                "symbol",
                "module",
                "stackDepth",
            },
            "StringIds": {"id", "value"},
        }
        for table, expected in required_columns.items():
            if table not in tables:
                errors.append(f"Nsight SQLite export has no {table} table")
                continue
            columns = _sqlite_columns(connection, table)
            result.setdefault("schema", {})[table] = sorted(columns)
            missing = expected - columns
            if missing:
                errors.append(
                    f"Nsight {table} schema is missing columns {sorted(missing)}"
                )

        if errors and any("table" in error or "schema" in error for error in errors):
            result["valid"] = False
            return result

        sched_rows = connection.execute(
            """
            SELECT globalTid,
                   COUNT(*) AS event_count,
                   SUM(CASE WHEN isSchedIn != 0 THEN 1 ELSE 0 END) AS sched_in,
                   SUM(CASE WHEN isSchedIn = 0 THEN 1 ELSE 0 END) AS sched_out
            FROM SCHED_EVENTS
            WHERE globalTid IS NOT NULL
            GROUP BY globalTid
            """
        ).fetchall()
        sample_rows = connection.execute(
            """
            SELECT globalTid,
                   COUNT(*) AS composite_count,
                   SUM(CASE WHEN cpuCycles = 1 THEN 1 ELSE 0 END) AS cpu_samples
            FROM COMPOSITE_EVENTS
            WHERE globalTid IS NOT NULL
            GROUP BY globalTid
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        errors.append(f"cannot query Nsight SQLite export: {exc}")
        result["valid"] = False
        return result
    finally:
        connection.close()

    sched_by_tid: dict[int, dict[str, int]] = {}
    observed_sched_pids: set[int] = set()
    for global_tid, event_count, sched_in, sched_out in sched_rows:
        pid, tid = _nsys_linux_ids(int(global_tid))
        observed_sched_pids.add(pid)
        if pid == target_pid:
            sched_by_tid[tid] = {
                "events": int(event_count),
                "sched_in": int(sched_in or 0),
                "sched_out": int(sched_out or 0),
            }

    samples_by_tid: dict[int, dict[str, int]] = {}
    sample_global_by_tid: dict[int, int] = {}
    observed_sample_pids: set[int] = set()
    for global_tid, composite_count, cpu_samples in sample_rows:
        pid, tid = _nsys_linux_ids(int(global_tid))
        observed_sample_pids.add(pid)
        if pid == target_pid:
            sample_global_by_tid[tid] = int(global_tid)
            samples_by_tid[tid] = {
                "composite_events": int(composite_count),
                "cpu_samples": int(cpu_samples or 0),
            }

    table_counts = {
        "SCHED_EVENTS": sum(int(row[1]) for row in sched_rows),
        "COMPOSITE_EVENTS": sum(int(row[1]) for row in sample_rows),
        "CPU_CYCLE_SAMPLES": sum(int(row[2] or 0) for row in sample_rows),
    }
    result["table_counts"] = table_counts
    result["observed_pids"] = {
        "SCHED_EVENTS": sorted(observed_sched_pids),
        "COMPOSITE_EVENTS": sorted(observed_sample_pids),
    }
    if table_counts["SCHED_EVENTS"] == 0:
        errors.append("Nsight SCHED_EVENTS contains no scheduling events")
    if table_counts["COMPOSITE_EVENTS"] == 0:
        errors.append("Nsight COMPOSITE_EVENTS contains no rows")
    if table_counts["CPU_CYCLE_SAMPLES"] == 0:
        errors.append("Nsight COMPOSITE_EVENTS contains no cpuCycles=1 CPU samples")
    if target_pid not in observed_sched_pids:
        errors.append(f"target PID {target_pid} has no Nsight scheduling events")
    if target_pid not in observed_sample_pids:
        errors.append(f"target PID {target_pid} has no Nsight composite events")

    comm_summaries: dict[str, Any] = {}
    target_cpu_samples = sum(row["cpu_samples"] for row in samples_by_tid.values())
    for comm, identities in required_identities.items():
        tids = [identity[0] for identity in identities]
        per_thread = []
        for tid in tids:
            per_thread.append(
                {
                    "tid": tid,
                    **sched_by_tid.get(
                        tid,
                        {"events": 0, "sched_in": 0, "sched_out": 0},
                    ),
                    **samples_by_tid.get(
                        tid,
                        {"composite_events": 0, "cpu_samples": 0},
                    ),
                }
            )
        aggregate: dict[str, Any] = {
            key: sum(int(row[key]) for row in per_thread)
            for key in (
                "events",
                "sched_in",
                "sched_out",
                "composite_events",
                "cpu_samples",
            )
        }
        aggregate["cpu_sample_fraction_of_target"] = (
            aggregate["cpu_samples"] / target_cpu_samples
            if target_cpu_samples
            else None
        )
        persistent_tids = [
            identity[0] for identity in identities if identity in after_rows
        ]
        comm_summaries[comm] = {
            "tids": tids,
            "persistent_tids": persistent_tids,
            "aggregate": aggregate,
            "threads": per_thread,
        }
        global_tids = [
            sample_global_by_tid[tid] for tid in tids if tid in sample_global_by_tid
        ]
        if global_tids:
            try:
                comm_summaries[comm]["hotspots"] = {
                    "leaf": _nsys_cpu_hotspots(
                        source,
                        global_tids=global_tids,
                        denominator=aggregate["cpu_samples"],
                        leaf_only=True,
                    ),
                    "inclusive": _nsys_cpu_hotspots(
                        source,
                        global_tids=global_tids,
                        denominator=aggregate["cpu_samples"],
                        leaf_only=False,
                    ),
                }
            except sqlite3.DatabaseError as exc:
                errors.append(
                    f"cannot query sampled symbols for native thread comm "
                    f"{comm!r}: {exc}"
                )
                comm_summaries[comm]["hotspots"] = {
                    "leaf": [],
                    "inclusive": [],
                }
        else:
            comm_summaries[comm]["hotspots"] = {
                "leaf": [],
                "inclusive": [],
            }
        if identities and aggregate["events"] == 0:
            errors.append(
                f"required native thread comm {comm!r} has no scheduling events"
            )
        if identities and aggregate["cpu_samples"] == 0:
            errors.append(
                f"required native thread comm {comm!r} has no cpuCycles=1 samples"
            )
        if identities and not comm_summaries[comm]["hotspots"]["leaf"]:
            errors.append(
                f"required native thread comm {comm!r} has no sampled leaf symbols"
            )
    result["required_thread_coverage"] = comm_summaries
    result["target_totals"] = {
        "sched_events": sum(row["events"] for row in sched_by_tid.values()),
        "sched_in": sum(row["sched_in"] for row in sched_by_tid.values()),
        "sched_out": sum(row["sched_out"] for row in sched_by_tid.values()),
        "composite_events": sum(
            row["composite_events"] for row in samples_by_tid.values()
        ),
        "cpu_samples": target_cpu_samples,
    }
    result["valid"] = not errors
    return result


class NsysSystemWideCpuCollector:
    """Bounded, non-injected Nsight CPU sampling and scheduling collection."""

    def __init__(
        self,
        *,
        target_pid: int,
        artifact_dir: str | Path,
        session_name: str,
        required_thread_comms: Sequence[str],
        executable: str = "nsys",
        command_timeout_s: float = 30.0,
        finalize_timeout_s: float = 180.0,
    ) -> None:
        self.name = "nsys_system_wide_cpu"
        self.target_pid = int(target_pid)
        self.artifact_dir = Path(artifact_dir).resolve()
        self.session_name = session_name
        self.required_thread_comms = tuple(required_thread_comms)
        self.executable = executable
        self.command_timeout_s = float(command_timeout_s)
        self.finalize_timeout_s = float(finalize_timeout_s)
        self.output_prefix = self.artifact_dir / "system-wide-cpu"
        self.report_path = self.output_prefix.with_suffix(".nsys-rep")
        self.sqlite_path = self.output_prefix.with_suffix(".sqlite")
        self.started_monotonic_ns: int | None = None
        self.started_wall_ns: int | None = None
        self.stopped_monotonic_ns: int | None = None
        self.stopped_wall_ns: int | None = None
        self._before: dict[str, Any] | None = None
        self._active = False
        self._commands: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _capture_dict(capture: CommandCapture) -> dict[str, Any]:
        return asdict(capture)

    def _run(self, label: str, argv: Sequence[str], timeout_s: float) -> CommandCapture:
        capture = capture_command(argv, timeout_s=timeout_s)
        self._commands[label] = self._capture_dict(capture)
        return capture

    def start(self) -> None:
        if self._active:
            raise RuntimeError("Nsight system-wide collector is already running")
        if not Path(f"/proc/{self.target_pid}/task").is_dir():
            raise RuntimeError(f"target process {self.target_pid} does not exist")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", self.session_name):
            raise ValueError(f"invalid Nsight session name: {self.session_name!r}")
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        collisions = [
            path for path in (self.report_path, self.sqlite_path) if path.exists()
        ]
        if collisions:
            raise FileExistsError(f"Nsight artifacts already exist: {collisions}")

        environment = self._run(
            "environment",
            [self.executable, "status", "--environment"],
            self.command_timeout_s,
        )
        if not environment.available or environment.returncode != 0:
            write_json(self.artifact_dir / "lifecycle.json", self._lifecycle())
            raise RuntimeError(
                "Nsight CPU profiling environment check failed: "
                f"{environment.stderr[-1000:] or environment.stdout[-1000:]}"
            )

        self._before = read_thread_snapshot(self.target_pid)
        start_capture = self._run(
            "start",
            [
                self.executable,
                "start",
                f"--session-new={self.session_name}",
                "--stop-on-exit=false",
                "--trace=none",
                "--sample=system-wide",
                "--cpuctxsw=system-wide",
                "--resolve-symbols=false",
                "--force-overwrite=true",
                f"--output={self.output_prefix}",
            ],
            self.command_timeout_s,
        )
        if not start_capture.available or start_capture.returncode != 0:
            self._run(
                "start_failure_cleanup",
                [self.executable, "stop", f"--session={self.session_name}"],
                self.command_timeout_s,
            )
            write_json(self.artifact_dir / "lifecycle.json", self._lifecycle())
            raise RuntimeError(
                "could not start non-injected Nsight system-wide CPU session: "
                f"{start_capture.stderr[-1000:] or start_capture.stdout[-1000:]}"
            )
        self.started_monotonic_ns = time.monotonic_ns()
        self.started_wall_ns = time.time_ns()
        self._active = True

        status_capture = self._run(
            "status",
            [self.executable, "status", f"--session={self.session_name}"],
            self.command_timeout_s,
        )
        if status_capture.returncode != 0:
            self.stop()
            raise RuntimeError(
                "Nsight session did not become queryable: "
                f"{status_capture.stderr[-1000:] or status_capture.stdout[-1000:]}"
            )
        write_json(self.artifact_dir / "lifecycle_started.json", self._lifecycle())

    def _wait_for_finalized_report(self) -> tuple[bool, str | None]:
        deadline = time.monotonic() + self.finalize_timeout_s
        prior_signature: tuple[int, int] | None = None
        stable_observations = 0
        while time.monotonic() < deadline:
            if self.report_path.is_file():
                stat = self.report_path.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                if signature == prior_signature and stat.st_size > 0:
                    stable_observations += 1
                    if stable_observations >= 2:
                        return True, None
                else:
                    stable_observations = 0
                    prior_signature = signature
            time.sleep(0.5)
        return (
            False,
            f"Nsight report did not finalize within {self.finalize_timeout_s}s",
        )

    def _lifecycle(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_pid": self.target_pid,
            "session_name": self.session_name,
            "capture_contract": "non-injected-system-wide-cpu",
            "trace_domains": [],
            "sample_scope": "system-wide",
            "context_switch_scope": "system-wide",
            "required_thread_comms": list(self.required_thread_comms),
            "started_monotonic_ns": self.started_monotonic_ns,
            "started_wall_ns": self.started_wall_ns,
            "stopped_monotonic_ns": self.stopped_monotonic_ns,
            "stopped_wall_ns": self.stopped_wall_ns,
            "report_path": str(self.report_path),
            "sqlite_path": str(self.sqlite_path),
            "commands": self._commands,
        }

    def stop(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        if not self._active or self._before is None:
            raise RuntimeError("Nsight system-wide collector was not started")
        # Snapshot identity at the workload boundary, before report generation.
        self.stopped_monotonic_ns = time.monotonic_ns()
        self.stopped_wall_ns = time.time_ns()
        after = read_thread_snapshot(self.target_pid)
        stop_capture = self._run(
            "stop",
            [self.executable, "stop", f"--session={self.session_name}"],
            timeout_s or self.finalize_timeout_s,
        )
        self._active = False
        errors: list[str] = []
        if stop_capture.returncode != 0:
            errors.append(
                "nsys stop failed: "
                f"{stop_capture.stderr[-1000:] or stop_capture.stdout[-1000:]}"
            )

        finalized, finalize_error = self._wait_for_finalized_report()
        if finalize_error:
            errors.append(finalize_error)
        export_capture: CommandCapture | None = None
        evidence: dict[str, Any] | None = None
        if finalized:
            export_capture = self._run(
                "export",
                [
                    self.executable,
                    "export",
                    "--type=sqlite",
                    "--force-overwrite=true",
                    "--lazy=false",
                    f"--output={self.sqlite_path}",
                    str(self.report_path),
                ],
                self.finalize_timeout_s,
            )
            if export_capture.returncode != 0:
                errors.append(
                    "Nsight SQLite export failed: "
                    f"{export_capture.stderr[-1000:] or export_capture.stdout[-1000:]}"
                )
            else:
                evidence = inspect_nsys_cpu_sqlite(
                    self.sqlite_path,
                    target_pid=self.target_pid,
                    thread_snapshot_before=self._before,
                    thread_snapshot_after=after,
                    required_thread_comms=self.required_thread_comms,
                )
                errors.extend(str(error) for error in evidence.get("errors", []))

        result = {
            **self._lifecycle(),
            "returncode": 0 if not errors else 1,
            "error": "; ".join(errors) if errors else None,
            "finalized": finalized,
            "report_bytes": (
                self.report_path.stat().st_size if self.report_path.is_file() else 0
            ),
            "sqlite_bytes": (
                self.sqlite_path.stat().st_size if self.sqlite_path.is_file() else 0
            ),
            "thread_snapshot_before": self._before,
            "thread_snapshot_after": after,
            "evidence": evidence,
            "valid": not errors,
        }
        write_json(self.artifact_dir / "lifecycle.json", result)
        return result


class ThreadSnapshotCollector:
    """Low-overhead procfs CPU/runnable-delay sampler for every native TID."""

    def __init__(
        self,
        *,
        pid: int,
        output_path: str | Path,
        interval_ms: int = 100,
    ) -> None:
        self.name = "thread_snapshots"
        self.pid = int(pid)
        self.output_path = Path(output_path).resolve()
        self.partial_path = self.output_path.with_suffix(
            self.output_path.suffix + ".partial"
        )
        self.interval_s = max(int(interval_ms), 20) / 1000.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._samples = 0
        self.started_monotonic_ns: int | None = None
        self.stopped_monotonic_ns: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("thread snapshot collector is already running")
        if not Path(f"/proc/{self.pid}/task").is_dir():
            raise RuntimeError(f"process {self.pid} does not exist")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists() or self.partial_path.exists():
            raise FileExistsError(f"collector artifact exists: {self.output_path}")
        self.started_monotonic_ns = time.monotonic_ns()
        self._thread = threading.Thread(
            target=self._run,
            name=f"thread-snapshot-{self.pid}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            with self.partial_path.open("x", encoding="utf-8") as handle:
                while not self._stop_event.is_set():
                    sample = {
                        "monotonic_ns": time.monotonic_ns(),
                        "pid": self.pid,
                        "threads": self._read_threads(),
                    }
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
                    self._samples += 1
                    self._stop_event.wait(self.interval_s)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(self.partial_path, self.output_path)
        except Exception as exc:  # noqa: BLE001 - preserve collector failure artifact
            self._error = f"{type(exc).__name__}: {exc}"

    def _read_threads(self) -> list[dict[str, Any]]:
        return read_thread_snapshot(self.pid)["threads"]

    def stop(self, *, timeout_s: float = 30.0) -> dict[str, Any]:
        thread = self._thread
        if thread is None:
            raise RuntimeError("thread snapshot collector was not started")
        self._stop_event.set()
        thread.join(timeout=timeout_s)
        self.stopped_monotonic_ns = time.monotonic_ns()
        if thread.is_alive():
            self._error = f"collector did not stop within {timeout_s}s"
        result = {
            "name": self.name,
            "pid": self.pid,
            "interval_ms": int(self.interval_s * 1000),
            "samples": self._samples,
            "returncode": 0 if self._error is None else 1,
            "error": self._error,
            "started_monotonic_ns": self.started_monotonic_ns,
            "stopped_monotonic_ns": self.stopped_monotonic_ns,
            "output": str(self.output_path),
            "finalized": self.output_path.is_file(),
        }
        self._thread = None
        return result


class CpuFrequencyCollector:
    """Sample scaling frequency and utilization counters without linux-tools."""

    def __init__(
        self,
        *,
        output_path: str | Path,
        interval_ms: int = 1000,
        cpus: Sequence[int] | None = None,
    ) -> None:
        self.name = "cpu_frequency"
        self.output_path = Path(output_path).resolve()
        self.partial_path = self.output_path.with_suffix(
            self.output_path.suffix + ".partial"
        )
        self.interval_s = max(int(interval_ms), 100) / 1000.0
        self.cpus = (
            tuple(sorted({int(cpu) for cpu in cpus})) if cpus is not None else None
        )
        if self.cpus is not None and any(cpu < 0 for cpu in self.cpus):
            raise ValueError("CPU frequency sample CPUs must be non-negative")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._samples = 0
        self.started_monotonic_ns: int | None = None
        self.stopped_monotonic_ns: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("CPU frequency collector is already running")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists() or self.partial_path.exists():
            raise FileExistsError(f"collector artifact exists: {self.output_path}")
        self.started_monotonic_ns = time.monotonic_ns()
        self._thread = threading.Thread(
            target=self._run,
            name="cpu-frequency-snapshot",
            daemon=True,
        )
        self._thread.start()

    def _read_sample(self) -> dict[str, Any]:
        counters: dict[int, tuple[int, int]] = {}
        try:
            lines = Path("/proc/stat").read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            fields = line.split()
            if not fields or not re.fullmatch(r"cpu\d+", fields[0]):
                continue
            try:
                cpu = int(fields[0][3:])
                ticks = [int(value) for value in fields[1:]]
            except ValueError:
                continue
            if self.cpus is not None and cpu not in self.cpus:
                continue
            # guest and guest_nice are already included in user and nice.
            total = sum(ticks[:8])
            idle = sum(ticks[index] for index in (3, 4) if index < len(ticks))
            counters[cpu] = (total, idle)

        rows: list[dict[str, Any]] = []
        for cpu, (total, idle) in sorted(counters.items()):
            raw_frequency = _read_text(
                f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
            )
            try:
                frequency_khz = int(raw_frequency) if raw_frequency else None
            except ValueError:
                frequency_khz = None
            rows.append(
                {
                    "cpu": cpu,
                    "frequency_khz": frequency_khz,
                    "total_ticks": total,
                    "idle_ticks": idle,
                }
            )
        return {
            "monotonic_ns": time.monotonic_ns(),
            "requested_cpus": list(self.cpus) if self.cpus is not None else None,
            "cpus": rows,
        }

    def _run(self) -> None:
        try:
            with self.partial_path.open("x", encoding="utf-8") as handle:
                # Capture explicit leading and trailing boundaries. Periodic
                # samples fill the interval between them.
                handle.write(json.dumps(self._read_sample(), sort_keys=True) + "\n")
                self._samples += 1
                while not self._stop_event.wait(self.interval_s):
                    handle.write(json.dumps(self._read_sample(), sort_keys=True) + "\n")
                    self._samples += 1
                # Finalize with an explicit trailing boundary sample so the
                # final workload window can be bracketed.
                handle.write(json.dumps(self._read_sample(), sort_keys=True) + "\n")
                self._samples += 1
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(self.partial_path, self.output_path)
        except Exception as exc:  # noqa: BLE001 - preserve collector failure artifact
            self._error = f"{type(exc).__name__}: {exc}"

    def stop(self, *, timeout_s: float = 30.0) -> dict[str, Any]:
        thread = self._thread
        if thread is None:
            raise RuntimeError("CPU frequency collector was not started")
        self._stop_event.set()
        thread.join(timeout=timeout_s)
        self.stopped_monotonic_ns = time.monotonic_ns()
        if thread.is_alive():
            self._error = f"collector did not stop within {timeout_s}s"
        result = {
            "name": self.name,
            "interval_ms": int(self.interval_s * 1000),
            "requested_cpus": list(self.cpus) if self.cpus is not None else None,
            "samples": self._samples,
            "returncode": 0 if self._error is None else 1,
            "error": self._error,
            "started_monotonic_ns": self.started_monotonic_ns,
            "stopped_monotonic_ns": self.stopped_monotonic_ns,
            "output": str(self.output_path),
            "finalized": self.output_path.is_file(),
        }
        self._thread = None
        return result


def perf_stat_collector(
    *,
    pid: int,
    output_path: str | Path,
    events: Sequence[str] = _PERF_BASE_EVENTS,
    interval_ms: int = 1000,
    executable: str = "perf",
) -> ManagedCollector:
    return ManagedCollector(
        name="perf_stat",
        argv=[
            executable,
            "stat",
            "-x",
            ",",
            "-I",
            str(max(interval_ms, 100)),
            "-e",
            ",".join(events),
            "-p",
            str(pid),
        ],
        stdout_path=output_path,
    )


def perf_sched_collector(
    *,
    pid: int,
    output_path: str | Path,
    executable: str = "perf",
) -> ManagedCollector:
    return ManagedCollector(
        name="perf_sched",
        argv=[
            executable,
            "sched",
            "record",
            "-o",
            str(Path(output_path).resolve()),
            "-p",
            str(pid),
        ],
    )


def turbostat_collector(
    *,
    output_path: str | Path,
    cpus: str | None = None,
) -> ManagedCollector:
    argv = ["turbostat", "--quiet", "--interval", "1"]
    if cpus:
        argv.extend(["-c", cpus])
    return ManagedCollector(
        name="turbostat",
        argv=argv,
        stdout_path=output_path,
    )


def gpu_dmon_collector(
    *,
    gpu_index: int,
    output_path: str | Path,
    interval_s: int = 1,
) -> ManagedCollector:
    return ManagedCollector(
        name="gpu_dmon",
        argv=[
            "nvidia-smi",
            "dmon",
            "-i",
            str(gpu_index),
            "-s",
            "pucmt",
            "-d",
            str(max(interval_s, 1)),
        ],
        stdout_path=output_path,
    )


def parse_perf_stat(path: str | Path) -> dict[str, Any]:
    """Sum interval-mode perf CSV counters, preserving units and bad rows."""
    counters: dict[str, dict[str, Any]] = {}
    ignored: list[str] = []
    source = Path(path)
    if not source.is_file():
        return {"counters": counters, "ignored": ["file is missing"]}
    for raw_line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4:
            ignored.append(raw_line)
            continue
        raw_value = fields[1].replace(" ", "")
        event = fields[3]
        try:
            value = float(raw_value)
        except ValueError:
            ignored.append(raw_line)
            continue
        entry = counters.setdefault(
            event,
            {"value": 0.0, "unit": fields[2], "intervals": 0},
        )
        entry["value"] += value
        entry["intervals"] += 1
    return {"counters": counters, "ignored": ignored}


def parse_turbostat(path: str | Path) -> dict[str, Any]:
    """Extract numeric Bzy_MHz/Busy%/power/temperature samples by column name."""
    source = Path(path)
    if not source.is_file():
        return {"rows": 0, "columns": {}, "error": "file is missing"}
    header: list[str] | None = None
    values: dict[str, list[float]] = {}
    wanted = {
        "Bzy_MHz",
        "Busy%",
        "PkgWatt",
        "CorWatt",
        "PkgTmp",
        "CoreTmp",
    }
    for raw_line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw_line.split()
        if "Bzy_MHz" in fields:
            header = fields
            for column in wanted & set(header):
                values.setdefault(column, [])
            continue
        if header is None or len(fields) != len(header):
            continue
        for column, samples in values.items():
            try:
                samples.append(float(fields[header.index(column)]))
            except ValueError:
                pass
    columns = {
        column: {
            "samples": len(samples),
            "mean": sum(samples) / len(samples) if samples else None,
            "min": min(samples) if samples else None,
            "max": max(samples) if samples else None,
        }
        for column, samples in values.items()
    }
    return {
        "rows": max((entry["samples"] for entry in columns.values()), default=0),
        "columns": columns,
    }


def parse_thread_snapshots(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {"threads": [], "error": "file is missing"}
    ThreadIdentity = tuple[int, int | None]
    first: dict[ThreadIdentity, tuple[int, dict[str, Any]]] = {}
    last: dict[ThreadIdentity, tuple[int, dict[str, Any]]] = {}
    sample_count = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        timestamp = int(sample["monotonic_ns"])
        sample_count += 1
        for row in sample.get("threads", []):
            tid = int(row["tid"])
            starttime = row.get("starttime_ticks")
            identity = (
                tid,
                int(starttime) if starttime is not None else None,
            )
            first.setdefault(identity, (timestamp, row))
            last[identity] = (timestamp, row)
    clock_ticks = float(os.sysconf("SC_CLK_TCK"))
    rows: list[dict[str, Any]] = []
    for identity in sorted(
        last,
        key=lambda value: (
            value[0],
            value[1] if value[1] is not None else -1,
        ),
    ):
        start_timestamp, start = first[identity]
        end_timestamp, end = last[identity]
        window_ns = max(0, end_timestamp - start_timestamp)
        runtime_ns = max(0, int(end["runtime_ns"]) - int(start["runtime_ns"]))
        delay_ns = max(
            0,
            int(end["runqueue_delay_ns"]) - int(start["runqueue_delay_ns"]),
        )
        cpu_ticks = max(
            0,
            int(end["utime_ticks"])
            + int(end["stime_ticks"])
            - int(start["utime_ticks"])
            - int(start["stime_ticks"]),
        )
        rows.append(
            {
                "tid": identity[0],
                "starttime_ticks": identity[1],
                "comm": end.get("comm"),
                "window_ms": window_ns / 1e6,
                "cpu_ms": cpu_ticks * 1000.0 / clock_ticks,
                "runtime_ms": runtime_ns / 1e6,
                "runqueue_delay_ms": delay_ns / 1e6,
                "timeslices": max(
                    0,
                    int(end["timeslices"]) - int(start["timeslices"]),
                ),
                "runtime_fraction": runtime_ns / window_ns if window_ns else None,
                "runqueue_delay_fraction": (
                    delay_ns / (runtime_ns + delay_ns) if runtime_ns + delay_ns else 0.0
                ),
                "last_processor": end.get("processor"),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["cpu_ms"]),
            int(row["tid"]),
            int(row["starttime_ticks"] or -1),
        )
    )
    return {"samples": sample_count, "threads": rows}


def parse_gpu_dmon(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {"samples": 0, "columns": {}, "error": "file is missing"}
    header: list[str] | None = None
    values: dict[str, list[float]] = defaultdict(list)
    for raw_line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        if fields[0] == "#" and len(fields) > 2 and fields[1] == "gpu":
            header = fields[1:]
            continue
        if fields[0].startswith("#") or header is None or len(fields) != len(header):
            continue
        for index, name in enumerate(header):
            if name == "gpu":
                continue
            try:
                values[name].append(float(fields[index]))
            except ValueError:
                continue
    columns = {
        name: {
            "samples": len(samples),
            "mean": statistics.mean(samples) if samples else None,
            "min": min(samples) if samples else None,
            "max": max(samples) if samples else None,
            "zero_fraction": (
                sum(value == 0 for value in samples) / len(samples) if samples else None
            ),
        }
        for name, samples in values.items()
    }
    return {
        "samples": max((entry["samples"] for entry in columns.values()), default=0),
        "columns": columns,
    }


def parse_cpu_frequency(
    path: str | Path,
    *,
    start_monotonic_ns: int | None = None,
    stop_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {"samples": 0, "error": "file is missing"}
    all_samples = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_samples.sort(key=lambda sample: int(sample["monotonic_ns"]))
    if start_monotonic_ns is None and stop_monotonic_ns is None:
        samples = all_samples
    else:
        first_index = 0
        if start_monotonic_ns is not None:
            preceding = [
                index
                for index, sample in enumerate(all_samples)
                if int(sample["monotonic_ns"]) <= start_monotonic_ns
            ]
            first_index = preceding[-1] if preceding else 0
        last_index = len(all_samples) - 1
        if stop_monotonic_ns is not None:
            following = [
                index
                for index, sample in enumerate(all_samples)
                if int(sample["monotonic_ns"]) >= stop_monotonic_ns
            ]
            last_index = following[0] if following else len(all_samples) - 1
        samples = (
            all_samples[first_index : last_index + 1]
            if all_samples and first_index <= last_index
            else []
        )
    frequencies_mhz: list[float] = []
    busy_frequency_sum = 0.0
    busy_ticks_sum = 0
    for sample in samples:
        for row in sample.get("cpus", []):
            frequency = row.get("frequency_khz")
            if isinstance(frequency, (int, float)):
                frequencies_mhz.append(float(frequency) / 1000.0)
    for before, after in zip(samples, samples[1:], strict=False):
        first = {int(row["cpu"]): row for row in before.get("cpus", [])}
        last = {int(row["cpu"]): row for row in after.get("cpus", [])}
        for cpu in first.keys() & last.keys():
            total = int(last[cpu]["total_ticks"]) - int(first[cpu]["total_ticks"])
            idle = int(last[cpu]["idle_ticks"]) - int(first[cpu]["idle_ticks"])
            busy = max(0, total - idle)
            frequency = last[cpu].get("frequency_khz")
            if busy and isinstance(frequency, (int, float)):
                busy_ticks_sum += busy
                busy_frequency_sum += busy * float(frequency) / 1000.0
    observed_cpus = sorted(
        {int(row["cpu"]) for sample in samples for row in sample.get("cpus", [])}
    )
    requested_cpus = next(
        (
            sample.get("requested_cpus")
            for sample in samples
            if "requested_cpus" in sample
        ),
        None,
    )
    sampled_frequency = {
        "mean": statistics.mean(frequencies_mhz) if frequencies_mhz else None,
        "min": min(frequencies_mhz) if frequencies_mhz else None,
        "max": max(frequencies_mhz) if frequencies_mhz else None,
    }
    busy_weighted = busy_frequency_sum / busy_ticks_sum if busy_ticks_sum else None
    sample_timestamps = [int(sample["monotonic_ns"]) for sample in samples]
    first_timestamp = sample_timestamps[0] if sample_timestamps else None
    last_timestamp = sample_timestamps[-1] if sample_timestamps else None
    brackets_start = start_monotonic_ns is None or (
        first_timestamp is not None and first_timestamp <= start_monotonic_ns
    )
    brackets_stop = stop_monotonic_ns is None or (
        last_timestamp is not None and last_timestamp >= stop_monotonic_ns
    )
    interior_samples = sum(
        1
        for timestamp in sample_timestamps
        if (start_monotonic_ns is None or timestamp >= start_monotonic_ns)
        and (stop_monotonic_ns is None or timestamp <= stop_monotonic_ns)
    )
    return {
        "samples": len(samples),
        "interior_samples": interior_samples,
        "total_file_samples": len(all_samples),
        "cpu_samples": len(frequencies_mhz),
        "scope": {
            "kind": (
                "selected_cpus" if requested_cpus is not None else "all_online_cpus"
            ),
            "requested_cpus": requested_cpus,
            "observed_cpus": observed_cpus,
        },
        "sampled_scaling_frequency_mhz": sampled_frequency,
        "busy_weighted_sampled_scaling_frequency_mhz": busy_weighted,
        # Compatibility aliases. These values are sampled scaling_cur_freq,
        # never APERF/MPERF-derived effective frequency.
        "frequency_mhz": sampled_frequency,
        "busy_weighted_frequency_mhz": busy_weighted,
        "busy_ticks": busy_ticks_sum,
        "start_monotonic_ns": start_monotonic_ns,
        "stop_monotonic_ns": stop_monotonic_ns,
        "coverage": {
            "first_sample_monotonic_ns": first_timestamp,
            "last_sample_monotonic_ns": last_timestamp,
            "brackets_start": brackets_start,
            "brackets_stop": brackets_stop,
        },
    }


def write_json(path: str | Path, data: Any) -> str:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial JSON artifact exists: {partial}")
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    return str(destination)


__all__ = [
    "CpuFrequencyCollector",
    "ManagedCollector",
    "NsysSystemWideCpuCollector",
    "ThreadSnapshotCollector",
    "capture_command",
    "collect_static_manifest",
    "find_processes_with_thread_comms",
    "gpu_dmon_collector",
    "parse_gpu_dmon",
    "parse_cpu_frequency",
    "parse_perf_stat",
    "parse_thread_snapshots",
    "parse_turbostat",
    "perf_sched_collector",
    "perf_stat_collector",
    "inspect_nsys_cpu_sqlite",
    "psi_delta",
    "read_process_cgroup_psi",
    "read_psi",
    "read_thread_snapshot",
    "summarize_thread_snapshot_delta",
    "turbostat_collector",
    "write_json",
]
