# SPDX-License-Identifier: Apache-2.0
"""Linux host evidence collectors for SGLang-Omni profiling runs."""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

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
    completed = subprocess.run(
        [executable, *argv[1:]],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
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
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,"
            "temperature.gpu,power.limit",
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
    return {
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
                "/sys/devices/system/cpu/cpu0/cpufreq/" "energy_performance_preference"
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


def read_psi(root: str | Path = "/proc/pressure") -> dict[str, Any]:
    snapshot: dict[str, Any] = {"captured_monotonic_ns": time.monotonic_ns()}
    for resource in ("cpu", "memory", "io"):
        path = Path(root) / resource
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


def perf_stat_collector(
    *,
    pid: int,
    output_path: str | Path,
    events: Sequence[str] = _PERF_BASE_EVENTS,
    interval_ms: int = 1000,
) -> ManagedCollector:
    return ManagedCollector(
        name="perf_stat",
        argv=[
            "perf",
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
) -> ManagedCollector:
    return ManagedCollector(
        name="perf_sched",
        argv=[
            "perf",
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
        for column in values:
            try:
                values[column].append(float(fields[header.index(column)]))
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


def write_json(path: str | Path, data: Any) -> str:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return str(destination)


__all__ = [
    "ManagedCollector",
    "capture_command",
    "collect_static_manifest",
    "perf_sched_collector",
    "perf_stat_collector",
    "parse_perf_stat",
    "parse_turbostat",
    "psi_delta",
    "read_psi",
    "turbostat_collector",
    "write_json",
]
