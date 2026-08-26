"""Shared process and HTTP helpers for router qualification scripts."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_PROXY_KEYS = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}


def local_process_env() -> dict[str, str]:
    """Return the environment without proxy variables for loopback-only work."""
    return {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in _PROXY_KEYS
    }


def benchmark_process_env() -> dict[str, str]:
    """Preserve external access while forcing benchmark traffic onto loopback."""
    environment = dict(os.environ)
    no_proxy = environment.get("NO_PROXY", environment.get("no_proxy", ""))
    entries = [item.strip() for item in no_proxy.split(",") if item.strip()]
    for host in ("127.0.0.1", "localhost"):
        if host not in entries:
            entries.append(host)
    environment["NO_PROXY"] = ",".join(entries)
    environment["no_proxy"] = environment["NO_PROXY"]
    return environment


def fetch_json(url: str, timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        with _LOCAL_OPENER.open(url, timeout=timeout) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError):
        return None
    try:
        value = json.loads(payload)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def fetch_text(url: str, timeout: float = 2.0) -> str | None:
    try:
        with _LOCAL_OPENER.open(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return None


def request_counters(metrics: str | None) -> dict[str, float]:
    if metrics is None:
        return {}
    counters: dict[str, float] = {}
    for raw_line in metrics.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            series, value_text = line.rsplit(None, 1)
            name = series.split("{", 1)[0]
            value = float(value_text)
        except (ValueError, IndexError):
            continue
        lowered = name.lower()
        if "request" in lowered and lowered.endswith(("_total", "_count")):
            counters[series] = value
    return counters


def counter_moved(before: dict[str, float], after: dict[str, float]) -> bool | None:
    shared = before.keys() & after.keys()
    if not shared:
        return None
    return any(after[key] > before[key] for key in shared)


def wait_http(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with _LOCAL_OPENER.open(url, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise TimeoutError(f"{url} was not ready within {timeout:.1f}s: {last_error}")


def terminate_process_group(
    process: subprocess.Popen[Any], grace_s: float = 15.0
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5.0)


def substitute_placeholders(
    command: Sequence[str], values: dict[str, str]
) -> list[str]:
    rendered: list[str] = []
    for argument in command:
        value = argument
        for key, replacement in values.items():
            value = value.replace("{" + key + "}", replacement)
        rendered.append(value)
    return rendered


def assert_zero_in_flight(diagnostics: dict[str, Any]) -> None:
    entries: list[tuple[str, Any]] = []
    admission = diagnostics.get("admission")
    if isinstance(admission, list):
        entries.extend(
            (f"admission[{index}]", item) for index, item in enumerate(admission)
        )
    workers = diagnostics.get("workers")
    if isinstance(workers, list):
        for worker_index, worker in enumerate(workers):
            if isinstance(worker, dict) and isinstance(worker.get("capacity"), list):
                entries.extend(
                    (f"workers[{worker_index}].capacity[{index}]", item)
                    for index, item in enumerate(worker["capacity"])
                )
    if not entries:
        raise AssertionError("Rust diagnostics contain no admission/capacity entries")
    nonzero: list[str] = []
    for path, item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("in_flight"), int):
            raise AssertionError(  # noqa: TRY004 -- this validates a test assertion
                f"{path}.in_flight is missing or not an integer"
            )
        if item["in_flight"] != 0:
            nonzero.append(f"{path}={item['in_flight']}")
    if nonzero:
        raise AssertionError("nonzero Rust ownership: " + ", ".join(nonzero))


def assert_workers_healthy(snapshot: dict[str, Any], candidate: str) -> None:
    workers = snapshot.get("workers")
    if not isinstance(workers, list) or len(workers) != 2:
        raise AssertionError("post-trial diagnostics must contain exactly two workers")
    if candidate == "rust":
        unhealthy = [
            worker.get("worker_id", "unknown")
            for worker in workers
            if not isinstance(worker, dict)
            or worker.get("health") != "healthy"
            or worker.get("disposition") != "serving"
        ]
        if snapshot.get("ready") is not True or unhealthy:
            raise AssertionError(
                f"Rust router/workers are not ready and healthy: {unhealthy}"
            )
        return
    if snapshot.get("healthy_workers") != 2 or snapshot.get("routable_workers") != 2:
        raise AssertionError("Python router does not have two healthy routable workers")
    unhealthy = [
        worker.get("display_id", "unknown")
        for worker in workers
        if not isinstance(worker, dict)
        or worker.get("health_state") != "healthy"
        or worker.get("routable") is not True
        or worker.get("active_requests") != 0
    ]
    if unhealthy:
        raise AssertionError(f"Python workers are not idle and healthy: {unhealthy}")


@dataclass
class ProcessMetrics:
    available: bool
    cpu_seconds: float | None = None
    peak_rss_bytes: int | None = None
    samples: int = 0
    reason: str | None = None


class ProcessGroupSampler:
    """Sample aggregate Linux /proc CPU and RSS for one process group."""

    def __init__(self, process_group_id: int, interval_s: float = 0.1) -> None:
        self.process_group_id = process_group_id
        self.interval_s = interval_s
        self.metrics = ProcessMetrics(available=Path("/proc").is_dir())
        if not self.metrics.available:
            self.metrics.reason = "Linux /proc is unavailable"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._first_ticks: int | None = None
        self._last_ticks: int | None = None

    def start(self) -> None:
        if not self.metrics.available:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> ProcessMetrics:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        if self._first_ticks is not None and self._last_ticks is not None:
            ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            self.metrics.cpu_seconds = (
                max(0, self._last_ticks - self._first_ticks) / ticks_per_second
            )
        return self.metrics

    def _run(self) -> None:
        page_size = os.sysconf("SC_PAGE_SIZE")
        while not self._stop.is_set():
            ticks = 0
            rss_pages = 0
            found = False
            try:
                proc_entries = list(Path("/proc").iterdir())
            except OSError as exc:
                self.metrics.available = False
                self.metrics.reason = str(exc)
                return
            for entry in proc_entries:
                if not entry.name.isdigit():
                    continue
                try:
                    process_group, process_ticks, process_rss = parse_proc_stat(
                        (entry / "stat").read_text(encoding="utf-8")
                    )
                    if process_group != self.process_group_id:
                        continue
                    ticks += process_ticks
                    rss_pages += process_rss
                    found = True
                except (OSError, ValueError, IndexError):
                    continue
            if found:
                if self._first_ticks is None:
                    self._first_ticks = ticks
                self._last_ticks = ticks
                rss_bytes = rss_pages * page_size
                self.metrics.peak_rss_bytes = max(
                    self.metrics.peak_rss_bytes or 0, rss_bytes
                )
                self.metrics.samples += 1
            self._stop.wait(self.interval_s)


def parse_proc_stat(stat: str) -> tuple[int, int, int]:
    """Return process group, CPU ticks, and RSS pages from Linux proc stat."""
    _prefix, separator, suffix = stat.rpartition(")")
    if not separator:
        raise ValueError("process stat has no command terminator")
    fields = suffix.split()
    return int(fields[2]), int(fields[11]) + int(fields[12]), max(0, int(fields[21]))


def metrics_dict(metrics: ProcessMetrics) -> dict[str, Any]:
    return {
        "available": metrics.available,
        "cpu_seconds": metrics.cpu_seconds,
        "peak_rss_bytes": metrics.peak_rss_bytes,
        "samples": metrics.samples,
        "reason": metrics.reason,
    }
