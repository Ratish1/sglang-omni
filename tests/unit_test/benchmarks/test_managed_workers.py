# SPDX-License-Identifier: Apache-2.0
"""GPU placement, startup failure, and shutdown contracts for managed workers."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
from concurrent.futures import Future

import pytest

import benchmarks.benchmarker.workers as managed_workers
from benchmarks.benchmarker.workers import (
    ManagedWorkerProcess,
    build_gpu_assignments,
    start_workers,
    wait_for_workers,
)


def test_gpu_assignment_uses_explicit_worker_gpu_ids() -> None:
    assert build_gpu_assignments(2, 2, worker_gpu_ids=["0,1", "2,3"]) == ["0,1", "2,3"]


def test_gpu_assignment_groups_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert build_gpu_assignments(2, 2) == ["4,5", "6,7"]


def test_gpu_assignment_allows_default_process_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(managed_workers, "infer_available_cuda_devices", lambda: [])
    assert build_gpu_assignments(1, 1) is None


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> None:
        return None


@pytest.fixture
def created_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[FakeProcess, list[str], dict[str, str]]]:
    created: list[tuple[FakeProcess, list[str], dict[str, str]]] = []

    def fake_popen(
        command: list[str],
        *,
        env: dict[str, str],
        stdout: None,
        stderr: None,
        cwd: None,
        start_new_session: bool,
    ) -> FakeProcess:
        assert start_new_session
        process = FakeProcess(12345 + len(created))
        created.append((process, command, env))
        return process

    monkeypatch.setattr(managed_workers.shutil, "which", lambda command: command)
    monkeypatch.setattr(managed_workers, "port_is_available", lambda port, host: True)
    monkeypatch.setattr(managed_workers.subprocess, "Popen", fake_popen)
    return created


def test_worker_command_preserves_model_arguments(
    created_workers: list[tuple[FakeProcess, list[str], dict[str, str]]],
) -> None:
    workers = start_workers(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        model_name="qwen3-omni",
        worker_host="127.0.0.1",
        worker_ports=[8011],
        num_gpus_per_worker=2,
        worker_gpu_ids=["0,1"],
        worker_extra_args="--mem-fraction-static 0.6 --thinker-tp-size 2 --thinker-gpus '0,1'",
    )
    process, command, environment = created_workers[0]
    assert command[:8] == [
        "sgl-omni",
        "serve",
        "--model-path",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "--host",
        "127.0.0.1",
        "--port",
        "8011",
    ]
    assert command[command.index("--model-name") + 1] == "qwen3-omni"
    assert command[command.index("--mem-fraction-static") + 1] == "0.6"
    assert command[command.index("--thinker-tp-size") + 1] == "2"
    assert command[command.index("--thinker-gpus") + 1] == "0,1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert workers[0].process_group_id == process.pid


@pytest.mark.parametrize("interrupted", [False, True])
def test_startup_failure_cleans_up_workers(
    monkeypatch: pytest.MonkeyPatch,
    created_workers: list[tuple[FakeProcess, list[str], dict[str, str]]],
    interrupted: bool,
) -> None:
    terminated: list[ManagedWorkerProcess] = []

    def fail_health(worker: ManagedWorkerProcess, deadline: float) -> None:
        raise TimeoutError("worker did not become healthy")

    def interrupt_wait(futures: set[Future[None]], *, return_when: str) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(managed_workers, "stop_managed_workers", terminated.extend)
    monkeypatch.setattr(managed_workers, "wait_for_worker_health", fail_health)
    if interrupted:
        monkeypatch.setattr(managed_workers, "wait", interrupt_wait)
        expected_error = KeyboardInterrupt
    else:
        expected_error = TimeoutError
    workers = start_workers(
        model_path="model",
        model_name="model",
        worker_host="127.0.0.1",
        worker_ports=[8011],
        num_gpus_per_worker=1,
        worker_gpu_ids=["7"],
    )
    with pytest.raises(expected_error):
        wait_for_workers(workers, 1)
    assert [worker.process for worker in terminated] == [created_workers[0][0]]
    assert workers == []


def test_workers_wait_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
    created_workers: list[tuple[FakeProcess, list[str], dict[str, str]]],
) -> None:
    terminated: list[ManagedWorkerProcess] = []
    waiting_urls: list[str] = []
    waiting_lock = threading.Lock()
    both_waiting = threading.Event()

    def wait_health(worker: ManagedWorkerProcess, deadline: float) -> None:
        with waiting_lock:
            waiting_urls.append(worker.url)
            if len(waiting_urls) == 2:
                both_waiting.set()
            else:
                pass
        assert both_waiting.wait(timeout=2)
        if worker.port == 8012:
            raise RuntimeError("worker 2 failed")
        else:
            pass

    monkeypatch.setattr(managed_workers, "wait_for_worker_health", wait_health)
    monkeypatch.setattr(managed_workers, "stop_managed_workers", terminated.extend)
    workers = start_workers(
        model_path="model",
        model_name="model",
        worker_host="127.0.0.1",
        worker_ports=[8011, 8012],
        num_gpus_per_worker=1,
        worker_gpu_ids=["0", "1"],
    )
    with pytest.raises(RuntimeError, match="worker 2 failed"):
        wait_for_workers(workers, 10)
    assert set(waiting_urls) == {"http://127.0.0.1:8011", "http://127.0.0.1:8012"}
    assert [worker.process for worker in terminated] == [
        created[0] for created in created_workers
    ]
    assert workers == []


def test_managed_shutdown_signals_only_worker_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int, int]] = []

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def send_signal(self, sig: int) -> None:
            events.append(("parent", self.pid, sig))

        def wait(self, timeout: float) -> int:
            assert timeout >= 0
            events.append(("wait", self.pid, 0))
            return 0

    def check_group(process_group_id: int, sig: int) -> None:
        events.append(("group", process_group_id, sig))
        raise ProcessLookupError()

    monkeypatch.setattr(managed_workers.os, "killpg", check_group)
    workers = [
        managed_workers.ManagedWorkerProcess(
            url=f"http://127.0.0.1:{port}",
            port=port,
            cuda_visible_devices=None,
            process=FakeProcess(port),
            process_group_id=port,
        )
        for port in (8011, 8012)
    ]

    managed_workers.stop_managed_workers(workers)

    assert events == [
        ("parent", 8011, signal.SIGINT),
        ("parent", 8012, signal.SIGINT),
        ("wait", 8011, 0),
        ("group", 8011, 0),
        ("wait", 8012, 0),
        ("group", 8012, 0),
    ]


@pytest.mark.parametrize("parent_timed_out", [False, True])
def test_managed_shutdown_kills_remaining_group(
    monkeypatch: pytest.MonkeyPatch, parent_timed_out: bool
) -> None:
    group_signals: list[int] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.wait_count = 0
            self.signals: list[int] = []

        def send_signal(self, sig: int) -> None:
            self.signals.append(sig)

        def wait(self, timeout: float) -> int:
            self.wait_count += 1
            if parent_timed_out and self.wait_count == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            return 0

    def signal_group(process_group_id: int, sig: int) -> None:
        assert process_group_id == 8011
        group_signals.append(sig)

    def group_exited(process_group_id: int) -> bool:
        assert process_group_id == 8011
        return False

    monkeypatch.setattr(managed_workers.os, "killpg", signal_group)
    monkeypatch.setattr(managed_workers, "process_group_has_live_members", group_exited)
    process = FakeProcess()
    worker = managed_workers.ManagedWorkerProcess(
        url="http://127.0.0.1:8011",
        port=8011,
        cuda_visible_devices=None,
        process=process,
        process_group_id=8011,
    )

    managed_workers.stop_managed_workers([worker])

    assert process.signals == [signal.SIGINT]
    assert process.wait_count == 2
    assert group_signals == (
        [signal.SIGKILL] if parent_timed_out else [0, signal.SIGKILL]
    )


def test_managed_shutdown_waits_for_orphaned_worker_child() -> None:
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print('ready', flush=True)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert parent.stdout is not None
        assert parent.stdout.readline().strip() == "ready"
        parent.wait(timeout=5)
        assert managed_workers.process_group_has_live_members(parent.pid)

        worker = managed_workers.ManagedWorkerProcess(
            url="http://127.0.0.1:8011",
            port=8011,
            cuda_visible_devices=None,
            process=parent,
            process_group_id=parent.pid,
        )
        managed_workers.stop_managed_workers([worker])

        assert not managed_workers.process_group_has_live_members(parent.pid)
    finally:
        try:
            managed_workers.os.killpg(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent.wait(timeout=5)
        if parent.stdout is not None:
            parent.stdout.close()
