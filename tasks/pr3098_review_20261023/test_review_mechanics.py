"""Review-only process/protocol checks; run remotely, never added to the PR."""

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import httpx
import pytest

from benchmarks.eval.benchmark_asr_seedtts import _fetch_worker_snapshot
from tests.test_model.omni_router_utils import (
    find_available_port_range,
    launch_managed_router,
    router_get_json,
)
from tests.test_model.rust_router_config import CiRouterTopology
from tests.utils import managed_workers

REPOSITORY = Path(__file__).resolve().parents[2]
FAKE_WORKER = r"""#!/usr/bin/env python3
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

signal.signal(signal.SIGINT, signal.default_int_handler)
parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--model-path")
parser.add_argument("--model-name")
parser.add_argument("--host")
parser.add_argument("--port", type=int)
arguments = parser.parse_args()
if arguments.model_name == "startup-failure":
    sys.exit(23)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
record = Path(os.environ["REVIEW_WORKER_DIRECTORY"]) / f"{arguments.port}.json"
record.write_text(json.dumps({"parent": os.getpid(), "child": child.pid,
    "devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "stopped": False}))

class WorkerHTTP(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"healthy")

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = json.dumps({"object": "chat.completion", "model": arguments.model_name,
            "choices": [{"index": 0, "message": {"role": "assistant",
                "content": request["messages"][0]["content"]}, "finish_reason": "stop"}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *arguments):
        pass

try:
    with HTTPServer((arguments.host, arguments.port), WorkerHTTP) as server:
        server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    child.terminate()
    child.wait(timeout=5)
    state = json.loads(record.read_text())
    state["stopped"] = True
    record.write_text(json.dumps(state))
"""


@pytest.fixture
def worker_directory(tmp_path, monkeypatch):
    executable = tmp_path / "sgl-omni"
    executable.write_text(FAKE_WORKER)
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("REVIEW_WORKER_DIRECTORY", str(tmp_path))
    monkeypatch.setenv(
        "SGLANG_OMNI_ROUTER_BIN",
        str(REPOSITORY / "sglang_omni_router/target/debug/sgl-omni-router"),
    )
    return tmp_path


@pytest.mark.parametrize("external", [False, True])
def test_complete_router_topology_and_cleanup(
    tmp_path_factory, worker_directory, external
):
    """Real routing must work, while teardown stops only owned worker processes."""
    externally_owned = []
    if external:
        base_port = find_available_port_range(2)
        externally_owned = managed_workers.start_workers(
            model_path="review-model",
            model_name="review-model",
            worker_host="127.0.0.1",
            worker_ports=[base_port, base_port + 1],
            worker_gpu_ids=managed_workers.build_gpu_assignments(["0", "1"], 2, 1),
        )
        managed_workers.wait_for_workers(externally_owned, 15)
    try:
        with launch_managed_router(
            tmp_path_factory=tmp_path_factory,
            model_path="review-model",
            model_name="review-model",
            worker_extra_args="",
            router_topology=CiRouterTopology.OMNI_TEXT,
            wait_timeout=20,
            external_worker_urls=(
                [worker.url for worker in externally_owned] if external else None
            ),
        ) as handle:
            router_process = handle.proc
            with httpx.Client(trust_env=False) as client:
                for ordinal in range(6):
                    content = f"request-{ordinal}-λ"
                    response = client.post(
                        f"http://127.0.0.1:{handle.port}/v1/chat/completions",
                        json={
                            "model": "review-model",
                            "messages": [{"role": "user", "content": content}],
                        },
                        timeout=5,
                    )
                    response.raise_for_status()
                    assert (
                        response.json()["choices"][0]["message"]["content"] == content
                    )
            snapshot = router_get_json(handle.port, "/diagnostics")
            counts = [
                sum(dispatch["requests"] for dispatch in worker["dispatches"])
                for worker in snapshot["workers"]
            ]
            assert sum(counts) == 6 and all(count > 0 for count in counts)
            benchmark_snapshot = _fetch_worker_snapshot("127.0.0.1", handle.port)
            assert (
                sum(
                    worker["routed_requests"]
                    for worker in benchmark_snapshot["workers"]
                )
                == 6
            )
            if not external:
                records = [
                    json.loads(path.read_text())
                    for path in worker_directory.glob("*.json")
                ]
                assert {record["devices"] for record in records} == {"0", "1"}
        assert router_process.poll() is not None
        if external:
            assert all(worker.process.poll() is None for worker in externally_owned)
        else:
            for path in worker_directory.glob("*.json"):
                state = json.loads(path.read_text())
                assert state["stopped"]
                assert not managed_workers.process_group_has_live_members(
                    state["parent"]
                )
            handle.stop()
    finally:
        managed_workers.stop_managed_workers(externally_owned)


def test_partial_worker_spawn_failure_releases_first_worker(
    worker_directory, monkeypatch
):
    """Failure to spawn replica two must not leave replica one or its children alive."""
    actual_popen = subprocess.Popen
    started = []

    def fail_second(command, **options):
        if command[0] != "sgl-omni":
            return actual_popen(command, **options)
        if len(started) == 1:
            managed_workers.wait_for_worker_health(
                managed_workers.ManagedWorkerProcess(
                    url=f"http://127.0.0.1:{base_port}",
                    process=started[0],
                    process_group_id=started[0].pid,
                ),
                time.monotonic() + 10,
            )
            raise OSError("injected second spawn failure")
        process = actual_popen(command, **options)
        started.append(process)
        return process

    monkeypatch.setattr(managed_workers.subprocess, "Popen", fail_second)
    base_port = find_available_port_range(2)
    with pytest.raises(OSError, match="injected second spawn failure"):
        managed_workers.start_workers(
            model_path="review-model",
            model_name="review-model",
            worker_host="127.0.0.1",
            worker_ports=[base_port, base_port + 1],
            worker_gpu_ids=["0", "1"],
        )
    assert len(started) == 1 and started[0].poll() is not None
    monkeypatch.setattr(managed_workers.subprocess, "Popen", actual_popen)
    assert not managed_workers.process_group_has_live_members(started[0].pid)


def test_failed_health_cleans_entire_topology(tmp_path_factory, worker_directory):
    """A real worker exit during startup must close router and worker groups."""
    with pytest.raises(RuntimeError, match="exited before becoming healthy"):
        with launch_managed_router(
            tmp_path_factory=tmp_path_factory,
            model_path="review-model",
            model_name="startup-failure",
            worker_extra_args="",
            router_topology=CiRouterTopology.OMNI_TEXT,
            wait_timeout=10,
        ):
            pytest.fail("failed workers reached the serving boundary")
    manifests = list(
        tmp_path_factory.getbasetemp().glob("omni_router_cleanup*/router_pgids.txt")
    )
    assert manifests
    for manifest in manifests:
        for group in manifest.read_text().splitlines():
            assert not managed_workers.process_group_has_live_members(int(group))


def test_retirement_in_all_package_manifests():
    """Each hardware package must stop advertising the deleted command/package."""
    manifests = list(REPOSITORY.glob("pyproject*.toml"))
    assert len(manifests) == 6
    for path in manifests:
        manifest = tomllib.loads(path.read_text())
        assert "sgl-omni-router-py" not in manifest["project"]["scripts"]
        assert not any(
            "sglang_omni_router" in value
            for value in manifest["tool"]["setuptools"]["packages"]["find"]["include"]
        )
    distribution = importlib.metadata.distribution("sglang-omni")
    assert not any(
        entry.name == "sgl-omni-router-py" for entry in distribution.entry_points
    )


def test_manual_evaluation_runner_still_depends_on_deleted_module(tmp_path):
    """Confirm the explicitly deferred tool's live default launch fails before workers start."""
    path = REPOSITORY / ".claude/skills/running-eval-suite/runner.py"
    spec = importlib.util.spec_from_file_location("review_eval_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    configuration = runner.load_yaml(path.parent / "models/qwen3-omni/config.yaml")
    profile = runner._resolve_server_profile(
        {}, configuration, configuration["hf_model_id"]
    )
    log_path = tmp_path / "runner.log"
    handle = runner.launch_managed_router_server(
        profile,
        sys.executable,
        port=49199,
        model="review-model",
        gpus_csv="",
        log_path=log_path,
        round_dir=tmp_path,
        repo_root=REPOSITORY,
    )
    try:
        assert handle.popen.wait(timeout=10) != 0
        assert "sglang_omni_router.python" in log_path.read_text()
        assert "ModuleNotFoundError" in log_path.read_text()
    finally:
        handle.stop()
