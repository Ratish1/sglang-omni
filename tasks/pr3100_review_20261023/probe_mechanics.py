"""Review-only controls for PR 3100; execute in its Radix container."""

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def ipc_probe(mode: str) -> None:
    test_path = REPOSITORY / "tests/unit_test/cli/test_serve_exit.py"
    syntax = ast.parse(test_path.read_text())
    source = next(
        ast.literal_eval(statement.value)
        for statement in syntax.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SEND_CUDA_TENSOR"
            for target in statement.targets
        )
    )
    instrumentation = r"""
import subprocess
from pathlib import Path
from multiprocessing import reduction
import torch.multiprocessing.reductions as reducers
import sglang_omni.utils.process as exit_helpers

probe_mode, record_directory = sys.argv[1:]
record_directory = Path(record_directory)
actual_reducer = reducers.reduce_tensor
def record_tensor(tensor):
    function, arguments = actual_reducer(tensor)
    if function is reducers.rebuild_cuda_tensor:
        reference_path = "/dev/shm/" + os.fsdecode(arguments[11]).lstrip("/")
        (record_directory / "actual_reference.json").write_text(json.dumps(reference_path))
    return function, arguments
reduction.register(torch.Tensor, record_tensor)
if probe_mode == "missing-cleanup":
    exit_helpers.release_cuda_ipc_file = lambda: None

actual_glob = glob.glob
snapshot_count = 0
def snapshot(pattern):
    global snapshot_count
    if snapshot_count == 0 and probe_mode == "stale":
        foreign = f"/dev/shm/torch_{os.getpid()}_3900310031_999999991"
        (record_directory / "foreign.json").write_text(json.dumps(foreign))
        subprocess.run([sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch(exist_ok=False)", foreign], check=True)
    observed = actual_glob(pattern)
    if snapshot_count == 0 and probe_mode == "overlap":
        foreign = f"/dev/shm/torch_{os.getpid()}_3900310031_999999992"
        (record_directory / "foreign.json").write_text(json.dumps(foreign))
        subprocess.run([sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch(exist_ok=False)", foreign], check=True)
    snapshot_count += 1
    return observed
glob.glob = snapshot
"""
    source = source.replace("import glob, json, os", "import glob, json, os, sys")
    source = source.replace(
        "def receive(queue, done):", instrumentation + "\ndef receive(queue, done):"
    )
    with tempfile.TemporaryDirectory(prefix="review3100-ipc-") as temporary:
        directory = Path(temporary)
        script = directory / "sender.py"
        script.write_text(source)
        try:
            sender = subprocess.run(
                [sys.executable, str(script), mode, str(directory)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert sender.returncode == 0, sender.stderr
            observed = json.loads(sender.stdout)
            actual = json.loads((directory / "actual_reference.json").read_text())
            remaining = [path for path in observed if Path(path).exists()]
            actual_remains = Path(actual).exists()
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "actual_reference": actual,
                        "observed": observed,
                        "remaining": remaining,
                        "actual_reference_remains": actual_remains,
                    }
                )
            )
            assert observed and actual in observed
            if mode == "missing-cleanup":
                assert actual_remains and remaining == [actual]
            elif mode == "overlap":
                foreign = json.loads((directory / "foreign.json").read_text())
                assert not actual_remains and remaining == [foreign]
            else:
                assert not actual_remains and remaining == []
        finally:
            for name in ("actual_reference.json", "foreign.json"):
                record = directory / name
                if record.exists():
                    Path(json.loads(record.read_text())).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["normal", "stale", "overlap", "missing-cleanup", "dots-without-guard"],
    )
    options = parser.parse_args()
    if options.mode == "dots-without-guard":
        program = """
from contextlib import nullcontext
import pytest
import sglang_omni.models.dots_tts.vocoder_cuda_graph as graphs
graphs.freeze_gc = lambda **options: nullcontext()
raise SystemExit(pytest.main(["-q", "--tb=short", "tests/unit_test/dots_tts/test_vocoder_cuda_graph.py::test_capture_completes_when_a_collection_runs_while_an_earlier_pool_is_dead"]))
"""
        completed = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=180
        )
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        assert completed.returncode == 1
        assert (
            "stream capture status" in completed.stdout
            or "previous error during capture" in completed.stdout
        )
        assert (
            "operation not permitted when stream is capturing"
            in completed.stdout + completed.stderr
        )
    else:
        ipc_probe(options.mode)


if __name__ == "__main__":
    main()
