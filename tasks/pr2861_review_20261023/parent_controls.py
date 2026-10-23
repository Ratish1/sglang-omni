"""Run the PR's functional regression cases against the exact parent's components."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import sglang_omni.models.qwen3_omni.components.talker_prefill as talker_prefill
import sglang_omni.preprocessing.audio as audio

PARENT = "0ee9d47dffe16a5b1cc8d1dcbfbb04769884107f"
OUTPUT_DIRECTORY = Path("/data/review2861")


def parent_module(relative_path: str, module_name: str) -> ModuleType:
    source = subprocess.check_output(
        ["git", "show", f"{PARENT}:{relative_path}"], text=True
    )
    path = OUTPUT_DIRECTORY / f"{module_name.rsplit('.', 1)[-1]}.py"
    path.write_text(source)
    specification = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


class ParentFailures:
    def __init__(self) -> None:
        self.failures: dict[str, str] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call" and report.failed:
            self.failures[report.nodeid] = str(report.longrepr)
        else:
            pass


parent_audio = parent_module(
    "sglang_omni/preprocessing/audio.py",
    "sglang_omni.preprocessing.review2861_parent_audio",
)
parent_talker = parent_module(
    "sglang_omni/models/qwen3_omni/components/talker_prefill.py",
    "sglang_omni.models.qwen3_omni.components.review2861_parent_talker",
)
audio.AudioMediaIO = parent_audio.AudioMediaIO
talker_prefill.TalkerPrefillBuilder = parent_talker.TalkerPrefillBuilder
recorder = ParentFailures()
status = pytest.main(
    [
        "-q",
        "--disable-warnings",
        "tests/unit_test/preprocessing/test_audio.py::test_audio_at_another_rate_loads_as_the_reference_loader",
        "tests/unit_test/qwen3_omni/test_talker_prefill_embed_cache.py::test_multimodal_prompt_uses_captured_thinker_states",
    ],
    plugins=[recorder],
)
assert status == 1
assert len(recorder.failures) == 2
assert all("AssertionError" in failure for failure in recorder.failures.values())
print("EXPECTED_PARENT_ASSERTION_FAILURES_CONFIRMED", sorted(recorder.failures))
