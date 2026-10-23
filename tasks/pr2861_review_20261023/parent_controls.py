"""Run the PR's functional regression cases against the exact parent's components."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_context

import sglang_omni.models.qwen3_omni.components.talker_prefill as talker_prefill
import sglang_omni.preprocessing.audio as audio
from sglang_omni.models.qwen3_omni.request_builders import compute_mrope_positions
from tests.unit_test.qwen3_omni.test_mrope_positions import (
    video_span,
    thinker_config_ns,
)

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

parent_builder = parent_module(
    "sglang_omni/models/qwen3_omni/request_builders.py",
    "sglang_omni.models.qwen3_omni.review2861_parent_builders",
)
tokens = [10] + video_span([8, 16, 16]) + [11, 12]
model_inputs = {
    "video_grid_thw": torch.tensor([[8, 16, 16]]),
    "video_second_per_grid": torch.tensor([1.1611]),
}
decode_positions = {}
for label, builder in (
    ("parent", parent_builder.compute_mrope_positions),
    ("head", compute_mrope_positions),
):
    positions, delta = builder(torch.tensor(tokens), model_inputs, thinker_config_ns())
    multimodal = MultimodalInputs(mm_items=[])
    multimodal.mrope_positions = positions
    multimodal.mrope_position_delta = delta
    forward_batch = SimpleNamespace(
        seq_lens_cpu=torch.tensor([len(tokens) + 1]),
        seq_lens=torch.tensor([len(tokens) + 1], device="cuda"),
    )
    with get_context().override_server_args():
        ForwardBatch._compute_mrope_positions_decode(
            forward_batch,
            SimpleNamespace(device="cuda"),
            SimpleNamespace(multimodal_inputs=[multimodal]),
        )
    decode_positions[label] = int(forward_batch.mrope_positions[0, 0])
expected_position = int(positions.max()) + 1
assert decode_positions["parent"] == expected_position + 1
assert decode_positions["head"] == expected_position
print("FAST_CUDA_DECODE_GAP_CONFIRMED", decode_positions, "expected", expected_position)
