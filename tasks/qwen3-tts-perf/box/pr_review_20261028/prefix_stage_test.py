"""Calls the stage step-down case of our predictor suite against the tree in PYTHONPATH (main has no step-down), to
show it fails before the fix. main lacks REDUCED_NUM_STAGES, the one name the suite imports that it does not define,
so that name is set first; everything the case runs is main's.

usage: PYTHONPATH=MAIN python3 prefix_stage_test.py OURS/tests/unit_test/qwen3_omni/test_predictor_kernels.py
"""

import importlib.util
import sys

import pytest
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig, PhaseConfig
from sglang.srt.runtime_context import get_context

from sglang_omni.utils import predictor_layers

predictor_layers.REDUCED_NUM_STAGES = 2
spec = importlib.util.spec_from_file_location("ours_predictor_suite", sys.argv[1])
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
print(f"predictor_layers from {predictor_layers.__file__}", flush=True)
with get_context().override_server_args(
    cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(backend=Backend.DISABLED))
):
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            suite.test_launches_beyond_the_shared_memory_pipeline_fewer_stages_with_the_same_bits(monkeypatch)
            print("passed", flush=True)
        except BaseException as error:
            print(f"failed: {type(error).__name__}: {error}", flush=True)
