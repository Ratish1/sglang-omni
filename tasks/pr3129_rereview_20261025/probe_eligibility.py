"""Check real factory eligibility before the heavyweight model-loading boundary."""

import contextlib
import json
from unittest.mock import patch

import torch

from sglang_omni.models.fun_cosyvoice3 import packed_dit, stages


class LoaderReached(Exception):
    pass


def main():
    records = []
    cases = (
        ("cpu_default", "cpu", "bfloat16", None, False, False, False),
        ("cpu_positive_cap", "cpu", "bfloat16", 2.0, False, False, False),
        ("cuda_float32", "cuda:0", "float32", None, False, False, False),
        ("cuda_disabled_cache", "cuda:0", "bfloat16", 0.0, False, False, False),
        ("cuda_tensor_rt", "cuda:0", "bfloat16", None, True, False, False),
        ("cuda_no_fa3", "cuda:0", "bfloat16", None, False, True, False),
        ("cuda_default", "cuda:0", "bfloat16", None, False, False, True),
        ("cuda_float16", "cuda:0", "float16", None, False, False, True),
    )
    for name, device, dtype, cache_size, tensor_rt, no_fa3, expects_queries in cases:
        with (
            patch.object(stages, "resolve_checkpoint", return_value="/checkpoint"),
            patch.object(
                stages, "load_cosyvoice3_flow_hift", side_effect=LoaderReached
            ) as loader,
            patch.object(
                torch.accelerator,
                "get_memory_info",
                wraps=torch.accelerator.get_memory_info,
            ) as memory_query,
            (
                patch.object(packed_dit, "_is_fa3_supported", return_value=False)
                if no_fa3
                else contextlib.nullcontext()
            ),
        ):
            try:
                stages.create_vocoder_executor(
                    "checkpoint",
                    device=device,
                    dtype=dtype,
                    flow_prefix_cache_gb=cache_size,
                    enable_dit_torch_compile=False,
                    enable_flow_cuda_graph=False,
                    enable_flow_prefix_cuda_graph=False,
                    enable_flow_whole_history_cuda_graph=False,
                    enable_hift_cuda_graph=False,
                    enable_flow_estimator_trt=tensor_rt,
                )
            except LoaderReached:
                pass
            assert loader.called, name
            assert bool(memory_query.call_count) == expects_queries, name
            records.append(
                dict(
                    case=name,
                    loader_reached=loader.called,
                    memory_queries=memory_query.call_count,
                    outcome="passed",
                )
            )
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
