# SPDX-License-Identifier: Apache-2.0
"""The builder hands out a resolved ServerArgs.

A real config.json is needed because the resolution pipeline returns before
the cuda graph handler on a dummy model path. The device is pinned to cuda
the way upstream's own resolution tests pin it, so the record resolves the
same on an accelerator-less host.
"""

from __future__ import annotations

import json
from pathlib import Path

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.model_executor.cuda_graph_config import Backend

from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)

_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 128,
    "intermediate_size": 256,
    "max_position_embeddings": 2048,
    "model_type": "llama",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "bfloat16",
    "vocab_size": 1000,
}


def _checkpoint(tmp_path: Path) -> str:
    (tmp_path / "config.json").write_text(json.dumps(_MINI_CONFIG))
    return str(tmp_path)


def test_builder_record_is_resolved_with_the_cuda_graph_config_declared(
    tmp_path: Path,
) -> None:
    server_args = build_sglang_server_args(
        _checkpoint(tmp_path), context_length=2048, device="cuda"
    )

    assert server_args._resolution_finished is True
    assert server_args.cuda_graph_config is None
    cuda_graph_config = resolution_result(server_args, "cuda_graph_config")
    assert cuda_graph_config.prefill.backend == Backend.DISABLED
