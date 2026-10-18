"""Exercise the talker attention's KV-store decision with real CUDA RoPE dispatch."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context

from sglang_omni.models.qwen3_omni.components.thinker_model import (
    Qwen3OmniMoeThinkerTextAttention,
)
from sglang_omni.vendor.sglang.layers import RMSNorm


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(2914)
    with get_context().override_server_args(
        cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(backend=Backend.DISABLED))
    ):
        pool = MHATokenToKVPool(
            size=32,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=8,
            head_dim=128,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        with torch.device("cuda"):
            text_rope = get_rope(128, 128, 32768, 1_000_000, dtype=torch.bfloat16)
            multimodal_rope = get_rope(
                128,
                128,
                32768,
                1_000_000,
                dtype=torch.bfloat16,
                rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20]},
            )
        attention = SimpleNamespace(
            use_fused_qk_norm_rope=False,
            q_size=16 * 128,
            kv_size=8 * 128,
            head_dim=128,
            q_norm=RMSNorm(128, eps=1e-6).to("cuda", torch.bfloat16),
            k_norm=RMSNorm(128, eps=1e-6).to("cuda", torch.bfloat16),
            alt_stream=None,
            attn=SimpleNamespace(layer_id=0, k_scale=None, v_scale=None),
        )
        batch = SimpleNamespace(
            out_cache_loc=torch.tensor([3, 7], device="cuda"),
            dcp_kv_mask=None,
        )
        positions = torch.tensor([17, 43], device="cuda")
        packed = torch.randn(2, (16 + 2 * 8) * 128, device="cuda", dtype=torch.bfloat16)
        previous_backend = get_fused_op_backend()
        observations: dict[str, str] = {}
        try:
            with forward_context(
                ForwardContext(SimpleNamespace(token_to_kv_pool=pool))
            ):
                for name, rope, backend in (
                    ("before_mrope_native", multimodal_rope, KernelBackend.TORCH),
                    ("after_text_cuda", text_rope, None),
                    ("after_text_native", text_rope, KernelBackend.TORCH),
                ):
                    attention.rotary_emb = rope
                    attention.compatible_with_fused_kv_buffer = not isinstance(
                        rope, MRotaryEmbedding
                    )
                    set_fused_op_backend(backend)
                    try:
                        Qwen3OmniMoeThinkerTextAttention.apply_qk_norm_rope(
                            attention, packed.clone(), positions, batch
                        )
                    except AssertionError as error:
                        observations[name] = str(error)
                    else:
                        observations[name] = "ok"
                attention.compatible_with_fused_kv_buffer = False
                Qwen3OmniMoeThinkerTextAttention.apply_qk_norm_rope(
                    attention, packed.clone(), positions, batch
                )
                observations["after_text_native_without_fused_store"] = "ok"
        finally:
            set_fused_op_backend(previous_backend)
        assert observations["before_mrope_native"] == "ok"
        assert observations["after_text_cuda"] == "ok"
        assert observations["after_text_native"] == (
            "fused_set_kv_buffer_arg is not supported for native implementation"
        )
        print(json.dumps(observations, indent=2))


if __name__ == "__main__":
    main()
else:
    pass
