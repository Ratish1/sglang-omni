"""Does the Qwen3-TTS predictor keep a request's bits across batch sizes in batch-invariant mode?

Deterministic inference turns on SGLang's batch-invariant mode, which promises the same bits
for a request at every batch size. With the mode on, the predictor at the checkpoint's shapes
(the fixtures of tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py) runs the opening pair
and 14 single-token passes at batch 1, 32 and 40, plain and fused; request 0's outputs and
K/V rows are compared bit for bit with batch 1. The plain path is main's code path.

usage: PYTHONPATH=<tree> python predictor_deterministic_probe.py
"""

from __future__ import annotations

import torch

BATCHES = (1, 32, 40)


def main() -> None:
    from sglang.srt.batch_invariant_ops import (
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        CudaGraphConfig,
        PhaseConfig,
    )
    from sglang.srt.runtime_context import get_context

    import sglang_omni.models.qwen3_tts.sglang_model as tts_model
    from sglang_omni.vendor.sglang.models import apply_qk_norm
    from tests.unit_test.qwen3_tts import test_predictor_cuda_graph as tts

    tts_model.apply_qk_norm = apply_qk_norm
    enable_batch_invariant_mode()
    print(f"batch-invariant mode: {is_batch_invariant_mode_enabled()}")
    device = torch.device("cuda")
    torch.manual_seed(0)
    layers = [tts.real_shape_layer(device) for _ in range(tts.FUSED_LAYERS)]
    final_norm = tts.real_shape_norm(tts.FUSED_HIDDEN, device)
    generator = torch.Generator(device=device).manual_seed(1)
    largest = max(BATCHES)
    hidden = tts.FUSED_HIDDEN
    pair = torch.randn(largest, 2, hidden, device=device, generator=generator)
    singles = [
        torch.randn(largest, 1, hidden, device=device, generator=generator)
        for _ in range(tts.FUSED_PREDICTOR_LEN - 3)
    ]
    pair = pair.to(tts.DTYPE)
    singles = [single.to(tts.DTYPE) for single in singles]
    tts.FUSED_MAX_BS = largest
    with (
        get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            )
        ),
        torch.no_grad(),
    ):
        for path in ("plain", "fused"):
            talker = tts.real_shape_talker(
                device, layers, final_norm, fused=path == "fused"
            )
            reference = None
            for batch in BATCHES:
                talker.predictor_k_cache.zero_()
                talker.predictor_v_cache.zero_()
                rows = [
                    talker.predictor_forward_tokens(
                        token_embeds=pair[:batch].clone(), batch_size=batch, cache_len=0
                    )[:1].clone()
                ]
                for index, single in enumerate(singles):
                    rows.append(
                        talker.predictor_forward_tokens(
                            token_embeds=single[:batch].clone(),
                            batch_size=batch,
                            cache_len=2 + index,
                        )[:1].clone()
                    )
                rows.append(talker.predictor_k_cache[:, :1].clone())
                rows.append(talker.predictor_v_cache[:, :1].clone())
                if reference is None:
                    reference = rows
                    print(f"{path} batch {batch:3d}: reference")
                else:
                    same = sum(torch.equal(a, b) for a, b in zip(rows, reference))
                    print(
                        f"{path} batch {batch:3d}: request 0 bit equal to batch 1 in "
                        f"{same} of {len(rows)} tensors"
                    )


if __name__ == "__main__":
    main()
