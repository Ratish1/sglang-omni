"""Bit identity of the fused predictor layers across a refactor, for Qwen3-Omni and Qwen3-TTS.

run      with PYTHONPATH=<tree>: the tree's own unit test fixtures build each model's
         predictor, fused and plain (Qwen3-Omni's plain path runs the exact add-norm kernel) (Qwen3-Omni: build_talker and fuse of
         tests/unit_test/qwen3_omni/test_predictor_kernels.py; Qwen3-TTS: real_shape_layer
         and real_shape_talker of tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py),
         run the opening pair and fourteen single-token passes at several batch sizes and
         seeds, and save every output and the K and V caches
compare  torch.equal over every saved tensor of two run files

usage: PYTHONPATH=<tree> python predictor_layers_refactor_identity.py run OUT.pt
       python predictor_layers_refactor_identity.py compare A.pt B.pt
"""

from __future__ import annotations

import sys

import torch


def run(out: str) -> None:
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        CudaGraphConfig,
        PhaseConfig,
    )
    from sglang.srt.runtime_context import get_context

    import sglang_omni.models.qwen3_tts.sglang_model as tts_model
    from sglang_omni.vendor.sglang.models import apply_qk_norm
    from tests.unit_test.qwen3_omni import test_predictor_kernels as omni
    from tests.unit_test.qwen3_tts import test_predictor_cuda_graph as tts

    tts_model.apply_qk_norm = apply_qk_norm
    device = torch.device("cuda")
    saved = {}
    with (
        get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            )
        ),
        torch.no_grad(),
    ):
        omni_cases = [
            (path, batch_size, seed)
            for path in ("fused", "plain")
            for batch_size in (1, 3, 12, 64)
            for seed in range(2)
        ]
        for path, batch_size, seed in omni_cases:
            talker = omni.build_talker(device, seed=seed)
            if path == "fused":
                talker = omni.fuse(talker)
            else:
                pass
            steps = omni.predictor_inputs(device, batch_size, seed=100 + seed)
            outputs = omni.run_sequence(talker, steps)
            name = f"omni {path} bs{batch_size} seed{seed}"
            for index, output in enumerate(outputs):
                saved[f"{name} out{index}"] = output.cpu()
            saved[f"{name} k"] = talker.predictor_k_cache.cpu()
            saved[f"{name} v"] = talker.predictor_v_cache.cpu()
        tts_cases = [
            (path, batch_size, seed)
            for path in ("fused", "plain")
            for batch_size in (1, 5, 64)
            for seed in range(2)
        ]
        for path, batch_size, seed in tts_cases:
            torch.manual_seed(seed)
            layers = [tts.real_shape_layer(device) for _ in range(tts.FUSED_LAYERS)]
            final_norm = tts.real_shape_norm(tts.FUSED_HIDDEN, device)
            talker = tts.real_shape_talker(
                device, layers, final_norm, fused=path == "fused"
            )
            generator = torch.Generator(device=device).manual_seed(200 + seed)
            shapes = [(batch_size, 2, tts.FUSED_HIDDEN)] + [
                (batch_size, 1, tts.FUSED_HIDDEN)
            ] * (tts.FUSED_PREDICTOR_LEN - 3)
            steps = [
                torch.randn(shape, device=device, generator=generator)
                for shape in shapes
            ]
            name = f"tts {path} bs{batch_size} seed{seed}"
            cache_len = 0
            for index, step in enumerate(steps):
                output = talker.predictor_forward_tokens(
                    token_embeds=step.to(tts.DTYPE),
                    batch_size=batch_size,
                    cache_len=cache_len,
                )
                saved[f"{name} out{index}"] = output.cpu()
                cache_len += step.shape[1]
            saved[f"{name} k"] = talker.predictor_k_cache.cpu()
            saved[f"{name} v"] = talker.predictor_v_cache.cpu()
    torch.save(saved, out)
    print(f"saved {len(saved)} tensors to {out}")


def compare(first: str, second: str) -> None:
    a, b = torch.load(first), torch.load(second)
    assert a.keys() == b.keys(), "the runs saved different tensors"
    differing = [key for key in a if not torch.equal(a[key], b[key])]
    for group in ("omni fused", "omni plain", "tts fused", "tts plain"):
        keys = [key for key in a if key.startswith(group)]
        bad = [key for key in differing if key.startswith(group)]
        print(f"{group}: {len(keys) - len(bad)} of {len(keys)} tensors bit identical")
    for key in differing[:10]:
        print("  differs:", key)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
