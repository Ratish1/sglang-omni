"""Compare trained Flow outputs in separate parent and head processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.stages import (
    FlowBatchInput,
    create_vocoder_executor,
)


def make_inputs(
    token_lengths: tuple[int, ...], prompt_lengths: tuple[int, ...]
) -> list[FlowBatchInput]:
    generator = torch.Generator().manual_seed(sum(token_lengths) + sum(prompt_lengths))
    return [
        FlowBatchInput(
            token=torch.randint(1, 6500, (1, tokens), generator=generator).int(),
            prompt_token=torch.randint(
                1, 6500, (1, prompts), generator=generator
            ).int(),
            prompt_feat=torch.randn(1, prompts * 2, 80, generator=generator),
            embedding=torch.randn(1, 192, generator=generator),
        )
        for tokens, prompts in zip(token_lengths, prompt_lengths, strict=True)
    ]


def digest(tensor: torch.Tensor) -> dict[str, object]:
    assert torch.isfinite(tensor).all()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(
            tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=("parent", "head"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2944)
    started = time.monotonic()
    scheduler = create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=0,
        dtype="bfloat16",
        max_batch_size=3,
        flow_batch_admission_frames=256,
        flow_prefix_cache_gb=0.5,
        enable_dit_torch_compile=True,
        enable_flow_cuda_graph=True,
        flow_cuda_graph_capture_shapes=((1, 128),),
        enable_flow_prefix_cuda_graph=True,
        enable_flow_whole_history_cuda_graph=True,
        enable_hift_cuda_graph=False,
    )
    torch.cuda.synchronize()
    factory_seconds = time.monotonic() - started
    flow = scheduler.vocoder.flow
    assert flow.is_buffered_packed and flow.packed_estimator.is_compiled
    assert (flow.cuda_graph_runner is None) == (args.label == "head")
    assert ("forward" not in vars(flow.decoder.estimator)) == (args.label == "head")
    assert flow.prefix_cuda_graph_runner is not None
    runner = flow.whole_history_cuda_graph_runner
    assert runner is not None
    records = []
    cases = [
        ((1, 25, 40), (0, 7, 3)),
        ((20, 37, 52), (1, 3, 15)),
        ((129,), (0,)),
        ((2, 3, 4, 5), (0, 1, 2, 3)),
        ((1, 25, 40), (0, 7, 3)),
    ]
    with torch.inference_mode(), scheduler.vocoder.stream_context:
        angles = [
            digest(flow.packed_estimator.rope_angles(length))
            for length in (37, 128, 15000)
        ]
        for token_lengths, prompt_lengths in cases:
            inputs = make_inputs(token_lengths, prompt_lengths)
            frames = 2 * (sum(token_lengths) + sum(prompt_lengths))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = flow.inference(inputs)
                flow.whole_history_cuda_graph_runner = None
                try:
                    eager_outputs = flow.inference(inputs)
                finally:
                    flow.whole_history_cuda_graph_runner = runner
            assert all(
                torch.equal(graph, eager)
                for graph, eager in zip(outputs, eager_outputs, strict=True)
            )
            record = {
                "tokens": token_lengths,
                "prompts": prompt_lengths,
                "frames": frames,
                "expected_graph_hit": frames <= 256 and len(inputs) <= 3,
                "rows": [digest(mel) for mel in outputs],
            }
            records.append(record)
            print(json.dumps(record), flush=True)
        assert records[0]["rows"] == records[-1]["rows"]
        item = make_inputs((28,), (0,))[0]
        cache = scheduler.vocoder.prefix_cache_rows(50)
        assert cache is not None
        try:
            first = scheduler.vocoder.hop_batch_prefix([item], [cache])
            longer = FlowBatchInput(
                token=torch.cat((item.token, item.token[:, :25]), dim=1),
                prompt_token=item.prompt_token,
                prompt_feat=item.prompt_feat,
                embedding=item.embedding,
            )
            assert scheduler.vocoder.grow_prefix_cache(cache, 100)
            followup = scheduler.vocoder.hop_batch_prefix([longer], [cache])
            prefix = [digest(first[0]), digest(followup[0])]
            assert [row.committed_frames for row in cache] == [100, 100]
        finally:
            scheduler.vocoder.release_prefix_cache(cache)
        fallback = make_inputs((32,), (7,))[0]
        torch.manual_seed(2945)
        fallback_started = time.monotonic()
        wav = scheduler.vocoder.token2wav(
            fallback.token,
            fallback.prompt_token,
            fallback.prompt_feat,
            fallback.embedding,
        )
        fallback_digest = digest(wav)
        fallback_seconds = time.monotonic() - fallback_started
        assert wav.numel() > 0
    scheduler.stop()
    summary = {
        "label": args.label,
        "source": stages.__file__,
        "factory_seconds_reduced_capture_config": factory_seconds,
        "rope_angles": angles,
        "cases": records,
        "prefix_hops": prefix,
        "native_fallback": fallback_digest,
        "native_fallback_seconds": fallback_seconds,
    }
    Path(f"/data/review2944/{args.label}.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
