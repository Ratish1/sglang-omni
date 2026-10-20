"""Mode 2 of the 10-16 lifecycle profile: the Fun-CosyVoice3 vocoder's components as mapping and formal torch trace
pairs (the omni-gpu-deep-dive contract). Formal runs the served vocoder executor (compiled DiT, prefix cache, prefix
and whole history graphs); mapping runs the same work on an eager twin (no compile, no graphs, stacks on).
Workloads, each one realistic c16 step shape:
  flow_prefix_hops_rows16   a stream start for 16 rows: first hop (75 prompt + 28 tokens, the 3200 frame tier),
                            25 more tokens, the next hop (an 800 frame tier), caches released
  flow_prefix_first_rows16_eager  16 first hops of a 125 token prompt (4800 frames, above the top prefix tier)
  flow_final_rows4          4 finals of 150 to 400 tokens (whole history graphs)
  hift_step_rows16          one streaming HiFT step of 16 rows, a 50 frame hop each over 150 to 600 frame histories
usage (in the container, from the tree): PYTHONPATH=<tree> python3 vocoder_pairs_c20261016.py <out dir> [workload ...]
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, ".claude/skills/omni-gpu-deep-dive/scripts")
from omni_trace_pair import capture_pair  # noqa: E402

from sglang_omni.models.fun_cosyvoice3 import stages  # noqa: E402
from sglang_omni.models.fun_cosyvoice3.config import (  # noqa: E402
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
)
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import causal_hop_frames  # noqa: E402

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
SERVED = dict(
    device="cuda",
    gpu_id=0,
    dtype="bfloat16",
    flow_batch_admission_frames=8000,
    flow_merge_max_gap_frames=384,
    flow_merge_pad_budget_percent=25.0,
    max_batch_size=16,
    max_batch_wait_ms=30,
    flow_cuda_graph_capture_shapes=FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
    enable_flow_estimator_trt=False,
    token_hop_len=25,
    token_max_hop_len=100,
    disable_hop_growth=False,
)
ITERS = 5
WARMUP = 3


def build(eager):
    graphs = not eager
    return stages.create_vocoder_executor(
        MODEL,
        **SERVED,
        enable_dit_torch_compile=graphs,
        enable_flow_cuda_graph=graphs,
        enable_flow_prefix_cuda_graph=graphs,
        enable_flow_whole_history_cuda_graph=graphs,
        flow_prefix_cache_gb=24.0,
    ).vocoder


def make_item(generator, token_count, prompt_count, flow):
    return stages.FlowBatchInput(
        token=torch.randint(0, 6561, (1, token_count), generator=generator, dtype=torch.int32),
        prompt_token=torch.randint(0, 6561, (1, prompt_count), generator=generator, dtype=torch.int32),
        prompt_feat=torch.randn(1, prompt_count * flow.token_mel_ratio, flow.output_size, generator=generator),
        embedding=torch.randn(1, flow.spk_embed_affine_layer.in_features, generator=generator),
    )


def grown(item, generator, extra):
    more = torch.randint(0, 6561, (1, extra), generator=generator, dtype=torch.int32)
    return stages.FlowBatchInput(
        token=torch.cat((item.token, more), dim=1),
        prompt_token=item.prompt_token,
        prompt_feat=item.prompt_feat,
        embedding=item.embedding,
    )


def prefix_hops(vocoder, first_items, next_items):
    ratio = vocoder.flow.token_mel_ratio
    caches = [vocoder.prefix_cache_rows(causal_hop_frames(item, token_mel_ratio=ratio)) for item in first_items]
    vocoder.hop_batch_prefix(first_items, caches)
    if next_items is not None:
        for item, cache in zip(next_items, caches):
            assert vocoder.grow_prefix_cache(cache, causal_hop_frames(item, token_mel_ratio=ratio))
        vocoder.hop_batch_prefix(next_items, caches)
    for cache in caches:
        vocoder.release_prefix_cache(cache)


def hift_rows(vocoder, generator, history_frames):
    hop_frames = 50
    rows = []
    for frames in history_frames:
        emitted_frame = frames - vocoder.hift_hold_frames - hop_frames
        rows.append(
            stages.HiftStepRow(
                history=torch.randn(1, vocoder.flow.output_size, frames, generator=generator).cuda(),
                emitted_samples=emitted_frame * vocoder.hift_samples_per_mel_frame,
                is_final=False,
            )
        )
    return rows


def main(out_dir, selected):
    formal = build(eager=False)
    mapping = build(eager=True)
    flow = formal.flow
    generator = torch.Generator().manual_seed(0)
    first16 = [make_item(generator, 28, 75, flow) for _ in range(16)]
    next16 = [grown(item, generator, 25) for item in first16]
    long_prompt16 = [make_item(generator, 28, 125, flow) for _ in range(16)]
    finals4 = [make_item(generator, length, 75, flow) for length in (150, 250, 300, 400)]
    histories = [150 + 30 * index for index in range(16)]
    hift16 = hift_rows(formal, generator, histories)
    workloads = {
        "flow_prefix_hops_rows16": lambda vocoder: prefix_hops(vocoder, first16, next16),
        "flow_prefix_first_rows16_eager": lambda vocoder: prefix_hops(vocoder, long_prompt16, None),
        "flow_final_rows4": lambda vocoder: vocoder.leftover_batch(finals4),
        "hift_step_rows16": lambda vocoder: vocoder.hift_step(hift16),
    }
    for name, work in workloads.items():
        if selected and name not in selected:
            continue
        with torch.inference_mode():
            capture_pair(
                output_dir=Path(out_dir) / name,
                mapping_body=lambda work=work: work(mapping),
                formal_body=lambda work=work: work(formal),
                iters=ITERS,
                warmup=WARMUP,
            )
        print(f"done {name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], set(sys.argv[2:]))
