# Run the served Fun-CosyVoice3 vocoder of the tree on PYTHONPATH (create_vocoder_executor with the vocoder stage's
# config.py arguments) over fixed streams: three causal hops through run_step (the prefix pool path), the same first
# hop through the plain packed path, the leftover final with HiFT, and the buffered Flow. Every mel, audio delta and
# conditioning tensor is saved, so two trees can be compared bit for bit. The blocking CUDA calls of each part are
# counted by Python call site under torch.cuda.set_sync_debug_mode("warn").
# usage: python3 flow_uploads_identity_c20261012.py <out.pt>   |   python3 ... --compare <a.pt> <b.pt> [<c.pt> ...]
import collections
import sys
import warnings

import torch

ROWS = 8
TOKENS = 260


def make_streams(scheduler):
    from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import CosyVoice3StreamState

    generator = torch.Generator().manual_seed(0)
    participants = []
    for row in range(ROWS):
        prompt_tokens = 40 + 7 * row
        state = CosyVoice3StreamState(hop_len=scheduler.token_hop_len)
        scheduler.latch_prompts(
            f"r{row}",
            state,
            prompt_token=torch.randint(0, 6561, (1, prompt_tokens), generator=generator, dtype=torch.int32),
            prompt_feat=torch.randn(1, 2 * prompt_tokens, 80, generator=generator) * 2,
            embedding=torch.randn(1, 192, generator=generator),
        )
        state.tokens = torch.randint(0, 6561, (TOKENS + 5 * row,), generator=generator).tolist()
        participants.append((f"r{row}", state))
    return participants


def census(name, counts, caught):
    for warning in caught:
        if "synchronizing" in str(warning.message):
            counts[(name, f"{warning.filename.split('/sglang_omni/')[-1]}:{warning.lineno}")] += 1


def run(out):
    from sglang_omni.models.fun_cosyvoice3 import stages
    from sglang_omni.models.fun_cosyvoice3.config import FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES
    from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput, HiftStepRow

    scheduler = stages.create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=0,
        dtype="bfloat16",
        max_batch_size=16,
        max_batch_wait_ms=30,
        flow_batch_admission_frames=8000,
        flow_merge_max_gap_frames=384,
        flow_merge_pad_budget_percent=25.0,
        enable_flow_cuda_graph=True,
        enable_flow_prefix_cuda_graph=True,
        enable_flow_whole_history_cuda_graph=True,
        flow_cuda_graph_capture_shapes=FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
        enable_flow_estimator_trt=False,
        token_hop_len=25,
        token_max_hop_len=100,
        disable_hop_growth=False,
        flow_prefix_cache_gb=24.0,
    )
    vocoder = scheduler.vocoder
    print(
        f"tree={stages.__file__} prefix_pool={vocoder.flow.prefix_pool is not None} "
        f"flow_dtype={next(vocoder.flow.parameters()).dtype} noise={vocoder.flow.decoder.rand_noise.device}",
        flush=True,
    )
    saved = {}
    counts = collections.Counter()
    autocast = torch.autocast(device_type="cuda", dtype=vocoder.autocast_dtype)
    with torch.inference_mode():
        participants = make_streams(scheduler)
        print(f"latched prompt_feat {participants[0][1].prompt_feat.device} {participants[0][1].prompt_feat.dtype}")
        saved["prompt_feat"] = torch.cat([state.prompt_feat.reshape(-1).float().cpu() for _, state in participants])
        torch.cuda.set_sync_debug_mode("warn")
        for hop in range(3):
            with warnings.catch_warnings(record=True) as caught, vocoder.stream_context:
                warnings.simplefilter("always")
                decoded = scheduler.run_step(participants, "causal_window")
            census(f"hop{hop}", counts, caught)
            for request_id, delta in decoded.items():
                saved[f"hop{hop}/{request_id}"] = delta
        first_items = [
            FlowBatchInput(
                token=torch.tensor(state.tokens[:28], dtype=torch.int32).unsqueeze(0),
                prompt_token=state.prompt_token,
                prompt_feat=state.prompt_feat,
                embedding=state.embedding,
            )
            for _, state in participants
        ]
        with warnings.catch_warnings(record=True) as caught, vocoder.stream_context:
            warnings.simplefilter("always")
            plain = vocoder.hop_batch(first_items)
            with autocast:
                packed = stages.pack_flow_inputs(vocoder.flow.flow, first_items)
                conditioning = stages.prepare_flow_conditioning(vocoder.flow, packed, finalize=False)
        census("plain", counts, caught)
        for row, mel in enumerate(plain):
            saved[f"plain/{row}"] = mel.float().cpu()
        for name in ("token_condition", "speaker_embedding", "prompt_mel", "noisy_mel"):
            saved[f"conditioning/{name}"] = getattr(conditioning, name).float().cpu()
        saved["packed/token"] = packed.token.cpu()
        final_items = [
            FlowBatchInput(
                token=torch.tensor(state.tokens, dtype=torch.int32).unsqueeze(0),
                prompt_token=state.prompt_token,
                prompt_feat=state.prompt_feat,
                embedding=state.embedding,
            )
            for _, state in participants
        ]
        with warnings.catch_warnings(record=True) as caught, vocoder.stream_context:
            warnings.simplefilter("always")
            mels = vocoder.leftover_batch(final_items)
            finals = scheduler.hift_step(participants, mels, is_final=True)
        census("leftover", counts, caught)
        for (request_id, _), (delta, _) in zip(participants, finals):
            saved[f"leftover/{request_id}"] = delta
        buffered_items = [
            FlowBatchInput(
                token=item.token[:, :120],
                prompt_token=item.prompt_token,
                prompt_feat=item.prompt_feat.cpu(),
                embedding=item.embedding.cpu(),
            )
            for item in final_items[:4]
        ]
        with warnings.catch_warnings(record=True) as caught, vocoder.stream_context, autocast:
            warnings.simplefilter("always")
            buffered = vocoder.flow.inference(buffered_items)
        census("buffered", counts, caught)
        for row, mel in enumerate(buffered):
            saved[f"buffered/{row}"] = mel.float().cpu()
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    for (part, site), count in sorted(counts.items()):
        print(f"blocking {part} {site} {count}")
    print(f"blocking total {sum(counts.values())}")
    torch.save(saved, out)


def compare(paths):
    first = torch.load(paths[0])
    for path in paths[1:]:
        other = torch.load(path)
        different = 0
        for name in first:
            same = name in other and first[name].dtype == other[name].dtype and torch.equal(first[name], other[name])
            if not same:
                different += 1
                diff = (first[name].float() - other[name].float()).abs().max().item() if name in other else None
                print(f"{paths[0]} vs {path} {name}: DIFFERENT max abs {diff}")
        print(f"{paths[0]} vs {path}: {len(first) - different} of {len(first)} IDENTICAL")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2:])
    else:
        run(sys.argv[1])
