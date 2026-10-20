# C1's component check before any repo edit: the served Fun-CosyVoice3 vocoder of the tree on PYTHONPATH
# (create_vocoder_executor with the vocoder stage's config.py arguments) over 8 fixed streams, three causal hops through
# run_step and the leftover final with HiFT, with SineGen2 as the tree has it, or with C1_PATCH=1 patched here by the
# proposed patch_sine_harmonics (DESIGN_NEXT_20261012.md 13, the same text). A warmup pass on other stream ids first,
# then a timed pass (device synchronized around each part, no sync debug mode) whose outputs are saved for a bit for bit
# comparison between arms, then a census pass counting the blocking calls by call site under sync debug mode "warn".
# usage: [C1_PATCH=1] python3 c1_sine_component_c20261013.py <out.pt>   |   python3 ... --compare <a.pt> <b.pt>
import collections
import os
import sys
import time
import warnings

import torch

ROWS = 8
TOKENS = 260


def patch_sine_harmonics(hift, device: torch.device) -> None:
    sine_generator = hift.m_source.l_sin_gen
    harmonics = torch.arange(
        1, sine_generator.harmonic_num + 2, dtype=torch.float32, device=device
    ).view(1, 1, -1)

    def forward(f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sine_waves = (
            sine_generator._f02sine(torch.multiply(f0, harmonics))
            * sine_generator.sine_amp
        )
        uv = sine_generator._f02uv(f0)
        noise_amp = uv * sine_generator.noise_std + (1 - uv) * sine_generator.sine_amp / 3
        noise = noise_amp * sine_generator.sine_waves[:, : sine_waves.shape[1]]
        return sine_waves * uv + noise, uv, noise

    sine_generator.forward = forward


def make_streams(scheduler, prefix):
    from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import CosyVoice3StreamState

    generator = torch.Generator().manual_seed(0)
    participants = []
    for row in range(ROWS):
        prompt_tokens = 40 + 7 * row
        state = CosyVoice3StreamState(hop_len=scheduler.token_hop_len)
        scheduler.latch_prompts(
            f"{prefix}{row}",
            state,
            prompt_token=torch.randint(0, 6561, (1, prompt_tokens), generator=generator, dtype=torch.int32),
            prompt_feat=torch.randn(1, 2 * prompt_tokens, 80, generator=generator) * 2,
            embedding=torch.randn(1, 192, generator=generator),
        )
        state.tokens = torch.randint(0, 6561, (TOKENS + 5 * row,), generator=generator).tolist()
        participants.append((f"{prefix}{row}", state))
    return participants


def one_pass(scheduler, vocoder, prefix, timings, saved, counts):
    from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput

    participants = make_streams(scheduler, prefix)
    for hop in range(3):
        with warnings.catch_warnings(record=True) as caught, vocoder.stream_context:
            warnings.simplefilter("always")
            torch.cuda.synchronize()
            start = time.perf_counter()
            decoded = scheduler.run_step(participants, "causal_window")
            torch.cuda.synchronize()
            timings[f"hop{hop}"] = time.perf_counter() - start
        census(f"hop{hop}", counts, caught)
        for request_id, delta in decoded.items():
            saved[f"hop{hop}/{request_id[len(prefix):]}"] = delta
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
        torch.cuda.synchronize()
        start = time.perf_counter()
        mels = vocoder.leftover_batch(final_items)
        torch.cuda.synchronize()
        middle = time.perf_counter()
        finals = scheduler.hift_step(participants, mels, is_final=True)
        torch.cuda.synchronize()
        timings["leftover flow"] = middle - start
        timings["leftover hift"] = time.perf_counter() - middle
    census("leftover", counts, caught)
    for (request_id, _), (delta, _) in zip(participants, finals):
        saved[f"leftover/{request_id[len(prefix):]}"] = delta


def census(name, counts, caught):
    for warning in caught:
        if "synchronizing" in str(warning.message):
            site = warning.filename.split("/sglang_omni/")[-1].split("/cosyvoice/")[-1]
            counts[(name, f"{site}:{warning.lineno}")] += 1


def run(out):
    from sglang_omni.models.fun_cosyvoice3 import stages
    from sglang_omni.models.fun_cosyvoice3.config import FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES

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
    patched = os.environ.get("C1_PATCH") == "1"
    spectra = os.environ.get("HIFT_SPECTRA") == "1"
    if patched:
        patch_sine_harmonics(vocoder.hift, next(vocoder.flow.parameters()).device)
    if spectra:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1_patch"))
        from c1_impl import patch_hift_spectra

        patch_hift_spectra(vocoder.hift, next(vocoder.flow.parameters()).device)
    print(f"tree={stages.__file__} patched={patched} spectra={spectra}", flush=True)
    with torch.inference_mode():
        for repeat in range(2):
            one_pass(scheduler, vocoder, f"w{repeat}_", {}, {}, collections.Counter())
        timings = {}
        saved = {}
        one_pass(scheduler, vocoder, "t_", timings, saved, collections.Counter())
        counts = collections.Counter()
        torch.cuda.set_sync_debug_mode("warn")
        one_pass(scheduler, vocoder, "c_", {}, {}, counts)
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    for part, seconds in timings.items():
        print(f"time {part} {seconds * 1e3:.2f} ms")
    for (part, site), count in sorted(counts.items()):
        print(f"blocking {part} {site} {count}")
    print(f"blocking total {sum(counts.values())}")
    saved["timings"] = torch.tensor([timings[k] for k in sorted(timings)], dtype=torch.float64)
    torch.save(saved, out)


def compare(paths):
    first = torch.load(paths[0])
    for path in paths[1:]:
        other = torch.load(path)
        names = [name for name in first if name != "timings"]
        different = 0
        for name in names:
            same = name in other and first[name].dtype == other[name].dtype and torch.equal(first[name], other[name])
            if not same:
                different += 1
                diff = (first[name].float() - other[name].float()).abs().max().item() if name in other else None
                print(f"{paths[0]} vs {path} {name}: DIFFERENT max abs {diff}")
        print(f"{paths[0]} vs {path}: {len(names) - different} of {len(names)} IDENTICAL")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2:])
    else:
        run(sys.argv[1])
