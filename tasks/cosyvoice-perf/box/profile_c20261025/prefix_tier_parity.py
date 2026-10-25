"""#3132 component parity (10-25 evening): the Flow prefix hop on the trained model, real conditioning and the served
hop schedule, at main's prefix tiers, at #3132's and eagerly, each arm on its own prefix pool, against upstream
CosyVoice's DiT in fp32.

Streams: the first --streams distinct seed-tts EN references, conditioned as the server conditions them (main's ONNX
Runtime speech tokenizer, the 24 kHz prompt mel, the CAM++ embedding, aligned to two frames a token and padded to a
hop as the scheduler latches them); each stream's generated tokens are other references' speech tokens. Hops of 25,
50, then 100 tokens plus the 3 token lookahead, as next_stream_hop_len grows them; stream i joins at step i and each
step batches every live stream's next hop.
Arms: main_tiers (main's prefix_cuda_graph.py, loaded from --main-tree), head_tiers (the tree on PYTHONPATH), eager
(no runner, solve_flow_euler_prefix), all through hop_batch_prefix, the served call, with the compiled prefix forward.
Per step: each arm's tier; between arms, bit for bit, the returned mels, every row's committed frames and
convolution context, and every committed K and V page of every Euler step and layer; each arm's new frames against
inference_causal over the whole history with the upstream DiT (packed estimator detached), fp32, TF32 off: max
absolute error and relative L2.
usage: CUDA_VISIBLE_DEVICES=<card> PYTHONPATH=<head tree> python3 prefix_tier_parity.py --main-tree <tree> --out <json>
"""

import argparse
import bisect
import importlib.util
import json
import os
import sys

import torch

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    BLOCK_FRAMES,
    PrefixCacheRow,
    PrefixKVPool,
    compile_forward_prefix,
    grow_rows,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import PrefixCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    align_flow_prompt,
    cosyvoice3_reference_input_key,
    load_prompt_audio,
    load_prompt_audio_24k,
)
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.models.fun_cosyvoice3.stages import (
    FlowBatchInput,
    create_vocoder_executor,
    load_cosyvoice3_flow_hift,
    pack_flow_inputs,
    prepare_flow_conditioning,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MAX_HOP_LEN,
    next_stream_hop_len,
    pad_flow_prompt_to_hop,
)
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import causal_hop_frames, upload_prompt
from sglang_omni.models.fun_cosyvoice3.utils import (
    SpeakerEncoder,
    SpeechTokenizerV3,
    extract_prompt_speech_feat,
)
from sglang_omni.platforms import current_platform
from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
HOPS = 4


def load_main_runner(main_tree):
    spec = importlib.util.spec_from_file_location(
        "main_prefix_cuda_graph", os.path.join(main_tree, "sglang_omni/models/fun_cosyvoice3/prefix_cuda_graph.py")
    )
    module = importlib.util.module_from_spec(spec)
    # note (ratish): a dataclass under postponed annotations looks its module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PrefixCudaGraphRunner


def build_streams(vocoder, count, device):
    checkpoint = resolve_checkpoint(MODEL)
    tokenizer = SpeechTokenizerV3(os.path.join(checkpoint, "speech_tokenizer_v3.onnx"), device="cuda")
    speaker_encoder = SpeakerEncoder(os.path.join(checkpoint, "campplus.onnx"), device="cuda")
    samples, seen = [], set()
    for sample in load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en"):
        key = cosyvoice3_reference_input_key(sample.ref_audio)
        if key not in seen:
            seen.add(key)
            samples.append(sample)
        else:
            pass
    codes = [tokenizer.extract_speech_token(load_prompt_audio(sample.ref_audio), 16000) for sample in samples]
    tokens_needed = TOKEN_HOP_LEN + 2 * TOKEN_HOP_LEN + (HOPS - 2) * TOKEN_MAX_HOP_LEN + PRE_LOOKAHEAD_LEN
    parameter = next(vocoder.flow.parameters())
    ratio = vocoder.flow.token_mel_ratio
    streams = []
    for index in range(count):
        sample = samples[index]
        prompt_token, prompt_feat = align_flow_prompt(
            codes[index], extract_prompt_speech_feat(load_prompt_audio_24k(sample.ref_audio), 24000)
        )
        prompt_token, prompt_feat = pad_flow_prompt_to_hop(
            prompt_token, prompt_feat, hop_len=TOKEN_HOP_LEN, token_mel_ratio=ratio
        )
        embedding = speaker_encoder.extract_embedding(load_prompt_audio(sample.ref_audio), 16000)
        generated = []
        donor = count + index
        while len(generated) < tokens_needed:
            generated += codes[donor % len(codes)][0].tolist()
            donor += count
        streams.append({
            "sample": sample.sample_id,
            "prompt_token": upload_prompt(prompt_token, parameter.device, torch.int32),
            "prompt_feat": upload_prompt(prompt_feat, parameter.device, parameter.dtype),
            "embedding": upload_prompt(embedding, parameter.device, parameter.dtype),
            "generated": generated[:tokens_needed],
        })
    return streams


def row_state(pool, pair):
    rows = []
    for row in pair:
        pages = row.pages(pool.device)[: row.committed_frames]
        rows.append({"committed": row.committed_frames, "context": row.conv_context,
                     "keys": [[layer[pages] for layer in step] for step in pool.keys],
                     "values": [[layer[pages] for layer in step] for step in pool.values]})
    return rows


def states_equal(first, second):
    for a, b in zip(first, second):
        if a["committed"] != b["committed"] or not torch.equal(a["context"], b["context"]):
            return False
        else:
            pass
        for kind in ("keys", "values"):
            if not all(torch.equal(x, y) for step_a, step_b in zip(a[kind], b[kind]) for x, y in zip(step_a, step_b)):
                return False
            else:
                pass
    return True


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-tree", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--streams", type=int, default=6)
    args = parser.parse_args()
    torch.manual_seed(3132)
    scheduler = create_vocoder_executor(
        MODEL, device="cuda", gpu_id=0, dtype="bfloat16", max_batch_size=16, flow_prefix_cache_gb=0,
        enable_dit_torch_compile=False, enable_flow_cuda_graph=False, enable_flow_prefix_cuda_graph=False,
        enable_flow_whole_history_cuda_graph=False, enable_hift_cuda_graph=False,
    )
    vocoder = scheduler.vocoder
    flow = vocoder.flow
    estimator = flow.packed_estimator
    assert estimator is not None and estimator.is_ragged
    device = torch.device("cuda", 0)
    attention = estimator.dit.transformer_blocks[0].attn
    ratio = flow.token_mel_ratio
    # note (ratish): the served DiT's Linear and Conv1d weights are cast to bf16 at load, so the fp32 reference is a
    # second Flow left in fp32, its upstream DiT (no packed estimator), on the served Flow's noise.
    reference_flow, _ = load_cosyvoice3_flow_hift(
        resolve_checkpoint(MODEL), device="cuda:0", fp16=False, autocast_dtype=None, enable_flow_estimator_trt=False
    )
    reference_flow.packed_estimator = None
    reference_flow.decoder.rand_noise = flow.decoder.rand_noise.clone()
    streams = build_streams(vocoder, args.streams, device)
    total_frames = [
        (int(stream["prompt_token"].shape[1]) + len(stream["generated"]) - PRE_LOOKAHEAD_LEN) * ratio for stream in streams
    ]
    capacity = sum(2 * -(-frames // BLOCK_FRAMES) * BLOCK_FRAMES for frames in total_frames)
    with vocoder.stream_context, torch.autocast("cuda", dtype=vocoder.autocast_dtype):
        conditioning = prepare_flow_conditioning(
            flow, pack_flow_inputs(flow.flow, [scheduler.make_warmup_flow_input(TOKEN_HOP_LEN)]), finalize=False
        )
    prefix_forward = compile_forward_prefix()
    backend = current_platform.get_device_graph_backend(device)
    runner_types = {"main_tiers": load_main_runner(args.main_tree), "head_tiers": PrefixCudaGraphRunner, "eager": None}
    arms = {}
    for name, runner_type in runner_types.items():
        pool = PrefixKVPool(
            layer_num=len(estimator.dit.transformer_blocks), euler_steps=len(estimator.block_modulations),
            head_num=attention.heads, head_dim=attention.inner_dim // attention.heads,
            capacity_frames=capacity + BLOCK_FRAMES * 4, device=device, dtype=vocoder.autocast_dtype,
        )
        pool.forward = prefix_forward
        flow.prefix_pool, flow.prefix_cuda_graph_runner = pool, None
        scheduler.warmup_prefix_compile()
        if runner_type is None:
            runner = None
        else:
            runner = runner_type(
                estimator, pool,
                graphs=SolveGraphCapture(backend, device=device, autocast_dtype=vocoder.autocast_dtype),
                frame_dtype=conditioning.noisy_mel.dtype, speaker_dtype=conditioning.speaker_embedding.dtype,
                cfg_rate=flow.decoder.inference_cfg_rate, mel_channels=flow.output_size,
                speaker_channels=flow.spk_embed_affine_layer.out_features, max_rows=16,
                hop_frames=TOKEN_MAX_HOP_LEN * ratio, min_hop_frames=TOKEN_HOP_LEN * ratio,
                max_frames=flow.decoder.rand_noise.shape[2],
            )
            runner.capture()
        arms[name] = {"pool": pool, "runner": runner, "caches": {}}
        print(f"arm {name}: pool {pool.free_frames} frames, tiers "
              f"{runner.tier_frames[:3] + ['..', runner.tier_frames[-1]] if runner else 'eager'}", flush=True)
    state = [{"offset": 0, "hop": TOKEN_HOP_LEN, "hops": 0} for _ in streams]
    records = []
    step = 0
    while any(item["hops"] < HOPS for item in state):
        live = [index for index in range(len(streams)) if index <= step and state[index]["hops"] < HOPS]
        items = [
            FlowBatchInput(
                token=torch.tensor(
                    streams[index]["generated"][: state[index]["offset"] + state[index]["hop"] + PRE_LOOKAHEAD_LEN],
                    dtype=torch.int32,
                ).unsqueeze(0),
                prompt_token=streams[index]["prompt_token"],
                prompt_feat=streams[index]["prompt_feat"],
                embedding=streams[index]["embedding"],
            )
            for index in live
        ]
        totals = [causal_hop_frames(item, token_mel_ratio=ratio) for item in items]
        outputs, row_states, tiers, prefixes = {}, {}, {}, {}
        for name, arm in arms.items():
            pairs = []
            for index, total in zip(live, totals):
                pair = arm["caches"].setdefault(index, (PrefixCacheRow(), PrefixCacheRow()))
                assert grow_rows(arm["pool"], list(pair), [total, total])
                pairs.append(pair)
            prefixes[name] = [pair[0].committed_frames for pair in pairs]
            new_frames = sum(total - prefix for total, prefix in zip(totals, prefixes[name]))
            runner = arm["runner"]
            if runner is None:
                tiers[name] = None
            else:
                tier = bisect.bisect_left(runner.tier_frames, new_frames)
                tiers[name] = runner.tier_frames[tier] if tier < len(runner.tier_frames) else "eager above top"
            flow.prefix_pool, flow.prefix_cuda_graph_runner = arm["pool"], runner
            with vocoder.stream_context:
                mels = vocoder.hop_batch_prefix(items, pairs)
            torch.cuda.synchronize()
            outputs[name] = [mel.float().clone() for mel in mels]
            row_states[name] = [row_state(arm["pool"], pair) for pair in pairs]
        flow.prefix_pool, flow.prefix_cuda_graph_runner = None, None
        matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            with vocoder.stream_context, torch.autocast("cuda", enabled=False):
                reference = [
                    mel.float()
                    for mel in reference_flow.inference_causal([
                        FlowBatchInput(token=item.token, prompt_token=item.prompt_token,
                                       prompt_feat=item.prompt_feat.float(), embedding=item.embedding.float())
                        for item in items
                    ])
                ]
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul_tf32, cudnn_tf32
        torch.cuda.synchronize()
        record = {"step": step, "rows": len(live), "new_frames": sum(t - p for t, p in zip(totals, prefixes["eager"])),
                  "tiers": tiers, "equal": {}, "fp32": {}}
        names = list(arms)
        for first_index, first in enumerate(names):
            for second in names[first_index + 1:]:
                record["equal"][f"{first}={second}"] = {
                    "mels": all(torch.equal(a, b) for a, b in zip(outputs[first], outputs[second])),
                    "state": all(states_equal(a, b) for a, b in zip(row_states[first], row_states[second])),
                }
        for name in names:
            errors, norms = [], []
            for index, (mel, whole) in enumerate(zip(outputs[name], reference)):
                prompt_frames = int(streams[live[index]]["prompt_token"].shape[1]) * ratio
                start = max(prefixes[name][index] - prompt_frames, 0)
                new, exact = mel[..., start:], whole[..., start: mel.shape[-1]]
                assert new.shape == exact.shape, (new.shape, exact.shape)
                errors.append((new - exact).abs().max().item())
                norms.append(((new - exact).norm() / exact.norm()).item())
            record["fp32"][name] = {"max_abs": max(errors), "rel_l2": max(norms)}
        records.append(record)
        print(json.dumps(record), flush=True)
        for index in live:
            stream = state[index]
            stream["offset"] += stream["hop"]
            stream["hop"] = next_stream_hop_len(stream["hop"], max_hop_len=TOKEN_MAX_HOP_LEN)
            stream["hops"] += 1
        step += 1
    pairs_equal = {key: all(record["equal"][key]["mels"] and record["equal"][key]["state"] for record in records)
                   for key in records[0]["equal"]}
    summary = {"steps": len(records), "bit_identical": pairs_equal,
               "fp32_max_abs": {name: max(record["fp32"][name]["max_abs"] for record in records) for name in arms},
               "fp32_rel_l2": {name: max(record["fp32"][name]["rel_l2"] for record in records) for name in arms},
               "streams": [stream["sample"] for stream in streams]}
    print("summary " + json.dumps(summary), flush=True)
    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=1)
    scheduler.stop()


if __name__ == "__main__":
    main()
