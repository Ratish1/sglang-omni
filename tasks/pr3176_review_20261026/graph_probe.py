"""Review-only trained ring graph replay and storage differential diagnostic."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    IncrementalCodecGraphPool,
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

apply_qwen_tts_transformers_compatibility_patches()
from qwen_tts import Qwen3TTSTokenizer

ROOT = Path("/data/review3176oct26")


def metrics(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "snr_db": float(
            10
            * torch.log10(
                expected.float().square().sum() / delta.square().sum().clamp_min(1e-30)
            )
        ),
        "max_abs": float(delta.abs().max()),
        "equal": torch.equal(actual, expected),
    }


def main():
    torch.manual_seed(3176)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    decoder = tokenizer.model.decoder.eval()
    fp32 = copy.deepcopy(decoder).float()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    device = torch.device("cuda", 0)
    arenas = [
        Qwen3TTSCodecStateArena(
            incremental,
            num_slots=5,
            device=device,
            dtype=torch.bfloat16,
            in_place_frames=width,
        )
        for width in (8, 0)
    ]
    rings, buffered = arenas
    assert rings.ring_frames == 79 and rings.key_rings
    for arena in arenas:
        assert [arena.acquire() for _ in range(4)] == [0, 1, 2, 3]
    pool = IncrementalCodecGraphPool(stream_priority=0)
    runners = {}
    for name, mode, widths, compiled, emit in [
        ("cold", "cold", (1, 2), (1, 2), 0),
        ("window", "window", (8, 64), (8,), 0),
        ("tail", "tail", (581,), (), 1),
    ]:
        runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
            incremental,
            device=device,
            dtype=torch.bfloat16,
            num_quantizers=decoder.config.num_quantizers,
            mode=mode,
            fresh_frames=widths,
            batch_sizes=(4,),
            min_free_gb=0,
            compile_fresh_frames=compiled,
            arena=rings,
            graph_pool=pool,
            emit_frames=emit,
        )
        runner.capture()
        assert runner.enabled, runner.stats()
        runners[name] = runner
        print("CAPTURED", name, flush=True)
    generator = torch.Generator(device=device).manual_seed(3176)
    records = []
    history = {slot: [] for slot in range(4)}
    last_by_slot = {}

    def step(name, slots, width, valid=None):
        codes = torch.randint(
            decoder.config.codebook_size,
            (len(slots), decoder.config.num_quantizers, width),
            device=device,
            generator=generator,
        )
        reference_state = buffered.gather(slots)
        if valid is None:
            expected = incremental.decode(codes, reference_state)
        else:
            expected = incremental.decode_tail(
                codes, torch.tensor(valid, device=device), reference_state, 1
            )
        buffered.scatter(slots, reference_state)
        actual = runners[name].decode_slots(codes, slots, valid_frames=valid)
        assert actual is not None
        actual = actual.clone()
        torch.cuda.synchronize()
        assert torch.equal(
            rings.storage.frame_positions, buffered.storage.frame_positions
        )
        assert actual.shape == expected.shape and torch.isfinite(actual).all()
        record = {
            "runner": name,
            "slots": slots,
            "width": width,
            "valid": valid,
            **metrics(actual, expected),
        }
        fp_rows = []
        for row, slot in enumerate(slots):
            history[slot].append(
                codes[row : row + 1, :, : width if valid is None else valid[row]]
            )
            frames = width if valid is None else 1
            fp_rows.append(
                fp32(torch.cat(history[slot], dim=-1))[
                    ..., -frames * decoder.total_upsample :
                ]
            )
            last_by_slot[slot] = actual[row : row + 1].clone()
        fp_pcm = torch.cat(fp_rows)
        record["ring_to_fp32"] = metrics(actual, fp_pcm)
        record["buffered_to_fp32"] = metrics(expected, fp_pcm)
        assert record["snr_db"] > 30, record
        assert (
            record["ring_to_fp32"]["snr_db"]
            >= record["buffered_to_fp32"]["snr_db"] - 1.5
        ), record
        # Verify the logical retained K/V independently of the attention's output.
        for row, slot in enumerate(slots):
            position = int(rings.storage.frame_positions[slot])
            count = min(position, 71)
            indices = (
                slot * rings.ring_frames
                + torch.arange(position - count, position, device=device)
                % rings.ring_frames
            )
            for layer, ring in enumerate(rings.key_rings):
                actual_keys = ring[indices].view(count, -1, 64).transpose(0, 1)
                expected_keys = buffered.storage.transformer_keys[layer][
                    slot, :, -count:
                ]
                torch.testing.assert_close(
                    actual_keys, expected_keys, rtol=0.03, atol=0.03
                )
        assert int(rings.storage.frame_positions[4]) == 0
        assert not any(
            torch.count_nonzero(ring.view(6, 79, -1)[4])
            for ring in (*rings.key_rings, *rings.value_rings)
        )
        records.append(record)
        print(json.dumps(record), flush=True)

    step("cold", [0, 1, 2], 1)
    step("tail", [3], 581, [581])
    step("window", [2, 0, 3], 8)
    step("window", [1, 3, 0], 64)
    step("tail", [2, 3, 1], 581, [581, 579, 577])
    for index in range(12):
        step("window", [3, 0, 2] if index % 2 else [2, 1], 8)
    for arena in arenas:
        arena.zero_slot(0)
    history[0] = []
    step("cold", [0], 2)
    step("window", [0, 1], 8)
    result = {
        "ring_frames": rings.ring_frames,
        "ring_bytes_per_slot": rings.bytes_per_slot,
        "buffered_bytes_per_slot": buffered.bytes_per_slot,
        "records": records,
        "runners": {name: runner.stats() for name, runner in runners.items()},
    }
    (ROOT / "graph_probe.json").write_text(json.dumps(result, indent=2))
    print("TRAINED RING GRAPH CHECKS COMPLETE", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
