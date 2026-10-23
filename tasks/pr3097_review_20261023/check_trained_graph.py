"""Exercise compiled, padded, reordered vocoder replays on the trained tokenizer."""

from __future__ import annotations

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


def main() -> None:
    torch.manual_seed(3097)
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    decoder = tokenizer.model.decoder.eval()
    replaced = fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    device = torch.device("cuda", 0)
    arenas = [
        Qwen3TTSCodecStateArena(
            incremental, num_slots=4, device=device, dtype=torch.bfloat16
        )
        for _ in range(2)
    ]
    graph_arena, reference_arena = arenas
    for arena in arenas:
        assert [arena.acquire() for _ in range(3)] == [0, 1, 2]
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        incremental,
        device=device,
        dtype=torch.bfloat16,
        num_quantizers=decoder.config.num_quantizers,
        mode="warm",
        fresh_frames=(8,),
        batch_sizes=(4,),
        min_free_gb=0,
        compile_fresh_frames=(8,),
        arena=graph_arena,
        graph_pool=IncrementalCodecGraphPool(stream_priority=0),
    )
    runner.capture()
    assert runner.enabled, runner.stats()
    records = []
    positions = [0, 0]
    generator = torch.Generator(device=device).manual_seed(3097)
    for slots in ([0, 1], [1, 0], [1], [0, 1]):
        codes = torch.randint(
            decoder.config.codebook_size,
            (len(slots), decoder.config.num_quantizers, 8),
            device=device,
            generator=generator,
        )
        state = reference_arena.gather(slots)
        expected = incremental.decode(codes, state)
        reference_arena.scatter(slots, state)
        actual = runner.decode_slots(codes, slots)
        assert actual is not None
        torch.cuda.synchronize()
        signal = expected.float().square().sum()
        error = (actual.float() - expected.float()).square().sum()
        snr_db = 10 * torch.log10(signal / error.clamp_min(1e-30))
        for slot in slots:
            positions[slot] += 8
        assert graph_arena.storage.frame_positions[:2].tolist() == positions
        assert reference_arena.storage.frame_positions[:2].tolist() == positions
        assert actual.shape == expected.shape
        for pool in (
            *graph_arena.storage.conv_histories.values(),
            *graph_arena.storage.transconv_overlaps.values(),
        ):
            assert not pool[2].any()
        assert not graph_arena.storage.frame_positions[2]
        record = {
            "slots": slots,
            "positions": positions.copy(),
            "snr_to_eager_db": snr_db.item(),
            "max_absolute_pcm_error": (actual.float() - expected.float())
            .abs()
            .max()
            .item(),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "replaced_activations": replaced,
        "compiled_shapes": sorted(incremental.compiled_shapes),
        "runner": runner.stats(),
        "replays": records,
    }
    Path("/data/review3097/trained-graph.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        "trained compiled graph, slot ownership and reordered replays checked",
        flush=True,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
