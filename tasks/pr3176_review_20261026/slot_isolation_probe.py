"""A deterministic interleaving distinguishes real slot ownership from cohort row routing."""

import json
from dataclasses import replace
from pathlib import Path

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

apply_qwen_tts_transformers_compatibility_patches()
from qwen_tts import Qwen3TTSTokenizer

with torch.inference_mode():
    torch.manual_seed(3176)
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    decoder = tokenizer.model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    codes = [
        torch.randint(
            decoder.config.codebook_size,
            (1, decoder.config.num_quantizers, width),
            device="cuda",
        )
        for width in (12, 64, 8)
    ]
    original = Qwen3TTSCodecStateArena.gather_by_index
    records = []
    for mutant in (False, True):

        def gather(self, index):
            state = original(self, index)
            if mutant and state.kv_rings is not None:
                state.kv_rings = replace(
                    state.kv_rings,
                    slot_rows=torch.arange(index.numel(), device=index.device)
                    * self.ring_frames,
                )
            return state

        Qwen3TTSCodecStateArena.gather_by_index = gather
        arenas = [
            Qwen3TTSCodecStateArena(
                incremental,
                num_slots=4,
                device=torch.device("cuda", 0),
                dtype=torch.bfloat16,
                in_place_frames=8,
            )
            for _ in range(2)
        ]
        for arena in arenas:
            assert [arena.acquire() for _ in range(4)] == [0, 1, 2, 3]

        def step(arena, slot, frames):
            state = arena.gather([slot])
            waveform = incremental.decode(frames, state).clone()
            arena.scatter([slot], state)
            return waveform

        alone, mixed = arenas
        assert torch.equal(step(alone, 3, codes[0]), step(mixed, 3, codes[0]))
        step(mixed, 1, codes[1])
        expected = step(alone, 3, codes[2])
        actual = step(mixed, 3, codes[2])
        equal = torch.equal(actual, expected)
        records.append(
            {
                "mutant": mutant,
                "continuation_equal": equal,
                "max_pcm_difference": float((actual - expected).abs().max()),
            }
        )
        assert equal != mutant, records[-1]
    Qwen3TTSCodecStateArena.gather_by_index = original
    Path("/data/review3176oct26/slot_isolation.json").write_text(
        json.dumps(records, indent=2)
    )
    print(json.dumps(records), flush=True)
