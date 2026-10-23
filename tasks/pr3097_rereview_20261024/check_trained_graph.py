"""Exercise compiled, padded, reordered vocoder replays on the trained tokenizer."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

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


def parent_module(
    name: str, package: str = "sglang_omni/models/qwen3_tts"
) -> ModuleType:
    source = subprocess.run(
        [
            "git",
            "show",
            f"a248446c7802cdca8525a72bfd7e705b0b60fe9a:{package}/{name}.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = Path(f"/data/review3097oct24/parent_{name}.py")
    path.write_text(source)
    specification = importlib.util.spec_from_file_location(
        f"pr3097_parent_{name}", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", action="store_true")
    arguments = parser.parse_args()
    torch.manual_seed(3097)
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    decoder = tokenizer.model.decoder.eval()
    reference_decoder = copy.deepcopy(decoder).float()
    replaced = fuse_vocoder_decoder(decoder)
    if arguments.parent:
        parent_decoder = parent_module("incremental_codec")
        parent_arena = parent_module("codec_state_arena")
        parent_bias = parent_module("bias_snake_beta", "sglang_omni/kernels")
        parent_exchange = parent_module("codec_state", "sglang_omni/kernels")
        parent_decoder.fused_bias_snake_beta = parent_bias.fused_bias_snake_beta
        parent_decoder.history_exchange = parent_exchange.history_exchange
        parent_decoder.overlap_exchange = parent_exchange.overlap_exchange
        incremental = parent_decoder.Qwen3TTSIncrementalDecoder(decoder)
        arena_class = parent_arena.Qwen3TTSCodecStateArena
    else:
        incremental = Qwen3TTSIncrementalDecoder(decoder)
        arena_class = Qwen3TTSCodecStateArena
    device = torch.device("cuda", 0)
    arenas = [
        arena_class(incremental, num_slots=4, device=device, dtype=torch.bfloat16)
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
    code_history: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    pcm_snapshots: list[torch.Tensor] = []
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
        fp32_rows = []
        for row, slot in enumerate(slots):
            code_history[slot].append(codes[row : row + 1])
            fp32_rows.append(
                reference_decoder(torch.cat(code_history[slot], dim=-1))[
                    ..., -8 * decoder.total_upsample :
                ]
            )
        fp32_pcm = torch.cat(fp32_rows)
        fp32_signal = fp32_pcm.square().sum()
        fp32_error = (actual.float() - fp32_pcm).square().sum()
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
            if not arguments.parent:
                assert not pool[graph_arena.scratch_slot].any()
        assert not graph_arena.storage.frame_positions[2]
        record = {
            "tree": "parent" if arguments.parent else "head",
            "slots": slots,
            "positions": positions.copy(),
            "snr_to_eager_db": snr_db.item(),
            "snr_to_fp32_db": (
                10 * torch.log10(fp32_signal / fp32_error.clamp_min(1e-30))
            ).item(),
            "max_absolute_pcm_error": (actual.float() - expected.float())
            .abs()
            .max()
            .item(),
        }
        records.append(record)
        pcm_snapshots.append(actual.clone())
        print(json.dumps(record), flush=True)
    result = {
        "replaced_activations": replaced,
        "compiled_shapes": sorted(incremental.compiled_shapes),
        "runner": runner.stats(),
        "replays": records,
    }
    tree = "parent" if arguments.parent else "head"
    torch.save(pcm_snapshots, f"/data/review3097oct24/trained-graph-{tree}-pcm.pt")
    Path(f"/data/review3097oct24/trained-graph-{tree}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        "trained compiled graph, slot ownership and reordered replays checked",
        flush=True,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
