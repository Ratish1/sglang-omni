"""Layer by layer, the fused decoder transformer (V3) against the plain bf16 step, both against fp32, on the trained
checkpoint and real codes.

usage: python3 v3_layer_probe.py OUT_JSON CARD   (from the V3 tree, so its incremental codec imports)

The speech tokenizer loads as the vocoder loads it (load_qwen3_tts_tokenizer, fp32 on cuda). The eight longest of the
first 64 SeedTTS EN references (one row starting at absolute position 4,000) are encoded to codes and turned into transformer inputs as the decoder does (quantizer decode, pre_conv),
then streamed through the incremental step as the vocoder streams: a 16-frame first chunk and 4-frame follow-ups, rows
at their own positions, with a per-row valid tail on the last step. Three copies of the trained transformer run every
step: fp32 (the reference), the plain bf16 step and the fused bf16 step. Per step and per layer it records the
relative L2 error against fp32 of the retained keys and values, and of the output of the transformer cut after that
layer (copies holding the first k layers, k = 1 to 8). It also checks that the bf16 decoder selects the fused step.
"""

import copy
import json
import sys

import soundfile
import torch

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.incremental_codec import (
    FusedIncrementalTransformer,
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
    incremental_transformer,
)
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ROWS = 8
FIRST_CHUNK = 16
FOLLOWUP = 4


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected).norm() / expected.norm())


def truncated(transformer: torch.nn.Module, layers: int) -> torch.nn.Module:
    copied = copy.deepcopy(transformer)
    copied.layers = copied.layers[:layers]
    return copied


def main() -> None:
    device = f"cuda:{sys.argv[2]}"
    torch.cuda.set_device(device)
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=device, dtype="float32", attn_implementation=None)
    decoder = tokenizer.model.decoder
    samples = sorted(
        load_seedtts_samples(SEEDTTS_DATASET_ID, 64, split="en"),
        key=lambda sample: soundfile.info(sample.ref_audio).duration,
    )[-ROWS:]
    audios = [soundfile.read(sample.ref_audio, dtype="float32")[0] for sample in samples]
    rate = soundfile.info(samples[0].ref_audio).samplerate
    codes = tokenizer.encode(audios, sr=rate).audio_codes
    frames = min(int(code.shape[0]) for code in codes)
    with torch.no_grad():
        stacked = torch.stack([code[:frames].t() for code in codes]).to(device)
        inputs = decoder.pre_conv(decoder.quantizer.decode(stacked)).transpose(1, 2)
    layer_count = len(decoder.pre_transformer.layers)
    depths = {}
    for layers in range(1, layer_count + 1):
        reference = truncated(decoder.pre_transformer, layers).eval()
        depths[layers] = {
            "reference": reference,
            "plain": copy.deepcopy(reference).bfloat16(),
            "fused": FusedIncrementalTransformer(copy.deepcopy(reference).bfloat16()),
        }
    starts = torch.arange(ROWS, device=device) * 7
    starts[-1] = 4000
    states = {
        (layers, path): Qwen3TTSIncrementalCodecState(frame_positions=starts.clone())
        for layers in depths
        for path in ("reference", "plain", "fused")
    }
    widths = [FIRST_CHUNK]
    while sum(widths) + FOLLOWUP <= frames:
        widths.append(FOLLOWUP)
    steps = []
    offset = 0
    with torch.no_grad():
        for index, width in enumerate(widths):
            chunk = inputs[:, offset : offset + width]
            valid = (
                torch.tensor([width - (row % 3) for row in range(ROWS)], device=device)
                if index == len(widths) - 1
                else None
            )
            record = {"step": index, "width": width, "output": {}, "keys": {}, "values": {}}
            for layers, paths in depths.items():
                expected = incremental_transformer(paths["reference"], chunk, states[(layers, "reference")], valid)
                plain = incremental_transformer(paths["plain"], chunk.bfloat16(), states[(layers, "plain")], valid)
                fused = paths["fused"](chunk.bfloat16(), states[(layers, "fused")], valid)
                record["output"][layers] = {
                    "plain": relative_error(plain, expected),
                    "fused": relative_error(fused, expected),
                }
            full = layer_count
            for kind in ("keys", "values"):
                truths = getattr(states[(full, "reference")], f"transformer_{kind}")
                plains = getattr(states[(full, "plain")], f"transformer_{kind}")
                fuseds = getattr(states[(full, "fused")], f"transformer_{kind}")
                record[kind] = {
                    layer: {
                        "plain": relative_error(plains[layer], truth),
                        "fused": relative_error(fuseds[layer], truth),
                    }
                    for layer, truth in truths.items()
                }
            for state in states.values():
                state.advance(width)
            offset += width
            steps.append(record)
    bf16 = load_qwen3_tts_tokenizer(MODEL, device=device, dtype="bfloat16", attn_implementation=None)
    selects_fused = Qwen3TTSIncrementalDecoder(bf16.model.decoder).fused_transformer is not None
    worst = {}
    for record in steps:
        for section in ("output", "keys", "values"):
            for layer, errors in record[section].items():
                key = f"{section} {layer}"
                ratio = errors["fused"] / errors["plain"]
                current = worst.get(key, {"ratio": 0.0})
                if ratio > current["ratio"]:
                    worst[key] = {"ratio": ratio, "step": record["step"], **errors}
                else:
                    pass
    result = {
        "frames": frames,
        "rows": ROWS,
        "widths": widths,
        "selects_fused": selects_fused,
        "worst_fused_over_plain": worst,
        "steps": steps,
    }
    json.dump(result, open(sys.argv[1], "w"), indent=1)
    print(f"frames {frames}, steps {len(widths)}, bf16 decoder selects the fused step: {selects_fused}")
    print("| layer | output fused / plain (worst step) | keys fused / plain | values fused / plain |")
    print("| --- | --- | --- | --- |")
    for layer in range(1, layer_count + 1):
        output = worst[f"output {layer}"]
        keys = worst[f"keys {layer - 1}"]
        values = worst[f"values {layer - 1}"]
        print(
            f"| {layer} | {output['fused']:.5f} / {output['plain']:.5f} ({output['ratio']:.3f}) | "
            f"{keys['fused']:.5f} / {keys['plain']:.5f} ({keys['ratio']:.3f}) | "
            f"{values['fused']:.5f} / {values['plain']:.5f} ({values['ratio']:.3f}) |"
        )


if __name__ == "__main__":
    main()
