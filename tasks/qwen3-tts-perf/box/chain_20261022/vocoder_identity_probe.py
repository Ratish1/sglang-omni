"""The Qwen3-TTS vocoder's incremental decodes on the trained speech tokenizer, loaded and fused as serving loads it
(bf16, SnakeBeta fused), on the tree on PYTHONPATH; one tree per process. Seeded codes; three scenarios on full-width
states as the arena holds them: streaming (4 rows, a 16-frame first chunk then follow-ups of 1 to 8 frames), tail (3
rows of 40, 25 and 30 valid frames, then a follow-up of 8) and compiled (the streaming scenario with its 8-frame
steps through the compiled kernel, as the warm width-8 graphs serve). Waveforms and every conv history, overlap and
K/V tensor are saved, conv state in the (rows, channels, samples) layout, so two trees compare with --compare.

usage: PYTHONPATH=<tree> python3 vocoder_identity_probe.py OUT.pt [--card 3]
       python3 vocoder_identity_probe.py --compare A.pt B.pt
"""

import argparse

import torch

from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
STREAM_PARTS = (16, 1, 2, 4, 8, 8, 8, 3)
TAIL_WIDTHS = (40, 25, 30)


def canonical_state(incremental: Qwen3TTSIncrementalDecoder, state, prefix: str) -> dict[str, torch.Tensor]:
    spec = incremental.state_spec()
    tensors = {}
    for pools, entries, name in (
        (state.conv_histories, spec.conv_histories, "history"),
        (state.transconv_overlaps, spec.transconv_overlaps, "overlap"),
    ):
        for key, channels, _ in entries:
            value = pools[key]
            tensors[f"{prefix}.{name}.{key}"] = (value if value.shape[1] == channels else value.transpose(1, 2)).cpu()
    for layer, value in state.transformer_keys.items():
        tensors[f"{prefix}.key.{layer}"] = value.cpu()
    for layer, value in state.transformer_values.items():
        tensors[f"{prefix}.value.{layer}"] = value.cpu()
    tensors[f"{prefix}.positions"] = state.frame_positions.cpu()
    return tensors


def stream(incremental, codes, device, dtype, prefix: str, compiled: bool) -> dict[str, torch.Tensor]:
    state = incremental.init_state(codes.shape[0], device=device, dtype=dtype)
    outputs = {}
    offset = 0
    for index, width in enumerate(STREAM_PARTS):
        part = codes[..., offset : offset + width].contiguous()
        outputs[f"{prefix}.wave.{index}"] = incremental.decode(part, state, compiled=compiled and width == 8).cpu()
        offset += width
    outputs.update(canonical_state(incremental, state, prefix))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?")
    parser.add_argument("--card", type=int, default=3)
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.compare:
        first, second = (torch.load(path) for path in args.compare)
        assert first.keys() == second.keys(), "the probes saved different entries"
        differing = 0
        for key in first:
            if torch.equal(first[key], second[key]):
                pass
            else:
                differing += 1
                delta = (first[key].float() - second[key].float()).abs().max().item()
                print(f"differs: {key} max |delta| {delta:.3e}")
        print(f"{len(first) - differing} of {len(first)} entries bit identical")
        return
    else:
        pass

    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    decoder = tokenizer.model.decoder.eval()
    print(f"fused SnakeBeta modules: {fuse_vocoder_decoder(decoder)}")
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    dtype = torch.bfloat16
    config = decoder.config
    generator = torch.Generator(device="cpu").manual_seed(1022)
    stream_codes = torch.randint(
        config.codebook_size, (4, config.num_quantizers, sum(STREAM_PARTS)), generator=generator
    ).to(device)
    tail_codes = torch.randint(
        config.codebook_size, (len(TAIL_WIDTHS), config.num_quantizers, max(TAIL_WIDTHS)), generator=generator
    ).to(device)
    follow_codes = torch.randint(
        config.codebook_size, (len(TAIL_WIDTHS), config.num_quantizers, 8), generator=generator
    ).to(device)

    outputs = {}
    with torch.inference_mode():
        outputs.update(stream(incremental, stream_codes, device, dtype, "stream", compiled=False))

        state = incremental.init_state(len(TAIL_WIDTHS), device=device, dtype=dtype)
        valid = torch.tensor(TAIL_WIDTHS, device=device)
        outputs["tail.wave"] = incremental.decode_tail(tail_codes, valid, state, 1).cpu()
        outputs["tail.follow"] = incremental.decode(follow_codes, state).cpu()
        outputs.update(canonical_state(incremental, state, "tail"))

        trace_state = incremental.init_state(4, device=device, dtype=dtype)
        incremental.precompile(stream_codes[..., :8].contiguous(), trace_state)
        outputs.update(stream(incremental, stream_codes, device, dtype, "compiled", compiled=True))
    torch.cuda.synchronize(device)
    torch.save(outputs, args.out)
    print(f"saved {len(outputs)} entries to {args.out}")


if __name__ == "__main__":
    main()
