"""Time the whole-utterance decode (chunked_decode, the non-streaming path) with the conv
weights as loaded and with them relaid channels last in place, as the incremental
decoder does, and compare the waveforms.

Each cell is the median ms of 10 eager decodes of random codes. The waveform columns are
the max abs difference between the two layouts and whether they are equal.

usage: python vocoder_full_decode_layout.py
"""

from __future__ import annotations

import statistics

import torch

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
FRAMES = (24, 50, 150)
BATCHES = (1, 8, 16)
REPEATS = 10


def decode_ms(decoder, codes) -> tuple[float, torch.Tensor]:
    with torch.inference_mode():
        decoder.chunked_decode(codes)
        torch.cuda.synchronize()
        times = []
        for _ in range(REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            waveform = decoder.chunked_decode(codes)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    return statistics.median(times), waveform


def main() -> None:
    from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

    tokenizer = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="bfloat16", attn_implementation=None
    )
    decoder = tokenizer.model.decoder
    config = getattr(tokenizer.model.config, "decoder_config", tokenizer.model.config)
    torch.manual_seed(0)
    cases = {
        (frames, rows): torch.randint(
            0,
            int(config.codebook_size),
            (rows, int(config.num_quantizers), frames),
            device="cuda",
        )
        for frames in FRAMES
        for rows in BATCHES
    }
    as_loaded = {case: decode_ms(decoder, codes) for case, codes in cases.items()}
    for subtree in (decoder.pre_conv, decoder.upsample, decoder.decoder):
        for layer in subtree.modules():
            if isinstance(layer, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                layer.weight.data = (
                    layer.weight.data.transpose(1, 2).contiguous().transpose(1, 2)
                )
            else:
                pass
    relaid = {case: decode_ms(decoder, codes) for case, codes in cases.items()}
    print(torch.cuda.get_device_name(0))
    print(
        f"{'frames':>7}{'rows':>6}{'as loaded ms':>14}{'relaid ms':>11}{'max diff':>11}{'equal':>7}"
    )
    for case in cases:
        loaded_ms, loaded_waveform = as_loaded[case]
        relaid_ms, relaid_waveform = relaid[case]
        difference = float(
            (loaded_waveform.float() - relaid_waveform.float()).abs().max()
        )
        print(
            f"{case[0]:>7}{case[1]:>6}{loaded_ms:>14.2f}{relaid_ms:>11.2f}"
            f"{difference:>11.2e}{str(torch.equal(loaded_waveform, relaid_waveform)):>7}"
        )


if __name__ == "__main__":
    main()
