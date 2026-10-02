"""Bit identity of the Qwen3-TTS incremental decoder across a refactor: the module at
OLD_FILE against the importable one.

Builds both incremental decoders over one real Qwen3-TTS decoder, checks their
channels-last weights hold the same values with the same strides,
then for widths 1, 2 and 8 frames at buckets 1, 2, 4 and 8 decodes three chained steps
from a full-width state with each and checks torch.equal on every waveform and every
conv history and transposed-conv overlap.

usage: python incremental_codec_refactor_identity.py OLD_FILE
"""

from __future__ import annotations

import importlib.util
import sys

import torch

from sglang_omni.models.qwen3_tts import incremental_codec as new_codec
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WIDTHS = (1, 2, 8)
BUCKETS = (1, 2, 4, 8)
STEPS = 3


def load_old(path: str):
    spec = importlib.util.spec_from_file_location("old_incremental_codec", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    old_codec = load_old(sys.argv[1])
    tokenizer = load_qwen3_tts_tokenizer(
        MODEL, device="cuda", dtype="bfloat16", attn_implementation=None
    )
    decoder = tokenizer.model.decoder
    config = getattr(tokenizer.model.config, "decoder_config", tokenizer.model.config)
    old_decoder = old_codec.Qwen3TTSIncrementalDecoder(decoder)
    new_decoder = new_codec.Qwen3TTSIncrementalDecoder(decoder)
    assert old_decoder.channels_last_weights is not None
    assert new_decoder.channels_last_weights is not None
    assert (
        old_decoder.channels_last_weights.keys()
        == new_decoder.channels_last_weights.keys()
    )
    weights_equal = all(
        torch.equal(old_weight, new_decoder.channels_last_weights[key])
        and old_weight.stride() == new_decoder.channels_last_weights[key].stride()
        for key, old_weight in old_decoder.channels_last_weights.items()
    )
    print(
        f"channels-last weights: {len(new_decoder.channels_last_weights)} keys, "
        f"equal values and strides {weights_equal}"
    )
    torch.manual_seed(0)
    all_identical = True
    for frames in WIDTHS:
        for rows in BUCKETS:
            old_state = old_decoder.init_state(
                rows, device="cuda", dtype=torch.bfloat16
            )
            new_state = new_decoder.init_state(
                rows, device="cuda", dtype=torch.bfloat16
            )
            identical = True
            for _ in range(STEPS):
                codes = torch.randint(
                    0,
                    int(config.codebook_size),
                    (rows, int(config.num_quantizers), frames),
                    device="cuda",
                )
                with torch.inference_mode():
                    old_waveform = old_decoder.decode(codes, old_state)
                    new_waveform = new_decoder.decode(codes, new_state)
                identical &= torch.equal(old_waveform, new_waveform)
                identical &= all(
                    torch.equal(old_state.conv_histories[key], tensor)
                    for key, tensor in new_state.conv_histories.items()
                )
                identical &= all(
                    torch.equal(old_state.transconv_overlaps[key], tensor)
                    for key, tensor in new_state.transconv_overlaps.items()
                )
            all_identical &= identical
            print(
                f"width {frames} bucket {rows}: {STEPS} steps bit identical {identical}"
            )
    print(f"all shapes bit identical {all_identical}")


if __name__ == "__main__":
    main()
