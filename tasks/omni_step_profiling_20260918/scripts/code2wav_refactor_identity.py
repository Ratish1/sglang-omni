"""Bit identity of Qwen3-Omni code2wav across a refactor: the module at OLD_FILE against
the importable one.

Builds two code2wav models from the real Qwen3-Omni code2wav config with one seeded set
of weights, runs use_channels_last on each (old code on one, new on the other), checks the
conv weights are equal with equal strides, then runs both channels-last forwards on the
same codes and checks torch.equal. Two builds: the HF pre-transformer and SnakeBeta, and
serving's, which since #2466 also swaps in the fused pre-transformer when the module has
one, then fuses the decoder's SnakeBeta.

usage: python code2wav_refactor_identity.py OLD_FILE
"""

from __future__ import annotations

import importlib.util
import sys

import torch
from transformers import AutoConfig

from sglang_omni.models.qwen3_omni.components import code2wav as new_code2wav
from sglang_omni.platforms import current_platform
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SHAPES = ((1, 8), (1, 33), (3, 24), (4, 120))


def load_old(path: str):
    spec = importlib.util.spec_from_file_location("old_code2wav", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(module, config, state_dict, fused: bool):
    model = module.Qwen3OmniCode2Wav._from_config(config)  # noqa: leading-underscore
    model.load_state_dict(state_dict)
    model = model.to("cuda", torch.bfloat16).eval()
    model.use_channels_last()
    if fused:
        if hasattr(model, "use_fused_transformer"):
            model.use_fused_transformer(
                current_platform.get_joint_rope_inplace_kernel()
            )
        else:
            pass
        fuse_vocoder_decoder(model.decoder)
    else:
        pass
    return model


def main() -> None:
    old_code2wav = load_old(sys.argv[1])
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).code2wav_config
    torch.manual_seed(0)
    state_dict = (
        new_code2wav.Qwen3OmniCode2Wav._from_config(  # noqa: leading-underscore
            config
        ).state_dict()
    )
    for fused in (False, True):
        old_model = build(old_code2wav, config, state_dict, fused)
        new_model = build(new_code2wav, config, state_dict, fused)
        weights_equal = all(
            torch.equal(old.weight, new.weight)
            and old.weight.stride() == new.weight.stride()
            for old, new in zip(old_model.modules(), new_model.modules())
            if isinstance(old, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))
        )
        print(
            f"serving build {fused} (fused transformer "
            f"{type(new_model.pre_transformer).__name__}): conv weights equal with "
            f"equal strides {weights_equal}"
        )
        for batch, frames in SHAPES:
            codes = torch.randint(
                0,
                int(config.codebook_size),
                (batch, int(config.num_quantizers), frames),
                device="cuda",
            )
            with torch.inference_mode():
                old_waveform = old_model(codes)
                new_waveform = new_model(codes)
            print(
                f"  codes {tuple(codes.shape)}: waveform {tuple(new_waveform.shape)} "
                f"bit identical {torch.equal(old_waveform, new_waveform)}"
            )


if __name__ == "__main__":
    main()
