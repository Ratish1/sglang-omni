"""Measure the native compatibility decode after the startup preparation changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from checkpoint_probe import make_inputs

from sglang_omni.models.fun_cosyvoice3 import stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=("parent", "head"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2944)
    checkpoint = stages.resolve_checkpoint("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    flow, hift = stages.load_cosyvoice3_flow_hift(
        checkpoint, device="cuda:0", fp16=False, autocast_dtype=torch.bfloat16
    )
    stages.patch_chunk_mask()
    if args.label == "parent":
        stages.compile_dit_backbone(flow, autocast_dtype=torch.bfloat16)
    else:
        stages.set_dit_compile_config()
    vocoder = stages.CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    item = make_inputs((32,), (7,))[0]
    native_inference = flow.flow.inference
    mels = []

    def observe_inference(**kwargs):
        output = native_inference(**kwargs)
        mels.append(output[0].clone())
        return output

    torch.manual_seed(2945)
    with (
        torch.inference_mode(),
        vocoder.stream_context,
        patch.object(flow.flow, "inference", observe_inference),
    ):
        wav = vocoder.token2wav(
            item.token, item.prompt_token, item.prompt_feat, item.embedding
        )
    assert len(mels) == 1
    outputs = {"mel": mels[0].cpu(), "wav": wav.cpu()}
    if args.label == "parent":
        estimator = flow.decoder.estimator
        estimator.forward = estimator.forward._torchdynamo_orig_callable
        torch.manual_seed(2945)
        with (
            torch.inference_mode(),
            vocoder.stream_context,
            patch.object(flow.flow, "inference", observe_inference),
        ):
            eager_wav = vocoder.token2wav(
                item.token, item.prompt_token, item.prompt_feat, item.embedding
            )
        outputs["eager_mel"] = mels[1].cpu()
        outputs["eager_wav"] = eager_wav.cpu()
    else:
        pass
    torch.save(
        outputs,
        Path(f"/data/review2944/{args.label}_fallback.pt"),
    )
    print(
        json.dumps(
            {"label": args.label, "mel": list(mels[0].shape), "wav": list(wav.shape)}
        )
    )


if __name__ == "__main__":
    main()
