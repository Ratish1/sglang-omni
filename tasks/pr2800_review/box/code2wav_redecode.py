"""Whether code2wav decoded each served request's codes correctly under load: every request's talker codes (saved by
code2wav_dump/sitecustomize.py while a server ran) decoded again alone, two ways, against the audio it served.

Run on the box from an arm's tree (its sglang_omni and benchmarks):
    cd <tree> && PYTHONPATH=<tree> CUDA_VISIBLE_DEVICES=<card> python3 code2wav_redecode.py --arm-dir <run>/<arm>
        --model marksverdhei/Qwen3-Omni-30B-A3B-FP8
For each saved request: omni's whole-sequence forward (load_code2wav_model, the loader the server uses, codes in one
call, no slots) and HF's own Qwen3OmniMoeCode2Wav on the same checkpoint weights decoded as HF's generate does
(chunked_decode, 300 frames, 25 frames of left context). Each is compared with the served audio: length, max
absolute difference and SNR. The re-decoded audio is written next to the dumps for transcription. Writes
<arm-dir>/redecode.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile
import torch
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import load_code2wav_model
from sglang_omni.models.weight_loader import load_module

CHUNK_FRAMES = 300
LEFT_CONTEXT_FRAMES = 25


def snr_db(value: np.ndarray, reference: np.ndarray) -> float:
    length = min(len(value), len(reference))
    noise = float(np.sum((value[:length] - reference[:length]) ** 2))
    signal = float(np.sum(reference[:length] ** 2))
    return math.inf if noise == 0 else 10 * math.log10(max(signal, 1e-12) / noise)


def compare(value: np.ndarray, served: np.ndarray) -> dict[str, float]:
    length = min(len(value), len(served))
    return dict(
        length_delta=int(len(value) - len(served)),
        max_abs_diff=float(np.max(np.abs(value[:length] - served[:length]))) if length else 0.0,
        snr_db=snr_db(value, served))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    omni = load_code2wav_model(args.model, device="cuda", dtype="bfloat16")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hf = Qwen3OmniMoeCode2Wav._from_config(config.code2wav_config)  # noqa: leading-underscore
    hf = load_module(hf, args.model, prefix="code2wav.", dtype=torch.bfloat16, device="cuda", strict=False).eval()
    out_dir = args.arm_dir / "redecoded"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for path in sorted((args.arm_dir / "code2wav_dump").glob("*.pt")):
        dump = torch.load(path, weights_only=False)
        if dump["codes"].numel() == 0:
            continue
        else:
            pass
        codes = dump["codes"].to("cuda").long().transpose(0, 1).unsqueeze(0)
        served = dump["audio"]
        with torch.no_grad():
            whole = omni(codes).reshape(-1).float().cpu().numpy()
            reference = hf.chunked_decode(
                codes, chunk_size=CHUNK_FRAMES, left_context_size=LEFT_CONTEXT_FRAMES).reshape(-1).float().cpu().numpy()
        request_id = dump["request_id"]
        soundfile.write(out_dir / f"{request_id}__omni_whole.wav", whole, dump["sample_rate"])
        soundfile.write(out_dir / f"{request_id}__hf.wav", reference, dump["sample_rate"])
        rows.append(dict(
            request_id=request_id, frames=int(codes.shape[-1]), served_seconds=round(len(served) / dump["sample_rate"], 2),
            omni_whole_vs_served=compare(whole, served), hf_vs_served=compare(reference, served),
            hf_vs_omni_whole=compare(reference, whole)))
        print(json.dumps(rows[-1]), flush=True)
    (args.arm_dir / "redecode.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
