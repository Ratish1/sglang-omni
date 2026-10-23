"""Which op of the speech tokenizer depends on the padded width under TF32 (10-23): the real V3 model on references
whose served codes moved, run eagerly at the reference's own mel width and at its graph's padded width, with TF32
matmuls on (and off, as the control); every intermediate of the conv front end and of the first and last blocks is
compared over the real positions (largest absolute difference, relative to the value's scale), then the codes.
usage: CUDA_VISIBLE_DEVICES=<card> PYTHONPATH=<tree with L3>:<CosyVoice dirs> python3 tokenizer_width_ops.py [<references>]"""

import bisect
import os
import sys

import torch
import torch.nn.functional as F
import whisper
from sglang.srt.arg_groups.cuda_graph_hook import generate_prefill_cuda_graph_batch_sizes

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.fun_cosyvoice3.request_builders import load_prompt_audio
from sglang_omni.models.fun_cosyvoice3.speech_tokenizer import (
    FSQ_SCALE,
    MAX_FRAMES,
    conv_frames,
    load_speech_tokenizer,
    rotate,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def block_steps(block, x, position_mask, attention_bias, cos, sin):
    steps = {}
    heads = (1, x.shape[1], block.head_count, -1)
    hidden = block.attn_ln(x)
    steps["attn_ln"] = hidden
    query = rotate(block.query(hidden).view(heads), cos, sin)
    key = rotate(block.key(hidden).view(heads), cos, sin)
    value = block.value(hidden)
    steps["query"], steps["key"], steps["value"] = query, key, value
    memory_input = value * position_mask
    left = (block.fsmn.kernel_size[0] - 1) // 2
    right = block.fsmn.kernel_size[0] - 1 - left
    memory = block.fsmn(F.pad(memory_input.transpose(1, 2), (left, right))).transpose(1, 2)
    memory = (memory + memory_input) * position_mask
    steps["fsmn"] = memory
    scale = query.shape[-1] ** -0.25
    scores = (query.permute(0, 2, 1, 3) * scale) @ (key.permute(0, 2, 3, 1) * scale)
    steps["scores"] = scores
    weights = torch.softmax(scores + attention_bias, dim=-1)
    steps["weights"] = weights
    attended = weights @ value.view(heads).transpose(1, 2)
    steps["attended"] = attended
    out = block.out(attended.transpose(1, 2).reshape_as(x))
    steps["out"] = out
    x = x + (out + memory)
    mlp = block.mlp_out(F.gelu(block.mlp_in(block.mlp_ln(x))))
    steps["mlp"] = mlp
    return x + mlp, steps


def trace(model, mel, frames):
    device = mel.device
    steps = {}
    x = F.gelu(model.conv1(mel * (torch.arange(mel.shape[2], device=device) < frames)))
    steps["conv1"] = x.transpose(1, 2)
    conv1_frames = conv_frames(frames)
    x = F.gelu(model.conv2(x * (torch.arange(x.shape[2], device=device) < conv1_frames)))
    steps["conv2"] = x.transpose(1, 2)
    positions = x.shape[2]
    is_code = torch.arange(positions, device=device) < conv_frames(conv1_frames)
    position_mask = is_code.to(x.dtype)[None, :, None]
    attention_bias = (~is_code).to(x.dtype)[None, None, None, :] * -1.0e10
    cos = model.rotary_cos[:positions, None]
    sin = model.rotary_sin[:positions, None]
    x = x.transpose(1, 2)
    last = len(model.blocks) - 1
    for index, block in enumerate(model.blocks):
        x, block_trace = block_steps(block, x, position_mask, attention_bias, cos, sin)
        if index in (0, last):
            steps.update({f"block{index}.{name}": value for name, value in block_trace.items()})
        else:
            pass
        steps[f"block{index}.output"] = x
    projected = model.project_down(x)
    steps["project_down"] = projected
    levels = torch.round(torch.tanh(projected) * FSQ_SCALE) + 1
    return steps, (levels * model.fsq_powers).sum(dim=-1).to(torch.int32)


def real(name, value, codes):
    if name.endswith("scores") or name.endswith("weights"):
        return value[..., :codes, :codes]
    elif name.endswith("attended"):
        return value[..., :codes, :]
    else:
        return value[:, :codes]


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    # note (ratish): CUDNN_TF32=0 turns cuDNN's TF32 convs off, torch's default being on.
    torch.backends.cudnn.allow_tf32 = os.environ.get("CUDNN_TF32", "1") == "1"
    device = torch.device("cuda", 0)
    model = load_speech_tokenizer(os.path.join(resolve_checkpoint(MODEL), "speech_tokenizer_v3.onnx"), device)
    widths = [*generate_prefill_cuda_graph_batch_sizes(MAX_FRAMES - 1), MAX_FRAMES]
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")
    moved = 0
    for sample in samples:
        mel = whisper.log_mel_spectrogram(torch.from_numpy(load_prompt_audio(sample.ref_audio).reshape(1, -1)), n_mels=128)
        frames = mel.shape[2]
        width = widths[bisect.bisect_left(widths, frames)]
        padded = torch.cat((mel, torch.zeros(1, 128, width - frames)), dim=2).to(device)
        codes = conv_frames(conv_frames(frames))
        report = {}
        for allow_tf32 in (True, False):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            with torch.inference_mode():
                true_steps, true_codes = trace(model, mel.to(device), torch.tensor([frames], device=device))
                pad_steps, pad_codes = trace(model, padded, torch.tensor([frames], device=device))
            flips = int((true_codes[0, :codes] != pad_codes[0, :codes]).sum())
            report[allow_tf32] = (flips, true_steps, pad_steps)
        torch.backends.cuda.matmul.allow_tf32 = False
        if report[True][0] == 0:
            continue
        else:
            moved += 1
        print(f"{sample.sample_id[:40]} frames {frames} -> width {width}, code flips tf32 {report[True][0]} fp32 {report[False][0]}")
        _, true_steps, pad_steps = report[True]
        _, fp32_true, fp32_pad = report[False]
        for name in true_steps:
            a, b = real(name, true_steps[name], codes), real(name, pad_steps[name], codes)
            fa, fb = real(name, fp32_true[name], codes), real(name, fp32_pad[name], codes)
            scale = float(a.abs().max()) or 1.0
            tf32_diff = float((a - b).abs().max()) / scale
            fp32_diff = float((fa - fb).abs().max()) / scale
            print(
                f"   {name:22s} tf32 {tf32_diff:.2e}  fp32 {fp32_diff:.2e}  equal {torch.equal(a, b)}  "
                f"tf32 is fp32: true width {torch.equal(a, fa)}, padded {torch.equal(b, fb)}"
            )
        if moved >= count:
            break
        else:
            pass


if __name__ == "__main__":
    main()
