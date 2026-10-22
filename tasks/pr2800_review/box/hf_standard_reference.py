"""HF's Qwen3-Omni the standard way: Qwen's cookbook on transformers 4.57.1, the version the checkpoint was saved with.

Runs only in the isolated reference environment (no datasets, no omni code):
    PYTHONPATH=/data/pr2800/pylib/nokernels:/data/pr2800/pylib/hf457:/data/pr2800/pylib/qou CUDA_VISIBLE_DEVICES=<card>
    python3 hf_standard_reference.py --task speech|mmsu --samples <json> --shard i --shards n --out <dir>
Each request is the cookbook's: a user turn of content parts (video, audio, text as omni orders them), the processor's
chat template, qwen_omni_utils.process_mm_info with use_audio_in_video False, the processor, model.generate.
speech: the thinker greedy at 256 tokens, the talker with the checkpoint's sampling, speaker Ethan, DRAWS draws a
sample each under torch.manual_seed(draw); writes <out>/audio/<id>__<draw>.wav and <out>/speech_<shard>.json.
mmsu: the talker disabled, the thinker greedy at 32 tokens; writes <out>/mmsu_<shard>.json with each raw response.
--zero-multimodal-hidden gives HF's talker zeros at the multimodal prompt rows, as omni's main builds them.
--fp8-talker <repo> runs every talker linear that checkpoint quantizes on its dequantized FP8 weight, its input
quantized to e4m3 per token and 128 features as a W8A8 block-FP8 linear does; the thinker stays bf16.
--argmax-code-predictor takes the 15 residual codes by argmax, as omni does, where HF's generate samples them
(top_k 50, top_p 0.8, `prepare_inputs_for_generation`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DRAWS = 6
FP8_BLOCK = 128
FP8_MAX = 448.0


def dequantize_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The checkpoint's weight_scale_inv is the quantization scale: the weight is w / s, one s a 128 x 128 block."""
    rows, cols = weight.shape
    expanded = scale.float().repeat_interleave(FP8_BLOCK, 0)[:rows].repeat_interleave(FP8_BLOCK, 1)[:, :cols]
    return weight.float() / expanded


def quantize_input_per_token_group(module: torch.nn.Module, args: tuple) -> tuple:
    """The W8A8 block-FP8 linear's activation: e4m3 per token and group of 128 features, scale amax / 448."""
    del module
    x = args[0]
    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // FP8_BLOCK, FP8_BLOCK)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / FP8_MAX
    quantized = (groups / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale
    return (quantized.reshape(x.shape).to(x.dtype), *args[1:])


def use_fp8_talker(model: Qwen3OmniMoeForConditionalGeneration, fp8_model: str) -> int:
    """Every talker linear the FP8 checkpoint quantizes runs on its dequantized FP8 weight with an FP8 input."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(snapshot_download(fp8_model, allow_patterns=["*.json"]))
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    replaced = 0
    for name, module in model.talker.named_modules():
        key = f"talker.{name}"
        if not isinstance(module, torch.nn.Linear) or f"{key}.weight_scale_inv" not in weight_map:
            continue
        else:
            pass
        shard = Path(snapshot_download(fp8_model, allow_patterns=[weight_map[f"{key}.weight"]]))
        with safe_open(str(shard / weight_map[f"{key}.weight"]), "pt") as handle:
            weight = handle.get_tensor(f"{key}.weight")
        with safe_open(str(shard / weight_map[f"{key}.weight_scale_inv"]), "pt") as handle:
            scale = handle.get_tensor(f"{key}.weight_scale_inv")
        module.weight.data.copy_(dequantize_fp8_block(weight, scale).to(module.weight.dtype))
        module.register_forward_pre_hook(quantize_input_per_token_group)
        replaced += 1
    return replaced


def cookbook_inputs(processor: Qwen3OmniMoeProcessor, content: list[dict[str, object]], device: torch.device,
                    dtype: torch.dtype) -> dict[str, torch.Tensor]:
    conversation = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                       use_audio_in_video=False)
    return inputs.to(device).to(dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("speech", "mmsu"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--zero-multimodal-hidden",
        action="store_true",
        help="speech: the talker's multimodal prompt rows project zeros in place of the thinker's hidden states",
    )
    parser.add_argument("--fp8-talker", help="speech: the talker on this FP8 checkpoint's weights and W8A8 inputs")
    parser.add_argument("--argmax-code-predictor", action="store_true",
                        help="speech: the 15 residual codes by argmax, as omni takes them, in place of HF's sampling")
    args = parser.parse_args()
    spec = json.loads(args.samples.read_text())
    samples = spec["samples"][args.shard::args.shards]
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype="auto", device_map="auto")
    if args.fp8_talker:
        print(f"FP8 talker: {use_fp8_talker(model, args.fp8_talker)} linears", flush=True)
    else:
        pass
    if args.argmax_code_predictor:
        predictor_generate = model.talker.code_predictor.generate

        def argmax_generate(*generate_args, **generate_kwargs):
            return predictor_generate(*generate_args, **{**generate_kwargs, "do_sample": False, "top_k": None,
                                                         "top_p": None})

        model.talker.code_predictor.generate = argmax_generate
    else:
        pass
    if args.zero_multimodal_hidden:
        user_parts = model._get_talker_user_parts  # noqa: leading-underscore  # HF's own method

        def zero_hidden_user_parts(im_start_index, segment_end_index, multimodal_mask, thinker_hidden,
                                   thinker_embed):
            return user_parts(im_start_index, segment_end_index, multimodal_mask, torch.zeros_like(thinker_hidden),
                              thinker_embed)

        model._get_talker_user_parts = zero_hidden_user_parts  # noqa: leading-underscore  # HF's own method
    else:
        pass
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    if args.task == "mmsu":
        model.disable_talker()
    else:
        (args.out / "audio").mkdir(parents=True, exist_ok=True)
    records = []
    for index, sample in enumerate(samples):
        if args.task == "speech":
            content = [
                {"type": "video", "video": sample["video_path"], "fps": 2, "max_frames": 128, "max_pixels": 401408},
                {"type": "audio", "audio": sample["audio_path"]},
                {"type": "text", "text": spec["prompt"]},
            ]
            inputs = cookbook_inputs(processor, content, model.device, model.dtype)
            for draw in range(DRAWS):
                torch.manual_seed(draw)
                start = time.perf_counter()
                text_ids, audio = model.generate(
                    **inputs, speaker="Ethan", thinker_return_dict_in_generate=True, use_audio_in_video=False,
                    thinker_do_sample=False, thinker_max_new_tokens=256)
                seconds = time.perf_counter() - start
                text = processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0]
                wav = audio.reshape(-1).float().cpu().numpy()
                path = args.out / "audio" / f"{sample['sample_id']}__{draw}.wav"
                soundfile.write(path, wav, 24000)
                records.append(dict(sample_id=f"{sample['sample_id']}__{draw}", raw_response=text, wav_path=str(path),
                                    is_success=True, prompt_tokens=int(inputs["input_ids"].shape[1]),
                                    latency_s=round(seconds, 2), audio_duration_s=round(len(wav) / 24000, 2)))
                print(f"{time.strftime('%H:%M:%S')} {sample['sample_id']} draw {draw}: {len(wav) / 24000:.1f} s audio, "
                      f"{seconds:.0f} s, text {text[:100]!r}", flush=True)
        else:
            content = [{"type": "audio", "audio": sample["audio_path"]}, {"type": "text", "text": sample["question"]}]
            inputs = cookbook_inputs(processor, content, model.device, model.dtype)
            text_ids, _ = model.generate(**inputs, return_audio=False, thinker_return_dict_in_generate=True,
                                         use_audio_in_video=False, thinker_do_sample=False, thinker_max_new_tokens=32)
            sequences = text_ids.sequences
            text = processor.batch_decode(sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
            records.append(dict(sample_id=sample["sample_id"], response=text.strip(),
                                prompt_tokens=int(inputs["input_ids"].shape[1])))
            if index % 100 == 0:
                print(f"{time.strftime('%H:%M:%S')} mmsu shard {args.shard} {index}/{len(samples)}", flush=True)
            else:
                pass
    (args.out / f"{args.task}_{args.shard}.json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
