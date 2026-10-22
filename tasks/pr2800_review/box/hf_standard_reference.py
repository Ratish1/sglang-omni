"""HF's Qwen3-Omni the standard way: Qwen's cookbook on transformers 4.57.1, the version the checkpoint was saved with.

Runs only in the isolated reference environment (no datasets, no omni code):
    PYTHONPATH=/data/pr2800/pylib/nokernels:/data/pr2800/pylib/hf457:/data/pr2800/pylib/qou CUDA_VISIBLE_DEVICES=<card>
    python3 hf_standard_reference.py --task speech|mmsu --samples <json> --shard i --shards n --out <dir>
Each request is the cookbook's: a user turn of content parts (video, audio, text as omni orders them), the processor's
chat template, qwen_omni_utils.process_mm_info with use_audio_in_video False, the processor, model.generate.
speech: the thinker greedy at 256 tokens, the talker with the checkpoint's sampling, speaker Ethan, DRAWS draws a
sample each under torch.manual_seed(draw); writes <out>/audio/<id>__<draw>.wav and <out>/speech_<shard>.json.
mmsu: the talker disabled, the thinker greedy at 32 tokens; writes <out>/mmsu_<shard>.json with each raw response.
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
    args = parser.parse_args()
    spec = json.loads(args.samples.read_text())
    samples = spec["samples"][args.shard::args.shards]
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype="auto", device_map="auto")
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
                      f"{seconds:.0f} s", flush=True)
        else:
            content = [{"type": "audio", "audio": sample["audio_path"]}, {"type": "text", "text": sample["question"]}]
            inputs = cookbook_inputs(processor, content, model.device, model.dtype)
            text_ids = model.generate(**inputs, return_audio=False, thinker_return_dict_in_generate=True,
                                      use_audio_in_video=False, thinker_do_sample=False, thinker_max_new_tokens=32)
            sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
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
