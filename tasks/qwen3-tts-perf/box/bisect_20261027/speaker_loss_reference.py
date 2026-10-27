"""Official qwen-tts 0.1.1 drawing chosen parity cases many times, as speaker_loss.py serve does on our server: the Base
voice clone with the reference and its transcript, language auto, the stream-text prompt (the clone default), the
checkpoint's sampling defaults, a seed per draw; each draw's audio and frame count kept in speaker_loss.py's layout so
its score step reads both arms alike.

usage: /data/refenv/bin/python speaker_loss_reference.py CASES.jsonl --ids ID [ID ...] --model PATH --out DIR
       [--draws 100] [--batch 16] [--card 0] [--seed 0] [--encoder-fp32]

--encoder-fp32 runs the speech tokenizer's encoder in its checkpoint dtype (float32), the rest in bf16.

Runs in a venv holding qwen-tts 0.1.1's pins (transformers 4.57.3, accelerate 1.12.0, torch 2.10.0).
"""

import argparse
import json
from pathlib import Path

import soundfile
import torch
from qwen_tts import Qwen3TTSModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-fp32", action="store_true")
    args = parser.parse_args()
    cases = {
        case["id"]: case
        for case in (json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line)
    }
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map=f"cuda:{args.card}", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if args.encoder_fp32:
        model.model.speech_tokenizer.model.encoder.to(torch.float32)
    else:
        pass
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    records = []
    for case_id in args.ids:
        case = cases[case_id]
        for start in range(0, args.draws, args.batch):
            count = min(args.batch, args.draws - start)
            torch.manual_seed(args.seed + start)
            with torch.no_grad():
                wavs, sample_rate = model.generate_voice_clone(
                    text=[case["input"]] * count,
                    language=["Auto"] * count,
                    ref_audio=[case["ref_audio"]] * count,
                    ref_text=[case["ref_text"]] * count,
                )
            for offset, wav in enumerate(wavs):
                draw = start + offset
                audio = args.out / "audio" / f"{case_id}_{draw:04d}.wav"
                soundfile.write(str(audio), wav, sample_rate, subtype="PCM_16")
                records.append({
                    "id": case_id, "draw": draw, "input": case["input"], "ref_audio": case["ref_audio"],
                    "finish_reason": "official", "frames": int(round(len(wav) / sample_rate * 12.5)),
                    "audio": str(audio),
                })
            print(f"{case_id} draws {start}-{start + count - 1}", flush=True)
    (args.out / "results.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


if __name__ == "__main__":
    main()
