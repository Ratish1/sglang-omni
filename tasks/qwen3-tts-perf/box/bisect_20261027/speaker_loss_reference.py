"""Official qwen-tts 0.1.1 drawing chosen parity cases many times, as speaker_loss.py serve does on our server: the Base
voice clone with the reference and its transcript, language auto, the stream-text prompt (the clone default), the
checkpoint's sampling defaults, a seed per draw; each draw's audio and frame count kept in speaker_loss.py's layout so
its score step reads both arms alike.

usage: /data/refenv/bin/python speaker_loss_reference.py CASES.jsonl --ids ID [ID ...] --model PATH --out DIR
       [--draws 100] [--batch 16] [--card 0] [--seed 0] [--encoder-fp32]

--encoder-fp32 reloads the speech tokenizer from its float32 checkpoint and casts only its decoder to bf16, so the
reference encoder runs on the checkpoint's own weights and dtype.

--ref-codes FILE.pt (ref_codes_case.py's output) builds official's own voice-clone prompt for each case, prints its
reference codes against FILE's `--ref-codes-key` frame by frame, then draws with that tensor in place of official's
codes, or with only its last frame in place (--last-frame-only); the speaker embedding and text stay official's.

Runs in a venv holding qwen-tts 0.1.1's pins (transformers 4.57.3, accelerate 1.12.0, torch 2.10.0).
"""

import argparse
import json
from pathlib import Path

import soundfile
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer


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
    parser.add_argument("--ref-codes", type=Path)
    parser.add_argument("--ref-codes-key", default="ours_graphs")
    parser.add_argument("--last-frame-only", action="store_true")
    args = parser.parse_args()
    cases = {
        case["id"]: case
        for case in (json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line)
    }
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map=f"cuda:{args.card}", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if args.encoder_fp32:
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(Path(snapshot_download(args.model)) / "speech_tokenizer"),
            device_map=f"cuda:{args.card}",
            dtype=torch.float32,
        )
        tokenizer.model.decoder.to(torch.bfloat16)
        model.model.load_speech_tokenizer(tokenizer)
    else:
        pass
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    records = []
    for case_id in args.ids:
        case = cases[case_id]
        if args.ref_codes is not None:
            items = model.create_voice_clone_prompt(ref_audio=case["ref_audio"], ref_text=case["ref_text"])
            official = items[0].ref_code
            swap = torch.load(args.ref_codes)[args.ref_codes_key].to(device=official.device, dtype=official.dtype)
            frames = min(official.shape[0], swap.shape[0])
            differing = torch.nonzero(~(official[:frames] == swap[:frames]).all(dim=-1)).flatten().tolist()
            print(f"{case_id}: official {tuple(official.shape)} / {args.ref_codes_key} {tuple(swap.shape)}, frames differing {differing}", flush=True)
            torch.save(official.cpu(), args.out / f"{case_id}_official_codes.pt")
            if args.last_frame_only:
                replaced = official.clone()
                replaced[-1] = swap[-1]
            else:
                replaced = swap
            items[0].ref_code = replaced
            prompt = {"voice_clone_prompt": items}
        else:
            prompt = {"ref_audio": [case["ref_audio"]], "ref_text": [case["ref_text"]]}
        for start in range(0, args.draws, args.batch):
            count = min(args.batch, args.draws - start)
            torch.manual_seed(args.seed + start)
            with torch.no_grad():
                wavs, sample_rate = model.generate_voice_clone(
                    text=[case["input"]] * count,
                    language=["Auto"] * count,
                    **{name: value * count for name, value in prompt.items()},
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
