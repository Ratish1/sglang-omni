"""HF's own Qwen3-Omni pipeline on SeedTTS EN voice-clone requests (correctness plan section 5, the reference rate).

Each named sample goes through Qwen3OmniMoeProcessor (reference audio by librosa.load at 16 kHz) and
Qwen3OmniMoeForConditionalGeneration.generate with the bench's request: the VoiceCloneOmni prompt, speaker Ethan,
thinker temperature 0.7 with no top-k or top-p (the server applies only the field the bench sends), HF's talker
defaults (sampled, top-k 50, top-p 1.0, temperature 0.9, repetition penalty 1.05). The default samples the residual
codebooks as HF does (top-k 50, top-p 0.8, mirrored from the talker's do_sample at every generate call,
modeling_qwen3_omni_moe.py:3261-3267); RESIDUALS=argmax overrides that mirror so the code predictor is greedy, as our
server is (DECISIONS 11). Writes OUT/audio/<id>__r<k>.wav at 24 kHz, OUT/reply_text.json, OUT/generated.json in the
bench's record format, and one timing line a sample, so run_bench.py's scorer and bad_output_census.py read a run as
they read a served pass. A run restarted on OUT continues after its last finished request.

usage: python3 hf_reference_rate.py OUT SAMPLE_IDS_FILE [REPEATS]
"""

import json
import os
import sys
import time
from pathlib import Path

import librosa
import soundfile
import torch
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from benchmarks.dataset.seedtts import load_seedtts_samples

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEEDTTS_META = "zhaochenyang20/seed-tts-eval-arrow"
RESIDUALS = os.environ.get("RESIDUALS", "sampled")
DEVICE = os.environ.get("DEVICE", "cuda:0")


def force_greedy_residuals(talker: torch.nn.Module) -> None:
    """Keep the code predictor greedy after the talker's generate mirrors its do_sample onto it."""
    prepare = talker._prepare_generation_config  # noqa: leading-underscore

    def prepare_then_greedy(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        talker.code_predictor.generation_config.do_sample = False
        return prepared

    talker._prepare_generation_config = prepare_then_greedy  # noqa: leading-underscore


def main() -> None:
    out = Path(sys.argv[1])
    sample_ids = Path(sys.argv[2]).read_text().split()
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    (out / "audio").mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(MODEL)
    processor = Qwen3OmniMoeProcessor.from_pretrained(snapshot)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        device_map={"": DEVICE},
        enable_audio_output=True,
    ).eval()
    if RESIDUALS == "argmax":
        force_greedy_residuals(model.talker)
    else:
        pass
    print(f"residuals {RESIDUALS} device {DEVICE}", flush=True)
    samples = {
        sample.sample_id: sample
        for sample in load_seedtts_samples(SEEDTTS_META, None, split="en")
    }
    generated_path = out / "generated.json"
    replies_path = out / "reply_text.json"
    generated = (
        json.loads(generated_path.read_text()) if generated_path.exists() else []
    )
    replies = json.loads(replies_path.read_text()) if replies_path.exists() else {}
    for repeat in range(repeats):
        for sample_id in sample_ids:
            key = f"{sample_id}__r{repeat}"
            if key in replies:
                continue
            else:
                pass
            sample = samples[sample_id]
            prompt = (
                f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
                f"Now please read the following text out loud in the same voice and style: "
                f"{sample.target_text}"
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": sample.ref_audio},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            audio, _ = librosa.load(sample.ref_audio, sr=16000)
            inputs = processor(
                text=text, audio=[audio], return_tensors="pt", padding=True
            ).to(model.device)
            inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
            began = time.time()
            with torch.inference_mode():
                sequences, wav = model.generate(
                    **inputs,
                    speaker="Ethan",
                    thinker_do_sample=True,
                    thinker_temperature=0.7,
                    thinker_top_k=0,
                    thinker_top_p=1.0,
                    return_audio=True,
                )
            elapsed = time.time() - began
            reply = processor.batch_decode(
                sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
            replies[key] = reply
            waveform = wav.reshape(-1).float().cpu().numpy()
            wav_path = out / "audio" / f"{key}.wav"
            soundfile.write(wav_path, waveform, 24000)
            audio_seconds = len(waveform) / 24000
            generated.append(
                {
                    "sample_id": key,
                    "target_text": sample.target_text,
                    "wav_path": str(wav_path),
                    "is_success": True,
                    "latency_s": elapsed,
                    "audio_duration_s": audio_seconds,
                }
            )
            print(
                f"sample {key} {elapsed:.1f} s, audio {audio_seconds:.2f} s, reply {reply[:80]!r}",
                flush=True,
            )
            replies_path.write_text(json.dumps(replies, indent=1, ensure_ascii=False))
            generated_path.write_text(
                json.dumps(generated, indent=1, ensure_ascii=False)
            )


if __name__ == "__main__":
    main()
