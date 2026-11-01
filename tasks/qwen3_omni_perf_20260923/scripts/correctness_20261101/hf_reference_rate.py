"""HF's own Qwen3-Omni pipeline on SeedTTS EN voice-clone requests (correctness plan section 5, the reference rate).

Each named sample goes through Qwen3OmniMoeProcessor (reference audio by librosa.load at 16 kHz) and
Qwen3OmniMoeForConditionalGeneration.generate with the bench's request: the VoiceCloneOmni prompt, speaker Ethan,
thinker temperature 0.7 with no top-k or top-p (the server applies only the field the bench sends), HF's talker
defaults (sampled, top-k 50, top-p 1.0, temperature 0.9, repetition penalty 1.05). RESIDUALS=argmax makes the code
predictor greedy, as our server is (DECISIONS 11); the default samples it, as HF does. Writes OUT/audio/<id>.wav at 24
kHz, OUT/reply_text.json, and one timing line a sample, so a run can be sized and scored by run_bench.py's scorer.

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


def main() -> None:
    out = Path(sys.argv[1])
    sample_ids = Path(sys.argv[2]).read_text().split()
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    (out / "audio").mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(MODEL)
    processor = Qwen3OmniMoeProcessor.from_pretrained(snapshot)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.bfloat16, device_map="auto", enable_audio_output=True
    ).eval()
    if RESIDUALS == "argmax":
        model.talker.code_predictor.generation_config.do_sample = False
    else:
        pass
    print(
        f"thinker generation config: {model.thinker.generation_config}; residuals {RESIDUALS}",
        flush=True,
    )
    samples = {
        sample.sample_id: sample
        for sample in load_seedtts_samples(SEEDTTS_META, None, split="en")
    }
    replies = {}
    for repeat in range(repeats):
        for sample_id in sample_ids:
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
            key = f"{sample_id}" if repeats == 1 else f"{sample_id}__r{repeat}"
            reply = processor.batch_decode(
                sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
            replies[key] = reply
            waveform = wav.reshape(-1).float().cpu().numpy()
            soundfile.write(out / "audio" / f"{key}.wav", waveform, 24000)
            print(
                f"sample {key} {elapsed:.1f} s, audio {len(waveform) / 24000:.2f} s, reply {reply[:80]!r}",
                flush=True,
            )
            (out / "reply_text.json").write_text(
                json.dumps(replies, indent=1, ensure_ascii=False)
            )


if __name__ == "__main__":
    main()
