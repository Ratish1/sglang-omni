"""Which positions does HF's talker actually use on an audio prompt, given its own get_rope_index cannot read audio_config?

Runs HF's generate on the host for a few steps with the talker's position calls recorded: its config's relevant fields,
every get_rope_index call and whether it raised, and the position_ids its forward receives at the prefill.
"""

import librosa
import torch
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from benchmarks.dataset.seedtts import load_seedtts_samples

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

snapshot = snapshot_download(MODEL)
processor = Qwen3OmniMoeProcessor.from_pretrained(snapshot)
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    snapshot, dtype=torch.bfloat16, device_map="cpu", enable_audio_output=True
).eval()
talker = model.talker
print("talker config has audio_config:", hasattr(talker.config, "audio_config"), flush=True)
print("talker config audio_start_token_id:", getattr(talker.config, "audio_start_token_id", "missing"), flush=True)
original = talker.get_rope_index


def recorded(*args, **kwargs):
    print("get_rope_index called; audio_seqlens", kwargs.get("audio_seqlens"), flush=True)
    try:
        result = original(*args, **kwargs)
        print("get_rope_index returned; delta", result[1], flush=True)
        return result
    except Exception as error:
        print("get_rope_index raised", repr(error), flush=True)
        raise


talker.get_rope_index = recorded
original_forward = talker.forward


def forward(*args, **kwargs):
    positions = kwargs.get("position_ids")
    print(
        "talker forward: position_ids",
        None if positions is None else (positions.shape, positions[..., :6].tolist(), positions[..., -3:].tolist()),
        "keys",
        sorted(key for key in kwargs if key in ("talker_input_ids", "audio_feature_lengths", "attention_mask")),
        flush=True,
    )
    return original_forward(*args, **kwargs)


talker.forward = forward
sample = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", 1, split="en")[0]
prompt = f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". Now please read: {sample.target_text}'
text = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "audio", "audio": sample.ref_audio}, {"type": "text", "text": prompt}]}],
    add_generation_prompt=True,
    tokenize=False,
)
waveform, _ = librosa.load(sample.ref_audio, sr=16000)
inputs = processor(text=text, audio=[waveform], return_tensors="pt", padding=True)
inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
print("input keys", sorted(inputs.keys()), "lengths", inputs["feature_attention_mask"].sum(dim=1), flush=True)
with torch.inference_mode():
    try:
        model.generate(
            **inputs, speaker="Ethan", thinker_max_new_tokens=6, talker_max_new_tokens=3, return_audio=True
        )
        print("generate finished", flush=True)
    except Exception as error:
        print("generate raised", repr(error), flush=True)
