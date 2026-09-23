"""HF reference for the Qwen3-Omni talker's multimodal prompt rows (item 12, step 0b).

parity:   for every omni dump (talker prompt ids, multimodal positions, rows; written by
          the prototype with SGLANG_OMNI_DUMP_PROMPT_HIDDEN), run the HF thinker on the
          same SeedTTS voice-clone prompt and compare hidden_states[23..25] at those
          positions with omni's rows. Prints whether the prompt ids match, then per layer
          the mean and min cosine and the max relative row difference.
generate: HF generate with audio output on SeedTTS EN (all samples, or the first N) with
          the benchmark's voice-clone prompt and thinker sampling; shard k of n writes
          OUT/seedtts_en/generated_k.json and wavs.
layers:   with omni's thinker dumps (full prompt rows at the inputs of layers 0 and 24),
          compares HF hidden_states[0] and [24] on text rows and audio rows separately.
merge:    joins the shards into OUT/seedtts_en/generated.json for run_bench.py score/sim.

usage: python hf_voice_clone_reference.py parity --dump-dir DIR
       python hf_voice_clone_reference.py generate --out DIR --shard 0 --num-shards 3
       python hf_voice_clone_reference.py merge --out DIR
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEEDTTS_META = "zhaochenyang20/seed-tts-eval-arrow"
AUDIO_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
# benchmarks/tasks/tts.py sends this temperature with every SeedTTS request
THINKER_TEMPERATURE = 0.7


def voice_clone_prompt(sample) -> str:
    return (
        f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
        f"Now please read the following text out loud in the same voice and style: "
        f"{sample.target_text}"
    )


def load_model():
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    return processor, model


def build_inputs(processor, model, sample):
    import librosa

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": sample.ref_audio},
                {"type": "text", "text": voice_clone_prompt(sample)},
            ],
        }
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    waveform, _ = librosa.load(sample.ref_audio, sr=AUDIO_SAMPLE_RATE)
    inputs = processor(text=text, audio=[waveform], return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    inputs["input_features"] = inputs["input_features"].to(model.dtype)
    return inputs


def seedtts_samples(count: int | None):
    from benchmarks.dataset.seedtts import load_seedtts_samples

    samples = load_seedtts_samples(SEEDTTS_META, split="en")
    return samples if count is None else samples[:count]


@torch.no_grad()
def parity(dump_dir: str) -> None:
    processor, model = load_model()
    samples = seedtts_samples(None)
    for dump_path in sorted(Path(dump_dir).glob("*.pt")):
        dump = torch.load(dump_path)
        prompt_text = processor.tokenizer.decode(dump["prompt_ids"])
        sample = next(
            (
                s
                for s in samples
                if s.target_text in prompt_text and s.ref_text in prompt_text
            ),
            None,
        )
        if sample is None:
            print(f"{dump_path.name}: no SeedTTS sample matches the dumped prompt")
            continue
        inputs = build_inputs(processor, model, sample)
        hf_ids = inputs["input_ids"][0].cpu()
        ids_match = torch.equal(hf_ids, dump["prompt_ids"])
        print(
            f"{sample.sample_id}: prompt ids match={ids_match} "
            f"(hf {hf_ids.numel()}, omni {dump['prompt_ids'].numel()}), "
            f"multimodal rows {dump['positions'].numel()}"
        )
        if not ids_match:
            print(f"  omni prompt: {prompt_text[:300]!r}")
            print(f"  hf prompt:   {processor.tokenizer.decode(hf_ids)[:300]!r}")
            continue
        thinker_inputs = {
            key: inputs[key]
            for key in (
                "input_ids",
                "attention_mask",
                "input_features",
                "feature_attention_mask",
            )
            if key in inputs
        }
        hidden_states = model.thinker(
            **thinker_inputs, output_hidden_states=True
        ).hidden_states
        omni_rows = dump["rows"].float().cuda()
        positions = dump["positions"].cuda()
        for layer in (23, 24, 25):
            hf_rows = hidden_states[layer][0][positions].float()
            cosine = torch.nn.functional.cosine_similarity(hf_rows, omni_rows, dim=-1)
            relative = (hf_rows - omni_rows).norm(dim=-1) / hf_rows.norm(dim=-1)
            print(
                f"  hidden_states[{layer}]: cosine mean {cosine.mean():.4f} min {cosine.min():.4f}, "
                f"relative diff max {relative.max():.4f}, hf norm {hf_rows.norm(dim=-1).mean():.2f}, "
                f"omni norm {omni_rows.norm(dim=-1).mean():.2f}"
            )


def row_agreement(hf_rows: torch.Tensor, omni_rows: torch.Tensor) -> str:
    cosine = torch.nn.functional.cosine_similarity(hf_rows, omni_rows, dim=-1)
    relative = (hf_rows - omni_rows).norm(dim=-1) / hf_rows.norm(dim=-1).clamp_min(1e-6)
    return (
        f"cosine mean {cosine.mean():.4f} min {cosine.min():.4f}, "
        f"relative diff mean {relative.mean():.4f} max {relative.max():.4f}"
    )


@torch.no_grad()
def layers(dump_dir: str) -> None:
    """Compare omni's full prompt rows at layer 0 and layer 24 with HF, text and audio apart."""
    processor, model = load_model()
    samples = seedtts_samples(None)
    for talker_dump_path in sorted(Path(dump_dir).glob("*.pt")):
        if talker_dump_path.name.startswith("thinker_"):
            continue
        request_id = talker_dump_path.stem
        chunks = sorted(
            (
                torch.load(path)
                for path in Path(dump_dir).glob(f"thinker_{request_id}_*.pt")
            ),
            key=lambda chunk: chunk["prefix"],
        )
        if not chunks or chunks[0]["prefix"] != 0:
            print(f"{request_id}: no full thinker dump")
            continue
        talker_dump = torch.load(talker_dump_path)
        prompt_text = processor.tokenizer.decode(talker_dump["prompt_ids"])
        sample = next(
            (
                s
                for s in samples
                if s.target_text in prompt_text and s.ref_text in prompt_text
            ),
            None,
        )
        if sample is None:
            continue
        inputs = build_inputs(processor, model, sample)
        if not torch.equal(inputs["input_ids"][0].cpu(), talker_dump["prompt_ids"]):
            print(f"{sample.sample_id}: prompt ids differ, skipped")
            continue
        thinker_inputs = {
            key: inputs[key]
            for key in (
                "input_ids",
                "attention_mask",
                "input_features",
                "feature_attention_mask",
            )
            if key in inputs
        }
        hidden_states = model.thinker(
            **thinker_inputs, output_hidden_states=True
        ).hidden_states
        prompt_len = talker_dump["prompt_ids"].numel()
        audio_mask = torch.zeros(prompt_len, dtype=torch.bool)
        audio_mask[talker_dump["positions"]] = True
        print(f"{sample.sample_id}: {prompt_len} rows, {int(audio_mask.sum())} audio")
        for layer in (0, 24):
            omni_rows = torch.cat([chunk[f"layer{layer}"] for chunk in chunks])[
                :prompt_len
            ]
            hf_rows = hidden_states[layer][0].float().cpu()
            for region, mask in (("text", ~audio_mask), ("audio", audio_mask)):
                print(
                    f"  layer {layer:2d} {region:5s}: "
                    f"{row_agreement(hf_rows[mask], omni_rows[mask])}"
                )


@torch.no_grad()
def generate(count: int | None, out: str, shard: int, num_shards: int) -> None:
    import soundfile

    processor, model = load_model()
    arm_dir = Path(out) / "seedtts_en"
    audio_dir = arm_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in seedtts_samples(count)[shard::num_shards]:
        inputs = build_inputs(processor, model, sample)
        began = time.perf_counter()
        # the benchmark's thinker sampling for omni: temperature 0.7, top-p 1, no top-k
        _, audio = model.generate(
            **inputs,
            speaker="Ethan",
            thinker_max_new_tokens=256,
            thinker_do_sample=True,
            thinker_temperature=THINKER_TEMPERATURE,
            thinker_top_p=1.0,
            thinker_top_k=0,
        )
        latency = time.perf_counter() - began
        waveform = audio.reshape(-1).float().cpu().numpy()
        wav_path = audio_dir / f"{sample.sample_id}.wav"
        soundfile.write(wav_path, waveform, OUTPUT_SAMPLE_RATE)
        duration = waveform.shape[0] / OUTPUT_SAMPLE_RATE
        records.append(
            {
                "sample_id": sample.sample_id,
                "target_text": sample.target_text,
                "wav_path": str(wav_path),
                "is_success": True,
                "latency_s": latency,
                "audio_duration_s": duration,
            }
        )
        print(
            f"{sample.sample_id}: {duration:.2f} s audio in {latency:.1f} s", flush=True
        )
    (arm_dir / f"generated_{shard}.json").write_text(json.dumps(records, indent=1))


def merge(out: str) -> None:
    arm_dir = Path(out) / "seedtts_en"
    records = [
        record
        for path in sorted(arm_dir.glob("generated_*.json"))
        for record in json.loads(path.read_text())
    ]
    (arm_dir / "generated.json").write_text(json.dumps(records, indent=1))
    print(f"merged {len(records)} records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("parity", "layers", "generate", "merge"))
    parser.add_argument("--dump-dir")
    parser.add_argument("--samples", type=int, help="first N samples; unset is all")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.mode == "parity":
        parity(args.dump_dir)
    elif args.mode == "layers":
        layers(args.dump_dir)
    elif args.mode == "generate":
        generate(args.samples, args.out, args.shard, args.num_shards)
    else:
        merge(args.out)


if __name__ == "__main__":
    main()
