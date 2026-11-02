"""Does BF16 arithmetic make the talker's bad outputs? HF's talker in factorial cells on the samples that go bad.

rows: for each sample, HF's own talker prefill and trailing text rows from the HF audio tower and thinker layers (through
the accept layer only; the reply is the target text teacher-forced, which the served thinker emits in about 98 % of
requests), once in FP32 and once in BF16, saved per dtype.
cell: HF's talker generate over those rows, with talker arithmetic and residual sampling chosen by the cell, HF's
code2wav in the talker's dtype, HF's sampling (top-k 50, top-p 1.0, temperature 0.9, repetition penalty 1.05, the top
1,024 ids but EOS suppressed), unseeded. Each pass writes the bench's generated.json and reply_text.json so
run_bench.py's scorer and bad_output_census.py read a cell as they read a served pass.

usage: python3 talker_cells.py rows ROWS.pt float32|bfloat16 SAMPLE_IDS_FILE
       python3 talker_cells.py cell OUT ROWS.pt float32|bfloat16 sampled|argmax SAMPLE_IDS_FILE REPEATS
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
DEVICE = os.environ.get("TALKER_CELLS_DEVICE", "cuda:0")
SPEAKER = "ethan"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def samples_by_id() -> dict:
    return {s.sample_id: s for s in load_seedtts_samples(SEEDTTS_META, None, split="en")}


def build_rows(out: str, dtype_name: str, ids_path: str) -> None:
    dtype = getattr(torch, dtype_name)
    sample_ids = Path(ids_path).read_text().split()
    snapshot = snapshot_download(MODEL)
    processor = Qwen3OmniMoeProcessor.from_pretrained(snapshot)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        snapshot, dtype=dtype, device_map="cpu", enable_audio_output=True
    ).eval()
    config = model.config
    thinker = model.thinker
    accept = config.talker_config.accept_hidden_layer
    # note (ratish): hidden_states[accept] is the accept layer's input, recorded when that layer
    # runs; the layers above it are never read, so they do not go to the card.
    del thinker.model.layers[accept + 1 :]
    thinker.lm_head = torch.nn.Identity()
    thinker.to(DEVICE)
    talker = model.talker
    talker.text_projection.to(DEVICE)
    talker.hidden_projection.to(DEVICE)
    talker.get_input_embeddings().to(DEVICE)
    speaker_id = config.talker_config.speaker_id[SPEAKER]
    samples = samples_by_id()
    rows = {}
    with torch.inference_mode():
        specials = torch.tensor(
            [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]], device=DEVICE
        )
        tts_bos, tts_eos, tts_pad = talker.text_projection(thinker.get_input_embeddings()(specials)).chunk(3, dim=1)
        for sample_id in sample_ids:
            sample = samples[sample_id]
            prompt = (
                f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
                f"Now please read the following text out loud in the same voice and style: "
                f"{sample.target_text}"
            )
            text = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": sample.ref_audio},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            waveform, _ = librosa.load(sample.ref_audio, sr=16000)
            inputs = processor(text=text, audio=[waveform], return_tensors="pt", padding=True)
            prompt_ids = inputs["input_ids"].to(DEVICE)
            reply_ids = processor.tokenizer(sample.target_text, return_tensors="pt")["input_ids"].to(DEVICE)
            thinker.rope_deltas = None
            output = thinker(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                input_features=inputs["input_features"].to(DEVICE, dtype),
                feature_attention_mask=inputs["feature_attention_mask"].to(DEVICE),
                output_hidden_states=True,
                use_cache=False,
            )
            prompt_embed = output.hidden_states[0]
            prompt_hidden = output.hidden_states[accept]
            reply_embed = thinker.get_input_embeddings()(reply_ids)
            thinker_embed = torch.cat([prompt_embed, reply_embed], dim=1)
            thinker_hidden = torch.cat([prompt_hidden, torch.zeros_like(reply_embed)], dim=1)
            sequences = torch.cat([prompt_ids, reply_ids], dim=1)
            multimodal = sequences == config.thinker_config.audio_token_id
            assistant_start = (
                torch.nonzero(
                    (prompt_ids[0, :-1] == config.im_start_token_id)
                    & (prompt_ids[0, 1:] == config.assistant_token_id)
                )
                .flatten()[-1]
                .item()
            )
            user = model._get_talker_user_parts(  # noqa: leading-underscore
                0, assistant_start, multimodal, thinker_hidden, thinker_embed
            )
            assistant, _, trailing = model._get_talker_assistant_parts(  # noqa: leading-underscore
                assistant_start,
                thinker_embed.shape[1],
                speaker_id,
                thinker_embed,
                tts_pad,
                tts_bos,
                tts_eos,
            )
            rows[sample_id] = {
                "prefill": torch.cat([user, assistant], dim=1)[0].float().cpu(),
                "trailing": trailing[0].float().cpu(),
                "pad": tts_pad[0, 0].float().cpu(),
            }
            print(f"rows {sample_id} {rows[sample_id]['prefill'].shape[0]}", flush=True)
    torch.save(rows, out)


def force_greedy_residuals(talker: torch.nn.Module) -> None:
    prepare = talker._prepare_generation_config  # noqa: leading-underscore

    def prepare_then_greedy(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        talker.code_predictor.generation_config.do_sample = False
        return prepared

    talker._prepare_generation_config = prepare_then_greedy  # noqa: leading-underscore


def run_cell(out: str, rows_path: str, dtype_name: str, residuals: str, ids_path: str, repeats: int) -> None:
    dtype = getattr(torch, dtype_name)
    out_dir = Path(out)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    sample_ids = Path(ids_path).read_text().split()
    snapshot = snapshot_download(MODEL)
    rows = torch.load(rows_path, map_location="cpu")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        snapshot, dtype=dtype, device_map="cpu", enable_audio_output=True
    ).eval()
    config = model.config
    talker = model.talker.to(DEVICE)
    code2wav = model.code2wav.to(DEVICE)
    del model.thinker
    if residuals == "argmax":
        force_greedy_residuals(talker)
    else:
        pass
    vocab = config.talker_config.text_config.vocab_size
    codec_eos = config.talker_config.codec_eos_token_id
    suppressed = [i for i in range(vocab - 1024, vocab) if i != codec_eos]
    samples = samples_by_id()
    generated_path = out_dir / "generated.json"
    replies_path = out_dir / "reply_text.json"
    generated = json.loads(generated_path.read_text()) if generated_path.exists() else []
    replies = json.loads(replies_path.read_text()) if replies_path.exists() else {}
    torch.manual_seed(int(time.time() * 1000) % 2**31)
    for repeat in range(repeats):
        for sample_id in sample_ids:
            key = f"{sample_id}__r{repeat}"
            if key in replies:
                continue
            else:
                pass
            sample = samples[sample_id]
            row = rows[sample_id]
            prefill = row["prefill"].to(DEVICE, dtype)[None]
            began = time.time()
            with torch.inference_mode():
                result = talker.generate(
                    inputs_embeds=prefill,
                    trailing_text_hidden=row["trailing"].to(DEVICE, dtype)[None],
                    tts_pad_embed=row["pad"].to(DEVICE, dtype)[None, None],
                    talker_input_ids=torch.full(
                        (1, prefill.shape[1]), config.tts_pad_token_id, device=DEVICE
                    ),
                    attention_mask=torch.ones(1, prefill.shape[1], device=DEVICE, dtype=torch.long),
                    max_new_tokens=4096,
                    do_sample=True,
                    top_k=50,
                    top_p=1.0,
                    temperature=0.9,
                    eos_token_id=codec_eos,
                    repetition_penalty=1.05,
                    suppress_tokens=suppressed,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                codes = (
                    torch.stack([hid[-1] for hid in result.hidden_states if hid[-1] is not None], dim=1)
                    .transpose(1, 2)
                    .to(DEVICE)
                )
                wav = code2wav.chunked_decode(codes, chunk_size=300, left_context_size=25)
            elapsed = time.time() - began
            waveform = wav.reshape(-1).float().cpu().numpy()
            wav_path = out_dir / "audio" / f"{key}.wav"
            soundfile.write(wav_path, waveform, 24000)
            seconds = len(waveform) / 24000
            replies[key] = sample.target_text
            generated.append(
                {
                    "sample_id": key,
                    "target_text": sample.target_text,
                    "wav_path": str(wav_path),
                    "is_success": True,
                    "latency_s": elapsed,
                    "audio_duration_s": seconds,
                }
            )
            print(f"sample {key} {elapsed:.1f} s, audio {seconds:.2f} s, frames {codes.shape[-1]}", flush=True)
            replies_path.write_text(json.dumps(replies, indent=1, ensure_ascii=False))
            generated_path.write_text(json.dumps(generated, indent=1, ensure_ascii=False))


def main() -> None:
    if sys.argv[1] == "rows":
        build_rows(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        run_cell(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]))


if __name__ == "__main__":
    main()
