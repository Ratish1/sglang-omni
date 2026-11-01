"""Our talker prefill rows against the rows HF builds from its own processor and thinker (correctness plan, row 10).

For each talker capture (test/qwen3-omni-talker-capture-20261101, SGLANG_OMNI_TALKER_CAPTURE_DIR) of a SeedTTS EN
voice-clone request, HF rebuilds the talker prefill from scratch: the request's messages through
Qwen3OmniMoeProcessor (reference audio by librosa.load at 16 kHz, the model card's loader), the HF thinker over the
prompt (hidden_states[0] and [accept_hidden_layer]), the reply's token embeddings, and HF's own
_get_talker_user_parts and _get_talker_assistant_parts, assembled as Qwen3OmniMoeForConditionalGeneration.generate
assembles them. "build" runs it in one dtype and saves the rows; "compare" reads the FP32 rows as the truth and the
BF16 rows as the floor and reports, per request and per row group (user text rows, user multimodal rows, assistant
rows): prompt id equality, and the relative L2 of ours and of the floor against the truth.

Pass criterion, fixed before the run: prompt ids equal; on every row group ours within 1.25 times the floor's mean
relative L2 to the truth (the predictor kernel tests' multiple).

usage: python3 prompt_rows_parity.py build float32|bfloat16 OUT.pt CAPTURE.pt [CAPTURE.pt ...]
       python3 prompt_rows_parity.py compare TRUTH.pt FLOOR.pt CAPTURE.pt [CAPTURE.pt ...]
"""

import sys

import librosa
import torch
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from benchmarks.dataset.seedtts import load_seedtts_samples

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEEDTTS_META = "zhaochenyang20/seed-tts-eval-arrow"
SPEAKER = "ethan"
PASS_MULTIPLE = 1.25


def voice_clone_prompt(sample) -> str:
    """The prompt benchmarks.tasks.tts.VoiceCloneOmni sends for an English voice clone."""
    return (
        f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
        f"Now please read the following text out loud in the same voice and style: "
        f"{sample.target_text}"
    )


def samples_by_prompt() -> dict:
    return {
        voice_clone_prompt(sample): sample
        for sample in load_seedtts_samples(SEEDTTS_META, None, split="en")
    }


def build(dtype_name: str, out: str, paths: list[str]) -> None:
    dtype = getattr(torch, dtype_name)
    snapshot = snapshot_download(MODEL)
    processor = Qwen3OmniMoeProcessor.from_pretrained(snapshot)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=dtype,
        device_map="auto",
        max_memory={0: "76GiB", 1: "76GiB", "cpu": "400GiB"},
        enable_audio_output=True,
        experts_implementation="eager",
    ).eval()
    config = model.config
    thinker = model.thinker
    embedding = thinker.get_input_embeddings()
    prompts = samples_by_prompt()
    speaker_id = config.talker_config.speaker_id[SPEAKER]
    rows = {}
    with torch.inference_mode():
        for path in paths:
            record = torch.load(path, map_location="cpu")
            # note (ratish): the capture's request_text is empty on this tree; the talker's prompt ids hold the user turn.
            user_turn = processor.tokenizer.decode(record["prefill_input_ids"])
            matches = [prompt for prompt in prompts if prompt in user_turn]
            if len(matches) != 1:
                print(f"skip {record['request_id']}: {len(matches)} prompt matches")
                continue
            else:
                pass
            prompt_text = matches[0]
            sample = prompts[prompt_text]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": sample.ref_audio},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            audio, _ = librosa.load(sample.ref_audio, sr=16000)
            inputs = processor(
                text=text, audio=[audio], return_tensors="pt", padding=True
            )
            device = embedding.weight.device
            input_ids = inputs["input_ids"].to(device)
            thinker.rope_deltas = None
            outputs = thinker(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"].to(device),
                input_features=inputs["input_features"].to(device, dtype),
                feature_attention_mask=inputs["feature_attention_mask"].to(device),
                output_hidden_states=True,
                use_cache=False,
            )
            prompt_embed = outputs.hidden_states[0].to(device)
            prompt_hidden = outputs.hidden_states[
                config.talker_config.accept_hidden_layer
            ].to(device)
            reply = list(record["thinker_token_ids"])
            # note (ratish): generate() never feeds its last token, so HF's thinker_embed ends before it.
            fed_reply = torch.tensor([reply[:-1]], dtype=torch.long, device=device)
            thinker_embed = torch.cat([prompt_embed, embedding(fed_reply)], dim=1)
            thinker_hidden = torch.cat(
                [prompt_hidden, torch.zeros_like(embedding(fed_reply))], dim=1
            )
            sequences = torch.cat(
                [input_ids, torch.tensor([reply], dtype=torch.long, device=device)],
                dim=1,
            )
            multimodal_mask = (
                (sequences == config.thinker_config.audio_token_id)
                | (sequences == config.thinker_config.image_token_id)
                | (sequences == config.thinker_config.video_token_id)
            )
            sequence_length = input_ids.shape[1]
            positions = torch.arange(sequence_length, device=device)[None]
            im_start_positions = torch.where(
                input_ids == config.im_start_token_id, positions, -1
            )
            previous_im_start = torch.cummax(
                im_start_positions, dim=-1
            ).values.clamp_min(0)
            role_token_ids = input_ids.gather(
                1, (previous_im_start + 1).clamp_max(sequence_length - 1)
            )
            user_mask = role_token_ids == config.user_token_id
            full_user = model._get_talker_user_parts(
                0, sequence_length, multimodal_mask, thinker_hidden, thinker_embed
            )
            user_embeds = full_user[user_mask][None]
            user_ids = input_ids[user_mask][None]
            user_multimodal = multimodal_mask[:, :sequence_length][user_mask]
            assistant_start = (
                torch.nonzero(
                    (input_ids[0, :-1] == config.im_start_token_id)
                    & (input_ids[0, 1:] == config.assistant_token_id)
                )
                .flatten()[-1]
                .item()
            )
            specials = torch.tensor(
                [
                    [
                        config.tts_bos_token_id,
                        config.tts_eos_token_id,
                        config.tts_pad_token_id,
                    ]
                ],
                device=device,
            )
            tts_bos, tts_eos, tts_pad = (
                model.talker.text_projection(embedding(specials))
                .to(device)
                .chunk(3, dim=1)
            )
            assistant_embeds, assistant_ids, trailing = (
                model._get_talker_assistant_parts(
                    assistant_start,
                    thinker_embed.shape[1],
                    speaker_id,
                    thinker_embed,
                    tts_pad,
                    tts_bos,
                    tts_eos,
                )
            )
            rows[record["request_id"]] = {
                "prefill_embeds": torch.cat([user_embeds, assistant_embeds], dim=1)[0]
                .float()
                .cpu(),
                "prefill_ids": torch.cat([user_ids, assistant_ids.to(device)], dim=1)[
                    0
                ].cpu(),
                "user_rows": user_embeds.shape[1],
                "user_multimodal": user_multimodal.cpu(),
                "trailing": trailing[0].float().cpu(),
                "sample_id": sample.sample_id,
            }
            print(f"built {record['request_id']} {dtype_name}", flush=True)
    torch.save(rows, out)


def relative_l2(actual: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return (actual.float() - truth).norm(dim=-1) / truth.norm(dim=-1).clamp_min(1e-12)


def compare(truth_path: str, floor_path: str, paths: list[str]) -> None:
    truth_rows = torch.load(truth_path)
    floor_rows = torch.load(floor_path)
    totals = {}
    failures = 0
    for path in paths:
        record = torch.load(path, map_location="cpu")
        request_id = record["request_id"]
        if request_id not in truth_rows or request_id not in floor_rows:
            print(f"skip {request_id}: not built")
            continue
        else:
            pass
        truth = truth_rows[request_id]
        floor = floor_rows[request_id]
        ours = record["prefill_input_embeds"].float()
        ids_equal = torch.equal(record["prefill_input_ids"], truth["prefill_ids"])
        if not ids_equal or ours.shape != truth["prefill_embeds"].shape:
            failures += 1
            print(
                f"FAIL {request_id}: prompt ids equal {ids_equal}, rows ours {tuple(ours.shape)} "
                f"truth {tuple(truth['prefill_embeds'].shape)}"
            )
            continue
        else:
            pass
        user_rows = truth["user_rows"]
        groups = {
            "user text": torch.cat(
                [
                    ~truth["user_multimodal"],
                    torch.zeros(ours.shape[0] - user_rows, dtype=torch.bool),
                ]
            ),
            "user multimodal": torch.cat(
                [
                    truth["user_multimodal"],
                    torch.zeros(ours.shape[0] - user_rows, dtype=torch.bool),
                ]
            ),
            "assistant": torch.arange(ours.shape[0]) >= user_rows,
        }
        line = [truth["sample_id"]]
        for name, mask in groups.items():
            if not mask.any():
                continue
            else:
                pass
            ours_error = relative_l2(ours[mask], truth["prefill_embeds"][mask])
            floor_error = relative_l2(
                floor["prefill_embeds"][mask], truth["prefill_embeds"][mask]
            )
            total = totals.setdefault(name, [0.0, 0.0, 0])
            total[0] += ours_error.mean().item()
            total[1] += floor_error.mean().item()
            total[2] += 1
            line.append(
                f"{name}: ours {ours_error.mean().item():.2e} (max {ours_error.max().item():.2e}) "
                f"floor {floor_error.mean().item():.2e}"
            )
        print(" | ".join(line))
    for name, (ours_sum, floor_sum, count) in totals.items():
        ratio = ours_sum / max(floor_sum, 1e-30)
        verdict = "pass" if ratio <= PASS_MULTIPLE else "FAIL"
        print(
            f"== {name}: {count} requests, mean relative L2 ours {ours_sum / count:.3e}, "
            f"floor {floor_sum / count:.3e}, ratio {ratio:.2f} {verdict}"
        )
    print(f"== prompt id or shape failures: {failures}")


def main() -> None:
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4:])


if __name__ == "__main__":
    main()
