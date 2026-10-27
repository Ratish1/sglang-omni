"""Official qwen-tts 0.1.1's talker teacher-forced on frames our server generated. For each chosen case: the official
prompt (Qwen3TTSModel's voice-clone steps up to the talker, captured where they call the talker's generate), then the
served frames as the talker's step inputs (each frame's 16 codebook embeddings summed, plus the trailing text or the
pad, as official's own step builds them, `modeling_qwen3_tts.py:1670-1692`), all in one causal forward. Per step,
after official's control-id suppression and two-frame minimum: codec EOS's log-probability over the whole vocabulary
at temperature 1, EOS under temperature 0.9 and top-k 50, EOS's rank, the top id and its log-probability; the same
fields `probe/qwen3-tts-teacher-forced` logs on our side.

usage: /data/refenv/bin/python teacher_forced_reference.py SERVED_DIR PROBE_DIR CASES.jsonl OUT.jsonl
       --model Qwen/Qwen3-TTS-12Hz-0.6B-Base [--card 0] [--natural-per-set 50] [--language English]

Runs in a venv holding qwen-tts 0.1.1's pins (transformers 4.57.3, accelerate 1.12.0, torch 2.10.0): its talker does
not load under the serving image's Transformers.
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel

SPEECH_ID = re.compile(r"speech-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
CONTROL_IDS = 1024
MIN_NEW_FRAMES = 2
TEMPERATURE = 0.9
TOP_K = 50


class TalkerInputs(Exception):
    def __init__(self, inputs: dict[str, object]) -> None:
        super().__init__("talker inputs captured")
        self.inputs = inputs


def capture_talker_inputs(**kwargs: object) -> None:
    raise TalkerInputs(kwargs)


def served_frames(probe: Path) -> dict[str, list[list[int]]]:
    frames: dict[str, dict[int, list[int]]] = defaultdict(dict)
    for path in sorted(probe.glob("codes_*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            request_id, frame, *codes = line.split()
            match = SPEECH_ID.search(request_id)
            if match is None:
                continue
            else:
                pass
            frames[match.group(0)][int(frame)] = [int(code) for code in codes]
    return {
        speech_id: [by_index[index] for index in range(len(by_index))]
        for speech_id, by_index in frames.items()
        if sorted(by_index) == list(range(len(by_index)))
    }


def chosen_cases(served: Path, natural_per_set: int) -> list[dict[str, object]]:
    records = [
        json.loads(line) for line in (served / "results.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    records = [record for record in records if "speech_id" in record and "frames" in record]
    chosen = [record for record in records if record.get("finish_reason") == "length"]
    by_set: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record.get("finish_reason") == "stop":
            by_set[str(record["set"])].append(record)
        else:
            pass
    for natural in by_set.values():
        natural.sort(key=lambda record: str(record["id"]))
        stride = max(1, len(natural) // natural_per_set)
        chosen += natural[::stride][:natural_per_set]
    return chosen


@torch.no_grad()
def step_fields(logits: torch.Tensor, codec_eos: int) -> list[list[float]]:
    shaped = logits.float()
    shaped[:, shaped.shape[1] - CONTROL_IDS : codec_eos] = float("-inf")
    shaped[:, codec_eos + 1 :] = float("-inf")
    shaped[:MIN_NEW_FRAMES, codec_eos] = float("-inf")
    log_probs = torch.log_softmax(shaped, dim=-1)
    top_log_prob, top_id = log_probs.max(dim=-1)
    scaled = shaped / TEMPERATURE
    top_values = torch.topk(scaled, TOP_K, dim=-1).values
    eos_scaled = scaled[:, codec_eos]
    eos_top_k = torch.where(
        eos_scaled >= top_values[:, -1],
        torch.exp(eos_scaled - torch.logsumexp(top_values, dim=-1)),
        torch.zeros_like(eos_scaled),
    )
    eos_rank = (shaped > shaped[:, codec_eos : codec_eos + 1]).sum(dim=-1)
    return torch.stack(
        (log_probs[:, codec_eos], eos_top_k, eos_rank.float(), top_id.float(), top_log_prob), dim=1
    ).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("served", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--natural-per-set", type=int, default=50)
    parser.add_argument("--language", choices=("auto", "English"), default="English")
    args = parser.parse_args()
    cases = {
        case["id"]: case
        for case in (json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line)
    }
    frames_by_speech = served_frames(args.probe)
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map=f"cuda:{args.card}", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    talker = model.model.talker
    talker.generate = capture_talker_inputs
    codec_eos = int(talker.config.codec_eos_token_id)
    groups = talker.config.num_code_groups
    began = time.perf_counter()
    written = 0
    with args.out.open("w", encoding="utf-8") as out:
        for record in chosen_cases(args.served, args.natural_per_set):
            frames = frames_by_speech.get(str(record["speech_id"]))
            if not frames:
                print(f"no frames for {record['id']}", flush=True)
                continue
            else:
                pass
            case = cases[str(record["id"])]
            with torch.no_grad():
                items = model.create_voice_clone_prompt(
                    ref_audio=[case["ref_audio"]],
                    ref_text=[case.get("ref_text")],
                    x_vector_only_mode=[case["mode"] == "xvec"],
                )
                input_ids = model._tokenize_texts([model._build_assistant_text(case["input"])])
                ref_ids = [
                    model._tokenize_texts([model._build_ref_text(item.ref_text)])[0] if item.ref_text else None
                    for item in items
                ]
                try:
                    model.model.generate(
                        input_ids=input_ids,
                        ref_ids=ref_ids,
                        voice_clone_prompt=model._prompt_items_to_voice_clone_prompt(items),
                        languages=[args.language],
                        non_streaming_mode=False,
                        **model._merge_generate_kwargs(max_new_tokens=len(frames) + 1),
                    )
                    raise RuntimeError("the talker's generate was not reached")
                except TalkerInputs as captured:
                    inputs = captured.inputs
                prefill = inputs["inputs_embeds"][0]
                trailing = inputs["trailing_text_hidden"][0]
                pad = inputs["tts_pad_embed"][0, 0]
                codes = torch.tensor(frames, dtype=torch.long, device=prefill.device)
                codec_hiddens = torch.cat(
                    [talker.get_input_embeddings()(codes[:, :1])]
                    + [
                        talker.code_predictor.get_input_embeddings()[index](codes[:, index + 1 : index + 2])
                        for index in range(groups - 1)
                    ],
                    dim=1,
                )
                step_inputs = codec_hiddens.sum(1)
                text = torch.stack([trailing[step] if step < trailing.shape[0] else pad for step in range(len(frames))])
                sequence = torch.cat((prefill, step_inputs + text), dim=0).unsqueeze(0)
                hidden = talker.model(
                    inputs_embeds=sequence,
                    attention_mask=torch.ones(sequence.shape[:2], dtype=torch.long, device=sequence.device),
                    use_cache=False,
                ).last_hidden_state
                logits = talker.codec_head(hidden[0, prefill.shape[0] - 1 :])
            out.write(
                json.dumps({
                    "id": record["id"], "speech_id": record["speech_id"], "set": record["set"], "mode": record["mode"],
                    "finish_reason": record["finish_reason"], "frames": len(frames),
                    "prefill_positions": int(prefill.shape[0]), "trailing_text": int(trailing.shape[0]),
                    "official": step_fields(logits, codec_eos),
                })
                + "\n"
            )
            written += 1
            if written % 50 == 0:
                print(f"{written} cases, {time.perf_counter() - began:.0f} s", flush=True)
            else:
                pass
    print(f"teacher forced {written} cases in {time.perf_counter() - began:.0f} s", flush=True)


if __name__ == "__main__":
    main()
