"""Our talker prompt against official qwen-tts 0.1.1's for the same request: the prompt our server built and saved
(`probe/qwen3-tts-teacher-forced`, QWEN3_TTS_PROBE_PROMPT_DIR: the prefill embeddings, the trailing text and the ICL
reference codes) beside official's for that case (its voice-clone steps up to the talker, captured where they call
the talker's generate, and the reference codes its prompt item carries). Prints per mode and set: shapes that agree,
the largest absolute difference of the prefill rows and of the trailing text, the first prefill row past the
tolerance, and for ICL how many reference codes agree and the first frame where they part; then the cases that part
most.

usage: /data/refenv/bin/python prompt_compare_reference.py CASES.jsonl SERVED_DIR PROMPT_DIR
       --model Qwen/Qwen3-TTS-12Hz-0.6B-Base [--card 0] [--language English] [--tolerance 0.05]

Runs in a venv holding qwen-tts 0.1.1's pins (transformers 4.57.3, accelerate 1.12.0, torch 2.10.0).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel

WORST_CASES = 20


class TalkerInputs(Exception):
    def __init__(self, inputs: dict[str, object]) -> None:
        super().__init__("talker inputs captured")
        self.inputs = inputs


def capture_talker_inputs(**kwargs: object) -> None:
    raise TalkerInputs(kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("served", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--language", choices=("auto", "English"), default="English")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    cases = {
        case["id"]: case
        for case in (json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line)
    }
    records = [
        json.loads(line) for line in (args.served / "results.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map=f"cuda:{args.card}", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.model.talker.generate = capture_talker_inputs
    rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        saved = sorted(args.prompts.glob(f"*{record.get('speech_id', 'none')}*.pt"))
        if not saved:
            continue
        else:
            pass
        ours = torch.load(saved[0])
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
                    **model._merge_generate_kwargs(max_new_tokens=2),
                )
                raise RuntimeError("the talker's generate was not reached")
            except TalkerInputs as captured:
                inputs = captured.inputs
        official_prefill = inputs["inputs_embeds"][0].float().cpu()
        official_trailing = inputs["trailing_text_hidden"][0].float().cpu()
        our_prefill = ours["input_embeds"].reshape(-1, official_prefill.shape[-1])
        our_trailing = ours["trailing_text_hidden"].reshape(-1, official_trailing.shape[-1])
        row: dict[str, object] = {
            "id": record["id"], "finish_reason": record.get("finish_reason"),
            "prefill_rows": f"{our_prefill.shape[0]}/{official_prefill.shape[0]}",
            "trailing_rows": f"{our_trailing.shape[0]}/{official_trailing.shape[0]}",
        }
        if our_prefill.shape == official_prefill.shape:
            per_row = (our_prefill - official_prefill).abs().amax(dim=-1)
            beyond = torch.nonzero(per_row > args.tolerance).flatten()
            row["prefill_max"] = float(per_row.max())
            row["first_row_beyond"] = int(beyond[0]) if beyond.numel() else -1
            row["rows_beyond"] = int(beyond.numel())
        else:
            row["prefill_max"] = float("inf")
            row["first_row_beyond"] = 0
            row["rows_beyond"] = -1
        if our_trailing.shape == official_trailing.shape:
            row["trailing_max"] = float((our_trailing - official_trailing).abs().max())
        else:
            row["trailing_max"] = float("inf")
        official_code = items[0].ref_code
        our_code = ours["ref_code"]
        if official_code is not None and our_code is not None:
            official_code = official_code.cpu().reshape(-1, official_code.shape[-1])
            our_code = our_code.cpu().reshape(-1, official_code.shape[-1])
            row["ref_frames"] = f"{our_code.shape[0]}/{official_code.shape[0]}"
            frames = min(our_code.shape[0], official_code.shape[0])
            equal = our_code[:frames] == official_code[:frames]
            row["ref_codes_equal"] = float(equal.float().mean())
            parted = torch.nonzero(~equal.all(dim=-1)).flatten()
            row["ref_first_frame_parted"] = int(parted[0]) if parted.numel() else -1
            row["ref_codebook0_equal"] = float(equal[:, 0].float().mean())
        else:
            pass
        rows[f"{case['mode']} {case['set']}"].append(row)
    print(
        "mode set | cases | prefill shapes equal | prefill max |diff| q50 / max | rows past tolerance (cases) | "
        "trailing max |diff| q50 / max | ICL reference codes equal (mean) | codebook 0 equal | frames equal in count"
    )
    every: list[dict[str, object]] = []
    for key in sorted(rows):
        values = rows[key]
        every += values
        same_shape = [value for value in values if value["prefill_max"] != float("inf")]
        prefill = sorted(float(value["prefill_max"]) for value in same_shape)
        trailing = sorted(float(value["trailing_max"]) for value in values if value["trailing_max"] != float("inf"))
        codes = [float(value["ref_codes_equal"]) for value in values if "ref_codes_equal" in value]
        codebook0 = [float(value["ref_codebook0_equal"]) for value in values if "ref_codebook0_equal" in value]
        frames_equal = sum(
            str(value["ref_frames"]).split("/")[0] == str(value["ref_frames"]).split("/")[1]
            for value in values
            if "ref_frames" in value
        )
        print(
            f"{key} | {len(values)} | {len(same_shape)} | "
            f"{prefill[len(prefill) // 2] if prefill else float('nan'):.3g} / {prefill[-1] if prefill else float('nan'):.3g} | "
            f"{sum(int(value['rows_beyond']) > 0 for value in same_shape)} | "
            f"{trailing[len(trailing) // 2] if trailing else float('nan'):.3g} / {trailing[-1] if trailing else float('nan'):.3g} | "
            f"{sum(codes) / len(codes) if codes else float('nan'):.4f} | "
            f"{sum(codebook0) / len(codebook0) if codebook0 else float('nan'):.4f} | {frames_equal}/{len(codes)}"
        )
    print("cases that part most: id | finish | prefill rows ours/official | prefill max | first row past | rows past | "
          "trailing max | ref frames ours/official | ref codes equal | first ref frame parted")
    for row in sorted(every, key=lambda value: -float(value["prefill_max"]))[:WORST_CASES]:
        print(
            f"{row['id']} | {row['finish_reason']} | {row['prefill_rows']} | {float(row['prefill_max']):.3g} | "
            f"{row['first_row_beyond']} | {row['rows_beyond']} | {float(row['trailing_max']):.3g} | "
            f"{row.get('ref_frames', '-')} | {row.get('ref_codes_equal', '-')} | {row.get('ref_first_frame_parted', '-')}"
        )


if __name__ == "__main__":
    main()
