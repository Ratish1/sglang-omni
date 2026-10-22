"""Whether omni's talker MRoPE positions equal HF's on stage 10's video prompts.

Two runs on the box, CPU only:
    PYTHONPATH=<isolated 4.57.1 env> python3 talker_rope_check.py --hf --out <dir>
    cd <omni tree> && PYTHONPATH=<omni tree> python3 talker_rope_check.py --omni --out <dir>
--hf builds each of stage 10's 10 requests the cookbook way (hf_standard_reference.cookbook_inputs), takes the talker's
prompt as HF's generate does (the user segment from its <|im_start|>, then 9 tts_pad ids for the assistant part; the
system segment dropped) and records HF's talker get_rope_index over it with the request's grids, video seconds per
grid and audio feature lengths. --omni feeds the same ids and inputs to omni's compute_mrope_positions with the
thinker config, as build_sglang_talker_request does, and reports per prompt whether positions and delta are equal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SAMPLES = Path("/data/pr2800/videoamme10.json")
ASSISTANT_ROWS = 9


def talker_prompt_ids(input_ids: torch.Tensor, config) -> torch.Tensor:
    starts = (input_ids == config.im_start_token_id).nonzero().view(-1).tolist() + [input_ids.shape[0]]
    parts = []
    for start, end in zip(starts[:-1], starts[1:]):
        role = int(input_ids[start + 1])
        if role == config.user_token_id:
            parts.append(input_ids[start:end])
        elif role == config.assistant_token_id:
            parts.append(torch.full((ASSISTANT_ROWS,), config.tts_pad_token_id, dtype=input_ids.dtype))
        else:
            pass
    return torch.cat(parts)


def run_hf(out: Path) -> None:
    from hf_standard_reference import cookbook_inputs
    from transformers import AutoConfig, Qwen3OmniMoeProcessor
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(MODEL)
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    talker = Qwen3OmniMoeTalkerForConditionalGeneration.__new__(Qwen3OmniMoeTalkerForConditionalGeneration)
    object.__setattr__(talker, "config", config.talker_config)
    object.__setattr__(talker, "spatial_merge_size", config.talker_config.spatial_merge_size)
    spec = json.loads(SAMPLES.read_text())
    records = []
    for sample in spec["samples"]:
        content = [
            {"type": "video", "video": sample["video_path"], "fps": 2, "max_frames": 128, "max_pixels": 401408},
            {"type": "audio", "audio": sample["audio_path"]},
            {"type": "text", "text": spec["prompt"]},
        ]
        inputs = cookbook_inputs(processor, content, torch.device("cpu"), torch.float32)
        ids = talker_prompt_ids(inputs["input_ids"][0], config)
        audio_lengths = inputs["feature_attention_mask"].sum(dim=1)
        positions, delta = talker.get_rope_index(
            ids[None], inputs.get("image_grid_thw"), inputs.get("video_grid_thw"), torch.ones_like(ids[None]),
            None, audio_lengths, inputs.get("video_second_per_grid"))
        records.append(dict(sample_id=sample["sample_id"], ids=ids, video_grid_thw=inputs.get("video_grid_thw"),
                            video_second_per_grid=inputs.get("video_second_per_grid"), audio_feature_lengths=audio_lengths,
                            positions=positions[:, 0], delta=delta.view(-1)))
        print(f"{sample['sample_id']} talker prompt {ids.shape[0]} rows, delta {int(delta.view(-1)[0])}", flush=True)
    torch.save(records, out / "rope_hf.pt")


def run_omni(out: Path) -> None:
    from transformers import AutoConfig

    from sglang_omni.models.qwen3_omni.request_builders import compute_mrope_positions

    thinker_config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).thinker_config
    rows = []
    for record in torch.load(out / "rope_hf.pt", weights_only=False):
        model_inputs = dict(video_grid_thw=record["video_grid_thw"],
                            video_second_per_grid=record["video_second_per_grid"],
                            audio_feature_lengths=record["audio_feature_lengths"])
        positions, delta = compute_mrope_positions(record["ids"], model_inputs, thinker_config)
        hf_positions = record["positions"].to(positions.dtype)
        differing = (positions != hf_positions).any(dim=0)
        first = int(differing.nonzero()[0]) if differing.any() else None
        rows.append(dict(sample_id=record["sample_id"], rows=int(record["ids"].shape[0]),
                         positions_equal=not bool(differing.any()), differing_rows=int(differing.sum()),
                         first_differing_row=first, delta_hf=int(record["delta"][0]), delta_omni=int(delta.view(-1)[0])))
        print(json.dumps(rows[-1]), flush=True)
    (out / "rope_check.json").write_text(json.dumps(rows, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--omni", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.hf:
        run_hf(args.out)
    else:
        run_omni(args.out)


if __name__ == "__main__":
    main()
