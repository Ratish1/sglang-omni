"""The baseline for "HF BF16 shares omni's departures": how many of HF BF16's own departures it makes again uncached.

test_thinker_concurrency.py counts omni's FP32 departures where an uncached HF BF16 forward over omni's tokens also picks
omni's token. HF BF16's own recording is the same question without omni: cached against uncached in one implementation.
Reads the recordings and departures from the run's reference.json and reports the share, by modality and margin.

usage: PYTHONPATH=TREE python3 floor_shared_baseline.py REFERENCE_JSON REPORT_JSON OUT_JSON   (HF on cuda:1)
"""

import json
import random
import sys
from pathlib import Path

import qwen_omni_utils
import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

from tests.test_model.qwen3_omni_component_accuracy.test_thinker_concurrency import (
    ORDER_SEED,
    audio_requests,
    image_requests,
    load_thinker,
    teacher_forced_logits,
    video_requests,
)

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEVICE = "cuda:1"
MARGINS = (0.0, 0.1, 0.25, 0.5)


def shares(margins_and_shared: list[tuple[float, bool]]) -> dict[str, list[int]]:
    return {
        f"margin_above_{threshold}": [
            sum(shared for margin, shared in margins_and_shared if margin > threshold),
            sum(margin > threshold for margin, _ in margins_and_shared),
        ]
        for threshold in MARGINS
    }


def main() -> None:
    reference = json.loads(Path(sys.argv[1]).read_text())
    report = json.loads(Path(sys.argv[2]).read_text())
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    requests = [
        *image_requests(processor),
        *audio_requests(processor),
        *video_requests(processor, qwen_omni_utils.process_mm_info),
    ]
    random.Random(ORDER_SEED).shuffle(requests)
    rows = report["per_request"]
    assert [row["sample_id"] for row in rows] == [request.sample_id for request in requests], "request order differs"
    thinker = load_thinker(MODEL, torch.bfloat16, DEVICE)
    floor_pairs: list[tuple[float, bool]] = []
    omni_pairs: list[tuple[float, bool]] = []
    per_modality: dict[str, list[tuple[float, bool]]] = {"image": [], "audio": [], "video": []}
    for index, request in enumerate(requests):
        tokens = reference["recordings"]["floor"][index]["tokens"]
        follow = reference["follows"]["floor"][index]
        choices = teacher_forced_logits(thinker, request.inputs, tokens).argmax(dim=-1)
        for step, margin in zip(follow["departure_steps"], follow["departure_margins"]):
            pair = (margin, int(choices[step]) == tokens[step])
            floor_pairs.append(pair)
            per_modality[request.modality].append(pair)
        shared = rows[index]["omni_departure_margins_hf_bf16_shares"]
        remaining = list(shared)
        for margin in rows[index]["omni"]["departure_margins"]:
            if margin in remaining:
                remaining.remove(margin)
                omni_pairs.append((margin, True))
            else:
                omni_pairs.append((margin, False))
        print(f"{index + 1} of {len(requests)} {request.sample_id}", flush=True)
    result = {
        "hf_bf16_own_departures_it_repeats_uncached": shares(floor_pairs),
        "omni_departures_hf_bf16_makes_uncached": shares(omni_pairs),
        "hf_bf16_by_modality": {modality: shares(pairs) for modality, pairs in per_modality.items()},
    }
    Path(sys.argv[3]).write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    print("baseline done", flush=True)


if __name__ == "__main__":
    main()
