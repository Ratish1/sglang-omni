# SPDX-License-Identifier: Apache-2.0
"""The served Qwen3-Omni thinker on the MMMU talker stage's image requests against the HF thinker in FP32.

QWEN3_OMNI_ACCURACY_SERVER is the base URL of a running omni server in the configuration under test. It receives HF's
processor outputs, which equal omni's own preprocessing of these requests (test_vision_preprocessing.py), at the
stage's concurrency with text and audio out, so the image encoder, the deepstack injection, the vision MRoPE positions
and the thinker run as the stage runs them. The HF thinker generates greedily in BF16 on
QWEN3_OMNI_ACCURACY_FLOOR_DEVICE, the floor a correct BF16 implementation sets, and in FP32 on
QWEN3_OMNI_ACCURACY_REFERENCE_DEVICE, which teacher-forces every recording. A departure is a step whose recorded token
is not the FP32 argmax. Omni passes when its logprob deviation from FP32 per request stays within the floor's by
comparison.py's bounds and it departs nowhere the FP32 top-two margin is wider than the widest margin at which HF's
own BF16 copy departs.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from benchmarks.benchmarker.data import RequestResult
from benchmarks.tasks.visual_understand import build_mmmu_result_records
from tests.test_model.qwen3_omni_component_accuracy.comparison import (
    CASE_BOUND,
    GEOMETRIC_MEAN_BOUND,
)
from tests.test_model.qwen3_omni_component_accuracy.mmmu_requests import (
    load_image_questions,
    reference_inputs,
)
from tests.test_model.qwen3_omni_component_accuracy.test_thinker import (
    serialized_tensor,
)
from tests.test_model.test_qwen3_omni_mmmu_talker_ci import CONCURRENCY, MAX_TOKENS

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

REQUEST_TIMEOUT_S = 600


@dataclass(frozen=True, kw_only=True)
class Recording:
    tokens: list[int]
    logprobs: list[float]


def media(inputs: BatchFeature) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in inputs.items()
        if name not in ("input_ids", "attention_mask")
    }


def device_media(
    inputs: BatchFeature, thinker: torch.nn.Module
) -> dict[str, torch.Tensor]:
    return {
        name: (
            tensor.to(thinker.device, dtype=thinker.dtype)
            if tensor.is_floating_point()
            else tensor.to(thinker.device)
        )
        for name, tensor in media(inputs).items()
    }


def served_greedy(server_url: str, inputs: BatchFeature) -> Recording:
    with httpx.Client(
        base_url=server_url, timeout=REQUEST_TIMEOUT_S, trust_env=False
    ) as server:
        response = server.post(
            "/generate",
            json={
                "input_ids": inputs["input_ids"][0].tolist(),
                "multimodal_train_inputs": {
                    "version": 1,
                    "tensors": {
                        name: serialized_tensor(tensor)
                        for name, tensor in media(inputs).items()
                    },
                },
                "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_TOKENS},
                "output_modalities": ["text", "audio"],
                "return_logprob": True,
            },
        )
    response.raise_for_status()
    pairs = response.json()["meta_info"]["output_token_logprobs"]
    return Recording(
        tokens=[int(pair[1]) for pair in pairs],
        logprobs=[float(pair[0]) for pair in pairs],
    )


def load_thinker(model_path: str, dtype: torch.dtype, device: str) -> torch.nn.Module:
    return Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map={"": device},
        enable_audio_output=False,
        experts_implementation="eager",
    ).thinker.eval()


def reference_greedy(
    thinker: torch.nn.Module, inputs: BatchFeature, eos_token_id: int
) -> Recording:
    prompt_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = thinker.generate(
            input_ids=inputs["input_ids"].to(thinker.device),
            attention_mask=inputs["attention_mask"].to(thinker.device),
            **device_media(inputs, thinker),
            max_new_tokens=MAX_TOKENS,
            eos_token_id=eos_token_id,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
    tokens = output.sequences[0, prompt_length:].tolist()
    logits = torch.stack([step[0].float().cpu() for step in output.logits])
    logprobs = torch.log_softmax(logits, dim=-1)[torch.arange(len(tokens)), tokens]
    return Recording(tokens=tokens, logprobs=logprobs.tolist())


def teacher_forced_logits(
    thinker: torch.nn.Module, inputs: BatchFeature, tokens: list[int]
) -> torch.Tensor:
    prompt_ids = inputs["input_ids"][0]
    full_ids = torch.cat([prompt_ids, torch.tensor(tokens, dtype=torch.long)])
    with torch.inference_mode():
        logits = thinker(
            input_ids=full_ids[None].to(thinker.device),
            attention_mask=torch.ones_like(full_ids)[None].to(thinker.device),
            **device_media(inputs, thinker),
            use_cache=False,
        ).logits
    start = prompt_ids.shape[0] - 1
    return logits[0, start : start + len(tokens)].float().cpu()


def follow(truth_logits: torch.Tensor, recording: Recording) -> dict[str, object]:
    steps = torch.arange(len(recording.tokens))
    tokens = torch.tensor(recording.tokens)
    truth_logprobs = torch.log_softmax(truth_logits, dim=-1)[steps, tokens]
    top2 = truth_logits.topk(2, dim=-1).values
    margins = top2[:, 0] - top2[:, 1]
    departures = (truth_logits.argmax(dim=-1) != tokens).nonzero().flatten()
    return {
        "steps": len(recording.tokens),
        "deviation_sum": (truth_logprobs - torch.tensor(recording.logprobs))
        .abs()
        .sum()
        .item(),
        "departure_steps": departures.tolist(),
        "departure_margins": margins[departures].tolist(),
    }


def test_thinker_images_follow_reference(
    model_path: str, report_dir: Path | None
) -> None:
    server_url = os.environ.get("QWEN3_OMNI_ACCURACY_SERVER")
    if server_url is None:
        pytest.skip("Set QWEN3_OMNI_ACCURACY_SERVER to the omni server under test")
    else:
        pass
    questions = load_image_questions()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    request_inputs = [reference_inputs(processor, sample) for sample in questions]
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        served = list(
            pool.map(lambda inputs: served_greedy(server_url, inputs), request_inputs)
        )

    truth_thinker = load_thinker(
        model_path,
        torch.float32,
        os.environ.get("QWEN3_OMNI_ACCURACY_REFERENCE_DEVICE", "cuda:0"),
    )
    floor_thinker = load_thinker(
        model_path,
        torch.bfloat16,
        os.environ.get("QWEN3_OMNI_ACCURACY_FLOOR_DEVICE", "cuda:1"),
    )
    recordings: dict[str, list[Recording]] = {"omni": served, "floor": [], "truth": []}
    follows: dict[str, list[dict[str, object]]] = {name: [] for name in recordings}
    for index, inputs in enumerate(request_inputs):
        recordings["floor"].append(
            reference_greedy(floor_thinker, inputs, eos_token_id)
        )
        recordings["truth"].append(
            reference_greedy(truth_thinker, inputs, eos_token_id)
        )
        for name in recordings:
            truth_logits = teacher_forced_logits(
                truth_thinker, inputs, recordings[name][index].tokens
            )
            follows[name].append(follow(truth_logits, recordings[name][index]))

    report: dict[str, object] = {}
    for name, recording_list in recordings.items():
        texts = [
            processor.tokenizer.decode(recording.tokens, skip_special_tokens=True)
            for recording in recording_list
        ]
        scored = build_mmmu_result_records(
            questions,
            [
                RequestResult(request_id=sample.sample_id, text=text, is_success=True)
                for sample, text in zip(questions, texts)
            ],
        )
        steps = sum(entry["steps"] for entry in follows[name])
        margins = [
            margin for entry in follows[name] for margin in entry["departure_margins"]
        ]
        report[name] = {
            "summary": {
                "correct": sum(record["is_correct"] for record in scored),
                "steps": steps,
                "deviation_mean": sum(entry["deviation_sum"] for entry in follows[name])
                / steps,
                "departures": len(margins),
                "departure_margin_max": max(margins, default=0.0),
            },
            "per_request": [
                {
                    "sample_id": record["sample_id"],
                    "predicted": record["predicted"],
                    "is_correct": record["is_correct"],
                    "is_mc_fallback": record["is_mc_fallback"],
                    **entry,
                    "text": text,
                }
                for record, entry, text in zip(scored, follows[name], texts)
            ],
        }
    truth_letters = [entry["predicted"] for entry in report["truth"]["per_request"]]
    for name in recordings:
        report[name]["summary"]["answers_equal_to_truth"] = sum(
            entry["predicted"] == letter
            for entry, letter in zip(report[name]["per_request"], truth_letters)
        )
    deviation_ratios = [
        (omni["deviation_sum"] / omni["steps"])
        / max(floor["deviation_sum"] / floor["steps"], 1e-12)
        for omni, floor in zip(follows["omni"], follows["floor"])
    ]
    report["omni"]["summary"]["deviation_floor_ratio"] = math.exp(
        statistics.fmean(math.log(max(ratio, 1e-12)) for ratio in deviation_ratios)
    )
    report["omni"]["summary"]["deviation_floor_ratio_max"] = max(deviation_ratios)

    if report_dir is not None:
        (report_dir / "thinker_images.json").write_text(json.dumps(report, indent=1))
    else:
        pass
    print(
        "\n"
        + "\n".join(
            f"[thinker_images] {name} {json.dumps(report[name]['summary'])}"
            for name in recordings
        )
    )
    omni_summary = report["omni"]["summary"]
    floor_margin = report["floor"]["summary"]["departure_margin_max"]
    assert (
        report["truth"]["summary"]["departures"] == 0
    ), "teacher forcing does not reproduce the reference's greedy"
    assert (
        omni_summary["deviation_floor_ratio"] <= GEOMETRIC_MEAN_BOUND
        and omni_summary["deviation_floor_ratio_max"] <= CASE_BOUND
    ), f"omni's logprob deviation from FP32 strays past the BF16 floor's: {omni_summary}"
    assert (
        omni_summary["departure_margin_max"] <= floor_margin
    ), f"omni departs where the FP32 margin passes {floor_margin:.3f}, the widest HF's BF16 departs at"
