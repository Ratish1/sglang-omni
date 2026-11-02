# SPDX-License-Identifier: Apache-2.0
"""The served Qwen3-Omni thinker on image, audio and video requests at once against the HF thinker in FP32.

QWEN3_OMNI_ACCURACY_SERVER is the base URL of a running omni server in the configuration under test. Every request goes
out with text and audio, mixed in one seeded order, at QWEN3_OMNI_ACCURACY_CONCURRENCY in flight, so the encoders, the
deepstack injection, the MRoPE positions, mixed batches and, on video's 14,000-token prompts, the chunked prefill all
run as serving runs them. Images and audio carry HF's processor outputs; video goes through omni's preprocessing, which
equals HF's (test_vision_preprocessing.py), because its tensors are too large for a request body.

HF generates greedily in BF16 (the floor a correct BF16 implementation sets) and then in FP32 on
QWEN3_OMNI_ACCURACY_REFERENCE_DEVICE, one at a time because together they do not fit one card; FP32 teacher-forces every
recording. A departure is a step whose recorded token is not the FP32 argmax. Omni passes when its logprob deviation
from FP32 per request stays within the floor's by comparison.py's bounds and it departs nowhere the FP32 top-two margin
is past twice the widest margin at which HF's own BF16 copy departs. HF BF16 also teacher-forces omni's tokens, which
counts the departures HF's BF16 makes itself on the same prefix. QWEN3_OMNI_ACCURACY_REFERENCE_CACHE, when set, names a
JSON file that keeps the HF recordings and their teacher-forced numbers, which do not depend on the server.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import librosa
import pytest
import torch
import transformers
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from benchmarks.dataset.mmmu import load_mmmu_samples
from benchmarks.dataset.prepare import DATASETS
from tests.test_model.qwen3_omni_component_accuracy.comparison import (
    CASE_BOUND,
    GEOMETRIC_MEAN_BOUND,
)
from tests.test_model.qwen3_omni_component_accuracy.media_requests import (
    load_video_media_questions,
)
from tests.test_model.qwen3_omni_component_accuracy.mmmu_requests import (
    reference_inputs,
)
from tests.test_model.qwen3_omni_component_accuracy.stage6_requests import (
    load_audio_questions,
)
from tests.test_model.qwen3_omni_component_accuracy.test_thinker import (
    REFERENCE_SAMPLE_RATE,
    serialized_tensor,
)
from tests.test_model.test_qwen3_omni_mmmu_talker_ci import MAX_TOKENS, MMMU_TTS_PROMPT
from tests.test_model.test_qwen3_omni_videomme_talker_ci import (
    MAX_SAMPLES as VIDEO_MAX_SAMPLES,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

REQUEST_TIMEOUT_S = 1800
ORDER_SEED = 20261102
# note (ratish): the largest departures of two equal runs differ by about a third: over 9,300 steps omni's reached 1.065
# and HF BF16's 0.868, with 6 each past 0.5. A departure past twice BF16's largest needs an error no BF16 rounding made.
CONFIDENT_MARGIN_FACTOR = 2.0


@dataclass(frozen=True, kw_only=True)
class ThinkerRequest:
    sample_id: str
    modality: str
    inputs: BatchFeature
    body: dict[str, object]


@dataclass(frozen=True, kw_only=True)
class Recording:
    tokens: list[int]
    logprobs: list[float]


@dataclass(frozen=True, kw_only=True)
class Reference:
    """The HF thinker's greedy recordings in BF16 ("floor") and FP32 ("truth"), each teacher-forced through FP32."""

    recordings: dict[str, list[Recording]]
    follows: dict[str, list[dict[str, object]]]


def media(inputs: BatchFeature) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in inputs.items()
        if name not in ("input_ids", "attention_mask")
    }


def tensor_body(inputs: BatchFeature) -> dict[str, object]:
    return {
        "input_ids": inputs["input_ids"][0].tolist(),
        "multimodal_train_inputs": {
            "version": 1,
            "tensors": {
                name: serialized_tensor(tensor) for name, tensor in media(inputs).items()
            },
        },
    }


def image_requests(processor: Qwen3OmniMoeProcessor) -> list[ThinkerRequest]:
    requests = []
    for sample in load_mmmu_samples(
        repo_id=DATASETS["mmmu-ci-50"], instruction_override=MMMU_TTS_PROMPT
    ):
        inputs = reference_inputs(processor, sample)
        requests.append(
            ThinkerRequest(
                sample_id=sample.sample_id,
                modality="image",
                inputs=inputs,
                body=tensor_body(inputs),
            )
        )
    return requests


def audio_requests(processor: Qwen3OmniMoeProcessor) -> list[ThinkerRequest]:
    requests = []
    for question in load_audio_questions():
        waveform, _ = librosa.load(question.audio_path, sr=REFERENCE_SAMPLE_RATE)
        text = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": question.audio_path},
                        {"type": "text", "text": question.question_text},
                    ],
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            text=text, audio=[waveform], return_tensors="pt", padding=True
        )
        requests.append(
            ThinkerRequest(
                sample_id=question.sample_id,
                modality="audio",
                inputs=inputs,
                body=tensor_body(inputs),
            )
        )
    return requests


def video_requests(
    processor: Qwen3OmniMoeProcessor, process_mm_info
) -> list[ThinkerRequest]:
    requests = []
    for question in load_video_media_questions()[:VIDEO_MAX_SAMPLES]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": question.media["videos"][0],
                        "fps": question.media["video_fps"],
                        "max_frames": question.media["video_max_frames"],
                    },
                    {"type": "text", "text": question.question_text},
                ],
            }
        ]
        _, _, videos, video_kwargs = process_mm_info(
            conversation, use_audio_in_video=False, return_video_kwargs=True
        )
        inputs = processor(
            text=processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            ),
            videos=videos,
            fps=video_kwargs["fps"][0],
            do_sample_frames=False,
            return_tensors="pt",
            padding=True,
        )
        requests.append(
            ThinkerRequest(
                sample_id=question.sample_id,
                modality="video",
                inputs=inputs,
                body={
                    "messages": [{"role": "user", "content": question.question_text}],
                    "metadata": question.media,
                },
            )
        )
    return requests


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


def served_greedy(server_url: str, request: ThinkerRequest) -> Recording:
    with httpx.Client(
        base_url=server_url, timeout=REQUEST_TIMEOUT_S, trust_env=False
    ) as server:
        response = server.post(
            "/generate",
            json={
                **request.body,
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


def reference_key(
    model_path: str, requests: list[ThinkerRequest]
) -> dict[str, object]:
    return {
        "model_path": model_path,
        "transformers": transformers.__version__,
        "max_tokens": MAX_TOKENS,
        "input_ids": [request.inputs["input_ids"][0].tolist() for request in requests],
        "media_sha256": [
            {
                name: hashlib.sha256(
                    tensor.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
                for name, tensor in media(request.inputs).items()
            }
            for request in requests
        ],
    }


def load_reference(path: Path | None, key: dict[str, object]) -> Reference | None:
    if path is None or not path.exists():
        return None
    else:
        stored = json.loads(path.read_text())
        if stored["key"] != key:
            return None
        else:
            return Reference(
                recordings={
                    name: [Recording(**recording) for recording in recordings]
                    for name, recordings in stored["recordings"].items()
                },
                follows=stored["follows"],
            )


def compute_reference(
    truth_thinker: torch.nn.Module,
    requests: list[ThinkerRequest],
    floor: list[Recording],
    eos_token_id: int,
) -> Reference:
    recordings: dict[str, list[Recording]] = {"floor": floor, "truth": []}
    follows: dict[str, list[dict[str, object]]] = {"floor": [], "truth": []}
    for index, request in enumerate(requests):
        recordings["truth"].append(
            reference_greedy(truth_thinker, request.inputs, eos_token_id)
        )
        for name in recordings:
            truth_logits = teacher_forced_logits(
                truth_thinker, request.inputs, recordings[name][index].tokens
            )
            follows[name].append(follow(truth_logits, recordings[name][index]))
    return Reference(recordings=recordings, follows=follows)


def summary(
    follows: list[dict[str, object]],
    recordings: list[Recording],
    truth: list[Recording],
) -> dict[str, object]:
    steps = sum(entry["steps"] for entry in follows)
    margins = [margin for entry in follows for margin in entry["departure_margins"]]
    return {
        "requests": len(follows),
        "steps": steps,
        "deviation_mean": sum(entry["deviation_sum"] for entry in follows) / steps,
        "departures": len(margins),
        "departure_margin_max": max(margins, default=0.0),
        "tokens_equal_to_truth": sum(
            recording.tokens == reference.tokens
            for recording, reference in zip(recordings, truth)
        ),
    }


def test_thinker_follows_reference_under_load(
    model_path: str, report_dir: Path | None
) -> None:
    server_url = os.environ.get("QWEN3_OMNI_ACCURACY_SERVER")
    if server_url is None:
        pytest.skip("Set QWEN3_OMNI_ACCURACY_SERVER to the omni server under test")
    else:
        pass
    qwen_omni_utils = pytest.importorskip("qwen_omni_utils")
    device = os.environ.get("QWEN3_OMNI_ACCURACY_REFERENCE_DEVICE", "cuda:1")
    concurrency = int(os.environ.get("QWEN3_OMNI_ACCURACY_CONCURRENCY", 64))
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    requests = [
        *image_requests(processor),
        *audio_requests(processor),
        *video_requests(processor, qwen_omni_utils.process_mm_info),
    ]
    random.Random(ORDER_SEED).shuffle(requests)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        served = list(
            pool.map(lambda request: served_greedy(server_url, request), requests)
        )

    cache = os.environ.get("QWEN3_OMNI_ACCURACY_REFERENCE_CACHE")
    cache_path = Path(cache) if cache is not None else None
    key = reference_key(model_path, requests)
    reference = load_reference(cache_path, key)
    floor_thinker = load_thinker(model_path, torch.bfloat16, device)
    omni_floor_choices = [
        teacher_forced_logits(floor_thinker, request.inputs, recording.tokens).argmax(
            dim=-1
        )
        for request, recording in zip(requests, served)
    ]
    floor = None
    if reference is None:
        floor = [
            reference_greedy(floor_thinker, request.inputs, eos_token_id)
            for request in requests
        ]
    else:
        pass
    del floor_thinker
    gc.collect()
    torch.cuda.empty_cache()
    truth_thinker = load_thinker(model_path, torch.float32, device)
    if reference is None:
        reference = compute_reference(truth_thinker, requests, floor, eos_token_id)
        if cache_path is not None:
            cache_path.write_text(
                json.dumps(
                    {
                        "key": key,
                        "recordings": {
                            name: [asdict(recording) for recording in recordings]
                            for name, recordings in reference.recordings.items()
                        },
                        "follows": reference.follows,
                    }
                )
            )
        else:
            pass
    else:
        pass
    recordings = {"omni": served, **reference.recordings}
    follows = {
        "omni": [
            follow(
                teacher_forced_logits(truth_thinker, request.inputs, recording.tokens),
                recording,
            )
            for request, recording in zip(requests, served)
        ],
        **reference.follows,
    }

    def picked(name: str, modality: str | None) -> dict[str, object]:
        rows = [
            index
            for index, request in enumerate(requests)
            if modality is None or request.modality == modality
        ]
        return summary(
            [follows[name][index] for index in rows],
            [recordings[name][index] for index in rows],
            [reference.recordings["truth"][index] for index in rows],
        )

    report: dict[str, object] = {
        scope: {name: picked(name, modality) for name in recordings}
        for scope, modality in (
            ("all", None),
            ("image", "image"),
            ("audio", "audio"),
            ("video", "video"),
        )
    }
    deviation_ratios = [
        (omni["deviation_sum"] / omni["steps"])
        / max(floor_entry["deviation_sum"] / floor_entry["steps"], 1e-12)
        for omni, floor_entry in zip(follows["omni"], follows["floor"])
    ]
    report["deviation_floor_ratio"] = math.exp(
        statistics.fmean(math.log(max(ratio, 1e-12)) for ratio in deviation_ratios)
    )
    report["deviation_floor_ratio_max"] = max(deviation_ratios)
    shared_margins = [
        [
            margin
            for step, margin in zip(
                follows["omni"][index]["departure_steps"],
                follows["omni"][index]["departure_margins"],
            )
            if int(omni_floor_choices[index][step]) == served[index].tokens[step]
        ]
        for index in range(len(requests))
    ]
    shared = sorted(margin for margins in shared_margins for margin in margins)
    report["omni_departures_hf_bf16_shares"] = {
        "shared": len(shared),
        "of": report["all"]["omni"]["departures"],
        "shared_margin_max": max(shared, default=0.0),
    }
    report["per_request"] = [
        {
            "sample_id": request.sample_id,
            "modality": request.modality,
            "prompt_tokens": int(request.inputs["input_ids"].shape[1]),
            "omni_tokens": served[index].tokens,
            "omni_departure_margins_hf_bf16_shares": shared_margins[index],
            **{name: follows[name][index] for name in recordings},
            "deviation_floor_ratio": deviation_ratios[index],
        }
        for index, request in enumerate(requests)
    ]

    if report_dir is not None:
        (report_dir / "thinker_concurrency.json").write_text(
            json.dumps(report, indent=1)
        )
    else:
        pass
    print(
        "\n"
        + "\n".join(
            f"[thinker] {scope} {name} {json.dumps(report[scope][name])}"
            for scope in ("all", "image", "audio", "video")
            for name in recordings
        )
        + f"\n[thinker] floor ratio {report['deviation_floor_ratio']:.3f}, "
        f"worst {report['deviation_floor_ratio_max']:.3f}; "
        f"omni departures HF BF16 shares {json.dumps(report['omni_departures_hf_bf16_shares'])}"
    )
    floor_margin = report["all"]["floor"]["departure_margin_max"]
    # note (ratish): FP32's cached and uncached forwards round differently: 8 of 9,278 steps depart, up to margin 0.095.
    assert (
        report["all"]["truth"]["departure_margin_max"] <= floor_margin
    ), "the reference's own teacher forcing departs as widely as its BF16 copy"
    assert (
        report["deviation_floor_ratio"] <= GEOMETRIC_MEAN_BOUND
        and report["deviation_floor_ratio_max"] <= CASE_BOUND
    ), f"omni's logprob deviation from FP32 strays past the BF16 floor's: {report['deviation_floor_ratio']}"
    assert (
        report["all"]["omni"]["departure_margin_max"]
        <= CONFIDENT_MARGIN_FACTOR * floor_margin
    ), f"omni departs where the FP32 margin passes {CONFIDENT_MARGIN_FACTOR} times {floor_margin:.3f}, HF BF16's widest"
