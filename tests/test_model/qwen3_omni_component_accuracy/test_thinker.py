# SPDX-License-Identifier: Apache-2.0
"""The served Qwen3-Omni thinker against the HF thinker, teacher-forced on omni's own greedy tokens.

QWEN3_OMNI_ACCURACY_SERVER is the base URL of a running omni server in the configuration under test;
QWEN3_OMNI_ACCURACY_REFERENCE_DEVICE is the card the HF thinker runs on. Arm A sends each request through omni's
preprocessing; arm B sends HF's processor outputs, so the thinker sees identical inputs on both sides and any
confident departure from the reference is the thinker's.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx
import librosa
import pytest
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from tests.test_model.qwen3_omni_component_accuracy.stage6_requests import AudioQuestion
from tests.test_model.test_qwen3_omni_mmsu_talker_ci import MAX_TOKENS

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

REFERENCE_SAMPLE_RATE = 16000
REQUEST_TIMEOUT_S = 600


def serialized_tensor(tensor: torch.Tensor) -> dict[str, str | list[int]]:
    contiguous = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(contiguous.dtype).removeprefix("torch."),
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.numpy().tobytes()).decode("ascii"),
    }


def greedy_with_logprobs(
    server: httpx.Client, body: dict[str, object]
) -> tuple[list[int], list[float]]:
    response = server.post(
        "/generate",
        json={
            **body,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_TOKENS},
            "output_modalities": ["text"],
            "return_logprob": True,
        },
    )
    response.raise_for_status()
    pairs = response.json()["meta_info"]["output_token_logprobs"]
    return [int(pair[1]) for pair in pairs], [float(pair[0]) for pair in pairs]


def teacher_forced_logits(
    thinker: torch.nn.Module,
    prompt_ids: torch.Tensor,
    generated_ids: list[int],
    reference_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Reference logits at every step omni generated, as float32 [steps, vocab]."""
    full_ids = torch.cat([prompt_ids, torch.tensor(generated_ids, dtype=torch.long)])
    device = thinker.device
    with torch.inference_mode():
        output = thinker(
            input_ids=full_ids[None].to(device),
            attention_mask=torch.ones_like(full_ids)[None].to(device),
            input_features=reference_inputs["input_features"].to(
                device, dtype=thinker.dtype
            ),
            feature_attention_mask=reference_inputs["feature_attention_mask"].to(
                device
            ),
            use_cache=False,
        )
    start = prompt_ids.shape[0] - 1
    return output.logits[0, start : start + len(generated_ids)].float().cpu()


def test_thinker_follows_reference(
    model_path: str,
    audio_questions: list[AudioQuestion],
    report_dir: Path | None,
) -> None:
    server_url = os.environ.get("QWEN3_OMNI_ACCURACY_SERVER")
    if server_url is None:
        pytest.skip("Set QWEN3_OMNI_ACCURACY_SERVER to the omni server under test")
    else:
        pass
    reference_device = os.environ.get("QWEN3_OMNI_ACCURACY_REFERENCE_DEVICE", "cuda:0")
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": reference_device}
    )
    thinker = model.thinker.eval()
    records = []
    with httpx.Client(
        base_url=server_url, timeout=REQUEST_TIMEOUT_S, trust_env=False
    ) as server:
        for question in audio_questions:
            waveform, _ = librosa.load(question.audio_path, sr=REFERENCE_SAMPLE_RATE)
            prompt_text = processor.apply_chat_template(
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
            reference_inputs = processor(
                text=prompt_text, audio=[waveform], return_tensors="pt", padding=True
            )
            prompt_ids = reference_inputs["input_ids"][0]
            omni_preprocessed_ids, _ = greedy_with_logprobs(
                server,
                {
                    "messages": [{"role": "user", "content": question.question_text}],
                    "metadata": {"audios": [question.audio_path]},
                },
            )
            generated_ids, omni_logprobs = greedy_with_logprobs(
                server,
                {
                    "input_ids": prompt_ids.tolist(),
                    "multimodal_train_inputs": {
                        "version": 1,
                        "tensors": {
                            name: serialized_tensor(reference_inputs[name])
                            for name in ("input_features", "feature_attention_mask")
                        },
                    },
                },
            )
            logits = teacher_forced_logits(
                thinker, prompt_ids, generated_ids, reference_inputs
            )
            reference_logprobs = torch.log_softmax(logits, dim=-1)
            omni_tokens = torch.tensor(generated_ids)
            steps = torch.arange(len(generated_ids))
            reference_choice = logits.argmax(dim=-1)
            preference_gap = (
                logits[steps, reference_choice] - logits[steps, omni_tokens]
            )
            departures = (reference_choice != omni_tokens).nonzero().flatten().tolist()
            records.append(
                {
                    "sample_id": question.sample_id,
                    "generated_tokens": len(generated_ids),
                    "omni_preprocessing_same_tokens": omni_preprocessed_ids
                    == generated_ids,
                    "logprob_abs_delta_mean": (
                        reference_logprobs[steps, omni_tokens]
                        - torch.tensor(omni_logprobs)
                    )
                    .abs()
                    .mean()
                    .item(),
                    "departure_steps": departures,
                    "departure_gaps": [
                        preference_gap[step].item() for step in departures
                    ],
                    "reference_top2_margin_at_departures": [
                        (
                            logits[step].topk(2).values[0]
                            - logits[step].topk(2).values[1]
                        ).item()
                        for step in departures
                    ],
                }
            )
    if report_dir is not None:
        (report_dir / "thinker.json").write_text(json.dumps(records, indent=1))
    else:
        pass
    print(
        "\n[thinker] departures from the reference argmax per request: "
        + ", ".join(str(len(record["departure_steps"])) for record in records)
    )
    assert all(record["generated_tokens"] > 0 for record in records)
