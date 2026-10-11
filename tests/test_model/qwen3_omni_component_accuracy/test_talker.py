# SPDX-License-Identifier: Apache-2.0
"""Omni's served talker, code predictor and code2wav against HF's, teacher-forced on what omni computed.

QWEN3_OMNI_ACCURACY_SERVER is a speech server started with SGLANG_OMNI_TALKER_CAPTURE_DIR set to the directory
QWEN3_OMNI_ACCURACY_TALKER_CAPTURE names. Each stage 6 request goes out with text and audio, greedy thinker, the
talker's own sampling; the capture keeps the talker's prefill and decode input rows, its final hidden state at every
step and the 16 codes of every frame. HF then runs on omni's own inputs, its FP32 copy as truth, its BF16 copy as the
floor:
- decode inputs: every decode row must be the codec embeddings of the previous frame's 16 codes plus the next text
  row (the projected thinker tokens after the first, then tts_eos, then tts_pad), built with HF's modules;
- talker: HF's talker over omni's prefill and decode rows; omni's hidden at each step and the first-code logits
  codec_head makes from it must stay within the floor;
- code predictor: HF's predictor over omni's talker hidden and codes; omni takes each residual codebook's argmax, so it
  may depart from HF FP32's argmax only where FP32's margin is within the largest margin HF BF16 departs at;
- code2wav: the audio omni served for its own codes against HF FP32 decoding the same codes in the same windows
  (floor: HF BF16 the same way), and that window walk against HF's whole-sequence decode, a reported semantic gap.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path

import httpx
import pytest
import soundfile
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    create_code2wav_scheduler,
)
from tests.test_model.qwen3_omni_component_accuracy.comparison import (
    CaseRecord,
    deviation,
    floor_violations,
    geometric_mean_ratio,
    reference_copy,
)
from tests.test_model.qwen3_omni_component_accuracy.stage6_requests import AudioQuestion
from tests.test_model.qwen3_omni_component_accuracy.test_code2wav import (
    best_alignment,
    window_walk,
)
from tests.test_model.test_qwen3_omni_mmsu_talker_ci import MAX_TOKENS

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

REQUEST_TIMEOUT_S = 900
CAPTURE_WAIT_S = 120


def generate_with_speech(
    server: httpx.Client, question: AudioQuestion
) -> tuple[list[int], torch.Tensor]:
    response = server.post(
        "/generate",
        json={
            "messages": [{"role": "user", "content": question.question_text}],
            "metadata": {
                "audios": [question.audio_path],
                "sample_id": question.sample_id,
            },
            "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_TOKENS},
            "output_modalities": ["text", "audio"],
            "return_logprob": True,
        },
    )
    response.raise_for_status()
    body = response.json()
    tokens = [int(pair[1]) for pair in body["meta_info"]["output_token_logprobs"]]
    audio, _ = soundfile.read(
        io.BytesIO(base64.b64decode(body["audio"]["data"])), dtype="float32"
    )
    return tokens, torch.from_numpy(audio)


def captured_records(directory: Path, sample_ids: set[str]) -> dict[str, dict]:
    deadline = time.monotonic() + CAPTURE_WAIT_S
    records: dict[str, dict] = {}
    while time.monotonic() < deadline:
        records = {}
        for path in directory.glob("*.pt"):
            record = torch.load(path, map_location="cpu")
            sample_id = record["metadata"].get("sample_id")
            if sample_id in sample_ids:
                records[sample_id] = record
            else:
                pass
        if set(records) == sample_ids:
            break
        else:
            time.sleep(2)
    return records


def test_talker_follows_reference(
    audio_questions: list[AudioQuestion],
    checkpoint_dir: str,
    reference_model: Qwen3OmniMoeForConditionalGeneration,
    component_device: str,
    report_dir: Path | None,
) -> None:
    server_url = os.environ.get("QWEN3_OMNI_ACCURACY_SERVER")
    capture_dir = os.environ.get("QWEN3_OMNI_ACCURACY_TALKER_CAPTURE")
    if server_url is None or capture_dir is None:
        pytest.skip(
            "Set QWEN3_OMNI_ACCURACY_SERVER and QWEN3_OMNI_ACCURACY_TALKER_CAPTURE to a speech server "
            "capturing its talker"
        )
    else:
        pass
    responses = {}
    with httpx.Client(
        base_url=server_url, timeout=REQUEST_TIMEOUT_S, trust_env=False
    ) as server:
        for question in audio_questions:
            responses[question.sample_id] = generate_with_speech(server, question)
    records = captured_records(Path(capture_dir), set(responses))
    assert set(records) == set(
        responses
    ), f"no capture for {sorted(set(responses) - set(records))}"

    config = reference_model.config
    truth = reference_copy(
        reference_model.talker, device=component_device, dtype=torch.float32
    )
    floor = reference_copy(
        reference_model.talker, device=component_device, dtype=torch.bfloat16
    )
    truth_code2wav = reference_copy(
        reference_model.code2wav, device=component_device, dtype=torch.float32
    )
    floor_code2wav = reference_copy(
        reference_model.code2wav, device=component_device, dtype=torch.bfloat16
    )
    thinker_embedding = reference_model.thinker.get_input_embeddings()
    code2wav_defaults = create_code2wav_scheduler.__kwdefaults__
    chunk = code2wav_defaults["stream_chunk_size"]
    context = code2wav_defaults["left_context_size"]
    upsample = truth_code2wav.total_upsample
    codec_eos = config.talker_config.codec_eos_token_id
    special_ids = torch.tensor(
        [config.tts_eos_token_id, config.tts_pad_token_id], dtype=torch.long
    )

    def codec_feedback(talker: torch.nn.Module, codes: torch.Tensor) -> torch.Tensor:
        """HF's next talker input from a frame's codes (modeling_qwen3_omni_moe.py:3247-3258)."""
        rows = talker.model.get_input_embeddings()(codes[:, 0])
        for group, embedding in enumerate(talker.code_predictor.get_input_embeddings()):
            rows = rows + embedding(codes[:, group + 1])
        return rows

    def predictor_logits(
        talker: torch.nn.Module, hidden: torch.Tensor, codes: torch.Tensor
    ) -> torch.Tensor:
        """HF's code predictor teacher-forced on one frame per row: [frames, groups - 1, codebook]."""
        embeddings = talker.code_predictor.get_input_embeddings()
        inputs = torch.stack(
            [hidden, talker.model.get_input_embeddings()(codes[:, 0])]
            + [
                embeddings[group](codes[:, group + 1])
                for group in range(len(embeddings) - 1)
            ],
            dim=1,
        )
        states = talker.code_predictor.model(inputs_embeds=inputs).last_hidden_state
        return torch.stack(
            [
                head(states[:, group + 1])
                for group, head in enumerate(talker.code_predictor.lm_head)
            ],
            dim=1,
        ).float()

    case_records: list[CaseRecord] = []
    reports = []
    with torch.inference_mode():
        for sample_id, (tokens, served_audio) in responses.items():
            record = records[sample_id]
            generated = [token for token in tokens if token != config.im_end_token_id]
            codes = record["codes"].to(component_device)
            hidden = record["hidden"].to(component_device)
            prefill = record["prefill_input_embeds"].to(component_device)
            decode = record["decode_input_embeds"].to(component_device)
            frames = codes.shape[0]
            assert (
                decode.shape[0] == frames - 1
            ), f"{sample_id}: {decode.shape[0]} decode rows for {frames} frames"

            text_rows = {}
            for name, talker, dtype in (
                ("truth", truth, torch.float32),
                ("floor", floor, torch.bfloat16),
            ):
                rows = talker.text_projection(
                    thinker_embedding(torch.tensor(generated[1:], dtype=torch.long)).to(
                        component_device, dtype
                    )
                )
                eos, pad = talker.text_projection(
                    thinker_embedding(special_ids).to(component_device, dtype)
                )
                padding = pad.expand(max(decode.shape[0] - rows.shape[0] - 1, 0), -1)
                text_rows[name] = torch.cat([rows, eos[None], padding])[
                    : decode.shape[0]
                ]
            expected = {
                name: codec_feedback(talker, codes[: decode.shape[0]]) + text_rows[name]
                for name, talker in (("truth", truth), ("floor", floor))
            }
            case_records.append(
                CaseRecord(
                    case=f"{sample_id}/decode_inputs",
                    omni=deviation(decode, expected["truth"]),
                    floor=deviation(expected["floor"], expected["truth"]),
                )
            )

            sequence = torch.cat([prefill, decode])
            steps = torch.arange(
                prefill.shape[0] - 1,
                prefill.shape[0] - 1 + frames,
                device=component_device,
            )
            truth_hidden = truth.model(
                inputs_embeds=sequence[None].float()
            ).last_hidden_state[0, steps]
            floor_hidden = floor.model(
                inputs_embeds=sequence[None].bfloat16()
            ).last_hidden_state[0, steps]
            case_records.append(
                CaseRecord(
                    case=f"{sample_id}/talker_hidden",
                    omni=deviation(hidden, truth_hidden),
                    floor=deviation(floor_hidden, truth_hidden),
                )
            )
            truth_logits = truth.codec_head(truth_hidden)
            omni_logits = truth.codec_head(hidden.float())
            floor_logits = truth.codec_head(floor_hidden.float())
            case_records.append(
                CaseRecord(
                    case=f"{sample_id}/code0_logits",
                    omni=deviation(omni_logits, truth_logits),
                    floor=deviation(floor_logits, truth_logits),
                )
            )

            truth_predicted = predictor_logits(truth, hidden.float(), codes)
            floor_predicted = predictor_logits(floor, hidden.bfloat16(), codes)
            top2 = truth_predicted.topk(2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
            truth_choice = truth_predicted.argmax(dim=-1)
            omni_departures = truth_choice != codes[:, 1:]
            floor_departures = truth_choice != floor_predicted.argmax(dim=-1)

            spoken = codes[codes[:, 0] != codec_eos].T[None]
            truth_walk = window_walk(truth_code2wav, spoken, chunk, context, upsample)
            floor_walk = window_walk(floor_code2wav, spoken, chunk, context, upsample)
            served = served_audio.to(component_device)
            shared = min(served.numel(), truth_walk.numel())
            case_records.append(
                CaseRecord(
                    case=f"{sample_id}/served_audio",
                    omni=deviation(served[:shared], truth_walk.flatten()[:shared]),
                    floor=deviation(
                        floor_walk.flatten()[:shared], truth_walk.flatten()[:shared]
                    ),
                )
            )
            whole = truth_code2wav(spoken)
            trim = spoken.shape[-1] * upsample - whole.shape[-1]
            reports.append(
                {
                    "sample_id": sample_id,
                    "frames": frames,
                    "spoken_frames": int(spoken.shape[-1]),
                    "served_samples": int(served.numel()),
                    "walk_samples": int(truth_walk.numel()),
                    "code0_top1_agreement": (
                        omni_logits.argmax(-1) == truth_logits.argmax(-1)
                    )
                    .float()
                    .mean()
                    .item(),
                    "residual_departures": int(omni_departures.sum()),
                    "residual_codes": int(omni_departures.numel()),
                    "residual_departure_margins": sorted(
                        margin[omni_departures].tolist()
                    )[-10:],
                    "floor_residual_departures": int(floor_departures.sum()),
                    "floor_departure_margin_max": (
                        margin[floor_departures].max().item()
                        if floor_departures.any()
                        else 0.0
                    ),
                    "walk_vs_whole_relative_l2": best_alignment(
                        truth_walk, whole, expected=0, radius=trim
                    )[1],
                }
            )

    if report_dir is not None:
        (report_dir / "talker.json").write_text(
            json.dumps(
                {
                    "records": [record.as_json() for record in case_records],
                    "requests": reports,
                },
                indent=1,
            )
        )
    else:
        pass
    floor_margin = max(report["floor_departure_margin_max"] for report in reports)
    confident = [
        f"{report['sample_id']}: residual code departs at FP32 margin {margin_value:.3f} "
        f"beyond HF BF16's largest {floor_margin:.3f}"
        for report in reports
        for margin_value in report["residual_departure_margins"]
        if margin_value > floor_margin
    ]
    print(
        f"\n[talker] floor ratio geometric mean {geometric_mean_ratio(case_records):.3f}; residual departures "
        f"omni {sum(r['residual_departures'] for r in reports)} HF BF16 "
        f"{sum(r['floor_residual_departures'] for r in reports)} of {sum(r['residual_codes'] for r in reports)}; "
        f"code0 top-1 agreement {sum(r['code0_top1_agreement'] for r in reports) / len(reports):.4f}; "
        f"walk against whole decode relative L2 "
        f"{sorted(round(r['walk_vs_whole_relative_l2'], 4) for r in reports)}"
    )
    violations = floor_violations(case_records) + confident
    assert not violations, "talker path departs from HF:\n" + "\n".join(violations)
