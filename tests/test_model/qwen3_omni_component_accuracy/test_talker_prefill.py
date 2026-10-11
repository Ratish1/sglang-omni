# SPDX-License-Identifier: Apache-2.0
"""Omni's talker prefill against the one HF's generate builds, for stage 6's requests.

HF's Qwen3OmniMoeForConditionalGeneration.generate runs in BF16 up to its talker call, which is captured instead of
run, so the reference prefill is HF's own assembly (modeling_qwen3_omni_moe.py:3981-4051). Omni's served
TalkerPrefillBuilder then builds from the same prompt ids and the same thinker tokens over the same talker
projection modules, once with every thinker token (as HF does) and once from the first 5 tokens with the rest
streamed in (partial start). Text rows, the 9-row assistant tail, the input ids and the future text rows must
match; at multimodal prompt rows omni feeds zeros where HF feeds the thinker's accept_hidden_layer hidden state,
which is measured and reported, not gated.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import librosa
import pytest
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.test_model.qwen3_omni_component_accuracy.comparison import (
    deviation,
    reference_copy,
)
from tests.test_model.qwen3_omni_component_accuracy.stage6_requests import AudioQuestion
from tests.test_model.test_qwen3_omni_mmsu_talker_ci import MAX_TOKENS

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

SAMPLE_RATE = 16000
SPEAKER = "Ethan"
PARTIAL_START_CHUNKS = 5
MATCH_RELATIVE_L2 = 1e-3


class TalkerInputsCaptured(Exception):
    pass


class TalkerProjections(torch.nn.Module):
    """The attributes TalkerPrefillBuilder reads from the talker, served by the HF talker's own modules."""

    def __init__(self, talker: torch.nn.Module) -> None:
        super().__init__()
        self.model = talker.model
        self.text_projection = talker.text_projection
        self.hidden_projection = talker.hidden_projection
        self.activation_dtype = talker.model.codec_embedding.weight.dtype

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()


def hf_talker_inputs(
    model: Qwen3OmniMoeForConditionalGeneration, inputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    thinker_generate = model.thinker.generate

    def record_thinker(*args, **kwargs):
        result = thinker_generate(*args, **kwargs)
        captured["sequences"] = result.sequences
        return result

    def capture_talker(**kwargs):
        captured.update(kwargs)
        raise TalkerInputsCaptured

    model.thinker.generate = record_thinker
    model.talker.generate = capture_talker
    try:
        with torch.inference_mode():
            model.generate(
                **inputs,
                speaker=SPEAKER,
                return_audio=True,
                thinker_max_new_tokens=MAX_TOKENS,
            )
    except TalkerInputsCaptured:
        pass
    else:
        raise AssertionError("HF generate finished without calling the talker")
    finally:
        del model.thinker.generate
        del model.talker.generate
    return captured


def rows_deviation(omni: torch.Tensor, reference: torch.Tensor) -> float:
    return deviation(omni, reference).relative_l2 if reference.numel() else 0.0


def test_talker_prefill_matches_reference(
    model_path: str,
    checkpoint_dir: str,
    audio_questions: list[AudioQuestion],
    reference_model: Qwen3OmniMoeForConditionalGeneration,
    component_device: str,
    report_dir: Path | None,
) -> None:
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    model = reference_copy(
        reference_model, device=component_device, dtype=torch.bfloat16
    )
    config = model.config
    builder = TalkerPrefillBuilder(
        model=TalkerProjections(model.talker),
        model_path=checkpoint_dir,
        audio_token_id=config.thinker_config.audio_token_id,
        image_token_id=config.thinker_config.image_token_id,
        video_token_id=config.thinker_config.video_token_id,
        tts_bos_token_id=config.tts_bos_token_id,
        tts_eos_token_id=config.tts_eos_token_id,
        tts_pad_token_id=config.tts_pad_token_id,
        im_start_token_id=config.im_start_token_id,
        im_end_token_id=config.im_end_token_id,
        system_token_id=config.system_token_id,
        user_token_id=config.user_token_id,
        assistant_token_id=config.assistant_token_id,
        codec_bos_id=config.talker_config.codec_bos_id,
        codec_nothink_id=config.talker_config.codec_nothink_id,
        codec_think_bos_id=config.talker_config.codec_think_bos_id,
        codec_think_eos_id=config.talker_config.codec_think_eos_id,
        codec_pad_id=config.talker_config.codec_pad_id,
        speaker_map=config.talker_config.speaker_id,
    )
    multimodal_ids = (
        config.thinker_config.audio_token_id,
        config.thinker_config.image_token_id,
        config.thinker_config.video_token_id,
    )

    records = []
    for question in audio_questions:
        waveform, _ = librosa.load(question.audio_path, sr=SAMPLE_RATE)
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
        inputs = processor(
            text=prompt_text, audio=[waveform], return_tensors="pt", padding=True
        )
        inputs = {
            name: (
                tensor.to(component_device, dtype=torch.bfloat16)
                if tensor.is_floating_point()
                else tensor.to(component_device)
            )
            for name, tensor in inputs.items()
        }
        reference = hf_talker_inputs(model, inputs)
        prompt_ids = inputs["input_ids"][0].cpu()
        generated = reference["sequences"][0, prompt_ids.shape[0] :].tolist()
        chunks = [
            StreamItem(
                chunk_id=index,
                data=None,
                from_stage="thinker",
                metadata={"token_id": token},
            )
            for index, token in enumerate(generated)
        ]
        payload = StagePayload(
            request_id=question.sample_id,
            request=OmniRequest(inputs={}, params={"speaker": SPEAKER}),
            data={"prompt": {"input_ids": prompt_ids}},
        )
        with torch.inference_mode():
            whole = builder.build_prompt_prefill(payload, chunks, thinker_done=True)
            partial = builder.build_prompt_prefill(
                payload, chunks[:PARTIAL_START_CHUNKS], thinker_done=False
            )
            streamed = SimpleNamespace(
                thinker_chunks_done=False,
                pending_text_queue=partial["pending_text_queue"],
                tts_eos_embed=partial["tts_eos_embed"],
            )
            for chunk in chunks[PARTIAL_START_CHUNKS:]:
                builder.append_text_chunk(streamed, chunk)
            builder.mark_thinker_done(streamed)

        reference_embeds = reference["inputs_embeds"][0]
        reference_ids = reference["talker_input_ids"][0].cpu()
        reference_future = reference["trailing_text_hidden"][0]
        multimodal = torch.isin(reference_ids, torch.tensor(multimodal_ids))
        tail = torch.zeros_like(multimodal)
        tail[-9:] = True
        text = ~multimodal & ~tail
        omni_embeds = whole["input_embeds"]
        records.append(
            {
                "sample_id": question.sample_id,
                "generated_tokens": len(generated),
                "rows": int(reference_ids.shape[0]),
                "multimodal_rows": int(multimodal.sum()),
                "input_ids_equal": torch.equal(whole["input_ids"].cpu(), reference_ids),
                "partial_input_ids_equal": torch.equal(
                    partial["input_ids"].cpu(), reference_ids
                ),
                "text_rows_relative_l2": rows_deviation(
                    omni_embeds[text], reference_embeds[text]
                ),
                "tail_rows_relative_l2": rows_deviation(
                    omni_embeds[tail], reference_embeds[tail]
                ),
                "partial_tail_rows_relative_l2": rows_deviation(
                    partial["input_embeds"][tail], reference_embeds[tail]
                ),
                "multimodal_rows_relative_l2": rows_deviation(
                    omni_embeds[multimodal], reference_embeds[multimodal]
                ),
                "future_rows": [
                    len(whole["pending_text_queue"]),
                    int(reference_future.shape[0]),
                ],
                "future_rows_relative_l2": rows_deviation(
                    torch.stack(list(whole["pending_text_queue"])), reference_future
                ),
                "streamed_future_rows": len(streamed.pending_text_queue),
                "streamed_future_rows_relative_l2": rows_deviation(
                    torch.stack(list(streamed.pending_text_queue)), reference_future
                ),
            }
        )

    if report_dir is not None:
        (report_dir / "talker_prefill.json").write_text(json.dumps(records, indent=1))
    else:
        pass
    print(
        "\n[talker_prefill] multimodal rows relative L2 (zeros against the thinker hidden): "
        + ", ".join(
            f"{record['multimodal_rows_relative_l2']:.3f}" for record in records
        )
    )
    failures = [
        f"{record['sample_id']}: {name} {record[name]}"
        for record in records
        for name in (
            "input_ids_equal",
            "partial_input_ids_equal",
            "text_rows_relative_l2",
            "tail_rows_relative_l2",
            "partial_tail_rows_relative_l2",
            "future_rows_relative_l2",
            "streamed_future_rows_relative_l2",
        )
        if record[name] is False
        or (not isinstance(record[name], bool) and record[name] > MATCH_RELATIVE_L2)
    ]
    assert not failures, "omni's talker prefill departs from HF's:\n" + "\n".join(
        failures
    )
