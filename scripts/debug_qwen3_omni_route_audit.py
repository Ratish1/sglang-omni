# SPDX-License-Identifier: Apache-2.0
"""Enumerate model routing contracts without claiming model execution coverage."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeConfig

from sglang_omni.models.qwen3_omni.config import (
    COLOCATED_SPEECH_PROCESSES,
    qwen_session_stages,
    speech_stages,
    text_stages,
)
from sglang_omni.models.qwen3_omni.payload_types import (
    EncoderInputs,
    Qwen3OmniPipelineState,
)
from sglang_omni.models.qwen3_omni.request_builders import (
    resolve_encoder_next_stages,
    resolve_mm_aggregate_wait_sources,
    resolve_preprocessing_next_stages,
    resolve_preprocessing_next_stages_speech,
    resolve_terminal_stages,
    resolve_thinker_stream_done_targets,
)
from sglang_omni.proto.request import OmniRequest, StagePayload

INPUT_MODALITIES = ("text", "audio", "image", "video")
OUTPUT_MODALITIES = (None, ("text",), ("audio",), ("text", "audio"))


@dataclass(kw_only=True)
class RouteCase:
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...] | None
    stream: bool
    video_audio_extracted: bool
    speech_preprocessing_targets: list[str]
    text_preprocessing_targets: list[str]
    join_sources: list[str]
    encoder_targets: str | list[str]
    stream_done_targets: list[str]
    terminal_stages: list[str]
    coverage: str = "post-preprocessing routing only"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    else:
        pass
    cases: list[RouteCase] = []
    for modality_count in range(1, len(INPUT_MODALITIES) + 1):
        for input_modalities in combinations(INPUT_MODALITIES, modality_count):
            extracted_audio_options = (
                (False, True) if "video" in input_modalities else (False,)
            )
            for output_modalities, stream, video_audio_extracted in product(
                OUTPUT_MODALITIES, (False, True), extracted_audio_options
            ):
                encoder_inputs: dict[str, EncoderInputs] = {}
                vision_inputs: EncoderInputs = {}
                if "image" in input_modalities:
                    vision_inputs["pixel_values"] = torch.ones(1)
                else:
                    pass
                if "video" in input_modalities:
                    vision_inputs["pixel_values_videos"] = torch.ones(1)
                else:
                    pass
                if vision_inputs:
                    encoder_inputs["image_encoder"] = vision_inputs
                else:
                    pass
                if "audio" in input_modalities or video_audio_extracted:
                    encoder_inputs["audio_encoder"] = {"input_features": torch.ones(1)}
                else:
                    pass
                request = OmniRequest(
                    inputs=None,
                    params={"stream": stream},
                    metadata=(
                        {"output_modalities": output_modalities}
                        if output_modalities is not None
                        else {}
                    ),
                )
                payload = StagePayload(
                    request_id="route-audit",
                    request=request,
                    data=Qwen3OmniPipelineState(
                        encoder_inputs=encoder_inputs
                    ).to_dict(),
                )
                speech_targets = resolve_preprocessing_next_stages_speech(
                    payload.request_id, payload
                )
                text_targets = resolve_preprocessing_next_stages(
                    payload.request_id, payload
                )
                join_sources = resolve_mm_aggregate_wait_sources(
                    payload.request_id, "preprocessing", payload
                )
                assert join_sources is not None
                terminals = resolve_terminal_stages(request)
                encoder_targets = resolve_encoder_next_stages(
                    payload.request_id, payload
                )
                stream_done_targets = resolve_thinker_stream_done_targets(
                    payload.request_id, payload
                )
                expected_audio_output = (
                    output_modalities is None or "audio" in output_modalities
                )
                assert ("talker_ar" in speech_targets) == expected_audio_output
                assert ("code2wav" in terminals) == expected_audio_output
                assert "decode" in terminals
                assert set(join_sources) == {"preprocessing", *encoder_inputs}
                assert set(text_targets) == {"mm_aggregate", *encoder_inputs}
                assert set(speech_targets) - {"thinker", "talker_ar"} == set(
                    encoder_inputs
                )
                cases.append(
                    RouteCase(
                        input_modalities=input_modalities,
                        output_modalities=output_modalities,
                        stream=stream,
                        video_audio_extracted=video_audio_extracted,
                        speech_preprocessing_targets=speech_targets,
                        text_preprocessing_targets=text_targets,
                        join_sources=join_sources,
                        encoder_targets=encoder_targets,
                        stream_done_targets=stream_done_targets,
                        terminal_stages=terminals,
                    )
                )
    stage_graphs = {}
    for name, stages in (
        ("text", text_stages()),
        (
            "speech_colocated",
            speech_stages(
                thinker_gpu=0,
                talker_gpu=0,
                process_by_stage=COLOCATED_SPEECH_PROCESSES,
                enable_partial_start=False,
            ),
        ),
        ("session_model_subgraph", qwen_session_stages()),
    ):
        stage_graphs[name] = [
            {
                "name": stage.name,
                "process": stage.process,
                "next": stage.next,
                "stream_to": stage.stream_to,
                "wait_for": stage.wait_for,
                "merge_fn": stage.merge_fn,
                "terminal": stage.terminal,
            }
            for stage in stages
        ]
    snapshot = Path(snapshot_download(arguments.model_path, local_files_only=True))
    config = Qwen3OmniMoeConfig.from_pretrained(snapshot, local_files_only=True)
    source = Path(inspect.getfile(resolve_preprocessing_next_stages_speech))
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_path": arguments.model_path,
        "checkpoint_revision": snapshot.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Real route and join functions over explicit post-preprocessing encoder-presence states. No media decode, encoder, Thinker, Talker, lifecycle or task-quality execution is implied.",
        "video_audio_condition": "True represents successful extraction. Silent videos and mixed valid/missing tracks are separate preprocessing contracts, not an assumed audio branch.",
        "model_dimensions": {
            "talker_layers": config.talker_config.text_config.num_hidden_layers,
            "talker_hidden_size": config.talker_config.text_config.hidden_size,
            "talker_experts": config.talker_config.text_config.num_experts,
            "talker_selected_experts": config.talker_config.text_config.num_experts_per_tok,
            "codec_groups": config.talker_config.num_code_groups,
            "predictor_layers": config.talker_config.code_predictor_config.num_hidden_layers,
        },
        "stage_graphs": stage_graphs,
        "cases": [asdict(case) for case in cases],
        "case_count": len(cases),
        "unexecuted_boundaries": [
            "all media decoding, placeholder ordering and video audio extraction",
            "image/video encoder and deepstack numerical parity",
            "audio encoder segments, packing and graph numerical parity",
            "Thinker multimodal positions, hidden-state alignment and output quality",
            "Talker numerical parity outside the separately recorded traces",
            "full free-generation quality of the cast-only candidate",
            "realtime turn retention, cancellation, completion and resource reclamation",
            "multi-rank, mixed-batch, quantized and H100 performance qualification",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "output": str(arguments.output),
                "model_dimensions": report["model_dimensions"],
            }
        )
    )


if __name__ == "__main__":
    main()
