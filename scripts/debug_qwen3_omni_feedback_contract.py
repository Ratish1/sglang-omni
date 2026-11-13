"""Audit actual speech feedback and terminal records against their published codes."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)


class NativeFinishReason(BaseModel):
    type: Literal["stop", "length", "abort"]
    matched: int | str | list[int] | None = None
    length: int | None = None


class FeedbackObservation(BaseModel):
    event: Literal["feedback"]
    request_id: str
    committed_output_token_count: int
    codes: list[int]
    maximum_absolute_error: float
    changed_coordinates: int


class TerminalObservation(BaseModel):
    event: Literal["terminal"]
    request_id: str
    finish_reason: NativeFinishReason
    requested_frame_limit: int
    effective_frame_limit: int
    prompt_token_count: int
    output_codes: list[int]
    thinker_done: bool
    pending_text_rows: int


SpeechContractObservation = Annotated[
    FeedbackObservation | TerminalObservation, Field(discriminator="event")
]


class SpeechIdentity(BaseModel):
    request_id: str
    sample_id: str


class GenerationSummary(BaseModel):
    total_requests: int
    completed_requests: int
    failed_requests: int
    max_token_hits: int


class GenerationResults(BaseModel):
    summary: GenerationSummary


@dataclass(kw_only=True)
class RequestContract:
    request_id: str
    sample_id: str
    generation_directory: str
    observed_frames: int
    mismatched_feedback_frames: int
    maximum_absolute_error: float
    changed_coordinates: int
    main_codes_match: bool
    committed_output_counts_match: bool
    requested_frame_limit: int
    effective_frame_limit: int
    prompt_token_count: int
    finish_type: Literal["stop", "length", "abort"]
    matched_stop: int | str | list[int] | None
    matched_eos: bool
    output_codes: list[int]
    thinker_done: bool
    pending_text_rows: int


@dataclass(kw_only=True)
class ContractReport:
    model_path: str
    model_revision: str
    code_group_count: int
    codec_eos_id: int
    requests: int
    frames: int
    mismatched_feedback_frames: int
    maximum_absolute_error: float
    mismatched_main_code_histories: int
    mismatched_committed_output_counts: int
    eos_finishes: int
    length_finishes: int
    requests_with_pending_text: int
    excluded_request_ids: list[str]
    per_request: list[RequestContract]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument(
        "--generation-directory", type=Path, action="append", required=True
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        local_files_only=True,
    ).talker_config
    identities: dict[str, SpeechIdentity] = {}
    directories: dict[str, str] = {}
    for directory in arguments.generation_directory:
        generation = GenerationResults.model_validate_json(
            (directory / "results.json").read_bytes()
        ).summary
        assert generation.total_requests == generation.completed_requests > 0
        assert generation.failed_requests == generation.max_token_hits == 0
        speeches = list(directory.glob("*.speech.json"))
        assert len(speeches) == generation.completed_requests
        for source in speeches:
            identity = SpeechIdentity.model_validate_json(source.read_bytes())
            assert identity.request_id not in identities
            assert source.name == f"{identity.request_id}.speech.json"
            identities[identity.request_id] = identity
            directories[identity.request_id] = str(directory)
    observations = TypeAdapter(SpeechContractObservation)
    frames: defaultdict[str, list[FeedbackObservation]] = defaultdict(list)
    terminals: dict[str, TerminalObservation] = {}
    excluded_request_ids: set[str] = set()
    with arguments.frames_output.open("xb") as destination:
        with gzip.GzipFile(fileobj=destination, mode="wb") as compressed:
            for line in arguments.server_log.read_text().splitlines():
                _, marker, payload = line.partition("speech_contract ")
                if not marker:
                    continue
                else:
                    observation = observations.validate_json(payload)
                if observation.request_id not in identities:
                    excluded_request_ids.add(observation.request_id)
                    continue
                elif isinstance(observation, FeedbackObservation):
                    assert len(observation.codes) == configuration.num_code_groups
                    assert math.isfinite(observation.maximum_absolute_error)
                    frames[observation.request_id].append(observation)
                    compressed.write(f"{observation.model_dump_json()}\n".encode())
                else:
                    assert observation.request_id not in terminals
                    terminals[observation.request_id] = observation
    assert identities.keys() == frames.keys() == terminals.keys()
    requests = []
    for request_id, identity in identities.items():
        recorded = frames[request_id]
        terminal = terminals[request_id]
        requests.append(
            RequestContract(
                request_id=request_id,
                sample_id=identity.sample_id,
                generation_directory=directories[request_id],
                observed_frames=len(recorded),
                mismatched_feedback_frames=sum(
                    frame.changed_coordinates > 0 for frame in recorded
                ),
                maximum_absolute_error=max(
                    frame.maximum_absolute_error for frame in recorded
                ),
                changed_coordinates=sum(
                    frame.changed_coordinates for frame in recorded
                ),
                main_codes_match=[frame.codes[0] for frame in recorded]
                == terminal.output_codes,
                committed_output_counts_match=[
                    frame.committed_output_token_count for frame in recorded
                ]
                == list(range(len(recorded))),
                requested_frame_limit=terminal.requested_frame_limit,
                effective_frame_limit=terminal.effective_frame_limit,
                prompt_token_count=terminal.prompt_token_count,
                finish_type=terminal.finish_reason.type,
                matched_stop=terminal.finish_reason.matched,
                matched_eos=terminal.finish_reason.type == "stop"
                and terminal.finish_reason.matched == configuration.codec_eos_token_id,
                output_codes=terminal.output_codes,
                thinker_done=terminal.thinker_done,
                pending_text_rows=terminal.pending_text_rows,
            )
        )
    report = ContractReport(
        model_path=arguments.model_path,
        model_revision=arguments.model_revision,
        code_group_count=configuration.num_code_groups,
        codec_eos_id=configuration.codec_eos_token_id,
        requests=len(requests),
        frames=sum(request.observed_frames for request in requests),
        mismatched_feedback_frames=sum(
            request.mismatched_feedback_frames for request in requests
        ),
        maximum_absolute_error=max(
            request.maximum_absolute_error for request in requests
        ),
        mismatched_main_code_histories=sum(
            not request.main_codes_match for request in requests
        ),
        mismatched_committed_output_counts=sum(
            not request.committed_output_counts_match for request in requests
        ),
        eos_finishes=sum(request.matched_eos for request in requests),
        length_finishes=sum(request.finish_type == "length" for request in requests),
        requests_with_pending_text=sum(
            request.pending_text_rows > 0 for request in requests
        ),
        excluded_request_ids=sorted(excluded_request_ids),
        per_request=requests,
    )
    with arguments.output.open("x") as destination:
        json.dump(asdict(report), destination, indent=2)
    print(
        json.dumps(
            {
                name: value
                for name, value in asdict(report).items()
                if name != "per_request"
            }
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
