"""Check request presence and the bootstrap gate through real lowering paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import torch

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.request_builders import build_lm_request
from sglang_omni.models.qwen3_tts.request_builders import build_qwen3_tts_state
from sglang_omni.proto.request import StagePayload
from sglang_omni.serve.openai_api import (
    build_chat_generate_request,
    build_rollout_generate_request,
)
from sglang_omni.serve.protocol import (
    ChatCompletionRequest,
    CreateSpeechRequest,
    RolloutGenerateRequest,
)
from sglang_omni.serve.speech_service import SpeechRequestValidator


def check_presence() -> None:
    supplied = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "seed": 0,
        "max_new_tokens": 1,
    }
    for source in ("client", "chat", "rollout"):
        for scenario in ("omitted", "null", "supplied"):
            if scenario == "supplied":
                fields = supplied
            elif scenario == "null":
                fields = {name: None for name in supplied}
            else:
                fields = {}
            if source == "client":
                generation_request = GenerateRequest(
                    prompt="hello",
                    sampling=SamplingParams(**fields),
                    stage_sampling={"lm": SamplingParams(**fields)},
                )
            elif source == "chat":
                chat_fields = {
                    "max_tokens" if name == "max_new_tokens" else name: value
                    for name, value in fields.items()
                }
                generation_request = build_chat_generate_request(
                    ChatCompletionRequest(
                        messages=[{"role": "user", "content": "hello"}],
                        stage_sampling={"lm": fields},
                        **chat_fields,
                    )
                )
            else:
                generation_request = build_rollout_generate_request(
                    RolloutGenerateRequest(
                        prompt="hello",
                        sampling_params=fields,
                        stage_sampling={"lm": fields},
                    )
                )
            lowered = Client.build_omni_request(generation_request)
            for scope in (lowered.params, lowered.params["stage_sampling"]["lm"]):
                selected = {name: scope[name] for name in supplied if name in scope}
                assert selected == (supplied if scenario == "supplied" else {})
            print(json.dumps({"source": source, "scenario": scenario, "passed": True}))


def check_bootstrap_gate() -> None:
    service = SpeechRequestValidator(default_model="qwen3-tts")
    for field_name, supplied_value in (
        ("temperature", 0.0),
        ("top_p", 0.6),
        ("top_k", 25),
        ("repetition_penalty", 1.2),
    ):
        observed = []
        for scope in ("top_level", "tts_engine"):
            fields = {field_name: supplied_value}
            request_fields = (
                fields
                if scope == "top_level"
                else {"stage_params": {"tts_engine": fields}}
            )
            generation_request = service.build_generate_request(
                CreateSpeechRequest(
                    input="Please read this sentence.",
                    voice="Ryan",
                    task_type="CustomVoice",
                    language="English",
                    response_format="pcm",
                    stream=True,
                    **request_fields,
                ),
                validate=False,
            )
            state = build_qwen3_tts_state(
                StagePayload(
                    request_id=f"{scope}-{field_name}",
                    request=Client.build_omni_request(generation_request),
                    data={},
                )
            )
            assert state.generation_kwargs[field_name] == supplied_value
            observed.append(state.suppress_bootstrap_silence)
            print(
                json.dumps(
                    {
                        "field": field_name,
                        "scope": scope,
                        "generation_kwargs": state.generation_kwargs,
                        "suppress_bootstrap_silence": state.suppress_bootstrap_silence,
                    }
                )
            )
        assert observed == [False, False]


def check_bootstrap_boundaries() -> None:
    service = SpeechRequestValidator(default_model="qwen3-tts")
    cases: tuple[tuple[dict[str, object], dict[str, object], bool], ...] = (
        ({}, {"max_new_tokens": 2048}, True),
        (
            {
                "temperature": None,
                "stage_params": {"tts_engine": {"temperature": None}},
            },
            {"max_new_tokens": 2048},
            True,
        ),
        ({"max_new_tokens": 321, "seed": 0}, {"max_new_tokens": 321}, True),
        (
            {"stage_params": {"tts_engine": {"max_new_tokens": 321}}},
            {"max_new_tokens": 321},
            True,
        ),
        (
            {"temperature": 0.0, "stage_params": {"tts_engine": {"temperature": None}}},
            {"max_new_tokens": 2048, "temperature": 0.0},
            False,
        ),
        (
            {"temperature": 0.8, "stage_params": {"tts_engine": {"temperature": 0.0}}},
            {"max_new_tokens": 2048, "temperature": 0.0},
            False,
        ),
        (
            {"stage_params": {"tts_engine": {"do_sample": False}}},
            {"max_new_tokens": 2048, "do_sample": False},
            False,
        ),
        (
            {"stage_params": {"tts_engine": {"subtalker_dosample": False}}},
            {"max_new_tokens": 2048, "subtalker_dosample": False},
            False,
        ),
    )
    for fields, expected_generation, expected_suppression in cases:
        generation_request = service.build_generate_request(
            CreateSpeechRequest(
                input="Please read this sentence.",
                task_type="CustomVoice",
                voice="Ryan",
                language="English",
                stream=True,
                **fields,
            ),
            validate=False,
        )
        state = build_qwen3_tts_state(
            StagePayload(
                request_id="boundary",
                request=Client.build_omni_request(generation_request),
                data={},
            )
        )
        assert state.generation_kwargs == expected_generation
        assert state.suppress_bootstrap_silence == expected_suppression
    print(
        json.dumps(
            {"check": "bootstrap_boundaries", "cases": len(cases), "passed": True}
        )
    )


def check_personaplex_precedence() -> None:
    cases: tuple[tuple[float | None, float | None, float | None, float], ...] = (
        (None, None, None, 0.7),
        (0.4, None, None, 0.4),
        (0.4, 0.6, None, 0.6),
        (0.4, 0.6, 0.3, 0.3),
        (0.4, 0.6, 0.0, 0.0),
    )
    for top_value, stage_override, stage_value, expected_temperature in cases:
        generation_request = build_chat_generate_request(
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}],
                temperature=top_value,
                stage_params={"lm": {"temperature": stage_override}},
                stage_sampling={"lm": {"temperature": stage_value, "seed": 0}},
            )
        )
        state = PersonaPlexState(
            text_prompt_ids=[11, 12, 13], user_codes=torch.zeros(2, 8, dtype=torch.long)
        )
        data = build_lm_request(
            StagePayload(
                request_id="precedence",
                request=Client.build_omni_request(generation_request),
                data=state.to_dict(),
            ),
            vocab_size=32000,
        )
        selected = data.talker_model_inputs["sampling"]
        assert selected.text_temperature == expected_temperature
        assert selected.seed == 0
        if expected_temperature == 0.0:
            assert data.req.sampling_params.top_k == 1
        else:
            assert data.req.sampling_params.temperature == expected_temperature
    print(
        json.dumps(
            {"check": "personaplex_precedence", "cases": len(cases), "passed": True}
        )
    )


def check_streams(port: int, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, text in enumerate(
        (
            "The quick brown fox jumps over the lazy dog.",
            "Welcome to the library. The books are on the second floor.",
            "Please turn left at the next street.",
        )
    ):
        recorded = {}
        for variant, controls in (
            ("top_level", {"temperature": 0.0}),
            ("stage", {"stage_params": {"tts_engine": {"temperature": 0.0}}}),
            (
                "stage_no_trim",
                {
                    "stage_params": {"tts_engine": {"temperature": 0.0}},
                    "suppress_bootstrap_silence": False,
                },
            ),
        ):
            body = {
                "input": text,
                "task_type": "CustomVoice",
                "voice": "Ryan",
                "language": "English",
                "response_format": "pcm",
                "stream": True,
                "seed": 1234,
                **controls,
            }
            request = Request(
                f"http://127.0.0.1:{port}/v1/audio/speech",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=600) as response:
                samples = response.read()
                sample_rate = response.headers.get("X-Sample-Rate")
            recorded[variant] = samples
            (directory / f"{index}-{variant}.pcm").write_bytes(samples)
            print(
                json.dumps(
                    {
                        "case": index,
                        "variant": variant,
                        "bytes": len(samples),
                        "sample_rate": sample_rate,
                        "sha256": hashlib.sha256(samples).hexdigest(),
                    }
                ),
                flush=True,
            )
        print(
            json.dumps(
                {
                    "case": index,
                    "top_equals_stage_no_trim": recorded["top_level"]
                    == recorded["stage_no_trim"],
                    "top_equals_stage": recorded["top_level"] == recorded["stage"],
                    "bytes_removed_by_stage": len(recorded["stage_no_trim"])
                    - len(recorded["stage"]),
                }
            ),
            flush=True,
        )
        assert recorded["top_level"] == recorded["stage"] == recorded["stage_no_trim"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--output", type=Path, default=Path("/data/pr2772-rereview-20261014/streams")
    )
    arguments = parser.parse_args()
    check_presence()
    check_bootstrap_gate()
    check_bootstrap_boundaries()
    check_personaplex_precedence()
    if arguments.port is not None:
        check_streams(arguments.port, arguments.output)
    else:
        pass
