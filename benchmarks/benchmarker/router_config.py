# SPDX-License-Identifier: Apache-2.0
"""Render Rust router configurations for homogeneous worker pools."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Literal


class RouterTopology(StrEnum):
    ASR = "asr"
    TTS = "tts"
    TTS_SERVING = "tts_serving"
    OMNI_TEXT = "omni_text"
    OMNI_AUDIO = "omni_audio"


def render_router_config(
    *,
    topology: RouterTopology,
    router_port: int,
    worker_urls: list[str],
    model_name: str,
    max_inflight: int,
    strategy: Literal["round_robin", "least_requests"],
    batch_limit: int | None = None,
    generation_streaming: bool = True,
    named_voice: bool = False,
) -> str:
    """Render a worker pool with caller-owned admission limits and policy."""
    preamble = router_preamble(
        topology, router_port, max_inflight, strategy, batch_limit
    )
    worker_blocks = [
        worker_block(
            topology=topology,
            ordinal=ordinal,
            worker_url=worker_url,
            model_name=model_name,
            generation_streaming=generation_streaming,
            named_voice=named_voice,
            max_inflight=max_inflight,
            batch_limit=batch_limit,
        )
        for ordinal, worker_url in enumerate(worker_urls, start=1)
    ]
    return f"{preamble.rstrip()}\n\n" + "\n\n".join(worker_blocks) + "\n"


def router_preamble(
    topology: RouterTopology,
    router_port: int,
    max_inflight: int,
    strategy: Literal["round_robin", "least_requests"],
    batch_limit: int | None,
) -> str:
    if topology is RouterTopology.TTS_SERVING:
        assert batch_limit is not None
        admission = (
            f"global = {max_inflight}",
            f"speech_http = {max_inflight}",
            f"speech_batch = {max_inflight * batch_limit}",
            f"speech_websocket = {max_inflight}",
        )
    else:
        admission_class = {
            RouterTopology.ASR: "transcription_http",
            RouterTopology.TTS: "speech_http",
            RouterTopology.OMNI_TEXT: "generation_http",
            RouterTopology.OMNI_AUDIO: "generation_http",
        }[topology]
        admission = (f"global = {max_inflight}", f"{admission_class} = {max_inflight}")
    lines = [
        "schema_version = 1",
        "",
        "[server]",
        f'listen = {toml_string(f"127.0.0.1:{router_port}")}',
        "",
        "[shutdown]",
        "drain_timeout_ms = 30000",
        "",
        "[logging]",
        'format = "json"',
        'filter = "info"',
        "",
        "[router]",
        f"strategy = {toml_string(strategy)}",
    ]
    if topology is RouterTopology.TTS_SERVING:
        lines.append('voice_owner_worker_id = "tts-serving-1"')
    else:
        pass
    lines.extend(
        [
            "",
            "[admission]",
            *admission,
            "",
            "[health]",
            "",
        ]
    )
    if topology in {RouterTopology.OMNI_TEXT, RouterTopology.OMNI_AUDIO}:
        lines.extend(
            [
                "[http_generation]",
                'trust_domain = "local"',
            ]
        )
        return "\n".join(lines)
    else:
        pass

    routes = {
        RouterTopology.ASR: ["transcription"],
        RouterTopology.TTS: ["speech"],
        RouterTopology.TTS_SERVING: ["speech", "speech_batch"],
    }[topology]
    lines.extend(
        [
            "[http_media]",
            f"routes = {toml_array(routes)}",
            'trust_domain = "local"',
        ]
    )
    if topology is RouterTopology.TTS_SERVING:
        lines.extend(["", "[websocket.speech]", 'trust_domain = "local"'])
    else:
        pass
    return "\n".join(lines)


def worker_block(
    *,
    topology: RouterTopology,
    ordinal: int,
    worker_url: str,
    model_name: str,
    generation_streaming: bool,
    named_voice: bool,
    max_inflight: int,
    batch_limit: int | None,
) -> str:
    prefix = {
        RouterTopology.ASR: "asr",
        RouterTopology.TTS: "tts",
        RouterTopology.TTS_SERVING: "tts-serving",
        RouterTopology.OMNI_TEXT: "omni",
        RouterTopology.OMNI_AUDIO: "omni",
    }[topology]
    worker_id = f"{prefix}-{ordinal}"
    lines = [
        "[[workers]]",
        f"worker_id = {toml_string(worker_id)}",
        f"base_url = {toml_string(worker_url.rstrip('/') + '/')}",
        'trust_domain = "local"',
        f"default_model_id = {toml_string(model_name)}",
    ]
    if topology is RouterTopology.TTS_SERVING:
        lines.extend(
            [
                "",
                "[workers.capacity]",
                f"speech_websocket = {max_inflight}",
            ]
        )
    else:
        pass
    lines.extend(
        [
            "",
            service_profiles(
                topology,
                model_name,
                named_voice,
                generation_streaming=generation_streaming,
                batch_limit=batch_limit,
            ),
        ]
    )
    return "\n".join(lines)


def service_profiles(
    topology: RouterTopology,
    model_name: str,
    named_voice: bool,
    *,
    generation_streaming: bool,
    batch_limit: int | None,
) -> str:
    model_ids = toml_array([model_name])
    if topology is RouterTopology.ASR:
        return transcription_profile(
            model_ids,
            task="transcribe",
            formats=["json", "verbose_json", "sse"],
        )
    elif topology is RouterTopology.TTS:
        if named_voice:
            tasks = ["text_to_speech"]
            reference_forms = ["none"]
            voice_name_policy = "preset"
        else:
            tasks = ["voice_clone"]
            reference_forms = ["direct", "list"]
            voice_name_policy = "uploaded"
        return "\n\n".join(
            [
                speech_profile(
                    service="speech_http",
                    model_ids=model_ids,
                    response_formats=["wav"],
                    stream_modes=["non_streaming"],
                    tasks=tasks,
                    reference_forms=reference_forms,
                    voice_name_policy=voice_name_policy,
                ),
                speech_profile(
                    service="speech_http",
                    model_ids=model_ids,
                    response_formats=["pcm"],
                    stream_modes=["non_streaming", "streaming"],
                    tasks=tasks,
                    reference_forms=reference_forms,
                    voice_name_policy=voice_name_policy,
                ),
            ]
        )
    elif topology is RouterTopology.TTS_SERVING:
        assert batch_limit is not None
        profiles = [
            speech_profile(
                service="speech_http",
                model_ids=model_ids,
                response_formats=["mp3", "opus", "aac", "flac", "wav"],
                stream_modes=["non_streaming"],
                tasks=["text_to_speech", "voice_clone", "voice_design"],
                reference_forms=["none", "direct", "list"],
                voice_name_policy="uploaded",
            ),
            speech_profile(
                service="speech_http",
                model_ids=model_ids,
                response_formats=["pcm"],
                stream_modes=["non_streaming", "streaming"],
                tasks=["text_to_speech", "voice_clone", "voice_design"],
                reference_forms=["none", "direct", "list"],
                voice_name_policy="uploaded",
            ),
            speech_batch_profile(model_ids, batch_limit),
            speech_profile(
                service="speech_websocket",
                model_ids=model_ids,
                response_formats=["pcm"],
                stream_modes=["non_streaming", "streaming"],
                tasks=["text_to_speech", "voice_clone", "voice_design"],
                reference_forms=["none", "direct", "list"],
                voice_name_policy="uploaded",
            ),
        ]
        return "\n\n".join(profiles)
    else:
        return generation_profile(
            model_ids=model_ids,
            audio_output=topology is RouterTopology.OMNI_AUDIO,
            streaming=generation_streaming,
        )


def transcription_profile(
    model_ids: str,
    *,
    task: str,
    formats: list[str],
) -> str:
    return "\n".join(
        [
            "[[workers.service_profiles]]",
            'service = "transcription_http"',
            f"model_ids = {model_ids}",
            f"task = {toml_string(task)}",
            f"response_formats = {toml_array(formats)}",
            'stream_modes = ["non_streaming", "streaming"]',
        ]
    )


def speech_profile(
    *,
    service: str,
    model_ids: str,
    response_formats: list[str],
    stream_modes: list[str],
    tasks: list[str],
    reference_forms: list[str],
    voice_name_policy: str,
) -> str:
    return "\n".join(
        [
            "[[workers.service_profiles]]",
            f"service = {toml_string(service)}",
            f"model_ids = {model_ids}",
            f"response_formats = {toml_array(response_formats)}",
            f"stream_modes = {toml_array(stream_modes)}",
            f"tasks = {toml_array(tasks)}",
            f"reference_forms = {toml_array(reference_forms)}",
            f"voice_name_policy = {toml_string(voice_name_policy)}",
        ]
    )


def speech_batch_profile(model_ids: str, batch_limit: int) -> str:
    return "\n".join(
        [
            "[[workers.service_profiles]]",
            'service = "speech_batch"',
            f"model_ids = {model_ids}",
            'response_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]',
            'tasks = ["text_to_speech", "voice_clone", "voice_design"]',
            'reference_forms = ["none", "direct", "list"]',
            'voice_name_policy = "uploaded"',
            f"max_batch_size = {batch_limit}",
        ]
    )


def generation_profile(*, model_ids: str, audio_output: bool, streaming: bool) -> str:
    output_modalities = ["text", "audio"] if audio_output else ["text"]
    audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"] if audio_output else []
    stream_modes = ["non_streaming", "streaming"] if streaming else ["non_streaming"]
    return "\n".join(
        [
            "[[workers.service_profiles]]",
            'service = "generation_http"',
            f"model_ids = {model_ids}",
            'message_content_forms = ["string", "typed_parts"]',
            'media_placements = ["top_level", "typed_parts"]',
            'input_modalities = ["text", "image", "audio", "video"]',
            f"output_modalities = {toml_array(output_modalities)}",
            f"chat_audio_formats = {toml_array(audio_formats)}",
            f"stream_modes = {toml_array(stream_modes)}",
        ]
    )


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def toml_array(values: list[str]) -> str:
    return "[" + ", ".join(toml_string(value) for value in values) + "]"
