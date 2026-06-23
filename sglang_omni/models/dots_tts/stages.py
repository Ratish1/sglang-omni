# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the dots.tts wrapper pipeline."""

from __future__ import annotations

import logging
import hashlib
import threading
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner
from sglang_omni.models.dots_tts.native_adapter import DotsTTSNativeAdapter
from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.request_builders import (
    build_stream_output,
    make_dots_tts_scheduler_adapters,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_DOTS_TTS_INSTALL_HINT = (
    "dots.tts native support requires the dots inference dependencies "
    "to be installed in the serving environment."
)
_RUNTIME_TEMPLATE_NAMES = {
    "tts",
    "instruction_tts",
    "text_to_audio",
    "tts_interleave",
}
_SIDE_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}
_SIDE_RUNTIME_CACHE_LOCK = threading.Lock()
_VOCODER_RUNTIME_CACHE: dict[tuple[Any, ...], tuple[Any, threading.RLock]] = {}
_VOCODER_RUNTIME_CACHE_LOCK = threading.Lock()
_SGLANG_VIEW_ROOT = Path("/tmp/sglang_omni_dots_tts_llm_views")


def _ensure_sglang_llm_checkpoint_view(model_path: str) -> str:
    """Create a local checkpoint view whose config.json is dots llm_config.json."""

    root = Path(model_path).expanduser().resolve()
    llm_config_path = root / "llm_config.json"
    if not llm_config_path.exists():
        raise FileNotFoundError(f"dots TTS checkpoint is missing {llm_config_path}")
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    view = _SGLANG_VIEW_ROOT / digest
    view.mkdir(parents=True, exist_ok=True)

    config_view = view / "config.json"
    config_view.write_bytes(llm_config_path.read_bytes())
    for source in root.iterdir():
        target_name = "dots_config.json" if source.name == "config.json" else source.name
        target = view / target_name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source)
    return str(view)


def _runtime_cache_key(
    model_path: str,
    *,
    precision: str,
    optimize: bool,
    max_generate_length: int,
    device: str | None,
    cache_dir: str | None,
    revision: str | None,
) -> tuple[Any, ...]:
    return (
        str(model_path),
        str(precision),
        bool(optimize),
        int(max_generate_length),
        device,
        cache_dir,
        revision,
    )


def _resolve_worker_device(device: str | None, gpu_id: int) -> str:
    if device is None or device == "cuda":
        return f"cuda:{int(gpu_id)}"
    return device


def _get_or_load_side_runtime(
    model_path: str,
    *,
    precision: str,
    optimize: bool,
    max_generate_length: int,
    device: str | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> Any:
    key = _runtime_cache_key(
        model_path,
        precision=precision,
        optimize=optimize,
        max_generate_length=max_generate_length,
        device=device,
        cache_dir=cache_dir,
        revision=revision,
    )
    with _SIDE_RUNTIME_CACHE_LOCK:
        cached = _SIDE_RUNTIME_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from sglang_omni.models.dots_tts.native.side_runtime import (
                DotsTtsSideRuntime,
            )
        except ImportError as exc:
            raise RuntimeError(_DOTS_TTS_INSTALL_HINT) from exc
        runtime = DotsTtsSideRuntime.from_pretrained(
            model_path,
            precision=precision,
            optimize=optimize,
            max_generate_length=max_generate_length,
            device=device,
            cache_dir=cache_dir,
            revision=revision,
        )
        return _SIDE_RUNTIME_CACHE.setdefault(key, runtime)


def _get_or_load_vocoder_runtime(
    model_path: str,
    *,
    precision: str,
    device: str | None = None,
) -> tuple[Any, threading.RLock]:
    key = (str(model_path), str(precision), str(device))
    with _VOCODER_RUNTIME_CACHE_LOCK:
        cached = _VOCODER_RUNTIME_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from sglang_omni.models.dots_tts.native.vocoder_runtime import (
                DotsTTSNativeVocoderRuntime,
            )
        except ImportError as exc:
            raise RuntimeError(_DOTS_TTS_INSTALL_HINT) from exc
        runtime = DotsTTSNativeVocoderRuntime.from_pretrained(
            model_path,
            precision=precision,
            device=device,
        )
        cached = (runtime, threading.RLock())
        return _VOCODER_RUNTIME_CACHE.setdefault(key, cached)


def _as_inputs_dict(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, str):
        return {"text": inputs}
    if isinstance(inputs, dict):
        return dict(inputs)
    if inputs is None:
        return {}
    return {"text": str(inputs)}


def _first_reference(inputs: dict[str, Any]) -> dict[str, Any] | None:
    references = inputs.get("references")
    if isinstance(references, list) and references:
        first = references[0]
        if isinstance(first, dict):
            return first
    return None


def _reference_audio_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        path = value.get("audio_path") or value.get("path")
        return str(path) if path else None
    return str(value)


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        raise ValueError(f"{field_name} must be positive")
    return out


def _optional_seed(value: Any) -> int | None:
    if value is None:
        return None
    seed = int(value)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def _resolve_template_name(
    *, inputs: dict[str, Any], params: dict[str, Any], tts_params: dict[str, Any]
) -> str | None:
    raw = (
        inputs.get("template_name")
        or tts_params.get("template_name")
        or params.get("template_name")
    )
    if raw is None:
        task_type = tts_params.get("task_type") or params.get("task_type")
        instructions = tts_params.get("instructions") or inputs.get("instructions")
        if task_type in _RUNTIME_TEMPLATE_NAMES:
            raw = task_type
        elif instructions:
            raw = "instruction_tts"
        else:
            raw = "tts"
    template_name = str(raw)
    if template_name not in _RUNTIME_TEMPLATE_NAMES:
        raise ValueError(
            f"Unsupported dots.tts template_name={template_name!r}; "
            f"expected one of {sorted(_RUNTIME_TEMPLATE_NAMES)}"
        )
    return template_name


def preprocess_dots_tts_payload(payload: StagePayload) -> StagePayload:
    """Normalize OpenAI-style speech inputs for the dots.tts runtime."""

    inputs = _as_inputs_dict(payload.request.inputs)
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params") or {}
    if not isinstance(params, dict):
        raise TypeError("dots.tts request params must be a dict")
    if not isinstance(tts_params, dict):
        raise TypeError("dots.tts metadata['tts_params'] must be a dict")

    ref = _first_reference(inputs)
    text = inputs.get("input") or inputs.get("text") or ""
    prompt_audio_path = (
        _reference_audio_path(inputs.get("prompt_audio_path"))
        or _reference_audio_path(inputs.get("prompt_audio"))
        or _reference_audio_path(inputs.get("reference_audio"))
        or _reference_audio_path(tts_params.get("ref_audio"))
    )
    prompt_text = (
        inputs.get("prompt_text")
        or inputs.get("reference_text")
        or tts_params.get("ref_text")
    )
    if ref is not None:
        prompt_audio_path = prompt_audio_path or _reference_audio_path(ref)
        prompt_text = prompt_text or ref.get("text")

    max_generate_length = (
        _optional_positive_int(
            inputs.get("max_generate_length"),
            field_name="max_generate_length",
        )
        or _optional_positive_int(
            tts_params.get("max_generate_length"), field_name="max_generate_length"
        )
        or _optional_positive_int(
            params.get("max_generate_length"),
            field_name="max_generate_length",
        )
        or _optional_positive_int(
            params.get("max_new_tokens"),
            field_name="max_new_tokens",
        )
    )

    state = DotsTTSState(
        text=str(text),
        prompt_audio_path=prompt_audio_path,
        prompt_text=str(prompt_text) if prompt_text is not None else None,
        template_name=_resolve_template_name(
            inputs=inputs, params=params, tts_params=tts_params
        ),
        language=(
            str(inputs.get("language") or tts_params.get("language"))
            if (inputs.get("language") or tts_params.get("language")) is not None
            else None
        ),
        speaker_scale=float(
            inputs.get("speaker_scale")
            or tts_params.get("speaker_scale")
            or params.get("speaker_scale")
            or 1.5
        ),
        ode_method=str(
            inputs.get("ode_method")
            or tts_params.get("ode_method")
            or params.get("ode_method")
            or "euler"
        ),
        num_steps=int(
            inputs.get("num_steps")
            or tts_params.get("num_steps")
            or params.get("num_steps")
            or 10
        ),
        guidance_scale=float(
            inputs.get("guidance_scale")
            or tts_params.get("guidance_scale")
            or params.get("guidance_scale")
            or 1.2
        ),
        normalize_text=bool(
            inputs.get("normalize_text")
            or tts_params.get("normalize_text")
            or params.get("normalize_text")
            or False
        ),
        profile_inference=bool(
            inputs.get("profile_inference")
            or tts_params.get("profile_inference")
            or params.get("profile_inference")
            or False
        ),
        max_generate_length=max_generate_length,
        seed=_optional_seed(
            inputs.get("seed") or tts_params.get("seed") or params.get("seed")
        ),
        stream=bool(params.get("stream", False)),
    )
    if state.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    payload.data = state.to_dict()
    return payload


def _to_latent_tensor(latent: Any) -> torch.Tensor:
    if isinstance(latent, torch.Tensor):
        return latent.detach()
    return torch.as_tensor(latent)


def _runtime_tensor_device(runtime: Any) -> torch.device | None:
    model = getattr(runtime, "model", None)
    modules = [
        model,
        getattr(model, "core", None),
        getattr(model, "vocoder", None),
    ]
    for module in modules:
        parameters = getattr(module, "parameters", None)
        if parameters is None:
            continue
        try:
            first_param = next(parameters())
        except (StopIteration, TypeError):
            continue
        return first_param.device
    return None


def _to_runtime_latent_tensor(latent: Any, runtime: Any) -> torch.Tensor:
    tensor = _to_latent_tensor(latent)
    device = _runtime_tensor_device(runtime)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)
    return tensor


def _concat_latent_patches(
    latent_patches: list[Any],
    *,
    runtime: Any | None = None,
) -> torch.Tensor:
    if not latent_patches:
        raise RuntimeError("dots.tts latent engine produced no latent patches")
    tensors = [
        _to_runtime_latent_tensor(patch, runtime)
        if runtime is not None
        else _to_latent_tensor(patch)
        for patch in latent_patches
    ]
    try:
        return torch.cat(tensors, dim=1)
    except RuntimeError:
        return torch.cat(tensors, dim=0)


class DotsTTSVocoder:
    """Decode dots.tts continuous latent patches to a waveform payload."""

    def __init__(
        self,
        *,
        runtime: Any,
        runtime_lock: threading.RLock | None = None,
    ) -> None:
        self.runtime = runtime
        self.runtime_lock = runtime_lock or threading.RLock()

    def __call__(self, payload: StagePayload) -> StagePayload:
        if not isinstance(payload.data, dict):
            raise TypeError("dots.tts vocoder payload data must be a dict")
        if payload.data.get("modality") != "audio_latents":
            raise ValueError("dots.tts vocoder expects modality='audio_latents'")
        state = DotsTTSState.from_dict(payload.data.get("state") or {})
        latent_patches = payload.data.get("latent_patches")
        if not isinstance(latent_patches, list):
            raise TypeError("dots.tts vocoder expects latent_patches to be a list")
        latents = _concat_latent_patches(latent_patches, runtime=self.runtime)
        with self.runtime_lock:
            audio = self.runtime.model._decode_latents(latents)
        sample_rate = int(state.sample_rate or getattr(self.runtime, "sample_rate", 48000))
        payload.data = audio_waveform_payload(
            audio,
            sample_rate=sample_rate,
            modality="audio",
            source_hint="dots.tts",
        )
        payload.data["state"] = state.to_dict()
        usage: dict[str, Any] = {}
        if state.engine_time_s:
            usage["engine_time_s"] = state.engine_time_s
        if state.rtf is not None:
            usage["rtf"] = state.rtf
        if usage:
            payload.data["usage"] = usage
        return payload


class DotsTTSVocoderScheduler(StreamingSimpleScheduler):
    """Streaming AudioVAE scheduler for dots.tts latent patches."""

    def __init__(
        self,
        *,
        runtime: Any,
        runtime_lock: threading.RLock | None = None,
    ) -> None:
        self.runtime = runtime
        self.runtime_lock = runtime_lock or threading.RLock()
        self._vocoder = DotsTTSVocoder(runtime=runtime, runtime_lock=self.runtime_lock)
        self._stream_states: dict[str, Any] = {}
        self._audio_chunks: dict[str, list[Any]] = {}
        self._stream_enabled: dict[str, bool] = {}
        super().__init__(self._vocoder)
        self._payloads = self._stream_payloads

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        if not isinstance(payload.data, dict):
            return False
        if payload.data.get("modality") != "audio_latents":
            return False
        if payload.data.get("streamed") is True:
            return True
        state = DotsTTSState.from_dict(payload.data.get("state") or {})
        return bool(state.stream)

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        state = DotsTTSState.from_dict(payload.data.get("state") or {})
        self._stream_enabled[request_id] = bool(state.stream)
        self._ensure_stream_state(request_id)

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        if isinstance(item.metadata, dict):
            modality = item.metadata.get("modality")
            if modality is not None and modality != "audio_latents":
                raise ValueError(
                    f"dots.tts vocoder expected audio_latents stream, got {modality!r}"
                )
        stream_state = self._ensure_stream_state(request_id)
        latent_patch = _to_runtime_latent_tensor(item.data, self.runtime)
        with self.runtime_lock:
            audio_chunk = self.runtime.model._stream_vocoder_patch(
                latent_patch,
                stream_state=stream_state,
            )
        return self._append_audio_chunk(request_id, audio_chunk)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        stream_state = self._ensure_stream_state(request_id)
        messages: list[OutgoingMessage] = []
        with self.runtime_lock:
            final_chunk = self.runtime.model._flush_vocoder_stream(stream_state)
        messages.extend(self._append_audio_chunk(request_id, final_chunk))

        payload = self._payloads.get(request_id)
        if payload is None:
            raise RuntimeError(f"dots.tts vocoder is missing payload for {request_id!r}")
        state = (
            DotsTTSState.from_dict(payload.data.get("state") or {})
            if isinstance(payload.data, dict)
            else DotsTTSState()
        )
        sample_rate = int(state.sample_rate or getattr(self.runtime, "sample_rate", 48000))
        if self._stream_enabled.get(request_id, True):
            final_data: dict[str, Any] = {
                "modality": "audio",
                "sample_rate": sample_rate,
            }
        else:
            audio_parts = self._audio_chunks.get(request_id, [])
            if not audio_parts:
                raise RuntimeError(f"dots.tts vocoder produced no audio for {request_id!r}")
            full_audio = torch.cat(
                [_to_latent_tensor(part).reshape(-1).cpu() for part in audio_parts],
                dim=0,
            )
            final_data = audio_waveform_payload(
                full_audio,
                sample_rate=sample_rate,
                modality="audio",
                source_hint="dots.tts",
            )
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self._stream_states.pop(request_id, None)
        self._audio_chunks.pop(request_id, None)
        self._stream_enabled.pop(request_id, None)

    def _ensure_stream_state(self, request_id: str) -> Any:
        stream_state = self._stream_states.get(request_id)
        if stream_state is None:
            with self.runtime_lock:
                stream_state = self.runtime.model._init_vocoder_stream_state()
            self._stream_states[request_id] = stream_state
            self._audio_chunks.setdefault(request_id, [])
        return stream_state

    def _append_audio_chunk(
        self, request_id: str, audio_chunk: Any
    ) -> list[OutgoingMessage]:
        audio_tensor = _to_latent_tensor(audio_chunk).reshape(-1)
        if audio_tensor.numel() == 0:
            return []
        self._audio_chunks.setdefault(request_id, []).append(audio_tensor.detach().cpu())
        sample_rate = int(getattr(self.runtime, "sample_rate", 48000))
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,
                data=audio_waveform_payload(
                    audio_tensor,
                    sample_rate=sample_rate,
                    modality="audio",
                    source_hint="dots.tts",
                ),
                metadata={"modality": "audio"},
            )
        ]


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    del model_path
    return SimpleScheduler(
        preprocess_dots_tts_payload,
        max_concurrency=max_concurrency,
    )


def create_latent_engine_executor(
    model_path: str,
    *,
    precision: str = "bfloat16",
    optimize: bool = False,
    max_generate_length: int = 500,
    device: str | None = None,
    gpu_id: int | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    server_args_overrides: dict[str, Any] | None = None,
) -> OmniScheduler:
    return create_sglang_latent_engine_executor(
        model_path,
        precision=precision,
        optimize=optimize,
        max_generate_length=max_generate_length,
        device=device,
        gpu_id=gpu_id,
        cache_dir=cache_dir,
        revision=revision,
        server_args_overrides=server_args_overrides,
    )


def create_sglang_latent_engine_executor(
    model_path: str,
    *,
    precision: str = "bfloat16",
    optimize: bool = False,
    max_generate_length: int = 500,
    device: str | None = None,
    gpu_id: int | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    server_args_overrides: dict[str, Any] | None = None,
) -> OmniScheduler:
    """Create the SGLang-backed dots continuous-latent engine."""

    overrides: dict[str, Any] = {
        "disable_cuda_graph": True,
        "mem_fraction_static": 0.85,
        "max_running_requests": 8,
        "chunked_prefill_size": 4096,
        "dtype": precision,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)
    if int(overrides.get("tp_size", 1)) != 1:
        raise ValueError("Dots TTS native latent engine v1 supports only tp_size=1")

    if gpu_id is None:
        if device and ":" in device:
            gpu_id = int(device.split(":")[-1])
        else:
            gpu_id = 0
    side_device = _resolve_worker_device(device, int(gpu_id))

    sglang_model_path = _ensure_sglang_llm_checkpoint_view(model_path)
    server_args = build_sglang_server_args(
        sglang_model_path,
        context_length=4096,
        **overrides,
    )
    server_args.disable_overlap_schedule = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        int(gpu_id),
        model_arch_override="DotsTTSForConditionalGeneration",
    )

    runtime = _get_or_load_side_runtime(
        model_path,
        precision=precision,
        optimize=optimize,
        max_generate_length=max_generate_length,
        device=side_device,
        cache_dir=cache_dir,
        revision=revision,
    )

    model = model_worker.model_runner.model
    model.attach_native_model(runtime.model, precision=runtime.precision)
    model.native_adapter = DotsTTSNativeAdapter(runtime)

    output_proc = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=None,
        model=model,
    )
    model_runner = DotsTTSModelRunner(model_worker, output_proc)
    request_builder, result_adapter = make_dots_tts_scheduler_adapters(
        adapter=model.native_adapter,
    )

    scheduler = OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=build_stream_output,
        enable_async_decode=False,
    )
    model_runner.set_stream_outbox(scheduler.outbox)
    return scheduler


def create_vocoder_executor(
    model_path: str,
    *,
    precision: str = "bfloat16",
    optimize: bool = False,
    max_generate_length: int = 500,
    device: str | None = None,
    gpu_id: int | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> SimpleScheduler:
    if gpu_id is None:
        if device and ":" in device:
            gpu_id = int(device.split(":")[-1])
        else:
            gpu_id = 0
    vocoder_device = _resolve_worker_device(device, int(gpu_id))
    runtime, runtime_lock = _get_or_load_vocoder_runtime(
        model_path,
        precision=precision,
        device=vocoder_device,
    )
    del optimize, max_generate_length, cache_dir, revision
    return DotsTTSVocoderScheduler(
        runtime=runtime,
        runtime_lock=runtime_lock,
    )


__all__ = [
    "DotsTTSVocoder",
    "DotsTTSVocoderScheduler",
    "create_latent_engine_executor",
    "create_preprocessing_executor",
    "create_sglang_latent_engine_executor",
    "create_vocoder_executor",
    "preprocess_dots_tts_payload",
]
