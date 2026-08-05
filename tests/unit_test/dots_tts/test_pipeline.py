# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


def make_payload(
    *,
    inputs,
    params: dict | None = None,
    tts_params: dict | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="req-dots-tts",
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def test_dots_tts_config_and_registry_contracts() -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig
    from sglang_omni.utils.hf import architecture_from_hf_config

    config = DotsTTSPipelineConfig(model_path="model")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "latent_engine",
        "vocoder",
    ]
    stages_by_name = {stage.name: stage for stage in config.stages}
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"latent_engine": 0, "vocoder": 0}
    assert stages_by_name["latent_engine"].stream_to == ["vocoder"]
    assert (
        stages_by_name["latent_engine"].factory
        == "sglang_omni.models.dots_tts.stages.create_sglang_latent_engine_executor"
    )
    assert stages_by_name["vocoder"].can_accept_stream_before_payload is True
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert config.supports_uploaded_voice_references() is True
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("DotsTTSForConditionalGeneration")
        is DotsTTSPipelineConfig
    )
    assert (
        architecture_from_hf_config(SimpleNamespace(model_type="dots_tts"))
        == "DotsTTSForConditionalGeneration"
    )


def test_dots_tts_rejects_tensor_parallel_config() -> None:
    from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig

    config_data = DotsTTSPipelineConfig(model_path="model").model_dump()
    latent_stage = next(
        stage for stage in config_data["stages"] if stage["name"] == "latent_engine"
    )
    latent_stage["tp_size"] = 2
    latent_stage["parallelism"] = {"tp": 2}
    latent_stage["gpu"] = [0, 1]

    with pytest.raises(ValueError, match="tp_size=1"):
        DotsTTSPipelineConfig(**config_data)


def test_dots_tts_checkpoint_view_resolves_huggingface_model_id(
    monkeypatch, tmp_path
) -> None:
    from sglang_omni.models.dots_tts import stages

    (tmp_path / "llm_config.json").write_text('{"model_type": "qwen2"}')
    (tmp_path / "config.json").write_text('{"model_type": "dots_tts"}')
    monkeypatch.setattr(
        stages,
        "resolve_checkpoint",
        lambda model_path: str(tmp_path),
        raising=False,
    )
    monkeypatch.setattr(stages, "_SGLANG_VIEW_ROOT", tmp_path / "views")

    view = stages._ensure_sglang_llm_checkpoint_view("dots-studio/dots.tts-mf")

    assert (stages.Path(view) / "config.json").read_text() == (
        tmp_path / "llm_config.json"
    ).read_text()
    assert (stages.Path(view) / "dots_config.json").resolve() == (
        tmp_path / "config.json"
    )


def test_dots_tts_preprocessing_maps_speech_request_fields() -> None:
    from sglang_omni.models.dots_tts.payload_types import DotsTTSState
    from sglang_omni.models.dots_tts.stages import preprocess_dots_tts_payload

    payload = make_payload(
        inputs={
            "text": "hello",
            "references": [{"audio_path": "ref.wav", "text": "hi"}],
        },
        params={"stream": True, "max_new_tokens": 128},
        tts_params={
            "language": "en",
            "instructions": "speak warmly",
            "seed": 7,
        },
    )

    prepared = preprocess_dots_tts_payload(payload)
    state = DotsTTSState.from_dict(prepared.data)

    assert state.text == "hello"
    assert state.prompt_audio_path == "ref.wav"
    assert state.prompt_text == "hi"
    assert state.language == "en"
    assert state.template_name == "instruction_tts"
    assert state.normalize_text is False
    assert state.max_generate_length == 128
    assert state.seed == 7
    assert state.stream is True


def test_dots_tts_preprocessing_reads_explicit_speech_generation_params() -> None:
    from sglang_omni.models.dots_tts.payload_types import DotsTTSState
    from sglang_omni.models.dots_tts.stages import preprocess_dots_tts_payload

    payload = make_payload(
        inputs={"text": "hello", "references": [{"audio_path": "ref.wav"}]},
        tts_params={
            "max_new_tokens": 12,
            "explicit_generation_params": ["max_new_tokens"],
        },
    )

    prepared = preprocess_dots_tts_payload(payload)
    state = DotsTTSState.from_dict(prepared.data)

    assert state.max_generate_length == 12


def test_dots_tts_preprocessing_reads_latent_stage_params() -> None:
    from sglang_omni.models.dots_tts.payload_types import DotsTTSState
    from sglang_omni.models.dots_tts.stages import preprocess_dots_tts_payload

    payload = make_payload(
        inputs="hello",
        params={
            "stage_params": {
                "latent_engine": {
                    "num_steps": 4,
                    "guidance_scale": 1.1,
                    "speaker_scale": 1.25,
                    "ode_method": "midpoint",
                }
            }
        },
    )

    state = DotsTTSState.from_dict(preprocess_dots_tts_payload(payload).data)

    assert state.num_steps == 4
    assert state.guidance_scale == 1.1
    assert state.speaker_scale == 1.25
    assert state.ode_method == "midpoint"


def test_dots_tts_latent_engine_alias_uses_sglang_path(monkeypatch) -> None:
    from sglang_omni.models.dots_tts import stages

    captured: dict[str, object] = {}

    def fake_sglang_factory(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "sglang-scheduler"

    monkeypatch.setattr(
        stages,
        "create_sglang_latent_engine_executor",
        fake_sglang_factory,
    )

    scheduler = stages.create_latent_engine_executor(
        "model",
        precision="bfloat16",
        max_generate_length=12,
        gpu_id=1,
        server_args_overrides={"max_running_requests": 1},
    )

    assert scheduler == "sglang-scheduler"
    assert captured["args"] == ("model",)
    assert captured["kwargs"]["precision"] == "bfloat16"
    assert captured["kwargs"]["max_generate_length"] == 12
    assert captured["kwargs"]["gpu_id"] == 1
    assert captured["kwargs"]["server_args_overrides"] == {"max_running_requests": 1}


def test_dots_tts_vocoder_decodes_latent_patches_to_audio_payload() -> None:
    from sglang_omni.models.dots_tts.stages import DotsTTSVocoder

    seen = []

    class FakeModel:
        def _decode_latents(self, latents):
            seen.append(latents)
            return np.asarray([0.0, 0.5, -0.25], dtype=np.float32)

    class FakeRuntime:
        model = FakeModel()
        sample_rate = 48000

    payload = make_payload(inputs="hello")
    payload.data = {
        "modality": "audio_latents",
        "latent_patches": [
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.asarray([[3.0, 4.0]], dtype=np.float32),
        ],
        "state": {"text": "hello", "sample_rate": 48000},
    }

    result = DotsTTSVocoder(runtime=FakeRuntime())(payload)

    assert result.data["modality"] == "audio"
    assert result.data["sample_rate"] == 48000
    assert np.frombuffer(result.data["audio_waveform"], dtype=np.float32).tolist() == [
        0.0,
        0.5,
        -0.25,
    ]
    assert seen


def _drain_outbox(scheduler) -> list:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def test_dots_tts_vocoder_scheduler_streams_audio_chunks_and_finalizes() -> None:
    from sglang_omni.models.dots_tts.stages import DotsTTSVocoderScheduler

    calls = []

    class FakeModel:
        def _init_vocoder_stream_state(self):
            return {"state": "vocoder"}

        def _stream_vocoder_patch(self, latent_patch, *, stream_state):
            calls.append(("step", latent_patch, stream_state))
            return np.asarray([0.1, 0.2], dtype=np.float32)

        def _flush_vocoder_stream(self, stream_state):
            calls.append(("flush", stream_state))
            return np.asarray([0.3], dtype=np.float32)

    class FakeRuntime:
        model = FakeModel()
        sample_rate = 48000

    payload = make_payload(inputs="hello", params={"stream": True})
    payload.data = {
        "modality": "audio_latents",
        "streamed": True,
        "state": {"text": "hello", "stream": True, "sample_rate": 48000},
    }
    scheduler = DotsTTSVocoderScheduler(runtime=FakeRuntime())

    scheduler._handle_new_request_batch(
        [IncomingMessage("req-dots-tts", "new_request", payload)]
    )
    scheduler._on_chunk(
        "req-dots-tts",
        StreamItem(
            chunk_id=0,
            data=np.asarray([[[1.0, 2.0]]], dtype=np.float32),
            from_stage="latent_engine",
            metadata={"modality": "audio_latents"},
        ),
    )
    scheduler._on_done("req-dots-tts")
    messages = _drain_outbox(scheduler)

    assert [msg.type for msg in messages] == ["stream", "stream", "result"]
    assert [msg.metadata["modality"] for msg in messages[:2]] == ["audio", "audio"]
    np.testing.assert_allclose(
        np.frombuffer(messages[0].data["audio_waveform"], dtype=np.float32),
        np.asarray([0.1, 0.2], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.frombuffer(messages[1].data["audio_waveform"], dtype=np.float32),
        np.asarray([0.3], dtype=np.float32),
    )
    assert messages[2].data.data == {"modality": "audio", "sample_rate": 48000}
    assert calls[0][0] == "step"
    assert calls[1][0] == "flush"


def test_dots_tts_sglang_native_interfaces_import() -> None:
    from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner
    from sglang_omni.models.dots_tts.request_builders import (
        DotsTTSSGLangRequestData,
        build_stream_output,
    )
    from sglang_omni.models.dots_tts.sglang_model import DotsTTSSGLangModel

    data = DotsTTSSGLangRequestData()
    data.latest_latent_patch = "patch"
    data.stream_metadata = {"modality": "audio_latents"}

    messages = list(build_stream_output("rid", data, SimpleNamespace(data=None)))

    assert DotsTTSModelRunner.__name__ == "DotsTTSModelRunner"
    assert DotsTTSSGLangModel.__name__ == "DotsTTSSGLangModel"
    assert messages[0].request_id == "rid"
    assert messages[0].metadata == {"modality": "audio_latents", "chunk_id": 0}
    assert data.latest_latent_patch is None


def test_create_sglang_latent_engine_executor_uses_sglang_factory(monkeypatch) -> None:
    from sglang_omni.models.dots_tts import stages

    captured: dict[str, object] = {}

    def fail_fallback(*args, **kwargs):
        raise AssertionError("create_sglang_latent_engine_executor used fallback")

    def fake_build_sglang_server_args(model_path, context_length, **overrides):
        captured["server_args_call"] = (model_path, context_length, overrides)
        return SimpleNamespace(
            disable_overlap_schedule=False,
            tp_size=int(overrides.get("tp_size", 1)),
        )

    class FakeModel:
        def __init__(self) -> None:
            self.native_adapter = None
            self.attached_bundle = None
            self.attached_precision = None

        def attach_side_module_bundle(self, module_bundle, *, precision=None):
            self.attached_bundle = module_bundle
            self.attached_precision = precision

    fake_model = FakeModel()
    fake_model_worker = SimpleNamespace(
        model_runner=SimpleNamespace(model=fake_model),
        gpu_id=0,
    )

    def fake_create_sglang_infrastructure(
        server_args,
        gpu_id,
        *,
        model_arch_override=None,
        **kwargs,
    ):
        captured["infra_call"] = (server_args, gpu_id, model_arch_override, kwargs)
        return (
            fake_model_worker,
            "tree",
            "req_pool",
            "kv_pool",
            "prefill",
            "decode",
            "model_config",
        )

    class FakeOutputProcessor:
        def __init__(self, **kwargs) -> None:
            captured["output_processor_kwargs"] = kwargs

    class FakeRunner:
        def __init__(self, model_worker, output_proc) -> None:
            captured["runner_args"] = (model_worker, output_proc)
            self._outbox = None

        def set_stream_outbox(self, outbox) -> None:
            self._outbox = outbox
            captured["stream_outbox"] = outbox

    class FakeScheduler:
        def __init__(self, **kwargs) -> None:
            captured["scheduler_kwargs"] = kwargs
            self.outbox = object()

    class StreamFallbackMustNotRun:
        llm_config = SimpleNamespace(vocab_size=32000)

        def _generate_latents_stream(self, *args, **kwargs):
            raise AssertionError("_generate_latents_stream must not run")

    fake_runtime_model = StreamFallbackMustNotRun()
    fake_module_bundle = SimpleNamespace(model=fake_runtime_model)
    fake_side_runtime = SimpleNamespace(
        model=fake_runtime_model,
        module_bundle=fake_module_bundle,
        precision="bfloat16",
    )

    monkeypatch.setattr(stages, "create_latent_engine_executor", fail_fallback)
    monkeypatch.setattr(
        stages,
        "_ensure_sglang_llm_checkpoint_view",
        lambda model_path: f"{model_path}-llm-view",
    )
    monkeypatch.setattr(
        stages, "build_sglang_server_args", fake_build_sglang_server_args
    )
    monkeypatch.setattr(
        stages,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )
    monkeypatch.setattr(
        stages,
        "_get_or_load_side_runtime",
        lambda *args, **kwargs: fake_side_runtime,
        raising=False,
    )
    monkeypatch.setattr(stages, "SGLangOutputProcessor", FakeOutputProcessor)
    monkeypatch.setattr(stages, "DotsTTSModelRunner", FakeRunner)
    monkeypatch.setattr(stages, "OmniScheduler", FakeScheduler)

    scheduler = stages.create_sglang_latent_engine_executor(
        "dots-model",
        gpu_id=0,
        max_generate_length=12,
    )

    assert scheduler is not None
    assert captured["server_args_call"][0] == "dots-model-llm-view"
    assert captured["server_args_call"][1] == 4096
    assert captured["server_args_call"][2]["disable_cuda_graph"] is True
    assert captured["server_args_call"][2]["disable_radix_cache"] is True
    assert captured["server_args_call"][2]["dtype"] == "bfloat16"
    assert captured["infra_call"][2] == "DotsTTSForConditionalGeneration"
    assert captured["output_processor_kwargs"]["capture_hidden"] is True
    assert captured["output_processor_kwargs"]["model"] is fake_model
    assert fake_model.attached_bundle is fake_side_runtime.module_bundle
    assert fake_model.attached_precision == "bfloat16"
    assert fake_model.native_adapter.runtime is fake_side_runtime
    assert captured["server_args_call"][2]["max_running_requests"] == 8
    assert captured["scheduler_kwargs"]["model_runner"]._outbox is scheduler.outbox


def test_create_sglang_latent_engine_accepts_concurrent_requests(monkeypatch) -> None:
    from types import SimpleNamespace

    from sglang_omni.models.dots_tts import stages

    captured: dict[str, object] = {}

    def fake_build_sglang_server_args(model_path, context_length, **overrides):
        captured["overrides"] = overrides
        return SimpleNamespace(disable_overlap_schedule=False, tp_size=1)

    fake_model = SimpleNamespace(
        native_adapter=None,
        attach_side_module_bundle=lambda module_bundle, *, precision=None: None,
    )
    fake_runtime_model = SimpleNamespace(llm_config=SimpleNamespace(vocab_size=32000))

    monkeypatch.setattr(
        stages,
        "_ensure_sglang_llm_checkpoint_view",
        lambda model_path: f"{model_path}-llm-view",
    )
    monkeypatch.setattr(
        stages, "build_sglang_server_args", fake_build_sglang_server_args
    )
    monkeypatch.setattr(
        stages,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (
            SimpleNamespace(model_runner=SimpleNamespace(model=fake_model)),
            "tree",
            "req_pool",
            "kv_pool",
            "prefill",
            "decode",
            "model_config",
        ),
    )
    monkeypatch.setattr(
        stages,
        "_get_or_load_side_runtime",
        lambda *args, **kwargs: SimpleNamespace(
            model=fake_runtime_model,
            module_bundle=SimpleNamespace(model=fake_runtime_model),
            precision="bfloat16",
        ),
    )
    monkeypatch.setattr(
        stages,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(kwargs=kwargs),
    )
    monkeypatch.setattr(
        stages,
        "DotsTTSModelRunner",
        lambda model_worker, output_proc: SimpleNamespace(
            set_stream_outbox=lambda outbox: None
        ),
    )
    monkeypatch.setattr(
        stages,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(outbox=object(), kwargs=kwargs),
    )

    scheduler = stages.create_sglang_latent_engine_executor(
        "dots-model",
        server_args_overrides={
            "max_running_requests": 2,
            "disable_radix_cache": False,
        },
    )

    assert scheduler is not None
    assert captured["overrides"]["max_running_requests"] == 2
    assert captured["overrides"]["disable_radix_cache"] is True


def test_create_sglang_latent_engine_rejects_tensor_parallel() -> None:
    from sglang_omni.models.dots_tts import stages

    with pytest.raises(ValueError, match="tp_size=1"):
        stages.create_sglang_latent_engine_executor(
            "dots-model",
            server_args_overrides={"tp_size": 2},
        )


def test_create_sglang_latent_engine_loads_side_runtime_on_worker_device(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from sglang_omni.models.dots_tts import stages

    captured: dict[str, object] = {}
    fake_model = SimpleNamespace(
        native_adapter=None,
        attach_side_module_bundle=lambda module_bundle, *, precision=None: None,
    )

    monkeypatch.setattr(
        stages,
        "_ensure_sglang_llm_checkpoint_view",
        lambda model_path: f"{model_path}-llm-view",
    )
    monkeypatch.setattr(
        stages,
        "build_sglang_server_args",
        lambda *args, **kwargs: SimpleNamespace(
            disable_overlap_schedule=False, tp_size=1
        ),
    )
    monkeypatch.setattr(
        stages,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (
            SimpleNamespace(model_runner=SimpleNamespace(model=fake_model)),
            "tree",
            "req_pool",
            "kv_pool",
            "prefill",
            "decode",
            "model_config",
        ),
    )

    def fake_get_side_runtime(*args, **kwargs):
        captured["side_runtime_kwargs"] = kwargs
        fake_runtime_model = SimpleNamespace(
            llm_config=SimpleNamespace(vocab_size=32000)
        )
        return SimpleNamespace(
            model=fake_runtime_model,
            module_bundle=SimpleNamespace(model=fake_runtime_model),
            precision="bfloat16",
        )

    monkeypatch.setattr(stages, "_get_or_load_side_runtime", fake_get_side_runtime)
    monkeypatch.setattr(
        stages,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(kwargs=kwargs),
    )
    monkeypatch.setattr(
        stages,
        "DotsTTSModelRunner",
        lambda model_worker, output_proc: SimpleNamespace(
            set_stream_outbox=lambda outbox: None
        ),
    )
    monkeypatch.setattr(
        stages,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(outbox=object(), kwargs=kwargs),
    )

    stages.create_sglang_latent_engine_executor("dots-model", device="cuda", gpu_id=1)

    assert captured["side_runtime_kwargs"]["device"] == "cuda:1"


def test_dots_tts_model_runner_runs_model_audio_step() -> None:
    import torch

    from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner
    from sglang_omni.models.dots_tts.request_builders import DotsTTSSGLangRequestData
    from sglang_omni.models.dots_tts.sglang_model import DotsTTSLatentStepOutput

    runner = DotsTTSModelRunner.__new__(DotsTTSModelRunner)
    runner.device = torch.device("cpu")

    class FakeModel:
        def __init__(self) -> None:
            self.seen = []

        def decode_audio_batch(self, batch):
            self.seen.append(batch)
            return DotsTTSLatentStepOutput(
                batch_result=None,
                next_token_ids=torch.tensor([321]),
                latent_patches=[torch.tensor([[[5.0, 6.0]]])],
                feedback_embeddings=[torch.tensor([[[0.5, 0.6]]])],
                eos_scores=[torch.tensor([0.0])],
                finished=[False],
                hidden_states=batch.hidden_states,
            )

    fake_model = FakeModel()
    runner.model = fake_model
    hidden = torch.tensor([[[1.0, 2.0]]])
    data = DotsTTSSGLangRequestData(
        control_token_id=321,
        fm_state=object(),
        latest_hidden_state=hidden,
    )
    req = SimpleNamespace(finished_reason=None)
    data.req = req
    sched_req = SimpleNamespace(request_id="rid", data=data)
    result = SimpleNamespace(next_token_ids=None)

    runner._run_model_latent_batch(result, [sched_req])

    assert fake_model.seen[0].active_indices == [0]
    assert fake_model.seen[0].hidden_states == [hidden]
    assert result.next_token_ids.tolist() == [321]
    assert torch.equal(data.latest_latent_patch, torch.tensor([[[5.0, 6.0]]]))
    assert len(data.latent_patches) == 1
    assert len(data.decode_input_embeds) == 1
    assert data.position == 1


def test_dots_tts_native_stage_factories_cache_by_stage(monkeypatch) -> None:
    from sglang_omni.models.dots_tts import stages

    stages._VOCODER_RUNTIME_CACHE.clear()
    vocoder_loads = []

    def fake_vocoder_loader(model_path, *, precision, device=None):
        runtime = SimpleNamespace(model=SimpleNamespace(), sample_rate=48000)
        vocoder_loads.append((model_path, precision, device, runtime))
        return runtime, SimpleNamespace()

    monkeypatch.setattr(
        stages,
        "create_sglang_latent_engine_executor",
        lambda *args, **kwargs: "sglang-scheduler",
    )
    monkeypatch.setattr(stages, "_get_or_load_vocoder_runtime", fake_vocoder_loader)

    latent_scheduler = stages.create_latent_engine_executor(
        "model",
        precision="bfloat16",
        max_generate_length=16,
    )
    vocoder_scheduler = stages.create_vocoder_executor(
        "model",
        precision="bfloat16",
        max_generate_length=16,
        device="cuda",
        gpu_id=2,
    )

    assert latent_scheduler == "sglang-scheduler"
    assert len(vocoder_loads) == 1
    assert vocoder_loads[0][2] == "cuda:2"
    assert vocoder_scheduler._fn.runtime is vocoder_loads[0][3]
