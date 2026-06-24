# SPDX-License-Identifier: Apache-2.0
"""Shared test shims for dots TTS unit tests."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from typing import Any


def _install_fake_sglang() -> None:
    class FakeReq:
        def __init__(
            self,
            *,
            rid,
            origin_input_text,
            origin_input_ids,
            sampling_params,
            eos_token_ids=None,
            vocab_size=None,
            **kwargs,
        ) -> None:
            del kwargs
            self.rid = rid
            self.origin_input_text = origin_input_text
            self.origin_input_ids = origin_input_ids
            self.sampling_params = sampling_params
            self.eos_token_ids = eos_token_ids
            self.vocab_size = vocab_size
            self.output_ids = []
            self.prefix_indices = []
            self.extend_input_len = len(origin_input_ids)
            self.finished_reason = None

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def normalize(self, tokenizer) -> None:
            del tokenizer
            self.stop_token_ids = None

        def verify(self, vocab_size) -> None:
            self.vocab_size = vocab_size

    class FakeGenerationBatchResult:
        def __init__(self, *, logits_output=None, can_run_cuda_graph=False) -> None:
            self.logits_output = logits_output
            self.can_run_cuda_graph = can_run_cuda_graph
            self.next_token_ids = None

    class FakeFinishMatchedToken:
        def __init__(self, token_id) -> None:
            self.token_id = token_id

    class FakeModelRegistry:
        models = {}

    class FakeModelConfig:
        pass

    class FakeSGLangModelRunner:
        pass

    class FakePortArgs:
        nccl_port = 0

        @classmethod
        def init_new(cls, server_args):
            del server_args
            return cls()

    class FakeServerArgs:
        tp_size = 1
        mem_fraction_static = None

    class FakeQwen2ForCausalLM:
        pass

    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.configs": types.ModuleType("sglang.srt.configs"),
        "sglang.srt.configs.model_config": types.ModuleType(
            "sglang.srt.configs.model_config"
        ),
        "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.schedule_batch": types.ModuleType(
            "sglang.srt.managers.schedule_batch"
        ),
        "sglang.srt.managers.scheduler": types.ModuleType(
            "sglang.srt.managers.scheduler"
        ),
        "sglang.srt.model_executor": types.ModuleType("sglang.srt.model_executor"),
        "sglang.srt.model_executor.model_runner": types.ModuleType(
            "sglang.srt.model_executor.model_runner"
        ),
        "sglang.srt.models": types.ModuleType("sglang.srt.models"),
        "sglang.srt.models.qwen2": types.ModuleType("sglang.srt.models.qwen2"),
        "sglang.srt.models.registry": types.ModuleType("sglang.srt.models.registry"),
        "sglang.srt.sampling": types.ModuleType("sglang.srt.sampling"),
        "sglang.srt.sampling.sampling_params": types.ModuleType(
            "sglang.srt.sampling.sampling_params"
        ),
        "sglang.srt.server_args": types.ModuleType("sglang.srt.server_args"),
    }
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.configs",
        "sglang.srt.managers",
        "sglang.srt.model_executor",
        "sglang.srt.models",
        "sglang.srt.sampling",
    ):
        modules[name].__path__ = []

    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].configs = modules["sglang.srt.configs"]
    modules["sglang.srt"].managers = modules["sglang.srt.managers"]
    modules["sglang.srt"].model_executor = modules["sglang.srt.model_executor"]
    modules["sglang.srt"].models = modules["sglang.srt.models"]
    modules["sglang.srt"].sampling = modules["sglang.srt.sampling"]
    modules["sglang.srt"].server_args = modules["sglang.srt.server_args"]
    modules["sglang.srt.configs"].model_config = modules[
        "sglang.srt.configs.model_config"
    ]
    modules["sglang.srt.managers"].schedule_batch = modules[
        "sglang.srt.managers.schedule_batch"
    ]
    modules["sglang.srt.managers"].scheduler = modules[
        "sglang.srt.managers.scheduler"
    ]
    modules["sglang.srt.model_executor"].model_runner = modules[
        "sglang.srt.model_executor.model_runner"
    ]
    modules["sglang.srt.models"].qwen2 = modules["sglang.srt.models.qwen2"]
    modules["sglang.srt.models"].registry = modules["sglang.srt.models.registry"]
    modules["sglang.srt.sampling"].sampling_params = modules[
        "sglang.srt.sampling.sampling_params"
    ]

    modules["sglang.srt.configs.model_config"].ModelConfig = FakeModelConfig
    modules["sglang.srt.managers.schedule_batch"].FINISH_MATCHED_TOKEN = (
        FakeFinishMatchedToken
    )
    modules["sglang.srt.managers.schedule_batch"].Req = FakeReq
    modules["sglang.srt.managers.scheduler"].GenerationBatchResult = (
        FakeGenerationBatchResult
    )
    modules["sglang.srt.model_executor.model_runner"].ModelRunner = (
        FakeSGLangModelRunner
    )
    modules["sglang.srt.models.qwen2"].Qwen2ForCausalLM = FakeQwen2ForCausalLM
    modules["sglang.srt.models.registry"].ModelRegistry = FakeModelRegistry
    modules["sglang.srt.sampling.sampling_params"].SamplingParams = FakeSamplingParams
    modules["sglang.srt.server_args"].PortArgs = FakePortArgs
    modules["sglang.srt.server_args"].ServerArgs = FakeServerArgs

    for name, module in modules.items():
        sys.modules.setdefault(name, module)


if importlib.util.find_spec("sglang") is None:
    _install_fake_sglang()


@dataclass
class _FakeSGLangARRequestData:
    input_ids: Any = None
    attention_mask: Any = None
    model_inputs: dict[str, Any] = field(default_factory=dict)
    output_ids: list[int] = field(default_factory=list)
    extra_model_outputs: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    weight_version: str | None = None
    return_logprob: bool = False
    output_token_logprobs: list[Any] = field(default_factory=list)
    capture_model_output_keys: tuple[str, ...] = ()
    max_new_tokens: int | None = None
    enforce_request_limits: bool = False
    temperature: float = 0.0
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    suppress_tokens: list[int] | None = None
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    input_embeds_are_projected: bool = False
    prefill_input_embeds: Any = None
    decode_input_embeds: list[Any] = field(default_factory=list)
    stage_payload: Any = None
    talker_model_inputs: dict[str, Any] = field(default_factory=dict)
    pending_feedback_queue: Any = None
    pending_text_queue: Any = None
    tts_pad_embed: Any = None
    tts_eos_embed: Any = None
    thinker_chunks_done: bool = True


def _fake_build_sglang_server_args(
    model_path,
    context_length,
    **overrides,
):
    attrs = dict(overrides)
    attrs.setdefault("disable_overlap_schedule", False)
    attrs["tp_size"] = int(attrs.get("tp_size", 1))
    attrs["model_path"] = model_path
    attrs["context_length"] = context_length
    return types.SimpleNamespace(**attrs)


_fake_sglang_backend = types.ModuleType("sglang_omni.scheduling.sglang_backend")
_fake_sglang_backend.SGLangARRequestData = _FakeSGLangARRequestData
_fake_sglang_backend.SGLangDLLMRequestData = type("SGLangDLLMRequestData", (), {})
_fake_sglang_backend.SGLangOutputProcessor = lambda **kwargs: types.SimpleNamespace(
    kwargs=kwargs
)
_fake_sglang_backend.build_sglang_server_args = _fake_build_sglang_server_args
_fake_sglang_backend.apply_encoder_mem_reserve = lambda *args, **kwargs: None
_fake_sglang_backend.create_tree_cache = lambda *args, **kwargs: None
_fake_sglang_backend.DecodeManager = type("DecodeManager", (), {})
_fake_sglang_backend.PrefillManager = type("PrefillManager", (), {})
sys.modules.setdefault("sglang_omni.scheduling.sglang_backend", _fake_sglang_backend)


class _FakeOmniScheduler:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.outbox = object()


def _fake_create_sglang_infrastructure(*args, **kwargs):
    raise AssertionError("create_sglang_infrastructure must be monkeypatched")


_fake_omni_scheduler = types.ModuleType("sglang_omni.scheduling.omni_scheduler")
_fake_omni_scheduler.OmniScheduler = _FakeOmniScheduler
sys.modules.setdefault("sglang_omni.scheduling.omni_scheduler", _fake_omni_scheduler)

_fake_bootstrap = types.ModuleType("sglang_omni.scheduling.bootstrap")
_fake_bootstrap.create_sglang_infrastructure = _fake_create_sglang_infrastructure
sys.modules.setdefault("sglang_omni.scheduling.bootstrap", _fake_bootstrap)
