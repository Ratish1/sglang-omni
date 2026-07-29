# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS SGLang engine builder."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)

from sglang_omni.models.higgs_tts import request_builders
from sglang_omni.models.higgs_tts import utils as higgs_utils
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_HIGGS_STREAM_STRIDE,
)
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


class HiggsTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Higgs TTS"
    context_length = 4096

    def __init__(
        self,
        *,
        max_new_tokens: int | None,
        max_running_requests: int,
        cuda_graph_max_bs: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        stream_stride: int = DEFAULT_HIGGS_STREAM_STRIDE,
        stream_followup_stride: int = DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
        prefill_coalesce_requests: int = 0,
        prefill_coalesce_wait_ms: float = 60.0,
        total_gpu_memory_fraction: float | None = None,
    ) -> None:
        if total_gpu_memory_fraction is not None and not (
            0.0 < total_gpu_memory_fraction < 1.0
        ):
            raise ValueError(
                "Higgs tts_engine total_gpu_memory_fraction must be in (0, 1): "
                "it drives sglang mem_fraction_static, which requires < 1"
            )
        self.max_new_tokens = max_new_tokens
        self.max_running_requests = max_running_requests
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.stream_stride = stream_stride
        self.stream_followup_stride = stream_followup_stride
        self.prefill_coalesce_requests = prefill_coalesce_requests
        self.prefill_coalesce_wait_ms = prefill_coalesce_wait_ms
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.model: Any | None = None
        self._prefill_graph_model_runner: Any | None = None
        self._prefill_graph_alias_installed = False

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        del dtype
        # note (luojiaxuan): Radix cache is namespaced per ref-audio via
        # Req.extra_key (set in build_sglang_higgs_request); shared -100
        # placeholder prefixes from different ref audios can't cross-contaminate
        # the KV tree.
        return {
            "max_running_requests": self.max_running_requests,
            "cuda_graph_max_bs": self.cuda_graph_max_bs,
            "disable_cuda_graph": False,
            # Higgs composes request-specific multi-codebook embeddings before
            # the backbone. Breakable prefill CG captures graph-safe transformer
            # regions, runs attention at SGLang's eager graph breaks, and leaves
            # embedding composition plus sampling in the eager outer forward.
            # The CI prompt domain is <=512 tokens. Larger prefills fall back
            # to eager. Use 64-token buckets: 128-token buckets add about 43%
            # padding work on the Higgs CI prompt distribution, while this
            # halves that waste without making graph startup or memory scale
            # excessively.
            "cuda_graph_config": {
                "prefill": {
                    "backend": "breakable",
                    "max_bs": 512,
                    "bs": list(range(64, 513, 64)),
                }
            },
            "mem_fraction_static": (
                self.total_gpu_memory_fraction
                if self.total_gpu_memory_fraction is not None
                else 0.85
            ),
            "chunked_prefill_size": 8192,
            "dtype": "bfloat16",
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        # Note: (Jiaxin Deng) an explicit mem_fraction_static override (e.g.
        # --talker-mem-fraction-static) wins, but never silently.
        expected = self.total_gpu_memory_fraction
        if expected is None:
            return
        actual = overrides.get("mem_fraction_static")
        if actual is not None and abs(actual - expected) <= 1e-9:
            return
        logger.warning(
            "Higgs tts_engine mem_fraction_static=%s overrides the "
            "placement-validated total_gpu_memory_fraction=%s",
            actual,
            expected,
        )

    def customize_server_args(self, server_args: Any) -> None:
        override_server_args(
            server_args,
            "sglang_omni.higgs_tts.disable_overlap_schedule",
            disable_overlap_schedule=True,
        )

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id
        model_runner = model_worker.model_runner
        self.model = model_runner.model
        higgs_utils.truncate_rope_to_bf16(self.model)
        prefill_backend = server_args.cuda_graph_config.prefill.backend
        if prefill_backend == "disabled":
            return
        if prefill_backend != "breakable":
            raise RuntimeError(
                "Higgs prefill CUDA graph adapter only supports SGLang's "
                f"breakable backend, got {prefill_backend!r}"
            )
        if not model_runner.model_config.is_multimodal:
            raise RuntimeError(
                "Higgs prefill CUDA graph requires SGLang multimodal input "
                "buffers for request-specific audio embeddings"
            )

        # Higgs is a composition wrapper around Qwen3ForCausalLM and exposes
        # neither name SGLang's prefill graph discovery looks for, so install a
        # ``language_model`` view just for graph construction;
        # post_cuda_graph_setup removes it, while PrefillCudaGraphRunner retains
        # the resolved layer model.
        #
        # Only ``language_model``: 0.5.15 read that attribute independently of
        # resolve_language_model(), so a placeholder ``.model`` was also needed
        # to keep that helper from raising. 0.5.16 instead *uses* the helper's
        # return value, and it checks ``.model`` before ``.language_model`` —
        # a placeholder there would win and dead-end discovery on an object
        # with no ``layers``, silently disabling the prefill graph.
        if hasattr(self.model, "model") or hasattr(self.model, "language_model"):
            raise RuntimeError(
                "Higgs prefill CUDA graph discovery contract changed; "
                "refusing to capture against ambiguous model aliases"
            )
        object.__setattr__(self.model, "language_model", self.model.backbone)
        self._prefill_graph_alias_installed = True
        self._prefill_graph_model_runner = model_runner

    def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
        if server_args.cuda_graph_config.prefill.backend == "disabled":
            return
        if not self._prefill_graph_alias_installed:
            raise RuntimeError("Higgs prefill CUDA graph alias was not installed")
        try:
            runner = self._prefill_graph_model_runner
            prefill_runner = (
                None if runner is None else runner.prefill_cuda_graph_runner
            )
            if not isinstance(prefill_runner, PrefillCudaGraphRunner):
                raise RuntimeError(
                    "Higgs explicitly enabled prefill CUDA graph, but SGLang did "
                    "not construct PrefillCudaGraphRunner"
                )
            if prefill_runner.prefill_backend_name != "breakable":
                raise RuntimeError(
                    "Higgs prefill CUDA graph requires the breakable backend's "
                    "segmented body-replay contract"
                )
            configured_shapes = tuple(server_args.cuda_graph_config.prefill.bs)
            captured_shapes = tuple(prefill_runner.capture_num_tokens)
            if captured_shapes != tuple(sorted(configured_shapes)):
                raise RuntimeError(
                    "Higgs prefill CUDA graph capture shapes differ from the "
                    f"configured contract: configured={configured_shapes}, "
                    f"captured={captured_shapes}"
                )
            logger.info(
                "Higgs prefill CUDA graph active: backend=%s shapes=%s",
                server_args.cuda_graph_config.prefill.backend,
                server_args.cuda_graph_config.prefill.bs,
            )
        finally:
            model.__dict__.pop("language_model", None)
            self._prefill_graph_alias_installed = False

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return model.sampler_pool_max_running_requests

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.higgs_tts.model_runner"
        )

        return model_runner_mod.HiggsTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        return request_builders.make_higgs_scheduler_adapters(
            model,
            max_new_tokens_cap=self.max_new_tokens,
            stream_stride=self.stream_stride,
            stream_followup_stride=self.stream_followup_stride,
        )

    def make_abort_callback(self) -> Any | None:
        assert self.model is not None
        return self.model.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
            "prefill_coalesce_requests": self.prefill_coalesce_requests,
            "prefill_coalesce_wait_ms": self.prefill_coalesce_wait_ms,
        }

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
