# SPDX-License-Identifier: Apache-2.0
"""dots.tts SGLang engine builder."""

from __future__ import annotations

from typing import Any

from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class DotsTTSEngineBuilder(TtsEngineBuilder):
    model_name = "dots.tts"
    context_length = 2048

    def __init__(self, *, optimize: bool = False) -> None:
        from sglang_omni.models.dots_tts.hf_config import DOTS_TTS_MODEL_ARCH_OVERRIDE

        self.model_arch_override = DOTS_TTS_MODEL_ARCH_OVERRIDE
        self.optimize = bool(optimize)
        self._model_runner: Any | None = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        from sglang_omni.models.dots_tts.hf_config import register_dots_tts_hf_config

        register_dots_tts_hf_config()
        if self.optimize:
            # The process-global compile policy must exist before SGLang builds
            # the model; applying it in setup_model nests Dynamo under FX.
            from sglang_omni.models.dots_tts.stages import _configure_optimized_kernels

            _configure_optimized_kernels()

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        return {
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": 1,
            "chunked_prefill_size": -1,
            "mem_fraction_static": 0.20,
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("dots.tts base support does not implement TP")
        if int(overrides.get("max_running_requests", 1)) != 1:
            raise ValueError("dots.tts base support allows max_running_requests=1")
        if bool(overrides.get("enable_torch_compile", False)):
            raise ValueError(
                "dots.tts uses its DiT compile path; SGLang backbone compile is disabled"
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
        del checkpoint_dir, device, gpu_id, server_args
        model = model_worker.model_runner.model
        model.flow.optimize = self.optimize
        model.eval()

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner

        self._model_runner = DotsTTSModelRunner(model_worker, output_proc)
        return self._model_runner

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        from sglang_omni.models.dots_tts.request_builders import (
            apply_latent_result,
            build_sglang_dots_tts_request,
        )

        return build_sglang_dots_tts_request, apply_latent_result

    def make_abort_callback(self) -> Any | None:
        assert self._model_runner is not None
        return self._model_runner.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        from sglang_omni.models.dots_tts.request_builders import build_stream_output

        return {
            "stream_output_builder": build_stream_output,
            "enable_async_decode": False,
        }


__all__ = ["DotsTTSEngineBuilder"]
