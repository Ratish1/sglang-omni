# SPDX-License-Identifier: Apache-2.0
"""SGLang registry entry for native dots TTS."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from sglang_omni.models.dots_tts.native_adapter import (
    DotsTTSAudioStepResult,
    _as_tensor,
    _torch_dtype,
)
from sglang_omni.models.dots_tts.weight_loader import (
    map_dots_qwen2_key,
    validate_checkpoint_files,
)

try:
    from sglang.srt.models.qwen2 import Qwen2ForCausalLM
except ImportError:
    Qwen2ForCausalLM = None

if TYPE_CHECKING:
    from sglang_omni.models.dots_tts.native.models.dots_tts.model import DotsTtsModel


class DotsTTSSGLangModel(nn.Module):
    """Native dots TTS model shell registered with SGLang.

    The current implementation keeps the dots model assembly local to
    sglang-omni and exposes the class through SGLang's registry. The full
    paged-attention Qwen2 replacement is owned by the model runner path and is
    intentionally isolated from the pipeline fallback.
    """

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        if Qwen2ForCausalLM is None:
            raise RuntimeError("SGLang Qwen2ForCausalLM is required for native dots TTS")
        self.config = config
        self.qwen2 = Qwen2ForCausalLM(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.qwen2" if prefix else "qwen2",
        )
        self._native_model: "DotsTtsModel | None" = None
        self.native_adapter: Any | None = None
        self.precision = getattr(config, "torch_dtype", None) or "bfloat16"

    @property
    def native_model(self) -> "DotsTtsModel":
        if self._native_model is None:
            raise RuntimeError(
                "DotsTTSSGLangModel.native_model is not initialized. "
                "Load weights through the dots native stage factory."
            )
        return self._native_model

    def attach_native_model(self, native_model: Any, *, precision: str | None = None):
        self._native_model = native_model
        if precision is not None:
            self.precision = precision

    def validate_model_path(self, model_path: str) -> None:
        validate_checkpoint_files(model_path)

    def prepare_request_state(self, state: Any) -> Any:
        if self.native_adapter is None:
            raise RuntimeError("DotsTTSSGLangModel.native_adapter is not initialized")
        return self.native_adapter.prepare_inputs(state)

    def append_hidden_chunk(self, data: Any, hidden_state: torch.Tensor) -> None:
        append_hidden = getattr(self.native_model, "_append_hidden_chunk", None)
        if append_hidden is not None:
            append_hidden(data.fm_state, hidden_state)

    def encode_audio_patch_feedback(
        self,
        data: Any,
        latent_patch: torch.Tensor,
    ) -> torch.Tensor:
        feedback = getattr(self.native_model, "_encode_audio_patch_feedback", None)
        if feedback is not None:
            return feedback(data.fm_state, audio_patch=latent_patch)
        return self.native_model._encode_audio_patch(latent_patch)

    def step_audio_latent(
        self,
        data: Any,
        hidden_state: torch.Tensor,
    ) -> DotsTTSAudioStepResult:
        generation_kwargs = getattr(data, "generation_kwargs", {})
        decode_kwargs = {
            key: value
            for key, value in generation_kwargs.items()
            if key in {"device", "g_cond", "ode_method", "num_steps", "guidance_scale"}
        }
        self.append_hidden_chunk(data, hidden_state)
        device = decode_kwargs.get("device")
        dtype = _torch_dtype(str(self.precision))
        use_amp = (
            isinstance(device, torch.device)
            and device.type == "cuda"
            and dtype in {torch.float16, torch.bfloat16}
        )
        with torch.autocast(
            device_type=device.type if isinstance(device, torch.device) else "cuda",
            dtype=dtype,
            enabled=use_amp,
        ):
            latent_patch = _as_tensor(
                self.native_model._decode_next_audio(
                    state=data.fm_state,
                    **decode_kwargs,
                )
            )
            feedback_embedding = self.encode_audio_patch_feedback(data, latent_patch)

        io_helper = getattr(getattr(self.native_model, "core", None), "io_helper", None)
        payload_patch = (
            io_helper.denormalize(latent_patch)
            if io_helper is not None
            else latent_patch
        )
        eos_score = self._score_audio_eos(data, hidden_state, latent_patch)
        return DotsTTSAudioStepResult(
            latent_patch=_as_tensor(payload_patch),
            feedback_embedding=_as_tensor(feedback_embedding),
            eos_score=_as_tensor(eos_score) if eos_score is not None else None,
        )

    def _score_audio_eos(
        self,
        data: Any,
        hidden_state: torch.Tensor,
        latent_patch: torch.Tensor,
    ) -> torch.Tensor | None:
        generation_kwargs = getattr(data, "generation_kwargs", {})
        stop_predicate = getattr(
            self.native_model, "_should_stop_after_current_audio", None
        )
        if stop_predicate is not None:
            eos_threshold = float(generation_kwargs.get("eos_threshold", 0.8))
            return torch.tensor(
                [
                    1.0
                    if stop_predicate(data.fm_state, eos_threshold=eos_threshold)
                    else 0.0
                ],
                device=latent_patch.device,
            )
        eos_predictor = getattr(self.native_model, "_predict_eos", None)
        if eos_predictor is None:
            return None
        return eos_predictor(hidden_state, latent_patch)

    def generate_latent_patch(
        self,
        *,
        hidden_state: torch.Tensor,
        fm_state: Any,
        generation_kwargs: dict[str, Any],
    ) -> Any:
        return self.step_audio_latent(
            SimpleNamespace(
                fm_state=fm_state,
                generation_kwargs=generation_kwargs,
            ),
            hidden_state=hidden_state,
        )

    def load_weights(self, weights) -> set[str]:
        qwen2_weights = []
        for name, weight in weights:
            mapped_name = map_dots_qwen2_key(name)
            if mapped_name is None:
                continue
            qwen2_weights.append((mapped_name.removeprefix("qwen2."), weight))

        if not qwen2_weights:
            return set()
        return self.qwen2.load_weights(qwen2_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        if kwargs:
            return self.qwen2(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                **kwargs,
            )
        return self.qwen2(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )


EntryClass = DotsTTSSGLangModel


__all__ = ["DotsTTSSGLangModel", "EntryClass"]
