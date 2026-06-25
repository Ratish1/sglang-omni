# SPDX-License-Identifier: Apache-2.0
"""SGLang registry entry for native dots TTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult
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


@dataclass
class DotsTTSLatentBatch:
    requests: list[Any]
    active_indices: list[int]
    hidden_states: list[torch.Tensor | None]


@dataclass
class DotsTTSLatentStepOutput:
    batch_result: GenerationBatchResult | None
    next_token_ids: torch.Tensor
    latent_patches: list[torch.Tensor | None]
    feedback_embeddings: list[torch.Tensor | None]
    eos_scores: list[torch.Tensor | None]
    finished: list[bool]
    hidden_states: list[torch.Tensor | None]


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
            raise RuntimeError(
                "SGLang Qwen2ForCausalLM is required for native dots TTS"
            )
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
        native_model.set_token_embedding(self.qwen2.get_input_embeddings())

    def attach_side_module_bundle(self, bundle: Any, *, precision: str | None = None):
        self.attach_native_model(bundle.model, precision=precision)
        self.core = bundle.core
        self.xvector_extractor = bundle.xvector_extractor
        self.tokenizer = bundle.tokenizer
        self.dots_config = bundle.config
        self.llm_config = bundle.llm_config

    def validate_model_path(self, model_path: str) -> None:
        validate_checkpoint_files(model_path)

    def prepare_request_state(self, state: Any) -> Any:
        if self.native_adapter is None:
            raise RuntimeError("DotsTTSSGLangModel.native_adapter is not initialized")
        return self.native_adapter.prepare_inputs(state)

    def append_hidden_chunk(self, data: Any, hidden_state: torch.Tensor) -> None:
        self.native_model._append_hidden_chunk(data.fm_state, hidden_state)

    def encode_audio_patch_feedback(
        self,
        data: Any,
        latent_patch: torch.Tensor,
    ) -> torch.Tensor:
        return self.native_model._encode_audio_patch_feedback(
            data.fm_state, audio_patch=latent_patch
        )

    def step_audio_latent(
        self,
        data: Any,
        hidden_state: torch.Tensor,
    ) -> DotsTTSAudioStepResult:
        generation_kwargs = data.generation_kwargs
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

        payload_patch = self.native_model.core.io_helper.denormalize(latent_patch)
        return DotsTTSAudioStepResult(
            latent_patch=_as_tensor(payload_patch),
            feedback_embedding=_as_tensor(feedback_embedding),
            eos_score=self._score_audio_eos(data, latent_patch),
        )

    def _score_audio_eos(
        self,
        data: Any,
        latent_patch: torch.Tensor,
    ) -> torch.Tensor:
        eos_threshold = float(data.generation_kwargs.get("eos_threshold", 0.8))
        stopped = self.native_model._should_stop_after_current_audio(
            data.fm_state, eos_threshold=eos_threshold
        )
        return torch.tensor(
            [1.0 if stopped else 0.0],
            device=latent_patch.device,
        )

    def forward_latent_decode_step(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        requests: list[Any],
        input_embeds: torch.Tensor | None = None,
    ) -> DotsTTSLatentStepOutput:
        logits_output = self(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        hidden_rows = self._extract_request_hidden_states(
            logits_output,
            requests=requests,
        )
        active_indices = [
            index
            for index, sched_req in enumerate(requests)
            if self._can_run_latent_step(sched_req.data, hidden_rows[index])
        ]
        latent_output = self.decode_audio_batch(
            DotsTTSLatentBatch(
                requests=requests,
                active_indices=active_indices,
                hidden_states=hidden_rows,
            )
        )
        next_token_ids = latent_output.next_token_ids.to(device=input_ids.device)
        latent_output.next_token_ids = next_token_ids
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
        batch_result.next_token_ids = next_token_ids
        latent_output.batch_result = batch_result
        batch_result.dots_tts_latent_output = latent_output
        return batch_result.dots_tts_latent_output

    def decode_audio_batch(
        self,
        batch: DotsTTSLatentBatch,
    ) -> DotsTTSLatentStepOutput:
        device = self._latent_batch_output_device(batch)
        next_token_ids = torch.tensor(
            [int(sched_req.data.control_token_id) for sched_req in batch.requests],
            dtype=torch.long,
            device=device,
        )
        latent_patches: list[torch.Tensor | None] = [None for _ in batch.requests]
        feedback_embeddings: list[torch.Tensor | None] = [None for _ in batch.requests]
        eos_scores: list[torch.Tensor | None] = [None for _ in batch.requests]
        finished = [False for _ in batch.requests]

        for index in batch.active_indices:
            data = batch.requests[index].data
            hidden_state = batch.hidden_states[index]
            if hidden_state is None:
                continue
            audio_step = self.step_audio_latent(data, hidden_state)
            latent_patches[index] = audio_step.latent_patch
            feedback_embeddings[index] = audio_step.feedback_embedding
            eos_scores[index] = audio_step.eos_score
            finished[index] = self._latent_step_finished(data, audio_step.eos_score)

        return DotsTTSLatentStepOutput(
            batch_result=None,
            next_token_ids=next_token_ids,
            latent_patches=latent_patches,
            feedback_embeddings=feedback_embeddings,
            eos_scores=eos_scores,
            finished=finished,
            hidden_states=batch.hidden_states,
        )

    @staticmethod
    def _can_run_latent_step(data: Any, hidden_state: torch.Tensor | None) -> bool:
        return (
            getattr(data, "finish_reason", None) is None
            and getattr(data, "fm_state", None) is not None
            and hidden_state is not None
        )

    def _latent_batch_output_device(self, batch: DotsTTSLatentBatch) -> torch.device:
        for hidden_state in batch.hidden_states:
            if hidden_state is not None:
                return hidden_state.device
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _extract_request_hidden_states(
        logits_output: Any,
        *,
        requests: list[Any],
    ) -> list[torch.Tensor | None]:
        hidden_states = getattr(logits_output, "hidden_states", None)
        if hidden_states is None:
            nested = getattr(logits_output, "logits_output", None)
            hidden_states = getattr(nested, "hidden_states", None)
        if hidden_states is None:
            return [None for _ in requests]
        if hidden_states.ndim == 3:
            return [
                hidden_states[index : index + 1, -1:, :]
                for index in range(len(requests))
            ]
        if hidden_states.ndim == 2:
            return [
                hidden_states[index : index + 1].unsqueeze(0)
                for index in range(len(requests))
            ]
        raise RuntimeError(
            f"dots TTS expected 2D or 3D hidden states, got {hidden_states.ndim}D"
        )

    @staticmethod
    def _latent_step_finished(data: Any, eos_score: torch.Tensor | None) -> bool:
        if eos_score is not None and bool((eos_score > 0.5).any()):
            return True
        latent_patches = getattr(data, "latent_patches", [])
        max_generate_length = int(getattr(data, "max_generate_length", 500))
        return len(latent_patches) + 1 >= max_generate_length

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


__all__ = [
    "DotsTTSLatentBatch",
    "DotsTTSLatentStepOutput",
    "DotsTTSSGLangModel",
    "EntryClass",
]
