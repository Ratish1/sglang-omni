# SPDX-License-Identifier: Apache-2.0
"""SGLang registry entry for native dots TTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult
from torch import nn

from sglang_omni.models.dots_tts.serving_types import DotsTTSFlowBatchItem
from sglang_omni.models.dots_tts.weight_loader import (
    map_dots_qwen2_key,
    validate_checkpoint_files,
)

try:
    from sglang.srt.models.qwen2 import Qwen2ForCausalLM
except ImportError:
    Qwen2ForCausalLM = None

if TYPE_CHECKING:
    from sglang_omni.models.dots_tts._vendored.models.dots_tts.model import DotsTtsModel


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
        grouped_items: dict[Any, list[DotsTTSFlowBatchItem]] = {}
        for index in batch.active_indices:
            data = batch.requests[index].data
            hidden_state = batch.hidden_states[index]
            if hidden_state is None:
                continue
            key = self.native_model.prepare_flow_batch_key(
                fm_state=data.fm_state,
                generation_kwargs=data.generation_kwargs,
                precision=str(self.precision),
            )
            grouped_items.setdefault(key, []).append(
                DotsTTSFlowBatchItem(
                    request_index=index,
                    fm_state=data.fm_state,
                    hidden_state=hidden_state,
                    generation_kwargs=data.generation_kwargs,
                )
            )

        for items in grouped_items.values():
            audio_batch = self.native_model.decode_audio_batch_step(
                items,
                precision=str(self.precision),
            )
            for row, request_index in enumerate(audio_batch.request_indices):
                data = batch.requests[request_index].data
                eos_score = audio_batch.eos_scores[row]
                latent_patches[request_index] = audio_batch.latent_patches[row]
                feedback_embeddings[request_index] = audio_batch.feedback_embeddings[
                    row
                ]
                eos_scores[request_index] = eos_score
                finished[request_index] = self._latent_step_finished(data, eos_score)

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
