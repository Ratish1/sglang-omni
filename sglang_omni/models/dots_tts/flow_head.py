# SPDX-License-Identifier: Apache-2.0
"""dots.tts-specific continuous-latent head for an SGLang Qwen2 backbone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch import nn


@dataclass
class DotsFlowState:
    fm_sequence: torch.Tensor
    fm_cfg_sequence: torch.Tensor
    fm_null_g_cond: torch.Tensor
    fm_capacity: int
    g_cond: torch.Tensor | None
    fm_seq_len: int = 0
    patch_encoder_state: Any = None
    dit_state: Any = None
    prompt_patches: torch.Tensor | None = None
    drop_regenerated_prompt_patch: bool = False
    suppress_first_eos_check: bool = False
    decoded_patches: int = 0


@dataclass(frozen=True)
class DotsFlowStep:
    latent_patch: torch.Tensor
    feedback_embedding: torch.Tensor
    finished: bool
    emit: bool


class DotsTTSFlowHead(nn.Module):
    """Patch encoder, Flow/MeanFlow DiT and EOS head from the dots model."""

    _LENGTH_BUCKETS = (64, 128, 256, 512)

    def __init__(
        self,
        config_dict: dict[str, Any],
        *,
        llm_hidden_size: int,
        latent_stats_path: str,
        optimize: bool,
    ) -> None:
        super().__init__()
        from dots_tts.models.dots_tts.config import ModelConfig
        from dots_tts.models.dots_tts.core import IOHelper
        from dots_tts.modules.backbone.dit import DiT
        from dots_tts.modules.backbone.encoder import VAESemanticEncoder

        config = ModelConfig.model_validate(config_dict)
        self.config = config
        self.fm_hidden_size = int(config.DiT.hidden_size)
        self.hidden_patch_size = 1
        self.latent_patch_size = int(config.patch_size)
        self.latent_dim = int(config.latent_dim)
        self.optimize = bool(optimize)
        self.mode = (
            "meanflow"
            if config.meanflow is not None and config.meanflow.enabled
            else "flow_matching"
        )
        dit_mode = (
            "meanflow"
            if self.mode == "meanflow" and config.meanflow.use_duration_embedding
            else "flow_matching"
        )

        self.patch_encoder = VAESemanticEncoder(
            in_dim=self.latent_dim,
            out_dim=int(llm_hidden_size),
            config=config,
        )
        self.hidden_proj = nn.Linear(int(llm_hidden_size), self.fm_hidden_size)
        self.latent_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.coordinate_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.xvec_proj = nn.Sequential(
            nn.Linear(int(config.campplus_embedding_size), self.fm_hidden_size),
            nn.LayerNorm(self.fm_hidden_size),
        )
        self.velocity_field_predictor = DiT(
            in_dim=self.fm_hidden_size,
            out_dim=self.latent_dim,
            transformer_config=config.DiT,
            mode=dit_mode,
        )
        self.eos_proj = nn.Sequential(
            nn.Linear(int(llm_hidden_size), int(llm_hidden_size)),
            nn.SiLU(),
            nn.Linear(int(llm_hidden_size), 2),
        )
        self.io = IOHelper(Path(latent_stats_path))
        self._patch_inference: Any = None
        self._dit_solver: Any = None

    def _bucket(self, requested: int) -> int:
        if requested <= 0:
            raise ValueError("dots.tts max audio patch count must be positive")
        for bucket in self._LENGTH_BUCKETS:
            if requested <= bucket:
                return bucket
        raise ValueError(
            f"dots.tts supports at most {self._LENGTH_BUCKETS[-1]} audio patches"
        )

    def _patch_encoder_inference(self):
        if self._patch_inference is None:
            from dots_tts.modules.backbone.encoder_inference import (
                SemanticEncoderInference,
            )

            self._patch_inference = SemanticEncoderInference(self.patch_encoder)
        return self._patch_inference

    def _solver(self):
        if self._dit_solver is None:
            from dots_tts.modules.backbone.dit_inference import (
                DiTInferenceContext,
                DiTSolver,
            )

            self._dit_solver = DiTSolver(
                DiTInferenceContext.from_core(self),
                optimize=self.optimize,
                bucket_resolver=self._bucket,
                meanflow=self.mode == "meanflow",
            )
        return self._dit_solver

    @torch.inference_mode()
    def new_request(
        self,
        *,
        max_audio_patch_count: int,
        prompt_latents: torch.Tensor | None,
        speaker_embedding: torch.Tensor | None,
        speaker_scale: float,
    ) -> tuple[DotsFlowState, torch.Tensor | None]:
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        capacity_patches = (
            self._bucket(max_audio_patch_count)
            if self.optimize
            else int(max_audio_patch_count)
        )
        fm_capacity = capacity_patches * (
            self.hidden_patch_size + self.latent_patch_size
        )
        g_cond = None
        if speaker_embedding is not None:
            speaker_embedding = speaker_embedding.to(device=device, dtype=dtype)
            g_cond = self.xvec_proj(speaker_embedding * float(speaker_scale))
        state = DotsFlowState(
            fm_sequence=torch.zeros(
                (1, fm_capacity, self.fm_hidden_size), device=device, dtype=dtype
            ),
            fm_cfg_sequence=torch.zeros(
                (1, fm_capacity, self.fm_hidden_size), device=device, dtype=dtype
            ),
            fm_null_g_cond=torch.zeros(
                (1, self.fm_hidden_size), device=device, dtype=dtype
            ),
            fm_capacity=fm_capacity,
            g_cond=g_cond,
        )
        if prompt_latents is None or prompt_latents.numel() == 0:
            return state, None

        prompt_latents = prompt_latents.to(device=device, dtype=dtype)
        patch_input = self._patch_encoder_input(prompt_latents)
        (
            prompt_embeddings,
            state.patch_encoder_state,
        ) = self._patch_encoder_inference().prefill_with_state(
            patch_input,
            None,
            optimize=self.optimize,
            bucket_resolver=self._bucket,
            dtype=dtype,
        )
        state.prompt_patches = rearrange(
            self.io.normalize(prompt_latents).to(dtype=dtype),
            "b (s p) d -> b s p d",
            p=self.latent_patch_size,
        )
        state.drop_regenerated_prompt_patch = True
        state.suppress_first_eos_check = True
        return state, prompt_embeddings

    @torch.inference_mode()
    def initialize_history(
        self,
        state: DotsFlowState,
        *,
        hidden_states: torch.Tensor,
        prompt_span_positions: torch.Tensor,
        audio_span_token_ids: set[int],
        generation_schedule: torch.Tensor,
        prefill_end: int,
    ) -> None:
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        prompt_patches = state.prompt_patches
        cursor = 0
        for prompt_index, span_position in enumerate(
            prompt_span_positions.detach().cpu().tolist()
        ):
            if span_position > cursor:
                self.append_hidden(
                    state, hidden_states[:, span_position - 1 : span_position]
                )
            if prompt_patches is None:
                raise RuntimeError("dots.tts prompt spans require prompt latents")
            self._append_history(state, prompt_patches[:, prompt_index])
            next_position = span_position + 1
            if (
                next_position < generation_schedule.shape[1]
                and int(generation_schedule[0, next_position]) in audio_span_token_ids
            ):
                self.append_hidden(
                    state, hidden_states[:, span_position : span_position + 1]
                )
            cursor = next_position
        if prefill_end > cursor:
            self.append_hidden(state, hidden_states[:, prefill_end - 1 : prefill_end])

    @torch.inference_mode()
    def append_hidden(self, state: DotsFlowState, hidden_states: torch.Tensor) -> None:
        hidden = hidden_states[:, -self.hidden_patch_size :]
        projected = self.hidden_proj(hidden)
        null_projected = self.hidden_proj(torch.zeros_like(hidden))
        start, end = self._reserve(state, projected.shape[1])
        state.fm_sequence[:, start:end].copy_(projected)
        state.fm_cfg_sequence[:, start:end].copy_(null_projected)
        state.fm_seq_len = end

    @torch.inference_mode()
    def decode_next(
        self,
        state: DotsFlowState,
        *,
        hidden_states: torch.Tensor,
        num_steps: int,
        ode_method: str,
        guidance_scale: float,
        eos_threshold: float,
    ) -> DotsFlowStep:
        should_check_eos = not (
            state.suppress_first_eos_check and state.decoded_patches == 0
        )
        finished = should_check_eos and bool(
            (
                self.eos_proj(hidden_states)
                .softmax(dim=-1)[:, -1, 1]
                .gt(float(eos_threshold))
                .item()
            )
        )
        from dots_tts.modules.backbone.dit_inference import DiTSolverState

        if state.dit_state is None:
            state.dit_state = DiTSolverState()
        dtype = state.fm_sequence.dtype
        device_type = state.fm_sequence.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=device_type == "cuda" and dtype in {torch.float16, torch.bfloat16},
        ):
            normalized_patch = self._solver().decode_next(
                state.dit_state,
                sequence=state.fm_sequence,
                cfg_sequence=state.fm_cfg_sequence,
                fm_seq_len=state.fm_seq_len,
                null_g_cond=state.fm_null_g_cond,
                g_cond=state.g_cond,
                nfe=int(num_steps),
                ode_method=str(ode_method),
                guidance_scale=float(guidance_scale),
            )
        self._append_history(state, normalized_patch)
        (
            feedback,
            state.patch_encoder_state,
        ) = self._patch_encoder_inference().decode_patch_with_state(
            self._patch_encoder_input(normalized_patch, already_normalized=True),
            state.patch_encoder_state,
            optimize=self.optimize,
            bucket_resolver=self._bucket,
            dtype=state.fm_sequence.dtype,
        )
        emit = not state.drop_regenerated_prompt_patch
        state.drop_regenerated_prompt_patch = False
        state.decoded_patches += 1
        return DotsFlowStep(
            latent_patch=self.io.denormalize(normalized_patch),
            feedback_embedding=feedback,
            finished=finished,
            emit=emit,
        )

    def _patch_encoder_input(
        self, latents: torch.Tensor, *, already_normalized: bool = False
    ) -> torch.Tensor:
        if self.patch_encoder.expects_normalized_input:
            value = latents if already_normalized else self.io.normalize(latents)
        else:
            value = self.io.denormalize(latents) if already_normalized else latents
        return value.to(dtype=next(self.patch_encoder.parameters()).dtype)

    def _append_history(self, state: DotsFlowState, latent_patch: torch.Tensor) -> None:
        projected = self.latent_proj(latent_patch)
        start, end = self._reserve(state, projected.shape[1])
        state.fm_sequence[:, start:end].copy_(projected)
        state.fm_cfg_sequence[:, start:end].copy_(projected)
        state.fm_seq_len = end

    @staticmethod
    def _reserve(state: DotsFlowState, length: int) -> tuple[int, int]:
        start = int(state.fm_seq_len)
        end = start + int(length)
        if end > state.fm_capacity:
            raise RuntimeError(
                f"dots.tts flow history exceeded capacity ({end}>{state.fm_capacity})"
            )
        return start, end


__all__ = ["DotsFlowState", "DotsFlowStep", "DotsTTSFlowHead"]
