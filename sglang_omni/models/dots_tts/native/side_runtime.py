# SPDX-License-Identifier: Apache-2.0
"""Side-only dots TTS runtime used by the SGLang-backed latent engine."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import librosa
import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer, Qwen2Config

from sglang_omni.models.dots_tts.native.data.pipelines.templates import (
    DEFAULT_INSTRUCTION_TTS_TEMPLATE,
    DEFAULT_INTERLEAVE_TTS_TEMPLATE,
    DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
    DEFAULT_TTS_TEMPLATE,
)
from sglang_omni.models.dots_tts.native.data.pipelines.tokenizing import (
    build_generation_schedule,
)
from sglang_omni.models.dots_tts.native.models.dots_tts.config import ModelConfig
from sglang_omni.models.dots_tts.native.models.dots_tts.core import (
    CausalHelper,
    DotsTtsCore,
    FlowMatchingHelper,
    IOHelper,
)
from sglang_omni.models.dots_tts.native.models.dots_tts.model import (
    DotsTtsModel,
    _PromptFeatureCacheEntry,
)
from sglang_omni.models.dots_tts.native.modules.backbone.dit import DiT
from sglang_omni.models.dots_tts.native.modules.backbone.semantic_encoder import (
    VAESemanticEncoder,
)
from sglang_omni.models.dots_tts.native.modules.speaker.encoder import (
    SpeakerXVectorFeatures,
)
from sglang_omni.models.dots_tts.native.modules.vocoder.bigvgan import AudioVAE
from sglang_omni.models.dots_tts.native.utils.audio import high_quality_resample
from sglang_omni.models.dots_tts.native.utils.profiling import measure_inference
from sglang_omni.models.dots_tts.native.utils.text import (
    attach_language_tag,
    detect,
    normalize_language_code,
    normalize_text,
)
from sglang_omni.models.dots_tts.native.utils.tokenizer import (
    AUDIO_COMP_SPAN_TOKEN,
    AUDIO_GEN_SPAN_TOKEN,
    AUDIO_GEN_START_TOKEN,
    TEXT_COND_END_TOKEN,
    require_token_id,
)
from sglang_omni.models.dots_tts.native.utils.util import get_dtype
from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.serving_types import (
    DotsTTSAudioStepResult,
    DotsTTSBatchedAudioStepResult,
    DotsTTSFlowBatchItem,
    DotsTTSFlowBatchKey,
    DotsTTSPreparedInputs,
    as_tensor,
    torch_dtype,
)

RUNTIME_TEMPLATE_BY_NAME = {
    "tts": DEFAULT_TTS_TEMPLATE,
    "instruction_tts": DEFAULT_INSTRUCTION_TTS_TEMPLATE,
    "text_to_audio": DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
    "tts_interleave": DEFAULT_INTERLEAVE_TTS_TEMPLATE,
}


class RuntimeInputs(TypedDict, total=False):
    fid: str
    language: str
    text: str
    prompt_text: str
    template_name: str
    generation_schedule: torch.Tensor
    prompt_audio: torch.Tensor


@dataclass(frozen=True)
class DotsTtsSideModuleBundle:
    model: "DotsTtsSideModel"
    core: DotsTtsCore
    xvector_extractor: SpeakerXVectorFeatures
    tokenizer: Any
    config: ModelConfig
    llm_config: Qwen2Config


class DotsTtsSideCore(nn.Module):
    """Dots DiT/patch/speaker-side core without the Qwen2 LLM."""

    fm_solver_step = DotsTtsCore.fm_solver_step
    meanflow_solver_step = DotsTtsCore.meanflow_solver_step
    _flow_matching_step_fm = DotsTtsCore._flow_matching_step_fm
    _meanflow_step_fm = DotsTtsCore._meanflow_step_fm
    step_fm = DotsTtsCore.step_fm

    def __init__(
        self,
        config: ModelConfig,
        *,
        tokenizer: Any,
        latent_stats_path: str | Path,
        llm_hidden_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.fm_hidden_size = config.DiT.hidden_size
        self.hidden_patch_size = 1
        self.cfg_droprate = config.get("cfg_droprate", 0.2)
        self.latent_patch_size = config.patch_size
        self.latent_dim = config.latent_dim
        self.xvec_dim = config.campplus_embedding_size
        self.xvec_drop_rate = config.get("xvec_drop_rate", 0.2)
        self.tokenizer = tokenizer
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self.audio_gen_span_id = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)
        self.audio_comp_span_id = require_token_id(tokenizer, AUDIO_COMP_SPAN_TOKEN)
        self.text_cond_end_id = require_token_id(tokenizer, TEXT_COND_END_TOKEN)
        self.llm_hidden_size = int(llm_hidden_size)

        self.patch_encoder = VAESemanticEncoder(
            in_dim=self.latent_dim,
            out_dim=self.llm_hidden_size,
            config=config,
        )
        self.hidden_proj = nn.Linear(self.llm_hidden_size, self.fm_hidden_size)
        self.latent_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.coordinate_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.xvec_proj = nn.Sequential(
            nn.Linear(self.xvec_dim, self.fm_hidden_size),
            nn.LayerNorm(self.fm_hidden_size),
        )
        self.meanflow_config = config.meanflow if config.meanflow is not None else None
        self.mode = (
            "meanflow"
            if self.meanflow_config is not None and self.meanflow_config.enabled
            else "flow_matching"
        )
        dit_mode = (
            "meanflow"
            if self.mode == "meanflow" and self.meanflow_config.use_duration_embedding
            else "flow_matching"
        )
        self.velocity_field_predictor = DiT(
            in_dim=self.fm_hidden_size,
            out_dim=self.latent_dim,
            transformer_config=config.DiT,
            mode=dit_mode,
        )
        self.eos_proj = nn.Sequential(
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
            nn.SiLU(),
            nn.Linear(self.llm_hidden_size, 2),
        )
        self.fm_helper = FlowMatchingHelper(sigma=config.get("fm_sigma", 0.0))
        self.causal_helper = CausalHelper()
        self.io_helper = IOHelper(latent_stats_path=latent_stats_path)
        self.audio_span_token_ids = [
            self.audio_gen_span_id,
            self.audio_comp_span_id,
        ]


class DotsTtsSideModel(nn.Module):
    """Dots inference side modules without an owned Qwen2 backbone."""

    _GENERATE_LENGTH_BUCKETS = DotsTtsModel._GENERATE_LENGTH_BUCKETS
    _COMPILE_TARGETS = DotsTtsModel._COMPILE_TARGETS
    _PROMPT_FEATURE_CACHE_MAX_ENTRIES = DotsTtsModel._PROMPT_FEATURE_CACHE_MAX_ENTRIES
    CONFIG_FILENAME = DotsTtsModel.CONFIG_FILENAME
    LATENT_STATS_FILENAME = DotsTtsModel.LATENT_STATS_FILENAME
    LLM_CONFIG_FILENAME = DotsTtsModel.LLM_CONFIG_FILENAME
    MODEL_FILENAME = DotsTtsModel.MODEL_FILENAME
    VOCODER_FILENAME = DotsTtsModel.VOCODER_FILENAME
    SPEAKER_ENCODER_FILENAME = DotsTtsModel.SPEAKER_ENCODER_FILENAME
    REQUIRED_ARTIFACT_FILES = DotsTtsModel.REQUIRED_ARTIFACT_FILES

    set_optimize = DotsTtsModel.set_optimize
    set_cfg_droprate = DotsTtsModel.set_cfg_droprate
    _resolve_generate_length_bucket = classmethod(
        DotsTtsModel._resolve_generate_length_bucket.__func__
    )
    _resolve_state_audio_patch_count = DotsTtsModel._resolve_state_audio_patch_count
    _warmup_fm_bucket = DotsTtsModel._warmup_fm_bucket
    _warmup_patch_encoder_bucket = DotsTtsModel._warmup_patch_encoder_bucket
    _compile_callable = DotsTtsModel._compile_callable
    _get_compiled_model = DotsTtsModel._get_compiled_model
    _get_compiled_method = DotsTtsModel._get_compiled_method
    _allocate_generate_state = DotsTtsModel._allocate_generate_state
    _allocate_fm_state_buffers = DotsTtsModel._allocate_fm_state_buffers
    _prepare_prompt_audio_for_conditioning = (
        DotsTtsModel._prepare_prompt_audio_for_conditioning
    )
    _get_prompt_feature_cache_entry = DotsTtsModel._get_prompt_feature_cache_entry
    _store_prompt_feature_cache_entry = DotsTtsModel._store_prompt_feature_cache_entry
    _can_cache_speaker_embedding = DotsTtsModel._can_cache_speaker_embedding
    _prepare_prompt_conditioning = DotsTtsModel._prepare_prompt_conditioning
    _patch_encoder_compile_signature = staticmethod(
        DotsTtsModel._patch_encoder_compile_signature
    )
    _resolve_patch_encoder_audio_bucket = (
        DotsTtsModel._resolve_patch_encoder_audio_bucket
    )
    _copy_patch_encoder_state = DotsTtsModel._copy_patch_encoder_state
    _ensure_patch_encoder_state_capacity = (
        DotsTtsModel._ensure_patch_encoder_state_capacity
    )
    _prefill_prompt_latents = DotsTtsModel._prefill_prompt_latents
    _get_fm_decode_workspace = DotsTtsModel._get_fm_decode_workspace
    _resolve_fm_history_bucket_capacity = (
        DotsTtsModel._resolve_fm_history_bucket_capacity
    )
    _build_fm_attn_mask = DotsTtsModel._build_fm_attn_mask
    _build_fm_pos_ids = DotsTtsModel._build_fm_pos_ids
    _prepare_fm_decode_inputs = DotsTtsModel._prepare_fm_decode_inputs
    _append_to_fm_buffer = DotsTtsModel._append_to_fm_buffer
    _append_hidden_chunk = DotsTtsModel._append_hidden_chunk
    _append_history_chunk = DotsTtsModel._append_history_chunk
    _locate_prefill_boundary = DotsTtsModel._locate_prefill_boundary
    _find_audio_span_positions = staticmethod(DotsTtsModel._find_audio_span_positions)
    _next_token_is_audio_span = staticmethod(DotsTtsModel._next_token_is_audio_span)
    _decode_next_audio = DotsTtsModel._decode_next_audio
    _encode_audio_patch_feedback = DotsTtsModel._encode_audio_patch_feedback
    _should_stop_after_current_audio = DotsTtsModel._should_stop_after_current_audio
    _validate_pretrained_directory = classmethod(
        DotsTtsModel._validate_pretrained_directory.__func__
    )

    @torch.no_grad()
    def run_warmup(
        self,
        *,
        max_generate_length: int,
        precision: str = "bfloat16",
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 1.2,
    ) -> None:
        ceiling_bucket = self._resolve_generate_length_bucket(max_generate_length)
        warmup_buckets = tuple(
            bucket
            for bucket in self._GENERATE_LENGTH_BUCKETS
            if bucket.size <= ceiling_bucket.size
        )
        bucket_sizes = [bucket.size for bucket in warmup_buckets]
        logger.info(
            "Side inference warmup started: requested_max_generate_length={} bucket_sizes={}",
            int(max_generate_length),
            bucket_sizes,
        )
        for bucket in warmup_buckets:
            self._warmup_fm_bucket(
                max_audio_patch_count=bucket.size,
                precision=precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
            )
            self._warmup_patch_encoder_bucket(
                max_audio_patch_count=bucket.size,
                precision=precision,
            )
        logger.info(
            "Side inference warmup completed: requested_max_generate_length={} bucket_sizes={}",
            int(max_generate_length),
            bucket_sizes,
        )

    def __init__(
        self,
        config: ModelConfig,
        *,
        tokenizer: Any,
        latent_stats_path: str | Path,
        llm_config: Qwen2Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.llm_config = llm_config
        self.tokenizer = tokenizer
        self.latent_stats_path = Path(latent_stats_path)
        self.audio_gen_start_id = require_token_id(tokenizer, AUDIO_GEN_START_TOKEN)
        self.core = DotsTtsSideCore(
            config,
            tokenizer=tokenizer,
            latent_stats_path=self.latent_stats_path,
            llm_hidden_size=llm_config.hidden_size,
        )
        self.vocoder = AudioVAE(config.vocoder).eval()
        self.vocoder.remove_weight_norm()
        self.hop_size = self.vocoder.hop_size
        self.xvector_extractor = SpeakerXVectorFeatures(
            sample_rate=self.vocoder.sample_rate,
            campplus_embedding_size=config.campplus_embedding_size,
            max_audio_seconds=config.xvec_max_audio_seconds,
        ).eval()
        for param in self.vocoder.parameters():
            param.requires_grad = False
        for param in self.xvector_extractor.parameters():
            param.requires_grad = False
        self._optimize_enabled = True
        self._compiled_models: dict[tuple[str, tuple[Any, ...] | None], Any] = {}
        self._prompt_feature_cache: OrderedDict[str, _PromptFeatureCacheEntry] = (
            OrderedDict()
        )
        self._fm_decode_workspaces: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
        self._fm_batch_decode_workspaces: dict[
            tuple[DotsTTSFlowBatchKey, int],
            dict[str, torch.Tensor],
        ] = {}
        self._token_embedding: nn.Module | None = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path):
        pretrained_path = cls._validate_pretrained_directory(
            pretrained_model_name_or_path
        )
        config = ModelConfig.model_validate(
            json.loads(
                (pretrained_path / cls.CONFIG_FILENAME).read_text(encoding="utf-8")
            )
        )
        llm_config = Qwen2Config.from_dict(
            json.loads(
                (pretrained_path / cls.LLM_CONFIG_FILENAME).read_text(encoding="utf-8")
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(pretrained_path),
            local_files_only=True,
        )
        model = cls(
            config,
            tokenizer=tokenizer,
            latent_stats_path=pretrained_path / cls.LATENT_STATS_FILENAME,
            llm_config=llm_config,
        )
        model._load_pretrained_artifacts(pretrained_path)
        return model.eval()

    def set_token_embedding(self, token_embedding: nn.Module) -> None:
        self._token_embedding = token_embedding

    def _build_prefill_inputs_embeds(
        self,
        generation_schedule: torch.Tensor,
        *,
        prompt_patch_embeddings: torch.Tensor | None,
        prompt_span_positions: torch.Tensor,
    ) -> torch.Tensor:
        if self._token_embedding is None:
            raise RuntimeError("SGLang token embedding is not attached.")
        inputs_embeds = self._token_embedding(generation_schedule).clone()
        if prompt_span_positions.numel() > 0:
            if prompt_patch_embeddings is None:
                raise RuntimeError(
                    "Prompt patch embeddings are required when prefill includes prompt audio spans."
                )
            patch_embeddings = prompt_patch_embeddings[
                :, : prompt_span_positions.numel()
            ].to(inputs_embeds.dtype)
            if patch_embeddings.size(1) != prompt_span_positions.numel():
                raise RuntimeError(
                    f"Prompt patch embeddings ({patch_embeddings.size(1)}) do not match prompt span count ({prompt_span_positions.numel()})."
                )
            inputs_embeds[:, prompt_span_positions, :] = patch_embeddings
        return inputs_embeds

    def prepare_request(
        self,
        raw_inputs: dict[str, Any],
        state: DotsTTSState,
        *,
        generation_schedule: torch.Tensor,
        precision: str,
    ) -> DotsTTSPreparedInputs:
        device = next(self.core.parameters()).device
        dtype = torch_dtype(precision)
        generation_schedule = generation_schedule.to(device=device, dtype=torch.long)
        if generation_schedule.ndim == 1:
            generation_schedule = generation_schedule.unsqueeze(0)

        use_prompt_prefill = raw_inputs.get("prompt_audio") is not None and bool(
            raw_inputs.get("prompt_text")
        )
        prompt_conditioning = self._prepare_prompt_conditioning(
            raw_inputs.get("prompt_audio"),
            use_prompt_prefill=use_prompt_prefill,
            speaker_scale=state.speaker_scale,
        )
        prompt_patches = prompt_conditioning.prompt_patches
        prompt_patch_count = (
            0 if prompt_patches is None else int(prompt_patches.size(1))
        )
        audio_placeholder_ids = set(self.core.audio_span_token_ids)
        span_positions = self._find_audio_span_positions(
            generation_schedule,
            audio_placeholder_ids=audio_placeholder_ids,
        )
        span_count = int(span_positions.numel())
        minimum_required_spans = prompt_patch_count + 1
        if span_count < minimum_required_spans:
            raise ValueError(
                f"generation_schedule provides {span_count} audio spans, but "
                f"prompt prefill requires {prompt_patch_count} spans and generation "
                "requires at least one additional decode span."
            )
        fm_state = self._allocate_generate_state(
            max_audio_patch_count=span_count,
            device=device,
            dtype=dtype,
        )
        prompt_latents = prompt_conditioning.prompt_latents
        if prompt_latents is not None:
            prompt_latents = prompt_latents.to(dtype=fm_state.fm_sequence.dtype)
        prompt_patch_embeddings = self._prefill_prompt_latents(
            prompt_latents,
            state=fm_state,
        )
        prefill_end, prompt_span_positions = self._locate_prefill_boundary(
            span_positions=span_positions,
            prompt_patch_count=prompt_patch_count,
        )
        input_ids = generation_schedule[:, :prefill_end]
        prompt_input_embeds = None
        if prompt_span_positions.numel() > 0:
            prompt_input_embeds = self._build_prefill_inputs_embeds(
                input_ids,
                prompt_patch_embeddings=prompt_patch_embeddings,
                prompt_span_positions=prompt_span_positions,
            )
        raw_inputs["prompt_conditioning"] = prompt_conditioning
        raw_inputs["fm_state"] = fm_state
        raw_inputs["audio_span_positions"] = span_positions
        raw_inputs["prefill_end"] = prefill_end
        raw_inputs["prompt_span_positions"] = prompt_span_positions
        raw_inputs["audio_placeholder_ids"] = audio_placeholder_ids
        return DotsTTSPreparedInputs(
            raw_inputs=raw_inputs,
            input_ids=input_ids,
            generation_schedule=generation_schedule,
            audio_span_positions=span_positions,
            prefill_end=prefill_end,
            audio_placeholder_ids=audio_placeholder_ids,
            prompt_patch_embeddings=prompt_input_embeds,
            prompt_conditioning=prompt_conditioning,
            fm_state=fm_state,
            generation_kwargs=self._generation_kwargs(
                state,
                prompt_conditioning=prompt_conditioning,
                fm_state=fm_state,
            ),
        )

    def append_hidden(
        self,
        fm_state: Any,
        hidden_state: torch.Tensor,
    ) -> None:
        self._append_hidden_chunk(fm_state, hidden_state)

    def decode_audio_step(
        self,
        *,
        fm_state: Any,
        generation_kwargs: dict[str, Any],
        hidden_state: torch.Tensor,
        precision: str,
    ) -> DotsTTSAudioStepResult:
        decode_kwargs = {
            key: value
            for key, value in generation_kwargs.items()
            if key in {"device", "g_cond", "ode_method", "num_steps", "guidance_scale"}
        }
        self.append_hidden(fm_state, hidden_state)
        device = decode_kwargs.get("device")
        dtype = torch_dtype(str(precision))
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
            latent_patch = as_tensor(
                self._decode_next_audio(
                    state=fm_state,
                    **decode_kwargs,
                )
            )
            feedback_embedding = self._encode_audio_patch_feedback(
                fm_state,
                audio_patch=latent_patch,
            )

        payload_patch = self.core.io_helper.denormalize(latent_patch)
        eos_threshold = float(generation_kwargs.get("eos_threshold", 0.8))
        stopped = self._should_stop_after_current_audio(
            fm_state,
            eos_threshold=eos_threshold,
        )
        eos_score = torch.tensor(
            [1.0 if stopped else 0.0],
            device=latent_patch.device,
        )
        return DotsTTSAudioStepResult(
            latent_patch=as_tensor(payload_patch),
            feedback_embedding=as_tensor(feedback_embedding),
            eos_score=eos_score,
        )

    def prepare_flow_batch_key(
        self,
        *,
        fm_state: Any,
        generation_kwargs: dict[str, Any],
        precision: str,
    ) -> DotsTTSFlowBatchKey:
        del precision
        if fm_state.fm_sequence is None or fm_state.fm_cfg_sequence is None:
            raise RuntimeError("FM static buffers are not initialized.")
        if fm_state.fm_null_g_cond is None:
            raise RuntimeError("FM null conditioning buffer is not initialized.")

        post_hidden_seq_len = int(fm_state.fm_seq_len) + int(
            self.core.hidden_patch_size
        )
        if post_hidden_seq_len <= 0:
            raise RuntimeError(
                "Cannot batch decode audio before any conditioning state has been prefetched."
            )
        history_bucket_capacity = self._resolve_fm_history_bucket_capacity(
            post_hidden_seq_len
        )
        return DotsTTSFlowBatchKey(
            device=fm_state.fm_sequence.device,
            dtype=fm_state.fm_sequence.dtype,
            mode=str(self.core.mode),
            ode_method=str(generation_kwargs["ode_method"]),
            num_steps=int(generation_kwargs["num_steps"]),
            guidance_scale=float(generation_kwargs["guidance_scale"]),
            history_bucket_capacity=history_bucket_capacity,
            latent_patch_size=int(self.core.latent_patch_size),
            hidden_patch_size=int(self.core.hidden_patch_size),
        )

    def decode_audio_batch_step(
        self,
        items: list[DotsTTSFlowBatchItem],
        *,
        precision: str,
    ) -> DotsTTSBatchedAudioStepResult:
        if not items:
            return DotsTTSBatchedAudioStepResult([], [], [], [])

        key = self.prepare_flow_batch_key(
            fm_state=items[0].fm_state,
            generation_kwargs=items[0].generation_kwargs,
            precision=precision,
        )
        for item in items:
            item_key = self.prepare_flow_batch_key(
                fm_state=item.fm_state,
                generation_kwargs=item.generation_kwargs,
                precision=precision,
            )
            if item_key != key:
                raise ValueError(
                    "decode_audio_batch_step received incompatible dots flow rows."
                )

        for item in items:
            self.append_hidden(item.fm_state, item.hidden_state)

        dtype = torch_dtype(str(precision))
        use_amp = key.device.type == "cuda" and dtype in {
            torch.float16,
            torch.bfloat16,
        }
        with torch.autocast(
            device_type=key.device.type,
            dtype=dtype,
            enabled=use_amp,
        ):
            latent_batch = self._decode_batched_audio_latents(items, key=key)
            feedback_batch = [
                self._encode_audio_patch_feedback(
                    item.fm_state,
                    audio_patch=latent_batch[row : row + 1],
                )
                for row, item in enumerate(items)
            ]

        latent_patches: list[torch.Tensor] = []
        feedback_embeddings: list[torch.Tensor] = []
        eos_scores: list[torch.Tensor] = []
        for row, item in enumerate(items):
            latent_patch = latent_batch[row : row + 1]
            payload_patch = self.core.io_helper.denormalize(latent_patch)
            eos_threshold = float(item.generation_kwargs.get("eos_threshold", 0.8))
            stopped = self._should_stop_after_current_audio(
                item.fm_state,
                eos_threshold=eos_threshold,
            )
            latent_patches.append(as_tensor(payload_patch))
            feedback_embeddings.append(as_tensor(feedback_batch[row]))
            eos_scores.append(
                torch.tensor(
                    [1.0 if stopped else 0.0],
                    device=latent_patch.device,
                )
            )

        return DotsTTSBatchedAudioStepResult(
            request_indices=[item.request_index for item in items],
            latent_patches=latent_patches,
            feedback_embeddings=feedback_embeddings,
            eos_scores=eos_scores,
        )

    def _decode_batched_audio_latents(
        self,
        items: list[DotsTTSFlowBatchItem],
        *,
        key: DotsTTSFlowBatchKey,
    ) -> torch.Tensor:
        workspace = self._get_fm_batch_decode_workspace(
            key=key,
            batch_size=len(items),
        )
        input_sequence = workspace["input_sequence"]
        cfg_sequence = workspace["cfg_sequence"]
        attn_mask = workspace["attn_mask"]
        pos_ids = workspace["pos_ids"]

        g_cond_rows = []
        for row, item in enumerate(items):
            state = item.fm_state
            input_sequence[row : row + 1, : state.fm_seq_len].copy_(
                state.fm_sequence[:, : state.fm_seq_len]
            )
            cfg_sequence[row : row + 1, : state.fm_seq_len].copy_(
                state.fm_cfg_sequence[:, : state.fm_seq_len]
            )
            self._build_fm_attn_mask(
                state=state,
                attn_mask=attn_mask[row : row + 1],
            )
            self._build_fm_pos_ids(
                state=state,
                pos_ids=pos_ids[row : row + 1],
            )
            g_cond_rows.append(
                self._resolve_batch_g_cond(
                    item.generation_kwargs.get("g_cond"),
                    state=state,
                )
            )

        batch_g_cond = torch.cat(g_cond_rows, dim=0)
        solver_step = None
        compile_signature = (
            (len(items), key.history_bucket_capacity, key.dtype)
            if self._optimize_enabled
            else None
        )
        if self.core.mode == "meanflow":
            solver_step = self._get_compiled_method(
                "FM.meanflow.solver_step",
                self.core,
                "meanflow_solver_step",
                signature=compile_signature,
            )
            step_attn_mask = attn_mask.unsqueeze(1)
            step_pos_ids = pos_ids
        else:
            solver_step = self._get_compiled_method(
                "FM.flow_matching.solver_step",
                self.core,
                "fm_solver_step",
                signature=compile_signature,
            )
            step_attn_mask = torch.cat([attn_mask, attn_mask], dim=0).unsqueeze(1)
            step_pos_ids = torch.cat([pos_ids, pos_ids], dim=0)

        with measure_inference("FM"):
            return self.core.step_fm(
                input_sequence=input_sequence,
                cfg_sequence=cfg_sequence,
                attn_mask=step_attn_mask,
                pos_ids=step_pos_ids,
                hidden_size=key.hidden_patch_size,
                patch_size=key.latent_patch_size,
                g_cond=batch_g_cond,
                ode_method=key.ode_method,
                num_steps=key.num_steps,
                guidance_scale=key.guidance_scale,
                solver_step=solver_step,
            )

    def _get_fm_batch_decode_workspace(
        self,
        *,
        key: DotsTTSFlowBatchKey,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        if not hasattr(self, "_fm_batch_decode_workspaces"):
            self._fm_batch_decode_workspaces = {}
        workspace_key = (key, int(batch_size))
        workspace = self._fm_batch_decode_workspaces.get(workspace_key)
        total_len = key.history_bucket_capacity + key.latent_patch_size
        if workspace is None:
            workspace = {
                "input_sequence": torch.zeros(
                    (batch_size, total_len, self.core.fm_hidden_size),
                    dtype=key.dtype,
                    device=key.device,
                ),
                "cfg_sequence": torch.zeros(
                    (batch_size, total_len, self.core.fm_hidden_size),
                    dtype=key.dtype,
                    device=key.device,
                ),
                "attn_mask": torch.zeros(
                    (batch_size, total_len, total_len),
                    dtype=torch.bool,
                    device=key.device,
                ),
                "pos_ids": torch.zeros(
                    (batch_size, total_len),
                    dtype=torch.float32,
                    device=key.device,
                ),
            }
            self._fm_batch_decode_workspaces[workspace_key] = workspace
        else:
            workspace["input_sequence"].zero_()
            workspace["cfg_sequence"].zero_()
            workspace["attn_mask"].zero_()
            workspace["pos_ids"].zero_()
        return workspace

    @staticmethod
    def _resolve_batch_g_cond(
        g_cond: torch.Tensor | None,
        *,
        state: Any,
    ) -> torch.Tensor:
        if g_cond is None:
            return state.fm_null_g_cond
        return g_cond.to(
            device=state.fm_null_g_cond.device,
            dtype=state.fm_null_g_cond.dtype,
        )

    @staticmethod
    def _generation_kwargs(
        state: DotsTTSState,
        *,
        prompt_conditioning: Any,
        fm_state: Any,
    ) -> dict[str, Any]:
        device = None
        if fm_state is not None and fm_state.fm_sequence is not None:
            device = fm_state.fm_sequence.device
        g_cond = getattr(prompt_conditioning, "g_cond", None)
        return {
            "device": device,
            "g_cond": g_cond,
            "ode_method": state.ode_method,
            "num_steps": state.num_steps,
            "guidance_scale": state.guidance_scale,
            "speaker_scale": state.speaker_scale,
            "eos_threshold": 0.8,
        }

    def _load_pretrained_artifacts(self, pretrained_path: Path) -> None:
        self.latent_stats_path = pretrained_path / self.LATENT_STATS_FILENAME
        self.core.io_helper = type(self.core.io_helper)(
            latent_stats_path=self.latent_stats_path
        )
        self._load_core_side_artifact(pretrained_path / self.MODEL_FILENAME)
        self._load_artifact_module(
            self.vocoder, pretrained_path / self.VOCODER_FILENAME
        )
        self._load_artifact_module(
            self.xvector_extractor,
            pretrained_path / self.SPEAKER_ENCODER_FILENAME,
        )
        self.core.eval()
        self.vocoder.eval()
        self.xvector_extractor.eval()

    def _load_core_side_artifact(self, path: Path) -> None:
        state_dict = {
            key: value
            for key, value in load_file(path, device="cpu").items()
            if not key.startswith("llm.")
        }
        mismatch = self.core.load_state_dict(state_dict, strict=False)
        if mismatch.missing_keys or mismatch.unexpected_keys:
            raise RuntimeError(f"Failed to load {path}: {mismatch}")

    @staticmethod
    def _load_artifact_module(module: nn.Module, path: Path) -> nn.Module:
        mismatch = module.load_state_dict(load_file(path, device="cpu"), strict=False)
        if mismatch.missing_keys or mismatch.unexpected_keys:
            raise RuntimeError(f"Failed to load {path}: {mismatch}")
        return module


class DotsTtsSideRuntime:
    """Request prep and side-module owner for SGLang-native dots latent decode."""

    def __init__(
        self,
        model: DotsTtsSideModel,
        pretrained_path: Path,
        *,
        precision: str = "bfloat16",
        optimize: bool = False,
        max_generate_length: int = 500,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.pretrained_path = pretrained_path
        self.precision = precision
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        target_dtype = get_dtype(self.precision)
        self.model.core.to(dtype=target_dtype)
        self.model = self.model.to(self.device).eval()
        self.optimize = bool(optimize)
        self.max_generate_length = int(max_generate_length)
        self.model.set_optimize(self.optimize)
        self.module_bundle = DotsTtsSideModuleBundle(
            model=self.model,
            core=self.model.core,
            xvector_extractor=self.model.xvector_extractor,
            tokenizer=self.model.tokenizer,
            config=self.model.config,
            llm_config=self.model.llm_config,
        )
        self.sample_rate = int(self.model.config.vocoder.sample_rate)
        if self.optimize:
            self.model.run_warmup(
                max_generate_length=self.max_generate_length,
                precision=self.precision,
            )
        logger.info(
            "Side runtime initialized: pretrained_path={} device={} precision={} optimize={}",
            self.pretrained_path,
            self.device,
            self.precision,
            self.optimize,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        precision: str = "bfloat16",
        optimize: bool = False,
        max_generate_length: int = 500,
        device: str | torch.device | None = None,
    ) -> "DotsTtsSideRuntime":
        pretrained_path = cls._resolve_pretrained_path(
            model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        model = DotsTtsSideModel.from_pretrained(pretrained_path)
        return cls(
            model,
            pretrained_path,
            precision=precision,
            optimize=optimize,
            max_generate_length=max_generate_length,
            device=device,
        )

    @staticmethod
    def _resolve_pretrained_path(
        model_name_or_path: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> Path:
        resolved_path = Path(model_name_or_path).expanduser().resolve()
        if resolved_path.exists():
            return resolved_path
        snapshot_dir = snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        return Path(snapshot_dir).expanduser().resolve()

    @staticmethod
    def _build_request_id(
        *,
        text: str,
        prompt_audio_path: str | None,
        prompt_text: str | None,
        template_name: str,
        language: str | None = None,
    ) -> str:
        payload = {
            "text": text,
            "prompt_audio_path": prompt_audio_path,
            "prompt_text": prompt_text,
            "template_name": template_name,
        }
        if language is not None:
            payload["language"] = language
        digest = hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    def _load_prompt_audio(self, prompt_audio_path: str) -> torch.Tensor:
        prompt_audio, sample_rate = librosa.load(prompt_audio_path, sr=None, mono=True)
        prompt_audio = librosa.effects.trim(prompt_audio, top_db=30)[0]
        prompt_audio = torch.from_numpy(prompt_audio).unsqueeze(0)
        prompt_audio = high_quality_resample(
            prompt_audio,
            orig_sr=sample_rate,
            target_sr=self.sample_rate,
        )
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        return prompt_audio

    def _resolve_language(self, language: str | None, *, text: str) -> str | None:
        if language is None:
            return None
        stripped = language.strip()
        if not stripped or stripped.lower() == "none":
            return None
        if stripped.lower() == "auto_detect":
            return normalize_language_code(detect(text))
        normalized_language = normalize_language_code(stripped)
        if normalized_language is None:
            raise ValueError(
                f"Unsupported language={language!r}. "
                "Expected 'none', 'auto_detect', or a valid language code/name."
            )
        return normalized_language

    def _process_prompt_text(
        self,
        prompt_text: str | None,
        *,
        language: str | None = None,
    ) -> str:
        if prompt_text is None:
            return ""
        prompt_text = prompt_text.strip()
        if not prompt_text:
            return ""
        prompt_text += "\n"
        if language is not None:
            prompt_text = attach_language_tag(prompt_text, language)
        return prompt_text

    def _process_text(
        self,
        text: str,
        *,
        language: str | None = None,
        normalize: bool = False,
    ) -> tuple[str, str | None]:
        stripped = text.strip()
        if normalize:
            stripped = normalize_text(stripped)
        resolved_language = self._resolve_language(language, text=stripped)
        return stripped, resolved_language

    def _estimate_prompt_audio_patch_count(
        self,
        *,
        prompt_audio: torch.Tensor | None,
        prompt_text: str,
    ) -> int:
        if prompt_audio is None or not prompt_text:
            return 0
        samples_per_patch = int(self.model.config.patch_size * self.model.hop_size)
        prompt_samples = int(prompt_audio.shape[-1])
        return math.ceil(prompt_samples / samples_per_patch)

    def _normalize_template_name(self, template_name: str | None) -> str:
        if template_name is None:
            return "tts"
        if template_name not in RUNTIME_TEMPLATE_BY_NAME:
            raise ValueError(
                f"Unknown template_name={template_name!r}. "
                f"Expected one of {sorted(RUNTIME_TEMPLATE_BY_NAME)}."
            )
        return template_name

    def _prepare_inputs(
        self,
        *,
        text: str,
        prompt_audio_path: str | None,
        prompt_text: str | None,
        template_name: str | None,
        language: str | None = None,
        normalize_text: bool = False,
    ) -> RuntimeInputs:
        normalized_template_name = self._normalize_template_name(template_name)
        template = RUNTIME_TEMPLATE_BY_NAME[normalized_template_name]
        if prompt_text and not prompt_audio_path:
            raise ValueError("prompt_text requires prompt_audio_path.")
        normalized_text, normalized_language = self._process_text(
            text,
            language=language,
            normalize=normalize_text,
        )
        normalized_prompt_text = self._process_prompt_text(
            prompt_text,
            language=normalized_language,
        )
        if normalized_language is not None and not normalized_prompt_text:
            normalized_text = attach_language_tag(normalized_text, normalized_language)
        inputs: RuntimeInputs = {
            "fid": self._build_request_id(
                text=normalized_text,
                prompt_audio_path=prompt_audio_path,
                prompt_text=normalized_prompt_text,
                template_name=normalized_template_name,
                language=normalized_language,
            ),
            "language": normalized_language or "",
            "text": normalized_text,
            "prompt_text": normalized_prompt_text,
            "template_name": normalized_template_name,
        }
        if prompt_audio_path:
            inputs["prompt_audio"] = self._load_prompt_audio(prompt_audio_path)
        prompt_audio_patch_count = self._estimate_prompt_audio_patch_count(
            prompt_audio=inputs.get("prompt_audio"),
            prompt_text=normalized_prompt_text,
        )
        if (
            prompt_audio_patch_count > 0
            and self.max_generate_length <= prompt_audio_patch_count
        ):
            raise ValueError(
                "max_generate_length must exceed prompt audio patch count when prompt_text is provided: "
                f"max_generate_length={self.max_generate_length} "
                f"prompt_audio_patch_count={prompt_audio_patch_count}."
            )
        schedule_spec = build_generation_schedule(
            text=f"{normalized_prompt_text}{normalized_text}",
            tokenizer=self.model.tokenizer,
            template=template,
            max_audio_tokens=self.max_generate_length,
        )
        schedule = torch.tensor(
            schedule_spec["schedule_ids"],
            dtype=torch.long,
            device=self.device,
        )
        inputs["generation_schedule"] = schedule.unsqueeze(0)
        return inputs


__all__ = ["DotsTtsSideModel", "DotsTtsSideModuleBundle", "DotsTtsSideRuntime"]
