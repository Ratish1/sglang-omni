"""Vendored dots TTS model assembly.

sglang-omni serving uses `DotsTtsSideModel` as the boundary around this upstream model
math. The upstream full-runtime entrypoints and top-level prefill/decode loops are
intentionally removed so the native Omni path owns request scheduling, latent feedback,
and vocoder staging.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen2Config

from sglang_omni.models.dots_tts.native.models.dots_tts.config import ModelConfig
from sglang_omni.models.dots_tts.native.models.dots_tts.core import DotsTtsCore
from sglang_omni.models.dots_tts.native.modules.speaker.encoder import (
    SpeakerXVectorFeatures,
)
from sglang_omni.models.dots_tts.native.modules.vocoder.bigvgan import AudioVAE
from sglang_omni.models.dots_tts.native.utils.profiling import measure_inference
from sglang_omni.models.dots_tts.native.utils.tokenizer import (
    AUDIO_GEN_START_TOKEN,
    require_token_id,
)
from sglang_omni.models.dots_tts.native.utils.util import get_dtype


@dataclass
class _GenerateState:
    llm_cache: Any | None = None
    llm_hiddens: torch.Tensor | None = None
    patch_encoder_state: Any | None = None
    fm_seq_len: int = 0
    fm_capacity: int = 0
    fm_sequence: torch.Tensor | None = None
    fm_cfg_sequence: torch.Tensor | None = None
    fm_null_g_cond: torch.Tensor | None = None
    end_flag: bool = False


@dataclass(frozen=True)
class _PromptConditioning:
    prompt_patches: torch.Tensor | None = None
    prompt_latents: torch.Tensor | None = None
    g_cond: torch.Tensor | None = None


@dataclass
class _PromptFeatureCacheEntry:
    speaker_embedding: torch.Tensor | None = None
    prompt_latent_distribution: torch.Tensor | None = None


@dataclass(frozen=True)
class _GenerateLengthBucket:
    size: int


class DotsTtsModel(nn.Module):
    """Inference model assembly around the dots.tts core network."""

    _GENERATE_LENGTH_BUCKETS = (
        _GenerateLengthBucket(32),
        _GenerateLengthBucket(64),
        _GenerateLengthBucket(128),
        _GenerateLengthBucket(256),
        _GenerateLengthBucket(512),
        _GenerateLengthBucket(1024),
    )
    _COMPILE_TARGETS = frozenset(
        {
            "FM",
            "patch_encoder",
            "vocoder",
        }
    )
    _optimize_enabled = True
    _PROMPT_FEATURE_CACHE_MAX_ENTRIES = 256
    CONFIG_FILENAME = "config.json"
    HF_MODEL_TYPE = "dots_tts"
    HF_ARCHITECTURES = ["DotsTTSForConditionalGeneration"]
    LATENT_STATS_FILENAME = "latent_stats.pt"
    LLM_CONFIG_FILENAME = "llm_config.json"
    MODEL_FILENAME = "model.safetensors"
    VOCODER_FILENAME = "vocoder.safetensors"
    SPEAKER_ENCODER_FILENAME = "speaker_encoder.safetensors"
    _ARTIFACT_ALIASES = (("llm.lm_head.weight", "llm.model.embed_tokens.weight"),)
    REQUIRED_ARTIFACT_FILES = (
        CONFIG_FILENAME,
        LATENT_STATS_FILENAME,
        LLM_CONFIG_FILENAME,
        MODEL_FILENAME,
        VOCODER_FILENAME,
        SPEAKER_ENCODER_FILENAME,
    )

    # region Module assembly and checkpoint IO
    def __init__(
        self,
        config: ModelConfig,
        tokenizer,
        latent_stats_path: str | Path,
        llm_config: Qwen2Config,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.latent_stats_path = Path(latent_stats_path)
        self.audio_gen_start_id = require_token_id(
            self.tokenizer, AUDIO_GEN_START_TOKEN
        )

        self.core = DotsTtsCore(
            config,
            llm_config=llm_config,
            tokenizer=tokenizer,
            latent_stats_path=self.latent_stats_path,
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
        self._compiled_models: dict[
            tuple[str, tuple[Any, ...] | None], Callable[..., Any]
        ] = {}
        self._prompt_feature_cache: OrderedDict[str, _PromptFeatureCacheEntry] = (
            OrderedDict()
        )
        self._fm_decode_workspaces: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}

    def set_optimize(self, optimize: bool) -> None:
        self._optimize_enabled = bool(optimize)
        if not self._optimize_enabled:
            self._compiled_models.clear()

    def set_cfg_droprate(
        self,
        cfg_droprate: float | None = None,
        xvec_drop_rate: float | None = None,
    ) -> None:
        if cfg_droprate is not None:
            self.config.cfg_droprate = cfg_droprate
            self.core.config.cfg_droprate = cfg_droprate
            self.core.cfg_droprate = cfg_droprate

        if xvec_drop_rate is not None:
            self.config.xvec_drop_rate = xvec_drop_rate
            self.core.config.xvec_drop_rate = xvec_drop_rate
            self.core.xvec_drop_rate = xvec_drop_rate

    @classmethod
    def _resolve_generate_length_bucket(
        cls,
        max_generate_length: int,
    ) -> _GenerateLengthBucket:
        requested = int(max_generate_length)
        if requested <= 0:
            raise ValueError("max_generate_length must be positive.")
        for bucket in cls._GENERATE_LENGTH_BUCKETS:
            if requested <= bucket.size:
                return bucket
        raise ValueError(
            "max_generate_length exceeds the largest supported compile bucket: "
            f"max_generate_length={requested} "
            f"max_supported={cls._GENERATE_LENGTH_BUCKETS[-1].size}."
        )

    def _resolve_state_audio_patch_count(self, max_audio_patch_count: int) -> int:
        requested = int(max_audio_patch_count)
        if requested <= 0:
            raise ValueError("max_audio_patch_count must be positive.")
        if not self._optimize_enabled:
            return requested
        return self._resolve_generate_length_bucket(requested).size

    def _warmup_fm_bucket(
        self,
        *,
        max_audio_patch_count: int,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
    ) -> None:
        dtype = get_dtype(precision)
        device = next(self.core.parameters()).device
        use_amp = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            state = self._allocate_generate_state(
                max_audio_patch_count=max_audio_patch_count,
                device=device,
                dtype=dtype,
            )
            state.fm_seq_len = state.fm_capacity
            self._decode_next_audio(
                state,
                device=device,
                g_cond=None,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
            )

    def _warmup_patch_encoder_bucket(
        self,
        *,
        max_audio_patch_count: int,
        precision: str,
    ) -> None:
        dtype = get_dtype(precision)
        device = next(self.core.parameters()).device
        state_dtype = dtype if device.type == "cuda" else torch.float32
        use_amp = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            state_audio_patch_count = self._resolve_state_audio_patch_count(
                max_audio_patch_count
            )
            patch_encoder_state = self.core.patch_encoder.init_decode_state(
                max_audio_patch_count=state_audio_patch_count,
                batch_size=1,
                device=device,
                dtype=state_dtype,
            )
            audio_patch = torch.zeros(
                (
                    1,
                    self.core.patch_encoder.patch_size,
                    self.core.latent_dim,
                ),
                dtype=state_dtype,
                device=device,
            )
            audio_patch = self.core.io_helper.denormalize(audio_patch)
            patch_encoder_decode = self._get_compiled_model(
                "patch_encoder.decode_patch",
                self.core.patch_encoder.decode_patch,
                signature=self._patch_encoder_compile_signature(patch_encoder_state),
            )
            positions = torch.arange(
                self.core.patch_encoder.out_ds_rate,
                device=device,
                dtype=torch.long,
            )
            with measure_inference("patch_encoder"):
                patch_encoder_decode(
                    audio_patch,
                    patch_encoder_state.conv_tail,
                    patch_encoder_state.layer_caches,
                    positions,
                )

    def _compile_callable(
        self,
        key: str,
        model: Callable[..., Any],
        *,
        signature: tuple[Any, ...] | None = None,
    ) -> Callable[..., Any]:
        compile_target = key.split(".", maxsplit=1)[0]
        cache_key = (key, signature)
        compiled = self._compiled_models.get(cache_key)
        if compiled is None:
            mode = (
                "default" if key == "patch_encoder.decode_patch" else "reduce-overhead"
            )
            compiled = torch.compile(
                model,
                mode=mode,
                fullgraph=True,
                dynamic=False,
            )
            self._compiled_models[cache_key] = compiled
            logger.info(
                "Compiled inference target: key={} target={} signature={}",
                key,
                compile_target,
                signature,
            )
        return compiled

    def _get_compiled_model(
        self,
        key: str,
        model: Callable[..., Any],
        *,
        signature: tuple[Any, ...] | None = None,
    ) -> Callable[..., Any]:
        compile_target = key.split(".", maxsplit=1)[0]
        if not self._optimize_enabled or compile_target not in self._COMPILE_TARGETS:
            return model
        return self._compile_callable(
            key,
            model,
            signature=signature,
        )

    def _get_compiled_method(
        self,
        key: str,
        owner: Any,
        method_name: str,
        *,
        signature: tuple[Any, ...] | None = None,
    ) -> Callable[..., Any]:
        bound_method = getattr(owner, method_name)
        compile_target = key.split(".", maxsplit=1)[0]
        if not self._optimize_enabled or compile_target not in self._COMPILE_TARGETS:
            return bound_method

        raw_method = getattr(type(owner), method_name)
        raw_callable = getattr(raw_method, "__wrapped__", raw_method)
        compiled = self._compile_callable(
            key,
            raw_callable,
            signature=signature,
        )
        return partial(compiled, owner)

    def _allocate_generate_state(
        self,
        *,
        max_audio_patch_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _GenerateState:
        state_dtype = dtype if device.type == "cuda" else torch.float32
        state_audio_patch_count = self._resolve_state_audio_patch_count(
            max_audio_patch_count
        )
        fm_capacity = state_audio_patch_count * (
            self.core.hidden_patch_size + self.core.latent_patch_size
        )
        workspace = self._allocate_fm_state_buffers(
            fm_capacity=fm_capacity,
            device=device,
            dtype=state_dtype,
        )

        patch_encoder_state = None
        if not self._optimize_enabled:
            patch_encoder_state = self.core.patch_encoder.init_decode_state(
                max_audio_patch_count=state_audio_patch_count,
                batch_size=1,
                device=device,
                dtype=state_dtype,
            )

        return _GenerateState(
            patch_encoder_state=patch_encoder_state,
            fm_seq_len=0,
            fm_capacity=fm_capacity,
            fm_sequence=workspace["fm_sequence"],
            fm_cfg_sequence=workspace["fm_cfg_sequence"],
            fm_null_g_cond=workspace["fm_null_g_cond"],
        )

    def _allocate_fm_state_buffers(
        self,
        *,
        fm_capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            "fm_sequence": torch.zeros(
                (1, fm_capacity, self.core.fm_hidden_size),
                dtype=dtype,
                device=device,
            ),
            "fm_cfg_sequence": torch.zeros(
                (1, fm_capacity, self.core.fm_hidden_size),
                dtype=dtype,
                device=device,
            ),
            "fm_null_g_cond": torch.zeros(
                (1, self.core.fm_hidden_size),
                dtype=dtype,
                device=device,
            ),
        }

    @staticmethod
    def _tensor_storage_signature(tensor: torch.Tensor) -> tuple:
        return (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.size()),
            tuple(tensor.stride()),
            tensor.dtype,
        )

    @classmethod
    def _build_artifact_state_dict(cls, module) -> dict[str, torch.Tensor]:
        state_dict = module.state_dict()
        skip_keys = set()

        for redundant_key, canonical_key in cls._ARTIFACT_ALIASES:
            redundant_tensor = state_dict.get(redundant_key)
            canonical_tensor = state_dict.get(canonical_key)
            if (
                redundant_tensor is not None
                and canonical_tensor is not None
                and cls._tensor_storage_signature(redundant_tensor)
                == cls._tensor_storage_signature(canonical_tensor)
            ):
                skip_keys.add(redundant_key)

        cleaned_state_dict = {}
        seen_storage = set()
        for key, value in state_dict.items():
            if key in skip_keys:
                continue

            storage_signature = cls._tensor_storage_signature(value)
            if storage_signature in seen_storage:
                continue

            seen_storage.add(storage_signature)
            cleaned_state_dict[key] = value.detach().cpu().contiguous()

        return cleaned_state_dict

    @classmethod
    def _restore_artifact_state_dict(cls, state_dict: dict, module) -> dict:
        restored_state_dict = dict(state_dict)
        for redundant_key, canonical_key in cls._ARTIFACT_ALIASES:
            if (
                canonical_key in restored_state_dict
                and redundant_key not in restored_state_dict
                and redundant_key in module.state_dict()
            ):
                restored_state_dict[redundant_key] = restored_state_dict[canonical_key]
        return restored_state_dict

    @classmethod
    def _save_artifact_module(cls, module, path: Path) -> None:
        save_file(cls._build_artifact_state_dict(module), path)

    @classmethod
    def _load_artifact_module(cls, module, path: Path):
        state_dict = load_file(path, device="cpu")
        restored_state_dict = cls._restore_artifact_state_dict(state_dict, module)
        mismatch = module.load_state_dict(restored_state_dict, strict=False)
        if mismatch.missing_keys or mismatch.unexpected_keys:
            raise RuntimeError(f"Failed to load {path}: {mismatch}")
        return module

    @classmethod
    def _validate_pretrained_directory(
        cls, pretrained_model_name_or_path: str | Path
    ) -> Path:
        pretrained_path = Path(pretrained_model_name_or_path).expanduser().resolve()
        missing_files = [
            name
            for name in cls.REQUIRED_ARTIFACT_FILES
            if not (pretrained_path / name).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Pretrained path {pretrained_path} is missing required files: {missing_files}"
            )
        return pretrained_path

    @classmethod
    def _load_pretrained_config(cls, pretrained_path: Path) -> ModelConfig:
        return ModelConfig.model_validate(
            json.loads(
                (pretrained_path / cls.CONFIG_FILENAME).read_text(encoding="utf-8")
            )
        )

    @staticmethod
    def _save_llm_config(llm_config: Qwen2Config, path: Path) -> None:
        path.write_text(
            json.dumps(llm_config.to_dict(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _load_llm_config(path: Path) -> Qwen2Config:
        return Qwen2Config.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _tie_llm_weights(self) -> None:
        if hasattr(self.core.llm, "tie_weights"):
            self.core.llm.tie_weights()

    def save_pretrained(self, save_directory: str | Path) -> Path:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config_payload = self.config.to_declared_dict()
        config_payload["model_type"] = self.HF_MODEL_TYPE
        config_payload["architectures"] = list(self.HF_ARCHITECTURES)
        (save_directory / self.CONFIG_FILENAME).write_text(
            json.dumps(config_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self._save_llm_config(
            self.core.llm.config,
            save_directory / self.LLM_CONFIG_FILENAME,
        )
        self.tokenizer.save_pretrained(save_directory)
        shutil.copy2(
            self.latent_stats_path,
            save_directory / self.LATENT_STATS_FILENAME,
        )
        self._save_artifact_module(self.core, save_directory / self.MODEL_FILENAME)
        self._save_artifact_module(self.vocoder, save_directory / self.VOCODER_FILENAME)
        self._save_artifact_module(
            self.xvector_extractor,
            save_directory / self.SPEAKER_ENCODER_FILENAME,
        )
        return save_directory

    def _load_pretrained_artifacts(self, pretrained_path: Path) -> None:
        self.latent_stats_path = pretrained_path / self.LATENT_STATS_FILENAME
        self.core.io_helper = type(self.core.io_helper)(
            latent_stats_path=self.latent_stats_path
        )
        self._load_artifact_module(self.core, pretrained_path / self.MODEL_FILENAME)
        self._tie_llm_weights()
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

    def load_pretrained_weights(
        self, pretrained_model_name_or_path: str | Path
    ) -> None:
        pretrained_path = self._validate_pretrained_directory(
            pretrained_model_name_or_path
        )
        saved_config = self._load_pretrained_config(pretrained_path)
        if saved_config.to_declared_dict() != self.config.to_declared_dict():
            raise ValueError(
                f"Pretrained config at {pretrained_path} does not match the current model."
            )
        saved_llm_config = self._load_llm_config(
            pretrained_path / self.LLM_CONFIG_FILENAME
        )
        if saved_llm_config.to_dict() != self.core.llm.config.to_dict():
            raise ValueError(
                f"Pretrained LLM config at {pretrained_path} does not match the current model."
            )
        self._load_pretrained_artifacts(pretrained_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path):
        logger.info(
            "DotsTtsModel load started: pretrained_path={}",
            pretrained_model_name_or_path,
        )
        pretrained_model_name_or_path = cls._validate_pretrained_directory(
            pretrained_model_name_or_path
        )
        config = cls._load_pretrained_config(pretrained_model_name_or_path)
        llm_config = cls._load_llm_config(
            pretrained_model_name_or_path / cls.LLM_CONFIG_FILENAME
        )
        logger.info(
            "DotsTtsModel config loaded: pretrained_path={} sample_rate={} patch_size={}",
            pretrained_model_name_or_path,
            config.vocoder.sample_rate,
            config.patch_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(pretrained_model_name_or_path),
            local_files_only=True,
        )
        model = cls(
            config,
            tokenizer=tokenizer,
            latent_stats_path=pretrained_model_name_or_path / cls.LATENT_STATS_FILENAME,
            llm_config=llm_config,
        )
        model._load_pretrained_artifacts(pretrained_model_name_or_path)
        logger.info(
            "DotsTtsModel load completed: pretrained_path={}",
            pretrained_model_name_or_path,
        )
        return model.eval()

    # endregion Module assembly and checkpoint IO

    # region Prompt conditioning and decode state helpers
    def _prepare_prompt_audio_for_conditioning(
        self,
        prompt_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        prompt_audio = prompt_audio.detach().cpu().contiguous()

        samples_per_patch = self.config.patch_size * self.hop_size
        target_len = (
            math.ceil(prompt_audio.size(1) / samples_per_patch) * samples_per_patch
        )
        pad_len = target_len - prompt_audio.size(1)
        if pad_len > 0:
            prompt_audio = F.pad(prompt_audio, (0, pad_len))

        digest = hashlib.sha1()
        digest.update(str(tuple(prompt_audio.shape)).encode("ascii"))
        digest.update(str(prompt_audio.dtype).encode("ascii"))
        digest.update(prompt_audio.numpy().tobytes())
        return prompt_audio, digest.hexdigest()

    def _get_prompt_feature_cache_entry(
        self,
        cache_key: str,
    ) -> _PromptFeatureCacheEntry | None:
        entry = self._prompt_feature_cache.get(cache_key)
        if entry is not None:
            self._prompt_feature_cache.move_to_end(cache_key)
        return entry

    def _store_prompt_feature_cache_entry(
        self,
        cache_key: str,
        entry: _PromptFeatureCacheEntry,
    ) -> None:
        if entry.speaker_embedding is None and entry.prompt_latent_distribution is None:
            return
        self._prompt_feature_cache[cache_key] = entry
        self._prompt_feature_cache.move_to_end(cache_key)
        while len(self._prompt_feature_cache) > self._PROMPT_FEATURE_CACHE_MAX_ENTRIES:
            self._prompt_feature_cache.popitem(last=False)

    def _can_cache_speaker_embedding(self, prompt_sample_count: int) -> bool:
        max_audio_seconds = self.xvector_extractor.max_audio_seconds
        if max_audio_seconds <= 0:
            return True
        max_input_length = round(self.xvector_extractor.sample_rate * max_audio_seconds)
        return int(prompt_sample_count) <= max_input_length

    @torch.no_grad()
    def _prepare_prompt_conditioning(
        self,
        prompt_audio: torch.Tensor | None,
        *,
        use_prompt_prefill: bool,
        speaker_scale: float = 1.5,
    ) -> _PromptConditioning:
        if prompt_audio is None:
            logger.info("Prompt conditioning skipped: no prompt audio provided.")
            return _PromptConditioning()

        self.vocoder.eval()
        self.xvector_extractor.eval()
        device = next(self.core.parameters()).device
        prompt_audio, cache_key = self._prepare_prompt_audio_for_conditioning(
            prompt_audio
        )
        prompt_sample_count = int(prompt_audio.shape[-1])
        cache_entry = self._get_prompt_feature_cache_entry(cache_key)
        if cache_entry is None:
            cache_entry = _PromptFeatureCacheEntry()
        prompt_audio = prompt_audio.to(device=device)

        can_cache_speaker = self._can_cache_speaker_embedding(prompt_sample_count)
        speaker_embedding = cache_entry.speaker_embedding if can_cache_speaker else None
        if speaker_embedding is None:
            speaker_encoder = self._get_compiled_model(
                "speaker_encoder",
                self.xvector_extractor,
            )
            with measure_inference("speaker_encoder"):
                speaker_embedding = speaker_encoder(prompt_audio[None, :])
            if can_cache_speaker:
                cache_entry.speaker_embedding = speaker_embedding.detach()
        else:
            logger.info(
                "Prompt speaker cache hit: key={} prompt_samples={}",
                cache_key[:12],
                prompt_sample_count,
            )
        xvec_param = next(self.core.xvec_proj.parameters())
        speaker_embedding = speaker_embedding.to(
            device=xvec_param.device,
            dtype=xvec_param.dtype,
        )
        g_cond = self.core.xvec_proj(speaker_embedding * float(speaker_scale))
        if not use_prompt_prefill:
            self._store_prompt_feature_cache_entry(cache_key, cache_entry)
            logger.info(
                "Reference-audio-only conditioning prepared: prompt_samples={} speaker_scale={} device={}",
                prompt_sample_count,
                speaker_scale,
                device,
            )
            return _PromptConditioning(g_cond=g_cond)

        prompt_latents = cache_entry.prompt_latent_distribution
        if prompt_latents is None:
            latent_encoder = self._get_compiled_model(
                "latent_encoder",
                self.vocoder.extract_latents,
            )
            with measure_inference("latent_encoder"):
                prompt_latents = latent_encoder(prompt_audio[None, :])
            cache_entry.prompt_latent_distribution = prompt_latents.detach()
        else:
            logger.info(
                "Prompt latent cache hit: key={} prompt_samples={}",
                cache_key[:12],
                prompt_sample_count,
            )
        self._store_prompt_feature_cache_entry(cache_key, cache_entry)
        prompt_latents_sampled = self.core.io_helper.sample_from_latent(prompt_latents)
        prompt_latents_sampled = prompt_latents_sampled[:, : -self.config.patch_size]
        prompt_patches = rearrange(
            self.core.io_helper.normalize(prompt_latents_sampled),
            "b (s p) d -> b s p d",
            p=self.config.patch_size,
        )
        logger.info(
            "Prompt conditioning prepared: prompt_samples={} prompt_patch_count={} "
            "speaker_scale={} device={}",
            prompt_sample_count,
            prompt_patches.size(1),
            speaker_scale,
            device,
        )
        return _PromptConditioning(
            prompt_patches=prompt_patches,
            prompt_latents=prompt_latents_sampled,
            g_cond=g_cond,
        )

    @staticmethod
    def _patch_encoder_compile_signature(
        patch_encoder_state: Any,
    ) -> tuple[int, torch.dtype]:
        key_cache, _ = patch_encoder_state.layer_caches[0]
        return int(key_cache.size(2)), key_cache.dtype

    def _resolve_patch_encoder_audio_bucket(self, required_seq_len: int) -> int:
        requested = int(required_seq_len)
        if requested <= 0:
            raise ValueError("required_seq_len must be positive.")
        requested_patch_count = math.ceil(
            requested / self.core.patch_encoder.out_ds_rate
        )
        if not self._optimize_enabled:
            return requested_patch_count
        return self._resolve_generate_length_bucket(requested_patch_count).size

    def _copy_patch_encoder_state(self, source: Any, target: Any) -> None:
        seq_len = source.seq_len
        target_capacity = int(target.layer_caches[0][0].size(2))
        if seq_len > target_capacity:
            raise ValueError(
                "Patch encoder state copy exceeds target capacity: "
                f"seq_len={seq_len} capacity={target_capacity}."
            )

        target.conv_tail.copy_(source.conv_tail)
        target.seq_len = seq_len
        for (source_key, source_value), (target_key, target_value) in zip(
            source.layer_caches,
            target.layer_caches,
            strict=True,
        ):
            if seq_len > 0:
                target_key[:, :, :seq_len, :].copy_(source_key[:, :, :seq_len, :])
                target_value[:, :, :seq_len, :].copy_(source_value[:, :, :seq_len, :])

    def _ensure_patch_encoder_state_capacity(
        self,
        state: _GenerateState,
        *,
        required_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        current_state = state.patch_encoder_state
        if current_state is not None:
            current_capacity = int(current_state.layer_caches[0][0].size(2))
            if current_capacity >= required_seq_len:
                return

        target_audio_patch_count = self._resolve_patch_encoder_audio_bucket(
            required_seq_len
        )
        next_state = self.core.patch_encoder.init_decode_state(
            max_audio_patch_count=target_audio_patch_count,
            batch_size=1,
            device=device,
            dtype=dtype,
        )
        if current_state is not None:
            self._copy_patch_encoder_state(current_state, next_state)
        state.patch_encoder_state = next_state

    def _prefill_prompt_latents(
        self,
        prompt_latents: torch.Tensor | None,
        *,
        state: _GenerateState,
    ) -> torch.Tensor | None:
        if prompt_latents is None:
            return None
        if prompt_latents.size(1) == 0:
            return prompt_latents.new_zeros(
                (prompt_latents.size(0), 0, self.core.llm_hidden_size)
            )
        self._ensure_patch_encoder_state_capacity(
            state,
            required_seq_len=(
                (prompt_latents.size(1) // self.core.patch_encoder.patch_size)
                * self.core.patch_encoder.out_ds_rate
            ),
            device=prompt_latents.device,
            dtype=(
                state.fm_sequence.dtype
                if state.fm_sequence is not None
                else prompt_latents.dtype
            ),
        )
        with measure_inference("patch_encoder"):
            prompt_patch_embeddings, state.patch_encoder_state = (
                self.core.patch_encoder.prefill(
                    prompt_latents,
                    state.patch_encoder_state,
                )
            )
        return prompt_patch_embeddings

    def _get_fm_decode_workspace(
        self,
        *,
        total_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        workspace_key = (total_len, str(device), dtype)
        workspace = self._fm_decode_workspaces.get(workspace_key)
        if workspace is None:
            workspace = {
                "input_sequence": torch.zeros(
                    (1, total_len, self.core.fm_hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "cfg_sequence": torch.zeros(
                    (1, total_len, self.core.fm_hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "attn_mask": torch.zeros(
                    (1, total_len, total_len),
                    dtype=torch.bool,
                    device=device,
                ),
                "pos_ids": torch.zeros(
                    (1, total_len),
                    dtype=torch.float32,
                    device=device,
                ),
            }
            self._fm_decode_workspaces[workspace_key] = workspace
        else:
            workspace["input_sequence"].zero_()
            workspace["cfg_sequence"].zero_()
        return workspace

    def _resolve_fm_history_bucket_capacity(self, fm_seq_len: int) -> int:
        requested = int(fm_seq_len)
        if requested <= 0:
            raise ValueError("fm_seq_len must be positive.")
        if not self._optimize_enabled:
            return requested
        history_stride = self.core.hidden_patch_size + self.core.latent_patch_size
        requested_patch_count = math.ceil(requested / history_stride)
        return (
            self._resolve_generate_length_bucket(requested_patch_count).size
            * history_stride
        )

    def _build_fm_attn_mask(
        self,
        *,
        state: _GenerateState,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        if state.fm_seq_len <= 0:
            raise RuntimeError("FM sequence length must be positive before decode.")
        hidden_patch_size = self.core.hidden_patch_size
        latent_start = attn_mask.size(-1) - self.core.latent_patch_size
        attn_mask.zero_()
        block_start = state.fm_seq_len - hidden_patch_size
        if block_start > 0:
            causal_mask = (
                torch.ones(
                    (block_start, block_start),
                    device=attn_mask.device,
                    dtype=torch.bool,
                )
                .triu(1)
                .logical_not()
            )
            attn_mask[:, :block_start, :block_start] = causal_mask

        attn_mask[:, block_start : state.fm_seq_len, : state.fm_seq_len] = True
        attn_mask[:, block_start : state.fm_seq_len, latent_start:] = True
        attn_mask[:, latent_start:, : state.fm_seq_len] = True
        attn_mask[:, latent_start:, latent_start:] = True
        if latent_start > state.fm_seq_len:
            padding_indices = torch.arange(
                state.fm_seq_len,
                latent_start,
                device=attn_mask.device,
            )
            attn_mask[:, padding_indices, padding_indices] = True
        return attn_mask

    def _build_fm_pos_ids(
        self,
        *,
        state: _GenerateState,
        pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        if state.fm_seq_len <= 0:
            raise RuntimeError("FM sequence length must be positive before decode.")
        pos_ids.zero_()
        latent_start = pos_ids.size(-1) - self.core.latent_patch_size
        pos_ids[:, : state.fm_seq_len] = torch.arange(
            state.fm_seq_len,
            device=pos_ids.device,
            dtype=pos_ids.dtype,
        )
        pos_ids[:, latent_start:] = torch.arange(
            state.fm_seq_len,
            state.fm_seq_len + self.core.latent_patch_size,
            device=pos_ids.device,
            dtype=pos_ids.dtype,
        )
        return pos_ids

    def _prepare_fm_decode_inputs(
        self,
        state: _GenerateState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        sequence = state.fm_sequence
        cfg_sequence = state.fm_cfg_sequence
        if sequence is None or cfg_sequence is None:
            raise RuntimeError("FM static buffers are not initialized.")
        history_bucket_capacity = self._resolve_fm_history_bucket_capacity(
            state.fm_seq_len
        )
        total_len = history_bucket_capacity + self.core.latent_patch_size
        workspace = self._get_fm_decode_workspace(
            total_len=total_len,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        workspace["input_sequence"][:, : state.fm_seq_len].copy_(
            sequence[:, : state.fm_seq_len]
        )
        workspace["cfg_sequence"][:, : state.fm_seq_len].copy_(
            cfg_sequence[:, : state.fm_seq_len]
        )
        return (
            workspace["input_sequence"],
            workspace["cfg_sequence"],
            workspace["attn_mask"],
            workspace["pos_ids"],
            history_bucket_capacity,
        )

    def _append_to_fm_buffer(
        self,
        buffer: torch.Tensor | None,
        state: _GenerateState,
        chunk: torch.Tensor,
    ) -> tuple[int, int]:
        if buffer is None:
            raise RuntimeError("FM static buffer is not initialized.")
        start = state.fm_seq_len
        end = start + chunk.size(1)
        if end > state.fm_capacity:
            raise RuntimeError(
                "FM StaticBuffer capacity exceeded: "
                f"next_length={end} capacity={state.fm_capacity}."
            )
        buffer[:, start:end].copy_(chunk.to(buffer.dtype))
        return start, end

    def _append_hidden_chunk(
        self, state: _GenerateState, hidden_chunk: torch.Tensor
    ) -> None:
        last_hidden = hidden_chunk[:, -self.core.hidden_patch_size :, :]
        projected = self.core.hidden_proj(last_hidden)
        null_projected = self.core.hidden_proj(torch.zeros_like(last_hidden))
        _start, end = self._append_to_fm_buffer(
            state.fm_sequence,
            state,
            projected,
        )
        cfg_buffer = state.fm_cfg_sequence
        if cfg_buffer is None:
            raise RuntimeError("FM cfg static buffer is not initialized.")
        cfg_buffer[:, state.fm_seq_len : end].copy_(null_projected.to(cfg_buffer.dtype))
        state.fm_seq_len = end

    def _append_history_chunk(
        self, state: _GenerateState, latent_chunk: torch.Tensor
    ) -> None:
        history_latent = self.core.latent_proj(latent_chunk)
        _start, end = self._append_to_fm_buffer(
            state.fm_sequence,
            state,
            history_latent,
        )
        cfg_buffer = state.fm_cfg_sequence
        if cfg_buffer is None:
            raise RuntimeError("FM cfg static buffer is not initialized.")
        cfg_buffer[:, state.fm_seq_len : end].copy_(history_latent.to(cfg_buffer.dtype))
        state.fm_seq_len = end

    def _locate_prefill_boundary(
        self,
        *,
        span_positions: torch.Tensor,
        prompt_patch_count: int,
    ) -> tuple[int, torch.Tensor]:
        if span_positions.numel() > prompt_patch_count:
            return (
                int(span_positions[prompt_patch_count].item()),
                span_positions[:prompt_patch_count],
            )
        raise RuntimeError(
            "Prefill boundary discovery failed despite prior schedule validation."
        )

    @staticmethod
    def _find_audio_span_positions(
        generation_schedule: torch.Tensor,
        *,
        audio_placeholder_ids: set[int],
    ) -> torch.Tensor:
        schedule = generation_schedule[0]
        placeholder_ids = torch.tensor(
            sorted(audio_placeholder_ids),
            device=schedule.device,
            dtype=schedule.dtype,
        )
        return torch.nonzero(
            torch.isin(schedule, placeholder_ids),
            as_tuple=False,
        ).squeeze(-1)

    @staticmethod
    def _next_token_is_audio_span(
        generation_schedule: torch.Tensor,
        *,
        position: int,
        audio_placeholder_ids: set[int],
    ) -> bool:
        next_position = position + 1
        if next_position >= generation_schedule.size(1):
            return False
        return (
            int(generation_schedule[0, next_position].item()) in audio_placeholder_ids
        )

    def _build_prefill_inputs_embeds(
        self,
        generation_schedule: torch.Tensor,
        *,
        prompt_patch_embeddings: torch.Tensor | None,
        prompt_span_positions: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.core.llm.get_input_embeddings()(
            generation_schedule
        ).clone()
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

    def _decode_next_audio(
        self,
        state: _GenerateState,
        *,
        device: torch.device,
        g_cond: torch.Tensor | None,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        if state.fm_seq_len <= 0:
            raise RuntimeError(
                "Cannot decode audio before any conditioning state has been prefetched."
            )
        if state.fm_sequence is None or state.fm_cfg_sequence is None:
            raise RuntimeError("FM static buffers are not initialized.")
        if state.fm_null_g_cond is None:
            raise RuntimeError("FM null conditioning buffer is not initialized.")
        (
            fm_sequence,
            fm_cfg_sequence,
            fm_attn_mask,
            fm_pos_ids,
            history_bucket_capacity,
        ) = self._prepare_fm_decode_inputs(state)
        compile_signature = (
            (history_bucket_capacity, state.fm_sequence.dtype)
            if self._optimize_enabled
            else (state.fm_seq_len, state.fm_sequence.dtype)
        )
        if g_cond is None:
            g_cond = state.fm_null_g_cond
        else:
            g_cond = g_cond.to(
                device=state.fm_null_g_cond.device,
                dtype=state.fm_null_g_cond.dtype,
            )
        with measure_inference("FM"):
            attn_mask = self._build_fm_attn_mask(
                state=state,
                attn_mask=fm_attn_mask,
            )
            pos_ids = self._build_fm_pos_ids(
                state=state,
                pos_ids=fm_pos_ids,
            )
            if self.core.mode == "meanflow":
                fm_solver_step = self._get_compiled_method(
                    "FM.meanflow.solver_step",
                    self.core,
                    "meanflow_solver_step",
                    signature=compile_signature,
                )
                return self.core._meanflow_step_fm(
                    input_sequence=fm_sequence,
                    attn_mask=attn_mask,
                    pos_ids=pos_ids,
                    patch_size=self.core.latent_patch_size,
                    g_cond=g_cond,
                    nfe=num_steps,
                    solver_step=fm_solver_step,
                )

            fm_solver_step = self._get_compiled_method(
                "FM.flow_matching.solver_step",
                self.core,
                "fm_solver_step",
                signature=compile_signature,
            )
            return self.core._flow_matching_step_fm(
                input_sequence=fm_sequence,
                cfg_sequence=fm_cfg_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                hidden_size=self.core.hidden_patch_size,
                patch_size=self.core.latent_patch_size,
                g_cond=g_cond,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                solver_step=fm_solver_step,
            )

    def _encode_audio_patch_feedback(
        self,
        state: _GenerateState,
        *,
        audio_patch: torch.Tensor,
    ) -> torch.Tensor:
        audio_patch_for_llm = self.core.io_helper.denormalize(audio_patch)
        self._append_history_chunk(state, audio_patch)
        current_seq_len = (
            0
            if state.patch_encoder_state is None
            else state.patch_encoder_state.seq_len
        )
        self._ensure_patch_encoder_state_capacity(
            state,
            required_seq_len=current_seq_len + self.core.patch_encoder.out_ds_rate,
            device=audio_patch_for_llm.device,
            dtype=(
                state.fm_sequence.dtype
                if state.fm_sequence is not None
                else audio_patch_for_llm.dtype
            ),
        )
        patch_encoder_decode = self._get_compiled_model(
            "patch_encoder.decode_patch",
            self.core.patch_encoder.decode_patch,
            signature=self._patch_encoder_compile_signature(state.patch_encoder_state),
        )
        patch_positions = (
            torch.arange(
                self.core.patch_encoder.out_ds_rate,
                device=audio_patch_for_llm.device,
                dtype=torch.long,
            )
            + state.patch_encoder_state.seq_len
        )
        with measure_inference("patch_encoder"):
            llm_embedding, conv_tail = patch_encoder_decode(
                audio_patch_for_llm,
                state.patch_encoder_state.conv_tail,
                state.patch_encoder_state.layer_caches,
                patch_positions,
            )
        state.patch_encoder_state.conv_tail.copy_(conv_tail)
        state.patch_encoder_state.seq_len += self.core.patch_encoder.out_ds_rate
        return llm_embedding

    def _should_stop_after_current_audio(
        self, state: _GenerateState, *, eos_threshold: float
    ) -> bool:
        if state.llm_hiddens is None:
            return False
        eos = (
            self.core.eos_proj(state.llm_hiddens).softmax(dim=-1)[:, -1, 1]
            > eos_threshold
        )
        return state.end_flag or bool(eos.item())

    # endregion Prompt conditioning and decode state helpers
