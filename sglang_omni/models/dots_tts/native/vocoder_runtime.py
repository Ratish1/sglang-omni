# SPDX-License-Identifier: Apache-2.0
"""Lightweight native dots TTS vocoder runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from sglang_omni.models.dots_tts.native.models.dots_tts.config import ModelConfig
from sglang_omni.models.dots_tts.native.modules.vocoder.bigvgan import AudioVAE


class DotsTTSNativeVocoderModel:
    """Subset of the dots model API needed by the Omni vocoder stage."""

    def __init__(self, vocoder: AudioVAE, *, patch_size: int) -> None:
        self.vocoder = vocoder
        self.core = type(
            "_CoreView",
            (),
            {"latent_patch_size": int(patch_size)},
        )()

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vocoder.inference_from_latents(
            latents.transpose(1, 2).float(),
            do_sample=False,
        )

    @torch.no_grad()
    def _init_vocoder_stream_state(self) -> Any:
        return self.vocoder.init_stream_state(
            batch_size=1,
            chunk_size=int(self.core.latent_patch_size),
        )

    @torch.no_grad()
    def _stream_vocoder_patch(self, latent_patch: torch.Tensor, *, stream_state: Any):
        if latent_patch.dim() != 3:
            raise ValueError(
                f"dots.tts vocoder stream expects [B,T,D] latent patch, got {tuple(latent_patch.shape)}"
            )
        return self.vocoder.stream_step(
            latent_patch.transpose(1, 2).float(),
            stream_state,
        )

    @torch.no_grad()
    def _flush_vocoder_stream(self, stream_state: Any) -> torch.Tensor:
        return self.vocoder.stream_flush(stream_state)


class DotsTTSNativeVocoderRuntime:
    """Load only AudioVAE weights from a dots checkpoint."""

    REQUIRED_FILES = ("config.json", "vocoder.safetensors")

    def __init__(self, model: DotsTTSNativeVocoderModel, *, sample_rate: int) -> None:
        self.model = model
        self.sample_rate = int(sample_rate)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        precision: str = "bfloat16",
        device: str | None = None,
        **_: Any,
    ) -> "DotsTTSNativeVocoderRuntime":
        model_path = Path(model_name_or_path).expanduser().resolve()
        missing = [
            name for name in cls.REQUIRED_FILES if not (model_path / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Dots TTS vocoder path {model_path} is missing required files: {missing}"
            )
        cfg = ModelConfig.model_validate(
            json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        )
        vocoder = AudioVAE(cfg.vocoder).eval()
        vocoder.remove_weight_norm()
        mismatch = vocoder.load_state_dict(
            load_file(model_path / "vocoder.safetensors", device="cpu"),
            strict=False,
        )
        if mismatch.missing_keys or mismatch.unexpected_keys:
            raise RuntimeError(f"Failed to load dots vocoder weights: {mismatch}")

        target_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        del precision
        # AudioVAE decode casts latent inputs to float32 internally; keep the
        # vocoder in fp32 to match upstream runtime behavior.
        vocoder = vocoder.to(device=target_device).eval()
        return cls(
            DotsTTSNativeVocoderModel(vocoder, patch_size=int(cfg.patch_size)),
            sample_rate=int(cfg.vocoder.sample_rate),
        )


__all__ = ["DotsTTSNativeVocoderRuntime"]
