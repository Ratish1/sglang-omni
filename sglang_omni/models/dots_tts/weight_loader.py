# SPDX-License-Identifier: Apache-2.0
"""Weight loading helpers for SGLang-native dots TTS."""

from __future__ import annotations

from pathlib import Path


def map_dots_qwen2_key(key: str) -> str | None:
    """Map a dots checkpoint key to the native SGLang Qwen2 namespace."""

    if key.startswith("llm.model."):
        return "qwen2.model." + key.removeprefix("llm.model.")
    if key.startswith("llm.lm_head."):
        return "qwen2.lm_head." + key.removeprefix("llm.lm_head.")
    return None


def required_checkpoint_files(model_path: str | Path) -> list[Path]:
    root = Path(model_path)
    return [
        root / "config.json",
        root / "llm_config.json",
        root / "model.safetensors",
        root / "vocoder.safetensors",
        root / "speaker_encoder.safetensors",
        root / "latent_stats.pt",
    ]


def validate_checkpoint_files(model_path: str | Path) -> None:
    missing = [
        path for path in required_checkpoint_files(model_path) if not path.exists()
    ]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"dots TTS checkpoint is missing required files: {joined}"
        )


__all__ = [
    "map_dots_qwen2_key",
    "required_checkpoint_files",
    "validate_checkpoint_files",
]
