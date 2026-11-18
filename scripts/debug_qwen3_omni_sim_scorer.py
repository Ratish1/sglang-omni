# SPDX-License-Identifier: Apache-2.0
"""Compare batch scoring and independent embeddings on the same saved audio."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity, load_audio
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from scripts.debug_qwen3_omni_sim_bisect import Case, read_history

ARMS = (
    "late",
    "early",
    "late-repeat",
    "input-only",
    "input-inverse",
    "post-only",
    "post-inverse",
    "final-only",
    "final-inverse",
    "dispatch-late",
    "dispatch-early",
)


class PaddingProbe(TypedDict):
    extractor_mode: str
    short_file: str
    long_file: str
    short_sha256: str
    samples_at_16khz: list[int]
    valid_first_conv_frames: int
    max_abs_prefix_change: dict[str, float]
    equal_length_vs_singleton_cosine: float
    mixed_length_vs_singleton_cosine: float


def probe_padding(
    scorer: WavLMSpeakerSimilarity, short_audio_path: str, long_audio_path: str
) -> PaddingProbe:
    upstream = scorer.model.feature_extract.model
    first_conv = upstream.feature_extractor.conv_layers[0]
    (conv,) = (
        module for module in first_conv.modules() if isinstance(module, nn.Conv1d)
    )
    (group_norm,) = (
        module for module in first_conv.modules() if isinstance(module, nn.GroupNorm)
    )
    short_samples = load_audio(short_audio_path).numel()
    long_samples = load_audio(long_audio_path).numel()
    assert short_samples < long_samples
    valid_frames = (short_samples - conv.kernel_size[0]) // conv.stride[0] + 1
    activation_prefixes: dict[str, list[torch.Tensor]] = {"conv": [], "group_norm": []}
    hooks = [
        layer.register_forward_hook(
            lambda _module, _inputs, output, layer_name=name: activation_prefixes[
                layer_name
            ].append(output[0, :, :valid_frames].detach().cpu())
        )
        for name, layer in (("conv", conv), ("group_norm", group_norm))
    ]
    try:
        equal_length = scorer.embed([short_audio_path, short_audio_path])
        mixed_length = scorer.embed([short_audio_path, long_audio_path])
    finally:
        for hook in hooks:
            hook.remove()
    independent = scorer.embed([short_audio_path])[0]
    return {
        "extractor_mode": upstream.cfg.extractor_mode,
        "short_file": Path(short_audio_path).name,
        "long_file": Path(long_audio_path).name,
        "short_sha256": hashlib.sha256(Path(short_audio_path).read_bytes()).hexdigest(),
        "samples_at_16khz": [short_samples, long_samples],
        "valid_first_conv_frames": valid_frames,
        "max_abs_prefix_change": {
            name: (first - second).abs().max().item()
            for name, (first, second) in activation_prefixes.items()
        },
        "equal_length_vs_singleton_cosine": F.cosine_similarity(
            equal_length[0], independent, dim=0
        ).item(),
        "mixed_length_vs_singleton_cosine": F.cosine_similarity(
            mixed_length[0], independent, dim=0
        ).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    arguments = parser.parse_args()
    run_directory = arguments.run_directory.resolve(strict=True)
    manifest = json.loads((run_directory / "inputs/manifest.json").read_text())
    assets = ensure_speaker_similarity_assets()
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device="cuda:0",
    )
    reports = []
    sample_counts: dict[str, int] = {}
    for row in manifest["cases"]:
        case = Case(
            sample_id=row["sample_id"],
            request_id=row["request_id"],
            request_sha256=row["request_sha256"],
            reference=run_directory / "inputs/audio" / row["reference_filename"],
        )
        assert (
            hashlib.sha256(case.reference.read_bytes()).hexdigest()
            == row["reference_sha256"]
        )
        reference_path = str(case.reference)
        sample_counts[reference_path] = load_audio(reference_path).numel()
        reference_embedding = scorer.embed([reference_path])[0]
        scores: dict[str, float] = {}
        scores_by_sha: dict[str, float] = {}
        for arm in ARMS:
            history = read_history(run_directory / arm, case, forced=False)
            audio_path = str(history.waveform)
            sample_counts[audio_path] = load_audio(audio_path).numel()
            if history.waveform_sha256 not in scores_by_sha:
                embedding = scorer.embed([audio_path])[0]
                scores_by_sha[history.waveform_sha256] = (
                    F.cosine_similarity(reference_embedding, embedding, dim=0) * 100
                ).item()
            else:
                pass
            scores[arm] = scores_by_sha[history.waveform_sha256]
        assert scores["late"] == scores["late-repeat"]
        reports.append({"sample_id": case.sample_id, "singleton_sim": scores})
        print(case.sample_id, flush=True)
    padding = probe_padding(
        scorer,
        min(sample_counts, key=sample_counts.__getitem__),
        max(sample_counts, key=sample_counts.__getitem__),
    )
    (run_directory / "scorer-report.json").write_text(
        json.dumps({"padding": padding, "cases": reports}, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
