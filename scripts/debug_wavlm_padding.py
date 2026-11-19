# SPDX-License-Identifier: Apache-2.0
"""Trace WavLM scores for fixed waveforms under different padding boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Literal
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity, load_audio
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from benchmarks.tasks.tts import SPEAKER_SIMILARITY_BATCH_SIZE

ScoreMode = Literal["current", "exact", "independent"]
TensorOutput = torch.Tensor | tuple[torch.Tensor, ...]


def exact_feature_mask(
    features: torch.Tensor,
    waveform_padding_mask: torch.Tensor,
    convolutions: list[nn.Conv1d],
) -> torch.Tensor:
    valid_lengths = (~waveform_padding_mask).sum(dim=1)
    for conv in convolutions:
        assert conv.padding == (0,) and conv.dilation == (1,)
        valid_lengths = (valid_lengths - conv.kernel_size[0]) // conv.stride[0] + 1
    return (
        torch.arange(features.size(1), device=features.device) >= valid_lengths[:, None]
    )


class LayerTrace:
    def __init__(self) -> None:
        self.prefixes: dict[str, torch.Tensor] = {}
        self.valid_encoder_frames: int = 0

    def capture(
        self,
        name: str,
        activations: TensorOutput,
        layout: Literal["bct", "btc", "tbc", "row"],
        valid_frames: int,
    ) -> None:
        assert name not in self.prefixes
        values = activations[0] if isinstance(activations, tuple) else activations
        if layout == "bct":
            row = values[0, :, :valid_frames]
        elif layout == "btc":
            row = values[0, :valid_frames, :]
        elif layout == "tbc":
            row = values[:valid_frames, 0, :]
        else:
            row = values[0]
        self.prefixes[name] = row.detach().to(device="cpu", dtype=torch.float32).clone()

    def encoder_input(
        self,
        _module: nn.Module,
        inputs: tuple[torch.Tensor],
        arguments: dict[str, torch.Tensor | int | None],
    ) -> None:
        padding_mask = arguments["padding_mask"]
        assert isinstance(padding_mask, torch.Tensor)
        self.valid_encoder_frames = (~padding_mask[0]).sum().item()


def trace_embedding(
    scorer: WavLMSpeakerSimilarity, paths: list[str]
) -> tuple[torch.Tensor, LayerTrace]:
    trace = LayerTrace()
    model = scorer.model
    upstream = model.feature_extract.model
    hooks = [
        upstream.encoder.register_forward_pre_hook(
            trace.encoder_input, with_kwargs=True
        )
    ]
    valid_lengths = load_audio(paths[0]).numel()
    hooks.append(
        upstream.feature_extractor.register_forward_pre_hook(
            lambda _module, inputs, frames=valid_lengths: trace.capture(
                "waveform_normalized", inputs[0][:, :frames], "row", frames
            )
        )
    )
    for index, block in enumerate(upstream.feature_extractor.conv_layers):
        conv = block[0]
        valid_lengths = (valid_lengths - conv.kernel_size[0]) // conv.stride[0] + 1
        for name, layer in (("convolution", conv), ("block", block)):
            hooks.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, frames=valid_lengths, component=f"frontend.{index}.{name}": trace.capture(
                        component, output, "bct", frames
                    )
                )
            )
    for name, layer, layout in (
        ("frontend_norm", upstream.layer_norm, "btc"),
        ("projection", upstream.post_extract_proj, "btc"),
        ("positional_conv", upstream.encoder.pos_conv, "bct"),
    ):
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, component=name, axis_order=layout: (
                    trace.capture(component, output, axis_order, valid_lengths)
                )
            )
        )
    for index, layer in enumerate(upstream.encoder.layers):
        for name, component in (
            ("attention", layer.self_attn),
            ("attention_norm", layer.self_attn_layer_norm),
            ("ffn_in", layer.fc1),
            ("ffn_out", layer.fc2),
            ("layer_norm", layer.final_layer_norm),
            ("output", layer),
        ):
            hooks.append(
                component.register_forward_hook(
                    lambda _module, _inputs, output, label=f"encoder.{index}.{name}": (
                        trace.capture(label, output, "tbc", valid_lengths)
                    )
                )
            )

    def capture_head_input(_module: nn.Module, inputs: tuple[torch.Tensor]) -> None:
        if "head_input" in trace.prefixes:
            pass
        else:
            trace.capture("head_input", inputs, "row", valid_lengths)

    for name, layer in (
        ("instance_norm", model.instance_norm),
        ("tdnn", model.layer1),
        ("res2_depth1", model.layer2),
        ("res2_depth2", model.layer3),
        ("res2_depth3", model.layer4),
        ("channel_projection", model.conv),
        ("attentive_pooling", model.pooling),
        ("embedding_norm", model.bn),
        ("embedding_projection", model.linear),
    ):
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, label=f"head.{name}": (
                    None
                    if label in trace.prefixes
                    else trace.capture(label, output, "row", valid_lengths)
                )
            )
        )
    hooks.append(model.instance_norm.register_forward_pre_hook(capture_head_input))
    try:
        embedding = scorer.embed(paths)[0].detach().cpu()
    finally:
        for hook in hooks:
            hook.remove()
    return embedding, trace


def trace_deltas(
    baseline: LayerTrace, comparison: LayerTrace
) -> dict[str, list[float]]:
    assert baseline.prefixes.keys() == comparison.prefixes.keys()
    deltas = {}
    for layer, expected in baseline.prefixes.items():
        measured = comparison.prefixes[layer]
        assert expected.shape == measured.shape, (layer, expected.shape, measured.shape)
        difference = measured - expected
        rms = difference.square().mean().sqrt().item()
        baseline_rms = expected.square().mean().sqrt().item()
        deltas[layer] = [difference.abs().max().item(), rms, baseline_rms]
    return deltas


def score_layout(
    scorer: WavLMSpeakerSimilarity,
    paths: list[str],
    order: list[int],
    mode: ScoreMode,
    device: str,
) -> tuple[torch.Tensor, float]:
    embeddings = torch.empty((len(paths), scorer.model.linear.out_features))
    batch_rows = 2 * SPEAKER_SIMILARITY_BATCH_SIZE
    torch.cuda.synchronize(device)
    start_seconds = perf_counter()
    for start in range(0, len(order), batch_rows):
        row_indices = order[start : start + batch_rows]
        group = [paths[index] for index in row_indices]
        if mode == "independent":
            batch_embeddings = torch.cat([scorer.embed([path]) for path in group])
        else:
            batch_embeddings = scorer.embed(group)
        embeddings[row_indices] = batch_embeddings.cpu()
    torch.cuda.synchronize(device)
    return embeddings, perf_counter() - start_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--split", default="en")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    arguments = parser.parse_args()
    samples = load_seedtts_samples(
        arguments.dataset, split=arguments.split, revision=arguments.revision
    )
    paths = [sample.ref_audio for sample in samples]
    assert len(paths) >= 2 * SPEAKER_SIMILARITY_BATCH_SIZE
    prepared_lengths = [load_audio(path).numel() for path in paths]
    duration_order = sorted(range(len(paths)), key=prepared_lengths.__getitem__)
    mixed_order = [
        index
        for pair in zip(duration_order, reversed(duration_order))
        for index in pair
    ][: len(paths)]
    assert len(set(mixed_order)) == len(paths)

    assets = ensure_speaker_similarity_assets()
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device=arguments.device,
    )
    model = scorer.model
    upstream = model.feature_extract.model
    convolutions = [block[0] for block in upstream.feature_extractor.conv_layers]

    def corrected_mask(
        features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        return exact_feature_mask(features, padding_mask, convolutions)

    if arguments.trace or arguments.trace_only:
        long_index = duration_order[-1]
        long_path = paths[long_index]
        long_length = prepared_lengths[long_index]
        valid_lengths = model.get_feat_lengths(prepared_lengths, torch.device("cpu"))
        valid_long = valid_lengths[long_index]
        current_bins = long_length // valid_long
        short_index = next(
            index
            for index in duration_order
            if (prepared_lengths[index] + current_bins - 1) // current_bins
            != valid_lengths[index]
        )
        short_path = paths[short_index]
        short_length = prepared_lengths[short_index]
        valid_short = valid_lengths[short_index]
        before_tf32 = torch.backends.cudnn.allow_tf32
        unchanged, baseline = trace_embedding(scorer, [short_path])
        repeat, repeat_trace = trace_embedding(scorer, [short_path])
        assert torch.equal(unchanged, repeat)
        equal, equal_trace = trace_embedding(scorer, [short_path, short_path])
        current, current_trace = trace_embedding(scorer, [short_path, long_path])
        device = torch.device(arguments.device)
        lengths = torch.tensor([valid_short, valid_long], device=device)
        padding = torch.arange(valid_long, device=device) >= lengths[:, None]

        def mask_position_input(
            _module: nn.Module, inputs: tuple[torch.Tensor]
        ) -> tuple[torch.Tensor]:
            return (inputs[0].masked_fill(padding[:, None, :], 0),)

        def replace_attention_mask(
            _module: nn.Module,
            inputs: tuple[torch.Tensor],
            arguments: dict[str, torch.Tensor | bool | None],
        ) -> tuple[tuple[torch.Tensor], dict[str, torch.Tensor | bool | None]]:
            arguments["self_attn_padding_mask"] = padding
            return inputs, arguments

        position_hook = upstream.encoder.pos_conv.register_forward_pre_hook(
            mask_position_input
        )
        try:
            position_only, position_only_trace = trace_embedding(
                scorer, [short_path, long_path]
            )
        finally:
            position_hook.remove()
        attention_hooks = [
            layer.register_forward_pre_hook(replace_attention_mask, with_kwargs=True)
            for layer in upstream.encoder.layers
        ]
        try:
            attention_only, attention_only_trace = trace_embedding(
                scorer, [short_path, long_path]
            )
        finally:
            for hook in attention_hooks:
                hook.remove()
        with patch.object(upstream, "forward_padding_mask", corrected_mask):
            corrected, corrected_trace = trace_embedding(
                scorer, [short_path, long_path]
            )
            try:
                torch.backends.cudnn.allow_tf32 = False
                strict, strict_trace = trace_embedding(scorer, [short_path])
                strict_mixed, strict_mixed_trace = trace_embedding(
                    scorer, [short_path, long_path]
                )
            finally:
                torch.backends.cudnn.allow_tf32 = before_tf32
        assert (
            baseline.valid_encoder_frames
            == equal_trace.valid_encoder_frames
            == valid_short
        )
        assert current_trace.valid_encoder_frames != valid_short
        assert (
            corrected_trace.valid_encoder_frames
            == strict_mixed_trace.valid_encoder_frames
            == valid_short
        )
        trace_report = {
            "short_id": samples[short_index].sample_id,
            "long_id": samples[long_index].sample_id,
            "short_sha256": hashlib.sha256(Path(short_path).read_bytes()).hexdigest(),
            "long_sha256": hashlib.sha256(Path(long_path).read_bytes()).hexdigest(),
            "sample_counts": [short_length, long_length],
            "valid_short_frames": valid_short,
            "current_unmasked_frames": current_trace.valid_encoder_frames,
            "cudnn_tf32_default": before_tf32,
            "deltas_columns": ["max_absolute", "rms", "reference_rms"],
            "variants": {
                name: {
                    "layers": trace_deltas(reference_trace, other_trace),
                    "embedding_cosine": F.cosine_similarity(
                        reference, other, dim=0
                    ).item(),
                }
                for name, reference_trace, other_trace, reference, other in (
                    ("repeat", baseline, repeat_trace, unchanged, repeat),
                    ("equal", baseline, equal_trace, unchanged, equal),
                    ("current", baseline, current_trace, unchanged, current),
                    (
                        "exact_position_only",
                        baseline,
                        position_only_trace,
                        unchanged,
                        position_only,
                    ),
                    (
                        "exact_attention_only",
                        baseline,
                        attention_only_trace,
                        unchanged,
                        attention_only,
                    ),
                    ("exact", baseline, corrected_trace, unchanged, corrected),
                    (
                        "exact_no_cudnn_tf32",
                        strict_trace,
                        strict_mixed_trace,
                        strict,
                        strict_mixed,
                    ),
                )
            },
        }
    else:
        trace_report = None

    if arguments.trace_only:
        arguments.output_file.write_text(
            json.dumps(trace_report, indent=2, allow_nan=False)
        )
        return
    else:
        pass

    order = list(range(len(paths)))
    independent, independent_seconds = score_layout(
        scorer, paths, order, "independent", arguments.device
    )
    target_scores = (
        F.cosine_similarity(independent, independent.roll(1, 0), dim=-1) * 100
    )
    layouts = {}
    for name, indices in (("dataset", order), ("duration_extremes", mixed_order)):
        current, current_seconds = score_layout(
            scorer, paths, indices, "current", arguments.device
        )
        with patch.object(upstream, "forward_padding_mask", corrected_mask):
            exact, exact_seconds = score_layout(
                scorer, paths, indices, "exact", arguments.device
            )
        layouts[name] = {
            "seconds": {"current": current_seconds, "exact": exact_seconds},
            "score_errors": {
                variant: (
                    F.cosine_similarity(vectors, vectors.roll(1, 0), dim=-1) * 100
                    - target_scores
                ).tolist()
                for variant, vectors in (("current", current), ("exact", exact))
            },
        }
        print(f"{name} finished for {len(paths)} fixed waveforms", flush=True)
    arguments.output_file.write_text(
        json.dumps(
            {
                "dataset": arguments.dataset,
                "revision": arguments.revision,
                "split": arguments.split,
                "sample_ids": [sample.sample_id for sample in samples],
                "independent_seconds": independent_seconds,
                "extractor_mode": upstream.cfg.extractor_mode,
                "trace": trace_report,
                "layouts": layouts,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
