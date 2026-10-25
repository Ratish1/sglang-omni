"""Remote review diagnostics through real component constructors and checkpoints."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from sglang.srt.runtime_context import override_platform
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)

from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.vision_compat import (
    Qwen3OmniMoeVisionEncoderCompat,
)


def make_checkpoint(root):
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3131)
    audio = Qwen3OmniMoeAudioEncoderConfig(
        d_model=256,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=512,
        downsample_hidden_size=16,
        output_dim=64,
    )
    vision = Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=144,
        intermediate_size=288,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=64,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0, 1],
    )
    config = Qwen3OmniMoeConfig(
        thinker_config=Qwen3OmniMoeThinkerConfig(
            audio_config=audio, vision_config=vision
        ),
        enable_audio_output=False,
    )
    config.save_pretrained(root)
    towers = (
        ("audio_tower", hf.Qwen3OmniMoeAudioEncoder(audio).eval()),
        ("visual", Qwen3OmniMoeVisionEncoderCompat(vision).eval()),
    )
    weights = {
        f"thinker.{prefix}.{key}": value.detach().contiguous()
        for prefix, tower in towers
        for key, value in tower.state_dict().items()
    }
    save_file(weights, root / "model.safetensors")


def summary(tensor):
    return dict(
        shape=list(tensor.shape),
        finite=bool(tensor.isfinite().all()),
        sha256=hashlib.sha256(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    )


def run(root, label, output, dtype, trained):
    records = []
    root = str(root)
    for kind, constructor in (
        ("audio", Qwen3OmniAudioEncoder),
        ("vision", Qwen3OmniImageEncoder),
    ):
        try:
            model = constructor(root, device="cuda:0", dtype=dtype)
            torch.manual_seed(3131)
            if kind == "audio":
                lengths = torch.tensor([400, 850, 99])
                features = torch.randn(3, 128, 850)
                mask = torch.arange(850).unsqueeze(0) < lengths.unsqueeze(1)
                arguments = dict(input_features=features, feature_attention_mask=mask)
            else:
                config = model.visual.config
                patch_width = (
                    config.in_channels
                    * config.temporal_patch_size
                    * config.patch_size**2
                )
                grid = torch.tensor(
                    [[1, 16, 16], [2, 12, 16]] if trained else [[1, 4, 4], [2, 4, 6]]
                )
                arguments = dict(
                    pixel_values=torch.randn(int(grid.prod(-1).sum()), patch_width),
                    image_grid_thw=grid,
                )
            with (
                torch.no_grad(),
                (
                    sdpa_kernel(SDPBackend.MATH)
                    if dtype == "float32"
                    else contextlib.nullcontext()
                ),
            ):
                result = model(**arguments)
            main_key = "audio_embeds" if kind == "audio" else "image_embeds"
            first = result[main_key]
            record = dict(
                label=label,
                dtype=dtype,
                component=kind,
                outcome="passed",
                **summary(first),
            )
            attention = (
                model.packed_attention
                if kind == "audio"
                else model.visual.packed_attention
            )
            record["attention_backend"] = type(attention).__name__
            torch.save(
                {
                    key: (
                        value.detach().cpu()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in result.items()
                },
                output / f"{label}_{kind}_{dtype}.pt",
            )
            if kind == "audio":
                arguments["input_features"][1] += 20
                count = int(result["audio_output_lengths"][0])
            else:
                first_patches = int(arguments["image_grid_thw"][0].prod())
                arguments["pixel_values"][first_patches:] += 20
                count = first_patches // model.spatial_merge_size**2
            with (
                torch.no_grad(),
                (
                    sdpa_kernel(SDPBackend.MATH)
                    if dtype == "float32"
                    else contextlib.nullcontext()
                ),
            ):
                second = model(**arguments)[main_key]
            record["first_segment_unchanged"] = bool(
                torch.equal(first[:count], second[:count])
            )
            records.append(record)
            del model
        except Exception as error:
            records.append(
                dict(
                    label=label,
                    dtype=dtype,
                    component=kind,
                    outcome=type(error).__name__,
                    error=str(error),
                )
            )
        torch.cuda.empty_cache()
    print(json.dumps(records), flush=True)


def graph_transitions(root, output, label):
    model = Qwen3OmniAudioEncoder(
        str(root), device="cuda:0", dtype="bfloat16", enable_layer_cuda_graph=True
    )
    assert model.layer_graph_runner is not None
    runner = model.layer_graph_runner
    replay = runner.maybe_replay
    decisions = []

    def record_replay(hidden, cu_seqlens, segments):
        result = replay(hidden, cu_seqlens, segments)
        decisions.append(
            dict(
                tokens=hidden.shape[0],
                windows=len(segments),
                replayed=result is not None,
            )
        )
        return result

    runner.maybe_replay = record_replay
    records = []
    torch.manual_seed(3131)
    for case, lengths in (
        ("captured", [400, 850, 99]),
        ("slot_overflow", [1] * 80),
        ("token_overflow", [33000]),
    ):
        lengths = torch.tensor(lengths)
        width = int(lengths.max())
        arguments = dict(
            input_features=torch.randn(len(lengths), 128, width),
            feature_attention_mask=torch.arange(width).unsqueeze(0)
            < lengths.unsqueeze(1),
        )
        with torch.no_grad():
            actual = model(**arguments)["audio_embeds"]
            runner.disabled_reason = "review forced eager control"
            expected = model(**arguments)["audio_embeds"]
            runner.disabled_reason = None
        records.append(
            dict(
                case=case,
                max_abs=float((actual - expected).abs().max()),
                relative_l2=float(
                    (actual.float() - expected.float()).norm() / expected.float().norm()
                ),
                **summary(actual),
                decisions=decisions[-2:],
            )
        )
        torch.save(
            dict(actual=actual.detach().cpu(), eager=expected.detach().cpu()),
            output / f"{label}_{case}.pt",
        )
        assert actual.isfinite().all()
    print(json.dumps(dict(graph_transitions=records)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--label", default="head")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--trained", action="store_true")
    parser.add_argument("--triton", action="store_true")
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    if options.make:
        make_checkpoint(options.checkpoint)
    else:
        pass
    with (
        override_platform(is_sm90=False) if options.triton else contextlib.nullcontext()
    ):
        if options.graphs:
            graph_transitions(options.checkpoint, options.output, options.label)
        else:
            run(
                options.checkpoint,
                options.label,
                options.output,
                options.dtype,
                options.trained,
            )


if __name__ == "__main__":
    main()
