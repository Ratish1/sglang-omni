"""Layer by layer, where an audio encoder path departs from HF FP32 on one of the accuracy harness's edge clips.

Run in an arm's tree with the harness copied in (mmmu_accuracy_flow.py does that), one card:
    CUDA_VISIBLE_DEVICES=<card> PYTHONPATH=<arm tree> python3 audio_edge_layers.py --frames 101
The clip is test_audio_encoder.py's edge input: stage 6's clips' features joined and cut to --frames. Paths: HF's own
BF16 tower (the floor), omni's eager encoder as built, and the same encoder with its packed attention removed so every
layer runs Transformers' attention on the stacked q, k and v views. Each line is the relative L2 against HF FP32 at
the layer stack's input and after every layer; the last line per path names the tokens furthest from FP32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from tests.test_model.qwen3_omni_component_accuracy.stage6_requests import (
    load_audio_questions,
)
from tests.test_model.qwen3_omni_component_accuracy.test_audio_encoder import (
    EDGE_FRAMES,
    SAMPLE_RATE,
)

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEVICE = "cuda:0"


def hf_tower(checkpoint: str, dtype: torch.dtype) -> torch.nn.Module:
    tower = hf_modeling.Qwen3OmniMoeAudioEncoder._from_config(  # noqa: leading-underscore  # Transformers construction API
        load_thinker_config(checkpoint).audio_config
    )
    weight_map = json.loads(
        Path(checkpoint, "model.safetensors.index.json").read_text()
    )["weight_map"]
    state = {}
    for shard in sorted(
        {
            file
            for name, file in weight_map.items()
            if name.startswith("thinker.audio_tower.")
        }
    ):
        with safe_open(f"{checkpoint}/{shard}", framework="pt") as handle:
            for name in handle.keys():
                if name.startswith("thinker.audio_tower."):
                    state[name.removeprefix("thinker.audio_tower.")] = (
                        handle.get_tensor(name)
                    )
                else:
                    pass
    tower.load_state_dict(state, strict=True)
    return tower.to(DEVICE, dtype).eval()


def edge_features(frames: int) -> torch.Tensor:
    extractor = Qwen3OmniMoeProcessor.from_pretrained(MODEL).feature_extractor
    features = []
    for question in load_audio_questions():
        waveform, _ = librosa.load(question.audio_path, sr=SAMPLE_RATE)
        features.append(
            extractor(
                waveform,
                sampling_rate=SAMPLE_RATE,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )["input_features"][0]
        )
    joined = torch.cat(features, dim=-1)
    joined = joined.repeat(1, max(EDGE_FRAMES) // joined.shape[-1] + 1)
    return joined[:, :frames].unsqueeze(0)


def layer_states(layers: torch.nn.ModuleList, run) -> list[torch.Tensor]:
    """The layer stack's input, then every layer's output, as float32."""
    states: list[torch.Tensor] = []
    handles = [
        layers[0].register_forward_pre_hook(
            lambda module, args: states.append(args[0].detach().float())
        )
    ]
    handles.extend(
        layer.register_forward_hook(
            lambda module, args, output: states.append(output[0].detach().float())
        )
        for layer in layers
    )
    with torch.inference_mode():
        run()
    for handle in handles:
        handle.remove()
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=101)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = snapshot_download(MODEL)
    features = edge_features(args.frames)
    mask = torch.ones(1, args.frames, dtype=torch.long)
    lengths = mask.sum(dim=1)
    packed = features[0].to(DEVICE)

    def hf_run(tower: torch.nn.Module):
        return lambda: tower(packed.to(tower.dtype), feature_lens=lengths.to(DEVICE))

    def omni_run(encoder: Qwen3OmniAudioEncoder):
        return lambda: encoder(
            input_features=features,
            feature_attention_mask=mask,
            audio_feature_lengths=lengths,
        )

    truth_tower = hf_tower(checkpoint, torch.float32)
    truth = layer_states(truth_tower.layers, hf_run(truth_tower))
    floor_tower = hf_tower(checkpoint, torch.bfloat16)
    packed_encoder = Qwen3OmniAudioEncoder(
        checkpoint, device=DEVICE, enable_layer_cuda_graph=False
    )
    unpacked_encoder = Qwen3OmniAudioEncoder(
        checkpoint, device=DEVICE, enable_layer_cuda_graph=False
    )
    unpacked_encoder.packed_attention = None
    for layer in unpacked_encoder.audio_tower.layers:
        layer.self_attn.packed_attention = None
    paths = {
        "hf bf16": layer_states(floor_tower.layers, hf_run(floor_tower)),
        "omni eager": layer_states(
            packed_encoder.audio_tower.layers, omni_run(packed_encoder)
        ),
        "omni eager, hf attention": layer_states(
            unpacked_encoder.audio_tower.layers, omni_run(unpacked_encoder)
        ),
    }
    print(
        f"frames {args.frames}, tokens {truth[0].shape[0]}, packed attention "
        f"{type(packed_encoder.packed_attention).__name__}"
    )
    for name, states in paths.items():
        errors = [
            ((state - reference).norm() / reference.norm()).item()
            for state, reference in zip(states, truth)
        ]
        print(f"{name:26s} " + " ".join(f"{value:.1e}" for value in errors))
        token_errors = (states[-1] - truth[-1]).norm(dim=-1) / truth[-1].norm(dim=-1)
        worst = token_errors.topk(3)
        print(
            f"{'':26s} worst tokens "
            + ", ".join(
                f"{index} {value:.2e} (norm {truth[-1][index].norm().item():.1f})"
                for value, index in zip(worst.values.tolist(), worst.indices.tolist())
            )
        )


if __name__ == "__main__":
    main()
