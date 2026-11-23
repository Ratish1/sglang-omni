# SPDX-License-Identifier: Apache-2.0
"""Verify loaded Talker and waveform-decoder tensors against the raw checkpoint."""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

EXPERT_WEIGHT = re.compile(
    r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)


def load_talker(
    checkpoint: Path, gpu_index: int, destination: Path
) -> Qwen3OmniMoeTalkerForConditionalGeneration:
    config = Qwen3OmniMoeConfig.from_pretrained(checkpoint).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    model, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        checkpoint,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": gpu_index},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
    model.eval()
    model.code_predictor.generation_config.do_sample = False
    audit_weights(model, checkpoint, destination)
    return model


def audit_weights(
    model: Qwen3OmniMoeTalkerForConditionalGeneration | Qwen3OmniMoeCode2Wav,
    checkpoint: Path,
    destination: Path,
) -> None:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    checkpoint_prefix = (
        "code2wav." if isinstance(model, Qwen3OmniMoeCode2Wav) else "talker."
    )
    tensors = model.state_dict(keep_vars=True)
    shards: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        if name.startswith(checkpoint_prefix):
            shards.setdefault(shard, []).append(name)
        else:
            pass
    visited: set[int] = set()
    checked_tensors = 0
    for shard, names in sorted(shards.items()):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as weights:
            for checkpoint_name in names:
                expected = weights.get_tensor(checkpoint_name)
                name = checkpoint_name.removeprefix(checkpoint_prefix)
                expert = EXPERT_WEIGHT.fullmatch(name)
                if expert is None:
                    tensor = tensors[name]
                    actual = tensor
                else:
                    prefix, expert_index, projection = expert.groups()
                    if projection == "down_proj":
                        tensor = tensors[f"{prefix}.down_proj"]
                        actual = tensor[int(expert_index)]
                    else:
                        tensor = tensors[f"{prefix}.gate_up_proj"]
                        rows = expected.shape[0]
                        start = 0 if projection == "gate_proj" else rows
                        actual = tensor[int(expert_index), start : start + rows]
                assert actual.dtype == expected.dtype, checkpoint_name
                assert torch.equal(actual, expected.to(actual.device)), checkpoint_name
                visited.add(id(tensor))
                checked_tensors += 1
    assert all(id(tensor) in visited for tensor in tensors.values())
    (destination / "weights.json").write_text(
        json.dumps({"raw_checkpoint_tensors_verified": checked_tensors}, indent=2)
    )
