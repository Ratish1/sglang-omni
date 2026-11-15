"""Inspect YuE2 top-k/top-p support using the pinned reference and real weights."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn

from check import DIRECTORY, MODEL, VAE, reference_module

from sglang_omni.models.yue2 import model_runner
from sglang_omni.models.yue2.model_runner import Yue2ModelRunner
from sglang_omni.models.yue2.request import (
    CODEC_OFFSET,
    CODEC_SIZE,
    SamplingConfig,
    Yue2Tokenizer,
    build_request,
)
from sglang_omni.models.yue2.runtime import Yue2Runtime
from sglang_omni.models.yue2.sglang_request_builder import Yue2ARRequestData
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.utils.checkpoint import resolve_checkpoint


@torch.inference_mode()
def compare_support(logits: torch.Tensor) -> dict[str, int | float]:
    sampling = SamplingConfig(repetition_penalty=1.0, min_tokens=0)
    reference = reference_module("sampling.py")
    expected = reference.distribution(logits, sampling, [], 0, "semantic", True)
    captured: dict[str, torch.Tensor] = {}
    native_topk = torch.Tensor.topk
    native_multinomial = model_runner.multinomial_with_seed

    def capture_topk(tensor, *arguments, **keywords):
        selected = native_topk(tensor, *arguments, **keywords)
        captured["candidates"] = selected.indices.clone()
        return selected

    def capture_multinomial(values, seeds, positions):
        captured["values"] = values.clone()
        return native_multinomial(values, seeds, positions)

    with patch.object(torch.Tensor, "topk", capture_topk), patch.object(
        model_runner, "multinomial_with_seed", capture_multinomial
    ):
        Yue2ModelRunner.sample_group(
            logits,
            [Yue2ARRequestData(sampling=sampling, sampling_seed=42)],
            "semantic",
            True,
        )
    actual = torch.full_like(expected, -torch.inf)
    actual.scatter_(-1, captured["candidates"], captured["values"])
    expected_probabilities = expected.float().softmax(-1)
    actual_probabilities = actual.float().softmax(-1)
    threshold = logits[:, CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE].topk(100).values[:, -1:]
    return {
        "reference_support": int(torch.isfinite(expected).sum()),
        "pr_support": int(torch.isfinite(actual).sum()),
        "total_variation": float((expected_probabilities - actual_probabilities).abs().sum() / 2),
        "reference_probability_outside_pr_support": float(expected_probabilities.masked_select(~torch.isfinite(actual)).sum()),
        "top_k_boundary_ties": int((logits[:, CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] == threshold).sum()),
    }


@torch.inference_mode()
def main() -> None:
    output: dict[str, dict[str, int | float]] = {}
    flat = torch.zeros(1, 184704, device="cuda", dtype=torch.bfloat16)
    output["flat_bf16_contract"] = compare_support(flat)
    runtime = Yue2Runtime.from_pretrained(
        MODEL, VAE, torch.device("cuda:0"), vae_core_frames=512, vae_halo_frames=16
    )
    official = reference_module("modeling_yue2.py")
    configuration = SimpleNamespace(**runtime.model.config.model_dump())
    with torch.device("meta"):
        backbone = official.Backbone(configuration)
        token_head = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)
    backbone.load_state_dict(runtime.model.model.state_dict(), assign=True, strict=True)
    backbone.eval()
    directory = Path(resolve_checkpoint(MODEL))
    mapping = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(directory / mapping["lm_head.weight"], framework="pt", device="cpu") as archive:
        token_head.load_state_dict({"weight": archive.get_tensor("lm_head.weight").cuda()}, assign=True)
    token_head.eval()
    tokenizer = Yue2Tokenizer(directory / "qwen.tiktoken")
    for index, style in enumerate([
        "Gentle acoustic pop, warm female vocals",
        "Fast electronic dance music, powerful male vocals",
        "Jazz piano, relaxed female vocals",
    ]):
        payload = StagePayload(
            request_id=f"sampling-{index}", data={},
            request=OmniRequest(
                inputs="[Verse]\nMorning light is shining through",
                metadata={"tts_params": {"instructions": style, "seed": 42}},
                params={"stage_params": {"synthesis": {"cot": "off"}}},
            ),
        )
        request = build_request(payload)
        prefixes = [tokenizer.prefix(request, []), tokenizer.negative_prefix(request, [])]
        branch_logits = []
        for prefix in prefixes:
            tokens = torch.tensor([prefix], device="cuda")
            positions = torch.arange(len(prefix), device="cuda")[None]
            hidden, _ = backbone(input_ids=tokens, position_ids=positions, use_cache=False)
            branch_logits.append(token_head(hidden[:, -1]))
        conditional, unconditional = branch_logits
        logits = unconditional + request.guidance * (conditional - unconditional)
        output[f"checkpoint_prompt_{index}"] = compare_support(logits)
    (DIRECTORY / "sampling-support.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
