"""Where HF FP32 stops matching itself on a long prompt: determinism, then the forward's length, layer by layer.

Uncached forwards of one request: A and B are the prompt alone, twice; C is the prompt with EXTRA_TOKENS appended. A causal
model's prompt rows cannot depend on rows after them, so any difference in A against C is arithmetic. For each pair, per
layer over the prompt rows: bit identity, largest absolute difference, match (1 - relative L2), the rows whose top-k
expert set differs, and the first layer and row of each. Each pair runs at SGLang's and torch's default attention backend
and with the backend forced, which separates the attention kernel from the rest.

usage: PYTHONPATH=TREE python3 fp32_shape_bisect.py OUT_JSON   (HF on cuda:1)
"""

import json
import sys
from pathlib import Path

import qwen_omni_utils
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

from tests.test_model.qwen3_omni_component_accuracy.test_thinker_concurrency import (
    device_media,
    image_requests,
    load_thinker,
    video_requests,
)

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEVICE = "cuda:1"
SAMPLES = ("001-2", "validation_Agriculture_15")
EXTRA_TOKENS = 8
BACKENDS = {
    "default": None,
    "flash": [SDPBackend.FLASH_ATTENTION],
    "cudnn": [SDPBackend.CUDNN_ATTENTION],
}


def forward_rows(thinker: torch.nn.Module, request, extra_ids: torch.Tensor | None) -> dict[str, list[torch.Tensor]]:
    prompt_rows = request.inputs["input_ids"].shape[1]
    hidden: list[torch.Tensor] = []
    experts: list[torch.Tensor] = []
    hooks = []
    for layer in thinker.model.layers:

        def keep_hidden(_module, _args, output) -> None:
            value = output if isinstance(output, torch.Tensor) else output[0]
            hidden.append(value[0, :prompt_rows].cpu())

        def keep_experts(_module, _args, output) -> None:
            experts.append(output[2][:prompt_rows].sort(dim=-1).values.cpu())

        hooks.append(layer.register_forward_hook(keep_hidden))
        hooks.append(layer.mlp.gate.register_forward_hook(keep_experts))
    ids = request.inputs["input_ids"].to(thinker.device)
    if extra_ids is not None:
        ids = torch.cat([ids, extra_ids[None].to(thinker.device)], dim=1)
    else:
        pass
    with torch.inference_mode():
        thinker(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            **device_media(request.inputs, thinker),
            use_cache=False,
        )
    for hook in hooks:
        hook.remove()
    return {"hidden": hidden, "experts": experts}


def compare_rows(left: dict, right: dict) -> dict:
    layers = []
    for index, (left_hidden, right_hidden) in enumerate(zip(left["hidden"], right["hidden"])):
        difference = left_hidden - right_hidden
        different_rows = (difference != 0).any(dim=-1)
        flips = (left["experts"][index] != right["experts"][index]).any(dim=-1)
        layers.append(
            {
                "layer": index,
                "identical": not bool(different_rows.any()),
                "rows_different": int(different_rows.sum()),
                "first_row_different": int(different_rows.nonzero()[0]) if different_rows.any() else None,
                "max_abs": float(difference.abs().max()),
                "match": 1 - float(difference.norm() / right_hidden.norm()),
                "rows_expert_set_differs": int(flips.sum()),
                "first_row_expert_flip": int(flips.nonzero()[0]) if flips.any() else None,
            }
        )
    first_different = next((entry["layer"] for entry in layers if not entry["identical"]), None)
    first_flip = next((entry["layer"] for entry in layers if entry["rows_expert_set_differs"]), None)
    return {
        "first_layer_different": first_different,
        "first_layer_expert_flip": first_flip,
        "total_rows_expert_set_differs": sum(entry["rows_expert_set_differs"] for entry in layers),
        "last_match": layers[-1]["match"],
        "lowest_match": min(entry["match"] for entry in layers),
        "layers": layers,
    }


def show(label: str, result: dict) -> None:
    picked = " ".join(
        f"{entry['layer']}:{entry['match']:.6f}/{entry['rows_expert_set_differs']}"
        for entry in result["layers"]
        if entry["layer"] in (0, 6, 12, 18, 24, 30, 36, 42, 47)
    )
    print(
        f"  {label}: first different layer {result['first_layer_different']}, first expert flip layer "
        f"{result['first_layer_expert_flip']}, flips {result['total_rows_expert_set_differs']}, last match "
        f"{result['last_match']:.6f}, lowest {result['lowest_match']:.6f}; layer:match/flips {picked}",
        flush=True,
    )


def main() -> None:
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    requests = {
        request.sample_id: request
        for request in (*image_requests(processor), *video_requests(processor, qwen_omni_utils.process_mm_info))
        if request.sample_id in SAMPLES
    }
    extra_ids = processor.tokenizer("The answer is that it goes on", return_tensors="pt")["input_ids"][0][:EXTRA_TOKENS]
    thinker = load_thinker(MODEL, torch.float32, DEVICE)
    report = {}
    for sample_id in SAMPLES:
        request = requests[sample_id]
        print(f"== {sample_id} {request.modality} rows {request.inputs['input_ids'].shape[1]}", flush=True)
        for name, backends in BACKENDS.items():

            def run(extra: torch.Tensor | None) -> dict:
                if backends is None:
                    return forward_rows(thinker, request, extra)
                else:
                    with sdpa_kernel(backends):
                        return forward_rows(thinker, request, extra)

            try:
                first = run(None)
                second = run(None)
                longer = run(extra_ids)
            except RuntimeError as error:
                print(f"  {name}: unavailable: {str(error)[:160]}", flush=True)
                report[f"{sample_id}/{name}"] = {"error": str(error)[:400]}
                torch.cuda.empty_cache()
                continue
            repeat = compare_rows(first, second)
            shape = compare_rows(first, longer)
            show(f"{name} same forward twice", repeat)
            show(f"{name} prompt against prompt+{extra_ids.shape[0]}", shape)
            report[f"{sample_id}/{name}"] = {"repeat": repeat, "shape": shape}
            Path(sys.argv[1]).write_text(json.dumps(report, indent=1))
            del first, second, longer
    print("shape bisect done", flush=True)


if __name__ == "__main__":
    main()
