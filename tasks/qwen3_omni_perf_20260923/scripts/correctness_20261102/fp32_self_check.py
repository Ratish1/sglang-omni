"""Why HF FP32 departs from itself between its cached generate and an uncached forward over the same tokens.

Takes the requests whose FP32 recording departed in a test_thinker_concurrency report, and one departure-free image and
audio request as controls. For each: the greedy cached generate and the uncached forward over its tokens, then the
position ids the model read in both, the match of the logits at every step (1 - relative L2), and at the worst step the
match of the last row's hidden state at every layer. Equal positions with a match falling through the layers is rounding;
unequal positions is a position defect in the reference.

usage: PYTHONPATH=TREE python3 fp32_self_check.py REPORT_JSON OUT_JSON   (HF on cuda:1)
"""

import json
import sys
from pathlib import Path

import qwen_omni_utils
import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

from tests.test_model.qwen3_omni_component_accuracy.test_thinker_concurrency import (
    audio_requests,
    device_media,
    image_requests,
    load_thinker,
    video_requests,
)
from tests.test_model.test_qwen3_omni_mmmu_talker_ci import MAX_TOKENS

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEVICE = "cuda:1"


def match(left: torch.Tensor, right: torch.Tensor) -> float:
    return 1 - float((left.float() - right.float()).norm() / right.float().norm())


def spatial_positions(position_ids: torch.Tensor) -> torch.Tensor:
    """The model's three MRoPE axes: generate adds a leading text axis the uncached forward does not."""
    return position_ids[-3:].cpu()


def check(thinker: torch.nn.Module, request, eos_token_id: int) -> dict:
    positions: list[torch.Tensor] = []
    layer_rows: list[list[torch.Tensor]] = [[] for _ in thinker.model.layers]
    kept_rows = [slice(-1, None)]

    def record(_module, _args, kwargs) -> None:
        positions.append(spatial_positions(kwargs["position_ids"]))

    def keep_layer_rows(index: int):
        def keep(_module, _args, output) -> None:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            layer_rows[index].append(hidden[0, kept_rows[0]].float().cpu())

        return keep

    hooks = [thinker.model.register_forward_pre_hook(record, with_kwargs=True)]
    hooks += [layer.register_forward_hook(keep_layer_rows(index)) for index, layer in enumerate(thinker.model.layers)]
    inputs = request.inputs
    prompt_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        cached = thinker.generate(
            input_ids=inputs["input_ids"].to(thinker.device),
            attention_mask=inputs["attention_mask"].to(thinker.device),
            **device_media(inputs, thinker),
            max_new_tokens=MAX_TOKENS,
            eos_token_id=eos_token_id,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
        tokens = cached.sequences[0, prompt_length:]
        cached_positions = positions[:]
        positions.clear()
        cached_layers = [rows[:] for rows in layer_rows]
        for rows in layer_rows:
            rows.clear()
        start = prompt_length - 1
        kept_rows[0] = slice(start, start + tokens.shape[0])
        full_ids = torch.cat([inputs["input_ids"][0].to(thinker.device), tokens])
        uncached = thinker(
            input_ids=full_ids[None],
            attention_mask=torch.ones_like(full_ids)[None],
            **device_media(inputs, thinker),
            use_cache=False,
        )
    for hook in hooks:
        hook.remove()
    uncached_positions = positions[0]
    position_steps = []
    for step, step_positions in enumerate(cached_positions):
        row_start = 0 if step == 0 else prompt_length + step - 1
        rows = step_positions.shape[-1]
        position_steps.append(bool(torch.equal(step_positions, uncached_positions[..., row_start : row_start + rows])))
    cached_logits = torch.stack([step[0].float() for step in cached.logits])
    uncached_logits = uncached.logits[0, start : start + tokens.shape[0]].float()
    steps = [match(cached_logits[step], uncached_logits[step]) for step in range(tokens.shape[0])]
    departures = (uncached_logits.argmax(dim=-1) != tokens).nonzero().flatten().tolist()
    worst = departures[0] if departures else min(range(len(steps)), key=steps.__getitem__)
    layers = [
        match(cached_rows[worst][0], uncached_rows[0][worst])
        for cached_rows, uncached_rows in zip(cached_layers, layer_rows)
    ]
    top2 = uncached_logits.topk(2, dim=-1).values
    return {
        "sample_id": request.sample_id,
        "modality": request.modality,
        "prompt_tokens": prompt_length,
        "steps": len(steps),
        "positions_equal_all_steps": all(position_steps),
        "position_steps_unequal": [step for step, equal in enumerate(position_steps) if not equal][:10],
        "logits_match_mean": sum(steps) / len(steps),
        "logits_match_min": min(steps),
        "logits_match_first_step": steps[0],
        "departures": departures,
        "departure_margins": (top2[:, 0] - top2[:, 1])[departures].tolist(),
        "worst_step": worst,
        "layers_match_at_worst_step": layers,
    }


def main() -> None:
    report = json.loads(Path(sys.argv[1]).read_text())
    rows = report["per_request"]
    departed = {row["sample_id"] for row in rows if row["truth"]["departure_steps"]}
    controls = {
        next(row["sample_id"] for row in rows if row["modality"] == modality and not row["truth"]["departure_steps"])
        for modality in ("image", "audio", "video")
    }
    wanted = departed | controls
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    requests = [
        request
        for request in (*image_requests(processor), *audio_requests(processor), *video_requests(processor, qwen_omni_utils.process_mm_info))
        if request.sample_id in wanted
    ]
    thinker = load_thinker(MODEL, torch.float32, DEVICE)
    records = []
    for request in requests:
        record = check(thinker, request, eos_token_id)
        records.append(record)
        layers = record["layers_match_at_worst_step"]
        print(
            f"{record['sample_id']} {record['modality']} prompt {record['prompt_tokens']} steps {record['steps']}: positions equal "
            f"{record['positions_equal_all_steps']}, logits match mean {record['logits_match_mean']:.5f} min {record['logits_match_min']:.5f} "
            f"first {record['logits_match_first_step']:.5f}, departures {record['departures']} margins "
            f"{[round(margin, 3) for margin in record['departure_margins']]}, at step {record['worst_step']} layers "
            f"0 {layers[0]:.5f} 12 {layers[12]:.5f} 24 {layers[24]:.5f} 36 {layers[36]:.5f} last {layers[-1]:.5f}, "
            f"lowest {min(layers):.5f} at {layers.index(min(layers))}",
            flush=True,
        )
        Path(sys.argv[2]).write_text(json.dumps(records, indent=1))
    print("self check done", flush=True)


if __name__ == "__main__":
    main()
