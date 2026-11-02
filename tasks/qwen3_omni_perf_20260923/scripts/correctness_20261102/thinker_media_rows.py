"""Our thinker's prefill rows on images and video against HF, layer by layer and by row group (bisection checks 1 and 2).

hf:      HF's input_layernorm and post_attention_layernorm outputs at LAYERS for each request, in BF16 then FP32, one
         process and one thinker at a time, every row of the prompt.
send:    the requests one at a time to a server whose thinker runs eager with SGLang's dumper on and its radix cache off,
         so each prefill forward dumps its rows; one reply token, because decode steps are the logprob check's.
compare: match = 1 - relative L2 against HF FP32, ours and HF BF16, and ours against HF BF16, for each layer and point over
         text rows, media rows (where deepstack lands), each chunk of a chunked prefill and the rows around its boundary.
         The first layer where ours separates from HF BF16 names the owner; a boundary below its chunks names the split.

usage: PYTHONPATH=TREE python3 thinker_media_rows.py hf OUT DEVICE
       PYTHONPATH=TREE python3 thinker_media_rows.py send URL OUT
       PYTHONPATH=TREE python3 thinker_media_rows.py compare OUT DUMP_DIR
"""

import collections
import gc
import json
import re
import sys
from pathlib import Path

import httpx
import qwen_omni_utils
import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

from tests.test_model.qwen3_omni_component_accuracy.test_thinker_concurrency import (
    device_media,
    image_requests,
    load_thinker,
    video_requests,
)

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
LAYERS = (0, 1, 2, 3, 12, 23, 24, 47)
POINTS = {"input_layernorm": "self_attn.inputs.hidden_states", "post_attention_layernorm": "mlp.inputs.0"}
SAMPLES = ("validation_Agriculture_15", "validation_Agriculture_18", "validation_Accounting_17", "003-1", "007-3")
BOUNDARY_ROWS = 32


def selected_requests(processor: Qwen3OmniMoeProcessor) -> list:
    requests = [*image_requests(processor), *video_requests(processor, qwen_omni_utils.process_mm_info)]
    by_id = {request.sample_id: request for request in requests}
    return [by_id[sample_id] for sample_id in SAMPLES]


def capture(out: Path, device: str, dtype: torch.dtype, label: str) -> None:
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    requests = selected_requests(processor)
    thinker = load_thinker(MODEL, dtype, device)
    for request in requests:
        rows: dict[str, torch.Tensor] = {}
        hooks = []

        def keep(name: str):
            def record(_module, _args, output: torch.Tensor) -> None:
                rows[name] = output[0].float().cpu()

            return record

        for layer in LAYERS:
            for norm in POINTS:
                module = getattr(thinker.model.layers[layer], norm)
                hooks.append(module.register_forward_hook(keep(f"{layer}.{norm}")))
        inputs = request.inputs
        with torch.inference_mode():
            thinker(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                **device_media(inputs, thinker),
                use_cache=False,
            )
        for hook in hooks:
            hook.remove()
        torch.save(rows, out / f"{request.sample_id}.{label}.pt")
        print(f"{label} {request.sample_id} rows {inputs['input_ids'].shape[1]}", flush=True)
    del thinker
    gc.collect()
    torch.cuda.empty_cache()


def send(url: str, out: Path) -> None:
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    requests = selected_requests(processor)
    with httpx.Client(base_url=url, timeout=1800, trust_env=False) as client:
        for request in requests:
            response = client.post(
                "/generate",
                json={
                    **request.body,
                    "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                    "output_modalities": ["text"],
                    "return_logprob": True,
                },
            )
            response.raise_for_status()
            print(f"sent {request.sample_id} rows {request.inputs['input_ids'].shape[1]}", flush=True)
    (out / "sent.txt").write_text("\n".join(SAMPLES))


def load_forwards(dump_dir: Path) -> list[dict[str, Path]]:
    pattern = re.compile(r"step=(\d+).*name=(.*?)(?:___|\.pt$)")
    grouped: dict[int, dict[str, Path]] = collections.defaultdict(dict)
    for path in dump_dir.rglob("*.pt"):
        match = pattern.search(path.name)
        if match is not None:
            grouped[int(match.group(1))][match.group(2).removeprefix("non_intrusive__")] = path
        else:
            pass
    return [grouped[step] for step in sorted(grouped)]


def value_of(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)["value"].float()


def match(left: torch.Tensor, right: torch.Tensor) -> float:
    return 1 - float((left - right).norm() / right.norm().clamp_min(1e-20))


def compare(out: Path, dump_dir: Path) -> None:
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    requests = selected_requests(processor)
    config = load_config_ids()
    probe = f"model.layers.{LAYERS[0]}.{POINTS['input_layernorm']}"
    forwards = [forward for forward in load_forwards(dump_dir) if probe in forward]
    cursor = 0
    report = []
    for request in requests:
        ids = request.inputs["input_ids"][0]
        prompt_rows = ids.shape[0]
        truth = torch.load(out / f"{request.sample_id}.fp32.pt", map_location="cpu", weights_only=False)
        floor = torch.load(out / f"{request.sample_id}.bf16.pt", map_location="cpu", weights_only=False)
        # note (ratish): warmup forwards come first, and a request's chunks add up to its prompt.
        chunks = []
        while sum(chunk[1] for chunk in chunks) != prompt_rows:
            count = value_of(forwards[cursor][probe]).shape[0]
            if not chunks and count > prompt_rows:
                raise RuntimeError(f"{request.sample_id}: forward of {count} rows is past the prompt's {prompt_rows}")
            elif sum(chunk[1] for chunk in chunks) + count > prompt_rows:
                chunks = []
            else:
                chunks.append((cursor, count))
            cursor += 1
        media = (ids == config["image"]) | (ids == config["video"])
        offset = 0
        groups: dict[str, torch.Tensor] = {
            "all": torch.ones(prompt_rows, dtype=torch.bool),
            "text": ~media,
            "media": media,
        }
        bounds = []
        for index, (_, count) in enumerate(chunks):
            mask = torch.zeros(prompt_rows, dtype=torch.bool)
            mask[offset : offset + count] = True
            groups[f"chunk_{index}"] = mask
            offset += count
            bounds.append(offset)
        if len(chunks) > 1:
            mask = torch.zeros(prompt_rows, dtype=torch.bool)
            for bound in bounds[:-1]:
                mask[bound - BOUNDARY_ROWS : bound + BOUNDARY_ROWS] = True
            groups["boundary"] = mask
        else:
            pass
        layers = {}
        for layer in LAYERS:
            for norm, point in POINTS.items():
                name = f"model.layers.{layer}.{point}"
                ours = torch.cat([value_of(forwards[index][name]) for index, _ in chunks])
                key = f"{layer}.{norm}"
                layers[key] = {
                    group: {
                        "ours": match(ours[mask], truth[key][mask]),
                        "floor": match(floor[key][mask], truth[key][mask]),
                        "ours_vs_floor": match(ours[mask], floor[key][mask]),
                    }
                    for group, mask in groups.items()
                    if mask.any()
                }
        record = {
            "sample_id": request.sample_id,
            "modality": request.modality,
            "prompt_rows": prompt_rows,
            "media_rows": int(media.sum()),
            "chunks": [count for _, count in chunks],
            "layers": layers,
        }
        report.append(record)
        print(f"== {request.sample_id} {request.modality} rows {prompt_rows} media {int(media.sum())} chunks {record['chunks']}", flush=True)
        for key, entry in layers.items():
            if key.endswith("input_layernorm"):
                print(
                    f"  {key:>22} "
                    + " ".join(
                        f"{group} {100 * values['ours']:.2f}/{100 * values['floor']:.2f}"
                        for group, values in entry.items()
                    ),
                    flush=True,
                )
    (out / "compare.json").write_text(json.dumps(report, indent=1))
    print("compare done", flush=True)


def load_config_ids() -> dict[str, int]:
    config = Qwen3OmniMoeProcessor.from_pretrained(MODEL).tokenizer
    return {
        "image": config.convert_tokens_to_ids("<|image_pad|>"),
        "video": config.convert_tokens_to_ids("<|video_pad|>"),
    }


def main() -> None:
    if sys.argv[1] == "hf":
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        capture(out, sys.argv[3], torch.bfloat16, "bf16")
        capture(out, sys.argv[3], torch.float32, "fp32")
        print("hf done", flush=True)
    elif sys.argv[1] == "send":
        send(sys.argv[2], Path(sys.argv[3]))
    else:
        compare(Path(sys.argv[2]), Path(sys.argv[3]))


if __name__ == "__main__":
    main()
