"""Our served thinker layer by layer against HF, on identical inputs (correctness plan rows 6, 9, 10).

send: posts the ledger's requests to a server whose thinker runs eager with SGLang's dumper on (dump_env on PYTHONPATH,
DUMPER_* in the environment, one directory per process): HF's processor outputs as the input ids and media, so any
departure is the audio encoder's or the thinker's, greedy, text only, one at a time, the reply tokens and logprobs kept.

compare: reads each thinker forward's dumps (self_attn.inputs.hidden_states is HF's input_layernorm output of that layer,
mlp.inputs.0 HF's post_attention_layernorm output, logits_processor.inputs.1 HF's final norm output, rows as HF's rows)
and reports, per layer and row group, the relative L2 of ours and of HF BF16 against HF FP32 and of ours against HF BF16,
and at the output the overlap of the distributions at temperature 0.7 and top-1 agreement. Steps stop at the first reply
token that differs from the ledger's teacher-forced target text.

usage: python3 served_thinker_dump.py send URL OUT.json SAMPLE_IDS_FILE COUNT
       python3 served_thinker_dump.py compare OUT.json REPLIES.json DUMP_DIR LEDGER_OUT_PREFIX
"""

import base64
import collections
import json
import re
import sys
from pathlib import Path

import httpx
import librosa
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, Qwen3OmniMoeProcessor

from benchmarks.dataset.seedtts import load_seedtts_samples

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEEDTTS_META = "zhaochenyang20/seed-tts-eval-arrow"
TEMPERATURE = 0.7


def serialized_tensor(tensor: torch.Tensor) -> dict:
    contiguous = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(contiguous.dtype).removeprefix("torch."),
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.numpy().tobytes()).decode("ascii"),
    }


def send(url: str, out: str, ids_path: str, count: int) -> None:
    sample_ids = Path(ids_path).read_text().split()[:count]
    snapshot = snapshot_download(MODEL)
    processor = Qwen3OmniMoeProcessor.from_pretrained(snapshot)
    samples = {s.sample_id: s for s in load_seedtts_samples(SEEDTTS_META, None, split="en")}
    replies = []
    with httpx.Client(base_url=url, timeout=900, trust_env=False) as client:
        for sample_id in sample_ids:
            sample = samples[sample_id]
            prompt = (
                f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
                f"Now please read the following text out loud in the same voice and style: "
                f"{sample.target_text}"
            )
            text = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": sample.ref_audio},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            waveform, _ = librosa.load(sample.ref_audio, sr=16000)
            inputs = processor(text=text, audio=[waveform], return_tensors="pt", padding=True)
            response = client.post(
                "/generate",
                json={
                    "input_ids": inputs["input_ids"][0].tolist(),
                    "multimodal_train_inputs": {
                        "version": 1,
                        "tensors": {
                            name: serialized_tensor(inputs[name])
                            for name in ("input_features", "feature_attention_mask")
                        },
                    },
                    "sampling_params": {"temperature": 0.0, "max_new_tokens": 96},
                    "output_modalities": ["text"],
                    "return_logprob": True,
                },
            )
            response.raise_for_status()
            pairs = response.json()["meta_info"]["output_token_logprobs"]
            replies.append(
                {
                    "sample_id": sample_id,
                    "prompt_rows": int(inputs["input_ids"].shape[1]),
                    "tokens": [int(pair[1]) for pair in pairs],
                    "logprobs": [float(pair[0]) for pair in pairs],
                }
            )
            print(sample_id, len(pairs), flush=True)
    Path(out).write_text(json.dumps(replies, indent=1))


def relative(candidate: torch.Tensor, truth: torch.Tensor) -> float:
    return ((candidate - truth).norm() / truth.norm().clamp_min(1e-20)).item()


def overlap(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 1.0 - 0.5 * (p - q).abs().sum(-1)


def load_forwards(dump_dir: Path) -> list[dict]:
    """The dumps by forward, in order: {name: tensor} for each forward that dumped a layer."""
    pattern = re.compile(r"step=(\d+).*dump_index=(\d+).*name=(.*?)(?:___|\.pt$)")
    grouped = collections.defaultdict(dict)
    for path in dump_dir.rglob("*.pt"):
        match = pattern.search(path.name)
        if match is None:
            continue
        step, index, name = int(match.group(1)), int(match.group(2)), match.group(3)
        grouped[step][name.removeprefix("non_intrusive__")] = path
    return [grouped[step] for step in sorted(grouped)]


def value_of(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)["value"].float()


def compare(out: str, replies_path: str, dump_dir: str, ledger_prefix: str) -> None:
    replies = json.loads(Path(replies_path).read_text())
    forwards = [
        forward
        for forward in load_forwards(Path(dump_dir))
        if "model.layers.0.self_attn.inputs.hidden_states" in forward
    ]
    snapshot = snapshot_download(MODEL)
    audio_token_id = AutoConfig.from_pretrained(snapshot).thinker_config.audio_token_id
    index = json.loads((Path(snapshot) / "model.safetensors.index.json").read_text())
    name = "thinker.lm_head.weight"
    with safe_open(Path(snapshot) / index["weight_map"][name], "pt") as handle:
        head = handle.get_tensor(name).float()
    rows_table = collections.defaultdict(list)
    outputs = []
    cursor = 0
    # note (ratish): one request at a time: a prefill forward, then one forward a reply token.
    for request_index, reply in enumerate(replies):
        truth = torch.load(f"{ledger_prefix}.{request_index}.truth.pt", map_location="cpu", weights_only=False)
        assert truth["sample_id"] == reply["sample_id"]
        prompt_rows = truth["prompt_rows"]
        target = truth["input_ids"][prompt_rows:].tolist()
        shared = 0
        while shared < min(len(target), len(reply["tokens"])) and target[shared] == reply["tokens"][shared]:
            shared += 1
        audio_rows = truth["input_ids"] == audio_token_id
        # note (ratish): warmup forwards come first, so a request starts at its whole prompt's prefill.
        while (
            value_of(forwards[cursor]["model.layers.0.self_attn.inputs.hidden_states"]).shape[0]
            != prompt_rows
        ):
            cursor += 1
        steps = forwards[cursor : cursor + len(reply["tokens"])]
        cursor += len(reply["tokens"])
        for step_index, forward in enumerate(steps):
            if step_index > shared:
                break
            first = value_of(forward["model.layers.0.self_attn.inputs.hidden_states"])
            count = first.shape[0]
            if step_index == 0:
                row_slice = slice(prompt_rows - count, prompt_rows)
                phase = "prefill"
            else:
                row = prompt_rows + step_index - 1
                row_slice = slice(row, row + count)
                phase = "decode"
            for layer in range(48):
                for point, norm_name in (
                    (f"model.layers.{layer}.self_attn.inputs.hidden_states", "input_layernorm"),
                    (f"model.layers.{layer}.mlp.inputs.0", "post_attention_layernorm"),
                ):
                    if point not in forward:
                        continue
                    ours = value_of(forward[point])
                    truth_rows = truth["truth_norms"][f"{layer}.{norm_name}"][0, row_slice]
                    floor_rows = truth["floor_norms"][f"{layer}.{norm_name}"][0, row_slice]
                    rows_table[f"{phase}/layer_{layer:02d}/{norm_name}"].append(
                        {
                            "ours": relative(ours, truth_rows),
                            "floor": relative(floor_rows, truth_rows),
                            "ours_vs_floor": relative(ours, floor_rows),
                        }
                    )
                    audio_mask = audio_rows[row_slice]
                    if audio_mask.any():
                        rows_table[f"{phase}_audio/layer_{layer:02d}/{norm_name}"].append(
                            {
                                "ours": relative(ours[audio_mask], truth_rows[audio_mask]),
                                "floor": relative(floor_rows[audio_mask], truth_rows[audio_mask]),
                                "ours_vs_floor": relative(ours[audio_mask], floor_rows[audio_mask]),
                            }
                        )
                    else:
                        pass
            final_name = next((key for key in forward if key.startswith("logits_processor.inputs")), None)
            if final_name is not None and step_index < len(reply["tokens"]):
                ours_final = value_of(forward[final_name])[-1:]
                position = step_index
                if position < truth["truth_logits"].shape[0]:
                    truth_logits = truth["truth_logits"][position : position + 1]
                    floor_logits = truth["floor_logits"][position : position + 1]
                    ours_logits = ours_final @ head.T
                    p_truth = torch.softmax(truth_logits / TEMPERATURE, dim=-1)
                    outputs.append(
                        {
                            "request": request_index,
                            "step": step_index,
                            "ours_overlap": overlap(p_truth, torch.softmax(ours_logits / TEMPERATURE, dim=-1)).item(),
                            "floor_overlap": overlap(p_truth, torch.softmax(floor_logits / TEMPERATURE, dim=-1)).item(),
                            "ours_top1": bool(ours_logits.argmax(-1) == truth_logits.argmax(-1)),
                            "floor_top1": bool(floor_logits.argmax(-1) == truth_logits.argmax(-1)),
                        }
                    )
    summary = {
        key: {
            field: sum(entry[field] for entry in entries) / len(entries)
            for field in entries[0]
        }
        | {"count": len(entries)}
        for key, entries in sorted(rows_table.items())
    }
    report = {
        "rows": summary,
        "outputs": {
            field: sum(entry[field] for entry in outputs) / max(len(outputs), 1)
            for field in ("ours_overlap", "floor_overlap", "ours_top1", "floor_top1")
        }
        | {"steps": len(outputs), "worst_ours_overlap": min((e["ours_overlap"] for e in outputs), default=None)},
    }
    Path(out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["outputs"], indent=1))
    for key in sorted(summary):
        if key.endswith("input_layernorm") and re.search(r"layer_(00|08|16|23|24|32|40|47)", key):
            entry = summary[key]
            print(
                f"{key}: ours {entry['ours']:.4f} floor {entry['floor']:.4f} ours_vs_floor {entry['ours_vs_floor']:.4f} ({entry['count']})"
            )


def main() -> None:
    if sys.argv[1] == "send":
        send(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


if __name__ == "__main__":
    main()
