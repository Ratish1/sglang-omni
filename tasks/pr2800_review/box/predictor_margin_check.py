"""Whether omni's code predictor picks the residual codes HF's would, codebook by codebook, on the served inputs.

Runs only in the isolated reference environment, on the CPU (the predictor is 5 layers of 1024):
    PYTHONPATH=<4.57.1 env> python3 predictor_margin_check.py --trace-dir <dir> [--out <file>]
Reads the traces talker_trace/sitecustomize.py wrote. For every decode step with a served talker hidden, HF's code
predictor (bf16 Instruct weights, run in fp32) reads the served hidden, the served first code and the served residual
codes teacher-forced in one causal pass, so each of the 15 heads sees exactly the prefix omni's predictor saw. Per
codebook: whether HF's argmax equals the served code and, where it does not, the margin, HF's top logit less its logit
at the served code. A mismatch at a margin near zero is a near-tie the served bf16 arithmetic may break either way;
a large one is a code omni's predictor should not have picked. Writes one JSON row a request and prints a summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
)

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
PREDICTOR_PREFIX = "talker.code_predictor."
CODEC_EMBEDDING = "talker.model.codec_embedding.weight"
MARGIN_BOUNDS = (0.05, 0.25, 1.0)


def load_predictor() -> tuple[Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration, torch.Tensor]:
    """HF's code predictor and the talker's codec embedding (the first code's input row), in fp32."""
    config = AutoConfig.from_pretrained(MODEL).talker_config.code_predictor_config
    predictor = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration._from_config(config, torch_dtype=torch.float32)  # noqa: leading-underscore  # HF's own constructor
    root = Path(snapshot_download(MODEL, allow_patterns=["*.json"]))
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({shard for key, shard in weight_map.items()
                     if key.startswith(PREDICTOR_PREFIX) or key == CODEC_EMBEDDING})
    state, codec_embedding = {}, None
    for shard in shards:
        path = Path(snapshot_download(MODEL, allow_patterns=[shard])) / shard
        with safe_open(str(path), "pt") as handle:
            for key in handle.keys():
                if key.startswith(PREDICTOR_PREFIX):
                    state[key[len(PREDICTOR_PREFIX):]] = handle.get_tensor(key).float()
                elif key == CODEC_EMBEDDING:
                    codec_embedding = handle.get_tensor(key).float()
                else:
                    pass
    missing, unexpected = predictor.load_state_dict(state, strict=False)
    print(f"predictor loaded: {len(state)} tensors, missing {missing}, unexpected {unexpected}", flush=True)
    assert codec_embedding is not None
    return predictor.eval(), codec_embedding


@torch.no_grad()
def check_request(predictor, codec_embedding: torch.Tensor, trace: dict) -> dict | None:
    steps = [step for step, hidden in enumerate(trace.get("hidden") or []) if hidden is not None]
    if not steps:
        return None
    else:
        pass
    codebooks = predictor.config.num_code_groups - 1
    embeddings = predictor.get_input_embeddings()
    matches = torch.zeros(codebooks, dtype=torch.long)
    margins, first_mismatch, gaps = [], [], []
    for step in steps:
        codes = trace["codes"][step].view(-1).long()
        rows = [trace["hidden"][step].float().view(1, -1), codec_embedding[codes[0]].view(1, -1)]
        rows += [embeddings[index](codes[index + 1:index + 2]) for index in range(codebooks - 1)]
        hidden = predictor.model(inputs_embeds=torch.cat(rows)[None], use_cache=False).last_hidden_state[0]
        step_first = None
        for index in range(codebooks):
            logits = predictor.lm_head[index](hidden[index + 1])
            served = int(codes[index + 1])
            top2 = logits.topk(2)
            gaps.append(float(top2.values[0] - top2.values[1]))
            if int(top2.indices[0]) == served:
                matches[index] += 1
            else:
                margins.append((index, float(top2.values[0] - logits[served])))
                step_first = index if step_first is None else step_first
        first_mismatch.append(step_first)
    return dict(
        request_id=trace["request_id"], steps=len(steps),
        codebook_agreement=[round(float(count) / len(steps), 4) for count in matches],
        mismatches=len(margins),
        mismatch_margins_over={str(bound): sum(margin > bound for _, margin in margins) for bound in MARGIN_BOUNDS},
        largest_margins=sorted(margins, key=lambda item: -item[1])[:10],
        steps_with_a_mismatch=sum(index is not None for index in first_mismatch),
        median_top_gap=sorted(gaps)[len(gaps) // 2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    predictor, codec_embedding = load_predictor()
    rows = []
    for path in sorted(args.trace_dir.glob("*.pt")):
        row = check_request(predictor, codec_embedding, torch.load(path, weights_only=False))
        if row is None:
            continue
        else:
            pass
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out is not None:
        args.out.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        pass
    steps = sum(row["steps"] for row in rows)
    mismatches = sum(row["mismatches"] for row in rows)
    over = {str(bound): sum(row["mismatch_margins_over"][str(bound)] for row in rows) for bound in MARGIN_BOUNDS}
    print(f"requests {len(rows)}, steps {steps}, residual codes {15 * steps}, mismatches {mismatches}, "
          f"mismatch margins over {over}, steps with a mismatch "
          f"{sum(row['steps_with_a_mismatch'] for row in rows)}", flush=True)


if __name__ == "__main__":
    main()
