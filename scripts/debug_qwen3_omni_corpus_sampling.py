"""Replay every recorded main-code draw without changing its seed or history."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import xxhash
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from transformers import Qwen3OmniMoeConfig

from sglang_omni.models.qwen3_omni.corpus_diagnostics import RequestTrace, StepMetadata
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument(
        "--request-directory", type=Path, action="append", required=True
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    identities: set[str] = set()
    for directory in arguments.request_directory:
        for path in directory.glob("*.request.json"):
            request = json.loads(path.read_text())
            assert isinstance(request, dict)
            assert not any(
                name in request
                for name in (
                    "seed",
                    "sampling_seed",
                    "talker_temperature",
                    "talker_repetition_penalty",
                    "talker_top_k",
                    "talker_top_p",
                )
            )
            request_id = request["request_id"]
            assert isinstance(request_id, str)
            identities.add(request_id)
    assert identities
    traces = {
        request_id: RequestTrace.model_validate_json(
            (
                arguments.capture_directory / "requests" / f"{request_id}.json"
            ).read_text()
        )
        for request_id in identities
    }
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config
    vocab_size = config.text_config.vocab_size
    token_ids = torch.arange(vocab_size, device="cuda")
    allowed = (token_ids < config.code_predictor_config.vocab_size) | (
        token_ids == config.codec_eos_token_id
    )
    counts = {request_id: 0 for request_id in identities}
    mismatches = []
    draws_by_phase: Counter[str] = Counter()
    with (
        (arguments.capture_directory / "steps.jsonl").open() as manifest,
        (arguments.output_directory / "draws.jsonl").open("w") as output,
    ):
        for line in manifest:
            metadata = StepMetadata.model_validate_json(line)
            if not identities.intersection(metadata.request_ids):
                continue
            else:
                pass
            assert (
                ForwardMode[metadata.phase].is_decode()
                or ForwardMode[metadata.phase].is_extend()
            )
            saved = torch.load(
                arguments.capture_directory / f"step-{metadata.step:06d}.pt",
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
            padded_rows = metadata.projection_rows["codec"]
            logits = F.pad(
                saved["snapshots"]["codec.output"].float().cuda(),
                (0, 0, 0, padded_rows - metadata.live_rows),
            )
            repetition_mask = torch.zeros(padded_rows, vocab_size, dtype=torch.bool)
            seeds = torch.zeros(padded_rows, dtype=torch.int64)
            for row, (request_id, frame) in enumerate(
                zip(metadata.request_ids, metadata.frame_indices, strict=True)
            ):
                if request_id not in identities:
                    continue
                else:
                    pass
                seeds[row] = (
                    xxhash.xxh64_intdigest(request_id.encode()) & SAMPLING_SEED_MASK
                )
                previous_codes = [
                    codes[0] for codes in traces[request_id].codes[:frame]
                ]
                repetition_mask[row, previous_codes] = True
            penalties = torch.full(
                (padded_rows, 1),
                arguments.repetition_penalty,
                device="cuda",
                dtype=torch.float32,
            )
            penalized = torch.where(logits > 0, logits / penalties, logits * penalties)
            logits = torch.where(repetition_mask.cuda(), penalized, logits)
            logits.masked_fill_(~allowed, -torch.inf)
            temperature = torch.full(
                (padded_rows, 1),
                arguments.temperature,
                device="cuda",
                dtype=torch.float32,
            )
            probabilities = (logits / temperature).softmax(-1)
            if ForwardMode[metadata.phase].is_extend():
                actual_positions = torch.tensor(metadata.sequence_lengths) - 1
                assert torch.equal(actual_positions, saved["positions"])
            else:
                actual_positions = saved["positions"]
            positions = F.pad(
                actual_positions.cuda(), (0, padded_rows - metadata.live_rows)
            )
            sampled = top_k_top_p_min_p_sampling_from_probs_torch(
                probabilities,
                torch.full(
                    (padded_rows,), arguments.top_k, device="cuda", dtype=torch.int32
                ),
                torch.full(
                    (padded_rows,), arguments.top_p, device="cuda", dtype=torch.float32
                ),
                torch.zeros(padded_rows, device="cuda"),
                False,
                seeds.cuda(),
                positions,
            ).cpu()
            actual = saved["codes"][:, 0]
            report = {
                "step": metadata.step,
                "phase": metadata.phase,
                "request_ids": metadata.request_ids,
                "frames": metadata.frame_indices,
                "positions": saved["positions"].tolist(),
                "seeds": seeds[: metadata.live_rows].tolist(),
                "actual": actual.tolist(),
                "replayed": sampled[: metadata.live_rows].tolist(),
            }
            output.write(json.dumps(report) + "\n")
            for row, request_id in enumerate(metadata.request_ids):
                if request_id not in identities:
                    continue
                else:
                    pass
                counts[request_id] += 1
                draws_by_phase[metadata.phase] += 1
                if sampled[row] != actual[row]:
                    mismatches.append(
                        {
                            "step": metadata.step,
                            "request_id": request_id,
                            "frame": metadata.frame_indices[row],
                            "actual": int(actual[row]),
                            "replayed": int(sampled[row]),
                        }
                    )
                else:
                    pass
    assert all(
        counts[request_id] == len(trace.codes) for request_id, trace in traces.items()
    )
    summary = {
        "model_path": arguments.model_path,
        "model_revision": arguments.model_revision,
        "temperature": arguments.temperature,
        "repetition_penalty": arguments.repetition_penalty,
        "top_k": arguments.top_k,
        "top_p": arguments.top_p,
        "requests": len(identities),
        "main_draws": sum(counts.values()),
        "main_draws_per_request": counts,
        "draws_by_phase": draws_by_phase,
        "mismatches": mismatches,
        "scope": "Actual prefill and decode logits, recorded main-code histories and source-derived request seeds. Prefill uses sequence length minus one, matching the model runner's actual sampling call. Default Talker suppression and output-history repetition penalties are held fixed, not tuned.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(
        json.dumps(
            {
                name: value
                for name, value in summary.items()
                if name not in ("main_draws_per_request", "mismatches")
            }
            | {"mismatch_count": len(mismatches)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
