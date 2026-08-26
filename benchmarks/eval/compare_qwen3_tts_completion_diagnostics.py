# SPDX-License-Identifier: Apache-2.0
"""Compare paired Qwen3-TTS completion artifacts without rerunning generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArmArtifacts:
    output_dir: Path
    generated_by_id: dict[str, dict[str, Any]]
    completion_by_id: dict[str, dict[str, Any]]
    wer_by_id: dict[str, dict[str, Any]]
    wer_summary: dict[str, Any] | None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_identity(entry: dict[str, Any]) -> tuple[int, str]:
    if "seed" not in entry:
        raise ValueError(
            f"generated entry {entry.get('sample_id')!r} has no request seed"
        )
    target_text = str(entry.get("target_text", ""))
    return (
        int(entry["seed"]),
        hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
    )


def _completion_identity(record: dict[str, Any]) -> tuple[int, str]:
    public_seed = record.get("public_seed")
    text_sha256 = record.get("target_text_sha256")
    if public_seed is None or not text_sha256:
        raise ValueError(
            "completion record is missing public_seed or target_text_sha256"
        )
    return int(public_seed), str(text_sha256)


def load_arm_artifacts(output_dir: str | Path) -> ArmArtifacts:
    output_path = Path(output_dir).expanduser().resolve()
    generated_path = output_path / "generated.json"
    if not generated_path.is_file():
        raise FileNotFoundError(f"missing generated artifact: {generated_path}")

    generated_entries = _load_json(generated_path)
    generated_by_id: dict[str, dict[str, Any]] = {}
    identity_to_sample_id: dict[tuple[int, str], str] = {}
    for entry in generated_entries:
        sample_id = str(entry["sample_id"])
        if sample_id in generated_by_id:
            raise ValueError(f"duplicate generated sample ID: {sample_id}")
        generated_by_id[sample_id] = entry
        if not entry.get("is_success", False):
            continue
        identity = _sample_identity(entry)
        previous = identity_to_sample_id.setdefault(identity, sample_id)
        if previous != sample_id:
            raise ValueError(
                "sample-specific seed/text identity collision between "
                f"{previous!r} and {sample_id!r}"
            )

    completion_files = sorted(
        (output_path / "completion_diagnostics").glob("qwen3-tts-completions-*.jsonl")
    )
    if not completion_files:
        raise FileNotFoundError(
            f"no completion JSONL files under {output_path / 'completion_diagnostics'}"
        )

    completion_by_id: dict[str, dict[str, Any]] = {}
    unmatched_records: list[str] = []
    for completion_file in completion_files:
        for line_number, line in enumerate(
            completion_file.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            identity = _completion_identity(record)
            sample_id = identity_to_sample_id.get(identity)
            if sample_id is None:
                unmatched_records.append(f"{completion_file.name}:{line_number}")
                continue
            if sample_id in completion_by_id:
                raise ValueError(
                    "duplicate completion record for sample "
                    f"{sample_id!r}; disable benchmark warmup for diagnostics"
                )
            completion_by_id[sample_id] = record

    if unmatched_records:
        raise ValueError(
            "completion records do not match measured generated entries: "
            + ", ".join(unmatched_records[:10])
        )

    successful_ids = {
        sample_id
        for sample_id, entry in generated_by_id.items()
        if entry.get("is_success", False)
    }
    missing_ids = sorted(successful_ids - completion_by_id.keys())
    if missing_ids:
        raise ValueError(
            f"missing {len(missing_ids)} completion records: {missing_ids[:10]}"
        )

    wer_path = output_path / "wer_results.json"
    wer_by_id: dict[str, dict[str, Any]] = {}
    wer_summary = None
    if wer_path.is_file():
        wer_payload = _load_json(wer_path)
        wer_summary = wer_payload.get("summary")
        wer_by_id = {
            str(entry["id"]): entry for entry in wer_payload.get("per_sample", [])
        }

    return ArmArtifacts(
        output_dir=output_path,
        generated_by_id=generated_by_id,
        completion_by_id=completion_by_id,
        wer_by_id=wer_by_id,
        wer_summary=wer_summary,
    )


def _first_flat_difference(
    left: list[Any],
    right: list[Any],
) -> dict[str, Any] | None:
    shared_length = min(len(left), len(right))
    for index in range(shared_length):
        if left[index] != right[index]:
            return {
                "index": index,
                "left": left[index],
                "right": right[index],
            }
    if len(left) != len(right):
        return {
            "index": shared_length,
            "left": left[shared_length] if shared_length < len(left) else None,
            "right": right[shared_length] if shared_length < len(right) else None,
        }
    return None


def _first_codec_difference(
    left: list[list[Any]],
    right: list[list[Any]],
) -> dict[str, Any] | None:
    shared_frames = min(len(left), len(right))
    for frame in range(shared_frames):
        difference = _first_flat_difference(left[frame], right[frame])
        if difference is not None:
            return {
                "frame": frame,
                "codebook": difference["index"],
                "left": difference["left"],
                "right": difference["right"],
            }
    if len(left) != len(right):
        return {
            "frame": shared_frames,
            "codebook": None,
            "left": left[shared_frames] if shared_frames < len(left) else None,
            "right": right[shared_frames] if shared_frames < len(right) else None,
        }
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_sha256(arm: ArmArtifacts, sample_id: str) -> str:
    raw_path = Path(str(arm.generated_by_id[sample_id]["wav_path"]))
    wav_path = raw_path if raw_path.is_absolute() else arm.output_dir / raw_path
    if not wav_path.is_file():
        raise FileNotFoundError(f"missing WAV for {sample_id!r}: {wav_path}")
    return _sha256_file(wav_path)


def _wer_transcript(arm: ArmArtifacts, sample_id: str) -> str | None:
    entry = arm.wer_by_id.get(sample_id)
    if not entry or not entry.get("is_success", False):
        return None
    return str(entry.get("whisper_text", ""))


def compare_arms(left: ArmArtifacts, right: ArmArtifacts) -> dict[str, Any]:
    left_ids = set(left.completion_by_id)
    right_ids = set(right.completion_by_id)
    if left_ids != right_ids:
        raise ValueError(
            "paired arms have different successful sample IDs: "
            f"left_only={sorted(left_ids - right_ids)[:10]} "
            f"right_only={sorted(right_ids - left_ids)[:10]}"
        )
    if bool(left.wer_by_id) != bool(right.wer_by_id):
        raise ValueError("paired arms must either both have WER artifacts or neither")

    counts = {
        "total": len(left_ids),
        "seed_triplet_identical": 0,
        "completion_tokens_identical": 0,
        "semantic_tokens_identical": 0,
        "codec_codes_identical": 0,
        "wav_identical": 0,
        "finish_reason_identical": 0,
    }
    classifications = {
        "identical": 0,
        "semantic_decoder": 0,
        "code_predictor": 0,
        "vocoder": 0,
        "asr_only": 0,
    }
    differing_samples: list[dict[str, Any]] = []
    identical_wav_different_asr_ids: list[str] = []

    for sample_id in sorted(left_ids):
        left_record = left.completion_by_id[sample_id]
        right_record = right.completion_by_id[sample_id]
        left_tokens = left_record["semantic_token_ids"]
        right_tokens = right_record["semantic_token_ids"]
        left_codes = left_record["generated_codec_codes"]
        right_codes = right_record["generated_codec_codes"]
        semantic_difference = _first_flat_difference(left_tokens, right_tokens)
        codec_difference = _first_codec_difference(left_codes, right_codes)
        left_wav_sha256 = _wav_sha256(left, sample_id)
        right_wav_sha256 = _wav_sha256(right, sample_id)
        wav_identical = left_wav_sha256 == right_wav_sha256
        left_transcript = _wer_transcript(left, sample_id)
        right_transcript = _wer_transcript(right, sample_id)
        transcripts_comparable = (
            left_transcript is not None and right_transcript is not None
        )
        transcript_identical = (
            not transcripts_comparable or left_transcript == right_transcript
        )
        seed_triplet_identical = all(
            left_record.get(key) == right_record.get(key)
            for key in (
                "public_seed",
                "semantic_sampling_seed",
                "subtalker_sampling_seed",
            )
        )

        counts["seed_triplet_identical"] += int(seed_triplet_identical)
        counts["completion_tokens_identical"] += int(
            left_record.get("completion_tokens")
            == right_record.get("completion_tokens")
        )
        counts["semantic_tokens_identical"] += int(semantic_difference is None)
        counts["codec_codes_identical"] += int(codec_difference is None)
        counts["wav_identical"] += int(wav_identical)
        counts["finish_reason_identical"] += int(
            left_record.get("finish_reason") == right_record.get("finish_reason")
        )

        if semantic_difference is not None:
            classification = "semantic_decoder"
        elif codec_difference is not None:
            classification = "code_predictor"
        elif not wav_identical:
            classification = "vocoder"
        elif not transcript_identical:
            classification = "asr_only"
            identical_wav_different_asr_ids.append(sample_id)
        else:
            classification = "identical"
        classifications[classification] += 1

        if classification != "identical" or not seed_triplet_identical:
            left_wer = left.wer_by_id.get(sample_id, {})
            right_wer = right.wer_by_id.get(sample_id, {})
            differing_samples.append(
                {
                    "sample_id": sample_id,
                    "classification": classification,
                    "seed_triplet_identical": seed_triplet_identical,
                    "semantic_difference": semantic_difference,
                    "codec_difference": codec_difference,
                    "left_semantic_tokens": len(left_tokens),
                    "right_semantic_tokens": len(right_tokens),
                    "left_codec_frames": len(left_codes),
                    "right_codec_frames": len(right_codes),
                    "left_finish_reason": left_record.get("finish_reason"),
                    "right_finish_reason": right_record.get("finish_reason"),
                    "left_wav_sha256": left_wav_sha256,
                    "right_wav_sha256": right_wav_sha256,
                    "left_wer": left_wer.get("wer"),
                    "right_wer": right_wer.get("wer"),
                    "left_transcript": left_transcript,
                    "right_transcript": right_transcript,
                }
            )

    return {
        "left": {
            "output_dir": str(left.output_dir),
            "repetition_penalty_owners": sorted(
                {
                    str(record.get("repetition_penalty_owner"))
                    for record in left.completion_by_id.values()
                }
            ),
            "wer_summary": left.wer_summary,
        },
        "right": {
            "output_dir": str(right.output_dir),
            "repetition_penalty_owners": sorted(
                {
                    str(record.get("repetition_penalty_owner"))
                    for record in right.completion_by_id.values()
                }
            ),
            "wer_summary": right.wer_summary,
        },
        "counts": counts,
        "classifications": classifications,
        "identical_wav_different_asr_ids": identical_wav_different_asr_ids,
        "differing_samples": differing_samples,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare paired Qwen3-TTS completion diagnostic arms."
    )
    parser.add_argument("left_output_dir")
    parser.add_argument("right_output_dir")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = compare_arms(
        load_arm_artifacts(args.left_output_dir),
        load_arm_artifacts(args.right_output_dir),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    counts = result["counts"]
    classes = result["classifications"]
    print(f"Compared {counts['total']} paired samples")
    print(
        "Exact semantic / codec / WAV: "
        f"{counts['semantic_tokens_identical']} / "
        f"{counts['codec_codes_identical']} / {counts['wav_identical']}"
    )
    print(
        "First differing boundary: "
        f"semantic={classes['semantic_decoder']} "
        f"code_predictor={classes['code_predictor']} "
        f"vocoder={classes['vocoder']} asr_only={classes['asr_only']}"
    )
    if args.output is not None:
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
