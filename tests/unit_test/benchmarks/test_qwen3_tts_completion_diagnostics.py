from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.eval.compare_qwen3_tts_completion_diagnostics import (
    compare_arms,
    load_arm_artifacts,
)


def _write_arm(
    output_dir: Path,
    *,
    arm: str,
    sample_rows: list[dict],
) -> None:
    audio_dir = output_dir / "audio"
    diagnostics_dir = output_dir / "completion_diagnostics"
    audio_dir.mkdir(parents=True)
    diagnostics_dir.mkdir()
    generated = []
    completions = []
    wer_rows = []
    for row in sample_rows:
        sample_id = row["sample_id"]
        text = row["text"]
        wav_path = audio_dir / f"{sample_id}.wav"
        wav_path.write_bytes(row["wav"])
        generated.append(
            {
                "sample_id": sample_id,
                "target_text": text,
                "wav_path": str(wav_path),
                "is_success": True,
                "seed": row["seed"],
            }
        )
        completions.append(
            {
                "record_type": "qwen3_tts_completion",
                "request_id": f"{arm}-{sample_id}",
                "target_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "public_seed": row["seed"],
                "semantic_sampling_seed": row["semantic_seed"],
                "subtalker_sampling_seed": row["subtalker_seed"],
                "repetition_penalty_owner": arm,
                "semantic_token_ids": row["semantic_tokens"],
                "generated_codec_codes": row["codec_codes"],
                "completion_tokens": len(row["codec_codes"]),
                "finish_reason": "stop",
            }
        )
        wer_rows.append(
            {
                "id": sample_id,
                "wer": row["wer"],
                "whisper_text": row["transcript"],
                "is_success": True,
            }
        )

    (output_dir / "generated.json").write_text(json.dumps(generated))
    (diagnostics_dir / f"qwen3-tts-completions-{arm}.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in completions)
    )
    (output_dir / "wer_results.json").write_text(
        json.dumps({"summary": {"wer_corpus": 0.01}, "per_sample": wer_rows})
    )


def _rows(*, divergent: bool, asr_variant: bool) -> list[dict]:
    return [
        {
            "sample_id": "a",
            "text": "alpha",
            "seed": 11,
            "semantic_seed": 101,
            "subtalker_seed": 201,
            "semantic_tokens": [1, 2, 9],
            "codec_codes": [[10, 11], [12, 13]],
            "wav": b"same-wav",
            "wer": 0.0,
            "transcript": "ALPHA" if asr_variant else "alpha",
        },
        {
            "sample_id": "b",
            "text": "beta",
            "seed": 12,
            "semantic_seed": 102,
            "subtalker_seed": 202,
            "semantic_tokens": [3, 5 if divergent else 4, 9],
            "codec_codes": [[20, 21], [24 if divergent else 22, 23]],
            "wav": b"right-wav" if divergent else b"left-wav",
            "wer": 0.1 if divergent else 0.0,
            "transcript": "beta changed" if divergent else "beta",
        },
    ]


def test_completion_comparator_localizes_first_boundary_and_asr_only(
    tmp_path: Path,
) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    _write_arm(
        left_dir, arm="sglang", sample_rows=_rows(divergent=False, asr_variant=False)
    )
    _write_arm(
        right_dir, arm="qwen", sample_rows=_rows(divergent=True, asr_variant=True)
    )

    result = compare_arms(
        load_arm_artifacts(left_dir),
        load_arm_artifacts(right_dir),
    )

    assert result["counts"]["total"] == 2
    assert result["counts"]["seed_triplet_identical"] == 2
    assert result["counts"]["semantic_tokens_identical"] == 1
    assert result["counts"]["codec_codes_identical"] == 1
    assert result["counts"]["wav_identical"] == 1
    assert result["classifications"] == {
        "identical": 0,
        "semantic_decoder": 1,
        "code_predictor": 0,
        "vocoder": 0,
        "asr_only": 1,
    }
    assert result["identical_wav_different_asr_ids"] == ["a"]
    by_id = {row["sample_id"]: row for row in result["differing_samples"]}
    assert by_id["b"]["semantic_difference"] == {
        "index": 1,
        "left": 4,
        "right": 5,
    }


def test_completion_loader_rejects_duplicate_warmup_record(tmp_path: Path) -> None:
    output_dir = tmp_path / "arm"
    _write_arm(
        output_dir,
        arm="sglang",
        sample_rows=_rows(divergent=False, asr_variant=False),
    )
    path = next((output_dir / "completion_diagnostics").glob("*.jsonl"))
    first_line = path.read_text().splitlines()[0]
    with path.open("a") as handle:
        handle.write(first_line + "\n")

    with pytest.raises(ValueError, match="disable benchmark warmup"):
        load_arm_artifacts(output_dir)
