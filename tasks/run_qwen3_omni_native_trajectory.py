# SPDX-License-Identifier: Apache-2.0
"""Capture selected speech trajectories under full speech workloads."""

import argparse
import asyncio
import json
import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import run_qwen3_omni_rounded_add_screen as benchmark
from qwen3_omni_corpus_support import (
    MMSU_TTS_PROMPT,
    REQUEST_SEED,
    SERVER_TIMEOUT_SECONDS,
    server_command,
)

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.mmsu import MmsuSample, load_mmsu_samples
from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu
from benchmarks.tasks.audio_understanding import (
    _build_request_payload as build_request_payload,
)
from sglang_omni.utils import find_available_port

SELECTED_SAMPLES = (
    "common_voice_en_21555942-common_voice_en_21555950",
    "common_voice_en_24523505-common_voice_en_24523504",
    "common_voice_en_23639587-common_voice_en_23639589",
    "common_voice_en_19691556-common_voice_en_19691557",
)
STAGES = ("image_encoder", "audio_encoder", "thinker", "talker_ar", "code2wav")


def identified_mmsu_payload(
    sample: MmsuSample,
    *,
    model_name: str,
    prompt: str,
    modalities: list[str],
    max_tokens: int,
    temperature: float,
    seed: int | None = None,
) -> dict[str, object]:
    payload = build_request_payload(
        sample,
        model_name=model_name,
        prompt=prompt,
        modalities=modalities,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    payload["request_id"] = f"feedback-trace-native-{sample.sample_id}"
    return payload


def main(
    output: Path,
    checkpoint: Path,
    tactic_cache: Path,
    selected_samples: list[str],
    client_concurrency: int,
    mmsu: bool,
    attention_trial_sequences: list[int],
    attention_trial_splits: int,
    code_trial_reference: Path | None,
    code_trial_choice: list[int],
    cache_trial_reference: list[Path],
    cache_target_reference: Path | None,
    continuation_reference: Path | None,
    continuation_target: Path | None,
    continuation_sequences: list[int],
    layer_frames: list[int],
    conditioning_reference: list[Path],
    qk_trial: str | None,
) -> None:
    assert all(frame > 0 for frame in layer_frames)
    assert (code_trial_reference is None) == (not code_trial_choice)
    if code_trial_reference is not None:
        assert not attention_trial_sequences
        assert len(selected_samples) == 1
    else:
        pass
    if cache_trial_reference:
        assert code_trial_reference is None
        assert len(selected_samples) == 1
        assert len(cache_trial_reference) == len(attention_trial_sequences)
        assert attention_trial_sequences == sorted(set(attention_trial_sequences))
        assert attention_trial_splits == 1
    else:
        assert cache_target_reference is None
    if continuation_reference is not None:
        assert continuation_target is not None
        assert len(continuation_sequences) == 2
        assert continuation_sequences[0] < continuation_sequences[1]
        assert len(selected_samples) == 1
        assert code_trial_reference is None and not cache_trial_reference
        assert not attention_trial_sequences
    else:
        assert continuation_target is None and not continuation_sequences
    if conditioning_reference:
        assert {reference.name for reference in conditioning_reference} == set(
            selected_samples
        )
        assert len(conditioning_reference) == len(selected_samples)
        assert code_trial_reference is None and not cache_trial_reference
        assert continuation_reference is None and not attention_trial_sequences
    else:
        pass
    if qk_trial is not None:
        assert conditioning_reference and layer_frames
    else:
        pass
    output.mkdir()
    if conditioning_reference:
        (output / "conditioning-references.json").write_text(
            json.dumps(
                {
                    reference.name: str(reference)
                    for reference in conditioning_reference
                },
                indent=2,
            )
            + "\n"
        )
    else:
        pass
    samples = load_seedtts_samples(SEEDTTS_DATASET_ID)
    sample_ids = {sample.sample_id for sample in samples}
    assert len(samples) == len(sample_ids) == 1088
    if mmsu:
        mmsu_samples = load_mmsu_samples(repo_id="ddwang2000/MMSU")
        mmsu_ids = {sample.sample_id for sample in mmsu_samples}
        assert len(mmsu_samples) == len(mmsu_ids) == 5000
        assert set(selected_samples) <= mmsu_ids
    else:
        assert set(selected_samples) <= sample_ids
        assert (
            samples[0].sample_id not in selected_samples
        ), "Concurrent warmup copies must not share a trace request ID"
    shutil.copytree(tactic_cache, output / "cache")
    captures = output / "captures"
    captures.mkdir()
    port = find_available_port()
    server = start_server_from_cmd(
        [
            *server_command(),
            "--port",
            str(port),
            "--model-path",
            str(checkpoint),
            "--model-name",
            "qwen3-omni",
            "--config",
            "examples/configs/qwen3_omni_colocated_h200.yaml",
            *(argument for stage in STAGES for argument in (f"--{stage}.gpu", "0")),
        ],
        output / "server.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        env={
            "SGLANG_CACHE_DIR": str(output / "cache"),
            "OMNI_NATIVE_TRAJECTORY_DIRECTORY": str(captures),
            "OMNI_NATIVE_TRAJECTORY_SAMPLES": ",".join(selected_samples),
            "OMNI_NATIVE_LAYER_FRAMES": ",".join(str(frame) for frame in layer_frames),
            "OMNI_NATIVE_QK_TRIAL": qk_trial or "",
            "OMNI_NATIVE_CONDITIONING_REFERENCES": json.dumps(
                [str(reference) for reference in conditioning_reference]
            ),
            "OMNI_NATIVE_ATTENTION_TRIAL_SEQUENCES": ",".join(
                str(value) for value in attention_trial_sequences
            ),
            "OMNI_NATIVE_ATTENTION_TRIAL_SPLITS": str(attention_trial_splits),
            "OMNI_NATIVE_CODE_TRIAL_REFERENCE": (
                str(code_trial_reference) if code_trial_reference else ""
            ),
            "OMNI_NATIVE_CODE_TRIAL_CHOICE": ",".join(
                str(value) for value in code_trial_choice
            ),
            "OMNI_NATIVE_CACHE_TRIAL_REFERENCES": json.dumps(
                [str(reference) for reference in cache_trial_reference]
            ),
            "OMNI_NATIVE_CACHE_TARGET_REFERENCE": (
                str(cache_target_reference) if cache_target_reference else ""
            ),
            "OMNI_NATIVE_CONTINUATION_REFERENCE": (
                str(continuation_reference) if continuation_reference else ""
            ),
            "OMNI_NATIVE_CONTINUATION_TARGET": (
                str(continuation_target) if continuation_target else ""
            ),
            "OMNI_NATIVE_CONTINUATION_SEQUENCES": ",".join(
                str(value) for value in continuation_sequences
            ),
        },
        strip_proxy=True,
    )
    try:
        with patch.object(benchmark, "CLIENT_CONCURRENCY", client_concurrency):
            asyncio.run(
                benchmark.run_seedtts(
                    output / "seedtts",
                    port,
                    sample_ids,
                    trace_phase=None if mmsu else "native",
                    trace_sample_ids=set(selected_samples),
                    warmup_requests=client_concurrency,
                )
            )
        if mmsu:
            with patch(
                "benchmarks.tasks.audio_understanding._build_request_payload",
                identified_mmsu_payload,
            ):
                result = asyncio.run(
                    run_mmsu(
                        argparse.Namespace(
                            base_url=f"http://127.0.0.1:{port}",
                            model="qwen3-omni",
                            modalities="text+audio",
                            output_dir=str(output / "mmsu"),
                            max_samples=None,
                            max_tokens=256,
                            temperature=0.0,
                            seed=REQUEST_SEED,
                            warmup=0,
                            max_concurrency=client_concurrency,
                            request_rate=float("inf"),
                            timeout_s=500,
                            disable_tqdm=False,
                            fingerprint=False,
                            save_audio=True,
                            prompt=MMSU_TTS_PROMPT,
                        ),
                        samples=mmsu_samples,
                        compute_wer=False,
                    )
                )
            assert [row["sample_id"] for row in result["per_sample"]] == [
                sample.sample_id for sample in mmsu_samples
            ]
            assert all(
                row["is_success"] and row["has_audio"] for row in result["per_sample"]
            )
        else:
            pass
    finally:
        stop_server(server)
    for sample_id in selected_samples:
        if conditioning_reference:
            conditioning_report = json.loads(
                (captures / f"{sample_id}-conditioning.json").read_text()
            )
            assert all(conditioning_report["checks"].values())
            assert conditioning_report["outcomes"]["emitted_frames"] > 0
        else:
            pass
        assert (captures / sample_id / "trajectory.pt").is_file()
        report = json.loads((captures / sample_id / "trajectory.json").read_text())
        if not mmsu:
            assert report["finish_reason"] == "stop"
        else:
            pass
        assert len(report["frames"]) == len(report["output_ids"])
        for sequence_tokens in attention_trial_sequences:
            assert (
                captures
                / f"feedback-trace-native-{sample_id}-sequence-{sequence_tokens}.json"
            ).is_file()
    if code_trial_reference is not None:
        assert (captures / "code-trial.json").is_file()
    else:
        pass
    if cache_trial_reference:
        assert (captures / "cache-trial.json").is_file()
        for sequence_tokens in attention_trial_sequences[1:]:
            assert (captures / f"cache-trial-{sequence_tokens}.json").is_file()
    else:
        pass
    if continuation_reference is not None:
        assert (captures / "continuation-trial.json").is_file()
    else:
        pass
    print(json.dumps({"seedtts_samples": len(samples), "captures": selected_samples}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tactic_cache", type=Path)
    parser.add_argument("--mmsu", action="store_true")
    parser.add_argument("--selected-samples", nargs="+", default=list(SELECTED_SAMPLES))
    parser.add_argument("--client-concurrency", type=int, default=32)
    parser.add_argument("--layer-frames", type=int, nargs="+", default=[])
    parser.add_argument("--conditioning-reference", type=Path, nargs="+", default=[])
    parser.add_argument(
        "--qk-trial", choices=("control", "checkpoint", "fused_checkpoint")
    )
    parser.add_argument("--attention-trial-sequences", type=int, nargs="*", default=[])
    parser.add_argument("--attention-trial-splits", type=int, choices=(0, 1), default=0)
    parser.add_argument("--code-trial-reference", type=Path)
    parser.add_argument("--code-trial-choice", type=int, nargs=4, default=[])
    parser.add_argument("--cache-trial-reference", type=Path, nargs="+", default=[])
    parser.add_argument("--cache-target-reference", type=Path)
    parser.add_argument("--continuation-reference", type=Path)
    parser.add_argument("--continuation-target", type=Path)
    parser.add_argument("--continuation-sequences", type=int, nargs=2, default=[])
    arguments = parser.parse_args()
    main(
        arguments.output,
        arguments.checkpoint,
        arguments.tactic_cache,
        arguments.selected_samples,
        arguments.client_concurrency,
        arguments.mmsu,
        arguments.attention_trial_sequences,
        arguments.attention_trial_splits,
        arguments.code_trial_reference,
        arguments.code_trial_choice,
        arguments.cache_trial_reference,
        arguments.cache_target_reference,
        arguments.continuation_reference,
        arguments.continuation_target,
        arguments.continuation_sequences,
        arguments.layer_frames,
        arguments.conditioning_reference,
        arguments.qk_trial,
    )
else:
    pass
