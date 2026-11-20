# SPDX-License-Identifier: Apache-2.0
"""Run matched layer-zero normalizer interventions over distinct SeedTTS contexts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import httpx
import torch

from benchmarks.benchmarker.utils import (
    get_wav_duration,
    start_server_from_cmd,
    stop_server,
)
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    apply_wer,
    load_qwen3_asr,
    transcribe,
)
from sglang_omni.models.qwen3_omni.boundary_witness import WitnessControl
from sglang_omni.utils import find_available_port

SERVER_START_TIMEOUT_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 300
EXPECTED_SEEDTTS_SAMPLES = 1088
HISTORICAL_SEVERE_WER = 0.5
MODERATE_CASE_COUNT = 8
CLEAN_CASE_COUNT = 9
HELD_SAMPLE_ID = "common_voice_en_19690002-common_voice_en_19690001"
HELD_AUDIO_SHA256 = "b3180f36dadd42f6381c106b0538b842a1fb38f684e44c8a5674f9f1b87e247c"


def select_cases(
    h200_history: Path, h100_history: Path
) -> list[dict[str, str | float]]:
    h200_reports = json.loads(h200_history.read_bytes())["reports"]
    h100_records = json.loads(h100_history.read_bytes())["reports"]["h100-main-full"][
        "records"
    ]
    cases = []
    selected = set()
    for name in ("factor-full-main-a", "factor-full-main-b"):
        for row in h200_reports[name]["records"]:
            if row["spoken_wer"] > HISTORICAL_SEVERE_WER and row["id"] not in selected:
                cases.append(
                    {
                        "sample_id": row["id"],
                        "request_id": row["request_id"],
                        "cohort": "historic_severe",
                        "source": name,
                        "historic_spoken_wer": row["spoken_wer"],
                    }
                )
                selected.add(row["id"])
            else:
                pass
    assert len(cases) == 7
    moderate = sorted(
        (
            row
            for row in h100_records
            if row["id"] not in selected
            and 0 < row["spoken_wer"] <= HISTORICAL_SEVERE_WER
            and row["matches_requested_normalized"]
        ),
        key=lambda row: (-row["spoken_wer"], row["id"]),
    )[:MODERATE_CASE_COUNT]
    for row in moderate:
        cases.append(
            {
                "sample_id": row["id"],
                "request_id": row["request_id"],
                "cohort": "historic_moderate",
                "source": "h100-main-full",
                "historic_spoken_wer": row["spoken_wer"],
            }
        )
        selected.add(row["id"])
    clean = sorted(
        (
            row
            for row in h100_records
            if row["id"] not in selected
            and row["spoken_wer"] == 0
            and row["matches_requested_normalized"]
        ),
        key=lambda row: (row["audio_duration_s"], row["id"]),
    )
    for index in range(CLEAN_CASE_COUNT):
        row = clean[len(clean) * (2 * index + 1) // (2 * CLEAN_CASE_COUNT)]
        cases.append(
            {
                "sample_id": row["id"],
                "request_id": row["request_id"],
                "cohort": "historic_clean",
                "source": "h100-main-full",
                "historic_spoken_wer": row["spoken_wer"],
            }
        )
    assert (
        len({row["sample_id"] for row in cases})
        == 7 + MODERATE_CASE_COUNT + CLEAN_CASE_COUNT
    )
    return cases


def controlled_history(
    trace_directory: Path,
) -> tuple[list[torch.Tensor], list[torch.Tensor], bool]:
    frames = [
        torch.load(path, weights_only=True)["codec_ids"].reshape(-1)
        for path in sorted(trace_directory.glob("codec-*.pt"))
    ]
    text_rows = []
    for path in sorted(trace_directory.glob("backbone-decode-*.pt")):
        row = torch.load(path, weights_only=True)
        if not row["uses_pad"]:
            text_rows.append(row["text_row"].reshape(-1))
        else:
            pass
    finish = torch.load(trace_directory / "backbone-finish-000.pt", weights_only=True)
    text_rows.extend(finish["pending_rows"])
    complete = (
        bool(finish["thinker_done"]) and len(frames) == finish["codec_frames_seen"]
    )
    return frames, text_rows, complete


def compare_pair(
    baseline: Path, candidate: Path, held: bool
) -> dict[str, int | bool | None]:
    if not all(
        (directory / "first-decode-layers.pt").is_file()
        for directory in (baseline, candidate)
    ):
        return {"opening_state_equal": False, "future_text_equal": False}
    else:
        pass
    opening_equal: dict[str, bool] = {}
    for name in (
        "backbone-prefill-000.pt",
        "first-decode-prefix-cache-000.pt",
        "backbone-decode-001.pt",
        "codec-000.pt",
        "decode-sampling-001.pt",
    ):
        before = torch.load(baseline / name, weights_only=True)
        after = torch.load(candidate / name, weights_only=True)
        assert before.keys() == after.keys()
        fields = before.keys() - {"raw_logits", "sampled_code", "predictor_query"}
        opening_equal[name] = all(
            (
                torch.equal(value, after[key])
                if isinstance(value, torch.Tensor)
                else value == after[key]
            )
            for key in fields
            if (value := before[key]) is not None or after[key] is not None
        )
    before_layers = torch.load(baseline / "first-decode-layers.pt", weights_only=True)
    after_layers = torch.load(candidate / "first-decode-layers.pt", weights_only=True)
    assert not before_layers[0]["norm_swap_applied"]
    assert after_layers[0]["norm_swap_applied"]
    assert torch.equal(
        before_layers[0]["attention_normalized"],
        before_layers[0]["attention_normalized_before_override"],
    )
    before_frames, before_text, before_complete = controlled_history(baseline)
    after_frames, after_text, after_complete = controlled_history(candidate)
    first_frame_change = next(
        (
            index
            for index, (left, right) in enumerate(zip(before_frames, after_frames))
            if not torch.equal(left, right)
        ),
        None,
    )
    if first_frame_change is None and len(before_frames) != len(after_frames):
        first_frame_change = min(len(before_frames), len(after_frames))
    else:
        pass
    first_main_change = next(
        (
            index
            for index, (left, right) in enumerate(zip(before_frames, after_frames))
            if int(left[0]) != int(right[0])
        ),
        None,
    )
    prefix_equal = all(
        equal
        for name, equal in opening_equal.items()
        if not held
        or name not in ("first-decode-prefix-cache-000.pt", "backbone-decode-001.pt")
    )
    if held:
        before_held = torch.load(
            baseline / "held-first-decode-000.pt", weights_only=True
        )
        after_held = torch.load(
            candidate / "held-first-decode-000.pt", weights_only=True
        )
        prefix_equal = prefix_equal and torch.equal(
            before_held["embedding"], after_held["embedding"]
        )
    else:
        pass
    original_norm_exact = torch.equal(
        before_layers[0]["attention_normalized"],
        after_layers[0]["attention_normalized_before_override"],
    )
    combined_input_exact = torch.equal(
        before_layers[0]["input_contribution"], after_layers[0]["input_contribution"]
    )
    cast_changed = not torch.equal(
        after_layers[0]["attention_normalized"],
        after_layers[0]["attention_normalized_before_override"],
    )
    return {
        "opening_state_equal": prefix_equal
        and original_norm_exact
        and combined_input_exact,
        "future_text_equal": before_complete
        and after_complete
        and len(before_text) == len(after_text)
        and all(
            torch.equal(left, right) for left, right in zip(before_text, after_text)
        ),
        "first_changed_codec_frame": first_frame_change,
        "first_changed_main_codec_frame": first_main_change,
        "before_second_code": int(before_frames[1][0]),
        "after_second_code": int(after_frames[1][0]),
        "before_frames": len(before_frames),
        "after_frames": len(after_frames),
        "cast_changed": cast_changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--asr-path", required=True)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--h200-history", type=Path, required=True)
    parser.add_argument("--h100-history", type=Path, required=True)
    parser.add_argument("--held-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transfer-from", type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir()
    cases = select_cases(arguments.h200_history, arguments.h100_history)
    donor_population = arguments.transfer_from
    donor_rows = {}
    if donor_population is None:
        scopes = ("none", "first", "decode")
    else:
        scopes = ("transfer", "restore")
        eligible = []
        for line in (donor_population / "scored.jsonl").read_text().splitlines():
            row = json.loads(line)
            before, after, pair = (
                row["arms"]["none"],
                row["arms"]["first"],
                row["pairs"]["first"],
            )
            frame_index = pair.get("first_changed_codec_frame")
            if (
                pair["opening_state_equal"]
                and pair["future_text_equal"]
                and "spoken_wer" in before
                and "spoken_wer" in after
                and before["returned_text"] == after["returned_text"]
                and frame_index is not None
                and 0
                < frame_index
                < min(pair["before_frames"], pair["after_frames"]) - 1
            ):
                eligible.append(
                    (after["word_errors"] - before["word_errors"], row["sample_id"])
                )
                donor_rows[row["sample_id"]] = row
            else:
                pass
        worsening = sorted((entry for entry in eligible if entry[0] > 0), reverse=True)[
            :2
        ]
        improving = sorted(entry for entry in eligible if entry[0] < 0)[:2]
        selected = {sample_id for _, sample_id in (*worsening, *improving)}
        cases = [case for case in cases if case["sample_id"] in selected]
    (output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    samples = load_seedtts_samples(DATASETS["seedtts"])
    assert len(samples) == EXPECTED_SEEDTTS_SAMPLES
    samples_by_id = {sample.sample_id: sample for sample in samples}
    control_path = output / "control.json"
    control_path.write_text('{"request_id": ""}\n')
    for scope in scopes:
        control_path.write_text('{"request_id": ""}\n')
        talker_port = find_available_port()
        talker_server = start_server_from_cmd(
            [
                sys.executable,
                "-P",
                "-m",
                "sglang_omni.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(talker_port),
                "--config",
                str(arguments.server_config),
                "--model-path",
                arguments.model_path,
                "--model-name",
                "qwen3-omni",
            ],
            output / f"talker-{scope}.log",
            talker_port,
            timeout=SERVER_START_TIMEOUT_SECONDS,
            env={"OMNI_WITNESS_BATCH_CONTROL": str(control_path)},
            strip_proxy=True,
        )
        try:
            with httpx.Client(
                trust_env=False, timeout=REQUEST_TIMEOUT_SECONDS
            ) as client:
                for index, case in enumerate(cases, 1):
                    sample = samples_by_id[case["sample_id"]]
                    input_sha256 = hashlib.sha256(
                        Path(sample.ref_audio).read_bytes()
                    ).hexdigest()
                    is_held = case["sample_id"] == HELD_SAMPLE_ID
                    if is_held:
                        assert input_sha256 == HELD_AUDIO_SHA256
                    else:
                        pass
                    case_directory = output / f'{case["cohort"]}-{case["sample_id"]}'
                    arm_directory = case_directory / scope
                    arm_directory.mkdir(parents=True)
                    prompt = (
                        f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
                        f"Now please read the following text out loud in the same voice and style: "
                        f"{sample.target_text}"
                    )
                    payload = {
                        "model": "qwen3-omni",
                        "request_id": case["request_id"],
                        "messages": [{"role": "user", "content": prompt}],
                        "modalities": ["text", "audio"],
                        "audio": {"format": "wav"},
                        "max_tokens": 256,
                        "temperature": 0.7,
                        "stream": False,
                        "audios": [sample.ref_audio],
                    }
                    control: WitnessControl = {
                        "request_id": case["request_id"],
                        "output_directory": str(arm_directory),
                        "norm_scope": {"transfer": "none", "restore": "first"}.get(
                            scope, scope
                        ),
                        "held_input_directory": (
                            str(arguments.held_input.resolve()) if is_held else None
                        ),
                    }
                    if donor_population is not None:
                        donor_row = donor_rows[case["sample_id"]]
                        donor_scope = "first" if scope == "transfer" else "none"
                        frame_index = donor_row["pairs"]["first"][
                            "first_changed_codec_frame"
                        ]
                        control["codec_override_frame"] = frame_index
                        control["codec_override_file"] = str(
                            donor_population
                            / donor_row["directory"]
                            / donor_scope
                            / f"codec-{frame_index:03d}.pt"
                        )
                    else:
                        pass
                    next_control = control_path.with_suffix(".next")
                    next_control.write_text(json.dumps(control))
                    next_control.replace(control_path)
                    response = client.post(
                        f"http://127.0.0.1:{talker_port}/v1/chat/completions",
                        json=payload,
                    )
                    arm = {
                        "http_status": response.status_code,
                        "input_sha256": input_sha256,
                    }
                    if response.is_success:
                        completion = response.json()
                        assert completion["id"] == f'chatcmpl-{case["request_id"]}'
                        (choice,) = completion["choices"]
                        waveform = base64.b64decode(
                            choice["message"]["audio"]["data"], validate=True
                        )
                        (arm_directory / "audio.wav").write_bytes(waveform)
                        arm.update(
                            returned_text=choice["message"]["content"],
                            finish_reason=choice["finish_reason"],
                            waveform_sha256=hashlib.sha256(waveform).hexdigest(),
                            duration_seconds=get_wav_duration(waveform),
                        )
                    else:
                        arm["error"] = response.text
                    (arm_directory / "response.json").write_text(json.dumps(arm))
                    print(
                        f'{scope} {index}/{len(cases)}: {case["sample_id"]} HTTP {response.status_code}',
                        flush=True,
                    )
        finally:
            stop_server(talker_server)

    results = []
    for case in cases:
        case_directory = output / f'{case["cohort"]}-{case["sample_id"]}'
        is_held = case["sample_id"] == HELD_SAMPLE_ID
        arms = {
            scope: json.loads((case_directory / scope / "response.json").read_bytes())
            for scope in scopes
        }
        assert len({arm["input_sha256"] for arm in arms.values()}) == 1
        if donor_population is None:
            pairs = {
                scope: compare_pair(
                    case_directory / "none", case_directory / scope, is_held
                )
                for scope in ("first", "decode")
            }
        else:
            pairs = {}
            donor_row = donor_rows[case["sample_id"]]
            frame_index = donor_row["pairs"]["first"]["first_changed_codec_frame"]
            for scope, recipient_scope, donor_scope in (
                ("transfer", "none", "first"),
                ("restore", "first", "none"),
            ):
                reference_directory = donor_population / donor_row["directory"]
                recipient_frames, recipient_text, recipient_complete = (
                    controlled_history(reference_directory / recipient_scope)
                )
                donor_frames, _, _ = controlled_history(
                    reference_directory / donor_scope
                )
                replay_frames, replay_text, replay_complete = controlled_history(
                    case_directory / scope
                )
                assert (
                    case_directory / scope / f"codec-override-{frame_index:03d}.pt"
                ).is_file()
                assert torch.equal(
                    replay_frames[frame_index], donor_frames[frame_index]
                )
                sampling_name = f"decode-sampling-{frame_index:03d}.pt"
                recipient_sampling = torch.load(
                    reference_directory / recipient_scope / sampling_name,
                    weights_only=True,
                )
                replay_sampling = torch.load(
                    case_directory / scope / sampling_name, weights_only=True
                )
                assert recipient_sampling.keys() == replay_sampling.keys()
                pairs[scope] = {
                    "overridden_frame": frame_index,
                    "preintervention_sampling_exact": all(
                        torch.equal(value, replay_sampling[name])
                        for name, value in recipient_sampling.items()
                    ),
                    "preceding_codec_history_exact": all(
                        torch.equal(left, right)
                        for left, right in zip(
                            recipient_frames[:frame_index],
                            replay_frames[:frame_index],
                            strict=True,
                        )
                    ),
                    "future_text_equal": recipient_complete
                    and replay_complete
                    and len(recipient_text) == len(replay_text)
                    and all(
                        torch.equal(left, right)
                        for left, right in zip(recipient_text, replay_text)
                    ),
                    "returned_text_equal": arms[scope].get("returned_text")
                    == donor_row["arms"][recipient_scope]["returned_text"],
                    "recipient_wer": donor_row["arms"][recipient_scope]["spoken_wer"],
                    "donor_wer": donor_row["arms"][donor_scope]["spoken_wer"],
                }
        result = {
            **case,
            "used_frozen_h200_state": is_held,
            "directory": case_directory.name,
            "arms": arms,
            "pairs": pairs,
        }
        results.append(result)
        with (output / "generation.jsonl").open("a") as generated:
            generated.write(json.dumps(result) + "\n")

    asr_port = find_available_port()
    asr_server = start_server_from_cmd(
        [
            sys.executable,
            "-P",
            "-m",
            "sglang_omni.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(asr_port),
            "--model-path",
            arguments.asr_path,
            "--model-name",
            QWEN3_ASR_MODEL_PATH,
            "--asr.gpu",
            "0",
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        output / "asr.log",
        asr_port,
        timeout=SERVER_START_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    try:
        asr = load_qwen3_asr(asr_port)
        transcripts = {}
        for index, result in enumerate(results, 1):
            for scope, arm in result["arms"].items():
                if "waveform_sha256" not in arm:
                    continue
                else:
                    pass
                path = output / result["directory"] / scope / "audio.wav"
                digest = arm["waveform_sha256"]
                if digest not in transcripts:
                    transcripts[digest] = transcribe(asr, str(path), "en", "cuda:0")
                else:
                    pass
                spoken = apply_wer(
                    SampleOutput(
                        sample_id=result["sample_id"], target_text=arm["returned_text"]
                    ),
                    transcripts[digest],
                    "en",
                )
                assert spoken.is_success
                arm.update(
                    transcript=spoken.whisper_text,
                    spoken_wer=spoken.wer,
                    word_errors=spoken.substitutions
                    + spoken.deletions
                    + spoken.insertions,
                    reference_words=spoken.substitutions
                    + spoken.deletions
                    + spoken.hits,
                )
            with (output / "scored.jsonl").open("a") as scored:
                scored.write(json.dumps(result) + "\n")
            print(f'{index}/{len(results)} scored: {result["sample_id"]}', flush=True)
    finally:
        stop_server(asr_server)


if __name__ == "__main__":
    main()
else:
    pass
