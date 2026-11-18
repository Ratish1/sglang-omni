# SPDX-License-Identifier: Apache-2.0
"""Run matched natural-generation and fixed-history Talker SIM probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from benchmarks.benchmarker.utils import (
    no_proxy_env,
    start_server_from_cmd,
    stop_server,
)
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from sglang_omni.models.qwen3_omni.sim_bisect_trace import SimTraceEvent
from sglang_omni.utils import find_available_port


@dataclass(frozen=True, kw_only=True)
class Arm:
    name: str
    cast_mask: str
    donor_name: str = ""
    comparison_name: str = "late"


@dataclass(frozen=True, kw_only=True)
class Case:
    sample_id: str
    request_id: str
    request_sha256: str
    reference: Path


@dataclass(frozen=True, kw_only=True)
class History:
    input_events: list[SimTraceEvent]
    frames: list[SimTraceEvent]
    text: str
    waveform: Path
    waveform_sha256: str


def read_history(arm_directory: Path, case: Case, *, forced: bool) -> History:
    generation_directory = arm_directory / "generation"
    generated_file = generation_directory / case.request_id
    generation_request = generated_file.with_suffix(".request.json")
    assert (
        hashlib.sha256(generation_request.read_bytes()).hexdigest()
        == case.request_sha256
    )
    speech = json.loads(generated_file.with_suffix(".speech.json").read_text())
    assert (speech["request_id"], speech["sample_id"]) == (
        case.request_id,
        case.sample_id,
    )
    waveform = generated_file.with_suffix(".wav")
    waveform_sha256 = hashlib.sha256(waveform.read_bytes()).hexdigest()
    assert waveform_sha256 == speech["waveform_sha256"]
    events = [
        json.loads(line)
        for line in (arm_directory / "trace" / f"{case.request_id}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert events[0]["kind"] == "input" and events[0]["phase"] == "prefill"
    input_events = events[::2]
    frames = events[1::2]
    assert len(frames) == len(input_events) > 1
    assert all(
        input_event["kind"] == "input"
        and frame["kind"] == "frame"
        and input_event["phase"] == ("prefill" if frame_index == 0 else "decode")
        and frame["frame_index"] == frame_index
        and len(frame["codes"]) == len(frames[0]["codes"]) > 1
        and frame["forced"] == forced
        and len(frame["residual_code_margins"])
        == (len(frame["codes"]) - 1 if forced else 0)
        and (frame["codes"][0] == frame["end_token_id"])
        == (frame_index == len(frames) - 1)
        for frame_index, (input_event, frame) in enumerate(
            zip(input_events, frames, strict=True)
        )
    ), case.request_id
    return History(
        input_events=input_events,
        frames=frames,
        text=speech["text"],
        waveform=waveform,
        waveform_sha256=waveform_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("natural", "held", "sites"), required=True)
    parser.add_argument("--replay-driver", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    arguments = parser.parse_args()
    run_directory = arguments.run_directory.resolve(strict=True)
    replay_driver = arguments.replay_driver.resolve(strict=True)
    config_path = arguments.config.resolve(strict=True)
    input_directory = run_directory / "inputs"
    manifest_path = input_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases: list[Case] = []
    for record in manifest["cases"]:
        reference_file = input_directory / "audio" / record["reference_filename"]
        assert (
            hashlib.sha256(reference_file.read_bytes()).hexdigest()
            == record["reference_sha256"]
        )
        request_file = input_directory / f'{record["request_id"]}.request.json'
        request_bytes = request_file.read_bytes()
        assert hashlib.sha256(request_bytes).hexdigest() == record["request_sha256"]
        request = json.loads(request_bytes)
        assert request["audios"] == [str(reference_file)]
        assert request["seed"] == manifest["seed"]
        cases.append(
            Case(
                sample_id=record["sample_id"],
                request_id=record["request_id"],
                request_sha256=record["request_sha256"],
                reference=reference_file,
            )
        )
    assert len({case.request_id for case in cases}) == len(cases) > 1
    assert Path(snapshot_download(arguments.model)).name == arguments.model_revision

    if arguments.phase == "natural":
        arms = [
            Arm(name="late", cast_mask="000"),
            Arm(name="early", cast_mask="111"),
            Arm(name="late-repeat", cast_mask="000"),
        ]
    elif arguments.phase == "held":
        arms = [
            Arm(
                name=f"{donor_name}-donor-{cast_name}",
                cast_mask=cast_mask,
                donor_name=donor_name,
            )
            for donor_name in ("late", "early")
            for cast_name, cast_mask in (("late", "000"), ("early", "111"))
        ]
    else:
        arms = [
            Arm(
                name=arm_name,
                cast_mask=cast_mask,
                comparison_name="early" if arm_name.endswith("-inverse") else "late",
            )
            for arm_name, cast_mask in (
                ("input-only", "100"),
                ("input-inverse", "011"),
                ("post-only", "010"),
                ("post-inverse", "101"),
                ("final-only", "001"),
                ("final-inverse", "110"),
            )
        ]
    for arm in arms:
        if arm.donor_name:
            for case in cases:
                read_history(run_directory / arm.donor_name, case, forced=False)
        else:
            pass
    if arguments.phase in ("held", "sites"):
        assert (run_directory / "natural-report.json").is_file()
    else:
        pass

    for arm in arms:
        arm_directory = run_directory / arm.name
        trace_directory = arm_directory / "trace"
        trace_directory.mkdir(parents=True)
        port = find_available_port()
        server = start_server_from_cmd(
            [
                sys.executable,
                "-m",
                "sglang_omni.cli",
                "serve",
                "--model-path",
                arguments.model,
                "--model-name",
                "qwen3-omni",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--config",
                str(config_path),
            ],
            arm_directory / "server.log",
            port,
            timeout=600,
            env={
                "SGLANG_OMNI_SIM_CAST_MASK": arm.cast_mask,
                "SGLANG_OMNI_SIM_MANIFEST": str(manifest_path),
                "SGLANG_OMNI_SIM_TRACE_DIRECTORY": str(trace_directory),
                "SGLANG_OMNI_SIM_DONOR_DIRECTORY": (
                    str(run_directory / arm.donor_name / "trace")
                    if arm.donor_name
                    else ""
                ),
            },
            strip_proxy=True,
        )
        try:
            with (arm_directory / "replay.log").open("w") as replay_log:
                subprocess.run(
                    [
                        sys.executable,
                        str(replay_driver),
                        "--mode",
                        "replay",
                        "--workload",
                        "tts",
                        "--input-directory",
                        str(input_directory),
                        "--reference-results",
                        str(input_directory / "speed_results.json"),
                        "--output-directory",
                        str(arm_directory / "generation"),
                        "--base-url",
                        f"http://127.0.0.1:{port}",
                        "--concurrency",
                        "1",
                    ],
                    env={**no_proxy_env(), "PYTHONPATH": str(Path.cwd())},
                    stdout=replay_log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=True,
                )
        finally:
            stop_server(server)
        processes = subprocess.check_output(["ps", "-eo", "pgid=,stat="], text=True)
        assert not any(
            int(process_group) == server.pid and not state.startswith("Z")
            for process_group, state in (
                line.split() for line in processes.splitlines()
            )
        ), server.pid
        summary = json.loads(
            (arm_directory / "generation" / "results.json").read_text()
        )["summary"]
        assert summary["completed_requests"] == len(cases)
        assert summary["failed_requests"] == summary["max_token_hits"] == 0
        for case in cases:
            read_history(arm_directory, case, forced=bool(arm.donor_name))
        print(f"{arm.name} completed and validated {len(cases)} histories", flush=True)

    reports = []
    if arguments.phase == "held":
        for case in cases:
            for donor_name in ("late", "early"):
                donor = read_history(run_directory / donor_name, case, forced=False)
                late = read_history(
                    run_directory / f"{donor_name}-donor-late", case, forced=True
                )
                early = read_history(
                    run_directory / f"{donor_name}-donor-early", case, forced=True
                )
                assert late.text == early.text == donor.text, case.request_id
                assert (
                    late.waveform_sha256
                    == early.waveform_sha256
                    == donor.waveform_sha256
                ), case.request_id
                assert (
                    late.input_events == early.input_events == donor.input_events
                ), case.request_id
                assert (
                    [frame["codes"] for frame in late.frames]
                    == [frame["codes"] for frame in early.frames]
                    == [frame["codes"] for frame in donor.frames]
                ), case.request_id
                repeated_donor = late if donor_name == "late" else early
                assert all(
                    repeated_frame["chosen_code_logit"]
                    == donor_frame["chosen_code_logit"]
                    and repeated_frame["best_other_code_logit"]
                    == donor_frame["best_other_code_logit"]
                    and min(repeated_frame["residual_code_margins"]) >= 0.0
                    for repeated_frame, donor_frame in zip(
                        repeated_donor.frames, donor.frames, strict=True
                    )
                ), case.request_id
                reports.append(
                    {
                        "sample_id": case.sample_id,
                        "donor": donor_name,
                        "early_minus_late_margin_by_frame": [
                            (
                                early_frame["chosen_code_logit"]
                                - early_frame["best_other_code_logit"]
                            )
                            - (
                                late_frame["chosen_code_logit"]
                                - late_frame["best_other_code_logit"]
                            )
                            for late_frame, early_frame in zip(
                                late.frames, early.frames, strict=True
                            )
                        ],
                        "early_minus_late_residual_margins_by_frame": [
                            [
                                early_margin - late_margin
                                for late_margin, early_margin in zip(
                                    late_frame["residual_code_margins"],
                                    early_frame["residual_code_margins"],
                                    strict=True,
                                )
                            ]
                            for late_frame, early_frame in zip(
                                late.frames, early.frames, strict=True
                            )
                        ],
                    }
                )
    else:
        assets = ensure_speaker_similarity_assets()
        scorer = WavLMSpeakerSimilarity(
            finetune_checkpoint=assets.finetune_checkpoint,
            wavlm_base=assets.wavlm_base,
            device="cuda:0",
        )
        for case in cases:
            late = read_history(run_directory / "late", case, forced=False)
            repeat = read_history(run_directory / "late-repeat", case, forced=False)
            assert late == History(
                input_events=repeat.input_events,
                frames=repeat.frames,
                text=repeat.text,
                waveform=late.waveform,
                waveform_sha256=repeat.waveform_sha256,
            ), case.request_id
            arm_names = dict.fromkeys(("late", "early", *(arm.name for arm in arms)))
            histories = {
                name: read_history(run_directory / name, case, forced=False)
                for name in arm_names
            }
            for history in histories.values():
                assert history.input_events[0] == late.input_events[0], case.request_id
            scores = scorer.score_batch(
                [str(case.reference)] * len(histories),
                [str(history.waveform) for history in histories.values()],
            )
            similarity_by_arm = dict(zip(histories, scores, strict=True))
            reports.append(
                {
                    "sample_id": case.sample_id,
                    "late_reference_sim": float(similarity_by_arm["late"]),
                    "early_reference_sim": float(similarity_by_arm["early"]),
                    "arms": {
                        arm.name: {
                            "compared_to": arm.comparison_name,
                            "same_returned_text": (
                                histories[arm.name].text
                                == histories[arm.comparison_name].text
                            ),
                            "codec_frames": len(histories[arm.name].frames) - 1,
                            "reference_sim_delta": float(
                                similarity_by_arm[arm.name]
                                - similarity_by_arm[arm.comparison_name]
                            ),
                        }
                        for arm in arms
                    },
                }
            )
    (run_directory / f"{arguments.phase}-report.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
