"""Split speech components after an exact native replay."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-command-file", type=Path, required=True)
    parser.add_argument("--native-proof-file", type=Path, required=True)
    parser.add_argument(
        "--split-kind",
        choices=("backbone-predictor", "boundary-experts"),
        default="backbone-predictor",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    proof = json.loads(arguments.native_proof_file.read_bytes())
    assert proof["all_exact"] and proof["counts"]["frames"] > 0
    native_command = json.loads(arguments.native_command_file.read_bytes())["command"]
    assert native_command[:4] == [
        "python3",
        "-u",
        "-m",
        "scripts.debug_qwen3_omni_talker_generation",
    ]
    assert "--verify-native-projection-inputs" in native_command
    assert "--native-projection-batches" in native_command
    assert "--checkpoint-query-key-normalization" in native_command
    arguments.output_directory.mkdir(exist_ok=False)
    trace_arguments = native_command[native_command.index("--trace-directory") :]
    common_command = native_command[
        : native_command.index("--native-predictor-capture-directory")
    ]
    common_command.extend(["--audit-text-handoff", "--capture-layer-states"])
    native_capture = native_command[
        native_command.index("--native-predictor-capture-directory") + 1
    ]
    phases = (
        (
            "checkpoint-backbone-checkpoint-predictor",
            "native-backbone-checkpoint-predictor",
            "checkpoint-backbone-native-predictor",
        )
        if arguments.split_kind == "backbone-predictor"
        else (
            "checkpoint-boundary-native-experts",
            "native-boundary-checkpoint-experts",
            "checkpoint-boundary-checkpoint-experts",
        )
    )
    commands = []
    for phase in phases:
        output_directory = arguments.output_directory / phase
        if arguments.split_kind == "boundary-experts":
            command = native_command.copy()
            command.remove("--verify-native-projection-inputs")
            command[command.index("--output-directory") + 1] = str(output_directory)
            if phase.startswith("checkpoint-boundary-"):
                command[command.index("--normalization-semantics") + 1] = "checkpoint"
            else:
                pass
            if phase.endswith("-checkpoint-experts"):
                command.remove("--native-expert-batches")
                command[command.index("--expert-output-arithmetic") + 1] = "checkpoint"
            else:
                pass
        elif phase == "native-backbone-checkpoint-predictor":
            command = native_command.copy()
            command.remove("--verify-native-projection-inputs")
            command[command.index("--native-feedback-reduction") + 1] = "sequential"
            command[command.index("--output-directory") + 1] = str(output_directory)
            command.append("--checkpoint-residual-predictor")
        else:
            command = common_command + [
                "--normalization-semantics",
                "checkpoint",
                "--output-directory",
                str(output_directory),
            ]
            if phase == "checkpoint-backbone-native-predictor":
                command.extend(
                    [
                        "--native-predictor-capture-directory",
                        native_capture,
                        "--native-feedback-reduction",
                        "checkpoint",
                    ]
                )
            else:
                pass
            command.extend(trace_arguments)
        commands.append({"phase": phase, "command": command})
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "source_commit": source_commit,
        "split_kind": arguments.split_kind,
        "native_proof": proof,
        "commands": commands,
        "scope": "Same captured conditioning and sampling coordinates as the exact native control. Any recorded dense/expert peer mode stops at its recorded horizon. Backbone-predictor split changes whole components. Boundary-expert split retains the native predictor, Q/K normalization, attention, routing and shared expert, changing only boundary normalization and/or routed-expert arithmetic. Neither is a performance comparison, full-population quality gate, or universal repair claim.",
    }
    (arguments.output_directory / "commands.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    progress = []
    for phase in commands:
        print(f"Starting {phase['phase']}", flush=True)
        with (arguments.output_directory / f"{phase['phase']}.log").open("wb") as log:
            completed = subprocess.run(
                phase["command"], stdout=log, stderr=subprocess.STDOUT
            )
        progress.append({"phase": phase["phase"], "returncode": completed.returncode})
        (arguments.output_directory / "progress.json").write_text(
            json.dumps(progress, indent=2) + "\n"
        )
        completed.check_returncode()
        results = json.loads(
            (arguments.output_directory / phase["phase"] / "results.json").read_bytes()
        )
        for result in results:
            assert result["complete_text_projection_verified"]
            assert result["checkpoint_text_handoffs_verified"] == (
                len(result["frames"]) - int(result["sequence_complete"])
            )
        print(
            json.dumps(
                {
                    "phase": phase["phase"],
                    "requests": [
                        {
                            name: result[name]
                            for name in (
                                "native_trace_directory",
                                "reference_trace_directory",
                                "termination",
                                "sequence_complete",
                                "reference_frames_including_eos",
                                "first_codec_divergence",
                                "first_main_code_divergence",
                            )
                        }
                        for result in results
                    ],
                }
            ),
            flush=True,
        )
    print("All component phases finished", flush=True)


if __name__ == "__main__":
    main()
