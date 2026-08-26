# SPDX-License-Identifier: Apache-2.0
"""Run the paired Qwen3-TTS repetition-penalty tail investigation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from benchmarks.eval.compare_qwen3_tts_completion_diagnostics import (
    compare_arms,
    load_arm_artifacts,
)

OWNER_ENV = "SGLANG_OMNI_QWEN3_TTS_REPETITION_PENALTY_OWNER"
DIAGNOSTICS_DIR_ENV = "SGLANG_OMNI_QWEN3_TTS_COMPLETION_DIAGNOSTICS_DIR"
DIAGNOSTICS_RUN_LABEL_ENV = "SGLANG_OMNI_QWEN3_TTS_COMPLETION_DIAGNOSTICS_RUN_LABEL"
PUBLIC_REPETITION_PENALTY = 1.05


@dataclass(frozen=True)
class RepetitionArm:
    name: str
    owner: str
    public_penalty: float
    qwen_penalty: float
    sglang_penalty: float

    @property
    def nominal_effective_penalty(self) -> float:
        return self.qwen_penalty * self.sglang_penalty


def repetition_arms() -> tuple[RepetitionArm, ...]:
    square_root_penalty = math.sqrt(PUBLIC_REPETITION_PENALTY)
    return (
        RepetitionArm(
            name="sglang_once_p105",
            owner="sglang",
            public_penalty=PUBLIC_REPETITION_PENALTY,
            qwen_penalty=1.0,
            sglang_penalty=PUBLIC_REPETITION_PENALTY,
        ),
        RepetitionArm(
            name="qwen_once_p105",
            owner="qwen",
            public_penalty=PUBLIC_REPETITION_PENALTY,
            qwen_penalty=PUBLIC_REPETITION_PENALTY,
            sglang_penalty=1.0,
        ),
        RepetitionArm(
            name="double_sqrt_p105",
            owner="double",
            public_penalty=square_root_penalty,
            qwen_penalty=square_root_penalty,
            sglang_penalty=square_root_penalty,
        ),
        RepetitionArm(
            name="double_p105",
            owner="double",
            public_penalty=PUBLIC_REPETITION_PENALTY,
            qwen_penalty=PUBLIC_REPETITION_PENALTY,
            sglang_penalty=PUBLIC_REPETITION_PENALTY,
        ),
    )


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("--seeds values must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds values must be unique")
    return seeds


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run four unique Qwen3-TTS repetition-owner arms with paired "
            "sample seeds and completion diagnostics."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--meta",
        default="zhaochenyang20/seed-tts-eval-arrow",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/qwen3_tts_repetition_tail"),
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=(20260823,))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-timeout", type=int, default=1200)
    parser.add_argument("--server-config", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--task-type", default=None)
    parser.add_argument("--no-ref-audio", action="store_true")
    parser.add_argument("--no-ref-text", action="store_true")
    parser.add_argument("--ref-format", choices=["flat", "references"], default="flat")
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--asr-model-path",
        default="Qwen/Qwen3-ASR-1.7B",
    )
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _append_optional(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_generation_command(
    args: argparse.Namespace,
    *,
    arm: RepetitionArm,
    seed: int,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_tts_seedtts",
        "--model",
        args.model,
        "--meta",
        args.meta,
        "--output-dir",
        str(output_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--response-format",
        "wav",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--repetition-penalty",
        repr(arm.public_penalty),
        "--seed",
        str(seed),
        "--sample-specific-seeds",
        "--warmup",
        "0",
        "--concurrency",
        str(args.concurrency),
        "--server-engine-stage",
        "tts_engine",
        "--max-running-requests",
        str(args.max_running_requests),
        "--cuda-graph-max-bs",
        str(args.cuda_graph_max_bs),
        "--server-timeout",
        str(args.server_timeout),
        "--disable-tqdm",
        "--generate-only",
    ]
    _append_optional(command, "--max-samples", args.max_samples)
    if args.sample_offset:
        command.extend(["--sample-offset", str(args.sample_offset)])
    _append_optional(command, "--temperature", args.temperature)
    _append_optional(command, "--top-p", args.top_p)
    _append_optional(command, "--top-k", args.top_k)
    _append_optional(command, "--server-config", args.server_config)
    _append_optional(command, "--quantization", args.quantization)
    _append_optional(command, "--task-type", args.task_type)
    if args.no_ref_audio:
        command.append("--no-ref-audio")
    if args.no_ref_text:
        command.append("--no-ref-text")
    command.extend(["--ref-format", args.ref_format])
    return command


def _arm_environment(arm: RepetitionArm, output_dir: Path, seed: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment[OWNER_ENV] = arm.owner
    environment[DIAGNOSTICS_DIR_ENV] = str(
        (output_dir / "completion_diagnostics").resolve()
    )
    environment[DIAGNOSTICS_RUN_LABEL_ENV] = f"seed-{seed}-{arm.name}"
    return environment


def _git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=True,
    ).strip()


def _package_versions() -> dict[str, str | None]:
    packages = (
        "sglang",
        "sglang-omni",
        "torch",
        "transformers",
        "flashinfer-python",
        "qwen-tts",
    )
    resolved = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = None
    return resolved


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_arm(arm: RepetitionArm, output_dir: Path) -> dict[str, Any]:
    artifacts = load_arm_artifacts(output_dir)
    failed_ids = sorted(
        sample_id
        for sample_id, entry in artifacts.generated_by_id.items()
        if not entry.get("is_success", False)
    )
    if failed_ids:
        raise RuntimeError(
            f"{arm.name} has {len(failed_ids)} generation failures: {failed_ids[:10]}"
        )

    for sample_id, record in artifacts.completion_by_id.items():
        if record.get("repetition_penalty_owner") != arm.owner:
            raise RuntimeError(
                f"{arm.name}/{sample_id} recorded owner "
                f"{record.get('repetition_penalty_owner')!r}, expected {arm.owner!r}"
            )
        expected = {
            "public_repetition_penalty": arm.public_penalty,
            "qwen_repetition_penalty": arm.qwen_penalty,
            "sglang_repetition_penalty": arm.sglang_penalty,
            "nominal_effective_repetition_penalty": arm.nominal_effective_penalty,
        }
        for key, expected_value in expected.items():
            actual = float(record[key])
            if not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=0.0):
                raise RuntimeError(
                    f"{arm.name}/{sample_id} recorded {key}={actual}, "
                    f"expected {expected_value}"
                )

    speed_results = json.loads(
        (output_dir / "speed_results.json").read_text(encoding="utf-8")
    )
    return {
        "generated": len(artifacts.generated_by_id),
        "completion_records": len(artifacts.completion_by_id),
        "speed_summary": speed_results.get("summary"),
    }


def _transcribe_all(
    args: argparse.Namespace,
    *,
    runs: list[tuple[int, RepetitionArm, Path]],
    output_root: Path,
) -> None:
    from benchmarks.benchmarker.utils import managed_omni_server
    from benchmarks.eval.benchmark_tts_seedtts import (
        TtsSeedttsBenchmarkConfig,
        run_tts_seedtts_transcribe,
    )

    with managed_omni_server(
        model_path=args.asr_model_path,
        port=args.port,
        host=args.host,
        log_file=output_root / "server_logs" / "asr_server.log",
        timeout=args.server_timeout,
    ):
        for seed, arm, output_dir in runs:
            config = TtsSeedttsBenchmarkConfig(
                model=args.model,
                meta=args.meta,
                host=args.host,
                port=args.port,
                task_type=args.task_type,
                voice_clone=not args.no_ref_audio,
                ref_format=args.ref_format,
                no_ref_text=args.no_ref_text,
                output_dir=str(output_dir),
                max_samples=args.max_samples,
                sample_offset=args.sample_offset,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=arm.public_penalty,
                seed=seed,
                sample_specific_seeds=True,
                warmup=0,
                concurrency=args.concurrency,
                max_running_requests=args.max_running_requests,
                cuda_graph_max_bs=args.cuda_graph_max_bs,
                server_config=args.server_config,
                server_engine_stage="tts_engine",
                quantization=args.quantization,
                lang=args.lang,
                device=args.device,
                asr_model_path=args.asr_model_path,
                asr_concurrency=1,
            )
            run_tts_seedtts_transcribe(config, asr_router_port=args.port)


def _compare_panel(
    *,
    seed: int,
    arms: tuple[RepetitionArm, ...],
    panel_dir: Path,
) -> dict[str, Any]:
    arm_artifacts = {arm.name: load_arm_artifacts(panel_dir / arm.name) for arm in arms}
    comparison_summaries = {}
    for left_arm, right_arm in itertools.combinations(arms, 2):
        comparison_name = f"{left_arm.name}__vs__{right_arm.name}"
        report = compare_arms(
            arm_artifacts[left_arm.name],
            arm_artifacts[right_arm.name],
        )
        report_path = panel_dir / "comparisons" / f"{comparison_name}.json"
        _write_json(report_path, report)
        comparison_summaries[comparison_name] = {
            "report": str(report_path),
            "counts": report["counts"],
            "classifications": report["classifications"],
        }

    arm_summaries = {}
    for arm in arms:
        output_dir = panel_dir / arm.name
        speed = json.loads(
            (output_dir / "speed_results.json").read_text(encoding="utf-8")
        )
        wer_path = output_dir / "wer_results.json"
        wer_summary = (
            json.loads(wer_path.read_text(encoding="utf-8")).get("summary")
            if wer_path.is_file()
            else None
        )
        arm_summaries[arm.name] = {
            **asdict(arm),
            "nominal_effective_penalty": arm.nominal_effective_penalty,
            "speed_summary": speed.get("summary"),
            "wer_summary": wer_summary,
        }

    panel_summary = {
        "seed": seed,
        "arms": arm_summaries,
        "comparisons": comparison_summaries,
    }
    _write_json(panel_dir / "panel_summary.json", panel_summary)
    return panel_summary


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.sample_offset < 0:
        raise ValueError("--sample-offset must be non-negative")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    output_root = args.output_dir.expanduser().resolve()
    arms = repetition_arms()
    runs: list[tuple[int, RepetitionArm, Path]] = []
    commands: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(args.seeds):
        rotation = seed_index % len(arms)
        ordered_arms = arms[rotation:] + arms[:rotation]
        for arm in ordered_arms:
            output_dir = output_root / f"seed_{seed}" / arm.name
            command = build_generation_command(
                args,
                arm=arm,
                seed=seed,
                output_dir=output_dir,
            )
            commands.append(
                {
                    "seed": seed,
                    "arm": asdict(arm),
                    "nominal_effective_penalty": arm.nominal_effective_penalty,
                    "output_dir": str(output_dir),
                    "command": command,
                    "environment": {
                        OWNER_ENV: arm.owner,
                        DIAGNOSTICS_DIR_ENV: str(
                            (output_dir / "completion_diagnostics").resolve()
                        ),
                        DIAGNOSTICS_RUN_LABEL_ENV: f"seed-{seed}-{arm.name}",
                    },
                }
            )
            runs.append((seed, arm, output_dir))

    if args.dry_run:
        for item in commands:
            environment = " ".join(
                f"{key}={shlex.quote(value)}"
                for key, value in item["environment"].items()
            )
            print(f"{environment} {shlex.join(item['command'])}")
        return

    manifest_path = output_root / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to mix with an existing run: {manifest_path}")
    manifest: dict[str, Any] = {
        "status": "running",
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_status": _git_value(repo_root, "status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "arguments": vars(args) | {"output_dir": str(output_root)},
        "commands": commands,
        "completed_arms": [],
    }
    _write_json(manifest_path, manifest)

    try:
        for command_item, (seed, arm, output_dir) in zip(commands, runs):
            if (output_dir / "generated.json").exists():
                raise FileExistsError(
                    f"refusing to overwrite completed arm: {output_dir}"
                )
            print(f"Running seed={seed} arm={arm.name}")
            subprocess.run(
                command_item["command"],
                cwd=repo_root,
                env=_arm_environment(arm, output_dir, seed),
                check=True,
            )
            validation = _validate_arm(arm, output_dir)
            manifest["completed_arms"].append(
                {
                    "seed": seed,
                    "arm": arm.name,
                    "validation": validation,
                }
            )
            _write_json(manifest_path, manifest)

        if not args.generation_only:
            _transcribe_all(args, runs=runs, output_root=output_root)

        panel_summaries = {}
        for seed in args.seeds:
            panel_summaries[str(seed)] = _compare_panel(
                seed=seed,
                arms=arms,
                panel_dir=output_root / f"seed_{seed}",
            )
        manifest["panel_summaries"] = panel_summaries
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(manifest_path, manifest)
        raise
    _write_json(manifest_path, manifest)
    print(f"Completed Qwen3-TTS repetition-tail workflow: {output_root}")


if __name__ == "__main__":
    main()
