from __future__ import annotations

import math
from pathlib import Path

from benchmarks.eval.run_qwen3_tts_repetition_tail import (
    _build_arg_parser,
    build_generation_command,
    repetition_arms,
)


def test_repetition_tail_arms_isolate_owner_and_effective_strength() -> None:
    arms = {arm.name: arm for arm in repetition_arms()}

    sglang = arms["sglang_once_p105"]
    assert (sglang.owner, sglang.qwen_penalty, sglang.sglang_penalty) == (
        "sglang",
        1.0,
        1.05,
    )
    qwen = arms["qwen_once_p105"]
    assert (qwen.owner, qwen.qwen_penalty, qwen.sglang_penalty) == (
        "qwen",
        1.05,
        1.0,
    )
    equal_effective = arms["double_sqrt_p105"]
    assert equal_effective.owner == "double"
    assert math.isclose(equal_effective.nominal_effective_penalty, 1.05)
    original_double = arms["double_p105"]
    assert original_double.owner == "double"
    assert math.isclose(original_double.nominal_effective_penalty, 1.1025)


def test_repetition_tail_generation_command_has_one_to_one_capture_contract(
    tmp_path: Path,
) -> None:
    args = _build_arg_parser().parse_args(
        [
            "--model",
            "model",
            "--output-dir",
            str(tmp_path),
            "--seeds",
            "20260823,20260824",
            "--max-samples",
            "42",
        ]
    )
    arm = repetition_arms()[0]
    command = build_generation_command(
        args,
        arm=arm,
        seed=20260823,
        output_dir=tmp_path / arm.name,
    )

    assert command[command.index("--seed") + 1] == "20260823"
    assert "--sample-specific-seeds" in command
    assert command[command.index("--warmup") + 1] == "0"
    assert command[command.index("--concurrency") + 1] == "16"
    assert command[command.index("--repetition-penalty") + 1] == "1.05"
    assert command[command.index("--max-samples") + 1] == "42"
    assert "--generate-only" in command
