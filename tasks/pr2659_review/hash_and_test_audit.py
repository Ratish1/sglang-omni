from __future__ import annotations

import ast
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import replace

import pytest
import torch

from sglang_omni.models.qwen3_tts.prompt_frontend import Qwen3TTSPromptFrontend
from sglang_omni.preprocessing.cache_key import (
    PROMPT_ROW_KEY_BASES,
    PROMPT_ROW_KEY_MAX_WORDS,
    PROMPT_ROW_KEY_MODULUS,
    PromptRowCacheKeys,
)

BASE = "284381f3a6f6207eeb1f89310173caf410f123df"


def run() -> None:
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:sglang_omni/models/qwen3_tts/request_builders.py"],
        text=True,
    )
    reference_function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_embedding_cache_key_ids"
    )
    namespace = {"torch": torch, "hashlib": hashlib}
    exec(
        compile(
            ast.Module(body=[reference_function], type_ignores=[]),
            "reference-row-keys",
            "exec",
        ),
        namespace,
    )
    reference_keys = namespace["build_embedding_cache_key_ids"]
    torch.manual_seed(129)
    timings: list[dict[str, int | float]] = []
    for width in (1024, 2048):
        keys = PromptRowCacheKeys.for_rows(
            row_bytes=width * 2, device=torch.device("cuda")
        )
        for row_count in (1, 8, 32, 128, 512):
            rows = torch.randn(row_count, width, device="cuda", dtype=torch.bfloat16)
            for _ in range(10):
                reference_keys(rows)
                keys.key_ids(rows)
            old_times: list[float] = []
            new_times: list[float] = []
            for repeat in range(60):
                measurements = ((reference_keys, old_times), (keys.key_ids, new_times))
                if repeat % 2:
                    measurements = tuple(reversed(measurements))
                else:
                    pass
                for compute, durations in measurements:
                    started = time.perf_counter_ns()
                    compute(rows)
                    durations.append((time.perf_counter_ns() - started) / 1_000_000)
            timings.append(
                dict(
                    width=width,
                    rows=row_count,
                    reference_ms=statistics.median(old_times),
                    candidate_ms=statistics.median(new_times),
                )
            )
    maximum_words = torch.full((1, PROMPT_ROW_KEY_MAX_WORDS), -1, dtype=torch.int16)
    expected_lanes = [
        (
            65535
            * sum(
                pow(base, word, PROMPT_ROW_KEY_MODULUS)
                for word in range(PROMPT_ROW_KEY_MAX_WORDS)
            )
        )
        % PROMPT_ROW_KEY_MODULUS
        for base in PROMPT_ROW_KEY_BASES
    ]
    expected_id = (expected_lanes[0] << 31) | expected_lanes[1]
    for device in (torch.device("cpu"), torch.device("cuda")):
        keys = PromptRowCacheKeys.for_rows(
            row_bytes=maximum_words.numel() * 2, device=device
        )
        assert keys.key_ids(maximum_words.to(device)) == [expected_id]
    print(
        json.dumps({"timings": timings, "maximum_word_bound": "pass"}, indent=2),
        flush=True,
    )

    original_constants = Qwen3TTSPromptFrontend.build_prompt_constants

    def wrong_eos_constants(self):
        constants = original_constants(self)
        return replace(
            constants, tts_eos_embed=torch.zeros_like(constants.tts_eos_embed)
        )

    Qwen3TTSPromptFrontend.build_prompt_constants = wrong_eos_constants
    try:
        result = pytest.main(
            [
                "-q",
                "tests/unit_test/qwen3_tts/test_pipeline.py::test_qwen3_tts_prompt_frontend_builds_a_custom_voice_prompt",
            ]
        )
        assert result == 0, result
        print(
            "MUTATION: the existing prompt-constant test accepts a zero EOS embedding",
            flush=True,
        )
    finally:
        Qwen3TTSPromptFrontend.build_prompt_constants = original_constants


if __name__ == "__main__":
    run()
