# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path


def test_dots_dependencies_are_grouped_with_main_model_dependencies() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text()
    )
    dependencies = set(pyproject["project"]["dependencies"])
    dots_dependencies = {
        "torchdiffeq",
        "langcodes[data]",
        "lingua-language-detector",
        "WeTextProcessing",
    }

    assert dots_dependencies.issubset(dependencies)
    assert "optional-dependencies" not in pyproject["project"]


def test_dots_smoke_tests_do_not_append_local_runtime_paths() -> None:
    root = Path(__file__).resolve().parents[3]
    smoke_paths = [
        root / "tests/smoke/dots_tts_native_sglang_smoke.py",
        root / "tests/smoke/dots_tts_server_e2e_smoke.py",
    ]

    for path in smoke_paths:
        source = path.read_text(encoding="utf-8")
        assert ".conda-dots-tts-py310" not in source
        assert "site-packages" not in source
