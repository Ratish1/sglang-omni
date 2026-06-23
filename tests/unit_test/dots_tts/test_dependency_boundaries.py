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
