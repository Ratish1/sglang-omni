# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


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


def test_dots_pr_files_do_not_include_local_runtime_paths() -> None:
    root = Path(__file__).resolve().parents[3]
    paths = list((root / "sglang_omni/models/dots_tts").rglob("*.py"))
    paths.extend((root / "tests/unit_test/dots_tts").rglob("*.py"))
    local_conda_env = "." + "conda-dots-tts-py310"
    home_path_prefix = "/" + "home" + "/"
    workspace_models = "workspace" + "/" + "models"
    site_packages = "site" + "-packages"

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert local_conda_env not in source
        assert site_packages not in source
        assert home_path_prefix not in source
        assert workspace_models not in source


def test_dots_runner_does_not_invoke_model_private_latent_step() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (
        root / "sglang_omni/models/dots_tts/model_runner.py"
    ).read_text(encoding="utf-8")
    forbidden = [
        "step_audio_latent",
        "_decode_next_audio",
        "_generate_latents_stream",
        "generate_audio_stream",
    ]
    for symbol in forbidden:
        assert symbol not in source
