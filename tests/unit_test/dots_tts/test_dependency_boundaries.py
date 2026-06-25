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
    source = (root / "sglang_omni/models/dots_tts/model_runner.py").read_text(
        encoding="utf-8"
    )
    forbidden = [
        "step_audio_latent",
        "_decode_next_audio",
        "_generate_latents_stream",
        "generate_audio_stream",
    ]
    for symbol in forbidden:
        assert symbol not in source


def test_side_model_warmup_does_not_use_full_generation_api() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (
        root / "sglang_omni/models/dots_tts/native/side_runtime.py"
    ).read_text(encoding="utf-8")

    assert "run_warmup = DotsTtsModel.run_warmup" not in source
    assert "def run_warmup(" in source


def test_vendored_model_does_not_expose_upstream_full_runtime_surfaces() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (
        root / "sglang_omni/models/dots_tts/native/models/dots_tts/model.py"
    ).read_text(encoding="utf-8")
    forbidden_defs = [
        "def run_warmup(",
        "def generate_audio(",
        "def generate_audio_stream(",
        "def _generate_latents_stream(",
        "def _prefill(",
        "def _decode(",
        "def _consume_text_schedule(",
        "def _consume_audio_patch(",
    ]

    for symbol in forbidden_defs:
        assert symbol not in source


def test_native_adapter_uses_side_model_serving_api_boundary() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (
        root / "sglang_omni/models/dots_tts/native_adapter.py"
    ).read_text(encoding="utf-8")
    forbidden = [
        "_prepare_prompt_conditioning",
        "_find_audio_span_positions",
        "_allocate_generate_state",
        "_prefill_prompt_latents",
        "_locate_prefill_boundary",
        "_build_prefill_inputs_embeds",
    ]

    assert "prepare_request(" in source
    for symbol in forbidden:
        assert symbol not in source


def test_sglang_model_uses_side_model_serving_api_boundary() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (
        root / "sglang_omni/models/dots_tts/sglang_model.py"
    ).read_text(encoding="utf-8")
    forbidden = [
        "native_model._append_hidden_chunk",
        "native_model._decode_next_audio",
        "native_model._encode_audio_patch_feedback",
        "native_model._should_stop_after_current_audio",
        "native_model.decode_audio_step(",
    ]

    assert "native_model.decode_audio_batch_step(" in source
    for symbol in forbidden:
        assert symbol not in source
