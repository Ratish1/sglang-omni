"""Check local checkpoint overrides without remote metadata availability."""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from sglang_omni.cli.config import resolve_sources

parser = argparse.ArgumentParser()
parser.add_argument("--complete-checkpoints", action="store_true")
arguments = parser.parse_args()
observations = []
with tempfile.TemporaryDirectory() as directory:
    directory_path = Path(directory)

    def load_uncached_remote_metadata(
        repo_id: str, filename: str, revision: str | None = None
    ) -> str:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(directory_path / "empty-hub-cache"),
            local_files_only=True,
        )

    for variant in ("base", "instruct"):
        if arguments.complete_checkpoints:
            revision = (
                "e4133ef7fc8edb61883da42c584deb72f28c8c5a"
                if variant == "base"
                else "1fa25f92a13679e6f1a52a867197a6d3b4ebfef7"
            )
            checkpoint_path = Path(
                snapshot_download(
                    repo_id=f"FireRedTeam/FireRedTTS3-{variant.capitalize()}-HF",
                    revision=revision,
                    local_files_only=True,
                )
            )
            required_assets = [
                "config.json",
                "tokenizer.json",
                "model.safetensors",
                "redae/model.safetensors",
            ]
            if variant == "base":
                required_assets.append("campp/campplus_voxceleb.bin")
            else:
                pass
            assert all(
                (checkpoint_path / filename).is_file() for filename in required_assets
            )
        else:
            checkpoint_path = directory_path / variant
            checkpoint_path.mkdir()
            metadata = (Path(__file__).parent / f"{variant}_config.json").read_text()
            (checkpoint_path / "config.json").write_text(metadata)
        example_path = Path(f"examples/configs/firered_tts3_{variant}.yaml")
        profile = yaml.safe_load(example_path.read_text())
        profile["model_path"] = str(checkpoint_path)
        local_profile_path = directory_path / f"{variant}.yaml"
        local_profile_path.write_text(yaml.safe_dump(profile))
        for configuration_path in (example_path, local_profile_path):
            with patch(
                "sglang_omni.models.firered_tts3.hf_config.hf_hub_download",
                side_effect=load_uncached_remote_metadata,
            ) as download:
                observation = {
                    "variant": variant,
                    "complete_checkpoint": arguments.complete_checkpoints,
                    "profile": (
                        "example" if configuration_path == example_path else "local"
                    ),
                }
                try:
                    resolved = resolve_sources(
                        model_path=str(checkpoint_path),
                        config_file=str(configuration_path),
                        text_only=False,
                        variant=None,
                        mem_fraction_static=None,
                        argv=[],
                    )
                except LocalEntryNotFoundError as error:
                    observation.update(succeeded=False, error=str(error))
                else:
                    observation.update(
                        succeeded=True,
                        resolved_path=resolved.resolved.config.model_path,
                        resolved_variant=resolved.resolved.config.variant,
                    )
                observation["download_calls"] = [
                    call.kwargs for call in download.call_args_list
                ]
                observations.append(observation)
print(json.dumps(observations, indent=2))
assert all(row["succeeded"] for row in observations if row["profile"] == "local")
assert not any(row["succeeded"] for row in observations if row["profile"] == "example")
