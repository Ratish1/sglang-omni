"""Check local checkpoint overrides without remote metadata availability."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml
from huggingface_hub.errors import LocalEntryNotFoundError

from sglang_omni.cli.config import resolve_sources

observations = []
with tempfile.TemporaryDirectory() as directory:
    directory_path = Path(directory)
    for variant in ("base", "instruct"):
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
                side_effect=LocalEntryNotFoundError(
                    "offline: remote metadata is not cached"
                ),
            ) as download:
                observation = {
                    "variant": variant,
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
