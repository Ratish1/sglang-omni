# SPDX-License-Identifier: Apache-2.0
"""Resolve launcher values from an SGLang Omni pipeline config."""

from __future__ import annotations

import argparse
from pathlib import Path

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_static_factory_args


def resolve_max_total_tokens(config_path: str | Path) -> int:
    """Return the configured generation-stage KV token cap."""

    pipeline_config = ConfigManager.from_file(str(config_path)).config
    stage_name = (
        type(pipeline_config).generation_sglang_role_to_stage().get("generation")
    )
    if stage_name is None:
        raise ValueError(
            f"{type(pipeline_config).__name__} does not declare a generation stage"
        )

    stage = next(
        (stage for stage in pipeline_config.stages if stage.name == stage_name),
        None,
    )
    if stage is None:
        raise ValueError(
            f"generation stage {stage_name!r} is missing from the pipeline"
        )

    factory_args = resolve_stage_static_factory_args(stage, pipeline_config)
    value = factory_args.get("server_args_overrides", {}).get("max_total_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "the generation stage must define a positive integer max_total_tokens"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="SGLang Omni pipeline config")
    args = parser.parse_args()
    try:
        print(resolve_max_total_tokens(args.config))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
