"""Every pipeline edge that can carry a direct CUDA IPC payload, for every registered config.

For each PipelineConfig class the registry knows (variants included) and each YAML file given,
builds the runtime prep exactly as the server does (prepare_pipeline_runtime: process
compilation, replica expansion, placement, process topology), then applies the router's rule
for a direct CUDA IPC target (sglang_omni/comm/router.py, direct_cuda_ipc_targets): the target
runs on exactly the sender's one placement GPU, in another process, not remote. An edge is
printed with the sender's disable_direct_cuda_ipc_payload flag; only edges with the flag off
can reach serialize_direct_cuda_ipc_payload. A config that fails to build prints its error.

usage: python direct_ipc_edges.py [config.yaml ...]
"""

from __future__ import annotations

import importlib
import sys
import traceback

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY, resolve_variant
from sglang_omni.pipeline.mp_runner import resolve_same_process_targets
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime


def config_classes() -> dict[str, type[PipelineConfig]]:
    found: dict[str, type[PipelineConfig]] = {}
    for config_cls in PIPELINE_CONFIG_REGISTRY.configs.values():
        found[f"{config_cls.__module__}:{config_cls.__name__}"] = config_cls
        module = importlib.import_module(config_cls.__module__)
        for variant in getattr(module, "Variants", {}).values():
            try:
                resolved = resolve_variant(variant)
            except Exception as exc:
                print(f"VARIANT_IMPORT_FAILED {variant}: {exc!r}")
                continue
            found[f"{resolved.__module__}:{resolved.__name__}"] = resolved
    return found


def edges(label: str, config: PipelineConfig) -> None:
    prep = prepare_pipeline_runtime(config)
    try:
        plan = prep.placement_plan
        by_name = {stage.name: stage for stage in prep.stages_cfg}
        stage_gpu_ids = {
            name: placement.gpu_ids for name, placement in plan.stages.items()
        }
        processes = prep.process_plan.stage_to_process
        rows = []
        for stage in prep.stages_cfg:
            same_process = resolve_same_process_targets(
                stage, by_name, prep.process_plan, prep.replica_topology
            )
            targets: list[tuple[str, str]] = []
            next_stages = (
                [stage.next] if isinstance(stage.next, str) else list(stage.next or [])
            )
            for raw in next_stages:
                targets += [
                    (name, "next") for name in prep.replica_topology.instances(raw)
                ]
            for raw in stage.stream_to:
                targets += [
                    (name, "stream") for name in prep.replica_topology.instances(raw)
                ]
            sender_gpus = stage_gpu_ids.get(stage.name, ())
            for rank, placement_gpu in enumerate(sender_gpus or (None,)):
                for target, kind in targets:
                    eligible = (
                        placement_gpu is not None
                        and stage_gpu_ids.get(target) == (placement_gpu,)
                        and target not in same_process
                    )
                    rows.append(
                        f"  {stage.name}[rank {rank}, gpu {placement_gpu}, "
                        f"process {processes.get(stage.name, 'tp')}] -> {target} "
                        f"({kind}, gpus {stage_gpu_ids.get(target)}, "
                        f"process {processes.get(target, 'tp')}): "
                        f"{'DIRECT' if eligible else 'no'}"
                        f"{' disabled' if eligible and stage.disable_direct_cuda_ipc_payload else ''}"
                    )
        print(f"CONFIG {label}")
        for row in rows:
            print(row)
    finally:
        prep.runtime_dir.close()


def main() -> None:
    for label, config_cls in sorted(config_classes().items()):
        try:
            config = config_cls(model_path="unused/model")
            edges(label, config)
        except Exception as exc:
            print(f"CONFIG {label} BUILD_FAILED {exc!r}")
            traceback.print_exc(limit=2, file=sys.stdout)
    for path in sys.argv[1:]:
        try:
            edges(path, ConfigManager.from_file(path).config)
        except Exception as exc:
            print(f"CONFIG {path} BUILD_FAILED {exc!r}")


if __name__ == "__main__":
    main()
