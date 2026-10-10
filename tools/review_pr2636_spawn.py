import json
import multiprocessing
import os
import threading
from collections import Counter
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event

from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage_workers import (
    StageLaunchConfig,
    StageWorkerProcessSpec,
    spawn_affinity,
)
from sglang_omni.utils.cpu import gpu_local_affinity
from tests.unit_test.fixtures import affinity_probe

IMPORT_PARENT_PRESENT = multiprocessing.parent_process() is not None
IMPORT_PYTHON_THREADS = [thread.name for thread in threading.enumerate()]


def report_spawn_threads(
    spec: StageWorkerProcessSpec,
    ready_event: Event,
    startup_error_channel: Queue,
) -> None:
    masks = Counter(
        tuple(sorted(os.sched_getaffinity(int(thread_id))))
        for thread_id in os.listdir("/proc/self/task")
    )
    startup_error_channel.put(
        {
            "fixture_module": affinity_probe.__name__,
            "import_parent_present": IMPORT_PARENT_PRESENT,
            "target_parent_present": multiprocessing.parent_process() is not None,
            "python_threads_at_import": IMPORT_PYTHON_THREADS,
            "python_threads_at_target": [
                thread.name for thread in threading.enumerate()
            ],
            "native_thread_masks": [
                {"cpus": list(cpus), "threads": count} for cpus, count in masks.items()
            ],
            "planned_cpus": sorted(spec.cpu_affinity),
        }
    )


def main() -> None:
    launcher_cpus = os.sched_getaffinity(0)
    target_cpus = gpu_local_affinity([0])
    assert target_cpus
    spec = StageWorkerProcessSpec(
        "probe", [StageLaunchConfig("preprocessing")], cpu_affinity=target_cpus
    )
    stage_workers.stage_process_main = report_spawn_threads
    group = stage_workers.StageGroup("probe", [spec])
    context = multiprocessing.get_context("spawn")
    try:
        group.spawn(context)
        report = group.startup_error_channels[0].get(timeout=60)
        process = group.processes[0]
        process.join(timeout=30)
        assert process.exitcode == 0
        assert os.sched_getaffinity(0) == launcher_cpus
        report["launcher_restored_after_spawn"] = True
        try:
            with spawn_affinity(spec):
                assert os.sched_getaffinity(0) == target_cpus
                raise RuntimeError("intentional spawn-boundary failure")
        except RuntimeError as error:
            assert str(error) == "intentional spawn-boundary failure"
        assert os.sched_getaffinity(0) == launcher_cpus
        report["launcher_restored_after_exception"] = True
        print(json.dumps(report, indent=2))
    finally:
        for process in group.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            else:
                pass
            process.close()
        group.close_control_channels()


if __name__ == "__main__":
    main()
