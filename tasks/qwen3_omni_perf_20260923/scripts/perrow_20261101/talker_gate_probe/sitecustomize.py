"""How often the Qwen3-Omni talker's all-rows decode gate defers a step, and what it holds (audit E9 step 1).

Put this directory on the server's PYTHONPATH. Once sglang_omni.models.qwen3_omni.talker_scheduler is imported,
QwenTalkerScheduler.is_batch_ready_to_run is wrapped. For every decode batch it sees: ready counts a decode step and
its rows; not ready counts a deferred iteration (main prepares the decode, then rolls it back), its rows, and its
unready rows (QwenTalkerModelRunner.data_has_next_decode_input false, the runner's own test). A run of deferred
iterations is one episode, from its first deferral to the next ready decode batch; its wall time is gated time, its
first iteration's rows and unready rows are the episode's. At most once a second it prints one cumulative GATEPROBE
line to stderr with the wall clock (epoch seconds), so a reader takes any window's deltas.
"""

import importlib.abc
import importlib.util
import os
import sys
import time

TARGET = "sglang_omni.models.qwen3_omni.talker_scheduler"
PRINT_INTERVAL_S = 1.0
FIELDS = (
    "decode_steps",
    "rows_run",
    "deferred",
    "rows_deferred",
    "unready_rows",
    "episodes",
    "episode_rows",
    "episode_unready",
    "gated_s",
)


class GateState:
    def __init__(self):
        self.counts = {field: 0 for field in FIELDS}
        self.gate_started = None
        self.last_print = 0.0


def patch(module):
    from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

    scheduler_cls = module.QwenTalkerScheduler
    ready_original = scheduler_cls.is_batch_ready_to_run
    state = GateState()

    def is_batch_ready_to_run(self, batch):
        ready = ready_original(self, batch)
        if batch is None or not batch.forward_mode.is_decode():
            return ready
        else:
            pass
        now = time.perf_counter()
        counts = state.counts
        rows = len(batch.reqs)
        if ready:
            counts["decode_steps"] += 1
            counts["rows_run"] += rows
            if state.gate_started is not None:
                counts["gated_s"] += now - state.gate_started
                state.gate_started = None
            else:
                pass
        else:
            unready = sum(
                1
                for req in batch.reqs
                if not QwenTalkerModelRunner.data_has_next_decode_input(
                    getattr(req, "omni_data", None)
                )
            )
            counts["deferred"] += 1
            counts["rows_deferred"] += rows
            counts["unready_rows"] += unready
            if state.gate_started is None:
                state.gate_started = now
                counts["episodes"] += 1
                counts["episode_rows"] += rows
                counts["episode_unready"] += unready
            else:
                pass
        if now - state.last_print >= PRINT_INTERVAL_S:
            state.last_print = now
            values = " ".join(f"{field}={counts[field]:.6g}" for field in FIELDS)
            print(
                f"GATEPROBE wall={time.time():.3f} pid={os.getpid()} {values}",
                file=sys.stderr,
                flush=True,
            )
        else:
            pass
        return ready

    scheduler_cls.is_batch_ready_to_run = is_batch_ready_to_run


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        else:
            pass
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None:
            return None
        else:
            pass
        loader = spec.loader
        exec_original = loader.exec_module

        def exec_module(module):
            exec_original(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, PatchOnImport())
