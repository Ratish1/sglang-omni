"""#2406 serving counter: one stderr line per vocoder step with its plan, rows and
wall time, and on a tree with the prefix cache one line per causal step with the
rows that ran on the cache, the rows that fell back to the whole-history hop and
the pool's free frames. Put this directory on PYTHONPATH and set
OMNI_PREFIX_COUNT=1. With OMNI_MEMORY_PROBE=1 each voc.step line also carries the device
memory allocated before the step, the peak allocated during it (the process's, so AR
work in the same window counts), the reserved memory after it, and the step's history
tokens (prompt plus generated, summed and largest row). No runtime code changes."""

import importlib.abc
import importlib.util
import os
import sys
import time

TARGET = "sglang_omni.models.fun_cosyvoice3.streaming_vocoder"
MEMORY_PROBE = os.environ.get("OMNI_MEMORY_PROBE") == "1"
GIB = 2**30


def patch(module):
    scheduler = module.FunCosyVoice3StreamingVocoderScheduler
    run_step = scheduler.run_step
    ingest = scheduler.ingest
    first_chunk_at = {}

    def timed_ingest(self, request_id, state, codes):
        first_chunk_at.setdefault(request_id, self.clock())
        return ingest(self, request_id, state, codes)

    def timed_step(self, participants, plan):
        now = self.clock()
        waits = [
            (now - state.ready_since) * 1e3
            for _, state in participants
            if state.ready_since is not None
        ]
        taken = {request_id for request_id, _ in participants}
        unstarted = {
            request_id: (now - state.ready_since) * 1e3
            for request_id, state in participants
            if state.first_emit_at is None and state.ready_since is not None
        }
        left_unstarted = [
            (now - state.ready_since) * 1e3
            for request_id, state in self.stream_state_items()
            if request_id not in taken
            and state.first_emit_at is None
            and state.ready_since is not None
            and state.next_decode() == "causal_window"
        ]
        ready = sum(
            state.next_decode() != "wait" for _, state in self.stream_state_items()
        )
        memory = ""
        if MEMORY_PROBE:
            import torch

            history = [
                len(state.tokens)
                + (0 if state.prompt_token is None else state.prompt_token.shape[1])
                for _, state in participants
            ]
            torch.cuda.reset_peak_memory_stats()
            allocated_before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        decoded = run_step(self, participants, plan)
        step_ms = (time.perf_counter() - started) * 1e3
        if MEMORY_PROBE:
            memory = (
                f"alloc_gib={allocated_before / GIB:.3f} "
                f"peak_gib={torch.cuda.max_memory_allocated() / GIB:.3f} "
                f"reserved_gib={torch.cuda.memory_reserved() / GIB:.3f} "
                f"tokens={sum(history)} max_tokens={max(history, default=0)} "
            )
        print(
            f"voc.step t={time.time():.3f} plan={plan} rows={len(participants)} "
            f"ms={step_ms:.1f} {memory}"
            f"wait_max={max(waits, default=0):.1f} "
            f"wait_mean={sum(waits) / max(len(waits), 1):.1f} ready={ready} "
            f"first_rows={len(unstarted)} "
            f"first_wait_max={max(unstarted.values(), default=0):.1f} "
            f"first_left={len(left_unstarted)} "
            f"first_left_wait_max={max(left_unstarted, default=0):.1f}",
            file=sys.stderr,
            flush=True,
        )
        for request_id, state in participants:
            if request_id in unstarted and state.first_emit_at is not None:
                chunk_at = first_chunk_at.get(request_id, now)
                print(
                    f"voc.first t={time.time():.3f} "
                    f"chunk_to_ready_ms={(now - unstarted[request_id] / 1e3 - chunk_at) * 1e3:.1f} "
                    f"ready_to_step_ms={unstarted[request_id]:.1f} step_ms={step_ms:.1f} "
                    f"rows={len(participants)}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                pass
        return decoded

    scheduler.ingest = timed_ingest
    scheduler.run_step = timed_step
    if MEMORY_PROBE:
        vocode_payloads = scheduler.vocode_payloads

        async def measured_payloads(self, payloads):
            import torch

            torch.cuda.reset_peak_memory_stats()
            allocated_before = torch.cuda.memory_allocated()
            started = time.perf_counter()
            decoded = await vocode_payloads(self, payloads)
            print(
                f"voc.batch t={time.time():.3f} rows={len(payloads)} "
                f"ms={(time.perf_counter() - started) * 1e3:.1f} "
                f"alloc_gib={allocated_before / GIB:.3f} "
                f"peak_gib={torch.cuda.max_memory_allocated() / GIB:.3f} "
                f"reserved_gib={torch.cuda.memory_reserved() / GIB:.3f}",
                file=sys.stderr,
                flush=True,
            )
            return decoded

        scheduler.vocode_payloads = measured_payloads
    if not hasattr(scheduler, "hop_batch_with_prefix"):
        return
    hop_batch_with_prefix = scheduler.hop_batch_with_prefix

    def counted(self, participants, items):
        pool = self.vocoder.flow.prefix_pool
        before = {id(state): state.flow_cache is not None for _, state in participants}
        started = time.perf_counter()
        mels = hop_batch_with_prefix(self, participants, items)
        host_ms = (time.perf_counter() - started) * 1e3
        cached = sum(state.flow_cache is not None for _, state in participants)
        dropped = sum(
            before[id(state)] and state.flow_cache is None for _, state in participants
        )
        free = -1 if pool is None else pool.free_frames
        held = 0
        spent = 0
        spent_streams = 0
        for _, state in self.stream_state_items():
            if state.flow_cache is None:
                continue
            frames = sum(row.allocated_frames for row in state.flow_cache)
            held += frames
            if state.next_decode() == "leftover":
                spent += frames
                spent_streams += 1
        print(
            f"prefix.step t={time.time():.3f} rows={len(participants)} cached={cached} "
            f"plain={len(participants) - cached} dropped={dropped} free_frames={free} "
            f"host_ms={host_ms:.1f} held={held} spent={spent} "
            f"spent_streams={spent_streams}",
            file=sys.stderr,
            flush=True,
        )
        return mels

    scheduler.hop_batch_with_prefix = counted


def patch_prefix_graph() -> None:
    """One stderr line per prefix graph call (#2516 and its rework): hit or miss, the
    step's rows and new frames."""
    module = sys.modules.get("sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph")
    if module is None:
        return
    for name in ("PrefixCudaGraphCache", "PrefixCudaGraphRunner"):
        cls = getattr(module, name, None)
        if cls is None:
            continue
        run = cls.run

        def counted_run(self, *args, _run=run, **kwargs):
            started = time.perf_counter()
            generated = _run(self, *args, **kwargs)
            new_frames = kwargs["new_frames"]
            print(
                f"graph.call t={time.time():.3f} hit={int(generated is not None)} "
                f"rows={len(new_frames)} frames={sum(new_frames)} "
                f"host_ms={(time.perf_counter() - started) * 1e3:.1f}",
                file=sys.stderr,
                flush=True,
            )
            return generated

        cls.run = counted_run


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen = False

    def find_spec(self, name, path, target=None):
        if name != TARGET or self.seen:
            return None
        self.seen = True
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)
            patch_prefix_graph()

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("OMNI_PREFIX_COUNT") == "1":
    sys.meta_path.insert(0, PatchOnImport())
