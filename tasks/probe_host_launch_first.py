"""Host-time breakdown probe for the prefill and decode paths (experiment only).

Wall clocks (perf_counter) around the scheduler step, the runner sub-steps,
and the pieces inside sglang's forward that the code says dominate host time:
the breakable prefill graph replay (graph segments vs eager attention breaks),
attention metadata build, multimodal embed, LM tail, sampling, and the
scheduler-side prepare_for_extend/prepare_for_decode. No cProfile. Every 5 s
one `[host-probe]` line (window) and one `[host-probe-cum]` line (cumulative);
each key: n, med, mean, p90, sum (ms).

Usage (from the checkout root, arm A or arm B):
    python tasks/probe_host_launch_first.py sglang_omni/scheduling/omni_scheduler.py
Revert with `git checkout -- sglang_omni/scheduling/omni_scheduler.py`
(only after committing or stashing any other edits to that file).

Keys (kind is extend or decode; per-launch sums, nested where noted):
    input                           recv_requests + process_input_requests wall, loop turns with >=1 request
    input_reqs                      requests handled in that turn (count, not ms)
    schedule:<kind>                 get_next_batch_to_run wall (kind = batch it returned; none if no batch)
    sched.new_prefill               get_new_batch_prefill wall (inside schedule:extend)
    sched.prepare_extend            ScheduleBatch.prepare_for_extend (inside sched.new_prefill)
    sched.prepare_decode            ScheduleBatch.prepare_for_decode (inside schedule:decode)
    launch:<kind>                   _run_batch_launch wall (async lookahead)
    sync_run:<kind>                 run_batch wall (sync path)
    resolve:<kind>                  _run_batch_resolve wall
    process_result:<kind>           process_batch_result wall
    runner.build:<kind>             _build_forward_batch
    runner.prepare_forward:<kind>   _prepare_and_forward (hooks + forward + sample)
    runner.forward_call:<kind>      tp_worker.forward_batch_generation (model_runner.forward + sample)
    fwd.mr_forward:<kind>           model_runner.forward
    fwd.pcg:<kind>                  PrefillCudaGraphRunner.execute (breakable prefill graph path)
    fwd.load_batch:<kind>           PrefillCudaGraphRunner.load_batch (inside fwd.pcg)
    fwd.attn_meta:<kind>            attn_backend.init_forward_metadata (eager, and BCG replay refresh)
    fwd.seg:<kind>                  sum of graph segment replays in one BreakableCUDAGraph.replay
    fwd.brk:<kind>                  sum of eager break fns (attention) in one BreakableCUDAGraph.replay
    fwd.brk_n:<kind>                number of break fns per replay
    fwd.attn:<kind>                 sum of attn_backend.forward calls per launch (eager path and BCG breaks)
    fwd.attn_n:<kind>               number of attn_backend.forward calls per launch
    fwd.embed:<kind>                mm_utils.embed_mm_inputs
    fwd.lm:<kind>                   language_model.forward (layer stack or replay + logits processor)
    fwd.logits:<kind>               logits_processor.forward
    fwd.sample:<kind>               model_runner.sample
"""

import pathlib
import sys

TARGET = pathlib.Path(sys.argv[1])

HELPERS = """
    # ---- host probe (experiment only) -------------------------------------
    def _hp_init(self) -> None:
        if getattr(self, "_hp_ready", False):
            return
        import functools

        self._hp_ready = True
        self._hp_window: dict = {}
        self._hp_cum: dict = {}
        self._hp_window_t0 = time.perf_counter()
        self._hp_started = self._hp_window_t0
        self._hp_kind = "none"
        self._hp_exec_kind = "none"
        self._hp_cur: dict = {}

        def add(key: str, ms: float) -> None:
            for stats in (self._hp_window, self._hp_cum):
                stats.setdefault(key, []).append(ms)

        self._hp_add = add

        def acc(name: str, ms: float, n: int = 1) -> None:
            cur = self._hp_cur
            cur[name] = cur.get(name, 0.0) + ms
            cur[name + "_n"] = cur.get(name + "_n", 0) + n

        def flush_cur() -> None:
            kind = self._hp_exec_kind
            for name, val in self._hp_cur.items():
                if name.endswith("_n"):
                    if name in ("fwd.brk_n", "fwd.attn_n"):
                        add(f"{name}:{kind}", float(val))
                else:
                    add(f"{name}:{kind}", val)
            self._hp_cur = {}

        def timed(key, fn, kind_fn=None):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    kind = kind_fn(*args, **kwargs) if kind_fn else self._hp_exec_kind
                    add(f"{key}:{kind}", (time.perf_counter() - t0) * 1e3)

            return wrapper

        def summed(name, fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    acc(name, (time.perf_counter() - t0) * 1e3)

            return wrapper

        def batch_kind(batch, *args, **kwargs):
            return self._hp_kind_of(batch)

        # scheduler-side
        self.get_new_batch_prefill = timed("sched.new_prefill", self.get_new_batch_prefill, lambda *a, **k: "extend")
        self._run_batch_resolve = timed("resolve", self._run_batch_resolve, batch_kind)
        self.process_batch_result = timed("process_result", self.process_batch_result, batch_kind)
        from sglang.srt.managers import schedule_batch as _sb

        _sb.ScheduleBatch.prepare_for_extend = timed("sched.prepare_extend", _sb.ScheduleBatch.prepare_for_extend, lambda *a, **k: "extend")
        _sb.ScheduleBatch.prepare_for_decode = timed("sched.prepare_decode", _sb.ScheduleBatch.prepare_for_decode, lambda *a, **k: "decode")

        runner = self._model_runner
        if runner is None:
            return

        orig_prepare = runner._prepare_and_forward

        @functools.wraps(orig_prepare)
        def prepare_wrapper(forward_batch, schedule_batch, requests, is_prefill, *args, **kwargs):
            self._hp_exec_kind = "extend" if is_prefill else "decode"
            self._hp_cur = {}
            t0 = time.perf_counter()
            try:
                return orig_prepare(forward_batch, schedule_batch, requests, is_prefill, *args, **kwargs)
            finally:
                add(f"runner.prepare_forward:{self._hp_exec_kind}", (time.perf_counter() - t0) * 1e3)
                flush_cur()

        runner._prepare_and_forward = prepare_wrapper
        runner._build_forward_batch = timed("runner.build", runner._build_forward_batch, lambda so, *a, **k: self._hp_kind_of(so.batch_data))
        tp_worker = runner.tp_worker
        tp_worker.forward_batch_generation = timed("runner.forward_call", tp_worker.forward_batch_generation)

        # inside the forward
        mr = tp_worker.model_runner
        mr.forward = summed("fwd.mr_forward", mr.forward)
        mr.sample = summed("fwd.sample", mr.sample)
        ab = mr.attn_backend
        ab.forward = summed("fwd.attn", ab.forward)
        ab.init_forward_metadata = summed("fwd.attn_meta", ab.init_forward_metadata)
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner as _PCG,
        )

        pcg = getattr(mr, "prefill_cuda_graph_runner", None)
        if isinstance(pcg, _PCG):
            pcg.execute = summed("fwd.pcg", pcg.execute)
            pcg.load_batch = summed("fwd.load_batch", pcg.load_batch)
        from sglang.srt.managers import mm_utils as _mm

        _mm.embed_mm_inputs = summed("fwd.embed", _mm.embed_mm_inputs)
        model = mr.model
        lm = getattr(model, "language_model", model)
        lm.forward = summed("fwd.lm", lm.forward)
        lp = getattr(lm, "logits_processor", None)
        if lp is not None:
            lp.forward = summed("fwd.logits", lp.forward)
        try:
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
                breakable_cuda_graph as _bcg,
            )
        except Exception:
            _bcg = None
        if _bcg is not None:

            def replay(graph_self):
                stream = torch.cuda.current_stream()
                token = _bcg._current_stream_var.set(stream)
                seg_ms = 0.0
                brk_ms = 0.0
                n_brk = 0
                try:
                    for i, seg in enumerate(graph_self._segments):
                        t0 = time.perf_counter()
                        seg.replay()
                        seg_ms += (time.perf_counter() - t0) * 1e3
                        if i < len(graph_self._break_fns):
                            t0 = time.perf_counter()
                            graph_self._break_fns[i]()
                            brk_ms += (time.perf_counter() - t0) * 1e3
                            n_brk += 1
                finally:
                    _bcg._current_stream_var.reset(token)
                    acc("fwd.seg", seg_ms)
                    acc("fwd.brk", brk_ms, n_brk)

            _bcg.BreakableCUDAGraph.replay = replay
        logger.warning(
            "[host-probe] installed: pcg=%s attn_backend=%s lm=%s",
            type(pcg).__name__ if pcg is not None else None,
            type(ab).__name__,
            type(lm).__name__,
        )

    def _hp_kind_of(self, batch) -> str:
        if batch is None:
            return "none"
        return "decode" if self._batch_is_decode(batch) else "extend"

    def _hp_flush(self) -> None:
        now = time.perf_counter()
        if now - self._hp_window_t0 >= 5.0:
            self._hp_log("host-probe", self._hp_window, now - self._hp_window_t0)
            self._hp_log("host-probe-cum", self._hp_cum, now - self._hp_started)
            self._hp_window = {}
            self._hp_window_t0 = now

    @staticmethod
    def _hp_log(tag: str, stats: dict, seconds: float) -> None:
        import statistics

        parts = [f"[{tag}] window={seconds:.1f}s"]
        for key in sorted(stats):
            vals = stats[key]
            if not vals:
                continue
            vs = sorted(vals)
            p90 = vs[int(0.9 * (len(vs) - 1))]
            parts.append(
                f"{key} n={len(vals)} med={statistics.median(vals):.3f} "
                f"mean={statistics.fmean(vals):.3f} p90={p90:.3f} sum={sum(vals):.1f}"
            )
        logger.info(" | ".join(parts))

"""

EDITS = [
    (
        "    def _event_loop_async_decode(self) -> None:\n",
        HELPERS + "    def _event_loop_async_decode(self) -> None:\n",
    ),
    (
        "        while self._running:\n            self._process_admin_requests()\n            recv_reqs = self.recv_requests()\n            recv_reqs.extend(self._take_deferred_request_payloads())\n            self.process_input_requests(recv_reqs)\n            if self._engine_paused:\n                self._process_admin_requests()\n                self._resolve_pending_async()\n",
        '        self._hp_init()\n        while self._running:\n            self._hp_flush()\n            self._process_admin_requests()\n            hp_t0 = time.perf_counter()\n            recv_reqs = self.recv_requests()\n            recv_reqs.extend(self._take_deferred_request_payloads())\n            self.process_input_requests(recv_reqs)\n            if recv_reqs:\n                self._hp_add("input", (time.perf_counter() - hp_t0) * 1e3)\n                self._hp_add("input_reqs", float(len(recv_reqs)))\n            if self._engine_paused:\n                self._process_admin_requests()\n                self._resolve_pending_async()\n',
    ),
    (
        "                self._resolve_pending_async()\n\n            batch = self.get_next_batch_to_run()\n            self.cur_batch = batch\n",
        '                self._resolve_pending_async()\n\n            hp_t0 = time.perf_counter()\n            batch = self.get_next_batch_to_run()\n            self._hp_kind = self._hp_kind_of(batch)\n            self._hp_add(f"schedule:{self._hp_kind}", (time.perf_counter() - hp_t0) * 1e3)\n            self.cur_batch = batch\n',
    ),
    (
        "                try:\n                    sched_output, pending_step = self._run_batch_launch(batch)\n                except Exception as exc:\n                    self._handle_batch_failure(batch, exc)\n                else:\n                    prev_pending = self._async_pending\n",
        '                try:\n                    hp_t0 = time.perf_counter()\n                    try:\n                        sched_output, pending_step = self._run_batch_launch(batch)\n                    finally:\n                        self._hp_add(f"launch:{self._hp_kind}", (time.perf_counter() - hp_t0) * 1e3)\n                except Exception as exc:\n                    self._handle_batch_failure(batch, exc)\n                else:\n                    prev_pending = self._async_pending\n',
    ),
    (
        "                if batch:\n                    result = self.run_batch(batch)\n                    if result is not _FAILED_BATCH_RESULT:\n                        self.process_batch_result(batch, result)\n",
        '                if batch:\n                    hp_t0 = time.perf_counter()\n                    try:\n                        result = self.run_batch(batch)\n                    finally:\n                        self._hp_add(f"sync_run:{self._hp_kind}", (time.perf_counter() - hp_t0) * 1e3)\n                    if result is not _FAILED_BATCH_RESULT:\n                        self.process_batch_result(batch, result)\n',
    ),
]

s = TARGET.read_text()
for old, new in EDITS:
    assert s.count(old) == 1, f"anchor not unique/found: {old[:60]!r}"
    s = s.replace(old, new)
TARGET.write_text(s)
print("host probe applied to", TARGET)
