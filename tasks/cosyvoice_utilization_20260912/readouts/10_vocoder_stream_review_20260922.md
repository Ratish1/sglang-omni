# Vocoder stream: code review of `exp/cosyvoice-vocoder-stream` (2026-09-22)

Branch `exp/cosyvoice-vocoder-stream` 589367020, worktree `.worktrees/cosy-x-stream`, based on
upstream main 9f1260e66. One commit, one file, 14 lines in
`sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py`: the scheduler builds one
`torch.cuda.Stream` in `__init__` and `pump_one_step` runs under it.

Measured before this review (`s17-r1`, `s18`): streaming c16 +4.9 % and +6.6 % req/s over two
pairs, TTFP mean -12 to -14 %, c1 +3.8 %, peak memory unchanged. Runs `s19` (buffered c16 pair),
`s20` (seeded c1, main against stream) and `s21` (seeded c1, main against main, the control) are
the identity and buffered gates; their numbers go in section 6.

## 1. What the change does, and why it works

Fun-CosyVoice3 is one process on one card. The AR (Qwen2 0.5B on SGLang) runs with
`disable_overlap_schedule: True` (`engine_builder.py:120`), so its scheduler takes
`event_loop_normal` and `run_batch` reaches the forward without `forward_stream_ctx`
(`scheduler.py:4044-4052`: the context is entered only under `enable_overlap`). Its kernels
therefore go to the thread's current stream, the legacy default stream. The vocoder scheduler
thread (`StreamingSimpleScheduler.start`, `streaming_simple_scheduler.py:135-165`) had the same
current stream. The `s14-gaps` capture shows it: 2,292,901 of 2,430,520 kernels on streamId 7,
the AR's 1.05 million and the vocoder's 1.29 million together (`schema_findings.txt:81`).

One stream serializes both threads' work in submission order: a vocoder step's 17,000 small
kernels and its per row host copies (`hift_delta`, `stages.py:1567`, a synchronous `.cpu()`)
queue behind whatever the AR's decode replay has already submitted, and the AR's next replay
queues behind the vocoder's kernels. With the pump on its own stream the two thread's submissions
are independent; the AR's graph replays fill the vocoder's launch gaps (the GPU idles 47 % of
the window at c16, 20.4 s of 56 s in gaps under 1 ms).

```
scheduler thread (one thread runs everything below, streaming_simple_scheduler.py:135)
  start() loop
    has_ready_work() -> run_ready_step()            [state_lock held]
      pump_one_step()                               <- override: with torch.cuda.stream(step_stream)
        select_step_participants()                  host only
        run_step()                                  streaming_vocoder.py:355
          "fallback": finish_stream() -> fallback_full_decode() -> vocoder.token2wav()
          "leftover": vocoder.leftover_batch() -> hift_delta() per row
          "causal_window": vocoder.hop_batch() -> hift_delta() per row
        outbox.put(...)                             host only
    handle_message() -> run_non_streaming_batch()   buffered path, default stream, same thread
      vocoder.decode_payloads() -> FlowCudaGraphRunner.run() (stages.py:447) -> mel2wav_batch()
  warmup_now()                                      before start(), default stream (finding 1)
AR scheduler thread: event_loop_normal -> run_batch -> forward on the default stream
```

## 2. Line by line

`streaming_vocoder.py:121-127`

- The note: "the AR shares this process and the default stream; on its own stream a step's
  kernels and host copies do not queue behind the AR's". Verified above; both halves are facts.
- `device = next(vocoder.hift.parameters()).device`. Correct on CUDA: `hift.to(device)` at
  `stages.py:906` moves every parameter, so any parameter answers. On the torch MPS path
  `vocoder.hift` is `MpsHiFTAdapter`, whose `parameters()` delegates to the wrapped HiFT
  (`stages.py:101`) after the f0 predictor was moved to the CPU (`stages.py:95`); the first
  parameter can then be a CPU one. The outcome is still right (not "cuda" either way), but the
  anchor is accidental. The earlier `next()` problem in this program (precast,
  `compile_dit_backbone` taking its dummy input dtype from the estimator's first parameter, which
  the precast had turned bfloat16) was a dtype anchor; it does not apply to a device read.
  Better anchors, either is fine: the Flow's parameter, the vocoder's own idiom
  (`stages.py:187`, `:1412`, `:1507`; the Flow runs first in every step and is never split across
  devices), or the executor's `device_obj` (`stages.py:2062`) passed in, the way
  `FlowCudaGraphRunner(flow, device=device_obj)` takes it.
- `torch.cuda.Stream(device=device) if device.type == "cuda" else None`. The check is needed:
  the same scheduler class serves the torch MPS path (`engine_builder.py:100-113`,
  `stages.py:2049-2054`) and `resolve_concrete_device` can return a CPU device; `torch.mps` has no
  `Stream`, so SGLang's device agnostic `create_device_stream` (`utils/common.py:568-572`,
  `torch.get_device_module(device).Stream(device=device)`) would raise there. The predicate is the
  one the flow graph gate uses in the same factory (`stages.py:2063-2066`) and the one the
  sibling vocoders use (`qwen3_tts/streaming_vocoder.py:815`, `moss_tts_local/streaming_vocoder.py:638`).
  A MUSA device has `type == "musa"` and gets no stream, the same as it gets no flow graph today;
  no failure, no gain there. `device=` matters: the stream must be on the vocoder's card, not on
  the process's current device.
- The attribute has no type hint. `self.step_stream: torch.cuda.Stream | None = ...`.

`streaming_vocoder.py:129-134`

- `pump_one_step` is the right seam: `run_ready_step` (`streaming_vocoder.py:394`) and
  `pump_streams` (`:383`) both go through it, so every step, including the fallback's
  `token2wav`, runs under the stream; `select_step_participants` and the outbox put are host
  work, which the context does not affect. Wrapping `run_step` instead would add an indent level
  to a 75 line method for the same kernels.
- The `if`/`else` with a `return` on each side is complete branching; a `nullcontext()` ternary
  would be shorter and less readable. `torch.cuda.stream(...)` sets the thread's current stream
  for the stream's device only, so nothing outside this thread sees it.

## 3. Cross stream ledger (what a second stream can break, and why it does not here)

A tensor written on one stream and read on another needs an event or a `wait_stream`; a tensor
freed while another stream still reads it needs `record_stream`. Every tensor that crosses the
pump boundary:

| tensor | producer | consumer | verdict |
|---|---|---|---|
| speech tokens (`state.tokens`) | `ingest`, scheduler thread, `codes.tolist()` (`:283`), default stream | `run_step` builds a CPU tensor from ints | host ints, no stream |
| `prompt_token`, `prompt_feat`, `embedding` | wire codec `tensor_list` (`pipeline_state.py:116`), `clone_reference_tensor` `.cpu()` (`request_builders.py:160`), `torch.as_tensor` of lists (`streaming.py:150-178`) | `pack_flow_inputs` `.to(device)` on the step stream (`stages.py:206-227`) | CPU tensors, H2D copies stream ordered on the step stream |
| `hift_mel` | `hift_delta` on the step stream (`stages.py:1568`) | next `hift_delta` on the step stream | one stream; freed by `release_stream_resources` from the scheduler thread, and the allocator returns a block to the pool of the stream it was allocated on |
| `delta`, `leftover` | `hift_delta` `.cpu()` (`stages.py:1567`) | outbox, `finish_stream` | a synchronous `.cpu()` copies on the current stream and waits for it: host complete before it leaves the pump |
| buffered path tensors | `decode_payloads` on the default stream, same thread | itself | never shared with the pump; each `pump_one_step` ends in `.cpu()` waits, so the step stream is drained when the thread moves on |
| model weights | load time | both streams | read only |

Nothing needs `wait_stream` or `record_stream`. Numerics: the same kernels in the same order per
stream, so the audio is bit identical by construction; `s20` against the `s21` control is the
measurement of that.

## 4. Findings

1. `[P1]` `warmup_now` (`:146-170`) runs on the default stream, the pump on the step stream.
   Two per stream costs are then paid by the first served request instead of the warmup: the
   caching allocator gives a stream only blocks cached on that stream
   (`c10/cuda/CUDACachingAllocator.cpp`, block comparator and `get_free_block`), so the step
   stream's pool starts empty and the first steps `cudaMalloc` every block size; and the cuBLAS and
   cuBLASLt workspaces are keyed by `(handle, stream)`
   (`aten/src/ATen/cuda/CublasHandlePool.cpp`), so the first matmul on the new stream allocates
   them. Fix: one context used by both, no new state:

   ```python
   def step_context(self):
       if self.step_stream is None:
           return contextlib.nullcontext()
       else:
           return torch.cuda.stream(self.step_stream)
   ```

   `pump_one_step` becomes `with self.step_context(): return super().pump_one_step()`, and
   `warmup_now` wraps its two calls the same way. Not visible in the c16 numbers (warmup 1 hides
   it); visible in the first request's TTFP after boot.
2. `[P3]` type hint on `step_stream` (section 2).
3. `[design, to decide]` Placement. The stream lives in the streaming scheduler, so the buffered
   path keeps the default stream. In a deployment that serves both kinds, the allocator keeps
   two pools and caches each path's peak. Putting the stream in `CosyVoice3Vocoder` (every entry
   point under it, graph replays included: a captured graph replays on whatever stream is
   current) would give one pool and the same AR overlap for buffered calls. Larger change, its own
   measurement; `s19` first tells whether the buffered path is untouched as it stands.
4. `[open, not a finding]` Stream priority is the default 0. The Qwen3-Omni vocoder runs at a
   raised priority; no CosyVoice measurement exists either way, so nothing to set.

## 5. Owed before a PR

- `s20`/`s21`: byte identity of the served audio against a control boot.
- `s19`: buffered c16 unchanged.
- Finding 1 fixed, one more streaming c16 pair on the fixed head (the pairs so far are
  589367020).
- Coding style pass on the fixed head; PR body per the program's rules.

## 6. Runs (filled when they finish)
