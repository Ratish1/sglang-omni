# 19. Qwen3-TTS predictor graph capture: measurement and plan

Run 2 (doc 17 section 4) showed the Qwen3-TTS code predictor graph
being captured inside serving steps, and every capture costing about
half a second with the whole batch waiting. The first response was a
startup warmup of the bucket ladder written straight from the ledger
finding, with no design step. It changed when captures happen and left
what a capture costs untouched, and its default sampling signature came
from dataclass field defaults rather than the checkpoint's merged
generation config. That draft was dropped (its patch is kept outside the
tree). This document is the design step it skipped: the mechanics as
read, the evidence, the hypotheses with the measurement that decides
each, and the design options gated on those numbers. No fix is written
until section 5 has numbers.

Anchors are main 15c4568bb for omni, v0.5.18 for sglang, and v2.13.0
for PyTorch, the pin in pyproject and the version on the H100 box. The
local clones at /Users/ratish/sglang and /Users/ratish/pytorch are
checked out at those tags.

## 1. Mechanics as read

### 1.1 The omni capture path

```
decode or prefill step
  qwen3_tts/model_runner.py:210 _collect_codes
    :230 model.code_predictor_forward(layer0_codes, hidden, semantic_positions)
      qwen3_tts/sglang_model.py:1177 code_predictor_forward
        :1306 _predictor_forward_graphed
          key = (bucket, mode, top_k ladder width, has_top_p, unbounded_top_k)
          hit  -> _PredictorDecodeGraph.replay :167  copies into static
                  buffers, graph.replay(), slices outputs
          miss -> :1340 cap check (32 keys, then eager forever for new keys)
               -> _PredictorDecodeGraph.__init__ :122 -> _capture :157
                    :133 new warmup stream, wait on the serving stream
                    :136 two passes of _code_predictor_forward_incremental
                         (for_capture=True) on the warmup stream
                    :143 serving stream waits on the warmup stream
                    :145 new capture stream, wait on the serving stream
                    :147 torch.cuda.graph(graph, pool, capture stream,
                         capture_error_mode="thread_local")
                           torch/cuda/graphs.py:439 torch.cuda.synchronize()
                           :449 torch.cuda.empty_cache()
                           :451 torch._C._host_emptyCache()
                           :462 capture_begin
                    :153 one pass of the chain, recorded
                           graphs.py:477 capture_end (instantiate)
                    :161 serving stream waits on the capture stream
```

The chain (`_code_predictor_forward_incremental`, :1358) runs, per
token, two one token forwards through the code predictor and then one
per remaining code group, each through every predictor layer with a
per step KV cache, and samples one code per group. With the checkpoint's
config (code predictor 5 layers, 16 code groups, hidden 1024, vocab
2048) that is 16 one token forwards over 5 layers plus 15 sampling
groups per token.

Three fused paths exist and are gated on
`torch.cuda.is_current_stream_capturing()`: the fused codec embedding
gather (:1496), the fused seeded top k top p sampler (:1712) and the
fused addmm for the attention output projection (:1820). The two warmup
passes are not capturing, so they run the eager branches. The fused
Triton kernels are launched for the first time inside the capture pass
of the first capture in the process.

The warmup passes and the capture pass all write the live output
buffers `_output_codes` and `_output_embeds` and the predictor KV cache
for rows up to the bucket. The replay that follows in the same step
overwrites them, so this is harmless today but it means the capture
cannot move to a moment when those buffers are live for another batch.

### 1.2 What sglang does for its own graphs

- Captures every shape at startup, never inside serving. The one
  serving time recapture is a weight reload with a flag
  (`weight_updater.py:209-217`), with no cost handling beyond the usual
  begin and end log lines. A batch that needs more than the captured
  graph raises rather than recaptures
  (`decode_cuda_graph_runner.py:1233-1238`).
- Two warmup passes per shape, each after a device synchronize and a tp
  barrier, with the comment "Two warmups so kernels are loaded and
  one-time setup is paid before capture"
  (`runner_backend/full_cuda_graph_backend.py:103-108`). No rationale
  for the count. PyTorch's training helper uses three
  (`graphs.py:494`, "hopefully prevents cudnn benchmarking and other
  lazy-initialization cuda work from ending up in any captures",
  :644).
- The same `torch.cuda.graph` context as ours, so it pays the device
  synchronize, the allocator flush and the pinned host cache flush per
  shape too, at startup where nothing waits
  (`full_cuda_graph_backend.py:126-129`). Default `capture_error_mode`
  "global" (`graphs.py:415`), ours is "thread_local".
- One capture stream for every shape, taken from the
  `graph_capture` context (`parallel_state.py:586-608`), and one
  process wide memory pool (`runner_utils/pool.py:34-40`). Shapes are
  captured from the largest down: "Capture the large shapes first so
  that the smaller shapes can reuse the memory pool allocated for the
  large shapes" (`decode_cuda_graph_runner.py:1033-1035`, :1065-1069).
- Garbage collection is frozen across the whole capture loop
  (`base_cuda_graph_runner.py:45-61`), and `torch.cuda.graph` itself no
  longer collects on entry unless `TORCH_CUDAGRAPH_GC` is set
  (`torch/compiler/config.py:182`).
- A one time kernel warmup and autotune runs before any capture
  (`base_runner.py:229-260`), for flashinfer workspaces and autotuning,
  not a per shape model warmup.

PyTorch's own note on pool sharing (`graphs.py:398-399`): "if you pass a
pool used by a previous capture and the previous capture used an
explicit stream argument, you should pass the same stream argument to
this capture". Our code takes two new streams per capture. PyTorch hands
streams out from a fixed per device pool, so the stream objects recycle,
but consecutive captures do not share one stream the way sglang's do.

### 1.3 Where the default sampling signature comes from

A request's subtalker sampling fields are the qwen_tts wrapper's hard
defaults overlaid with the checkpoint's `generation_config.json`, merged
at `request_builders.py:1067`, then overridden per request by the
`subtalker_*` fields the API exposes (`request_builders.py:52-63`). For
Qwen3-TTS-12Hz-1.7B-Base the file on the Hub has `subtalker_dosample`
true, `subtalker_top_k` 50 and `subtalker_top_p` 1.0, and the ladder
quantizes 50 to 50, so every default request lands on the key
`(bucket, "sampled", 50, False, False)`, which is the only signature the
run 2 log shows. Any startup capture must derive its signature from that
merge, not from constants.

## 2. Evidence from run 2

The twelve captures, host wall from the ledger's `host_ms` max of the
step that paid each (doc 17 section 4, corrected table):

| window | captures | host wall per capture |
|---|---|---|
| c1 | bucket 1, the first in the process | 1089 ms, of which the step's own prefill forward was 232 ms |
| c16 | 16 after the 15 row prefill burst, then 12, 8, 4, 2 | 915, 565, 531, 503, 545 ms |
| c32 | 32, 24 | 497, 523 ms |
| c64 | 64, 56, 48, 40 | 512, 513, 506, 496 ms |

Each capture step records about 4400 tensor allocations against about
50 for a normal step, which is three passes of the chain at about 1470
each. The chain's device time is milliseconds at every bucket.

The preprocessing stage (reference audio encoding) runs in the same
process as the tts engine (both pid 2100925 in the run 2 artifacts),
on its own thread and stream, so the device wide synchronize inside
`torch.cuda.graph` also waits for its work.

## 3. Hypotheses and the measurement that decides each

The measurement is the capture phase timing on perf/step-ledger: one
log line per capture with the host wall and allocator deltas of each
phase (warmup pass 1, warmup pass 2, the wait for the warmup work, the
wait for the rest of the device, the allocator flush, the context
manager entry, the capture pass, the context manager exit), two device
side figures from timing events (the serving stream's pending work when
the capture began, the device time of the two warmup passes), and the
timing's own cost. The first replay of every key logs its launch wall.
Nothing else in the capture changed, the two idempotent calls that the
context manager makes on entry are made once more, first, so that each
is timed alone.

- H1, the flat 500 ms is host work: three passes of a chain of roughly
  2500 PyTorch ops, plus instantiate. Decided by `warmup1`, `warmup2`
  and `capture_pass` against the total, on any capture after the first.
  If those three are the bulk, the cost scales with passes and ops and
  the warmups are two thirds of it.
- H2, the first capture pays the fused kernels' first launch inside the
  capture. Decided by `capture_pass` on ordinal 1 against ordinal 2 or
  later. The code reading already says the warmups do not exercise the
  fused branches, the number says what that costs.
- H3, the extra on the bucket 16 capture is the device drain and the
  allocator flush after a prefill burst in a process shared with
  preprocessing. Decided by `device_drain`, `step_pending_gpu`,
  `empty_cache` with its `frees` and `MiB_released` on that capture in
  a c16 window.
- H4, pool sharing across captures is ineffective with a new stream per
  capture. Decided by `mallocs` inside `capture_pass` on ordinal 2 and
  later: a shared pool serves later captures from its cached blocks with
  no device allocation.
- H5, the first replay of an instantiated graph carries an upload cost.
  Decided by the first replay line per key.

The warmups' device time `warmup_gpu` and the eager chain's device time
are the same number, so the log also gives the chain's device cost per
bucket without a trace.

## 4. Design options, decided on section 5

- A. Fix the capture in place. Replace the three capture state gates
  with an explicit graph path flag threaded through the chain, so the
  kernel set the capture records can be run outside capture. Run one
  process wide warm pass of that kernel set before the first capture.
  Capture each bucket with no per bucket warmup, since the state a
  warmup exists for is process wide (module loads, cuBLAS handle and
  workspace, Triton compilation) and per shape state (cuBLAS heuristics)
  is host side and legal during capture. Reuse one capture stream and
  the shared pool across captures. Expected cost per capture after A:
  one pass plus instantiate. Validated by the existing bit match tests
  between graph and eager, which pin the sampling bits.
- B. For captures that still happen inside serving after A, skip the
  device wide synchronize and the allocator flush by opening the
  capture on the capture stream after it waits on the serving stream,
  through `CUDAGraph.capture_begin` and `capture_end` directly. Only if
  H3's phases still matter after A.
- C. Capture the ladder at startup for the merged default signature
  (section 1.3), largest bucket first, sharing the pool and the stream,
  as a policy on top of A. Only if the residual cost of a lazy capture
  after A still matters at a rung crossing. Other signatures keep the
  lazy path.
- D. Reduce the ops per pass of the chain itself (task T22). Orthogonal,
  and the lever on whatever flat cost remains, on the eager chain and on
  the capture alike.

Order: A first, because it is a refactor of code that exists and every
other option sits on it. B and C are decided on the residual, measured
the same way after A.

## 5. Gates

- Measurement: run 3 of doc 15 with the capture lines present in the
  Qwen3-TTS server log. The c16 window alone decides H1, H3 and H4, the
  c1 window decides H2 and H5.
- Correctness of A: `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py`
  on the box, in particular the graph against eager bit match cases.
  The graph path flag must not change what the eager path computes, so
  the eager sampler stays the reference.
- Performance of A, then of B or C: the same c16 window on a fresh
  server, the capture lines before and after, the ledger's `host_ms`
  max at the rung rows, then the Qwen3-TTS benchmark at c1, c16 and c32
  with a fresh server per point.
- Memory: `reserved_bytes` before and after each capture from the log,
  and the pool growth across the ladder when C is on.

## 6. What run 3 must return for this document

The Qwen3-TTS server log with the capture and first replay lines, the
ledger JSON of the c1 and c16 windows, and `/model_info` per window, per
the runbook in doc 15 section 3.2.

## 7. Run 3 results (d6425827b, GPU 3, fresh server per point)

Seven captures, host wall per phase in ms, from the capture lines of
the two server logs. Ordinal is the capture's number in its process.

| window | bucket | ordinal | total | warmup 1 | warmup 2 | drain | flush | enter | capture pass | exit |
|---|---|---|---|---|---|---|---|---|---|---|
| c1 | 1 | 1 | 683.5 | 601.4 | 34.0 | 0.4 | 0.9 | 0.2 | 38.8 | 5.5 |
| c16 | 8 | 1 | 651.6 | 569.1 | 35.2 | 0.0 | 1.6 | 0.2 | 38.3 | 5.3 |
| c16 | 16 | 2 | 477.6 | 396.2 | 33.3 | 0.1 | 0.5 | 0.2 | 31.6 | 13.5 |
| c16 | 12 | 3 | 591.9 | 499.5 | 36.8 | 0.0 | 0.4 | 0.1 | 33.6 | 19.2 |
| c16 | 2 | 4 | 483.9 | 389.0 | 34.0 | 0.0 | 24.2 | 0.2 | 29.0 | 5.5 |
| c16 | 1 | 5 | 467.6 | 401.6 | 32.1 | 0.0 | 0.2 | 0.1 | 28.3 | 2.9 |
| c16 | 4 | 6 | 600.3 | 470.8 | 32.8 | 0.0 | 46.3 | 0.2 | 28.2 | 20.0 |

The allocator deltas on every line: warmup 1 makes 1422 tensor
allocations and 2 device allocations, warmup 2 makes 1421 and none, the
capture pass makes 1022 and 2. The flush freed 2 to 906 MiB in 1 to 26
device frees. The two device timings: the serving stream had at most
0.1 ms of work pending when the capture began, and the device timeline
of the two warmup passes equals their host wall to within a millisecond
with a zero wait for the warmup work afterwards, so the device was idle
waiting on the host throughout. First replay launches were 0.5 to 0.7
ms. The instrument cost 1.9 to 2.3 ms per capture.

Verdicts on section 3:

- H1 held in kind and failed in shape. The cost is host work, but it is
  not three equal passes. The second warmup pass and the capture pass
  cost 28 to 39 ms each, which is the chain's launch cost. The first
  warmup pass costs 389 to 601 ms on the same 1422 allocations, twelve
  to seventeen times the second pass, on every capture, on every bucket
  and in every order. Whatever it pays is paid once per new bucket, or
  once per new stream, and is not the allocator: two device allocations
  cannot cost 400 ms.
- H2 held in a small way. The first capture in each process pays about
  150 to 200 ms more in warmup 1 and about 8 ms more in the capture
  pass, so the fused kernels' first launch inside the capture is cheap
  on this box and the first process wide cost sits in the eager pass.
- H3 failed. The device drain is at most 0.4 ms and the flush at most
  46 ms, so the synchronize and empty cache of `torch.cuda.graph` are
  not the stall.
- H4 is consistent with a per stream cost. Every capture pass makes two
  device allocations into the shared pool although the pool already
  holds the previous captures' blocks, which is what a fresh capture
  stream needing its own cuBLAS workspace would do. The sizes are not
  logged.
- H5 failed. The first replay is under a millisecond.

Two further facts from the ledger of the same windows. The first prefill
of each process cost far more than its capture: c1 extend rows 1 host
1476 ms against a 683 ms capture and a 30 ms forward, c16 extend rows 8
host 1681 ms against a 652 ms capture and a 31 ms forward, so 800 to
1000 ms of first request cost sits outside the predictor capture and is
unattributed. And the accelerator test module on d6425827b passed 54 of
55, with `test_mixed_padded_bucket_bit_identity_and_reuse` failing at
`capture_end` with `cudaErrorStreamCaptureInvalidated` after 28 earlier
tests in the same process, and passing alone. The only work the timing
adds inside a capture is two reads of the allocator statistics, whose
binding calls `getDeviceStats` and no CUDA API
(`torch/csrc/cuda/Module.cpp:579`), so the failure is either order
dependent state that the base commit also has or something not yet
understood. It is decided by running the same module on cec7b6b11 in
the same order.

What the first warmup pass pays is the open question, and the reading
so far rules out the obvious candidates: the eager sampler is omni's
own small k Triton kernel (`sampling_kernels.py:450-495`, entry
conditions met for every bucket), not sglang's compiled
`multinomial_with_seed`, and its launch arguments are constant across
buckets, so there is no per bucket compilation on the path. The
attention is `scaled_dot_product_attention` on flash with no plan
cache. No compiled function is on the chain (`torch.compile` appears
in the engine only for the vocoder layers, which live in the vocoder
process). The candidates left are per stream first use costs inside
CUDA or PyTorch, and they are read off a trace, not off the code.

## 8. Run 4

One c16 window with the torch profiler on, through the existing
`/start_profile` route with `enable_torch` true and a
`trace_path_template`, which records every stage process continuously
between start and stop (`profiler/torch_profiler.py:111-120`, CPU and
CUDA activities, no stack). The capture lines stay in the log and the
ledger still writes, since the route also starts the event recorder.
The trace answers two questions at once: which ops carry the first
warmup pass of each capture, against the same ops in the second pass,
and what the first prefill of the process pays outside the capture.
Alongside it, the predictor test module on cec7b6b11 in the same order,
for the failure above.

The design options of section 4 stand, with one change already
decided by the numbers: the per bucket warmup passes are the cost, the
flush and the drain are not, so option B is dropped and option A's
first step is to find what the first pass initialises and move it out
of the serving step.
