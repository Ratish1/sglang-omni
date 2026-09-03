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

Anchors are main 15c4568bb for omni, v0.5.18 for sglang, 2.9.1 for
PyTorch.

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
                           torch/cuda/graphs.py:242 torch.cuda.synchronize()
                           :252 torch.cuda.empty_cache()
                           :258 capture_begin
                    :153 one pass of the chain, recorded
                           graphs.py:266 capture_end (instantiate)
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
  (`graphs.py:296`, "hopefully prevents cudnn benchmarking and other
  lazy-initialization cuda work from ending up in any captures",
  :422-424).
- The same `torch.cuda.graph` context as ours, so it pays the device
  synchronize and the allocator flush per shape too, at startup where
  nothing waits (`full_cuda_graph_backend.py:126-129`). Default
  `capture_error_mode` "global", ours is "thread_local".
- One capture stream for every shape, taken from the
  `graph_capture` context (`parallel_state.py:586-608`), and one
  process wide memory pool (`runner_utils/pool.py:34-40`). Shapes are
  captured from the largest down: "Capture the large shapes first so
  that the smaller shapes can reuse the memory pool allocated for the
  large shapes" (`decode_cuda_graph_runner.py:1033-1035`, :1065-1069).
- Garbage collection is frozen across the whole capture loop
  (`base_cuda_graph_runner.py:45-61`), and `torch.cuda.graph` itself no
  longer collects on entry unless `TORCH_CUDAGRAPH_GC` is set
  (`torch/compiler/config.py:117`).
- A one time kernel warmup and autotune runs before any capture
  (`base_runner.py:229-260`), for flashinfer workspaces and autotuning,
  not a per shape model warmup.

PyTorch's own note on pool sharing (`graphs.py:204-206`): "if you pass a
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
