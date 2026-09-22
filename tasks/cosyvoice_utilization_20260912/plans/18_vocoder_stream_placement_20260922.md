# Plan 18: the vocoder owns the stream (2026-09-22)

Follows readout 10 and PR #2300 (`exp/cosyvoice-vocoder-stream` 87621ea6a), where the streaming
scheduler owns a `torch.cuda.Stream` and runs its step pump and warmup under it. The buffered
path (`decode_batch`, `FlowCudaGraphRunner.run`, `mel2wav_batch`) still runs on the default
stream, which the AR's forwards use as well (`engine_builder.py:120` disables the overlap
schedule, so `run_batch` never enters `forward_stream_ctx`).

## What is wrong with the placement in #2300

1. The buffered path pays the same serialization streaming paid: its Flow replays and HiFT
   calls queue behind the AR's decode replays on one stream, and the AR's behind them. #2300
   removed that for streaming only.
2. After #2300 the warmup runs on the step stream, so the buffered path's first request after
   boot builds the default stream's memory pool and cuBLAS workspaces itself (readout 10
   finding 1, now on the other path). A first request cost, not a throughput one.
3. Two allocator pools in a deployment that serves both kinds, each caching its own peak.
4. The stream is a property of the vocoder's GPU work, not of one scheduler; the scheduler
   should not own a CUDA object it only passes through.

## Design

`CosyVoice3Vocoder` owns the stream and the context:

- `__init__`: `device = next(self.flow.parameters()).device`; `self.stream` is a
  `torch.cuda.Stream(device=device)` on CUDA, `None` elsewhere (the torch MPS path uses this
  class, `torch.mps` has no `Stream`). Right after creation `self.stream.wait_stream(
  torch.cuda.current_stream(device))`: everything submitted on the default stream before
  (weight loads, the precast, the flow graph capture that the default stream already waits
  on) is ordered before the vocoder stream's first kernel. PyTorch streams are non blocking
  with respect to the legacy default stream, so nothing orders them otherwise.
- `stream_context()`: `torch.cuda.stream(self.stream)` or `contextlib.nullcontext()`.
- The two entry points enter it: `decode_batch` (buffered, and through it `mel2wav_batch`),
  and the streaming scheduler's `pump_one_step` and `warmup_now` (through it `hop_batch`,
  `leftover_batch`, `hift_delta`, and the fallback's `token2wav`). The scheduler's own
  `step_stream` and `step_context` go away.
- `FlowCudaGraphRunner.run` and the TRT estimator need no change: a captured graph replays on
  the current stream, and `execute_flow_estimator` waits on `torch.cuda.current_stream` and
  hands back to it (`flow_estimator_trt.py:274-297`).

Not in scope: stream priority (no measurement), the MLX scheduler (its own class).

## Gates, on the branch `slice/cosyvoice-9-1-vocoder-stream` (worktree `cosy-9-1`), A = 87621ea6a

1. Buffered identity: seeded buffered c1, 64 requests, A against B with a second A boot as
   control (`g1_compare_audio.py --control`). Expected byte identical: the same kernels on
   another stream.
2. Buffered c16, 1,088 requests, two pairs with the cards swapped. This is the claim.
3. Streaming c16, 1,088 requests, one pair: the same mechanism as #2300, expected unchanged.
4. If 1 to 3 pass: fold into #2300 (the user prefers one PR when the fix fits) as the commit
   that moves the stream into the vocoder, and update the PR body with the buffered table.
