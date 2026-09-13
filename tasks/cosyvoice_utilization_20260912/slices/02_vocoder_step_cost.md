# Slice 02: vocoder step cost

Base: branch `perf/cosyvoice3-stream-scheduler-liveness` at `9b407e5bc`, which is the scheduler
structure at `a8700d833` (upstream main 51e2f7ec2 plus slice 01 and the admission rule) plus the
six commits below. Owners: `sglang_omni/models/fun_cosyvoice3/stages.py`,
`sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py`,
`sglang_omni/models/fun_cosyvoice3/streaming.py`. Reference implementation read for numerics:
CosyVoice `cosyvoice/cli/model.py` (CosyVoice3Model, lines 397-450), `cosyvoice/flow/DiT/dit.py`
(lines 76-166), `cosyvoice/utils/mask.py` (lines 127-238), `cosyvoice/flow/flow_matching.py`
(lines 71-125, 196-227), `cosyvoice/hifigan/generator.py` (causal HiFT, lines 596-726),
`cosyvoice/transformer/convolution.py` (CausalConv1d).

## 1. Measured basis

The slice 01 A/B on the H100 (English corpus, 1088 requests, c16 streaming, readout
`readouts/01_b_stream_c16_readout_20260913.md`): B `a8700d833` fixes WER (6.82 to 1.38 percent),
failures (23 to 0), the tail (p99 43.6 to 13.4 s) and throughput (1.62 to 2.05 req/s, 7.40 to 9.72
audio s/s), and loses first audio (0.91 to 3.32 s) and continuity (C50 79.8 to 13.2). Continuity at
c16 is capacity bound: 16 real-time streams need 16 audio seconds per second and the vocoder
delivers 9.72, so any admission order only chooses who underruns.

The 16 request Nsight capture on `622bcd198` (the `a8700d833` tree plus NVTX) pins where the
capacity goes:

| item | host ms | GPU ms | notes |
|---|---:|---:|---|
| native singleton Flow call (every hop below batch 2, every final) | about 280 | 70 to 90 | launch bound, about 18,100 kernel launches |
| packed causal Flow, 14 rows | 547 | 454 | GPU bound |
| one hop alone versus one hop inside a batch of 16 | 325 versus 53 | | 855 ms for all 16 |
| final (native, one per request) | 324 | | 1,088 of them, about 354 s of the 530 s B run |
| HiFT, one row | 26 to 54 | 4.5 to 8.7 | 83 host syncs per call |
| syncs per Flow call | | | 12 |

The capacity that follows: about 3 audio seconds per second if every hop is a singleton, about 19
if every hop and every final is batched, against the 9.7 measured on a run where 1907 of about 3264
hops went through 192 batched calls and every final ran alone. Batching is the whole gap.

## 2. What the PR now does

1. **Batch every runnable hop regardless of its token window** (`ad181263b`,
   `streaming_vocoder.py:304`). The `(token_offset, hop_len)` step key is gone. The step is one
   ranked list, started streams by playback slack least first, then new streams by age, cut at the
   batch size, and the batched call reads each row's own window and offset rather than the head's
   (`_run_hop_batch`, `:358`). The Flow adapter already padded and masked per row, so the key was
   stricter than the call it gated.
2. **Batch finals through the packed non-streaming Flow adapter** (`4a8d6f8c6`). A stream whose AR
   is done and has no ready hop is kind FINAL when it has tokens and FALLBACK when it has none
   (`_step_kind`, `streaming_vocoder.py:74`). When the head is a final the step is every runnable
   final up to the batch size, one packed `generate_flow(streaming=False, finalize=True)` call
   through the new `CosyVoice3Vocoder.leftover_batch` (`stages.py:1446`), each row sliced at its own
   `token_offset` and finalized in HiFT per row (`_run_leftover_batch`, `streaming_vocoder.py:390`).
   The leftover lands in `state.leftover` and `decode_delta(is_final=True)` returns it. A FALLBACK
   stream runs alone so its fallback error aborts only itself.
3. **Run every hop through the packed causal Flow adapter** (`138c41980`). The native singleton path
   (`_run_one_causal_hop`, `_run_flow_hift`) is deleted, `first_hop_batch` is renamed `hop_batch`
   (`stages.py:1435`), and batch size one goes through the same packed causal call. `token2wav_chunk`
   (`stages.py:1385`) remains only for the full-decode fallback.
4. **Build the streaming DiT chunk mask without a host sync** (`33c8b2731`). `_patch_chunk_mask`
   (`stages.py:844`) installs unconditionally after load and now covers the `static_chunk_size > 0`
   branch, building the chunk grid on the device and filling empty rows with `masked_fill_` instead
   of the upstream `.item()` check.
5. **Keep the HiFT window, sine tables and f0 predictor dtype resident at load** (`c03d7c226`).
   `_keep_hift_constants_on_device` (`stages.py:908`) moves the STFT window and the sine generator
   tables to the device and casts the f0 predictor to float64 once, because they are plain
   attributes that `hift.to(device)` leaves behind.
6. **Allocate the HiFT causal conv cache on the device** (`9b407e5bc`). `_patch_causal_conv_cache`
   (`stages.py:889`) wraps `CausalConv1d.forward` so the zero cache is `x.new_zeros(...)` instead of
   a CPU tensor copied in.

## 3. What each change removes

| change | removed cost |
|---|---|
| 1 | the step key that split equal work across steps; the affordability gate and its step timer go with it, because in a batched step admitting a new stream costs the started streams one row of GPU time, not a step |
| 2 | the 1,088 serial native finals, about 354 s of the 530 s run, and the `LEFTOVER_FLOW_STREAMING` constant, whose measured note now lives on `leftover_batch` |
| 3 | the remaining native singleton calls, 280 ms host for 70 to 90 ms GPU each |
| 4 | 10 of the 12 `.item()` drains per Flow call; capture no longer needs the graph flag to get the sync-free mask |
| 5 | 4 of the 83 syncs per HiFT call |
| 6 | 78 of the 83 syncs per HiFT call |

## 4. Exactness and the E2 gate

Mixed offset rows in one packed call are mathematically identical to solo rows: packing is left
aligned, the mel mask is per row, the chunk mask is anchored at frame 0, the shared noise is indexed
from frame 0, and no reduction crosses the batch. They are not bit identical, because tiling changes
reduction order. A packed call at batch size one is the same math on the same shapes as the native
singleton call it replaces. The mask patch and the conv cache patch are exact by construction: the
same boolean mask and the same zero cache, built on the device.

Gate: experiment E2, `experiments/e2_flow_path_exactness.py`, run on the box against real weights
before the benchmark. It compares the native singleton hop against the packed batch size one, a
mixed offset batch against the same rows run solo, the native leftover against the packed
non-streaming final, and each patch against its original, reporting max absolute, max relative and
mean absolute difference of the mel and of the HiFT waveform. WER on the c16 run is the second gate.

## 5. Run gates

One c16 streaming run against A (upstream main `51e2f7ec2`) and the previous B (`a8700d833`):

- C50 at 90 or above.
- req/s, WER and first audio within 2 percent of the best of A and B `a8700d833`.
- Zero failures.
- E2 run and passed before the benchmark boot.

## 6. Unit tests

At `9b407e5bc`, in `tests/unit_test/fun_cosyvoice3/test_streaming.py`:

- `test_hops_of_different_token_windows_share_one_causal_flow_batch` pins that the step key is gone.
- `test_step_takes_started_streams_before_new_ones_up_to_the_batch_size` pins the ranking and the cut.
- `test_finals_share_one_non_streaming_flow_batch` pins the packed final call.
- `test_stream_without_tokens_fails_in_its_own_step` pins that a FALLBACK stream runs alone.
- `test_finals_rank_by_slack_with_the_other_started_streams` pins that a final is ranked, not special
  cased ahead of the hops.

`tests/unit_test/fun_cosyvoice3/test_streaming_replay.py` has
`test_recorded_c16_inbox_replays_in_order_and_completes`, which replays a recorded c16 inbox through
the scheduler and asserts order and completion.

## 7. Out of scope, follow-up PRs on main

- Streaming Flow CUDA graphs with lazy, workload derived capture shapes. It needs the per row length
  tensors, the noise slice and the prompt loop moved outside the captured region first.
- Batched HiFT, `hift_delta_batch` on branch `perf/cosyvoice3-vocoder-step-cost`, with exactness
  experiment E1.
- Preprocessing: the campplus provider, the thread pools and the finalize lock.
- The full history recompute per hop. The reference does the same, so it is faithful, not a defect.
