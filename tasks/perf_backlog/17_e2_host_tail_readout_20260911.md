# 17. E2, the host tail of the Qwen3-TTS decode step, 2026-09-11

Main `80b5aaed7` (before S3, 1062 predictor kernels), H100, sglang 0.5.19, one server with
`SGLANG_TORCH_PROFILER_WITH_STACK=1`, windows of 12 requests at c1 and 192 at c16, read with
`perfkit.py hosttail` (corrected version, analysis branch `08dd949c5`) and `perfkit.py steps`.
Figures are per decode step, p50 unless stated, microseconds unless stated.

## 1. Where the step goes

| | c1, 1 row | c16, 16 rows |
| --- | ---: | ---: |
| step wall | 8.56 ms | 9.55 ms |
| backbone busy | 1.80 ms | 2.04 ms |
| predictor wall (busy) | 4.02 (3.35) ms | 4.27 (3.59) ms |
| idle inside the predictor replay | 0.67 ms | 0.68 ms |
| device idle in the step | 3.28 ms | 3.79 ms |
| host tail after the predictor replay | 1.40 ms | 2.05 ms |
| idle elsewhere, backbone end to predictor start | about 1.9 ms | about 1.7 ms |

The GPU is idle 38 percent of a c1 step and 40 percent of a full c16 step. At c16 the rows
below 16, the steps where a request finished or joined, are 60 percent of the window's steps
and idle 6 to 9 ms of a 12 to 15 ms step.

Two host bound windows per step:

- Between the backbone and the predictor. The host runs the eager layer 0 sampling
  (`sampler.py: top_k_top_p_min_p_sampling_from_probs_torch` 220 self, `sampler.py: forward`
  66, dynamo and triton launch wrappers about 240, `_apply_codec_suppress_tokens` 40,
  `sglang_model.py: replay` 57) and then the graph launch calls, `cuda/graphs.py: replay` 2.02
  ms self at c1 and 2.03 at c16 for the two replays of the step, the predictor's 1062 node
  graph being most of it. The GPU finishes the backbone before the predictor launch is
  submitted, and the in graph gaps add 0.67 ms.
- After the predictor replay, the tail, 1.40 ms at c1 and 2.05 ms at c16. Owners at c16: sglang
  52 percent (1.07 ms), omni 36 percent (0.73 ms), torch 5, python 6. At c1: sglang 61 percent
  (0.86 ms), omni 22 percent (0.30 ms).

## 2. The omni owned tail, c16, self time p50 (p90)

| frame | us |
| --- | ---: |
| `qwen3_tts/model_runner.py: _write_feedback_buffers` | 197 (203) |
| `qwen3_tts/model_runner.py: post_process_outputs` | 125 (158) |
| `qwen3_omni/talker_model_runner.py: _decode_row` | 47 (52) |
| `sglang_backend/output_processor.py: process` | 33 (37) |
| `model_runner/base.py: _finalize` | 31 (33) |
| `talker_model_runner.py: _append_decode_input_history` | 20 (23) |
| `qwen3_tts/sglang_model.py: prepare_decode_buffers` | 19 (293) |
| `omni_scheduler.py: _build_sched_output` | 14 (15) |
| `omni_scheduler.py: _emit_stream_output` | 12 (13) |
| `omni_scheduler.py: _make_batch_result` | 6 (7) |
| `omni_scheduler.py: stream_output` | 4 (41) |
| `request_builders.py: apply_sglang_qwen3_tts_result` | 0 (100) |

The doc 04 S4 candidates measure: the finish path copy 0 at p50 and 100 at p90 (it runs on the
steps with a finish), the per step walks 12, the decode buffer scan 19 at p50 and 293 at p90
(the restage when the batch changes, not the scan). Together they are under 3 percent of the
tail at p50. S4 as scoped in doc 04 is closed: nothing in it reaches the noise floor.

The sglang owned tail is the batch result processor (101, p90 381), MRoPE decode positions
(72), the graph buffer fill (59), forward batch init (54), the penalty bookkeeping (48 + 24 + 25
+ 13), decode allocation (44), the attention graph metadata (37), prepare_for_decode (34),
filter_batch (8, p90 171). None of it is patchable; some of it is switched by configuration
the model owns, which is a question for the next plan, not a slice.

## 3. What this ranks

1. The predictor graph's node count. Every node costs device busy, an in graph gap and about
   1.9 us of host launch time per step, and the launch time is what leaves the GPU idle between
   the backbone and the predictor. S3's 80 nodes: 0.076 ms busy, about 0.05 ms of gaps, about
   0.15 ms of launch, 0.28 ms of a 9.5 ms step, which is the 3.7 percent the A/B measured. The
   next fusions on the same path are doc 03 A3, qk norm plus rope plus the cache store as one
   kernel (160 nodes), and the split K down projection's reduce (80 nodes if the GEMM can run
   without split K). Both need their own E experiment for numerics before code.
2. The host work between the backbone and the predictor, about 0.6 ms of eager sampling and
   wrappers before the predictor launch can be issued. Capturing the layer 0 sampling into the
   replay, the way the sub step sampling already is, would let the predictor launch be issued
   right after the backbone launch and hide the 2 ms launch behind the backbone's 2 ms of
   device time. Its own plan: the layer 0 sampler is sglang's, the seam is the model runner.
3. The tail's omni half, about 0.5 ms addressable at c16 (`_write_feedback_buffers`,
   `post_process_outputs`, the per row history), half of it at best, 2 to 3 percent.
4. Overlapping the tail with the next forward, the one step lookahead every other AR model in
   the tree runs: up to the whole tail, 16 to 21 percent, held on the Qwen3-Omni first audio
   regression, its own plan.

## 4. One check owed before item 2 is planned

`perfkit.py timeline census_c1/tts_engine.pkl --rows 1` on the retained pickle: the host t0 of
the predictor `cudaGraphLaunch` against the device end of the backbone replay, to read the gap
directly rather than as the remainder of section 1.
