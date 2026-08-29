# Qwen3-Omni 0.5.18 numerics, clean H100 run of 2026-08-28 and root causes

Branch analysis/qwen3-omni-0518-numerics at 35f108381 (chunked-prefill logprob
fix in). Tarball numerics2.tar.gz. Same environment as 01_h100_results.md.
Every run is clean: 439 unit tests, both smoke checks (the chunked prompt had
9016 tokens, 2 logprob entries for 2 tokens), 24 FP8 Video-AMME runs, 8 bf16
MMSU runs, 3 stage 8 runs and 10 RTF runs with failed=0, one ready and one
stop line per server. Kernel A/B is unchanged from the first run: 117 case
tensors bitwise identical across two runs, moe_fp8_cutlass vs moe_fp8_triton
still not comparable (harness issue, validation task 3 of 01_h100_results.md).

CI artifacts used below: 3 pre-bump runs (32599828664, 32867559656,
32981649732, 6 stage 9 and 9 stage 8 attempts) and 10 post-bump runs
(33071447456 to 33147701232, 27 stage 9 and 23 stage 8 attempts). The two
oldest pre-bump runs have expired.

## 1. Results

FP8 Video-AMME, 50 samples, accuracy identical at c1 and in all three c16 runs
of each arm unless noted:

| arm | accuracy | c16 QPS |
|---|---|---|
| baseline | 0.64 | 1.00 to 1.04 |
| deepgemm_off | 0.68 | 0.96 to 1.02 |
| moe_triton | 0.66 | 1.04 to 1.07 |
| attn_triton | 0.66 (0.68 in one c16 run) | 0.96 to 1.02 |
| audio_graph_off | 0.66 | 1.01 to 1.04 |
| fp32_lm_head | 0.64 | 1.01 to 1.03 |

bf16 MMSU, 2000 samples: baseline 0.707 c1 and 0.708 c16, attn_triton 0.709
and 0.707, moe_flashinfer_cutlass 0.7095 and 0.707 (QPS 33 against 62),
fp32_lm_head 0.7085 and 0.7075.

Stage 8, 20 samples at c16, three runs: accuracy 0.65, 0.60, 0.60. rtf_mean
1.056, 1.040, 1.074. latency_mean 9.7 s. Sample 001-1 alone with a cached
prefix, ten runs: rtf 0.67 to 0.97, latency 1.69 to 1.83 s.

## 2. Stage 9 root cause

Per-sample bisect of the FP8 baseline against CI:

- The H100 baseline agrees with the CI post-bump majority on 49 of 50
  samples. The one difference is 008-2 (expected B): B in 4 of 4 H100 runs, D
  in 25 of 27 CI attempts. Its greedy path has an exact top-1 to top-2 tie at
  step 15. The H100 runs use the synchronous sampling path (logprobs disable
  the async path at thinker_model_runner.py:431), CI uses the async path.
- CI pre to post differs on exactly two samples: 007-1 (expected D) went from
  D in 5 of 6 attempts to A in 27 of 27, and 017-1 went from A to D, both
  wrong. 007-1 has an exact tie at step 40 of its H100 completion. That one
  sample is the whole 0.64 to 0.62 move.
- CI post attempts score 0.62 in 24 of 27 and 0.64 in 3. The three 0.64
  attempts are the attempts where 008-2 came out B (2) or 003-2 came out C
  (1), the only two samples that are not unanimous across attempts on one
  stack. Both have an exact tie in their H100 completion.
- 32 of the 50 baseline completions contain a step where the top-1 and top-2
  logprobs are exactly equal. The logits are bf16: sglang computes the lm_head
  as a bf16 matmul (logits_processor.py:745) and casts to float afterwards
  (logits_processor.py:888), so logits near 20 to 30 have a resolution of
  0.125 nat and gaps below that collapse to zero. With enable_fp32_lm_head
  (torch.mm with out_dtype float32, logits_processor.py:727) no gap is exactly
  zero, but the answer letters do not change (0.64 in 4 of 4 runs) and MMSU
  flips between c1 and c16 stay at 19 against 17 on the baseline, so the fp32
  head is not a fix.
- Every arm flips letters only on samples with such a tie: deepgemm_off 4
  (003-2, 017-1, 018-3, 020-1), moe_triton 4, attn_triton 4, audio_graph_off
  3, fp32_lm_head 0. The tie population, not any kernel, decides the score.
- Within an arm the letter is stable across c1 and three c16 runs on every
  sample except one attn_triton sample (005-3), and token counts move on 6
  samples for the baseline, 30 for attn_triton, 2 for fp32_lm_head. The Triton
  attention backend is the least stable under batch composition.

Conclusion: no kernel regression. The gate 0.64 equals the count the current
stack reaches only when two tie samples resolve favourably (3 of 27
attempts). It was calibrated on the pre-bump stack at zero margin: the
pre-bump attempts score 0.62 in 4 of 6 and 0.64 in 2.

MMSU is the same picture on 2000 samples: 117 answers have a gap below 0.5
nat, 16 are exact ties, concurrency alone flips 17 to 22 answers on the same
server, a kernel change flips up to 50, the net effect stays within 5 samples,
and one stack scores 0.7035 to 0.709 across CI attempts against a gate of
0.707.

Correction to 00_plan.md 2.5: stage 9 QPS does not measure the thinker. See
section 3.

## 3. Stage 8 root cause (rtf_mean)

Server log reconstruction of run 1 (all 16 requests submitted at t=0):

- The preprocessing stage is one process running one request at a time
  (SimpleScheduler._run_single, simple_scheduler.py). It decodes and
  processes one 14k-token video request every 0.63 to 0.86 s on the H100 box
  (torchcodec decode 0.20 to 0.40 s of that), and every 0.6 s on the CI runner
  (1.6 to 1.7 qps at stage 9). The thinker prefills a request within 0.5 s of
  its preprocessing and is otherwise idle, so at c16 the request latency is
  the preprocessing queue: 16 x 0.96 s = 15.4 s on the H100 (stage 9 c16
  median latency 15.5 s), 16 x 0.6 s on CI (stage 9 latency 8.2 to 9.1 s).
  Stage 9 QPS and latency are CPU preprocessing numbers on both machines.
- Sample 001-1 (video fFjv93ACGo8) was preprocessed 16th of 16 in run 1
  (slot at 11.3 s), then had a cached-prefix prefill of 132 tokens at 11.7 s,
  8 output tokens, the talker, and finished at 13.1 s. Alone with a cached
  prefix it takes 1.7 s. In CI it finished at 13.5 to 15.9 s in all 32
  attempts, with the run maximum at 13.6 to 16.3 s: the first request of the
  benchmark is processed last in every attempt on both machines.
- 001-1 answers "The audio response is not available." in 8 tokens on both
  stacks (38 tokens in 2 of 9 pre-bump attempts), which the talker turns into
  1.8 to 2.6 s of audio. Its per-request RTF is latency over audio, 5.7 to 8.4,
  and it contributes 0.28 to 0.42 to rtf_mean. rtf_mean without it is 0.83 to
  0.95 in CI and 0.77 to 0.79 on the H100.
- Pooled RTF (summed latency over summed audio, the audio-weighted mean of
  per-request RTF) is 0.775 to 0.861 pre-bump and 0.741 to 0.841 post-bump in
  CI, 0.68 to 0.71 on the H100. There is no speech generation slowdown across
  the bump on that measure.
- Stage 8 accuracy: the only sample whose correctness changes between
  attempts is 005-3 (expected C): C in 9 of 9 pre, C in 18 of 23 post (D 4,
  A 1), C then D then D on the H100 with an exact tie at step 5 or 22. The
  0.65 to 0.60 attempts are this sample. 003-1, named in 00_plan.md 2.2 as
  the flip, is wrong on both stacks (C pre, A or C post, expected B).

Conclusion: rtf_mean as a gate is a queue-position lottery on one 2 s sample
in front of a serial preprocessing stage, with a threshold (1.12) calibrated
on attempts where that sample happened to answer longer. Nothing in the
talker, code2wav or thinker changed speed across the bump on this benchmark.

## 4. Parked: gate and metric edits

The rtf_pooled metric, its threshold support, the stage 8 gate on it and the
accuracy gate margins are parked in patches/ci_gates_rtf_pooled.patch.gz
(gunzip, then git apply from the checkout root, stored compressed because
the whitespace hook strips patch context). They change what the gates measure, not the
server, and are held until the speed moves of section 5 are attributed.
Kept on the branch: h100_runs.sh backend_lines uses grep -a (three per-arm
readouts were empty because grep treated the server log as binary).

## 5. Component profiling protocol

The arms of sections 4.1 to 4.3 varied thinker kernels one at a time and
found no kernel that changes results beyond the tie band. Stage 8 and 9
latency is set by the serial CPU preprocessing stage and stage 10 adds the
FP8 TP=2 talker, and the image bump also moved torch 2.11.0 to 2.13.0,
torchvision 0.26 to 0.28 and torchcodec 0.11.1 to 0.15.0, the libraries that
stage runs. Measured moves across the bump in CI: stage 8 latency_mean 9.9 to
10.6 s against 10.2 to 11.2 s, stage 9 qps 1.58 to 1.75 against 1.53 to
1.70, stage 10 latency_mean 41.3 to 44.8 s against 43.2 to 45.8 s and
rtf_mean 3.04 to 3.42 against 3.26 to 3.81. The protocol below measures every
Qwen3-Omni component on the stage 8, 9 and 10 paths per request and per
kernel, on the current image first, then on the previous image with the same
commands, and attributes the moves by diff.

Components and where each shows up:

| component | process (stage events, trace file) | stages |
|---|---|---|
| video decode, frame resize, mel features (torchcodec, torchvision, transformers processor, CPU) | preprocessing | 8, 9, 10 |
| vision tower (transformers Qwen3OmniMoeVisionEncoder, SDPA) | image_encoder | 8, 9, 10 |
| audio tower (captured layer stack, FA3 varlen) | audio_encoder | 8, 9, 10 |
| thinker prefill and decode (attention, MoE, dense GEMM, norms, MRoPE, lm_head, sampler) | thinker | 8, 9, 10 |
| text detokenizer and stream assembly | decode | 8, 9, 10 |
| talker (bf16 flashinfer_cutlass MoE on stage 8, FP8 cutlass MoE plus Triton dense on stage 10) | talker_ar | 8, 10 |
| code2wav (4 CUDA graphs) | code2wav | 8, 10 |
| coordinator hops and IPC | coordinator, hop breakdown | all |

Step 1. CPU preprocessing A/B (scripts/preprocess_ab.py). Times, per video of
the stage 8 result file, the two heavy steps of the preprocessing stage with
the CI request settings (ensure_video_list_async, then the HF processor
call), median of 3 repeats after a warmup. With --full it also times the
real Qwen3OmniPreprocessor on the same payloads, to be checked against the
0.63 to 0.86 s per request seen in the server log. Runs in the server venv
and then in a CPU-only venv that differs in torch, torchvision and torchcodec
only:

    source tasks/qwen3_omni_0518_numerics/scripts/h100_runs.sh
    run_preprocess_ab
    make_old_cpu_venv && run_preprocess_ab_old

Outputs preprocess_ab_new.json and preprocess_ab_old.json (env versions,
per-video load_s, processor_s, total_s, summary).

Step 2. Per-request stage events (scripts/stage_events.py). The request
profiler writes request_admission, preprocess_start and preprocess_end,
encoder_start and encoder_end, scheduler_request_build_start and end,
stage_input_received, stage_complete and terminal_response per request
(profiler/event_recorder.py, routes /start_request_profile and
/stop_request_profile). The runs use the CI request shape (no logprobs):

    GPU=0,1 run_stage8_events
    run_stage9_events
    GPU=0,1 run_stage10_events

Outputs events_<stage>_stages.txt (python -m sglang_omni.profiler stage and
hop breakdown: per stage interval count, avg and p95, and the time spent on
each hop between stages) and events_<stage>_requests.txt (per request: submit
offset, preprocessing queue wait, preprocess duration, encoder durations,
thinker request build, total).

Step 3. Torch profiler traces per stage process (scripts/trace_kernels.py).
POST /start_profile with enable_torch true records CPU and CUDA activity in
every stage process between start and stop and writes
traces_<stage>/<process>_pid<pid>_rank<r>.trace.json.gz. The workload is
fixed: the first 8 samples at c1, then the first 16 at c16, both without
logprobs, with stage events recorded alongside:

    GPU=0,1 run_stage8_traces
    run_stage9_traces
    GPU=0,1 run_stage10_traces

Outputs traces_<stage>_kernels.txt: per process, total CUDA kernel time, time
per kernel family (attention, moe, gemm, norm, rope, sampling, copy, reduce,
other) and the top kernels by total time.

Step 4. Same three steps on the previous image. The pre-bump commit
8f8b73d3c has the profiler package and both profiler routes, and the tools
under tasks/ import nothing that changed, so on the old image check out
8f8b73d3c, copy tasks/qwen3_omni_0518_numerics/scripts into it, install the
checkout, and run the same commands into a second OUT. Then:

    python scripts/trace_kernels.py diff OLD/traces_stage8 NEW/traces_stage8
    python scripts/trace_kernels.py diff OLD/traces_stage9 NEW/traces_stage9
    python scripts/trace_kernels.py diff OLD/traces_stage10 NEW/traces_stage10

diff aligns the two runs by process and kernel name and prints the kernel
family and kernel totals that moved, per stage process. Together with the
stage breakdown diff (queue waits, preprocess, encoders, talker, code2wav
intervals) and the preprocessing A/B, this places the 3 to 10 percent moves
on a component and a kernel.

Step 5. Preprocessing replicas, once the above is attributed.
processes.<name>.num_replicas exists in the pipeline config
(config/schema.py:180), the coordinator binds requests round robin
(pipeline/replicas.py:239) and the preprocessing process has no GPU stage, so
it needs no replica_devices. Expected at c16: latency from 15 s toward 4 s
with identical per-sample letters.

Step 6. Field reports. The evidence here is CI plus the H100 arms. Reports of
quality or latency changes in real traffic are matched against these
measurements before any of them is called explained.
