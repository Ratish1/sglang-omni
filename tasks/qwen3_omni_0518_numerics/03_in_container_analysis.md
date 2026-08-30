# In-container analysis of the previous image versus the current image

The two full trees the protocol of 02_h100_clean_run.md section 5 produced are
751 MiB (current image) and 5.2 GiB (previous image). Almost all of that is
torch profiler traces, which do not need to leave the machine. This document
records what the calibration in PR #1811 and the logs of the clean run already
establish, then gives one command that turns the two trees into a report in
place, and what to read in that report.

## 1. What PR #1811 establishes

PR #1811 recalibrated every CI reference from five clean runs per stage on the
current image (sglang 0.5.18, torch 2.13.0, one H100 pair, cpuset pinned). The
per-run tables are in the PR's Qwen3-Omni calibration comment. Read against
the references that were on main, they separate real shifts from noise, with
one caveat: the main references were not all measured on that host. The voice
clone references come from PR #1021's 8xH100 calibration, so a cross-reference
delta on that stage compares hosts as well as images. The in-container A/B of
section 3 removes that confound.

| Stage | Metric | main reference | five runs on the current image | Reading |
|---|---|---|---|---|
| VideoMME talker (stage 8) | rtf_mean | 0.9986 | 1.155, 1.146, 1.079, 1.116, 1.123 (median 1.123) | Every run above the reference, 12 percent at the median. |
| VideoMME talker (stage 8) | latency_mean_s | 10.878 | 10.98, 10.64, 10.45, 10.22, 10.20 | Unchanged or better. |
| VideoMME talker (stage 8) | throughput_qps | 0.981 | 0.991 to 1.078 | Unchanged or better. |
| VideoAMME (stage 9) | accuracy | 0.64 | 62, 62, 62, 64, 64 percent | The one-sample tie flip of 02_h100_clean_run.md section 2 in three of five runs. |
| Voice clone (TTS speed) | throughput_qps | 8.921 | 8.277 to 9.223 (median 8.373) | Six percent below the reference at the median, reference from another host. |
| Voice clone (TTS speed) | latency_mean_s | 1.644 | 1.619 to 1.793 (median 1.749) | Same. |
| Voice clone (TTS speed) | rtf_mean | 0.5124 | 0.509 to 0.566 (median 0.544) | Same. |
| VideoAMME talker TP2 (stage 10) | latency_mean_s | 54.071 | 43.4 to 43.8 | Faster than the reference, stable within 1 percent. |
| VideoAMME talker TP2 (stage 10) | rtf_mean | 4.476 | 3.32 to 3.78 | Faster than the reference. |
| MMMU speed | latency_mean_s | 7.478 | 12.112, 7.043, 6.686, 5.706, 5.755 | Run 1 twice as slow as runs 2 to 5 and not rejected, the first stage run of the session. |
| MMSU | accuracy | 0.707 | 70.35 to 70.65 percent | Within the bf16 tie band. |

Stage 8 is the clearest case: rtf_mean moved while latency and throughput did
not. rtf is latency over generated audio seconds per request and rtf_mean
averages those ratios, so with the same latencies the mean moves only when the
audio durations or the order in which requests finish change. Sample 001-1
(eight tokens, about two seconds of audio, rtf 5.7 to 8.4 in every observed
run) contributes 0.28 to 0.42 of the mean on its own. The per-sample paired
table of the report (section 3) shows directly whether the audio durations or
the finishing order moved.

The voice clone stage runs the talker and code2wav with a short text prompt and
no video, so it isolates the talker path that stage 8 and stage 10 share. It
takes about six seconds per run, which makes it the cheapest place to repeat
an A/B many times.

## 2. What the clean run logs already show

The server logs of the clean run (numerics2, current image) contain, on the
stage 8 server (bf16, thinker on GPU 0, talker and code2wav on GPU 1):

- `Configured SGLang backend policy: arch=Qwen3OmniThinkerForCausalLM ... moe_runner_backend=auto`
  and `arch=Qwen3OmniTalker ... moe_runner_backend=flashinfer_cutlass`. The
  talker backend is chosen by sglang_omni/platforms/cuda.py, which is unchanged
  since the pre-bump commit 8f8b73d3c apart from a docstring.
- `flashinfer.jit: [Autotuner]: Loaded 26 configs from ~/.cache/sglang/flashinfer/autotune/0.6.17/sm90/...`
  at startup, then during the benchmark
  `No tuned config covers trtllm::fused_moe::gemm1 input_shapes=((14200, 1024), (128, 768, 1024), ...)` followed by `falling back`
  and the same for gemm2. Hidden size 1024 with 128 experts of intermediate
  size 768 is the talker MoE, and M=14200 is the full multimodal prompt, so the
  talker prefill of a video request runs FlashInfer's fused MoE on an untuned
  fallback configuration.
- `Config file not found at .../triton_utils/configs/triton_3_7_1/E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json. Fallback to triton version 3.2.0`
  and, for the `_down.json` file, `reusing the tuned up-projection config without TMA` for
  the bf16 thinker MoE on triton.

On the stage 9 server (FP8 colocated) the talker's eight dense FP8 GEMM shapes
(N=1024,K=768 up to N=6144,K=1024) log
`Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!`.

Checked against the two sglang tags: the E=128,N=768 H100 MoE config exists
only under triton_3_2_0 on both v0.5.16 and v0.5.18, the fallback search and
the separate down-projection lookup are in fused_moe_triton_config.py on both
tags, and neither tag ships a W8A8 block config for any of the eight talker
shapes (161 and 163 files, all DeepSeek shapes). So those warnings are not new
with the bump. What is version specific is the FlashInfer 0.6.17 autotuner on
the talker MoE. Whether its fallback configuration is slower than what
FlashInfer 0.5 ran on the previous image is a kernel-time question, answered
by the trace diff (report section 6) and priced by the talker backend arm
(section 4 below).

Compiled kernel caches also moved: sglang 0.5.18 puts triton, inductor,
FlashInfer and DeepGEMM caches under SGLANG_CACHE_DIR (default ~/.cache/sglang,
environ.py third_party_cache_defaults), sglang 0.5.16 used ~/.triton,
~/.cache/flashinfer and ~/.cache/deep_gemm. CI starts every job in a fresh
container, so whatever is compiled at first use inside the benchmark window is
paid on every CI run. The MMMU run 1 in PR #1811 is consistent with that but
does not prove it. The cold versus warm step of section 4 measures it.

## 3. One command in the container

On the current image, with the branch pulled and the two trees at OLD and NEW:

```bash
source tasks/qwen3_omni_0518_numerics/scripts/h100_runs.sh   # OMNI_ROOT, OUT=NEW, GPU set
capture_env                                    # versions, git head, GPU state, cache sizes into NEW/env
run_compare OLD NEW                            # writes NEW/compare/report.md and tables.json
make_slim_bundle OLD NEW /results/numerics3-slim.tar.gz
```

run_compare streams every trace once (a raw 7 GB trace takes a few minutes
per file, COMPARE_WORKERS=8 files in parallel) and caches the per-file
statistics as `<trace>.stats.json` next to the trace, so a second run is
immediate. The slim bundle holds both trees without traces and audio: the
report, the stats caches, the events, the result files, the logs and the env
capture. It is a few tens of MB.

If the previous image's container still exists, run `capture_env` there too
with OUT=OLD before bundling, so section 1 of the report has both package
lists.

## 4. Additional steps on the current image

Each is one server start and a few minutes of traffic.

```bash
run_tts_repeats        # voice clone stage, bf16 colocated, one GPU, five c16 runs, then stage 8 traffic
run_talker_moe_arm     # same server with --talker_ar.engine.moe_runner_backend triton
run_tts_events         # voice clone stage with request profiler events
run_tts_traces         # voice clone stage with torch traces
run_cold_warm          # stage 9 then stage 8 with the caches moved aside, then warm
```

run_talker_moe_arm prices the omni-owned backend choice: if the triton talker
MoE is as fast or faster than flashinfer_cutlass on this image, the policy in
sglang_omni/platforms/cuda.py is an omni-side fix that needs no sglang patch.
run_tts_repeats is its baseline on the same server. Both also run the stage 8
workload on the colocated server, so the talker backend is priced on the video
prompt shape as well.

run_cold_warm moves the cache directories aside (nothing is deleted, the
directories get an .aside_<timestamp> suffix), starts a server on empty caches
and runs the benchmark twice on it, then starts a fresh server on the caches the
first one filled and runs twice again. The first run of the first server is
what CI sees. The difference between it and the second run on the same server
is in-process warmup, the difference between the first server and the second
server is disk cache JIT.

The same steps on the previous image need only run_tts_repeats and
run_tts_events / run_tts_traces (the talker arm and the cold versus warm
question concern the current image). Then rerun run_compare.

## 5. Reading the report

Section 1 (inventory, versions): the package diff between the images and the
trace sizes. Both raw and gz present for a trace means gzip was still running
when the tree was tarred, the raw file is used.

Section 2 (logs): the backend policy lines, the config fallbacks, the
FlashInfer autotuner lines and the error counts for the same log name on both
sides, with a B only / A only flag. This is where a backend or kernel
selection difference between the images shows first. The decode and prefill
throughput lines from the sglang scheduler, bucketed by running requests and
prefill size, are a direct per-process comparison on the thinker-only servers
and a mixed one on the disaggregated speech server.

Section 3 (benchmarks): every result file, grouped by run family, with the
summary metrics side by side, then for each family shared on both sides the
per-sample paired deltas. For the talker stages the rtf_mean decomposition
line names the samples that carry the change and gives the total audio seconds
on both sides. A change carried by one or two samples with unchanged audio
seconds is a finishing-order effect. Audio seconds that differ is a talker
output change. A uniform latency ratio across samples is a compute change.
Answer flips list the samples whose correctness differs between the images.

Section 4 (preprocessing): the CPU A/B files from both trees with their library
versions and per-video times.

Section 5 (events): per stage interval means and percentiles on both sides,
sorted by the change weighted by count, the hops between stages, the per-request
component table (medians and sums) and the per-request rows with the largest
end to end change. The component whose sum moved by the same amount as the end
to end sum is where the time went.

Section 6 (traces): per stage process, kernel time, GPU span and busy
fraction, launch counts (cudaLaunchKernel versus cudaGraphLaunch is the CUDA
graph coverage), event categories, kernel families, the kernels whose total
time moved the most with launch counts and average duration on both sides,
kernels present on one side only, and the same by short name so that renamed
template instantiations still align. A kernel with the same launch count and a
higher average is a slower kernel. A kernel with a higher launch count is more
work or lost graph coverage. Kernels only on one side are a backend change.

## 6. Decision table

| Report finding | Meaning | Action |
|---|---|---|
| Section 3 stage 8: rtf change carried by 001-1 and audio seconds unchanged | Finishing order under serial preprocessing | Preprocessing replicas or the pooled rtf metric, an omni-side choice |
| Section 3: audio seconds differ on the same samples | Talker emits different codec lengths | Compare talker outputs per sample, then the talker kernels in section 6 |
| Section 6 talker: trtllm fused_moe kernels slower or only in B | FlashInfer 0.6.17 fused MoE on the talker | run_talker_moe_arm result decides the omni policy |
| Section 6 thinker: fused_moe_kernel same count, higher average | triton 3.7.1 with the 3.2.0 config | Tune a triton_3_7_1 config for E=128,N=768 (sglang benchmark script) and ship it through omni's SGLANG_MOE_CONFIG_DIR |
| Section 6: cudaLaunchKernel up, cudaGraphLaunch down on a stage | Lost CUDA graph coverage | Compare the capture lines in section 2 and the stage's graph settings |
| Section 5: preprocess interval up, everything else flat | CPU decode path (torchcodec, torchvision) | Section 4 confirms per library, then preprocessing replicas |
| Section 5: thinker.wait or hops up | Scheduling or transport, not kernels | Inspect the bootstrap.py change since 8f8b73d3c |
| run_cold_warm: first run of the first server slow, later runs flat | JIT inside the benchmark window | Warm the caches at startup or mount the cache dir in CI |

## 7. Validation tasks

- The previous image's logs must show the same MoE config fallback and W8A8
  warnings (section 2 of the report). If they do not, the previous image did
  not run those kernels through the same paths.
- The 5.2 GiB of the previous image's trace tree is either uncompressed traces
  or more trace events. The inventory and the event counts per stage process
  in section 6 decide which.
- The rtf_mean shift of stage 8 is either finishing order, audio length or
  compute. Section 3 decides.
- Whether FlashInfer 0.6.17's fallback configuration for the talker prefill is
  slower than the previous image's kernel. Section 6 and run_talker_moe_arm.
- Whether JIT compilation happens inside the benchmark window on a fresh
  container. run_cold_warm.
- The pairing of per-request event rows by admission order assumes the client
  submits samples in the same order on both images. The benchmark iterates the
  dataset in order under a semaphore, so this holds for c1 and for the first
  16 admissions at c16.
