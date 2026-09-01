# Where the time goes on the Qwen3-Omni stage paths

Measured on the numerics3 run: the previous image (sglang 0.5.16, torch 2.11,
triton 3.6.0, flashinfer 0.6.14, called A below) and the current image
(sglang 0.5.18, torch 2.13, triton 3.7.1, flashinfer 0.6.17, called B), same
8xH100 host, same omni tree apart from the bump. The compare report and every
number here are in `artifacts/2026-08-30-numerics3-slim/numerics3-current/compare/report.md`
and the `*.stats.json` caches next to it. Kernel and host figures come from
torch profiler traces of one server per stage config over a fixed workload
(8 requests at c1 then 16 at c16), request figures from the request profiler
events, end to end figures from the benchmark result files.

Caveats on the data:

- The previous image's runs needed several retries across GPU pairs (logs
  named `*.oom.log`, `*.exit9.log`, gpu01 through gpu67). The stage 9 c16
  event run on A (events_stage9) shows the thinker at 7 s per request with a
  p95 of 17 s, which is a contended GPU, and is not used below. The stage 10
  runs on A are contended in the same way and stage 10 did not start on B
  (NCCL in the container), so stage 10 has no valid numbers here.
- The stage 8 talker trace on B was not written, so the talker kernel diff
  between images is missing. The talker on B is characterised from the voice
  clone run instead (same process, short prompts).
- The videomme-talker workload against the bf16 colocated server failed 20 of
  20 on both arms, so the talker MoE backend arm has voice clone numbers only.

## 1. What the A/B settles about the bump

| Question | Answer | Evidence |
|---|---|---|
| Stage 8 rtf_mean up 12 to 17 percent | One sample. 001-1 answers with 38 tokens and 10.5 s of audio on A and with 8 tokens and 2.0 s of audio on B, in three runs of three on each image (c1, c16 and the 20 sample run). Its rtf goes from 1.5 to 6.3 to 7.0 and contributes +0.27 to +0.30 of the rtf_mean difference, more than the whole difference. Every other sample has a lower or equal rtf on B. | report section 3, stage8_events, stage8_traces_c1, stage8_traces_c16, rtf decomposition lines |
| Stage 8 latency and throughput | Not regressed. 20 sample c16 run: latency mean 10.41 s on A, 10.16 s on B, qps 1.063 to 1.113. c1: 3.61 s to 3.64 s. | report section 3 |
| Stage 9 latency at c1 | Not regressed. 2.21 s to 2.27 s, per sample ratio median 0.997. | stage9_traces_c1 |
| CPU preprocessing | B is faster. Two step decode plus processor per video 0.728 s with the A libraries (torch 2.11, torchvision 0.26, torchcodec 0.11.1) and 0.696 s with the B libraries, in the same container. In the server, preprocess_start to preprocess_end 797 ms on A and 754 ms on B (stage 8, 20 requests). | report section 4, events_stage8 |
| Thinker kernels, bf16 (stage 8) | Slower per launch on B. `fused_moe_kernel` 28.7 to 33.0 us per launch (+15 percent, 98k launches each side), FA3 decode kernel 17.8 to 18.9 us (+6 percent), fused_add_rmsnorm 3.0 to 3.3 us, act_and_mul 2.9 to 3.4 us, qknorm 2.3 to 2.5 us (the last three are the sgl-kernel rebuild, renamed from an anonymous namespace to `sglang::`). B also replaced `moe_align_block_size_kernel` plus `count_and_sort_expert_tokens_kernel` (2.9 plus 1.4 us per layer step) with `_moe_align_small_numel_kernel` (1.7 us). Net thinker kernel time +17 percent over the window, of which about 450 ms is the fused MoE kernel. | report section 6, traces_stage8 thinker |
| Thinker kernels, FP8 (stage 9) | Flat per launch. Grouped GEMM 13.8 to 13.8 us, FA3 decode 16.9 to 16.8 us, DeepGEMM 8.1 to 8.3 and 9.4 to 9.3 us, per token group quant 1.7 to 1.7 us. Total +4.5 percent with 6 percent more launches (more output tokens on B). | traces_stage9 thinker |
| Triton MoE config | Both images fall back from their triton version directory to the triton_3_2_0 config for E=128,N=768 on H100. Only the triton version differs (3.6.0 on A, 3.7.1 on B), so the +15 percent on `fused_moe_kernel` is the triton compiler with an unchanged tuning. | report section 2, serve_bf16_disagg_31002.log |
| FlashInfer untuned talker MoE shape | Present on both images (autotuner with 12 configs on 0.6.14, 12 on 0.6.17, the `No tuned config covers trtllm::fused_moe` warning at M=14238 on A and M=14208 on B). Not new. | report section 2 |
| Talker MoE backend (omni policy picks flashinfer_cutlass for bf16) | No difference on the voice clone workload. Five runs each: qps median 6.87 versus 6.84, latency mean 2.196 versus 2.194 s, rtf 0.661 versus 0.675. | tts_vc_c16 versus tts_vc_talker_triton_c16 |
| Cold container | The first run on empty compiled kernel caches costs +2.9 s of mean latency on stage 8 (13.04 s, then 10.23 s on the same server, 10.16 s on a fresh server with warm caches). Stage 9 shows +0.3 s on the first run and a contended second run. CI starts every job cold. | cold_stage8_c16, warm_stage8_c16, cold_stage9_c16, warm_stage9_c16 |
| Stage 9 accuracy | Tie flips as established before: 007-1 (D on A, A on B), 008-2 and 011-3 the other way. | report section 3 answer flips |

The end to end move that CI reports on stage 8 is therefore one tie flip on
one short answer sample, and the CPU and GPU sides moved in opposite
directions by a few percent each with the CPU side winning at c16 because the
GPU is mostly idle (next section).

## 2. Where the time goes

### 2.1 GPU busy fraction and host time per stage process

From the trace stats. `wall per graph launch` is the GPU span divided by the
number of `cudaGraphLaunch` calls, a proxy for the decode step period of the
process. `gpu per graph launch` is kernel time divided by the same count.
`host blocked` is host thread time inside `cudaEventSynchronize`,
`cudaStreamSynchronize`, `cudaDeviceSynchronize` and `cudaMemcpyAsync`.

| process | image | workload | GPU span s | kernel s | busy | graph launches | wall per launch ms | gpu per launch ms | host blocked s | memcpyAsync calls, avg ms |
|---|---|---|---|---|---|---|---|---|---|---|
| talker_ar | A | stage 8 | 45.3 | 16.8 | 37 percent | 2549 | 17.8 | 6.6 | 25.8 | 47111, 0.02 |
| talker_ar | B | voice clone | 13.5 | 3.5 | 26 percent | 631 | 21.4 | 5.5 | 3.6 | 12043, 0.01 |
| thinker | A | stage 8 | 44.1 | 8.6 | 20 percent | 993 | 44.4 | 8.7 | 7.5 | 14051, 0.53 |
| thinker | B | stage 8 | 42.3 | 10.0 | 24 percent | 1002 | 42.2 | 10.0 | 8.6 | 14331, 0.60 |
| thinker | A | stage 9 | 36.0 | 8.9 | 25 percent | 1230 | 29.3 | 7.2 | 7.7 | 13203, 0.58 |
| thinker | B | stage 9 | 37.9 | 9.3 | 25 percent | 1304 | 29.1 | 7.1 | 7.9 | 13961, 0.56 |
| thinker | B | voice clone | 11.3 | 1.2 | 10 percent | 713 | 15.8 | 1.7 | 1.1 | 3426, 0.30 |
| code2wav | B | stage 8 | 43.4 | 3.1 | 7 percent | 406 | 107 | 7.6 | 2.1 | 1228, 1.68 |
| image_encoder | B | stage 8 | 38.3 | 2.2 | 6 percent | 0 | | | 2.0 | 612, 3.25 |

Talker on the voice clone run in detail: 671 `cudaEventSynchronize` calls
totalling 3.1 s (4.7 ms each), 631 `cudaGraphLaunch` calls totalling 1.2 s
(1.9 ms each on the host), 2176 `cudaStreamSynchronize` calls totalling 0.4 s.
The talker step period is 21 ms for 5.5 ms of GPU work. The old image's stage
8 talker is the same shape: 25.8 s of the 45 s window blocked, 47k small
memcpys.

Thinker on stage 8 in detail: 14331 `cudaMemcpyAsync` calls at 0.6 ms each on
the host, 8.6 s in a 42 s window. The GPU side shows `Memcpy HtoD (Pageable
-> Device)` 1347 launches at 120 us, so these are pageable copies, which block
the calling thread until the copy completes. Also `aten::zeros` 42 calls
totalling 1.4 s (34 ms each) and `aten::copy_` 1149 calls totalling 1.0 s.

Image encoder on stage 8: `Memcpy DtoH (Device -> Pageable)` 540 launches at
1.2 ms (22 per request), host `cudaMemcpyAsync` 612 calls at 3.2 ms, and
`cudaLaunchKernel` 3954 calls at 194 us each, which is a launch queue stalled
behind synchronous copies.

code2wav on stage 8: 396k kernel launches for 24 requests (16k per request)
at 7 percent busy, `cudaIpcCloseMemHandle` 32 calls at 25 ms each,
`cudaIpcOpenEventHandle` 2379 calls (one per talker to code2wav stream chunk).

### 2.2 Per request, from the events

Stage 8 at c16, 20 requests, B (A in brackets), medians in seconds:

| component | B | A |
|---|---|---|
| wait for the preprocessing process | 6.01 | 6.66 |
| preprocess | 0.74 | 0.79 |
| thinker total | 1.05 | 1.29 |
| thinker prefill | 0.18 | 0.17 |
| thinker decode | 0.34 | 0.57 |
| talker total | 2.72 | 2.77 |
| talker prefill | 0.09 | 0.09 |
| talker decode | 1.82 | 1.66 |
| code2wav | 0.02 | 0.02 |
| end to end median | 7.80 | 8.92 |

The end to end time at c16 is the preprocessing queue. The preprocessing
stage is one process that handles one request at a time at 0.75 to 1.0 s per
video request, so stage 8 throughput is bounded at about 1.3 requests per
second and stage 9 at about 1.0 (measured 1.02 to 1.11 on B), while the
thinker process is busy 24 percent of the time.

Transport between stages for one video request (hop means from the events,
stage 8, B): preprocessing to image_encoder 65 to 78 ms, image_encoder to
thinker 97 to 121 ms, image_encoder to talker_ar 27 to 33 ms, thinker to
talker_ar 45 stream chunks at 11 to 13 ms each, talker_ar to code2wav 150
stream chunks at 0.6 to 1.0 ms each. The image encoder process copies to
pageable host memory 22 times per request (the D2H copies above) and the
thinker and code2wav processes open CUDA IPC handles (403
`cudaIpcGetMemHandle` in the thinker, 2379 `cudaIpcOpenEventHandle` in
code2wav). Which hop uses which path, and what the 100 to 130 ms is, is stated in section 5.4 from the code.

Voice clone at c16, 51 requests, B, means: talker_ar total 2.26 s of which
0.88 s is `scheduler_queue_enter` to `scheduler_prefill_start` (p50 1.01 s,
p95 1.52 s) and 0.90 s decode, thinker total 0.37 s (prefill 0.041 s, decode
0.244 s for about 15 tokens, 16 ms per token), audio_encoder 0.063 s, code2wav
0.043 s, preprocessing 0.014 s. At c1 the same request takes 0.65 s end to
end. The c16 latency of 2.2 to 2.4 s is therefore about 0.9 s of talker
admission wait and about 0.4 s of slower decode steps under batching, on a GPU
that is 26 percent busy in the talker process and 10 percent busy in the
thinker process.

## 3. Problems worth addressing, by measured cost

Only items with a measurement above are listed. The seam column is where the
change would live. None of these is an sglang patch.

| # | Problem | Measured cost | Seam |
|---|---|---|---|
| 1 | Serial CPU preprocessing bounds video throughput | qps equals one over the per video preprocess time (0.75 to 1.0 s). GPU stages 24 percent busy at c16. The c16 queue is 6 to 13 s of a 8 to 15 s request. | `SimpleScheduler` serial path (simple_scheduler.py:227-241). `processes.preprocessing.num_replicas: N` is available today (config/schema.py:159-169, no `replica_devices` needed for a CPU process), untested for this stage, see 5.4 |
| 2 | Talker decode step is host bound | 21 ms per step for 5.5 ms of GPU work (voice clone, B), 17.8 ms for 6.6 ms (stage 8, A). Per step one `cudaEventSynchronize` of 4.7 ms and one `cudaGraphLaunch` of 1.9 ms host time. The talker is the longest stage of every speech request (2.7 s of 7.8 s on stage 8, 2.3 s of 2.4 s on voice clone). | `_event_loop_normal` (omni_scheduler.py:1731-1741) with `_resolve_host_token_ids` `event.synchronize()` per step (base.py:185) and no overlap between the host work of step N and the GPU work of step N+1, see 5.4 |
| 3 | Thinker decode step is host bound in the speech pipelines | Voice clone: 15.8 ms per step for 1.7 ms of GPU. Stage 8: 42 ms for 10 ms. 14k pageable `cudaMemcpyAsync` calls at 0.6 ms in the stage 8 window, 1.4 s of `aten::zeros` (34 ms each, 42 calls). | `lookahead_eligible` returns False for audio output requests (thinker_model_runner.py:428-435) so the async loop degrades to synchronous, the thinker never calls `_stage_token_ids` so two blocking device `.tolist()` reads run per step (output_processor.py:38, upstream :934), and the deepstack `torch.zeros` per prefill (thinker_model_runner.py:384-389), see 5.4 |
| 4 | Encoder outputs leave the GPU on their way to the thinker | 22 D2H copies of 1.2 ms per request in the image encoder, 100 to 130 ms hop to the thinker, 65 to 78 ms hop from preprocessing, on the same GPU in the colocated configs. About 0.2 s per video request against a 0.17 s thinker prefill. | The encoder output cache is written to pageable host memory with a blocking copy before the send (stages.py:577-583, stage_cache.py:45, cache constructed without `pin_memory` at stages.py:849-853), and preprocessing to encoder moves a float32 pixel tensor through shared memory in three copies, see 5.4 |
| 5 | Talker admission wait under concurrency | 0.9 s of a 2.3 s voice clone request at c16, p95 1.5 s. Cause not determined (validation task). | Not determined. The talker prefill runs as its own batch step in the normal loop and the thinker to talker stream is per token with two relay objects per token (5.4), a per step batch log on the talker decides between admission, KV budget and stream arrival |
| 6 | Cold container JIT inside the benchmark window | +2.9 s mean latency on the first stage 8 run of a fresh container (+28 percent). CI starts every job cold. | CI runner cache mount for SGLANG_CACHE_DIR, or a warmup request set before the timed window that covers the prefill shapes |
| 7 | bf16 triton fused MoE 15 percent slower per launch on triton 3.7.1 with the triton 3.2.0 tuning | 450 ms of the 42 s stage 8 window, 54 percent of the voice clone thinker's GPU time is this kernel. Invisible at c16 where the GPU idles, visible at c1 and in the thinker decode step. | a tuned E=128,N=768 config for triton_3_7_1 shipped through SGLANG_MOE_CONFIG_DIR (omni owns the env), produced with sglang's benchmark/kernels/fused_moe_triton tuner |
| 8 | code2wav launch pattern | 16k kernel launches per request at 7 percent busy, `cudaIpcCloseMemHandle` 25 ms times 32, pageable memcpys 1.7 ms times 51 per request. Under 70 ms per request in the events, so low priority. | code2wav CUDA graph coverage (4 exact graphs today) |

Measured and not worth work: the talker MoE backend choice (flashinfer_cutlass
and triton within noise), the CPU decode libraries (torchcodec 0.15 is not
slower), the FP8 thinker kernels (flat), the FlashInfer untuned shape and the
MoE and W8A8 config warnings (identical on both images).

## 4. Validation tasks

- The talker admission wait: which condition holds a voice clone request
  between `scheduler_queue_enter` and `scheduler_prefill_start` for 0.9 s at
  c16 (KV budget, running request cap, the thinker chunk arrival, or prefill
  and decode not being mixed in the talker step). The section 5 audit names
  the code, a run with the talker's decode log interval at 1 shows the batch
  composition per step.
- The talker step anatomy: which host work fills the 16 ms between the end of
  one graph and the launch of the next (sampling readback, chunk send to
  code2wav, next batch build). A trace with `with_stack` on the talker process
  for 30 steps answers it directly (`SGLANG_TORCH_PROFILER_WITH_STACK=1`).
- Whether the 34 ms `aten::zeros` per request in the thinker is the hidden
  capture buffer and whether it is needed per request.
- Stage 8 talker trace on the current image (the file was not written), to
  diff talker kernels between images.
- Stage 10 on the current image with `NCCL_NVLS_ENABLE=0`, as PR #1811 did in
  its container.
- The prefill FA3 kernel per launch (+7 percent on B) was measured on mixed
  chunk shapes and needs a fixed shape pair to be a claim.

## 5. What SGLang main does on these paths

Read from sgl-project/sglang main at e6a6492057 (2026-08-30, 683 commits past
the v0.5.18 pin) and from v0.5.18, bit for bit, by code audit agents whose
decisive claims were re-read in the files. Line numbers are main unless
marked.

### 5.1 Prefill CUDA graphs: full versus breakable

The full prefill graph backend has not left experimental status. It landed on
2026-07-06 in commit 3cbb7568bd "[Experimental] Full Cuda Graph Support for
Prefill (#27988)", and at main HEAD the resolved config still logs
`cuda_graph_config[prefill].backend='full' is experimental. Use breakable or
tc_piecewise for production workloads.` (arg_groups/cuda_graph_hook.py:456,
moved verbatim from server_args.py:4457 in v0.5.18). Its incompatibility
rule list is empty, with a comment that says so and points at the
experimental warning (cuda_graph_hook.py:309). The only model that
gets full as a prefill default is Inkling, through the per architecture
registry (cuda_graph_hook.py:400-423). The breakable backend became the CUDA
prefill default on 2026-07-01 (commit 0543246184, `default_prefill_backend()`
in cuda_graph_config.py:112-121). The server argument surface for CUDA graphs
is byte identical between v0.5.18 and main (server_args.py:1842-1919 versus
v0.5.18:1852-1932), no argument added, removed or renamed, no default changed.
The help text of `--cuda-graph-config` still says `(full is decode-only)`
(server_args.py:1845), which is stale since 3cbb7568bd.

What full prefill does (runner/prefill_cuda_graph_runner.py): one whole
forward graph per num_tokens bucket with a fixed request slot count
(`full_prefill_max_req`, default `chunked_prefill_size // 512`, so 16 slots
for 8192, cuda_graph_setup.py:369-377), replay pads the token count up to the
bucket and rejects when the padded count exceeds twice the real one
(`_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2`, :151, predicate :1097-1160),
rejects batches with more requests than slots (:1097), captures the
transformer body only and runs the LM head eagerly (:528-531), refreshes
attention metadata out of graph on a slot padded view (:1058-1075), and
captures chunked cached prefix variants (1, 2, 4, 8, 16 chunks) only for the
FlashAttention backend (`supports_full_cuda_graph_chunked_prefix` is True in
flashattention_backend.py:150 alone). Both full and breakable reject a batch
whose ForwardBatch carries `input_embeds` or `replace_embeds` (:1124-1127),
and both copy embeddings into a static `input_embeds` slot inside the patched
layer forward (:1733-1749). Post v0.5.18 additions on this runner: DP
attention coordination (58ecbba0bd, 2026-08-26), pipeline parallel support
(26fd7fdaa2, 2026-08-27), graph pool borrow scopes (37c09ff3d8), and an
assert that mixed chunk prefill requires the breakable backend
(prefill_cuda_graph_runner.py:266-271, commit ff5578eb4e, 2026-08-27).

What breakable does: one captured region becomes a sequence of
torch.cuda.CUDAGraph segments separated by eager break points
(runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:14-22,
216-269, replay at :281-290). The 12 break sites are the attention custom
ops (layers/radix_attention.py:577, 580, 655), linear attention, DSA, the
MoE all to all (layers/moe/ep_moe/layer.py:215) and a few model specific
ops, identical in v0.5.18. Capture buckets are the same token ladder as
full, without the request slot limit (cuda_graph_setup.py:379). The
multimodal architectures allowed to use it are a seven entry allowlist
(configs/model_config.py:1986-1994).

How omni uses it: the only place a prefill graph is enabled for Qwen3-Omni
is the thinker in examples/configs/qwen3_omni_colocated_h100_bf16.yaml:10-11
(`cuda_graph_backend_prefill: breakable`, `cuda_graph_max_bs_prefill: 2048`,
the TTS CI config). Higgs TTS, Fun-ASR, Qwen3-ASR and MOSS-Transcribe-Diarize
set breakable as a stage default in their engine builders. Every other stage
is forced to disabled in
sglang_omni/scheduling/sglang_backend/server_args_builder.py:67, and the
talker_ar, Qwen3-TTS, MOSS-TTS, Dots, Zonos2, Voxtral, FishAudio, CosyVoice3,
Audar, MiniMax, Ming architectures declare
`supports_breakable_prefill_cuda_graph = False`. Omni's policy validator
(sglang_omni/scheduling/generation_batch_policy.py:285-373) rejects any
prefill backend other than breakable or disabled, so full cannot be selected
through omni today. Omni has no prefill graph runner of its own. Its parts
are the `is_multimodal` capture view shim (scheduling/bootstrap.py:43-60), the
runner composed embeddings sidecar (model_runner/prefill_inputs.py,
sglang_model_runner.py:234-259) that keeps ForwardBatch.input_embeds None so
the replay predicate passes, the `enable_prefill_input_embeds` flag (omni
only, zero hits in sglang), and the `/model_info` prefill_cuda_graph payload
(model_runner/model_worker.py:328-355). The thinker stage enables
`enable_mixed_chunk` (models/qwen3_omni/stages.py:1034), which main's runner
asserts is breakable only.

What this means for the question of replacing breakable with full: the code
supports it for the thinker in principle (FlashAttention backend, embeddings
through the same slot, no incompatibility rules), and forbids it for the
thinker's mixed chunk setting at main. It is off the table by omni's own
validator until that is lifted. For the small models the prefill is not
where the time is: the talker prefill is 30 to 100 ms and the thinker prefill
41 ms of a 2.3 s voice clone request (section 2.2), against a decode step
period of 21 ms for 5.5 ms of GPU work.

### 5.2 torch.compile and launch overhead

torch.compile is off for the model body by default and the same in v0.5.18.
`enable_torch_compile` defaults to False with the help text "Optimize the
model with torch.compile. Experimental feature." (server_args.py:2047-2051),
`torch_compile_max_bs` is 32 (:2055-2057), and with the flag off the decode
graph runner's `compile_bs` is empty
(runner/base_cuda_graph_runner.py:96-100) so `patch_model` yields the plain
`model.forward` (compilation/torch_compile_decoration.py:64). The default
prefill backend, breakable, states "No torch.compile." in its module
docstring (runner_backend/breakable_cuda_graph_backend.py:14-17). The
tc_piecewise backend is the one that wraps the language model forward in
`torch.compile` (compilation/compile.py:217-221), with `tc_compiler="eager"`
by default, i.e. dynamo tracing with the piecewise backend and no inductor
(cuda_graph_config.py:96-97), and it is selected only off CUDA or for a
validated list of multimodal decoders (arg_groups/cuda_graph_hook.py:122-141).
A decode request for tc_piecewise falls back to full with a warning
(runner_backend/utils.py:94-103). What does run compiled by default is a set
of import time `@torch.compile` helpers on the MoE path: the grouped topk
post processing on CUDA (layers/moe/topk.py:1578, reached through
`select_experts` :2144) and `moe_sum_reduce_torch_compile` on the triton MoE
runner for `num_tokens <= 32` (moe_runner/triton_utils/fused_moe.py:124-125,
:366). These execute during graph capture and their kernels are recorded, so
replay runs no Python. The code's own reasons for keeping compile out of the
hot path: "torch.compile of the native TopK only pays off at bs=1 ... the
compiled path regressed bs > 1" (topk.py:582-584), the same note on the
unquantized MoE layer (quantization/unquant.py:883-887), and the assertion
that `--disable-cuda-graph-padding` cannot be combined with compile because
every distinct batch size would get its own compile and autotune cycle
(arg_groups/validation_hook.py:43-49).

Decode CUDA graphs capture the whole forward including the LM head
(runner/decode_cuda_graph_runner.py:1137-1152, capture body :1187-1226). The
bucket list is `[1, 2, 4, 8, 12]` then 16 to 256 in steps of 8, 272 to 512 in
steps of 16, then steps of 32 up to `max_bs`
(arg_groups/cuda_graph_hook.py:494-527), `max_bs` from a GPU memory tier
table (256 on an 80 GB card at tp below 4, arg_groups/memory_hook.py:111),
clamped to the request pool size and the attention tp alignment
(base_cuda_graph_runner.py:64-101). A batch pads up to the next bucket
(`_pad_to_bucket`, :132-150), padded rows are not zeroed
(decode_cuda_graph_runner.py:1372-1373), the output is sliced back
(:1493-1508). The replay predicate is `cuda_graph_bs <= self.max_bs` plus
the spec, TBO, ngram and encoder decoder gates (:676-749), and a batch above
`max_bs` runs eager (model_runner.py:1716-1787). The only device read in the
predicate, `torch.all(forward_batch.encoder_lens > 0)`, is reached for
encoder decoder models only (:726). The memory reserve for the graphs is
`max_bs * 2` MB (memory_hook.py:278-319). Attention metadata is prepared out
of graph per step by `init_forward_metadata_out_graph` and only the static
shape part is recorded (base_attn_backend.py:36-59), and main adds an
optional metadata glue graph that captures that per step host op sequence
into one graph launch (runner/metadata_glue_graph.py:82-86, off by default
under `SGLANG_ENABLE_METADATA_GLUE_GRAPH`, environ.py:1198).

Per step host to device traffic on the decode path is nil in
`ForwardBatch.init_new`: positions come from `clamp_position(batch.seq_lens)`
on the device (forward_batch_info.py:893), `num_token_non_padded` is built
only when `moe_ep_size > 1` (:1724-1725), the global token tensors only under
DP. Sampling parameters are copied once at batch construction
(sampling/sampling_batch_info.py:87-180, pinned and `non_blocking`), and
per step `filter_batch` reindexes with a device index tensor (:324-348).
The attention backends' decode metadata init issues no host copies: FA3
reads the CPU mirror of `seq_lens` and comments that it does so to avoid a
D2H sync (flashattention_backend.py:688-697), triton's only `.item()` calls
are the fallback when that mirror is absent (triton_backend.py:751-774), and
flashinfer copies into preallocated device buffers with `non_blocking`
(flashinfer_backend.py:250-257). Device to host is one batched pinned copy
per step on `copy_stream` with a `copy_done` event (managers/utils.py:31-41,
:155, :178), allocating a fresh pinned `torch.empty` per copied tensor.
Finished flags are never copied, finish state is derived on the host from the
token ids. The one per batch difference between the versions is that main
pins the prefill staging tensors that v0.5.18 builds pageable
(forward_batch_info.py:898-904 versus v0.5.18 :877-882,
mem_cache/allocation.py:304-321 versus v0.5.18 :304-314), and a
`non_blocking=True` copy from pageable memory is synchronous.

Streams and events. The steady state runs on three named streams, schedule,
forward and copy (scheduler.py:1561-1567, :1762), forked by
`forward_stream.wait_stream(schedule_stream)` (:3903) and joined by
`copy_stream.wait_stream(forward_stream)` (:3974), with a write after read
barrier event handed from the graph runner to the scheduler
(decode_cuda_graph_runner.py:528-538, scheduler.py:1783-1794). Side streams
come from a named process level registry, `get_stream(name)`
(runtime_context.py:881-899, "Creation is a driver call that must stay
outside cuda-graph capture"). The two function multi stream helper is enabled
only under CUDA graph capture, "because switch stream has extra host
overhead" (utils/multi_stream_utils.py), as are the K and V split write in
the KV pool (mem_cache/memory_pool.py:183-189, gated on `get_is_capture_mode`)
and the MegaMoE shared expert overlap (layers/moe/mega_moe.py:155-169). Where
events are created per step on a hot path the code pools them and documents
why: "do NOT create a fresh event per gather/combine ... the HSA signal pool
is exhausted after a few hundred forwards" (layers/dp_attention.py:911-924).
Main captures all graphs on one shared capture stream
(runner_utils/pool.py:42, :87-93), v0.5.18 takes a fresh stream per capture
pass (v0.5.18 decode_cuda_graph_runner.py:1037).

Two batch overlap is a single stream software pipeline that interleaves the
stages of two micro batches (batch_overlap/operations.py:38-71, no stream or
event in the module) and relies on the asynchrony of the collective ops, it
exists for four MoE decoder layer classes (operations_strategy.py:34), and it
requires an a2a backend or DP attention (validation_hook.py:408-425). Single
batch overlap adds an alternate stream and an SM split for the DeepEP low
latency combine (batch_overlap/single_batch_overlap.py:81-108). The prefill
delayer is a DP attention scheduling mechanism that holds prefill admission
so decode keeps running, negotiated across DP ranks (managers/prefill_delayer.py),
and requires the overlap scheduler (:144-146). None of these applies to a
single GPU dense or small MoE stage.

What omni does not use from this list: the overlap loop (5.4), the pinned
result copy for the thinker (5.4), and the bucket padded decode graphs are
used by every AR stage (the talker's 1.2 million kernels in 631 graph
launches). The talker's 1.9 ms `cudaGraphLaunch` host time per step is not
explained by anything above and is a validation task (section 4).

### 5.3 Scheduler loop and per step host work

For a single GPU, no speculative decoding, default arguments, the loop that
runs is `event_loop_overlap` (managers/scheduler.py:5281-5294 dispatch,
:1831-1907 body). Its per iteration order is: drain the tokenizer and rpc zmq
sockets non blocking (scheduler_components/request_receiver.py:118-134, two
`recv_pyobj(NOBLOCK)` calls minimum), `process_input_requests`,
`get_next_batch_to_run`, `run_batch` (launches step N+1 on `forward_stream`,
:1879), append `(batch.copy(), result)` to a queue (:1882), then
`pop_and_process` the previous step's result (:1888-1890). The three streams
are created at :1561-1567 and :1762 (`schedule_stream`, `forward_stream`,
`copy_stream`).

The host blocks in exactly one place per decode step:
`result.copy_done.synchronize()` in
scheduler_components/batch_result_processor.py:874-875 (prefill at :249-250),
and by the loop order it does so after step N+1's kernels are already queued.
The sampled tokens never come back to the host for the next step: after the
forward, `future_map.stash` writes them into a device buffer indexed by
request slot (managers/overlap_utils.py:555-557), and the next decode batch
gathers its `input_ids` from that buffer on the device
(overlap_utils.py:108-113). The device to host copy of the result runs on
`copy_stream` behind an event (`_async_d2h`, managers/utils.py:31-41, into
pinned memory with `non_blocking=True`) and is read with one `.tolist()` on
the pinned host tensor after the event wait (batch_result_processor.py:1026).
The sampler has no host sync on its default path (layers/sampler.py:157 and
:236-238, the `.cpu().tolist()` calls at :364 and :418-422 are behind
`return_sampling_mask`). Finish detection is host side by design
(schedule_batch.py:1676-1717, no tensor reads).

Per step host to device traffic on the decode path is nil for tokens.
`prepare_for_decode` adds one to `seq_lens` on the device and to the host
mirror (schedule_batch.py:3216-3228), `alloc_for_decode` is host list
slicing plus one `req_to_token_pool.write` (mem_cache/allocation.py:524-587,
allocator at mem_cache/allocator/token.py:53-63). On the prefill path
`ForwardBatch.init_new` builds `extend_seq_lens` and `extend_prefix_lens` as
pinned tensors copied with `non_blocking=True`
(forward_batch_info.py:888-906), and `alloc_for_extend` builds three pinned
index tensors (allocation.py:305-321). In v0.5.18 those same tensors are
pageable (v0.5.18 forward_batch_info.py:876-881, allocation.py:303-313), the
one substantive per batch difference between the two versions on this path.
The overlap mechanism, the relay buffer, the copy stream and the blocking
line are the same in v0.5.18 (managers/scheduler.py:1753-1825,
batch_result_processor.py:203 and :811).

Batch independent work every iteration: the two zmq drains, five nvtx
decorated entry points, `os.getenv` reads for `SGLANG_REQ_WAITING_TIMEOUT`,
`SGLANG_REQ_RUNNING_TIMEOUT`, `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`,
`SGLANG_PROFILE_V2`, `SGLANG_LOG_DECODE_GRAPH_KEY` and
`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY`, a `for req in batch.reqs` scan
in `_maybe_clear_mm_inputs` (scheduler.py:2477-2485) and in
`_record_step_counters` (:4256), a load snapshot every 15th decode step
(:826-844), and in main a few new per step calls that short circuit without
beam search or auxiliary outputs (`beam_coordinator.commit_decode`,
`strip_beam_tail`, `append_beam_tail`, `alloc_aux_to_lengths`).

The per request Python loop of `process_batch_result_decode`
(batch_result_processor.py:919-956) appends the token, checks finish
conditions and releases KV for finished requests, then one payload per step
goes to the detokenizer over a blocking `send_pyobj` on a PUSH socket
(output_streamer.py:215, io_struct.py:2462-2465).

### 5.4 The omni side

Read from sglang-omni main at 507cfaa0d. This is what the numbers in section
2 come from.

Which loop runs. `OmniScheduler.start` picks `_event_loop_async_decode` when
`enable_async_decode` is set, else `_event_loop_overlap`, else
`_event_loop_normal` (sglang_omni/scheduling/omni_scheduler.py:1731-1741).
The overlap loop raises `NotImplementedError` (:2369-2374). The Qwen3-Omni
thinker sets `enable_async_decode=True` (models/qwen3_omni/config.py:144),
the talker does not, and Qwen3-TTS does not
(models/qwen3_tts/engine_builder.py:127-132 against the default False at
omni_scheduler.py:197). So the talker and the TTS engine run the normal
loop: run_batch, execute, forward, sample, `_finalize`,
`_resolve_host_token_ids` with `event.synchronize()`
(sglang_omni/model_runner/base.py:185), `.tolist()`, stream output, upstream
`process_batch_result`, and only then the next iteration. The host waits for
step N to complete on the GPU before it does any of the post processing or
builds step N+1, which is the 21 ms step period for 5.5 ms of GPU work in
the talker trace and the 4.7 ms `cudaEventSynchronize` per step. The thinker's
async loop applies its lookahead only when the batch has at least
`async_decode_min_batch_size` requests, the batch is a decode batch and
`lookahead_eligible` holds (omni_scheduler.py:2574-2594), and the thinker's
`lookahead_eligible` returns False for any request that produces audio output
and for any logprob request
(sglang_omni/model_runner/thinker_model_runner.py:428-435). On the speech
pipelines the thinker therefore also runs synchronously.

Token readback. The base runner stages sampled ids into a pinned ping pong
buffer with a `non_blocking` copy and an event at sample time
(base.py:121-134, :147-180), so the later `.tolist()` is host only. The
talker (talker_model_runner.py:98, :116) and the TTS runner
(qwen3_tts/model_runner.py:292) call `_stage_token_ids`, the thinker never
does, so for the thinker `_resolve_host_token_ids` returns None
(base.py:183-187) and
`sglang_omni/scheduling/sglang_backend/output_processor.py:38` runs
`.tolist()` on the device tensor, a blocking pageable D2H, and the upstream
`process_batch_result` repeats it on the device tensor
(v0.5.18 batch_result_processor.py:934), since the substitution at
omni_scheduler.py:1476-1485 only helps runners that staged. Two blocking
device reads per thinker decode step. The thinker trace shows 14331
`cudaMemcpyAsync` calls at 0.6 ms in the stage 8 window and 3426 at 0.3 ms
in the voice clone window.

Other per step host work in the shared runner: `_apply_repetition_penalty`
builds three host lists per request and three pageable `torch.tensor(...,
device=device)` copies per step whenever any request has
`repetition_penalty != 1.0` (base.py:923-951), `_apply_codec_suppress_tokens`
(base.py:968-1005) and seed installation (base.py:820-830) do the same when
active. The talker's `prepare_decode_buffers` builds six host lists per
request including a set comprehension over `req.output_ids`, then a pinned
staging copy and up to two pageable H2D copies (talker.py:1082-1170), skipped
only when `_reuse_decode_buffers` sees the same requests in the same order
each one token longer (talker.py:1029-1049). The thinker prefill allocates
`torch.zeros(total_tokens, layers * hidden)` for the deepstack injection per
prefill (thinker_model_runner.py:384-389), which is the 34 ms `aten::zeros`
per request in the trace, and the request build thread does five `.cpu()`
calls and a 14k element `input_ids.tolist()` per request
(models/qwen3_omni/request_builders.py:541-550, :628).

Encoders and their cache. Both encoder stages write every output to a
`StageOutputCache(max_size=64, max_bytes=4 GiB, cache_device="cpu")`
constructed without `pin_memory` (stages.py:849-853, :929-933), so
`_detach_value` takes the `value.to(device="cpu")` branch
(sglang_omni/scheduling/stage_cache.py:45), a blocking pageable D2H of the
whole output, and the store happens inside `_batch_image_encoder_payloads`
before the results are returned and sent (stages.py:577-583). That is the 612
`cudaMemcpyAsync` calls at 3.2 ms host time in the image encoder trace, about
80 ms per request ahead of the hop to the thinker, and the reason the
`cudaLaunchKernel` calls in that process average 194 us. The class has a
pinned path (`_to_pinned_host`, stage_cache.py:22-27) that is not selected.
The audio encoder additionally does `(cu_seqlens[1:] - cu_seqlens[:-1]).tolist()`
per batch (audio_encoder.py:233) and `torch.equal` plus `lengths.tolist()`
(audio_encoder.py:58, :66).

Transport. Control messages go over zmq ipc sockets as msgpack
(pipeline/control_plane.py:47-55). Tensors go one of three ways
(comm/router.py:310-356): shared memory for CPU tensors with three copies
(`torch.cat` at comm/stage_io.py:579, the shm write at relay/shm.py:31, the
copy out at :132), a CUDA IPC slot pool with three copies across GPUs
(stage_io.py:579, relay/cuda_ipc.py:777, :931), or direct CUDA IPC with zero
copies when sender and receiver are different processes on the same single
GPU (router.py:52-59, handles through `ForkingPickler`, stage_io.py:707-712).
For a video request: preprocessing to image encoder carries the HF
processor's `pixel_values_videos` on CPU through shared memory, three copies
of a float32 tensor of `[num_patches, 1536]` (request_builders.py:250-255,
preprocessor.py:620-635), the measured 65 to 78 ms hop. Image encoder to
thinker is direct CUDA IPC when both are on GPU 0, so the measured 100 to
130 ms hop is the cache write above, not the transfer. Audio encoder to
thinker is forced onto the relay path even on the same GPU
(`disable_direct_cuda_ipc_payload=True`, models/qwen3_omni/config.py:122-124).
Thinker to talker sends, per generated token, one `[hidden]` tensor as the
data and a second `[hidden]` tensor in the metadata, and the metadata tensor
travels as its own relay object with its own acknowledgement
(request_builders.py:981-992, stage_io.py:521-533), the measured 11 to 13 ms
per chunk on the two GPU stage 8 topology. Talker to code2wav is one 128 byte
message per decode step, code2wav to the coordinator carries the waveform
inside the msgpack stream message after a pinned copy
(code2wav_scheduler.py:398-400, :472, utils/audio_payload.py:34-49).

Preprocessing. `SimpleScheduler(_preprocess)` with no `max_concurrency` and
no batch function (models/qwen3_omni/stages.py:823-826) takes the serial
path `_start_serial` (sglang_omni/scheduling/simple_scheduler.py:227-241),
where `loop.run_until_complete` (:177) holds the scheduler thread for the
whole request. Video decoding runs on the CPU through `qwen_vl_utils` with a
torchvision fallback (preprocessing/video.py:324-335). No decoded media is
cached anywhere under sglang_omni/preprocessing, every request re-fetches and
re-decodes, only the encoder output cache above is reused. Replication
exists: `processes.<name>.num_replicas` on `ProcessConfig`
(config/schema.py:159-169), round robin at admission
(pipeline/replicas.py:238-250), each replica its own process, and a CPU only
process needs no `replica_devices` (config/topology.py:188-192). No example
config replicates a preprocessing process.

CUDA graphs. Thinker decode graphs at batch sizes 1 to 64 (stages.py:1030),
talker decode graphs 1 to 32 (stages.py:1151) plus an omni owned predictor
graph captured lazily per batch size bucket (talker.py:1474-1535), audio
layer graphs at six token buckets (audio_layer_graph.py:18), code2wav exact
shape graphs at frames 10, 20, 30, 35 for batch 1 plus batched keys
(code2wav_scheduler.py:43-75), Qwen3-TTS decode graphs 1 to 32, a predictor
graph and initial vocoder graphs at batch 1, 2, 4, 8
(streaming_vocoder.py:84). The talker trace shows 1.2 million kernels in 631
graph launches, so the predictor graph is in use. Every shape miss falls back
to eager silently: `maybe_replay` in the audio graph (audio_layer_graph.py:228-241),
the predictor bucket miss (talker.py:1499-1519), and code2wav's `key_miss`
counter (code2wav_cuda_graph.py:653-668).
