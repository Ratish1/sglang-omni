# Qwen3-Omni 0.5.18 numerics, H100 run of 2026-08-28

Branch analysis/qwen3-omni-0518-numerics at 1424f34aa, tarball
qwen3-omni-0518-numerics-results-1424f34aa.tar.gz. Environment from
ab/determinism.txt: H100 80GB, torch 2.13.0+cu130, triton 3.7.1, sglang 0.5.18,
sgl_kernel 0.4.6.post1, flashinfer 0.6.17, deep_gemm 0.1.5.post3,
transformers 5.12.1, torchcodec 0.15.0+cu130.

## 1. Run validity

- check_install: the first attempt imported sglang_omni from
  /sgl-workspace/sglang-omni, the installed package. After pip install -e the
  checkout imports and HEAD is 1424f34a (check_install_final_console.log).
- Unit tests: 436 passed.
- smoke_logprobs: 'B' at -0.0, alternatives 'A' -11.75, 'Б' -12.25, then
  '<|im_end|>' at -0.0001.
- Kernel A/B, ab/determinism.txt: all 117 case tensors bitwise identical
  between two runs of the same stack.
- Kernel A/B, ab/backend_pairs.txt: fp8 dense deepgemm vs triton differ on
  1e-4 of elements, max_abs 0.0156 at M=4096 and bitwise at M=1. Audio SDPA vs
  FA3 varlen differ on 46 to 48 percent of elements, max_abs 0.002. Vision SDPA
  default vs flash and vs math differ on 40 to 49 percent, max_abs 0.002. SDPA
  default vs cudnn bitwise. silu AOT vs JIT bitwise. moe_fp8_cutlass vs
  moe_fp8_triton differ on 100 percent of elements with max_rel 324 to 1440:
  the two harness cases do not compute the same function, so this pair is not
  evidence (validation task 2).
- Step 4 first pass is invalid. stop_server killed the background subshell
  that wrapped cd and nohup, not the server. The next launch found port 31000
  busy, moved to port 58255 (serve_fp8_31000.log lines 358 and 359) and its
  thinker and audio encoder died of CUDA OOM (lines 557 and 650). All six arms
  of that pass hit the baseline server, which is why every c1 run reported
  0.48. The requalified pass with one log per arm is the valid one and is what
  the sections below use.
- The overrides took effect in the requalified pass. serve_fp8_attn_triton:
  attention_backend=triton, decode and prefill. serve_fp8_moe_triton: thinker
  moe_runner_backend=triton. serve_fp8_fp32_lm_head: enable_fp32_lm_head:
  true. serve_fp8_audio_graph_off: no audio layer CUDA graph capture line.
  serve_bf16_attn_triton: attention_backend=triton.
  serve_bf16_moe_flashinfer_cutlass: moe_runner_backend=flashinfer_cutlass.
  deepgemm_off: no log line names the fp8 GEMM backend in either state, the arm
  is confirmed only by its distinct outputs in section 3 (validation task 3).
  bf16 fp32_lm_head: no config print in the log, but 0 of 2000 answer margins
  in that arm lie on the 0.125 nat grid against 61 to 63 percent in every other
  bf16 arm, so the fp32 head was active.

## 2. The HTTP 500 failures

Every FP8 Video-AMME run has 17 of 50 requests fail with HTTP 500 after a full
generation (latency 1.7 to 2.8 s, no server traceback). At c1 the failing set
is exactly the first question of each of the 17 videos in the subset. At c16
it is still one question per video, whichever the scheduler reached first for
that video. Stage 8 on the disagg speech server fails 16 or 17 of 20 the same
way.

Server log pattern: a 500 follows a two-chunk prefill (#new-token 8192 with
#pending-token, then the remainder), a 200 follows a cached-prefix prefill
(#new-token a few hundred, #cached-token 14086). On the disagg thinker the KV
pool ran at token usage 0.96 so prefixes were evicted and nearly every request
took the two-chunk path. The 001-1 RTF sample succeeded in all ten runs because
its prefix was still cached from the c16 runs (#new-token 1, #cached-token
14216).

Cause. The thinker runs with chunked_prefill_size 8192 (stages.py:1034), so an
uncached 14.4k-token prompt is prefilled in two chunks. SGLang samples the
first chunk's row like any other and discards the token: the batch result
processor appends to output_ids only once inflight_middle_chunks reaches 0
(sglang batch_result_processor.py:265 and 309). _record_rollout_logprobs
appended the logprob of that row anyway, so the request carried N+1 logprobs
for N output tokens and _chat_non_stream returned 500 with detail
"backend returned token_logprobs length N+1 for completion_tokens=N"
(openai_api.py:764). The detail is not logged server side and the benchmark
client keeps only the status line, which is why no message was visible. The
/generate rollout path on main has the same defect for prompts longer than the
chunk size (openai_api.py:1209 check, same recording code).

Fix on the branch, not yet run on the H100:

- base.py: _is_middle_chunk_row reads data.req.inflight_middle_chunks, the same
  signal the thinker's stream output and SGLang use. _enable_sampler_logprobs
  requests no top-k for such rows and _record_rollout_logprobs records nothing
  for them. Tests in test_rollout_logprobs.py, 17 pass.
- h100_runs.sh: _launch_server starts every server with setsid and refuses to
  start while the port is in use, stop_server ends the process group and waits
  for the port to close, smoke_logprobs_chunked sends a 9000-token text prompt
  with logprobs and checks that the entry count equals completion_tokens,
  backend_lines also prints attention_backend, enable_fp32_lm_head and
  DeepGEMM lines.

Consequence for the reported tables: the FP8 arm accuracies are computed over
50 with 17 failures counted wrong, and which question fails changes with
scheduling at c16, so the accuracy and QPS differences between arms are
failure-set noise. Only the per-sample comparison on the 33 requests that
succeed in every run is valid, and that is what section 3 uses.

## 3. FP8 Video-AMME, the 33 shared successes

Within one arm the greedy path is deterministic across c1 and the three c16
runs: identical letters, token counts and margins to all printed digits, with
two exceptions. attn_triton gives different paths at c1 and c16 for 003-2
(107 vs 77 tokens), 005-3 (D vs B) and 002-2 (43 vs 39 tokens). The baseline
gives 28 vs 44 tokens for 003-3 and 32 vs 42 tokens for 010-2 between c1 and
c16_3.

Between arms 6 of 33 samples change their letter, each deterministically:

| sample | expected | baseline | deepgemm_off | moe_triton | attn_triton | audio_graph_off | fp32_lm_head |
|---|---|---|---|---|---|---|---|
| 003-2 | C | D | C | D | D | C | D |
| 005-3 | C | B | B | D | D or B | B | B |
| 006-3 | A | A | A | A | B | A | A |
| 008-2 | B | B | B | B | B | D | B |
| 018-3 | D | D | C | C | D | D | D |
| 002-2 | D | A | A | A | A | A | A |

fp32_lm_head changes no letter. Two of its samples take a different path to
the same letter (008-2 41 vs 77 tokens, 002-2 38 vs 107 tokens).

The answer letter is never the weak point. In every flipped case the margin at
the letter position is 1.9 to 6.9 nat. The divergence happens earlier in the
completion: 22 of the 33 baseline completions contain a step where the top-1
and top-2 logprobs are exactly equal (min_margin 0.000 at index 2 to 63), the
text diverges from that step (003-2: 77 tokens in the baseline, 64 with
deepgemm_off, 85 with audio_graph_off) and the letter follows the text.

The exact ties are a bf16 artefact. In the baseline run 84.5 percent of all
top-1 minus top-2 gaps are multiples of 0.125 nat and 2.1 percent are exactly
zero. With enable_fp32_lm_head no gap lies on the grid and none is exactly
zero, but 4.0 percent are still below 0.1 nat: the ties are genuine near-ties
of the model on these prompts, bf16 rounding of the logits only makes them
exact.

QPS at c16, mean of three runs, valid only as a relative number under 17
failures: baseline 0.728, deepgemm_off 0.700, moe_triton 0.726, attn_triton
0.645, audio_graph_off 0.655, fp32_lm_head 0.702.

## 4. bf16 MMSU, 2000 samples, no failures

| arm | c1 | c16 | c16 QPS |
|---|---|---|---|
| baseline | 0.7070 | 0.7065 | 63.4 |
| attn_triton | 0.7090 | 0.7050 | 60.2 |
| moe_flashinfer_cutlass | 0.7095 | 0.7075 | 33.2 |
| fp32_lm_head | 0.7085 | 0.7095 | 62.1 |

The answer letter is the first output token in every MMSU record
(answer_token_index 0), so answer_margin is the only margin that matters.

Flips against baseline c1 (predicted letter differs): baseline c16 22 (net
correct -1), attn_triton c1 39 (+4) and c16 45 (-4), moe_flashinfer_cutlass c1
48 (+5) and c16 50 (+1), fp32_lm_head c1 7 (+3) and c16 23 (+5). Every flip is
on a sample whose baseline margin is at most 1.875 nat, 7 or 8 per arm on an
exact tie of 0.000 and 28 to 40 per arm below 0.5 nat. 99 distinct samples
flip in at least one arm, 32 in three or more. fp32_lm_head at c1 flips only
the 7 exact ties, to margins of 0.005 to 0.115 nat.

Concurrency alone moves 22 samples on the same server (c1 vs c16), so a CI run
at c16 has a run-to-run band of about 0.005 around 0.707 from batch composition
before any software change. The 12 systematic CI flips after the bump are of
this kind: 13 to 16 samples sit below 0.1 nat in either state and the
distribution median is 6.2 nat in both.

moe_flashinfer_cutlass runs at half the QPS: the flashinfer autotuner logs
"No tuned config covers trtllm::fused_moe::gemm1" and falls back to tactic -1
for every shape.

## 5. Stage 8 RTF

Sample 001-1 alone at c1 with a cached prefix: rtf 0.69 to 0.93 over ten runs,
latency 1.63 to 1.71 s, audio 1.8 to 2.5 s. The same sample inside the c16
runs: rtf 4.9 to 5.9, latency 12.7 to 13.0 s. The CI value of 5.7 to 8.4 is
queueing at c16, not generation speed. The thinker text for 001-1 is "The
audio response is not available." (8 tokens) in every run, with 'The' at -1.34
against 'Looking' -2.21, so the talker gets 8 tokens to speak in every stack.

## 6. Reading

The stack is bitwise deterministic run to run at the kernel level and the
greedy path is deterministic within an arm at fixed concurrency. Every arm that
changes any kernel, and the change of concurrency alone, moves the same
population: samples whose greedy path passes through a top-2 gap below about
0.1 nat, most of them exact bf16 ties. The bump changed which way those ties
resolve, not the quality of any kernel, and net accuracy stayed inside the tie
band on both benchmarks, which matches the CI history. The gates 0.64 and
0.707 sit inside that band on these subsets.

## 7. Validation tasks

1. Rerun steps 2 to 7 of h100_runs.sh with the fix. Step 2 is now
   serve_fp8_colocated 31000, smoke_logprobs 31000, smoke_logprobs_chunked
   31000, stop_server 31000, and the chunked smoke must print prompt_tokens
   above 8192 and exit 0 before anything else runs. Expected: failed=0 in
   every run.
2. kernel_ab.py moe_fp8_cutlass vs moe_fp8_triton: compare both cases to a
   torch bf16 reference MoE on the same inputs before using the pair.
3. deepgemm_off: find or add the log line that states the fp8 GEMM backend
   the thinker uses, so the arm is confirmed from the log and not from its
   outputs.
4. /generate with return_logprob on a prompt above 8192 tokens: confirm the
   same fix covers the rollout path.
