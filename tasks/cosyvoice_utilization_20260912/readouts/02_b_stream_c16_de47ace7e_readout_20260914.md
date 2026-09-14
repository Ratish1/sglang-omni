# Readout 02: B stream c16 at de47ace7e (2026-09-14)

Archive `artifacts/cosyvoice-de47ace7e-20260914T082850Z.tar.gz`, one boot, English corpus,
1,088 requests, c16 streaming, warmup 1, unseeded, benchmark client `max_new_tokens=2048`.
Head de47ace7e is the tree that became af843811b after the rebase: upstream main 51e2f7ec2 plus the
batching commits, the stage order and pool ordering commits, and the mask `any` fix. No step bound.

## 1. Results against the previous B run

| run | failures | WER | first audio mean / p95 s | C50 | req/s | audio s/s | latency p90 / p99 s |
|---|---|---|---|---|---|---|---|
| B 9b407e5bc (OOM run) | 15 (3 OOM) | 1.33 | 2.09 / 2.81 | 71.5 | 3.954 | 18.4 | 5.2 / 29.6 |
| B de47ace7e | 0 | 1.43 | 2.66 / 6.66 | 52.8 | 3.039 | 15.07 | 9.4 / 16.9 |

OOM count in serve.log: 0. Boot: pool 62.1 GB, "Memory pool end" 11.13 GB free, the same 15 percent
slack SGLang leaves regardless of where the vocoder loads. Batch histogram: causal calls 255, mean
8.6 rows, max 16; leftover calls 139, mean 7.8, max 16. E2 tables: chunk mask patch and conv cache
patch exact (0.0 difference); packed versus native mel 86 to 99 dB SNR; wav differences are HiFT
phase sensitivity as before. Unit tests: 1 failed (the executor test with `flow_batch_admission_frames=200`
drove warmup to a negative token count; warmup is removed in the rebased tree).

## 2. Where the throughput went

Six requests produced more than 20 s of audio, two exactly 81.92 s, which is 2,048 tokens, the
client's `max_new_tokens`. The old run had two such requests and its OOM killed the steps they
widened. Examples (words, audio seconds): "Can we please leave now?" 5 words, 58.8 s; "Don't take
any chances." 4 words, 38.6 s.

Step stalls from the serve.log timestamps between consecutive Flow batch lines, 359 s run:

| gap | count | total s |
|---|---|---|
| over 1.5 s | 48 | 135.5 |
| over 3 s | 20 | 77.9 |

The longest gaps, 3.5 to 6.1 s, follow causal calls of 10 to 16 rows during the windows when the
2,048 token requests were alive. E3 puts a 16 row step padded to about 4,200 frames at 5.9 s, so
these are the runaway rows padding every other row in the step to their width, exactly the shape
the OOM run could not finish.

## 3. Step rule model

`experiments/e4_step_padding_model.py` replays the run's own audio durations with hop growth
25, 50, 100, prompts assumed at 3 to 10 s, and the E3 cost model (280 ms in-server call floor, 80 us
per padded row frame). Round time for 16 rows, mean over 3,000 draws:

| rule | typical round s | round with one 4,200 frame row s |
|---|---|---|
| no bound (de47ace7e) | 1.91 | 5.66 |
| cap 8,000 padded frames | 1.61 | 2.14 |
| cap plus 25 percent pad budget | 1.71 | 2.25 |
| padding per step at most one call, 3,500 frames | 1.48 | 1.99 |

Sensitivity of the last rule to its budget: 1,750 frames 1.59 / 2.09, 3,500 1.48 / 1.99, 7,000
1.39 / 1.92, 14,000 1.53 / 2.04. The result is flat within a factor of four around the measured
break-even, so an H100 constant does not need to be exact to hold on a neighbouring GPU.

## 4. Decision

The padding budget (c7a12957a, 3,500 padded frames) was withdrawn the same day: its constant is
the ratio of a host launch floor to a GPU per frame rate and pins to the H100. The cost it bounded
is a layout defect, the packed Flow adapter padding every row to the widest, which SGLang never
does (flat token stream, `cu_seqlens`, ragged attention; its only padding rule is the prefill
graph runner's post formation two times check, prefill_cuda_graph_runner.py:1190-1195).

Commit 8701415d8 on af843811b, `packed_dit.py`: the streaming hops and the finals run the DiT over
the rows concatenated along the sequence. Per token modules (input projection, AdaLN, feed forward,
long skip, output projection) are unchanged; the causal conv position embedding runs on the rows
scattered to a padded layout and gathered back (exact, it is causal with left zero padding); rotary
frequencies are gathered by per token position; attention runs within each row through
flashinfer's ragged prefill kernel (`BatchPrefillWithRaggedKVCacheWrapper`, one plan per Flow
call, a flattened per row boolean mask for the chunk causal streaming case, no mask for finals),
with a per row SDPA loop as the CPU and exactness reference. The unconditional CFG twins are rows
of the same packed call. The scheduler admits every runnable hop up to the batch size with no
padding rule; a wide row costs its own frames only. The graphed non-streaming `inference` for
buffered requests and the TensorRT estimator keep the padded layout.

Local check on a tiny real DiT (vendored CosyVoice source, CPU, float64): packed forward against
`DiT.forward` 2.7e-14, packed Euler solve against the padded solve 0.0, rows packed together
against each row alone 3.2e-14, for both the chunk causal and the bidirectional mask. The same
checks are `tests/unit_test/fun_cosyvoice3/test_packed_dit.py` (skipped where cosyvoice is not
installed).

Follow-ups in the same series: ecf1512d2 warms one hop and one final in the factory before
readiness (the flashinfer kernel variants load there, not on the first request), 5f0f489e5
guards the conv cache patch against a second install.

## 5. The attention kernel (E2 runs at 16ade9019, 6befb4517, cc1d2238a and the forensics run)

The first packed version ran attention through flashinfer's ragged prefill. Four E2 runs found
two defects and settled the kernel choice:

1. flashinfer's `plan()` keeps the previous plan's custom mask buffer when no mask is passed and
   `run()` applies it whenever the buffer is set, so every final planned after a hop ran under the
   hop's chunk mask: finals 16 dB against the SDPA reference. Fixed by one wrapper per mask mode.
2. With that fixed, the packed SDPA path was bit identical to the shipped padded path (38 to 40 dB
   from float32, the serving dtype's own distance), but the flashinfer path sat at 26 to 31 dB. Per
   call, 90 of 220 attention calls in a batch were under 40 dB, the worst at 0.6 dB (hops) and
   minus 3.2 dB (finals), clustered on hop blocks 2 and 4 and final blocks 5 and 20 with key
   magnitudes of 500 to 1,000. On the saved worst call tensors fresh fa2, fa3 and auto wrappers
   reproduced it deterministically and disagreed with each other, while SDPA in bfloat16 agreed
   with SDPA in float32 at 58 to 70 dB. The mechanism, read in flashinfer's source at the v0.6.18
   tag and verified line by line: the kernels keep the logits in float32 (FA2 `DTypeQKAccum` is
   float for bfloat16, prefill.cuh:2554; FA3 `tSrS` is float, mainloop_mma.cuh:73) but use a finite
   sentinel for minus infinity, `constexpr float inf = 5e4` (math.cuh:33), as the running max
   initialiser (prefill.cuh:949), the mask fill (prefill.cuh:350) and FA3's fill (hopper/
   attention_updater.cuh:168), in the raw q.k domain before the scale. A query row whose raw
   logits all sit below minus 5e4 never raises the max and the output guard zeroes it
   (variant_helper.cuh:86). This DiT has no QK normalisation; on the worst hop call 81 percent of
   query rows and on the worst final call 53 percent have every raw logit below minus 5e4. A CPU
   replay of those semantics on the saved tensors gives 0.6 dB for the hop, the observed 0.6, and
   3.7 dB for the final, the observed fresh FA2 value. flashinfer fixed it in PR #4401 (merged
   2026-08-20, IEEE infinity on main), after the v0.6.18 tag SGLang pins; issues #4267, #4450,
   #4451, #4452. Our usage was correct. An earlier reading of mine, bfloat16 rounding of the
   logits, reproduced a similar SNR by coincidence and is withdrawn.

Decision, commit ee699a4a5: the packed rows keep every per token module packed and run attention
as the padded DiT runs it, SDPA on the rows scattered to the padded layout under the row key mask
and the chunk causal mask, masked with IEEE minus infinity. The packed forward is now bit identical
to `DiT.forward` on the tiny real DiT (0.0 in float64) and carries no flashinfer, workspace,
wrapper or dtype dispatch. Any future ragged kernel for this model must be qualified on its raw
logit range, below minus 1.8e5 and above 1.6e6 on the saved calls. The attention's own padded cost remains, about 8 percent of the per token work
at typical hop widths and a third at a 4,196 frame runaway row; a ragged attention with float32
logits (FlexAttention under torch.compile) is the follow up for that share.

Gate for the next run at ee699a4a5: E2 table 5 at 0.0 for hops and finals, table 6 packed rows
equal to the padded rows; then zero failures, zero OOM, C50 at or above 71.5, req/s within
2 percent of 3.954, WER within noise of 1.33 to 1.43, and the serve.log gap table with no gap over
3 s.
