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

Commit c7a12957a: a step admits rows in slack order while `rows * widest - useful` stays within
`flow_step_pad_frames` (3,500, `DEFAULT_FLOW_STEP_PAD_FRAMES` in stages.py, pinned by E3 and the
Nsight in-server floor). A wider row runs in its own step, as it does on main. The non-streaming
admission frames and pad budget from #1899 stay untouched on their own path.

Gate for the next run at c7a12957a: zero failures, zero OOM, C50 at or above 71.5, req/s within
2 percent of 3.954, WER within noise of 1.33 to 1.43, and the gap table above with no gap over 3 s
outside a singleton step.
