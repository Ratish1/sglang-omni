# 39. Runbook: the reference encoder, eager against captured graphs (doc 32 item 3, the bench)

Doc 38 puts the remaining early ids regression on the reference encoder thread: about
1,000 eager launches per encode, 15 device to host syncs from the conv padding
arithmetic, 32 quantizer stages for 16 kept codes, one lock handoff per launch. The
design has three parts (host side padding buffers, `num_quantizers=16`, graphs at
length buckets). This bench checks each part on real references before any runtime
code is written, the way doc 34 did for the window widths. No server, no Nsight.

## Steps

From `tmp/an` pulled to this doc's head, the venv active, GPU 1:

```bash
cd /sgl-workspace/sglang-omni/tmp/an
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tasks/perf_backlog/scripts/ref_encoder_graph_bench.py \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base --meta zhaochenyang20/seed-tts-eval-arrow --lang en \
  --samples 64 --bucket-frames 48,64,96,128,192 --batches 1,2 --reps 20 \
  --out /sgl-workspace/sglang-omni/tmp/ref_encoder_graph_bench.json 2>&1 \
  | tee /sgl-workspace/sglang-omni/tmp/ref_encoder_graph_bench.log
```

The script loads the speech tokenizer through `stages._load_qwen3_tts_tokenizer` in
bfloat16 as the vocoder does, takes the first 64 distinct reference files of the
corpus, and prints their frame count quantiles first. If the p90 frame count is above
the largest bucket, rerun with a larger `--bucket-frames` list so every reference fits.

## Reads

1. `host_pad.exact` and `q16.exact` must both be true: parts 1 and 2 change nothing
   in the codes. If either is false the design stops there and the mismatch is the
   finding.
2. `wrapper`, `host_pad` and `q16` host ms for one reference: the cost the thread pays
   today and after each part; device ms is the encoder's GPU time.
3. `graph.<frames>x<batch>`: replay host and device ms per key and the footprint in
   MiB. The bucket set is the smallest that covers the p90 frame count with replays
   whose device time stays near the eager device time of the same length.
4. `graph.batch1_check` and `batch2_check`: exact against q16 per reference; a
   mismatch lists positions and the first quantizer index. Mismatches confined to late
   quantizers are kernel selection noise of the kind batching already carries;
   mismatches at quantizer 0 mean the tail padding is not exact and the design is wrong.

Archive the json and the log. The runner is written only after read 1 passes and
reads 3 and 4 pick the keys.
