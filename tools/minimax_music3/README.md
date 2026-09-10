# MiniMax Music 3 remote qualification

Run on H200 with the model dependencies installed, including pinned SGLang v0.5.19. This is a qualification tool for the test branch; it does not add capture hooks to production model files. No GPU results are checked into this branch or implied by the presence of this tool.

The test branch is `test/minimax-music3-review-fixes`, based on PR #1559 head `ac288638261e95c0b895ebe005e677af8d12b163`. Keep these tooling changes separate when moving the production fixes back to the PR.

## What the three runs establish

- **Cookbook:** sends all 17 request payloads in the checked-in cookbook, including the three seed variations, structured blues caption, five genre examples, OpenAI example's request fields, and three parallel requests. The HTTP bodies are preserved in `cookbook_requests.json`, with source line numbers. All clients use the same HTTP transport in the harness; it does not execute installation commands or require curl/OpenAI SDK wrappers. It launches the real CLI in both documented single- and dual-GPU layouts. WAVs, individual failures and logs are retained for inspection/listening.
- **Performance:** uses c1 and c16 by default, 32 requests per measured group and three alternating base/candidate rounds. A c16 client load is meaningful: the model supports continuous batching and 16 logical requests occupy 32 CFG rows. This measures serving performance without requiring a pre-existing model benchmark. It records actual generated audio, AR frame counts when complete worker logs are available, request latencies, observed peak client concurrency and GPU telemetry. It does not assume frame caps equal actual generated work.
- **Accuracy:** serves the first full cookbook song with the same seed independently on each arm, without forced codes. It uses the existing AR hidden-window dump and a test-only forward hook to save each DIT condition and **final full latent before DAV**. Tensor capture is untimed and never enabled for performance runs. This is a free-generation comparison; diverging AR states also change subsequent acoustic latents.

The default tests FP32 acoustic serving. Add `--dtype bfloat16` for the precision-fix path. A BF16 result is not expected to equal the old incorrect BF16 implementation. Latent error metrics are diagnostic measurements, not a validated perceptual-quality threshold: listen to the cookbook outputs as well.

## Prepare the two source trees

Use the same Python environment for both arms. The launcher selects Omni source with `PYTHONPATH`; it does not reinstall or upgrade dependencies between runs. Use a complete local checkpoint snapshot, including `language_model`, legacy audio/tokenizer components, DIT and DAV. Each server gets its own lightweight checkpoint view with copied JSON and linked weights, so the base's config rewrite cannot alter the input seen by the candidate.

For attribution to this PR, the frozen merged-main base is:

```bash
git worktree add /sgl-workspace/minimax-main 428b73225f808f0d93669140983e90839c9aea0e
```

To compare current upstream main instead, create the base worktree at the exact desired upstream commit and use that path. This is a valid additional serving comparison, but main changes beyond the PR's merged parent can affect the result. The harness records both commits, tracked diffs, package versions, SGLang source/commit and GPU information. Keep tracked source clean when collecting final evidence.

The examples below use:

```bash
MINIMAX_CANDIDATE=/sgl-workspace/minimax-candidate
MINIMAX_BASE=/sgl-workspace/minimax-main
MINIMAX_SNAPSHOT=/models/minimax-music3-snapshot
MINIMAX_TOOL="$MINIMAX_CANDIDATE/tools/minimax_music3/qualify.py"
```

`MINIMAX_CANDIDATE` must contain this test branch. The tool reads its workload manifest and capture entry point from that checkout even when the server imports the base checkout.

## Cookbook and quality

```bash
python "$MINIMAX_TOOL" run --mode cookbook \
  --base-repo "$MINIMAX_BASE" --candidate-repo "$MINIMAX_CANDIDATE" \
  --model-path "$MINIMAX_SNAPSHOT" --output /results/minimax-cookbook
```

Both layouts run by default. With one H200, add `--layouts single`; this does not qualify the dual-GPU cookbook. Select physical devices with `--single-gpu 0` and `--dual-gpus 0,1`.

Repeat with `--max-running-requests 32` and a new output directory to check the cookbook's explicit 32-request engine configuration. To actually load 32 concurrent clients, use the performance mode with `--max-running-requests 32 --concurrency 1 16 32 --requests 32`; changing the engine limit alone does not create c32 traffic.

Inspect every group `events.jsonl` and WAV, not just HTTP status. The harness checks decoded sample rate, stereo shape, nonempty finite samples, and records peak/RMS/duration. It cannot judge intelligible vocals, musical quality or seam audibility. Listen to comparable base/candidate files, especially the longer blues example and acoustic overlaps. Early EOS can produce a shorter valid song.

## c1/c16 performance

```bash
python "$MINIMAX_TOOL" run --mode performance \
  --base-repo "$MINIMAX_BASE" --candidate-repo "$MINIMAX_CANDIDATE" \
  --model-path "$MINIMAX_SNAPSHOT" --requests 32 --rounds 3 \
  --output /results/minimax-performance

python "$MINIMAX_TOOL" summarize --results /results/minimax-performance \
  --output /results/minimax-performance/paired.json
```

Each concurrency warms with that many requests before a measured group. Requests use the first cookbook song, cap 750 frames, and a fixed seed list shared by both arms. Runs alternate A/B then B/A; servers are restarted between arms. Report paired distributions on an otherwise idle host, not one noisy comparison. No arbitrary percentage threshold is imposed.

The summary reports wall-time changes and audio-normalized throughput. `equal_generated_frames` only becomes true when complete AR completion logs establish equal frame counts. Different generated work makes raw wall-time comparison inconclusive. Audio-normalized throughput helps describe the result, but is not a substitute for equal-work kernel/stage profiling when diagnosing a small regression.

All request events are persisted as they finish. Any failed request or later server failure leaves the overall run in `harness_status="error"`; partial successful output cannot be summarized as valid performance evidence. Warmup/capture directories are distinct from measured directories. `c*_gpu.csv` records card-level telemetry; it includes other processes if the host is not isolated.

## Latent comparison

```bash
python "$MINIMAX_TOOL" run --mode accuracy \
  --base-repo "$MINIMAX_BASE" --candidate-repo "$MINIMAX_CANDIDATE" \
  --model-path "$MINIMAX_SNAPSHOT" --output /results/minimax-accuracy

python "$MINIMAX_TOOL" compare \
  --base /results/minimax-accuracy/single_round0_base/tensors \
  --candidate /results/minimax-accuracy/single_round0_candidate/tensors \
  --output /results/minimax-accuracy/single-latents.json --require-exact

python "$MINIMAX_TOOL" compare \
  --base /results/minimax-accuracy/dual_round0_base/tensors \
  --candidate /results/minimax-accuracy/dual_round0_candidate/tensors \
  --output /results/minimax-accuracy/dual-latents.json --require-exact
```

The comparator requires matching, nonempty hidden/condition/latent chunk sets. It reports shape/dtype differences, exact equality, maximum absolute error, RMSE, relative L2 and cosine similarity. Missing chunks, incompatible shapes and non-finite or empty tensors fail; `--require-exact` additionally fails numerical differences. For the default FP32 path, investigate any departure from the PR's claimed exact parity. For BF16, omit `--require-exact` to collect differences, then compare to an FP32 reference and perform listening checks; a successful metrics report is not an accuracy pass.

Interpret the first divergence in order: AR hidden window, projected condition, DIT latent, then waveform. If the AR states differ, latent differences alone do not isolate the acoustic implementation. Replaying the same hidden inputs through both acoustic decoders can localize that case, but must be reported separately from free-generation quality. This harness deliberately does not force reference codes into normal serving.

The capture entry point installs its hook in spawned stage processes and removes each hook after its decode call. A duplicate latent filename fails rather than silently overwriting evidence. Only one request is captured per server, so the existing seed/chunk naming remains unambiguous.

## Scheduler regression gates

The cookbook and performance modes do not force retraction. Run the focused tests remotely before expensive qualification:

```bash
cd "$MINIMAX_CANDIDATE"
python -m pytest tests/unit_test/minimax_music3 -q
```

The new tests use real SGLang Req, ScheduleBatch release/filter and PrefillAdder methods. They cover pair resource release, final-pair failure, admission equality, page alignment, generation clipping, replay embeddings, replay frame accounting and loaded FP32 phases before warmup. No local pytest was run when preparing the branch.

Also run a candidate server with `SGLANG_TEST_RETRACT=1 SGLANG_TEST_RETRACT_INTERVAL=100` and submit two concurrent 750-frame requests. Confirm a complete pair was actually retracted and resumed in the logs. Exercise public `/pause_generation` with `{"mode":"retract","stages":["minimax_music3_ar"]}` and `/continue_generation` with `{"stages":["minimax_music3_ar"]}` during generation. Repeat late in a request and with a constrained `--minimax_music3_ar.engine.max_total_tokens` pool. The benchmark launcher intentionally clears forced-retraction/capture environments, so run these lifecycle checks separately.

The scoped admission fix respects v0.5.19's existing conservative debit after each row. At the review's 2,200-token boundary the replay pair now fits. At a smaller capacity where upstream cannot admit both rows, the request gets an explicit reservation error instead of an indefinite wait. This does not change SGLang's accounting globally or temporarily rewrite generation limits. Removing that inherited conservatism would require a separate upstream admission change and qualification.

After H200 runs, retain the exact candidate commit, reports, logs and listening results. Move the validated production/test commits to the PR only after examining that evidence; do not treat these unexecuted remote gates as already passed.
