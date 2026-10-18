# PR #2909 independent verification

Reviewed production head: cefab51fc709735b12b3d25793229a8fed89722a.

## Environment and scope

- Radix node-radixark-16-0001, one assigned H200, visible device 0.
- Container sglang-omni-review2909, created through tasks/radix_gpu/up.sh.
- PyTorch 2.13.0+cu130, SGLang 0.5.21, Transformers 5.12.1.
- Qwen/Qwen3-Omni-30B-A3B-Instruct checkpoint snapshot 26291f793822fb6be9555850f06dfe95f2d7e695.
- Source/probes pushed only to review/pr2909-20261018 on the reviewer's fork. No author-branch changes.
- All pytest execution occurred inside the H200 container. No local pytest.

The old assignment had expired; a new one-GPU four-hour assignment was taken on the requested node. No assignment or container was released/deleted.

## Focused suite

Result: **724 passed, 28 skipped**, exit 0, 184.94 seconds.

Executed together:

```text
tests/unit_test/qwen3_omni/test_code2wav.py
tests/unit_test/qwen3_omni/test_codec_state_pool.py
tests/unit_test/qwen3_omni/test_code2wav_cuda_graph.py
tests/unit_test/qwen3_omni/test_code2wav_batching.py
tests/unit_test/qwen3_omni/test_code2wav_overlap.py
tests/unit_test/qwen3_omni/test_colocation_config.py
tests/unit_test/qwen3_omni/test_sglang_ar_budget.py
tests/unit_test/test_stage_device_contract.py
```

This covers the existing window path, graph budget/rollback behavior, batching, asynchronous output ownership, configuration/device contracts and the new stateful tests. Platform-gated skips are not claimed as tested accelerator coverage.

Node evidence: /data/review2909/pytest.log and pytest.exit.

## Mutation-verified test admission

The extra test uses the real tiny CUDA codec and graph runner. It ingests two streams before each ready step, executes first and next graphs with two rows, wraps the small ring and finishes final tails. Audio is compared by request, not merely checked for shape.

| Run | Result |
|---|---|
| Real implementation, new coalesced test | 1 passed |
| Swap slot IDs only for multi-row next graphs, existing PR scheduler parity test | 2 passed |
| Same mutation, new coalesced test | 1 failed as expected |

The failure reaches the audio comparison: 90 of 123 samples in the first compared stream differ beyond tolerance, with maximum violating absolute difference 0.18359375. The test does not fail in setup or by asserting a mocked return value.

This proves an uncovered, reachable state-ownership regression. The suggested test belongs in the existing test_code2wav.py file. The review task's standalone file is only a demonstrator.

Node evidence: cohort-test.log/.exit, mutation-existing.log/.exit, mutation-new.log/.exit.

## GPU lifecycle probes

Both passed using the real codec and graph runner:

1. Decode a first cohort, queue a delayed GPU follow-up, confirm its transfer has not completed, abort both streams, immediately reuse their codec slots, and compare new audio to fresh direct decoding. The injected device delay only establishes the in-flight interleaving; it is not a wall-clock performance assertion or a proposed CI timing test.
2. Raise after an actual graph has queued state writes. Verify both cohort participants get errors, all codec slots return, new streams still decode, and shutdown drains retired input/transfer ownership.

Node evidence: lifecycle.log/.exit.

## Trained decoder numerical checks

The probe downloads the checkpoint's actual code2wav weight shard, loads every code2wav parameter with strict=True, and compares against the complete HF fp32 decoder. The served decoder uses bf16, channels-last convolutions, the SGLang rotary kernel and fused SnakeBeta.

Seven stream lengths cross the attention boundary, ring boundary and repeated wraps: 11, 71, 72, 81, 82, 171 and 290 frames. Each runs with normal ten-frame chunks and an irregular 1/9/3/7 split, for **14 checks**. Every output length matches the whole forward. Worst incremental SNR loss relative to whole bf16 is approximately 0.64 dB in these checks.

Selected normal-chunk results, all relative to HF fp32:

| Codec frames | Whole bf16 SNR | Stateful chunks SNR | Old 25-frame-context windows SNR |
|---|---|---|---|
| 71 | 33.13 dB | 32.88 dB | 21.90 dB |
| 72 | 34.68 dB | 34.61 dB | 25.27 dB |
| 81 | 34.11 dB | 33.95 dB | 19.74 dB |
| 82 | 34.42 dB | 33.84 dB | 20.84 dB |
| 171 | 34.58 dB | 34.87 dB | 18.64 dB |
| 290 | 34.77 dB | 34.71 dB | 18.35 dB |

Dirty-slot reuse after a 290-frame stream is **bit identical** to a fresh pool for a new 1/9/10/9 split.

Eight trained streams were ingested as actual scheduler cohorts. Nine eight-row threshold decodes crossed the ring boundary, then each request finished a three-frame tail. The scheduler's graph output is **bit identical** to direct eager decoding with the same batch shapes and chunk boundaries. All 32 stream slots returned.

Matching the final-tail batch shape is necessary: comparing the serving path's one-row tail with an eight-row eager tail gives small bf16 GEMM/convolution rounding differences. The probe was corrected to use the same shapes rather than relaxing its assertion.

Node evidence: checkpoint.log, checkpoint.exit and checkpoint.json.

## Isolated later-window GPU work

Same trained served model on the same H200, one graph replay per timing sample, five warmup samples excluded, median of the next 20. The old arm computes a 35-frame stateless window; the new arm advances ten frames from retained state. Both exclude input-copy/Python/HTTP overhead.

| Rows | Stateless 35-frame graph | Stateful 10-frame graph | Reduction |
|---|---|---|---|
| 1 | 1.928 ms | 1.583 ms | 17.9% |
| 8 | 10.464 ms | 4.084 ms | 61.0% |

These are mechanism-level measurements, not a repeated main-versus-head end-to-end throughput benchmark. The stateless computation uses the unchanged whole-window operators after this PR's helper extraction. The first-window state-write cost remains a real tradeoff, documented by the author; this later-window table does not hide it or claim to measure it.

## Full serving lifecycle

Launch uses the repository's H200 colocated YAML, all seven logical stages, normal server warmup, and loopback binding:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_colocated_h200.yaml \
  --host 127.0.0.1 --port 8919
```

The normal server warmup completed through preprocessing, both encoders, thinker, talker and code2wav before /health became ready. The actual YAML's code2wav budget was 1.4% of H200 memory: 32 stream slots plus eight scratch slots occupied 112.0 MiB, and all 60 requested graphs published with a 512 MiB graph footprint. No runner disable occurred.

Final HTTP probe: **28/28 passed**, 14 streaming and 14 buffered, including 16 mixed requests submitted concurrently. Every WAV response/chunk decoded to nonempty finite 24 kHz audio; the health endpoint remained 200 afterward.

For each response mode, generation caps 1, 9, 10, 11, 20 and 21 exercised streams shorter than the first window, exact threshold endings and final partial windows. All 12 small-cap responses had exactly 1920×frames−555 samples, derived from the checkpoint's upsampling factors and the decoder's initial transposed-convolution trim:

| Frame cap | Samples in both streaming and buffered response |
|---|---|
| 1 | 1365 |
| 9 | 16725 |
| 10 | 18645 |
| 11 | 20565 |
| 20 | 37845 |
| 21 | 39765 |

The concurrent group used larger 90–92-frame caps to exercise multiple follow-up windows; natural EOS could finish earlier. This is a lifecycle/format check, not a corpus benchmark or an assertion that the generated speech has a particular WER.

The serving log contained no traceback, pool exhaustion warning, replay failure or no-audio error. An initial probe requested the unsupported client format pcm16, which fell back to WAV; the final probe requested WAV explicitly and decoded it. The format warning is not a PR finding.

Node evidence: server.log, serving-wav.log/.exit and serving.json. The owned server stopped cleanly after SIGINT: all stage workers exited, no serving/spawn worker processes remained in the task container, and H200 device memory returned to 0 MiB. The GPU lease, container, model cache and results remain allocated.

## Capture bounds at a one-stream cap

The exact production factory was exercised with one codec state slot and otherwise identical arguments. max_replay_rows=8 versus 1 gives:

| Item | Eight-row capture cap | One-row capture cap |
|---|---|---|
| Stream slots | 1 | 1 |
| Capture slots | 8 | 1 |
| Published graphs | 60 | 46 |
| Stateful graphs whose row count exceeds available slots | 14 | 0 |
| Pool bytes | 26415936 | 5870208 |
| Graph footprint bytes | 572522496 | 180355072 |
| Served audio bytes | 151380 | 151380 |
| Audio SHA-256 | faf13c896125086a8e0ae2e4cb6d0a9472facafc420cf6f2b8af07fec0abeff6 | faf13c896125086a8e0ae2e4cb6d0a9472facafc420cf6f2b8af07fec0abeff6 |
| Runtime | 2 graph replays, no fallback/error | 2 graph replays, no fallback/error |

This isolates reachable graph shapes: with one codec slot, two stateful participants cannot coexist, and stateless fallback participants use a separate group. The actual capture/pool saving is 393.6 MiB on this H200. Factory times were 12.76 and 10.75 seconds in these single runs; those timings are exploratory and are not treated as an established latency gain.

The factory accepts the model device as cuda with gpu_id=0, rather than an indexed device string; the probe follows that placement contract. Node evidence: capacity-8.json/.log/.exit and capacity-1.json/.log/.exit. This is a small-cap configuration finding, not a failure of the default 32-slot deployment.

## Reproduction

From the Mac, with the assignment active:

```bash
ssh ratish1@47.74.115.221 "podman exec -i sglang-omni-review2909 bash -s" <<'EOF'
cd /workspace/sglang-omni
git fetch -q origin review/pr2909-20261018 && git checkout -q FETCH_HEAD
uv pip install -q --python "$(command -v python3)" --prerelease=allow -e .
python3 -m tasks.pr2909_review_20261018.mutation_probe
python3 -m tasks.pr2909_review_20261018.mutation_probe --mutate --existing
python3 -m tasks.pr2909_review_20261018.mutation_probe --mutate
python3 -m tasks.pr2909_review_20261018.lifecycle_probe
python3 -m tasks.pr2909_review_20261018.checkpoint_probe
python3 -m tasks.pr2909_review_20261018.capacity_probe --rows 8
python3 -m tasks.pr2909_review_20261018.capacity_probe --rows 1
EOF
```

The third command intentionally returns nonzero. Run serving_probe only while the server is ready on port 8919. Do not run other GPU workloads during the timing part of checkpoint_probe.

## Limits and harness corrections

No independent full-corpus end-to-end A/B, WER or SIM run was performed. The author's paired full-corpus performance table is read as author evidence. The independent numerical oracle verifies the changed decoder, not the semantic quality of every generated transcript.

Initial exploratory probe runs used an unsupported generic batch-pump entry point, reused a mutable Transformers config across reference models, or compared different final-tail batch shapes. Those harness errors were corrected. They are not recorded as PR defects, and failed harness runs are not counted as successful verification.
