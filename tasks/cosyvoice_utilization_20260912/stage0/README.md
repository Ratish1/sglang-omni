# Stage 0: profiling and metrics before the Flow graph refactor

Written 2026-09-15 for `ROADMAP_20260915.md` stage 0. Everything here is new; no existing perfkit,
experiment, plan, readout or ledger file is changed by it.

## Tree under test

- `#2170` is squash merged into upstream main as `e7460794f`.
- `#2171` head `2eefbc4766` is rebased on it: 0 commits behind main, 8 ahead, base `main`,
  mergeable. `git merge-tree` of main and the head gives the head's own tree, so there is nothing to
  merge. CI on the head: CodeQL "Analyze (python)" failed, the GPU venv job was pending.
- Every run below uses `2eefbc4766`. Its only log line that main does not print is
  `Fun-CosyVoice3 vocoder warmup: hop` (streaming_vocoder.py:154); a boot without it is void.

## What stage 0 decides

| run | question | decides |
|---|---|---|
| V0 | which Python line issues each sync and each runtime copy per call kind | roadmap 0.2 and 1.3 |
| E5 | does hop k+1 reproduce hop k's frames; does a hop over cached K/V equal the full recompute; does the hop schedule change emitted mel | whether the hop graph key is (batch, hop frames) over a cached prefix (roadmap 1.2, 2.1) or total tokens over the packed rows |
| E6 | which attention kernel is within the serving dtype's own distance from float32 on this DiT's activations, hops and finals, and what each costs | the attention backend of the graphed packed step |
| G0 | what the 55 shape table costs at boot and in memory, and what replay buys over eager and over the packed call at each shape | whether the padded buffered graphs survive the refactor |
| census | the tree's own numbers at four points | A of the refactor's A/B |
| ledger | the shapes, waits and times the real workload produces per call | the key space and the first audio origin, measured, not taken from a table |

## The calls being measured, at `2eefbc4766`

```text
streaming step    streaming_vocoder.py:341 run_step (select_step_participants :300)
  hop             stages.py:1531 hop_batch -> :803 inference_causal
                    :151 pack_flow_inputs -> :530 prepare_flow_conditioning (lookahead 3 stripped)
                    :690 generate_flow_packed -> packed_dit.py:191 solve_flow_euler_packed
                      10 Euler steps x PackedDiT.forward (packed_dit.py:131), 2B CFG rows
                        per token modules on (1, total, 1024)
                        conv position embed: scatter to (rows, widest), conv, gather  :161
                        22 x RowAttention (:77): scatter Q,K,V to (rows, widest), SDPA with
                           (rows,1,W,W) key and chunk causal mask, gather  :97-109
  final           stages.py:1543 leftover_batch -> :784 inference_leftover (same, bidirectional)
  HiFT            stages.py:1559 hift_delta, whole accumulated mel, delta .cpu()
buffered          stages.py:1384 decode_batch -> :1249 adaptive_flow_requests_grouping
  group           stages.py:772 inference -> :630 generate_flow
                    runner.run :447  key (B, ceil(T/16)*16) in the 55 shape table (config.py:19-75)
                      hit: right pad, copy into static inputs, replay, crop
                      miss: :242 solve_flow_euler eager on (B, frames) padded
  HiFT            stages.py:1610 mel2wav_batch, padded to the longest mel
boot              stages.py:2054-2064 capture all 55 shapes, one pool; streaming_vocoder.py:132 warmup
```

## Variables and metrics

| name | meaning | source |
|---|---|---|
| P | prompt tokens after padding to a hop multiple | streaming.py pad_flow_prompt_to_hop |
| O, H | token offset already emitted, hop length (25, 50, then 100) | CosyVoice3StreamState |
| window | tokens a hop reads, O + H + 3 lookahead | streaming_vocoder.py:380 |
| row frames | 2 (P + O + H) for a hop, 2 (P + all tokens) for a final | prepare_flow_conditioning |
| new frames | 2H, the frames a hop emits | run_step slices at 2O |
| rows, B | requests in one Flow call; the DiT runs 2B rows for CFG | |
| total frames T | sum of row frames; the packed per token work | |
| widest W | largest row frames; attention and conv pay rows x W | |
| pad ratio | rows x W / T; 1.0 means no padding | |
| bucket | ceil(T / 16) x 16 for the buffered graph key | stages.py:78 |
| chunk | 50 frames: a frame sees keys before the end of its own chunk | DiT static_chunk_size |
| host_ms | Python wall of a call; for GPU work it is launch time unless the call syncs | ledger |
| gpu_ms | device time between CUDA events at call entry and exit | ledger |
| launches, syncs | CUDA runtime API counts per call | V0, G0 |
| wait since ready | time a stream was runnable before its step started; for a first hop it is the first audio queue | ledger step |
| raw logit | q.k before the 1/8 scale; FlashInfer 0.6.18 failed below -5e4 | E6 |
| SNR, worst row SNR | 20 log10 of truth norm over error norm, whole call and worst row | E5, E6 |
| float32 ulp mismatch | elements that differ by more than two float32 ulps of their magnitude | E5 |
| census | req/s, audio s/s, RTF, first audio mean p50 p95 p99, inter chunk, C50, C100, failures, WER, latency | benchmark_tts_seedtts |

## Setup on the box

```bash
cd /sgl-workspace/sglang-omni
git fetch https://github.com/sgl-project/sglang-omni.git perf/cosyvoice3-vocoder-load-path
git worktree add --detach /sgl-workspace/wt/cosy-2171 2eefbc4766
git fetch https://github.com/Ratish1/sglang-omni.git analysis/cosyvoice-utilization-20260912
ANALYSIS=$(git rev-parse FETCH_HEAD)
git -C /sgl-workspace/wt/cosyvoice-analysis checkout --detach "$ANALYSIS"

TREE=/sgl-workspace/wt/cosy-2171
S0=/sgl-workspace/wt/cosyvoice-analysis/tasks/cosyvoice_utilization_20260912/stage0
OUT=/sgl-workspace/wt/cosyvoice-analysis/artifacts/cosyvoice/stage0-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
cd "$TREE"
git rev-parse HEAD > "$OUT/head.txt"
python -c "import sglang_omni; print(sglang_omni.__file__)" > "$OUT/import_path.txt"
{ pip list 2>/dev/null | grep -iE "^(torch|sglang|sgl-kernel|sglang-kernel|kernels|flashinfer|x-transformers|onnxruntime) ";
  python -c "import sitecustomize; print('existing sitecustomize', sitecustomize.__file__)" 2>&1;
  echo "SGLANG_USE_SGL_FA3_KERNEL=${SGLANG_USE_SGL_FA3_KERNEL:-unset}"; nvidia-smi; } > "$OUT/env.txt"
nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv
```

`import_path.txt` must be under `/sgl-workspace/wt/cosy-2171`. GPU 0 must list no process before
each offline run. Every command runs from `$TREE`, so `sglang_omni` and `benchmarks` are the tree
under test.

## 1. Offline runs, GPU 0 alone, no server

```bash
cd "$TREE"
CUDA_VISIBLE_DEVICES=0 python "$S0/v0_flow_call_syncs.py" --device cuda:0 --json "$OUT/v0.json" 2>&1 | tee "$OUT/v0.txt"
CUDA_VISIBLE_DEVICES=0 python "$S0/e5_hop_prefix_exactness.py" --device cuda:0 --dtype float64 --json "$OUT/e5_float64.json" 2>&1 | tee "$OUT/e5_float64.txt"
CUDA_VISIBLE_DEVICES=0 python "$S0/e5_hop_prefix_exactness.py" --device cuda:0 --dtype bfloat16 --json "$OUT/e5_bfloat16.json" 2>&1 | tee "$OUT/e5_bfloat16.txt"
CUDA_VISIBLE_DEVICES=0 python "$S0/e6_attention_kernels.py" --device cuda:0 --json "$OUT/e6.json" --save-dir "$OUT/e6_worst" 2>&1 | tee "$OUT/e6.txt"
CUDA_VISIBLE_DEVICES=0 python "$S0/g0_flow_graph_cost.py" --device cuda:0 --json "$OUT/g0.json" 2>&1 | tee "$OUT/g0.txt"
```

- V0 prints, per call kind, every sync with the four innermost Python frames and the CUDA runtime
  API counts of one call.
- E5 float64 is the exactness gate: table 1 and table 3 with 0 float32 ulp mismatches means the
  prefix is stable and the cached hop is exact. The bfloat16 run shows the serving dtype's distance
  for the same comparisons.
- E6 prints the FA3 implementation SGLang loaded (`kernels-community/sgl-flash-attn3` from the hub,
  or `sgl_kernel`), the raw logit range, and one row per (kind, candidate): calls, min SNR and where,
  calls under 40 dB, worst row SNR, median kernel time. A candidate that errors is listed as
  unavailable with the error.
- G0 prints the shipped capture wall and memory, then one row per shape.

## 2. Census, bare tree, one boot per point

```bash
run_point() {
  local mode=$1 conc=$2 ledger=$3
  local name="${mode}-en-c${conc}"
  [ "$ledger" = 1 ] && name="${name}-ledger"
  local BOOT="$OUT/$name"
  mkdir -p "$BOOT"
  until [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
  git -C "$TREE" rev-parse HEAD > "$BOOT/head.txt"
  (cd "$TREE" && python -c "import sglang_omni; print(sglang_omni.__file__)") > "$BOOT/import_path.txt"
  nvidia-smi > "$BOOT/gpus_before.txt"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv >> "$BOOT/gpus_before.txt"
  local ledger_env=""
  [ "$ledger" = 1 ] && ledger_env="PYTHONPATH=$S0/call_ledger COSY_CALL_LEDGER_DIR=$BOOT/ledger"
  (cd "$TREE" && setsid bash -c "echo \$\$ > '$BOOT/server.pgid'; exec env CUDA_VISIBLE_DEVICES=0 $ledger_env python -m sglang_omni.cli serve --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000" > "$BOOT/serve.log" 2>&1 &)
  until curl -s http://127.0.0.1:8000/health | grep -q healthy; do sleep 5; done
  grep -q "Fun-CosyVoice3 vocoder warmup: hop" "$BOOT/serve.log" && echo marker > "$BOOT/marker.txt" || echo "no branch marker, boot void" | tee "$BOOT/marker.txt"
  nvidia-smi dmon -i 0 -s pucv -d 1 > "$BOOT/dmon.log" 2>&1 &
  local DMON=$!
  local stream_flag=""
  [ "$mode" = stream ] && stream_flag="--stream"
  (cd "$TREE" && python -m benchmarks.eval.benchmark_tts_seedtts \
      --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --lang en \
      --meta zhaochenyang20/seed-tts-eval-arrow \
      --use-existing-server --host 127.0.0.1 --port 8000 \
      --concurrency "$conc" --warmup 1 $stream_flag \
      --generate-only --output-dir "$BOOT/bench") > "$BOOT/bench.log" 2>&1
  kill "$DMON"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv > "$BOOT/gpus_after.txt"
  kill -TERM -- -"$(cat "$BOOT/server.pgid")"
  while kill -0 -- -"$(cat "$BOOT/server.pgid")" 2>/dev/null; do sleep 2; done
}

run_point stream 16 0
run_point stream 1 0
run_point buffered 16 0
run_point buffered 1 0
```

Quality on the census WAVs, with no server on GPU 0:

```bash
cd "$TREE"
for point in stream-en-c16 buffered-en-c16; do
  CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.benchmark_tts_seedtts --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --lang en --meta zhaochenyang20/seed-tts-eval-arrow \
    --transcribe-only --asr-model-path Qwen/Qwen3-ASR-1.7B \
    --output-dir "$OUT/$point/bench" > "$OUT/$point/transcribe.log" 2>&1
done
```

## 3. Call ledger, profiling boots, never compared with the census

`call_ledger/sitecustomize.py` loads `cosy_call_ledger.py` in every process of the serve when
`COSY_CALL_LEDGER_DIR` is set, after any sitecustomize the venv already has (listed in `env.txt`).
It wraps the calls in the diagram above and writes one JSON line per call; it reads shapes from host
metadata and times the device with CUDA events, so it adds no sync.

```bash
run_point stream 16 1
ls "$OUT/stream-en-c16-ledger/ledger" && head -c 300 "$OUT"/stream-en-c16-ledger/ledger/ledger_*.jsonl
run_point buffered 16 1
python "$S0/summarize_call_ledger.py" "$OUT/stream-en-c16-ledger/ledger" > "$OUT/stream-en-c16-ledger/summary.md"
python "$S0/summarize_call_ledger.py" "$OUT/buffered-en-c16-ledger/ledger" > "$OUT/buffered-en-c16-ledger/summary.md"
```

The ledger directory must hold a `graph_capture` line from boot; if it is empty the patch did not
load and the boot is void.

## 4. Return

```bash
cd /sgl-workspace/wt/cosyvoice-analysis
tar --exclude='*.wav' -czf "$(basename "$OUT").tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
```

`e6_worst/` holds up to four calls' query, key and value; copy the archive to the Mac's
`artifacts/`.

## 5. Readout

`../readouts/03_stage0_2eefbc476_<date>.md`, verdict first:

- V0: each remaining sync with its line and whether it is on the hop, final, buffered or HiFT path.
- E5: prefix stable yes or no, cached hop exact yes or no, schedule independent yes or no, with the
  float64 mismatch counts and the cached K/V bytes per frame.
- E6: per kind the candidates that are within the production bf16 path's SNR band on every call and
  every row, their kernel time against the production call, and the FA3 implementation that ran.
- G0: shipped capture seconds and memory; per shape replay, eager and packed ms.
- Census: one table per point.
- Ledger: rows, total and widest frames, pad ratio, distinct shape counts per call kind, first hop
  wait, graph hits and misses, HiFT time by accumulated frames.
