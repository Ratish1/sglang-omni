# Box steps for the slices (moss, container sglang-omni-ratish)

Every step runs through the moss skill form (`ssh moss 'docker exec -i
sglang-omni-ratish bash -s' <<'EOF' ... EOF`). Scripts are copied into `$S` from
`tasks/omni_step_profiling_20260918/scripts/` by the planner and checked by md5 before a
run; the runner does not edit them.

## 1. Rules for every run

- Census first: `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` and
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`. Take
  one card with no compute process and at most 3 MiB used. None free: report and stop.
- Never kill another user's process. Leave nothing running at the end.
- Long commands run in the background with `nohup ... > file 2>&1 &`, their PID saved.
  Check every 3 minutes: the PID is alive, the output file's size and mtime moved, the
  last 5 lines. Report each check in one line. A file that has not moved for 10 minutes
  while the PID is alive: send `py-spy dump --pid <pid>` if available, else the last 40
  lines, and wait for instructions.
- Results come back verbatim. No interpretation, no reruns with changed arguments.
- Every file a run writes stays in its `$OUT`. After the run the planner copies the run's
  analysis outputs (traces, logs, tables, tensors; never WAVs) to
  `artifacts/moss_omni_step_profiling/` on the Mac: a tarball built on the box with
  `--exclude='*.wav'`, fetched with `docker exec ... cat`, md5 compared, traces checked
  with `gzip -t`. Anything too slow to copy is skipped and named.

## 2. Run 03: V1-e0 to V1-e4 and P2-e1 (no server)

```bash
cd /workspace/sglang-omni
source .venv/bin/activate
export CARD=<card from the census>
export W=/workspace/sglang-omni/.tmp/wt/step-prof       # upstream 144bd6399 + profiler patch
export S=/workspace/sglang-omni/.tmp/omni_step_profiling/scripts
export OUT=/workspace/sglang-omni/.tmp/omni_step_profiling/run03
export MODEL=/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p $OUT
git -C $W rev-parse HEAD > $OUT/head.txt                # must be 144bd6399...
md5sum $S/vocoder_resident_bench.py $S/sampler_stage_bench.py > $OUT/md5.txt
cd $W
```

Then, one at a time, each with `CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$W`:

| # | command | output | expected duration |
| --- | --- | --- | --- |
| a | `python $S/vocoder_resident_bench.py dispatch --model $MODEL` | `$OUT/v1e0_dispatch.txt` | under 2 min |
| b | `python $S/sampler_stage_bench.py` | `$OUT/p2e1_stages.txt` | under 3 min |
| c | `python $S/vocoder_resident_bench.py timing --model $MODEL` | `$OUT/v1e1_timing.txt` | 10 to 30 min |
| d | `python $S/vocoder_resident_bench.py numerics --model $MODEL` | `$OUT/v1e2_numerics.txt` | 5 to 20 min |
| e | `python $S/vocoder_resident_bench.py fulldecode --model $MODEL` | `$OUT/v1e4_fulldecode.txt` | under 10 min |

If a command fails, send its traceback and continue with the next one. Return: head.txt,
md5.txt, and each output file in full.

## 3. Run 04: P1 unit tests, P1-e1 numerics, P1-e2 graph timing (no server)

Trees: base `$W` (above); pair `$P = /workspace/sglang-omni/.tmp/wt/p1`, upstream
144bd6399 plus `p1_pair_pass.patch` (branch `perf/qwen3-tts-predictor-pair-pass`; the
qwen3_tts files are identical at 144bd6399 and at main 27b5b0d4f).

```bash
cd /workspace/sglang-omni
source .venv/bin/activate
export CARD=<card from the census>
export W=/workspace/sglang-omni/.tmp/wt/step-prof
export P=/workspace/sglang-omni/.tmp/wt/p1
export S=/workspace/sglang-omni/.tmp/omni_step_profiling/scripts
export OUT=/workspace/sglang-omni/.tmp/omni_step_profiling/run04
export MODEL=/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p $OUT
git -C $P diff --stat > $OUT/p1_stat.txt
md5sum .tmp/omni_step_profiling/p1_pair_pass.patch $S/predictor_pair_bench.py > $OUT/md5.txt
for T in $W $P; do (cd $T && PYTHONPATH=$T python -c "import sglang_omni; print(sglang_omni.__file__)"); done > $OUT/import_paths.txt
```

(From `/workspace/sglang-omni` the current directory precedes PYTHONPATH on sys.path and
the main checkout would be imported; every step below runs from inside its tree.)

`import_paths.txt` must name `$W/...` then `$P/...`. Then, one at a time, each with
`CUDA_VISIBLE_DEVICES=$CARD`:

| # | from | command | output |
| --- | --- | --- | --- |
| a | `cd $P` | `PYTHONPATH=$P python -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider` | `$OUT/pytest_p1_qwen3_tts.txt` |
| b | `cd $W` | recording boot, below | `$OUT/record_serve.log`, `$OUT/record_driver.txt` |
| c | `cd $W` | `PYTHONPATH=$W python $S/predictor_pair_bench.py run --model $MODEL --inputs $OUT/predictor_inputs.pt --out $OUT/p1_base.pt` | `$OUT/run_base.txt` |
| d | `cd $P` | `PYTHONPATH=$P python $S/predictor_pair_bench.py run --model $MODEL --inputs $OUT/predictor_inputs.pt --out $OUT/p1_pair.pt` | `$OUT/run_pair.txt` |
| e | `cd $W` | `PYTHONPATH=$W python $S/predictor_pair_bench.py run --model $MODEL --inputs $OUT/predictor_inputs.pt --out $OUT/p1_truth.pt --fp32` | `$OUT/run_truth.txt` |
| f | `cd $W` | `python $S/predictor_pair_bench.py compare --base $OUT/p1_base.pt --pair $OUT/p1_pair.pt --truth $OUT/p1_truth.pt` | `$OUT/p1e1_compare.txt` |
| g | `cd $W` | `PYTHONPATH=$W python $S/predictor_pair_bench.py time --model $MODEL --inputs $OUT/predictor_inputs.pt` | `$OUT/p1e2_time_base.txt` |
| h | `cd $P` | `PYTHONPATH=$P python $S/predictor_pair_bench.py time --model $MODEL --inputs $OUT/predictor_inputs.pt` | `$OUT/p1e2_time_pair.txt` |

Step b, the recording boot (the served talker's predictor inputs; the qwen-tts reference
model does not run under the venv's transformers 5.12.1). `export PORT=8013`, then from
`cd $W`:

```bash
setsid bash -c "echo \$\$ > $OUT/record_server.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD \
  PYTHONPATH=$S/record_predictor_inputs:$W PREDICTOR_RECORD_OUT=$OUT/predictor_inputs.pt \
  python -u -m sglang_omni.cli serve --model-path $MODEL --port $PORT" > $OUT/record_serve.log 2>&1 &
```

Poll `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health` every 10 s
until 200 (fail after 15 min: send the last 80 lines of record_serve.log). Then
`PYTHONPATH=$W python $S/profile_workloads.py --url http://127.0.0.1:$PORT --model $MODEL
--out $OUT/record_driver --kind decode --batch 16 --steps 40 --label record --no-capture
> $OUT/record_driver.txt 2>&1`. Pass: record_serve.log has `predictor recorder: saved`
and `$OUT/predictor_inputs.pt` exists. Teardown: `kill -TERM -- -$(cat
$OUT/record_server.pgid)`, wait 10 s, `-KILL` only if the group is alive; the card must
be back near 0 MiB before step c.

Step a is compared against the base suite log of the same tests on `$W`
(`.tmp/omni_step_profiling/pytest_base_qwen3_tts.log`): return both failure lists.
Known on base (sm89): `test_eager_predictor_accepts_a_strided_input_and_leaves_its_neighbours`.
If b fails, stop and report (c to h need its output). Return: p1_stat.txt, md5.txt,
import_paths.txt, the failure lines and summary of a and of the base log, and every
other output file in full.

## 4. Run 05: P1-e1b, V1-e5 and V2-e1 (no server)

Same exports as run 04 (`W`, `P`, `S`, `MODEL`, HF offline) plus
`export OUT=/workspace/sglang-omni/.tmp/omni_step_profiling/run05 R4=/workspace/sglang-omni/.tmp/omni_step_profiling/run04`,
`mkdir -p $OUT`, `md5sum $S/predictor_pair_bench.py $S/predictor_pair_paired.py
$S/vocoder_resident_bench.py $S/vocoder_attribution_bench.py > $OUT/md5.txt`. All from
`cd $W` with `CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$W`, one at a time:

| # | command | output |
| --- | --- | --- |
| a | `python $S/predictor_pair_bench.py run --model $MODEL --inputs $R4/predictor_inputs.pt --out $OUT/p1_base_bs32.pt --batches 32` | `$OUT/run_base_bs32.txt` |
| b | same with `--fp32 --out $OUT/p1_truth_bs32.pt` | `$OUT/run_truth_bs32.txt` |
| c | `python $S/predictor_pair_paired.py --base $OUT/p1_base_bs32.pt --pair $OUT/p1_base_bs32.pt --truth $OUT/p1_truth_bs32.pt` | `$OUT/p1e1b.txt` |
| d | `python $S/vocoder_attribution_bench.py split --model $MODEL` | `$OUT/v1e5_split.txt` |
| e | `python $S/vocoder_attribution_bench.py compile --model $MODEL --arm eager` | `$OUT/v2e1_eager.txt` |
| f | `python $S/vocoder_attribution_bench.py compile --model $MODEL --arm static` | `$OUT/v2e1_static.txt` |
| g | `python $S/vocoder_attribution_bench.py compile --model $MODEL --arm dynamic` | `$OUT/v2e1_dynamic.txt` |

After c, delete `$OUT/p1_base_bs32.pt` and `$OUT/p1_truth_bs32.pt` (large, not needed
further). Expected durations: a and b 1 to 3 min each, d under 10 min, e under 10 min,
f 20 to 60 min (44 compiles), g 5 to 30 min.

## 5. Run 06: P2 unit tests and P2-e2 (no server)

Tree `$Q = /workspace/sglang-omni/.tmp/wt/p2`: upstream 144bd6399 plus
`p2_sampler_noise.patch` (branch `perf/qwen3-tts-sampler-noise`; the touched files are
identical at 144bd6399 and main 27b5b0d4f). Exports as run 04 plus `Q` and
`OUT=/workspace/sglang-omni/.tmp/omni_step_profiling/run06`; `mkdir -p $OUT`;
`md5sum .tmp/omni_step_profiling/p2_sampler_noise.patch $S/sampler_noise_bench.py >
$OUT/md5.txt`; `(cd $Q && PYTHONPATH=$Q python -c "import sglang_omni;
print(sglang_omni.__file__)") > $OUT/import_path.txt` (must name `$Q/...`). From `cd $Q`
with `CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$Q`:

| # | command | output |
| --- | --- | --- |
| a | `python -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider` | `$OUT/pytest_p2_qwen3_tts.txt` |
| b | `python $S/sampler_noise_bench.py` | `$OUT/p2e2_noise.txt` |

## 6. Run 07: full server profiles of base, P1, P2 (and base again), then nsys

Trees (all upstream 144bd6399 plus `step_profiler_v2.patch`, the profiler as committed on
`feat/omni-step-profiler`, md5 7f864b156cfd5623a7dd65f84c6a6384):
`.tmp/wt/prof-base`; `.tmp/wt/prof-pf` (+ `pf_base_prefill_graph.patch`);
`.tmp/wt/prof-p1` (+ `p1_pair_pass.patch`); `.tmp/wt/prof-p2`
(+ `p2_sampler_noise.patch`). Memory: the shipped default `mem_fraction_static` 0.85
(no fifth argument); `mem.csv` records the card's memory every second so the fraction
can be sized from the measured peak afterwards.

Each boot is one script call, run in the background with nohup, one boot at a time on
one card, in this order (base twice brackets the arms for drift):

```bash
cd /workspace/sglang-omni && source .venv/bin/activate
S=/workspace/sglang-omni/.tmp/omni_step_profiling/scripts
R=/workspace/sglang-omni/.tmp/omni_step_profiling/run07
nohup bash $S/run_profile_boot.sh /workspace/sglang-omni/.tmp/wt/prof-base $R/base1 $CARD 8021 > $R/base1.log 2>&1 &
# then prof-pf -> $R/pf (port 8026), prof-p1 -> $R/p1 (8022), prof-p2 -> $R/p2 (8023),
# prof-base -> $R/base2 (8024)
nohup bash $S/run_nsys_boot.sh /workspace/sglang-omni/.tmp/wt/prof-base $R/nsys_base $CARD 8025 > $R/nsys_base.log 2>&1 &
```

Per boot: `progress.txt` shows each capture's start, end and return code; check it every
3 minutes with `serve.log` recency. A boot takes about 15 to 25 minutes; the nsys boot
longer (server under nsys). `FAILED` in a boot dir means the server never became healthy:
send the last 80 lines of its serve.log and go on to the next boot. Return per boot: its
`progress.txt`, `armed_marker.txt`, `import_path.txt`, the `.err` files that are not
empty, and `ls -la` of the boot dir; for nsys also `c6_probe.log`, `nsys_start.log`,
`stats.log`. The planner reads the ledgers and traces from the copy.

## 7. Run 08: V-e6, the revised vocoder chain (no server; after run 07)

Exports as run 03 (`W` is `.tmp/wt/step-prof`, the base tree) with
`OUT=/workspace/sglang-omni/.tmp/omni_step_profiling/run08`; `mkdir -p $OUT`;
`md5sum $S/vocoder_chain_bench.py $S/vocoder_resident_bench.py
$S/vocoder_attribution_bench.py > $OUT/md5.txt`. From `cd $W` with
`CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$W`, one at a time:

| # | command | output | expected |
| --- | --- | --- | --- |
| a | `python $S/vocoder_chain_bench.py sweep --model $MODEL --depthwise resident` | `$OUT/ve6_sweep_dw_resident.txt` | 20 to 40 min |
| b | `python $S/vocoder_chain_bench.py sweep --model $MODEL --depthwise ncl` | `$OUT/ve6_sweep_dw_ncl.txt` | 20 to 40 min |
| c | `python $S/vocoder_chain_bench.py split --model $MODEL --depthwise resident` | `$OUT/ve6_split.txt` | under 15 min |
| d | `python $S/vocoder_chain_bench.py numerics --model $MODEL --depthwise resident` | `$OUT/ve6_numerics.txt` | 10 to 30 min |

## 8. Run 09: PF A/B, c16 stream first (six boots at once, cards 1 to 6)

A is upstream main `27b5b0d4f`, B is `origin/perf/qwen3-tts-base-prefill-graph`
`2f60fc1a8` (one commit on `27b5b0d4f`). Scripts come from the pushed
`feat/omni-step-profiler`, not a copy. The node is free (census at 1 MiB on all 8 cards,
128 CPUs); the census step still runs first and a busy card stops the run.

```bash
cd /workspace/sglang-omni
# the container clone does not map named fetches to origin/*, so the refspecs are explicit
git fetch origin +refs/heads/perf/qwen3-tts-base-prefill-graph:refs/remotes/origin/perf/qwen3-tts-base-prefill-graph \
  +refs/heads/feat/omni-step-profiler:refs/remotes/origin/feat/omni-step-profiler && git fetch upstream main
git worktree add --detach .tmp/wt/pf-a 27b5b0d4f
git worktree add --detach .tmp/wt/pf-b origin/perf/qwen3-tts-base-prefill-graph
git worktree add --detach .tmp/wt/step-prof-tools origin/feat/omni-step-profiler
S=/workspace/sglang-omni/.tmp/wt/step-prof-tools/tasks/omni_step_profiling_20260918/scripts
R=/workspace/sglang-omni/.tmp/omni_step_profiling/run09
mkdir -p $R
nohup bash $S/run_pf_ab.sh $R /workspace/sglang-omni/.tmp/wt/pf-a /workspace/sglang-omni/.tmp/wt/pf-b > $R/run.log 2>&1 &
```

Check: `git -C .tmp/wt/pf-a rev-parse HEAD` is `27b5b0d4f...`, `pf-b` is `2f60fc1a8...`.
Per boot dir (`a_c16_stream`, `b_c16_stream`, `a_c1_stream_seeded`, `b_c1_stream_seeded`,
`a_c16_buffered`, `b_c16_buffered`): `progress.txt` every 3 minutes with `serve.log` and
`bench.log` recency. Expected: c16 generation 10 to 25 min, seeded c1 60 to 120 min, WER
and similarity 10 to 20 min each after. `FAILED` means the server never became healthy.
`$R/DONE` ends the run; then `$R/identity.txt`. Return per boot: `progress.txt`,
`head.txt`, `import_path.txt`, `ls -la`, and any traceback. The planner copies
`$R` without WAVs and reads everything.

## 9. Later runs

The matrix A/B runbook (section 2 of 00_PRINCIPLES_AND_AUDIT.md) is added here once
P1-e1 passes; it is checked against both branch heads before the box runs it.
