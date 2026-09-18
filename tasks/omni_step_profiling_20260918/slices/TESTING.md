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
- Every file a run writes stays in its `$OUT`; after the run the planner copies the whole
  `.tmp/omni_step_profiling` tree to `artifacts/moss_omni_step_profiling/` on the Mac,
  traces, logs and tensors included, and reads from that copy.

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
| b | `cd $W` | `PYTHONPATH=$W python $S/predictor_pair_bench.py record --model $MODEL --out $OUT/predictor_inputs.pt` | `$OUT/record.txt` |
| c | `cd $W` | `PYTHONPATH=$W python $S/predictor_pair_bench.py run --model $MODEL --inputs $OUT/predictor_inputs.pt --out $OUT/p1_base.pt` | `$OUT/run_base.txt` |
| d | `cd $P` | `PYTHONPATH=$P python $S/predictor_pair_bench.py run --model $MODEL --inputs $OUT/predictor_inputs.pt --out $OUT/p1_pair.pt` | `$OUT/run_pair.txt` |
| e | `cd $W` | `PYTHONPATH=$W python $S/predictor_pair_bench.py run --model $MODEL --inputs $OUT/predictor_inputs.pt --out $OUT/p1_truth.pt --fp32` | `$OUT/run_truth.txt` |
| f | `cd $W` | `python $S/predictor_pair_bench.py compare --base $OUT/p1_base.pt --pair $OUT/p1_pair.pt --truth $OUT/p1_truth.pt` | `$OUT/p1e1_compare.txt` |
| g | `cd $W` | `PYTHONPATH=$W python $S/predictor_pair_bench.py time --model $MODEL --inputs $OUT/predictor_inputs.pt` | `$OUT/p1e2_time_base.txt` |
| h | `cd $P` | `PYTHONPATH=$P python $S/predictor_pair_bench.py time --model $MODEL --inputs $OUT/predictor_inputs.pt` | `$OUT/p1e2_time_pair.txt` |

Step a is compared against the base suite log of the same tests on `$W`
(`.tmp/omni_step_profiling/pytest_base_qwen3_tts.log`): return both failure lists.
Known on base (sm89): `test_eager_predictor_accepts_a_strided_input_and_leaves_its_neighbours`.
If b fails, stop and report (c to h need its output). Return: p1_stat.txt, md5.txt,
import_paths.txt, the failure lines and summary of a and of the base log, and every
other output file in full.

## 4. Later runs

The matrix A/B runbook (section 2 of 00_PRINCIPLES_AND_AUDIT.md) is added here once
P1-e1 passes; it is checked against both branch heads before the box runs it.
