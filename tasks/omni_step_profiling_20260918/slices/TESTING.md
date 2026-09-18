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

## 3. Later runs

P1 (P1-e1 numerics, P1-e2 graph timing) and the matrix A/B runbook are added here once
the P1 branch and its scripts exist; each is checked against the branch head before the
box runs it.
