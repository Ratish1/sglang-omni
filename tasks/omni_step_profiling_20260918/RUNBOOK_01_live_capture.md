# Runbook 01: first live step captures, Qwen3-TTS 1.7B Base on moss

Goal: checks C1, C3, C4, C5 of DESIGN.md section 5 and the first steady-step ledger.
One boot of the patched tree (`patches/step_profiler.patch` on upstream 144bd6399), one
card, every capture on that boot. No A/B in this runbook.

All commands go through the moss skill form. Never kill another user's process. Report
progress at least every 3 minutes while anything runs; leave nothing running at the end.

## 0. Setup (container, /workspace/sglang-omni, .venv active)

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # take a card at 0-1 MiB
export CARD=<card> PORT=8011
export MODEL=/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export W=/workspace/sglang-omni/.tmp/wt/step-prof          # patched tree from the unit test run
export S=/workspace/sglang-omni/.tmp/omni_step_profiling/scripts
export OUT=/workspace/sglang-omni/.tmp/omni_step_profiling/run01
mkdir -p $S $OUT
```

Copy `scripts/profile_workloads.py` and `scripts/step_ledger.py` from
`/Users/ratish/sglang-omni/tasks/omni_step_profiling_20260918/scripts/` into `$S`, md5
both sides. Clone the analysis skill once:

```bash
git clone https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS .tmp/ai-infra-skills
git -C .tmp/ai-infra-skills checkout 6238045
```

Archive: `git -C $W rev-parse HEAD > $OUT/head.txt; git -C $W diff --stat > $OUT/patch_stat.txt;
nvidia-smi > $OUT/gpus_before.txt`.

## 1. Boot

```bash
cd $W
setsid bash -c "echo \$\$ > $OUT/server.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$W \
  python -u -m sglang_omni.cli serve --model-path $MODEL --port $PORT" > $OUT/serve.log 2>&1 &
PYTHONPATH=$W python -c "import sglang_omni; print(sglang_omni.__file__)" > $OUT/import_path.txt
```

Poll `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health` every 10 s
until 200 (fail after 15 min, send the last 80 lines of serve.log). `import_path.txt`
must be under `$W`.

## 2. Captures, in this order, one at a time

Each from `$W` with `PYTHONPATH=$W python $S/profile_workloads.py --url http://127.0.0.1:$PORT
--model $MODEL --out $OUT` plus:

| # | args |
| --- | --- |
| a | `--kind decode --batch 16 --steps 40 --label uncaptured1 --no-capture` |
| b | `--kind decode --batch 16 --steps 40 --label formal` |
| c | `--kind decode --batch 16 --steps 40 --label uncaptured2 --no-capture` |
| d | `--kind decode --batch 1 --steps 40 --label formal` |
| e | `--kind prefill --batch 1 --steps 10 --label formal` |
| f | `--kind prefill --batch 16 --steps 8 --label formal` |
| g | `--kind decode --batch 16 --steps 20 --label mapping --with-stack` |

Save each run's stdout line to `$OUT/driver_<#>.json`. After b, grep serve.log for
`Torch profiler armed for 40 forwards of stage tts_engine` (the branch-only marker; send
the line). If a run fails, send its traceback and the last 80 lines of serve.log, and
continue with the next.

## 3. Analysis on the box

For every trace file `T` under `$OUT/{formal,mapping}/*/b*/`:

```bash
python $S/step_ledger.py $T --top 30 > <T dir>/ledger.txt 2>&1
python .tmp/ai-infra-skills/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py \
  --framework sglang --input $T > <T dir>/triage.txt 2>&1
```

And the two-trace pair:

```bash
python .tmp/ai-infra-skills/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py \
  --framework sglang --mapping-input <mapping decode b16 trace> --formal-input <formal decode b16 trace> \
  > $OUT/triage_pair_decode_b16.txt 2>&1
```

## 4. Teardown

`kill -TERM -- -$(cat $OUT/server.pgid)`, wait 10 s, `kill -KILL -- -<pgid>` only if the
group is still alive. `nvidia-smi > $OUT/gpus_after.txt`; the card must be back near 0 MiB.

## 5. Return

Verbatim: head.txt, import_path.txt, the armed marker line, every driver json line,
every ledger.txt, every triage.txt, triage_pair_decode_b16.txt, trace file sizes
(`ls -la`), serve.log lines with `ERROR` or `Traceback` (with 20 lines of context), and
the `client_ms_per_frame` of a, b, c. No interpretation.
