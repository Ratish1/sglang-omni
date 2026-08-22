# Loop profile: GPU-idle attribution for the OmniScheduler

Question: on main, in the regime the 2026-08-20 throughput numbers were
measured in, how much GPU time is idle and what is the scheduler thread
doing at the start of each idle gap? Plan, label set, analysis definitions
and the decision rules fixed before the data: tasks/loop_profile_plan_20260822.md
(on the Mac). This file is the runbook for the H100 box.

Contents:

- `nvtx_probe/omni_nvtx_probe.py`, `nvtx_probe/sitecustomize.py`: the
  runtime NVTX wrappers, loaded into every stage process when
  `OMNI_NVTX_PROBE=1` and this directory is on PYTHONPATH. No source edit on
  any arm.
- `nsys_gap_attribution.py`: sqlite export in, the attribution tables out.
- `checks/`: the two local self-checks (no GPU needed).

## 0. One-time setup on the box

```bash
git fetch origin loop-profile
git worktree add ../omni-profile origin/loop-profile    # or checkout anywhere
export PROBE_DIR=$(realpath ../omni-profile/profile/nvtx_probe)
export ATTR=$(realpath ../omni-profile/profile/nsys_gap_attribution.py)
export OUT=$PWD/tmp/loop-profile-20260822 && mkdir -p $OUT
```

Self-checks (optional, 10 s, from any arm checkout root with the server venv):

```bash
OMNI_NVTX_PROBE=1 PYTHONPATH=$PROBE_DIR:. python ../omni-profile/profile/checks/check_probe_local.py
python ../omni-profile/profile/checks/check_attribution_local.py
```

Both print `... CHECK PASSED`. On a CUDA box the first one reports
`backend=nvtx` and skips the recorder-only assertions; the line that
matters is `installed=22 missing=none`. Its idle-iteration part needs the
repo's unit-test constructor to import; when it does not, it prints
SKIPPED with the reason and the check still passes on the rest.

## 1. Arms and fixed factors

| Arm | Checkout | Server flag added | Isolates |
|---|---|---|---|
| A | upstream/main `2494125c9` | none | the shipped loop |
| A0 | same | `--prefill-coalesce-requests 1` | the loop with the hold-off removed |
| B | `prefill-launch-first` `10f7f2570` | none | the launch-first loop |

Fixed across arms (do not vary; record the launch line in the result):

- Model and revisions as on 2026-08-20: Qwen3-ASR
  `7278e1e70fe206f11671096ffdd38061171dd6e5`, SeedTTS EN
  `27f4c1adee83b5b29b7c4b375f6b976324bda308`, 1088 clips, `--lang en`,
  non-streaming (same mode as the throughput cells).
- Server defaults otherwise: cuda graphs on, prefill cuda graph backend
  default, `--mem-fraction-static` as used on 2026-08-20, tp 1, pre-LM
  encoder on with its defaults (`pre_lm_max_batch_size 8`,
  `pre_lm_max_batch_wait_ms 0`, encoder cuda graph on), coalesce defaults
  from qwen3_asr config (16 requests / 40 ms, when_idle, requires pending
  builds, after builds during decode), async lookahead min batch size 2,
  one request-build worker. None of these is passed explicitly; passing
  any of them is a new arm and must be reported as such.
- GPU: the same physical H100 for every cell; nothing else on it.
- Client: the repo benchmark, three measured repeats, one discarded warmup
  run done as a separate invocation (section 2 step 3) so the capture holds
  only measured passes.
- nsys: `--trace=cuda,nvtx --cuda-graph-trace=graph --sample=none
  --cpuctxsw=none`. Graph mode records one row per graph replay (low
  overhead); the analysis reads both the graph and kernel tables and prints
  their row counts. If `GRAPH_TRACE` shows 0 rows on a cell where cuda
  graphs are on, rerun that cell with `--cuda-graph-trace=node`.

Why whole passes instead of a 20 s steady-state window: at c32 one pass of
1088 clips takes about 3 s (350 req/s), so a fixed window would span ramps
anyway, and the 2026-08-20 req/s were measured over exactly these passes.
The capture therefore spans the three measured repeats and the analysis
uses the whole window (`--trim 0`). The inter-repeat idle lands in the
`sched:sleep` and `sched:idle_check` buckets, which the decision rules
treat as "not a loop cost"; everything else is the measured regime.

## 2. Per arm (three times: A, A0, B)

1. Checkout and environment, in the server shell:

```bash
git checkout <arm checkout>            # A and A0: 2494125c9; B: 10f7f2570
export OMNI_NVTX_PROBE=1
export PYTHONPATH=$PROBE_DIR${PYTHONPATH:+:$PYTHONPATH}
ARM=A   # or A0 or B
```

2. Launch the server under nsys (one server per arm; A0 is its own server
   because the flag is a launch argument):

```bash
nsys launch --trace=cuda,nvtx --cuda-graph-trace=graph --sample=none --cpuctxsw=none -- \
  python -m sglang_omni.cli serve \
    --model-path "$MODEL_PATH" --model-name Qwen/Qwen3-ASR-1.7B --port 8000 \
    <the same extra flags as the 2026-08-20 Qwen3-ASR cells, for example --mem-fraction-static> \
    $( [ "$ARM" = A0 ] && echo --prefill-coalesce-requests 1 ) \
  2>&1 | tee $OUT/${ARM}_server.log
```

   Confirm in the log, once per stage process:
   `[nvtx-probe] backend=nvtx pid=... installed=22 missing=none`.
   `missing` must be `none`; if it is not, stop and report the line.

3. Warmup, not captured (one pass at c32 is enough to settle graphs and the
   encoder cache):

```bash
python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 \
  --concurrencies 32 --repeats 1 --max-samples 0 --lang en \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --output $OUT/${ARM}_warmup.json
```

4. Capture, one per concurrency, c32 then c8:

```bash
for C in 32 8; do
  nsys start -o $OUT/${ARM}_qwen3asr_c${C}
  python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 \
    --concurrencies $C --repeats 3 --max-samples 0 --lang en \
    --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
    --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
    --output $OUT/${ARM}_qwen3asr_c${C}.json
  nsys stop
done
```

   Expected capture lengths: c32 about 10 to 15 s, c8 about 30 s. Record
   the client's req/s for the log only; under nsys it is not a gate.

5. Export and analyze:

```bash
for C in 32 8; do
  R=$OUT/${ARM}_qwen3asr_c${C}
  nsys export --type sqlite --force-overwrite true -o $R.sqlite $R.nsys-rep
  python $ATTR $R.sqlite --json $R.attr.json > $R.attr.md
done
```

6. Stop the server before the next arm.

Total: three servers, one warmup each, six captures, six `.attr.md` files.
Optional seventh: ArkASR c16 on arm A (eager prefill shape) with the same
steps and the ArkASR model flags from 2026-08-20.

## 3. Sanity checks before reading the numbers

- First line of each `.attr.md`: `scheduler threads 1, encoder threads 1`
  and the GPU row counts. `GRAPH_TRACE` rows must be non-zero on a graph
  mode capture (see section 1); `KERNEL` rows carry the eager work.
- The script exits with a message if no `sched:recv` range exists (probe
  not live in the LM stage) or no NVTX table exists (trace flag missing).
- `unlabeled` host time should be a small share of the window. A large
  share means a host phase is missing from the label set; report it with
  the host table rather than interpreting around it.
- A0 must show `0 hold marks`; A and B must show a non-zero count at c32.
- Window length should match the client run length to within a second.

## 4. Report

Keep per cell: `.nsys-rep`, `.sqlite`, `.attr.json`, `.attr.md`, the
client `.json`, the server log. Send back the six `.attr.md` files, the
six client req/s lines, and the `[nvtx-probe]` lines from the three server
logs. The Mac side writes tasks/loop_profile_results_20260822.md and takes
the decision from plan section 6.
