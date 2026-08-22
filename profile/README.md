# Loop profile: GPU-idle attribution for the OmniScheduler

Question: on main, in the cold-pass regime that is real ASR serving (each
clip seen once, encoder runs per request, full prefill), how much GPU time is idle and what is the scheduler thread
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
- nsys: `nsys launch --trace=cuda,nvtx --cuda-graph-trace=graph` on the
  server; `--sample=none --cpuctxsw=none` on each `nsys start` (Nsight
  2026.2.1 rejects them on launch). The analysis reads both the graph and
  kernel tables and prints their row counts.

### The cold-pass rule (the 2026-08-22 finding)

Every measured pass must be the first time the server sees each clip.
The pre-LM embedding cache (4096 entries) and the radix cache (keyed by
the audio content hash) both serve a repeated clip, so a pass that
follows any same-corpus pass measures no encoder work and a one-token
prefill per request. That is not ASR serving and it is what every earlier
gate measured.

Protocol: **one fresh server per measured pass, no client warmup** (no
`--warmup`, `--repeats 1`). The server captures its CUDA graphs at
startup, so the only cold-start cost left in the pass is the first few
requests' allocator and JIT work; that is real serving behaviour and it is
inside the window, once per pass, the same on every arm. Repeats are
separate fresh servers. The proof that a pass was cold is in the
attribution output: the extend `tok` median must be in the hundreds, not
equal to `bs`, and the encoder thread count must be 1. A pass failing
either check is discarded, not reported.

## 2. Per cell (arm x concurrency), repeated for each measured pass

One server, one capture, one pass. Cells: c32 and c8 for A, A0 and B, one
pass each first (six servers, six captures); a second pass per cell only
if the first six leave a decision on the margin.

```bash
ARM=A; C=32                               # A0 adds the flag; B checks out 10f7f2570
git checkout <arm checkout>
export OMNI_NVTX_PROBE=1
export PYTHONPATH=$PROBE_DIR${PYTHONPATH:+:$PYTHONPATH}
R=$OUT/${ARM}_qwen3asr_c${C}_p1           # p2, p3 for later passes

nsys launch --trace=cuda,nvtx --cuda-graph-trace=graph -- \
  python -m sglang_omni.cli serve \
    --model-path "$MODEL_PATH" --model-name Qwen/Qwen3-ASR-1.7B --port 8000 \
    <the same extra flags as the 2026-08-20 Qwen3-ASR cells, for example --mem-fraction-static> \
    $( [ "$ARM" = A0 ] && echo --prefill-coalesce-requests 1 ) \
  2>&1 | tee $R.server.log
```

Wait for the server to report ready and confirm once per stage process:
`[nvtx-probe] backend=nvtx pid=... installed=22 missing=none`. `missing`
must be `none`; if not, stop and report the line.

```bash
nsys start --sample=none --cpuctxsw=none -o $R
python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 \
  --concurrencies $C --repeats 1 --max-samples 0 --lang en \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --output $R.client.json
nsys stop
# stop the server now; the next pass gets a fresh one

nsys export --type sqlite --force-overwrite true -o $R.sqlite $R.nsys-rep
python $ATTR $R.sqlite --json $R.attr.json > $R.attr.md
head -3 $R.attr.md; grep "Extend batch shape" $R.attr.md
```

Expected pass lengths in the cold regime are longer than the 3 s per pass
of the cached regime (the encoder runs for every clip); record the
client's req/s for the log only, not as a gate.

Optional seventh cell: ArkASR c16 on arm A with the ArkASR model flags
from 2026-08-20.

## 3. Sanity checks before reading the numbers

- First line of each `.attr.md`: `threads: scheduler 1, encoder 1,
  builder 0 or 1` and the GPU row counts. `GRAPH_TRACE` rows must be
  non-zero on a graph mode capture; `KERNEL` rows carry the eager work.
- Cold-pass proof: `Extend batch shape ... tok median` in the hundreds.
  `tok median` equal to `bs` means a cached pass; discard it.
- The default window is the benchmark pass (first to last exec range), so
  lead-in and shutdown idle are excluded. Its length should match the
  client wall time to within a second.
- `unlabeled` host time should be a small share of the window. A large
  share means a host phase is missing from the label set; report it with
  the host table rather than interpreting around it.
- A0 must show `0 hold marks`; A and B must show a non-zero count at c32.
- The script exits with a message if no `sched:recv` or no `exec:*` range
  exists (probe not live in the LM stage, or nothing ran) or no NVTX
  table exists (trace flag missing).

## 4. Report

Keep per pass: `.nsys-rep`, `.sqlite`, `.attr.json`, `.attr.md`, the
client `.json`, the server log. Send back the `.attr.md` files, the client
req/s lines, and the `[nvtx-probe]` lines from the server logs. The Mac
side writes tasks/loop_profile_results_<date>.md and takes the decision
from plan section 6 plus the bubble rule added on 2026-08-22.
