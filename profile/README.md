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
| B | `launch-first-rebased` (the five launch-first commits on `2494125c9`) | none | the launch-first loop |

`prefill-launch-first` at `10f7f2570` is on base `24e9a3552` and must not
be compared with `2494125c9`: it lacks main's Qwen3-ASR encoder CUDA graph
(#1633), the encoder overlap (#1569) and the pinned cache (#1591).

Shared base, verified before any A/B (added 2026-08-22 after a campaign
was lost to mismatched bases): both arms must descend from the same base
commit, checked with `git merge-base --is-ancestor <base> <arm>` for each
arm, and the base hash goes in the report. A branch cut from an older
main is rebased first; comparing it against current main measures main's
intervening commits, not the branch.

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

## 5. Campaign 3: unprofiled A vs B, and GIL attribution (2026-08-22)

Why: the cold campaign (tasks/loop_profile_cold_results_20260822.md)
found B at 130 req/s vs A at 210 at c32, builder bound (8 workers / 60.9 ms
build = 131), with the builder and encoder threads 3 to 4x slower under B.
Two questions, in order: is the loss real without nsys; and is the
coupling the GIL. Four fresh servers, all Qwen3-ASR c32, cold-pass rule as
in section 1 (no warmup, `--repeats 1`, one pass per server).

Outcome (2026-08-22): run as written with B = `10f7f2570`, which is on an
older base; the A/B numbers were confounded and are retracted
(tasks/loop_profile_campaign3_results_20260822.md). The GIL tables for A
stand. Section 6 is the fair rerun.

`profile/gil/aggregate_pyspy_raw.py` turns a py-spy raw recording into a
per-thread GIL share table. py-spy: `pip install py-spy` in any venv (it
is a static binary) and ptrace permission: run it with sudo, or
`echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope` once.

### 5.1 Clean req/s, A then B (no nsys, no probe, no PYTHONPATH)

```bash
for ARM in A B; do                       # A: 2494125c9, B: 10f7f2570
  git checkout <arm checkout>
  R=$OUT/c3_${ARM}_c32_clean
  python -m sglang_omni.cli serve --model-path "$MODEL_PATH" --model-name Qwen/Qwen3-ASR-1.7B --port 8000 \
    <same extra flags as before> 2>&1 | tee $R.server.log &
  # wait for ready
  python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 --concurrencies 32 --repeats 1 \
    --max-samples 0 --lang en --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
    --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 --output $R.client.json
  # stop the server
done
```

Report: req/s and p50/p95 for A and B. This alone closes step 3 of the
work order: B at or below A retires it; B above A by more than the
single-pass spread reopens the nsys-artefact question.

### 5.2 GIL share per thread, A then B (py-spy, probe env on for the pid line)

The probe env is set only so the server log prints the LM stage pid
(`[nvtx-probe] ... pid=NNN`); without nsys attached the NVTX calls are
no-ops. py-spy samples at 200 Hz with `--gil`, which records a sample only
when some thread holds the GIL.

```bash
for ARM in A B; do
  git checkout <arm checkout>
  export OMNI_NVTX_PROBE=1 PYTHONPATH=$PROBE_DIR${PYTHONPATH:+:$PYTHONPATH}
  R=$OUT/c3_${ARM}_c32_gil
  python -m sglang_omni.cli serve ... 2>&1 | tee $R.server.log &
  # wait for ready; then
  PID=$(grep -o 'nvtx-probe\] backend=nvtx pid=[0-9]*' $R.server.log | tail -1 | grep -o '[0-9]*$')
  DUR=20                                 # c32 pass is about 6 s on A, 9 s on B; 20 s covers it
  sudo py-spy record --pid $PID --gil --threads --format raw --rate 200 --duration $DUR -o $R.gil.raw &
  sleep 1
  python -m benchmarks.eval.benchmark_asr_seedtts --port 8000 --concurrencies 32 --repeats 1 \
    --max-samples 0 --lang en --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
    --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 --output $R.client.json
  wait                                   # py-spy finishes its duration
  python $(dirname $ATTR)/gil/aggregate_pyspy_raw.py $R.gil.raw --rate 200 --duration $DUR > $R.gil.md
  unset OMNI_NVTX_PROBE
  # stop the server
done
```

Because the recording window (20 s) is longer than the pass, the "share
of wall" column is diluted by the idle tail; read the "share of GIL-held
time" column, and the first line's held-by-anyone percentage as a lower
bound. Threads to expect: `scheduler-<stage>`, `omni-request-build_0..7`,
`qwen3-asr-audio-encode`, the asyncio main thread and its default
executor threads.

Report: the two `.gil.md` files. The reading: if `scheduler-*` holds a
much larger share of GIL time on B than on A while the builders' share
shrinks, the coupling is the GIL; if the shares are similar on both arms,
the coupling is not the GIL (GPU stream contention is next).

### 5.3 Optional: switch-interval sweep on A (cheap second method)

`OMNI_SWITCH_INTERVAL=<seconds>` in the server environment (with
`PYTHONPATH=$PROBE_DIR`, probe env unset) sets `sys.setswitchinterval` in
every process. Two extra clean A c32 servers: `0.0005` and `0.05` against
the default `0.005` from 5.1. If req/s moves by more than the single-pass
spread, GIL scheduling is load bearing on A too, and the direction says
whether the scheduler thread or the builders are starved.

## 6. Campaign 4: fair launch-first test (cold, clean)

Arms: A = `2494125c9`; B = `launch-first-rebased` (five commits on
`2494125c9`). Verify before starting:

```bash
git merge-base --is-ancestor 2494125c9 launch-first-rebased && echo base-ok
git log --oneline 2494125c9..launch-first-rebased | wc -l      # 5
test -f sglang_omni/models/qwen3_asr/encoder_cuda_graph.py && echo encoder-graph-ok
```

Cells: c32 and c8, A then B, one fresh server per cell, clean (no nsys,
no probe, no PYTHONPATH), commands as in 5.1. Four servers. Report req/s,
p50, p95, WER per cell.

Reading, fixed in advance: launch-first was measured at +3.4 (c16) and
+4.9 percent (c32) on its own base in the cached regime; in the cold
regime its benefit is the extend resolve moved off the critical path
(13 ms total over 122 extends in the cold capture), so the expected gain
is small. B above A by more than about 3 percent at c32 with c8 not
worse: step 3 reopens for the three reads. Anything less: step 3 closes
as no measurable gain in the real regime and the branch is archived.

## 7. Campaign 5: inside the decode step and the extend (main, cold, c32 and c8)

Why: on main the GPU idle at c32 is 62 percent decode path and 25 percent
extend path, and the scheduler thread spends about 70 percent of its
time in C++ dispatch with the GIL released (campaign 3). The question is
which phase issues the launches: 56 eager kernels per decode step and
about 182 launch bubbles per extend. Findings are acted on at omni-owned
seams (sglang is a pinned dependency), so every phase below is labeled by
side: `run:` and `bridge:` are omni, `sgl:` is upstream, `sched:` is the
loop.

The probe now carries 45 wrappers (22 from before plus the launch
internals). New labels, all nested inside `exec:launch:decode` /
`exec:sync:extend`:

| side | label | wraps |
|---|---|---|
| omni | run:before_decode, run:before_prefill | runner hooks |
| omni | run:fwd_call | ModelWorker.forward_batch_generation |
| sgl | sgl:forward | ModelRunner.forward |
| sgl | sgl:decode_can_run, sgl:decode_load_batch, sgl:decode_graph | DecodeCudaGraphRunner can_run_graph, load_batch, execute |
| sgl | sgl:prefill_can_run, sgl:prefill_load_batch, sgl:prefill_graph | PrefillCudaGraphRunner same |
| sgl | sgl:attn_meta | the concrete attention backend's init_forward_metadata (wrapped on first forward; the log prints which backend) |
| omni | run:sample, run:rep_penalty, run:suppress, run:seeds | omni's sampler wrapper and its three pre-steps |
| sgl | sgl:sample | ModelRunner.sample |
| omni | run:post_launch, run:post_decode, run:post_prefill | post hooks (default post_decode_launch does the pinned copy) |
| omni | run:publish, bridge:publish, bridge:record_event | FutureMap stash and the completion event |
| sched | sched:batch_copy | ScheduleBatch.copy |
| omni | run:resolve | execute_resolve |

The analysis gains one table: CUDA launches by innermost label, from the
RUNTIME correlation rows: API calls, API host ms, GPU rows launched by
kind, GPU ms of those rows. Divide a decode-path row by the window's
decode steps (printed in the title) for the per-step view.

Cells: A = `2494125c9` only, c32 and c8, one cold pass each, two servers.
Same commands as section 2 with one change: `--cuda-graph-trace=node` on
`nsys launch`, so the kernels inside a graph replay are rows and the
per-phase kernel counts are complete. Expect a larger `.nsys-rep`.

```bash
nsys launch --trace=cuda,nvtx --cuda-graph-trace=node -- python -m sglang_omni.cli serve ...
nsys start --sample=none --cpuctxsw=none -o $R
...
python $ATTR $R.sqlite --json $R.attr.json > $R.attr.md
```

Confirm in the server log: `installed=45 missing=none` and one line
`[nvtx-probe] attention backend wrapped: <Backend>`.

Report: the two `.attr.md` files. Reading: the launch table answers, per
phase, how many API calls and how many GPU rows one decode step and one
extend issue, and the micro-gap table says where the bubbles sit. Each
omni-side row with a large count is a candidate seam; each sgl-side row
is a configuration question against the pinned sglang, not a patch.

## 8. Campaign 6: two omni-side configuration levers (cold, clean, A vs A+lever)

From campaign 5 (tasks/loop_profile_campaign5_results_20260823.md): the
decode graph replays 369 kernels per step at 5.7 to 6.3 us each, about
twice the bandwidth floor, because Qwen3-ASR compiles only buckets at or
below `torch_compile_max_bs: 2`; the extend issues 612 CUDA API calls per
extend in the breakable prefill graph. Both are omni-owned configuration.

Both levers are CLI overrides on the same checkout (A = `2494125c9`), so
the arms share a base by construction. Verify each override reaches the
stage by reading the engine builder's startup log line (it prints
torch_compile and the prefill backend).

| Arm | Override | Isolates |
|---|---|---|
| A | none | baseline |
| A+E1 | `--torch-compile-max-bs 32` | compiled decode graphs up to bs 32 |
| A+E2 | `--cuda-graph-backend-prefill tc_piecewise` | piecewise-compiled prefill graph |
| A+E1+E2 | both | the pair |

If a flag is not accepted by `sglang_omni.cli serve`, pass it as a
stage override the way `--prefill-coalesce-requests` is, and record the
exact form used. Cells: c32 and c8 per arm, clean (no nsys), one cold
pass per fresh server: eight servers. Record startup time to ready for
every server (E1 adds compile time per bucket; that is part of the
result). Report req/s, p50, p95, WER per cell.

Then one profiled cell per winning arm at c32 (section 7 commands, node
trace) so the launch table shows the new kernels per step and API calls
per extend; that is the proof the lever did what the budget said.

Reading, fixed in advance: E1 is expected to move c8 more than c32 (c8
is GPU bound per step, c32 is host bound); E2 is expected to cut the
extend's host time at both. A lever that does not change its own row in
the launch table did not engage, whatever req/s says.
