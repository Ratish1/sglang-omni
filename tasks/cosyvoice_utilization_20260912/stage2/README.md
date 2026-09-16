# Stage 2: the gates before the hop prefix K/V cache

Plan: `../plans/11_hop_prefix_cache.md`. This directory holds slice 1.0, the G0
numerics gate, which runs before any runtime code is written.

## Result, 2026-09-16

Run `g0-20260916T061846Z`, H100 80GB, head `cc85ddaa9`, all eight GPUs idle,
margin 1.0 dB, defaults otherwise.

- **Mel gate passes.** Cached minus production is -0.21 dB at the minimum and
  +0.15 dB at the median over 36 hops. Per hop the difference is +0.03 dB on
  average with a 0.90 dB spread, and cached is the closer of the two in 18 of
  36 hops: two unbiased bfloat16 roundings of the same quantity, which is what
  an exact cache should look like. No NaN, no Inf, every length equal.
- **The hop call gets 2.1x cheaper over the schedule**, 1,774 ms to 843 ms, at a
  new over window frame ratio of 0.348. Per step the speedup runs 1.06x at 400
  window frames to 3.70x at 6,850.
- **The cached call is flat**, 138 to 143 ms whether it computes 400 or 1,600
  new frames, because both paths pay the same eager launch floor of about 140 ms
  (stage 1: 19,072 launches, `flow_hop_first_rows1` wall 152.6 ms at busy over
  wall 0.41). The cache removes device work, so it moves the hop from device
  bound back to launch bound and makes hop CUDA graphs (roadmap 2.1) the next
  lever.
- Chunk alignment held on every hop. The pool used 13,700 of 13,700 slots,
  901,120 bytes per cached frame, 11.5 GiB, exactly the plan's arithmetic.
- The conditioning receptive field measured 3 tokens left and 2 right of the
  perturbed token, so a frame depends on tokens j-2 to j+3, confirming the
  window the plan's section 5.3 recomputes.

Two defects the run exposed, both fixed here, neither in the cached path:

- `write_report` shadowed its own `path` parameter with the loop variable, so
  the first run wrote a file called `cached` into the working directory instead
  of `g0.md`. The report for that run was regenerated from its `g0.json`.
- The raw waveform comparison is uninformative and is no longer gated. HiFT is a
  deterministic function of the mel (no module state; with `training=False` and
  `causal=True` both noise sources are fixed tensors), but its excitation phase
  is a cumulative sum of the predicted F0, so a bfloat16 level mel difference
  drifts the phase and the waveform decorrelates in L2. The shipped path itself
  scores -1.6 dB against the float32 truth while its mel scores 36.4 dB, and
  waveform SNR correlates with mel SNR at 0.11. The gate now uses the magnitude
  spectrum of the delta and keeps the raw waveform as a reported diagnostic.

## What G0 answers

`g0_hop_cache_numerics.py` runs every hop of a staggered 8 row schedule three
ways over the same inputs and compares them:

```text
truth       packed full window causal call, float32          flow.inference_causal, no autocast
production  the same call in bfloat16 autocast               vocoder.hop_batch, what main serves
cached      the new frames only, earlier frames' K/V from    solve_flow_euler_packed over a
            an SGLang MHATokenToKVPool read through FA3      PackedDiT subclass
            with a page table, bfloat16 autocast
```

| question | plan | how the run answers it |
|---|---|---|
| is the cached bfloat16 hop good enough to ship | G0, V1 | emitted mel and delta spectrum SNR of cached and of production, both against the float32 truth, per hop and row |
| does FA3 `causal=False` with `cache_seqlens` at the chunk end read exactly [0, chunk end) | V3 | the same comparison: a wrong key range shows up as a collapsed SNR on the rows whose chunk boundary moved |
| does the ragged causal conv position embedding with tails equal the full conv | V4 | same, and the per step tails are the only conv state the cached path carries |
| how much of a hop is new work | section 1 | new frames against window frames per step, with the wall of each path |
| what the cache costs per frame | P3, V2 | bytes per frame measured off the real pool, and the slots the schedule holds at the end |
| how much token context the cached conditioning needs | section 5.3 | receptive field probe: one token is perturbed and the mel frames that move are reported |

The cached path drives production's own solver, DiT forward, CFG combination and
Euler update. Only attention, the conv position embedding and the RoPE positions
are replaced, so the module order cannot drift and `flow_time` keeps production's
dtype, which the stage 0 E5 cached rows did not (E5 used float32 there, so its
17.6 to 43.1 dB does not answer this question).

Which float32 attention backend PyTorch picks for the truth is not forced: at
float32 the backends differ far below the bfloat16 differences under test.

## The gate

Decision P1 in the plan, section 9. Default `--gate-margin-db 1.0`:

- cached minimum SNR at least production minimum minus the margin,
- cached median SNR at least production median minus the margin,
- no NaN or Inf, and every emitted length equal to truth's.

The mel and the delta's magnitude spectrum are gated; the raw waveform is
reported but not gated, for the reason in the result above. The script prints
the verdict and stores it in `g0.json` under `verdict`. Change the margin on the
command line rather than in the code, and record which value the run used.

## The schedule

Eight SeedTTS EN references, prompts of 40 to 144 speech tokens so no prompt is a
hop multiple and the serving `pad_flow_prompt_to_hop` padding runs. Rows join
staggered over the first four steps and then grow their hop through
`next_stream_hop_len`, so one step mixes hop 25, 50 and 100 rows the way
`select_step_participants` does. Generated tokens are real speech tokens, taken
from the other references when one clip is too short.

With the defaults the schedule holds 6,850 mel frames at the end, so 13,700 pool
slots and 11.5 GiB of K/V. `--steps` and `--streams` move both.

## Run

```bash
cd /sgl-workspace/sglang-omni
git fetch https://github.com/sgl-project/sglang-omni.git main
[ -d /sgl-workspace/wt/cosy-main ] || git worktree add --detach /sgl-workspace/wt/cosy-main cc85ddaa9
git -C /sgl-workspace/wt/cosy-main checkout --detach cc85ddaa9
git fetch https://github.com/Ratish1/sglang-omni.git analysis/cosyvoice-utilization-20260912
ANALYSIS=$(git rev-parse FETCH_HEAD)
[ -d /sgl-workspace/wt/cosyvoice-analysis ] || git worktree add --detach /sgl-workspace/wt/cosyvoice-analysis "$ANALYSIS"
git -C /sgl-workspace/wt/cosyvoice-analysis checkout --detach "$ANALYSIS"

S2=/sgl-workspace/wt/cosyvoice-analysis/tasks/cosyvoice_utilization_20260912/stage2
OUT=/sgl-workspace/wt/cosyvoice-analysis/artifacts/cosyvoice/g0-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
cd /sgl-workspace/wt/cosy-main
COSYVOICE_PATH=/sgl-workspace/CosyVoice-utilization-20260912
export PYTHONPATH="/sgl-workspace/wt/cosy-main:$S2/../stage0:$S2:$COSYVOICE_PATH:$COSYVOICE_PATH/third_party/Matcha-TTS"
git rev-parse HEAD > "$OUT/head.txt"
python -c "import sglang_omni, cosyvoice, matcha.utils.audio, sgl_kernel.flash_attn, sglang; print(sglang_omni.__file__); print(cosyvoice.__file__); print(sglang.__version__)" | tee "$OUT/import_path.txt"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > "$OUT/gpus_before.csv"
nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv
CUDA_VISIBLE_DEVICES=0 python "$S2/g0_hop_cache_numerics.py" --device cuda:0 \
  --out "$OUT" 2>&1 | tee "$OUT/g0.log"
```

GPU 0 must list no process: the float32 truth and the pool together need about
20 GiB, and the wall of each path is reported. Outputs: `g0.md` (the tables),
`g0.json` (every field, including the per hop records) and `g0.log`.

The 2026-09-16 run took about a minute of GPU work after the checkpoint load.
`--repeats` sets how many times the two bfloat16 paths are timed; the numerics
do not depend on it. The pool is sized to exactly what the schedule needs, so
raising `--steps` or `--streams` raises the K/V footprint with it.

## Return

```bash
cd /sgl-workspace/wt/cosyvoice-analysis
tar -czf "$(basename "$OUT").tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
```

## Reading the result

- Both gates pass: slice 1.1 (the refactor, gated on byte identity against main)
  and then slice 1.2 (the cache itself) are unblocked, and section 6 of the plan
  records the measured numbers in place of its proposals.
- A gate fails: stop. Section 8 V4 of the plan says to trace per block inside
  this script before any runtime code is written; the per hop records in
  `g0.json` name the step, row and hop where the SNR collapsed.
- Section 5 of `g0.md` is the first measurement of how hop cost follows new
  frames instead of window frames. It replaces the derived 30 percent estimate
  in the plan's section 1. It is not a serving measurement: the cached path here
  lays out its slots and page table in Python and the solve is eager, and this
  schedule caches a larger share of frames (0.348 new over window) than the c16
  ledger does (0.491), so G2 will show a smaller gain.
