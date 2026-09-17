# Stage 2: the gates before the hop prefix K/V cache

Plan: `../plans/11_hop_prefix_cache.md`. This directory holds slice 1.0, the G0
numerics gate, which runs before any runtime code is written.

## Result, 2026-09-16, H100 80GB

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

## Result, 2026-09-16, RTX 4090 D

Run `g0-4090-20260916T193520Z`, card 5, main `27a8293c`, same margin and defaults.

**Both gates fail on this card and the cache is not the reason.** Mel: cached
-1.40 dB at the minimum and -1.11 dB at the median against production, where the
H100 gave -0.21 and +0.15. Production itself barely moved between the two cards
(minimum 23.78 to 24.80, median 36.38 to 36.10), so it is the cached path that
diverges here.

The cause is now isolated, and it is not the cache. Swapping production's
attention into the cached path makes the two **bit identical**: 36.13 against
36.13 dB and 23.68 against 23.68 dB, delta +0.00 on every row. So the conv
tails, the RoPE positions, the slicing, the packing, the CFG twins and the Euler
update in the cached path all reproduce production exactly, and the only thing
that differs is the attention kernel. Put paged FA3 back and the same two rows
move to +1.23 and +0.14 dB, better than production, not worse.

The two kernels are equally accurate against a float32 truth, 53.3 to 53.5 dB
each at every shape this run forms, with a measured difference of -0.0 dB. What
they do not share is the per sample rounding, and ten Euler steps over 22 blocks
amplify that into a few dB of jitter in either direction on the final mel. The
H100's 36 hops happened to average +0.03 dB; this card's happened to average
-0.99. Neither is evidence about the cache.

This says the gate is measuring the wrong thing. A cached hop cannot be bit
identical to production, because by design it attends through a different
kernel, so an end to end SNR margin between them is really a jitter budget that
has to be re-estimated per card. The sharper gate is the one this isolation
used: hold the attention fixed and require the cached path to be **bit
identical** to production, which is deterministic and architecture independent,
then qualify the kernel swap separately on its own accuracy against float32,
which E6 and this run both put at 53.3 to 54.0 dB.

Two further measurements, neither implicating the design:

- **FA3 is not the cause.** The kernel the cached path uses scores 53.3 to
  53.6 dB against float32 SDPA at every shape this run forms, from 100 to 2,300
  keys, matching the H100's 53.3 to 54.0 (E6). Chunk alignment held and the
  receptive field is identical to the H100's.
- **The first hop diverges too**, by -0.88 dB on average, and it reads no cached
  state: it computes the whole window from scratch. Caching adds almost nothing
  on top (-1.03 dB on the later hops), which is what pointed at the kernel.

So the H100 verdict stands, and this run is not evidence against the cache. It
is evidence that the gate needs the split above before it is used again.

Cost, for the record and not comparable to the H100's: the hop call falls from
1,228 ms to 254 ms at step 5, 4.8x, against 3.7x on the H100 at the same point.
The cache looks better here exactly as predicted, because this card is
relatively more device bound.

## Correctness gate, 2026-09-16, RTX 4090 D

Run `g0-4090-pooled-sdpa-20260916T201958Z`, `--attention pooled-sdpa`, 4 streams,
4 steps, main `27a8293c`. The pool is written and read exactly as the shipping
path does; only the kernel reading it is production's SDPA instead of FA3.

**It passes: cached is +0.01 dB from production at the median, +0.33 dB at the
minimum, over 10 hops.** Mean per hop is -0.018 dB, and the six hops that
actually read cached K/V average +0.468 dB, so reading the cache carries no
systematic penalty.

What this gate does and does not prove, stated precisely because the
distinction is the whole point:

- It is **not** bit exact, and cannot be. Production attends over the padded
  batch of all rows at once; the cached path attends over the new frames of one
  row at a time, so even with identical mathematics the batch dimension differs
  and the kernel tiles differently. The residual is +-1 to 2 dB per hop,
  stdev 1.19, with a mean of zero.
- Exactness **is** reachable on the first hop, where the cached path's rows are
  production's rows, and there it was checked directly: swapping production's
  own batched `RowAttention` into the cached path gives 36.13 against 36.13 dB
  and 23.68 against 23.68 dB, delta +0.00 on every row. Bit identical.

So the correctness case for the cache is four measurements, not one: bit
identity on the first hop with the attention held fixed; no systematic bias on
the hops that read the cache; FA3 qualified on its own at 53.3 to 54.0 dB on
both Hopper and Ada; and E5's float64 exactness of the cached hop across the
growth schedule on the H100.

Cost here is not meaningful: `pooled-sdpa` loops SDPA per row per layer, 220
times 2R calls, so the cached column is slower than production by construction.

## G1, slice 1.1, 2026-09-17, RTX 4090 D

`g1_stream_identity.sh` boots one arm and generates the first 16 English SeedTTS
samples at c1, streaming, seed 1234, warmup 1, on a local checkpoint.
`g1_compare_audio.py` compares the emitted WAVs byte for byte.
Arms: main `27a8293c` and slice 1.1 `c652aa5e`. Report `g1.json`.

**It passes: 14 of 14 gated samples byte identical, 0 differing.** The other two
of the sixteen are excluded by the control and are not a slice 1.1 effect, see
below. Run on card 5 the two arms are in fact identical on all 16.

Four boots, because the first comparison said 14 of 16 and the answer to "which
14" decides whether the gate means anything:

| pair | identical |
|---|---|
| main card 4 against slice card 5 | 14 of 16 |
| main card 4 against main card 5 | 14 of 16, the same two, the same hashes |
| main card 4 against main card 4, a second boot | 14 of 16, the same two |
| main card 5 against slice card 5 | 16 of 16 |

So two of the sixteen requests are not reproducible from one boot to the next on
this box, on the same card and the same revision, and the arms agree on
everything else. Both unstable samples are the two that use reference clip
`common_voice_en_1205005`; the other six references that appear twice are stable
in both of their samples. They diverge from their first audio sample, not their
last, and their generated lengths differ by 0.4 s, so the AR took a different
token at or near the first step, not a different stop decision. The seed is
applied: fourteen samples reproduce exactly across four boots, which unseeded
sampling would not do. `--tts_engine.factory.onnx_intra_op_threads 1` does not
change it, so it is not thread ordering in the ONNX preprocessing.

The gate is therefore stated against a control: a second run of the baseline
revision defines which samples are reproducible, and identity is demanded on
those. That is the sound form of the gate, and it is the form the script
implements. Chasing the last two is a separate question, filed for the reference
audio path, and it is not a blocker for slice 1.1.

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

From the Mac, one command:

```bash
ssh moss 'docker exec -i sglang-omni-ratish bash -s' < stage2/run_box.sh
```

It adds two detached worktrees off the container's own repo, one for the
revision under test and one for this branch, uses the repo's `.venv`, takes a
card nobody is on from `CARDS`, and writes to `.tmp/out/<script>-<stamp>/`.
There is one run and it is the real one: the script asserts its own shapes,
boundaries and outputs as it goes and raises rather than returning a result that
cannot be trusted. `REV`, `CARDS`, `SCRIPT` and `ARGS` override the defaults.

Nothing is patched. The tree under test is upstream main as published, and the
container's checkout stays on whatever branch it was on.

## The moss box

8x RTX 4090 D, sm89, 24,564 MiB each, shared with other users. Same software as
the H100: torch 2.13.0+cu130, sglang 0.5.19, sglang-kernel 0.4.6.post1.

What had to be true before any of this ran, each checked rather than assumed:

- **FA3 works on sm89.** sgl-kernel builds it for sm80/86/89/90a and
  `is_fa3_supported` accepts capability major 8. The paged configuration plan
  1.2 needs, page size 1, one query segment per (row, chunk), `cache_seqlens`
  at the chunk end, `causal=False`, scores 53.5 dB against float32 SDPA here,
  against 53.3 to 54.0 dB on the H100 (E6). The design survives the move.
- **The code under test is the same.** `packed_dit.py`, `streaming_vocoder.py`
  and `streaming.py` are byte identical between the H100 baseline `cc85ddaa9`
  and main `27a8293c`. In `stages.py` the only change inside the range these
  scripts import is 28 lines of underscore renames from the Ruff commit
  (`_patch_chunk_mask` to `patch_chunk_mask`), which `stage0/common.py` follows.
- **CosyVoice is a separate clone**, `/workspace/CosyVoice` at `074ca6dc`, the
  revision the plan's line anchors were read against, with its Matcha submodule.
  `matcha.utils.__init__` pulls lightning, gdown and matplotlib before
  `matcha.utils.audio`, so those are in the venv.
- **The container exports a SOCKS and an HTTP proxy.** httpx prefers SOCKS and
  then wants socksio, which is absent; `run_box.sh` unsets `ALL_PROXY`. The
  proxy is shared and congested, so the first checkpoint fetch is slow and
  parallel workers stall; `max_workers=1` with retries gets through.
- Cards report 1 MiB when idle, so a free card is one with no compute process,
  not one at 0 MiB.

## What a 4090 number means

Correctness transfers, speed does not. The 4090 has roughly a third of the H100's
memory bandwidth and a quarter of its bfloat16 throughput against a similar host,
so it is relatively more device bound: a cache that removes device work looks
better here than it will in production. Quote a 4090 speedup as an H100 gain and
it is wrong in the optimistic direction.

Amended by the 4090 run above: numerics do not transfer either. Two
implementations of the same arithmetic round about 1 dB apart on Ada and not on
Hopper, so a G0 threshold is per card. Byte identity still transfers, because it
compares one implementation against itself on one card.

| transfers | does not transfer |
|---|---|
| correctness gates that compare a path against itself: G1 byte identity | every wall time, the launch floor, the hop speedup |
| launch and sync counts, which are architecture independent | busy over wall, so the device bound against launch bound ranking, and any SNR threshold |
| WER and SIM as paired deltas between arms on the same box | the SM Issue target at c16 and RTF p99 below 1 |
| whether the 1.2 fallback path is correct | P3, the Flow cache memory budget |

Two consequences. P3 cannot be decided on 24 GB: the H100 sized its AR pool at
60.3 GB and the c16 Flow cache alone wants 17.1 GiB, so they cannot coexist and
c16 may not be reachable. And the 1.2 fallback, a stream releasing its cache and
running today's full window, stops being a rare corner here and becomes the
common case, so it gets real coverage instead of a counter that reads zero.

The H100 stays the machine of record for speed. Every performance table carries
the card it came from, and the existing H100 ledger is not overwritten.

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
