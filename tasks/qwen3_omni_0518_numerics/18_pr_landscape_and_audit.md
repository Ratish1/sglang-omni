# 18. The open PRs against the ledger findings, and the audit of what is already built

Read on 2026-09-02 against sgl-project/sglang-omni: 375 open PRs, the
trackers #1022, #1018, #1754, #1418, #1233, #1324, #1357, #1707, and the
new profiling methodology #1798 with its per model issues (#1914, #1887,
#1886, #1885). Every PR named below was read as its full diff plus body
and review thread. Code claims carry a file and line on the tree named.

## 1. The base drift, first

perf/step-ledger is based on 216e946dd. Upstream main is at 15c4568bb, 26
commits later, and these touch the paths the two runs measured:

| commit | what it changes | finding it touches |
|---|---|---|
| 86e06cca3 #1641 | fuses the Qwen3-TTS predictor kernels and its seeded sampling | the 5.6 ms predictor chain (doc 17 section 5.1) was measured without it |
| 931c612fb #1665 | fuses the 32 MOSS-TTS Delay audio heads into one stacked GEMM | the 33 eager head calls of doc 17 section 5.3 are gone on main |
| f148e4ae8 #1666 | fuses the MOSS-TTS-Local seeded frame sampler into one Triton kernel | the frame graph's kernel count |
| 4ecfde1e1 #1616 | per stage absolute KV byte budgets | the talker and thinker pool sizes behind the admission arithmetic |
| d6670de47 #1581 | breakable prefill graph support for Qwen3-TTS | Qwen3-TTS prefill rows |
| d346c46ff #1649 | removes the Qwen3-TTS talker compile path | Qwen3-TTS decode |
| 7d1bb690c #1756, e63a257c8 #1757, 4a464db1c #1848 | incremental and stateful codec streaming, chunk ramp | the Qwen3-TTS vocoder stage |
| e7d876b28 #1762 | configurable ASR audio chunking | Qwen3-ASR request build |

The Qwen3-TTS predictor graph is still captured lazily on main
(`sglang_model.py:125`, `:147`, `:1315-1329` on 15c4568bb), so that
finding stands. The predictor chain cost, the MOSS-TTS Delay head fan and
the MOSS-TTS-Local frame graph must be measured again on main before any
of them is worked. Run 3 of doc 15 runs on perf/step-ledger rebased onto
15c4568bb, not on 7b1bee5c4.

## 2. Coverage of each finding by open work

| finding (doc 16 and 17) | open or merged work | verdict | what stays ours |
|---|---|---|---|
| talker admission bound, 6 rows against 32, queue wait 870 ms at c16 | #1910 (ours) raises the pool 2.4x by fixing the layer count override, the only PR moving a term of the budget. Cap moves from 6 to about 17 rows, still under 32. #1702 clamps the same pool by device headroom, unconditional and unmeasured on the colocated profile. #1565 and #1563 touch the symptom, not the budget | complementary | the reservation itself: 4096 `talker_max_new_tokens` against about 43 frames, untouched by any PR. T15 stands, with #1910 as its baseline |
| one outbox message per talker row per step | #1346 coalesces frames into per chunk messages, 10x fewer by construction, right seam, conflicts with #1204 and #1320 and with main since 2026-08-15 | covered | nothing, help #1346 rebase |
| talker `prepare_decode_buffers` slow path, 1023 entry walk and staging sync | #1320 rekeys the fast path (draft, held by its author on a TTFA regression), #1409 removes two other syncs | not covered | T23 |
| code2wav final window eager at 60 to 108 ms | #1758 makes it visible per shape. #1862 names the mechanism and is a self declared negative result. #1677 and #1346 can add a second eager window at chunk 0 | not covered | T21 |
| Qwen3-TTS predictor graph captured lazily inside serving | nothing, open or merged. The only startup hook is the vocoder warmup (`stages.py:217`). #1855 has the right capture with rollback pattern, on the codec | not covered | T20 |
| Qwen3-TTS predictor chain 5.6 ms per step | #1641 merged, unmeasured by us. #1790 FP8 retracted by its tracker | remeasure | T22 after run 3 |
| Qwen3-TTS vocoder bound past c32 | #1912 widens vocoder graph coverage, the tracker credits it with r10 parity, but captures every shape twice into two pools. #1853 prunes nothing at shipped defaults. #1846 and #1855 build on a stateful path the PR's own data measures 3.6x slower. #1794 fuses one activation | partially covered | the stage capacity question (GPU fraction, workers), untouched |
| Qwen3-TTS prefill runs the predictor on a cache hit | nothing | not covered | small, folds into T20 |
| one token cache hit eager on Higgs and Qwen3-ASR | #1915 and #1916 fix the bucket ordering bug and are mutually exclusive, neither lowers the ladder below 4. #1614 (stale draft) lowers it to 1 and 2 in the shared ladder | not covered | T19 as a one line change to the shared ladder, rebased on whichever of #1915 or #1916 lands |
| MOSS-TTS Delay 33 eager heads | #1665 merged | done on main | remeasure |
| MOSS-TTS Delay 33 feedback embeddings | #1792 captures them in the sampling graph, right seam, no CI run yet. It makes the per row clone loop load bearing (its feedback tensor is a view of the static buffer) | covered | nothing |
| MOSS-TTS Delay two list uploads and per row clones | nothing | not covered | T24 |
| MOSS-TTS-Local two list uploads, 55 launch hash | #1136 removes one upload only above 12 rows behind an env var and a process global beacon. #762 folds the hash into the backbone graph, default off, author reports no gain, stale since June | not covered at c1 | T24 |
| MOSS-TD request build 16.9 ms at c1 | nothing. #1865 adds host work per token | not covered | T17 |
| Qwen3-ASR request build 5.6 ms | #1681 moves the mel FFT to the encoder stream, right seam through the existing `stage_host_copy` hook, the only PR here with green GPU CI. Decode, resample and fingerprint stay on the CPU | mostly covered | the coalesce hold |
| Qwen3-ASR 40 ms coalesce hold, 7 to 14 ms of queue wait | nothing. #1480 sets ARK-ASR's to 32 ms | not covered | T12 |
| Qwen3-ASR 39 percent eager prefills | nothing | not covered | T25 |
| the 1 ms idle sleep, 10 to 116 per request gap at c1 | #1809 blocks on the inbox with a 20 ms timeout, right seam, one open TP objection (followers would wait the full timeout before the broadcast) | covered | nothing, review #1809 |
| the step ledger itself | #1106 is a per phase wall share timer with no device axis, no per step rows, no percentiles. #1798 and #1850 are a methodology (GPU busy ratio, py-spy, sweeps, A/B) | complementary | the ledger is the per step instrument that methodology lacks |

## 3. Approach notes on the PRs we would comment on

- **#1910.** The line applies to all eleven entries of the arch map
  (`model_worker.py:48-60`), the test covers two. It assigns rather than
  taking the larger value, and sglang sets `num_attention_layers` above
  `num_hidden_layers` for three architectures (v0.5.18
  `model_config.py:1040-1056`), none in the map today.
- **#1702.** An unconditional `min()` against the aggregate device
  fraction (0.94 on the colocated profile), boot order dependent, deletes
  two profile helpers while leaving their tests, no benchmark.
- **#1565.** The gate sleeps before the fatal error raise, the duplicate
  id check and the in flight cap check in `_submit_request`, so the cap
  reads a stale count for the whole stagger window. A knob on a symptom.
- **#1915 against #1916.** Same bug, same function name with different
  signatures. #1915 writes once through `override_server_args` and deletes
  the dead helpers but adds `context_length` as a cap for every model
  against the maintainer's stated direction on #1528. #1916 pops the
  bucket list out of the overrides so sglang plans graph memory against a
  ladder omni later shrinks, double writes, and bypasses the write guard
  with `object.__setattr__`.
- **#1912.** Hands both graph holders the same frame set into two private
  pools, so every shape captures twice. That is its own +46 s of startup.
- **#1907.** Fabricates `[3, T]` mrope positions for a model with no
  multimodal input so an upstream slot refreshes, deletes a corruption
  guard whose trigger is unknown, fails the unit test on an unguarded
  attribute and the non streaming similarity gate.
- **#1846 and #1855.** Correct lifecycle machinery (a reviewer traced it)
  on a path the PR's own data measures at 25.8 percent r10 success against
  94.2 for the default. #1855 adds 676 lines of runner on top and has never
  run Omni CI.
- **#1136.** Couples an AR sync removal to a vocoder graph behind one env
  var, keeps a process global beacon beside the per batch flag it was
  meant to replace, and never fires below 12 rows.
- **#762.** A better state model shipped default off with the author
  conceding no measured win, plus an unrelated processor loading change.
- **#1772.** Flips six radix defaults on evidence that predates #1770,
  with a claim about Whisper the repo contradicts. Changes requested.
- **#1809.** Right mechanism. The env var has no CLI or validation, and
  the block is rank unconditional while only the entry rank receives on
  its inbox under TP.
- **#1454 and #832.** Already on main (#1509, the request build executor).

## 4. Audit of the optimizations already in the tree

Six optimizations the runs leaned on, read whole for semantic correctness
on 216e946dd, with the merged state on 15c4568bb noted where it differs.
Each defect names its trigger and its consequence.

### 4.1 Qwen3-TTS predictor graph cache (`qwen3_tts/sglang_model.py`)

What is right: per row temperature, top p, top k and seed are read at
replay from six persistent device buffers refreshed in place
(`:481-504`, `:1078-1092`, read at `:1551-1573`), only the four host
branches are baked and all four are in the key (`:1195-1249`), padded rows
compute on zeroed inputs with no cross row reduction and are sliced off
at replay and at publish (`:181`, `model_runner.py:254-256`), and the
feature is off under tp above 1 (`:1258-1266`).

Defects:

- **Every new key stalls the batch for about half a second.** All twelve
  captures of run 2 cost 496 to 565 ms of host wall, whatever the bucket
  and whatever the allocator held, and each ran the chain three times
  (about 4400 tensor allocations against 50 for a normal step). The
  first capture in the process and the one after a prefill burst cost
  more (doc 17 section 4, corrected table). What the flat part is made
  of is measured, not assumed, by the capture phase timing on
  perf/step-ledger (d6425827b) and decided in doc 19. `torch.cuda.graph`
  does synchronize the whole device and empty the caching allocator on
  entry (`torch/cuda/graphs.py:439`, `:449` on v2.13.0, the pinned
  torch), inside the serving step, in a process shared with the
  preprocessing stage. Run 3 measured both at under a millisecond and
  under 50 ms, so they are not the stall, the first warmup pass is
  (doc 19 section 7). `capture_error_mode thread_local` (`:151` on
  15c4568bb) exempts other threads from erroring, it does not keep them
  out.
- **The first warmup pass builds fifteen cuDNN attention plans per
  bucket.** The predictor's `scaled_dot_product_attention` over a cache
  slice of 1 to 16 keys (`:1798`, `:1809-1815`) runs on cuDNN on torch
  2.13 and H100, whose plan cache is keyed by batch and key length, so
  each new bucket builds fifteen plans at 23 to 37 ms each on its first
  eager pass (doc 19 section 9, from the run 4 trace). That is the
  stall. The same code pattern is in the talker predictor and the
  MOSS-TTS attentions (doc 19 section 9.1), and dots_tts and MiniMax
  already opt out of cuDNN attention.
- **The fused kernels are gated on capture state.** The fused gather,
  sampler and addmm are gated on `is_current_stream_capturing()`
  (`:1496`, `:1712`, `:1820` on 15c4568bb), so the warmup passes run the
  eager branches and the fused Triton kernels are first launched inside
  the capture pass. Measured at about 8 ms on the first capture, a
  defect but not the stall.
- **A capture with the cyclic collector running.** A talker and its
  graphs form a reference cycle, and a graph freed by the collector
  during another capture resets itself on a capturing stream and
  invalidates that capture. It surfaced as an order dependent test
  failure on the timing branch, the base has the same exposure (doc 19
  section 9.3). Sglang freezes the collector around its capture loop
  (`base_cuda_graph_runner.py:45-61`).
- **A new stream per capture with a shared pool.** Two fresh streams per
  capture (`:133`, `:145`) against PyTorch's note that captures sharing
  a pool should share the stream (`graphs.py:398-399`) and sglang's one
  stream for every shape, captured from the largest down. Whether the
  pool is reused across our captures is read off the device allocation
  count inside the capture pass (doc 19, H4).
- **The 32 key cap is reachable by a benign mix.** Per request top k enters
  the key through a ten rung ladder (`:43`, `:1054-1065`), so six buckets
  times 23 signatures gives 138 reachable keys against a cap of 32
  (`:39`, `:1300`). Past the cap every uncached key runs eager forever
  with one warning ever, and the fallback counter (`:1301`) is never
  read.
- **Chunked prefill emits a spurious code frame.** The engine builder
  leaves `chunked_prefill_size` to sglang's auto sizing, and `post_prefill`
  runs the predictor on every chunk's last position with no chunk guard
  (`model_runner.py:74-81`, `:257-264`). The talker (`talker_scheduler.py:31`)
  and zonos2 disable chunking for exactly this reason.
- Fixed on main by #1641: the small k Triton sampler's uniform clamp
  diverged from sglang's log clamp at hash 0 (`sampling_kernels.py:72-76`
  on our tree, `_gumbel_from_hash` on 15c4568bb).

### 4.2 Qwen3-Omni talker predictor graph (`qwen3_omni/components/talker.py`)

- **It never serves a decode step.** The decode graph captures the whole
  forward, and inside a capture the guard at `:1464` routes the predictor
  to its eager unroll, so the outer graph holds the 16 sub steps at the
  bucket's rows and the predictor graph's only live caller is prefill
  (`talker_model_runner.py:98`).
- **On that one path it pads to 32 rows.** The serve log shows a single
  capture at batch size 32, which is the `(max_batch_size,)` fallback at
  `:1479-1480`, so a one row prefill runs 512 row sub steps for 16 useful
  ones. This is inside the 18.6 ms prefill forward of doc 17 section 2.
- A second code dtype would capture lazily on the serving thread
  (`:1517-1529`) with a private pool.

### 4.3 Talker decode buffer fast path (`talker.py:1029-1184`)

What is right: the fast path's repetition update at `:1046` is correct in
every case it admits (single writer at `:1290`, replay at fixed
addresses, prefill excluded by `:1252-1253`).

Defects:

- **A constant mask rebuilt per row.** The suppress list is a fixed 1023
  token set of the model (`request_builders.py:1078-1084`), stored per
  request and walked per row on every slow step (`:1113-1122`), about 131
  KB of host built indices and one upload at eight rows, to reproduce one
  device row that could be broadcast.
- **Any batch size change is a full rebuild** (`:1033`), so a request
  replacement in a batch of eight costs two slow steps.
- **The staging event wait is wider than its note says.** `:1127` waits
  on an event recorded after the previous slow step's upload (`:1143`),
  which completes only when everything queued before it on the compute
  stream has, and the event spins.
- The prefill invalidation lives inside the Python forward (`:1253`), so
  a graph replayed prefill would silently bypass it. Unreachable today.

### 4.4 code2wav graph contract (`qwen3_omni/components/code2wav_scheduler.py`, `code2wav_cuda_graph.py`)

- **The tail is forced eager even when its key exists.** `decode_delta`
  passes `graph_eligible=not is_final` (`:340-343`) and the runner returns
  eager on that flag before looking the key up (`code2wav_cuda_graph.py:609-610`).
  The synchronous tail path does not need it: `is_final` already takes the
  non slot branch (`:351`).
- **No key covers the tail anyway.** The window is context plus 1 to 9
  staged frames, 11 to 34, against keys 10, 20, 30, 35.
- **Padding the tail is sound but not with the current trim.** The codec
  forward is stateless and strictly causal (transformers
  `modeling_qwen3_omni_moe.py:3283-3332`, `:3766-3778`), so padding to the
  next key and trimming is identical to within 4.5e-8, but the trailing
  slice at `:344` would then return the padding's audio unless the pad's
  tail is dropped first.
- **The fallback counters are write only after startup.** `stats()` is
  read at `:1033` and `:1047`, before any traffic. #1758 is the fix.
- In the shipped config batching is off, so concurrency 16 runs sixteen
  batch one windows and the batched path is unreachable.

### 4.5 Higgs codec decode graphs (`higgs_tts/audio_codec.py`, `vocoder_scheduler.py`)

- **Seven of 150 graphs are unreachable and the steady state uses one
  shape.** The streaming window is new frames plus the 8 frame overlap
  (`vocoder_scheduler.py:218-240`), which gives 20, 127, then 83 forever
  at the shipped defaults, so frames 144 to 150 never replay and one shape
  carries the run.
- **Exact shapes have no causal reason.** The codec decode is a pure
  function of the code window (transformers Higgs `decode`, `:564-589`),
  which is why the scheduler re decodes an overlap and trims. Bucketed
  padding would be safe.
- **The batched and non streaming paths never use graphs.** Multi item
  buckets and full decodes above 150 frames run eager
  (`audio_codec.py:420-424`, `vocoder_scheduler.py:262-266`, `:361-378`),
  warned once per shape.
- Startup pays 150 warmups and 150 captures, each with a device
  synchronize and an allocator flush.

### 4.6 MOSS-TTS-Local frame graph and hash (`moss_tts_local/`)

What is right: padded rows are inert and discarded (`sglang_model.py:558-574`),
every pool and KV write is indexed by real rows, and seeds and positions
are copied into the static buffers before every replay (`:559-570`).

Defects:

- **The hash is never captured.** `radix_hash.py:63-64` says the channel
  loop unrolls at capture time, but its input is assembled after
  `graph.replay()` returns (`model_runner.py:408-410` against
  `sglang_model.py:573`) and it runs eagerly at `model_runner.py:262` and
  `:471`, about 56 launches per step. What the design buys is the absent
  host copy on the lookahead path, not a capture.
- **Both per step uploads are derivable.** `emit_index_t`
  (`model_runner.py:418`) is always `arange(batch_size)` on a decode step
  because `inflight_middle_chunks` is nonzero only during chunked extends
  (`:554-565`), and the runner already built that arange on the device at
  `:243-247`. `pool_rows` (`state_pool.py:287`) changes only when the batch
  composition changes and is re uploaded every step. A third upload sits
  at `model_runner.py:607`.
- **Graph and eager disagree on a stopping row.** The graph writes the
  assistant slot id into channel 0 of the feedback (`sglang_model.py:424-427`)
  while the eager fallback uses the audio end id (`model_runner.py:402-415`),
  and both land in `pool.feedback_embeds` (`:446-449`). Bounded to a
  discarded overrun frame, but the two paths are not bit identical.
- `buf[batch_size:].fill_(1 if buf.dtype.is_floating_point else 1)` at
  `sglang_model.py:572` is a dead conditional.
- The MOSS-TD engine builder's bucket floor of 1 and 2 was read and is
  correct as written.

## 5. Order

1. Rebase perf/step-ledger onto 15c4568bb, run doc 15 (fresh server per
   point) on a quiet box with MiniMax, and re read sections 5.1 and 5.3 of
   doc 17 against #1641, #1665 and #1666.
2. T20, the Qwen3-TTS predictor graph, now doc 19. The startup warmup
   that was drafted on perf/qwen3-tts-predictor-warmup was dropped: it
   changed when captures happen, not what one costs, and took its
   signature from dataclass defaults instead of the checkpoint's merged
   generation config. The order is now: run 3 with the capture phase
   timing (d6425827b), fix the capture in place on those numbers (the
   graph path flag, one process wide warm pass of the captured kernel
   set, no per bucket warmup, one capture stream), then decide startup
   capture of the merged default signature as a policy on the residual.
   Verdict per doc 19 section 5.
3. T15 on top of #1910: the reservation against the observed output,
   with `schedule_conservativeness` as the control arm.
4. T21, the code2wav final window: drop the `is_final` eager gate, add
   the tail's frame counts to the key set or pad with the pad tail
   dropped before the trim, after #1758 lands so the effect shows in its
   telemetry.
5. T23 on the talker: broadcast the constant suppress mask, key the fast
   path so a batch size change does not rebuild unchanged rows, and give
   the prefill predictor a bucket ladder or run it eager below eight
   rows.
6. T19 as the shared ladder floor, rebased on whichever of #1915 or #1916
   lands.
7. T24 on MOSS-TTS-Local: the two derivable uploads and the eager hash,
   after #1792 lands for Delay.
8. Higgs codec graphs: bucketed capture over the reachable window shapes
   instead of 150 exact ones, with the batched decode path included.
9. Reviews to post: #1910's two precision points, #1912's double capture,
   #1809's TP rank condition, #1915 against #1916, #1758 as the fix for
   the write only counters.

Every runtime change above is validated by one A/B over the model's
entire CI stage set (doc 14 section 6.3).
