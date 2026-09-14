# Qwen3-TTS optimization handbook

State as of 2026-09-14. This is the whole stack, top to bottom: where things are, what
is done, what is open, how a bottleneck is found, how a fix is designed, built, tested,
measured and shipped, and the rules that came out of doing it. Everything here was
done at least once; the doc numbers point at the record.

## 1. Where things are

### Machines and checkouts

- Mac: `/Users/ratish/sglang-omni` is the main checkout, branch `main`. Never run
  pytest here (no torch); `python3 -m py_compile` only. `/Users/ratish/sglang` is the
  shared sglang checkout at the pinned tag; confirm `git describe` says v0.5.19 before
  reading it and never switch it.
- Worktrees under `.worktrees/`: `qwen3-omni-0518-numerics` is the analysis branch
  `analysis/qwen3-omni-0518-numerics` (docs, scripts, patches under `tasks/`, which is
  gitignored, so `git add -f`); one worktree per PR branch, created from `upstream/main`
  with `git worktree add -b <branch> .worktrees/<name> upstream/main`.
- Remotes: `origin` is the fork `Ratish1`, `upstream` is `sgl-project/sglang-omni`.
  PRs are opened with `gh pr create --repo sgl-project/sglang-omni --head Ratish1:<branch>`.
- H100 box: `/sgl-workspace/sglang-omni`, venv `.venv` (python 3.12, torch 2.13,
  transformers 5.12.1, qwen-tts 0.1.1, sglang 0.5.19). Worktrees under `tmp/`:
  `tmp/main` (upstream main, the control), `tmp/an` (the analysis branch, scripts),
  `tmp/<slice>` per PR branch (`tmp/bw` was #2151, `tmp/re` is #2172). GPU 1 for every
  boot; GPUs 6 and 7 usually have tenants. Bench port 31001.
- The venv's editable install points at the main checkout, so the `sgl-omni` console
  script always imports main. Servers start with `python -m sglang_omni.cli serve` from
  the worktree, and every boot archives
  `PYTHONPATH=. python -c "import sglang_omni; print(sglang_omni.__file__)"`. A B boot
  is also gated on a log line only the branch prints. Session 1 of doc 35 was void
  because of exactly this.
- Heavy work (nsys exports, sqlite scans, pytest, any GPU) runs on the box through
  runbook scripts; the Mac reads the text results only. Archives land under
  `artifacts/` and are unpacked there.

### Dependencies are consumed, never patched

sglang, torch, transformers and qwen-tts are pinned. A gap is closed at an omni owned
seam: the batcher, the stage factories, the loader, the runners. Calling a
dependency's public sub API (`encoder.encode(values, num_quantizers=16)`) is a seam;
monkeypatching its class is not. Moving a loaded module's buffers is instance
configuration, not a patch.

### The process being optimized

One process, one GPU, about 18 threads sharing the interpreter lock:

```
MainThread            zmq stage runtime, outbox drain
scheduler-tts_engine  the talker (sglang scheduler): prefill, decode steps, sampling,
                      code predictor, stream output builder     ~470 eager launches/request
qwen3-tts-ref-code    reference encoder batcher thread, own stream   (#2172: ~1 replay)
ThreadPoolExecutor-2  8 preprocessing workers: normalize, speaker encoder (ECAPA),
                      mel on CPU, tokenization, prompt build    ~220 launches/request
scheduler-vocoder     chunk validation, plans
qwen3-tts-vocoder-initial      bootstrap decode, arena slot zeroing   (#2151: ~47 launches)
qwen3-tts-vocoder-followup-0/1 steady stride replays              ~28 launches/chunk
omni-request-build_N, asyncio_N   small
```

Streams: the talker on the default stream, the vocoder decode stream at priority -2
(PyTorch range 0..-3, more negative is higher), preprocessing and the encoder on their
own streams at 0. Request path for a streaming voice clone request:

```
coordinator -> preprocessing pool thread: normalize audio, submit clip to the batcher,
   speaker embedding, wait for codes, tokenize, build_voice_clone_inputs (ICL prompt)
-> tts_engine: request build, queue, prefill (eager), decode steps (graphed) with the
   code predictor, stream chunks of codes
-> vocoder: initial worker decodes the reference prefixed bootstrap (window graphs),
   follow up workers replay the steady stride, audio chunks to the coordinator
```

Non streaming uses `chunked_decode` on the whole sequence and none of the incremental
machinery. CustomVoice and VoiceDesign carry no reference clip.

## 2. The mechanism every slice so far has attacked

Every eager kernel launch releases and reacquires the interpreter lock (torch releases
it around dispatch). With 18 threads, a reacquire is contended: CPython 3.12 waits in
`pthread_cond_timedwait`, about 20 us each, and the count is about one per launch.
Measured on Nsight OS runtime rows, 20 s at c16 (docs 38 and 40):

| thread | main, default launch | main, early ids | after #2151, early ids |
| --- | ---: | ---: | ---: |
| talker lock wait, percent of window | 16.0 | 31.1 | 24.0 |
| initial vocoder worker | 12.5 | 31.5 | 2.3 |
| reference encoder thread | 10.7 | 22.9 | 19.5 |

The GPU is not the bottleneck: SM issue is 13 to 15 percent, GR active 70 to 78. The
bottleneck is CPU side issue: launch count, host syncs, lock handoffs. SM issue rises
when a thread stops stalling the GPU's feed, not when more work is added. Every fix so
far removed launches from a thread: #2151 took the bootstrap decode from about 860
eager launches to 3 replays, #2172 took the reference encode from about 1,000 to 1.

## 3. Status

### Merged

- #2151 (2026-09-13): reference prefixed bootstraps replayed through captured window
  graphs; precompile on the runner's own tensors. Default launch TTFC mean 129.7 to
  117.0 ms, cold misses 2,688 to 0. Docs 32 to 36.

### Open

- #2172 `perf/qwen3-tts-reference-encoder-graphs` at c372ef10a: reference encoder
  padding buffers on the host, 16 quantizers, graphs at bucketed lengths. Census in doc
  44; PR body draft `pr_qwen3_tts_reference_encoder_graphs.md`. Pending on the box:
  the two test files at c372ef10a (the fakes were fixed at a1f2b6249, unrun), and the
  `A-nsys` metric files plus that arm's client log archived next to B's.
- #2123 `perf/qwen3-tts-stage-ids-early` (early ids: the layer 0 id staged before the
  code predictor, lookahead off). Held on its doc 31 gate: first chunk within 10 ms of
  main's control at early ids throughput. With #2172 it passes (103.7 against 116.5 ms
  at 19.1 against 15.9 req/s). After #2172 merges: rebase onto main, remeasure main
  against main plus #2123 on the doc 33 protocol, and add one open loop pass at main's
  rate (`--request-rate`) for the equal throughput first chunk read.
- #2126 `perf/qwen3-tts-nonblocking-copies` (draft): sampling restage from pinned host
  memory, finish payload copied without blocking the scheduler. Measured on top of
  #2123; remeasure on the new main after #2123.
- `perf/qwen3-tts-codec-precompile` (unopened, redundant, inside #2151): delete.

### Measured, for the record

Docs 31 to 44 under `tasks/perf_backlog/`. The census pattern, default launch, streaming
c16 full corpus, pass 2 of one boot per arm:

| slice | TTFC mean ms | preprocessing p50 ms | req/s |
| --- | ---: | ---: | ---: |
| main 3060470a8 | 129.7 | 39.4 | 15.43 |
| #2151 | 117.0 | 36.3 | 15.78 |
| main 69ddc6baa (after #2151) | 116.5 | 35.4 | 15.88 |
| #2172 | 99.5 | 23.5 | 15.89 |

Quality bands: WER 1.05 percent, speaker similarity 71.3 to 71.5 on the full seed-tts
en corpus. Repeatability: about 4 percent on req/s and 2 ms on a segment between passes
of one boot; the first pass after a boot is 28 to 30 ms slower on TTFC mean and is
never quoted. Streaming is not byte reproducible across boots even with a seed (10 to
38 of 1,088 identical), so identity is not a streaming gate.

## 4. Remaining bottlenecks, ranked by launches per request

1. The talker thread: about 470 eager launches per request, 81 per decode step around
   the graphed step (sampling, penalties in `cumulate_penalty_output_tokens`,
   `_build_forward_batch`, the `_foreach_copy` into graph inputs, `stream_output_builder`)
   plus the eager prefill. Doc 32 item 2. Its lock wait is the largest in the process
   (16 percent default, 24 with early ids after #2151). The reads to take first: the
   per step launch list from `nsys_threads.py` restricted to the talker's tid between
   two graph launches, and the leaf frames of its lock ownership from a py-spy record.
2. The preprocessing pool threads: about 220 launches per request. The speaker encoder
   (ECAPA, `extract_speaker_embedding` in `sglang_model.py`) is about 120 of them;
   `mel_spectrogram` in qwen-tts recomputes the librosa filterbank on every call and
   runs the STFT on the CPU under the lock; the prompt build (`build_voice_clone_inputs`,
   `generate_icl_prompt`) is about 40 tiny embedding launches. Same shape of fix as
   #2172: bucketed graphs for the speaker encoder at the same frame buckets, the
   filterbank computed once, the prompt build's constants built once.
3. The vocoder initial worker's `_zero_slot` in `codec_state_arena.py`: one `zero_`
   launch per arena buffer at every slot acquisition, about 35 of its remaining 47
   launches per request and 33 to 39 percent of its lock time. One
   `torch._foreach_zero_` over the slot's rows is a handful of launches.
4. `validate_chunk` in `streaming_vocoder.py` casts every chunk to long on the vocoder
   scheduler thread: one launch per chunk, half of that thread's lock time.
5. Under early ids the preprocessing segment still grows 6 ms (23.5 to 29.6 p50) and the
   bootstrap 3 ms, from items 1 and 2 competing with a talker stepping 22 percent
   faster. Item 1 is what remains of that.
6. The tokenizer encoder runs in bfloat16 because the vocoder loads it that way; its
   codes differ from the float32 checkpoint's on 58 percent of positions (doc 41).
   Whether float32 references clone better is a quality question with its own census.
7. The first pass after boot is 28 to 30 ms slower on TTFC mean; not diagnosed.

Not bottlenecks: the GPU (SM issue 15 percent), the arena's memory, the graph
footprints (2.1 GB window runner, 224 MiB encoder).

## 5. How a bottleneck is found

Four layers, always in this order, each pointing at the next. Never Nsight first.

### Layer 1: the client census and the anatomy

The benchmark: `python -m benchmarks.eval.benchmark_tts_seedtts --generate-only
--use-existing-server --stream --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --meta
zhaochenyang20/seed-tts-eval-arrow --ref-format references --lang en --warmup 1
--concurrency 16 --port 31001 --output-dir <dir>`; no seed; passes 1 and 2 per boot,
pass 2 quoted; seeded c1 once per default launch boot for the single stream read;
`--request-rate R` for an open loop pass at a fixed arrival rate (equal throughput
comparisons; Little's law gives the in flight count from the mean latency).

The event recorder: `/start_profile` with `enable_torch` false on pass 2, stopped
after 200 completions as doc 33 does; it writes `events_<stage>_<pid>.jsonl` per stage. `first_chunk_anatomy.py
<label>=<events_dir>[:<speed_results.json>] ...` prints the per segment p50 and p95
from admission to first audio (preprocessing, request build, queue, prefill, first
frame to first audio), the bootstrap cost by how many other bootstraps were in flight,
prefill by overlap, and the talker cadence. This is the read that says which stage
moved. The preprocessing segment has no sub events; split it by the reference cache's
modes (hit, follower, encode) and by pool occupancy with the events' timestamps as in
doc 38 when it is the one that moved.

Quality: `--transcribe-only` (WER with Qwen3-ASR) and `--similarity-only` on pass 2's
audio of the B arm.

### Layer 2: Nsight, one separate boot per arm, never on a measured boot

```
nsys launch -- python -m sglang_omni.cli serve ...          (from the arm's worktree)
nsys start --gpu-metrics-devices=1 --gpu-metrics-frequency=20000 -o <out>
<the c16 pass>                                               (a 20 s slice or the whole pass)
nsys stop
nsys export --type sqlite -o <out>.sqlite <out>.nsys-rep
```

Reads, all from the sqlite, all on the box:

- `nsys_threads.py`: per thread launches, graph launches, memcpys, syncs with their
  durations, kernel queue latency (launch to start, graph replays excluded), top
  kernels. Divide by the requests in the window for launches per request. This is the
  table that names the thread and the count.
- `nsys_lock_waits.py <sqlite> <threads.json>`: per thread `pthread_cond_timedwait`
  (the interpreter lock), `sem_wait` and `sem_clockwait` (Python locks and queues,
  idle), `pthread_rwlock_rdlock` (the CUDA runtime). Count, summed ms, percent of the
  window. A thread's lock wait is the direct cost of its launches.
- `nsys_gpu_metrics.py <sqlite> [--bench-log client.log | --window T0 T1]`: GR Active
  (any engine busy), SMs Active (SMs with a resident warp), SM Issue (issue slots used,
  the utilization rate), Tensor Active, warps in flight, unallocated warps, DRAM read
  and write, GPC clock (stored in Hz, printed in MHz), with sample count, span and the
  share of zero samples. Metrics are looked up by name; ids move with the metric set.
  The cohort window (the benchmark's "Benchmarking N requests" to "Results saved"
  lines, converted through the export's `localTime`) is the reproducible cut for a
  whole pass capture; it includes ramp and drain. A steady state slice excludes them.
  Quote the window with every number; compare only same window against same window.
  The teammate's `compute_sm_window.py` (branch `profie_cosy_workload`) reads the same
  counters on the cohort window; `patches/compute_sm_window_names_localtime.patch`
  gives it name lookup, the local clock and sample counts.
- What is not comparable: means over windows of different length or load (a 32 request
  cohort is all ramp and drain), nvidia-smi or dmon SM percent (it is not SM issue),
  numbers under different trace sets (CUDA and OS runtime tracing add CPU overhead to
  the very threads under study; a GPU metrics only capture is the least disturbing),
  boots with different tenants on the box.

### Layer 3: the lock record

`py-spy record --pid <stage pid> --gil --threads --nonblocking --rate 250 --duration 60
--format raw -o gil_raw.txt` about 15 s into a c16 pass, then `gil_share.py
gil_raw.txt`: each thread's share of lock ownership and, from the raw stacks, the leaf
frames it holds the lock in (`pad`, `cdist`, `_zero_slot`, `validate_chunk` were found
this way). The record does not give the held fraction: py-spy's "Samples" counts
traces written, not intervals. Wrap py-spy in `timeout 90`; it has hung once. Not a
latency pass: the sampler costs about 70 ms of TTFC.

### Layer 4: the code and a micro bench before any runtime change

Read the whole file, not the function: the thread it runs on, the stream, what
allocates, what syncs (`.item()`, `.tolist()`, `F.pad` with a device tensor argument,
`cudaStreamSynchronize` in the Nsight table), what is captured and what is eager. Then
a standalone bench on the tokenizer or the model alone, run on the box without a
server, that answers the design's questions with numbers before code is written:
doc 34 (`codec_window_bench.py`: replay cost per width and bucket, capture footprint),
doc 40 (`ref_encoder_graph_bench.py`: exactness of each part, replay cost per bucket),
doc 41 (`ref_encoder_padding_check.py`: where padded codes differ and why). Every
bench script is checked against the pinned dependency source (fetch the tagged file)
before the box runs it.

## 6. How a fix is designed

Rules, each with the case that produced it:

- Origin first. The origin is the launch count on a thread; the fix removes launches
  from that thread in the default single GPU launch. MPS, process splits and extra
  GPUs are never the fix.
- Derive constants from a measured cost curve, never from a corpus and never from one
  GPU. Good: the encoder ladder starts at 32 frames because a replay is a floor plus a
  small per frame term, so smaller keys save nothing measurable, and each key is 32
  MiB. Bad: `_DYNAMO_CACHE_ENTRIES = 1024` (an underived number; torch's default of 8
  held), a `min_free_gb` guard on 32 MiB keys.
- No branches for states the caller cannot produce, no fallbacks that keep dead code
  alive. The batcher's "device unresolvable" synchronize went away when the device
  came from the encoder's parameters.
- One path for the fast and the slow case. The encoder's miss path pads to whole
  frames exactly like the graph path, so the codes are defined once.
- Numerics: bit identity is claimed only where the structure gives it (a captured graph
  against its own eager shape; the padding buffers on the host; 16 stages of a residual
  chain). Where kernels change with shape (any batching or bucketing), the gate is the
  quality census, and the design makes the result deterministic per input (a reference
  encodes the same on every boot) even though it differs from the old path.
- Capture at omni seams with static buffers per key, warmup twice on the capture
  stream, one shared pool, largest key first, a failure disables the runner with its
  reason, and every replay's output is copied out before the next replay. Graphs are
  never captured on a thread that another thread might launch into at the time;
  capture happens before the worker thread starts and before the KV pool is sized.
- Hand writing a dependency's model is the last resort: the cost was never the math
  (10 ms alone against 50 in the server), and a rewrite carries checkpoint loading and
  every upstream fix. Remove the launches first; if GPU time then matters, compile the
  captured function.
- Batching by waiting is not a fix: the 2 ms batcher window against 40 ms arrival gaps
  never batches. Batch keys are added when a measured need shows up.
- Hardware agnostic: keys in frames, streams by device type, no GPU constants, no
  pinned metric sets. The only CUDA coupling is that graphs need a CUDA device;
  every other backend takes the eager path unchanged.

## 7. How a fix is built and shipped

1. Plan doc, under 200 lines: what was noticed with the numbers, how the code does it
   today (an ASCII flow with file anchors), the change as a numbered list, the tests,
   the gate. No history, no discussion references.
2. Micro bench runbook and readout (section 5 layer 4). The plan is revised on the
   readout, not defended.
3. Worktree from `upstream/main`, implement. Comments follow the sglang rule files:
   one to three lines, the fact only, `# note(ratish):` prefix on ours, no docstring
   notes, no doc or discussion references, no backticks. Names are mechanical and
   cheap (`split_frames_by_width`, `smallest_bucket`, `move_conv_padding_to_host`;
   never plan, resolve, orchestrate).
4. Tests are contracts and edge cases on real shapes: fakes return what the real object
   returns (a fake encoder returns `(1, quantizers, samples // hop)`), CUDA tests use
   a small real model (`MimiModel(MimiConfig(...))` with `upsample_groups` dividing
   `hidden_size`), buffer reuse is tested by replaying the same key twice, no comments
   inside tests, no mock and count. Unrun tests are never "good to go"; the box runs
   them before the census. Existing fakes that model the old contract are rewritten,
   not worked around.
5. Commits: short, lowercase, mechanical, no scope prefixes, no PR references, no
   trailers; one commit per mechanical unit (runner and loader; batcher and tests).
   Runtime commits are discussed before they are made; docs, runbooks and tests push
   directly. Pre-commit runs black and isort and may reformat; re-add and commit again.
6. Runbook for the box: worktrees to pull, the exact commands, the B gate log line,
   what to archive. Session rules: GPU recorded before each boot and clear, one
   independent server per arm, module launch, import path archived, dmon on, passes 1
   and 2, seeded c1 on default launch boots, quality on B pass 2, recompile grep,
   decode log gap check on the first boot, zero failed requests, Nsight as a separate
   boot per arm.
7. Readout doc: verdict first, the tables, what the numbers settle, what is next. The
   box operator's report is not the readout; its claims are checked against the files
   (a quoted row without a file behind it is not a measurement).
8. PR: title with one scope tag (`[Qwen3-TTS] ...`), body with mechanism, changes,
   census table main against the branch only, quality one line, footprint and knob;
   Nsight rows over the cohort window from separate boots; no testing history, no
   other arms. Social message: "I opened a fix here for qwen3-tts: <link>. <what it
   does>. <the numbers it moves>."
9. Review: a request for a CUDA test that covers real buffer reuse is answered with one
   on a small real model, same PR. Force pushes only when asked; a rewrite is a clean
   commit series reflecting the PR's changes, rebased onto upstream main.
10. After merge: dependent PRs rebase onto the new main and are remeasured as main
    against main plus the PR; nothing is stacked without its own census.

## 8. The runbook skeleton, per arm

```
S=/sgl-workspace/sglang-omni/tmp/<session>; ARM=<A|B>
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > $S/$ARM/gpus_before.txt
cd /sgl-workspace/sglang-omni/tmp/<worktree> && git rev-parse HEAD > $S/$ARM/head.txt
PYTHONPATH=. python -c "import sglang_omni; print(sglang_omni.__file__)" > $S/$ARM/import_path.txt
CUDA_VISIBLE_DEVICES=1 python -m sglang_omni.cli serve <the doc 33 serve line> > $S/$ARM/serve.log 2>&1 &
(B) grep -c "<gate line>" $S/$ARM/serve.log      # 1, or the boot is void
nvidia-smi dmon -s um -d 1 > $S/$ARM/dmon.log &
pass 1, pass 2 with the event recorder, seeded c1 (default launch), quality on B pass 2
grep -E "recompile_limit|disabled the" $S/$ARM/serve.log > $S/$ARM/recompile_grep.txt
```

Doc 33 holds the exact serve line and the recorder calls; doc 37 the py-spy pass; doc
43 the whole pass Nsight capture and the two script comparison.

## 9. Index

Docs (`tasks/perf_backlog/`): 31 gate and prefix prime; 32 the origin and the three
items; 33 session protocol; 34 window width bench; 35 session 1 void, repeatability;
36 #2151 readout; 37 preprocessing runbook; 38 preprocessing readout, lock waits; 39
encoder bench runbook; 40 encoder bench readout; 41 padding check readout; 42 encoder
graph plan; 43 encoder graph runbook; 44 encoder graph readout; `pr_*.md` PR bodies;
`research/perf_method_playbook.md` the earlier method doc; this file.

Scripts (`tasks/perf_backlog/scripts/`): `first_chunk_anatomy.py`, `nsys_threads.py`,
`nsys_lock_waits.py`, `nsys_gpu_metrics.py`, `gil_share.py`, `codec_window_bench.py`,
`ref_encoder_graph_bench.py`, `ref_encoder_padding_check.py`, `graph_replay_bench.py`,
`trace_*.py`. Patches (`tasks/perf_backlog/patches/`): the teammate's script fix and
its drop in copy. Early ids as a patch for A/B: `tasks/qwen3_tts_e4_investigation_20260912/early_ids.patch`.

Runtime files of the slices: `sglang_omni/models/qwen3_tts/` `incremental_codec.py`,
`incremental_codec_cuda_graph.py`, `codec_state_arena.py`, `streaming_vocoder.py`
(#2151); `reference_encoder_cuda_graph.py`, `request_builders.py`, `stages.py`,
`engine_builder.py` (#2172); `model_runner.py` (#2123, #2126).

## 10. The next slice, mechanically

Item 1 of section 4, the talker thread. Steps: (a) `nsys_threads.py` on the #2172
B-nsys export restricted to the talker tid, launches grouped between consecutive graph
launches, to get the per step list with counts; (b) py-spy leaf frames of the talker
under c16; (c) read `model_runner/base.py` (`_sample_next_token_ids`,
`_build_forward_batch`), `omni_scheduler.py` (`get_next_batch_to_run`,
`process_batch_result`), `request_builders.py` (`stream_output_builder`) and
`cuda_graph_buffer_registry.py` (`_foreach_copy`) whole, on the thread and stream they
run; (d) a plan doc: which of the 81 launches per step are capturable into the step's
graph, which are host reads that can be batched or moved, which are copies that can
be one foreach; (e) a micro bench on the model runner alone; (f) the slice, with the
early ids pair as its second census since that is where the talker's lock wait is
highest. Item 3 (`_foreach_zero_`) is a one commit slice measurable in the same
session.
