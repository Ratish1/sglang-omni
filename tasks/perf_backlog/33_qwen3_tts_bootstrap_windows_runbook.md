# 33. Runbook: the vocoder bootstrap through captured graphs (doc 32 item 1)

Branch `perf/qwen3-tts-bootstrap-graphs` at 1453ef538, three commits on upstream main
3060470a8: the runner's window schedule and bucket queries (7e5e2aa48), the window
runner and its knob (cb9cf8f46), the windowed decode path (1453ef538). Same session
rules as doc 31: plain server command, GPU 0, GPUs 1 to 3 recorded before every boot,
dmon on every boot, full corpus, warmup 1, no seed, two passes per arm, event recorder
in pass 2 stopped after 200 completions, decode log gap check on the first boot.

What the change does. A reference prefixed bootstrap (reference frames plus the first
chunk) used to be one eager decode of an uncaptured width, about 860 host launches. The
initial worker now owns a second graph runner, the window runner, and a same width
cohort whose width no runner captured is consumed as a sequence of the window runner's
widths against the cohort's arena slots, one replay per window, largest width first.
The widths are the powers of two up to the widest decode the scheduler already replays
through a graph, a left context plus one steady chunk (16 + 8 = 24 by default, so
1, 2, 4, 8, 16). They are a vocoder factory argument,
`incremental_codec_cuda_graph_window_frames`, and an empty list turns windowing off. The
serve log's codec state line carries `windowed_decodes: {rows, replays}` and a `window`
runner entry next to `cold` and `warm`.

## Step 0, the replay cost of the ladder against the eager decode, before any boot

```bash
git fetch origin analysis/qwen3-omni-0518-numerics perf/qwen3-tts-bootstrap-graphs
git worktree add tmp/bw 1453ef538
cd tmp/bw && CUDA_VISIBLE_DEVICES=0 python ../../tasks/perf_backlog/scripts/codec_window_bench.py \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base --widths 4,8,16 --totals 50,100,150 --reps 50 \
  --out ../../results/bw/codec_window_bench.json
```

(the script lives on the analysis branch; run it with the code branch's worktree on the
import path.) Read, per width: device and host ms of one replay at bucket 1 and 4. Per
total: the eager decode's device and host ms, and the window sequence's device and host
ms with its window list. Also the capture time and footprint.

This is a gate, not a search. The ladder is derived, not tuned, so the bench answers one
question: does the sequence for a 100 and a 150 frame bootstrap cost no more device time
than the eager decode while cutting its host time by an order of magnitude. If yes, the
pair runs. If no, the finding is reported with the numbers and the pair does not run
until the reason is understood.

## Step 1, unit tests on the box, full files

```bash
python -m pytest tests/unit_test/qwen3_tts/test_incremental_codec_cuda_graph.py -q
python -m pytest tests/unit_test/qwen3_tts/test_pipeline.py -q
python -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py -q
```

The two accelerator tests in the graph file run on the box only. All must pass before a
boot; archive the output.

## Step 2, the pair, default layout (2 boots)

A: upstream main 3060470a8. B: the branch head. Streaming c16, doc 31 protocol. Reads,
per arm, `first_chunk_anatomy.py` on the pass 2 events plus the client summary: TTFC
mean and p99, req/s, inter chunk, the first frame to first audio segment at ahead 0 and
its mean, the prefill by overlap table, preprocessing p50, the cadence, dmon GR active.
From B's serve log: the window runner's captured keys and footprint at boot, and the
last codec state line's `windowed_decodes` rows and replays, the window runner's
`replays`, and the cold runner's `uncaptured_fresh_frames` counter. Expected on B: the
cold counter near zero (only widths no ladder covers stay eager), `windowed_decodes.rows`
about the number of reference prefixed requests, replays per row about the window count
for the corpus's reference lengths (a 100 frame bootstrap is 16, 16, 16, 16, 16, 16, 4).

## Step 3, the origin check, Nsight on both arms (1 window each)

The doc 29 default backend protocol, full preconditioning pass, 20 s window in pass 2,
`--gpu-metrics-devices=0 --gpu-metrics-frequency=20000`, SQLite export, `nsys_threads.py`.
Read per thread: launches in the window, launches per bootstrap on the initial worker
(launches over decode done syncs), lock wait per bootstrap, the preprocessing workers'
and the scheduler's lock wait, GR and SM active. The slice reached the origin only if
the initial worker's launches per bootstrap fell from about 860 to tens and the other
threads' lock waits fell with them. If the launches fell and the waits did not, say so:
that is a finding about what the lock waits on, not a pass.

## Step 4, quality, on B's pass 2

WER and speaker similarity against the bands (WER about 1.0 percent, similarity about
71.2). A seeded c1 non streaming pass is not informative here (the bootstrap path is
streaming only); instead a seeded streaming c1 pass on A and B, identity count reported
not gated, since window boundaries change reduction order inside the decoder.

Archive per boot: head, import path, server command, gpus before, dmon log, both passes'
speed_results and client logs, every event file of pass 2, serve.log, the Nsight SQLite
and the `nsys_threads.py` JSON, plus step 0's JSON and step 1's output.
