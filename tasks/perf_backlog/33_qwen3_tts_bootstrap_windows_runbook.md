# 33. Runbook: the vocoder bootstrap through captured graphs (doc 32 item 1)

Branch `perf/qwen3-tts-bootstrap-graphs` on upstream main 3060470a8. Same session rules
as doc 31: plain server command, GPU 0, GPUs 1 to 3 recorded before every boot, dmon on
every boot, full corpus, warmup 1, no seed, two passes per arm, event recorder in pass 2
stopped after 200 completions, decode log gap check on the first boot.

What the change does. A reference prefixed bootstrap (reference frames plus the first
chunk) used to be one eager decode of an uncaptured width, about 860 host launches. The
initial worker's cold graph runner now also captures a set of window widths, and a plan
whose width is not captured is consumed as a sequence of captured widths against its
arena slot, one graph replay per window, largest width first. Requests of different
lengths share replays round by round. The window widths are a vocoder factory argument,
`incremental_codec_cuda_graph_window_frames`, default (1, 2, 4, 8, 16, 32); an empty
list turns windowing off. The serve log's codec state line carries
`windowed_decodes: {rows, replays}` and the cold runner's `window_frames`.

## Step 0, the replay time per width, before any unit test or boot

```bash
git fetch origin analysis/qwen3-omni-0518-numerics perf/qwen3-tts-bootstrap-graphs
git worktree add tmp/bw perf/qwen3-tts-bootstrap-graphs
cd tmp/bw && CUDA_VISIBLE_DEVICES=0 python ../../tasks/perf_backlog/scripts/codec_window_bench.py \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base --widths 4,8,16,32,64 --totals 50,100,150 --reps 50 \
  --out ../../results/bw/codec_window_bench.json
```

(the script lives on the analysis branch; run it with the code branch's worktree on the
import path.) Read, per width: device and host ms of one replay at bucket 1 and 4. Per
total: the eager decode's device and host ms, and for each largest width the window
sequence's device and host ms and its window list. Also the capture time and footprint.

Decision: the default window set is the largest width whose sequences stay at or under
the eager device time for the 100 and 150 frame totals while cutting the host time by
an order of magnitude, and whose capture footprint at bucket 8 is under 2 GB. If 32
holds, the shipped default stands; if 16 or 64 is the right largest width, the default
changes on the branch before step 1, and the reply records the numbers.

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
From B's serve log: the cold runner's captured keys and footprint at boot, and the last
codec state line's `windowed_decodes` rows and replays and the cold `replays` and
`uncaptured_fresh_frames` counters. Expected on B: `uncaptured_fresh_frames` near zero,
`windowed_decodes.rows` about the number of reference prefixed requests, replays per
row about the window count for the corpus's reference lengths.

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
