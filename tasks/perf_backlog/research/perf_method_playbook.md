# The performance method, as used on Qwen3-TTS, for reuse on other models

Four measurement layers, one tool per layer, one document per step. Nothing is quoted
that was not read from a file in an archive.

## 1. The layers, top down, each one locates the next

1. Client. `benchmarks/eval/benchmark_tts_seedtts.py` (`--generate-only
   --use-existing-server`, full corpus, `--warmup 1`, no seed, `--stream` for streaming).
   Reads `speed_results.json`: req/s, latency, RTF, TTFC mean and p99, inter chunk,
   c50/c100/c200, and `results.csv` for runaways. This says whether anything moved.
2. Stage events. The event recorder (`sglang_omni/profiler/event_recorder.py`) writes
   `events_<stage>_<pid>.jsonl` per process when `/start_profile` is called with
   `event_dir` and `enable_torch` false; stop with `/stop_profile` after about 200
   completions so the window is one steady state cohort. `python -m sglang_omni.profiler`
   gives the per request timeline and stage and hop breakdowns;
   `tasks/perf_backlog/scripts/first_chunk_anatomy.py` gives the segment p50 and p95
   from admission to first audio, binned by how many other requests were in the same
   stage at the time. This says which stage and which segment.
3. Kernels and host, one stage process. `/start_profile` with `enable_torch` true and
   `trace_path_template` writes a chrome trace per stage; 12 requests per concurrency
   unit is enough. `tasks/qwen3_omni_0518_numerics/scripts/perfkit.py`, standard library
   only: `ingest` once, then `steps` (per step device time by phase), `census` (kernels
   per replay by family, gaps between kernels), `timeline` (one step laid out), `hosttail`
   (which Python frames own the host time between launches), `diff` between two census
   JSONs, `memory` and `snapshot` for allocator events. This says whether a step is bound
   by kernel count, kernel time, host launch time, or waits.
4. Threads and the device, the whole process. `nsys` with
   `--gpu-metrics-devices --gpu-metrics-frequency`, a 20 s window in pass 2 after a full
   preconditioning pass, SQLite export, `tasks/perf_backlog/scripts/nsys_threads.py`:
   per thread launch counts and durations, kernels by correlation id, sync calls by API,
   lock wait and hold from NVTX. `trace_gpu_activity.py` gives GR active and SM proxies
   from a kineto trace when nsys is not affordable. This says which thread pays and
   whether the GPU is idle, and it is the layer that named the origin (launch count per
   request across threads) rather than the symptom.

Nsight is never on a measured boot. Measure first (layers 1 and 2 on every pair), profile
only the arm and concurrency that the numbers point at.

## 2. The session protocol, fixed

- One server per arm, from that arm's worktree, `python -m sglang_omni.cli serve` with
  only `CUDA_VISIBLE_DEVICES` in front, archive the import path before the boot and gate
  the arm on a log line only its code prints.
- Regime check before any pair is quoted: the decode log line gap (about 0.6 to 0.7 s per
  40 steps on H100 for Qwen3-TTS), clocks, no forced backend variable, other GPUs recorded
  before every boot (`gpus_before.txt`), `nvidia-smi dmon -i N -s pucv -d 1` on every boot.
- Two passes per arm: pass 1 unprofiled, pass 2 with the event recorder. Pairs are valid
  only inside one session on one GPU; deltas hold, absolute numbers across days do not.
  A pass with a runaway output (Qwen3-TTS: 163.84 s) is not quotable.
- Expected values are written in the runbook before the run, per read, with the source
  measurement they come from. The readout states verdict first, provenance second.
- Quality on the B arm of the default pair: WER and speaker similarity against the
  bands. Identity (seeded c1) is reported, gated only where the change is bit exact by
  construction and the path is reproducible across boots (streaming was not).

## 3. The document chain, one file per step in tasks/perf_backlog

census (what the numbers are, layer 3 or 4) -> plan (mechanism, candidate slices ordered
by what they remove at the origin, each with a named experiment) -> runbook (arms, heads,
boot command, reads, expected values, archive list) -> readout (verdict, provenance,
tables, what was learned, next runbook) -> `00_backlog.md` (order, open items by area,
done, decisions the user owns, status to establish). Scripts live in
`tasks/perf_backlog/scripts` and are verified against the exact branch head before the
box runs them (signature check, module level imports).

## 4. The decision rules

- Origin first: find the count that every symptom scales with (for Qwen3-TTS, eager
  launches per request across 18 threads sharing one interpreter lock), then remove
  launches at the source in the default single GPU launch. Moving work between threads,
  MPS, process splits or a second GPU are not fixes.
- One measured PR per issue on current upstream main; parts validated by the same run
  ship together; a slice that does not move the read it was written for is parked with
  its readout, not merged.
- No constant a measurement does not pin; a bucket or width list is derived from the
  workload structure or measured in a microbenchmark with its own script and readout.
- Bit exactness is checked with a script on the box against the pinned tree before a
  fused or reordered path is adopted.

## 5. What is model specific when porting

- perfkit's step definition is "one backbone graph launch to the next on the scheduler
  thread" with a predictor marker regex; set the marker to the model's second graph (the
  flow or vocoder replay for CosyVoice) or run `steps` without one. The kernel FAMILIES
  regexes are generic but worth a pass over the new model's kernel names.
- The anatomy script's segment list is the Qwen3-TTS stage event names; the recorder is
  generic, so map the new pipeline's stage names and the first audio event once.
- The regime numbers (decode line gap, runaway length, quality bands) are per model and
  are measured on the first clean boot, then written into the runbook as gates.
