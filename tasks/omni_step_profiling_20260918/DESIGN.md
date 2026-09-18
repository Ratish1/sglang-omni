# Step profiling for sglang-omni AR models, Qwen3-TTS first

Status 2026-09-18: design, before any runtime change. Every open question below is a
named check (C0 to C7) run on moss before the tooling is trusted.

## 1. What the BBuf skill does, mechanically

Source: `AI-Infra-Auto-Driven-SKILLS/skills/llm-torch-profiler-analysis`, read in full
(SKILL.md, analyze_llm_torch_profile.py, profile_common.py, the timing and attribution
parts of triage_kernel_helpers.py and triage_overlap_helpers.py).

- Two degenerate workloads, one trace each: prefill (input 4090, output 1) and decode
  (input 1, output 2048). Directory names `prefill/` and `decode/` are the stage labels
  (`profile_common.py:467 parse_stage`).
- Warmup before arming: 10 requests for prefill, one 10-token request for decode
  (`profile_common.py:726 build_probe_plan`).
- Server-side step bound: SGLang's `/start_profile` takes `num_steps`; the scheduler
  counts `forward_ct` in `run_batch` and stops at the target
  (`sglang/srt/managers/scheduler_components/profiler_manager.py:408`, called at
  `scheduler.py:4029` right after `forward_ct += 1`). The skill asks for `num_steps + 1`.
- Kernel table: every GPU kernel of the heaviest device pid, grouped by canonical name,
  `sum(dur)`, share of the stage's summed kernel time, rows under 1.0 percent hidden
  (`analyze_llm_torch_profile.py:29,712`).
- Attribution: kernel `External id` to cpu_op to the innermost ranked Python frame; if
  the kernel has no cpu_op, its `correlation` to the `cuda_runtime` launch event and the
  frames active on that thread at launch time (`triage_kernel_helpers.py:2058`). Graph
  replays resolve through the `cudaGraphLaunch` correlation to the replay call site.
  `sglang_omni/` frames rank highest (`:1393`, commit 6238045).
- Stage inside one trace: user annotations whose name contains `decode` or `prefill`
  (`:1609`). SGLang's own span is `step[DECODE bs=N]` / `step[EXTEND bs=N toks=T]`
  (`sglang/srt/utils/profile_utils.py:463`, opened in `ModelRunner.forward`
  `model_runner.py:1591`). `EXTEND` matches neither word, so mixed traces mislabel
  prefill kernels; the skill avoids this by one workload per trace.
- Overlap: a sweep over kernel start and end points per stream gives busy union,
  exclusive and hidden time per kernel (`triage_overlap_helpers.py:566`).
- Two-trace mode: a mapping trace (graphs off, readable sites) and a formal trace (real
  serving config, real timing). The mapping trace only names kernels.

## 2. How sglang-omni differs

Facts on main 144bd6399:

- The omni profiler records continuously between `/start_profile` and `/stop_profile`.
  There is no step bound, and `with_stack` / `record_shapes` come only from env vars
  (`sglang_omni/profiler/torch_profiler.py:125-137`). The `config` field of the request is
  dropped (`profiler_control.py:53`).
- The profiler starts on the stage control thread (`pipeline/stage/runtime.py:1889`),
  not on the scheduler thread. Qwen3-TTS runs three stages in one process
  (`models/qwen3_tts/config.py:57,69,78`) with its own worker threads:
  `qwen3-tts-ref-code` (`request_builders.py:809`) and `qwen3-tts-vocoder-*`
  (`streaming_vocoder.py:1104`). Whether CPU ops, annotations and Python frames of
  threads other than the starting one reach the trace is check C0.
- OmniScheduler does not call SGLang's profiler manager. It counts forwards itself:
  `_stamp_batch_launch` does `forward_ct += 1` (`omni_scheduler.py:1433`) for both the
  sync path (`_run_batch :1441`) and the lookahead launch (`_run_batch_launch :1530`).
- Qwen3-TTS runs the sync loop (`_event_loop_normal :2357`; it never sets
  `enable_async_decode`). One iteration: talker forward inside SGLang's step span, then
  sampling, then `code_predictor_forward` (a graph replay per batch size,
  `models/qwen3_tts/model_runner.py:197`), then collect. The predictor and sampling sit
  outside SGLang's span. FunASR, Zonos2 and MOSS use the lookahead loop, where launch
  and resolve of different steps interleave.
- The vocoder decodes on its own threads and stream, concurrently with talker steps.
  An LLM decode trace has one owner; ours has four (talker, predictor, vocoder,
  reference encode) sharing the device and the interpreter lock.
- `bug`: `TorchProfiler.start` reads `rank` before assigning it on the restart path
  (`torch_profiler.py:67,70` before `:85`). Unreached today because stages check
  `is_active()` first.

### C0 result, moss GPU 6, torch 2.13.0+cu130 (`scripts/thread_coverage_probe.py`)

Profiler started on the main thread; a thread alive before the start (pre) and one
created after it (post) each ran 20 iterations of eager ops plus an 8-node graph replay.

| config | cpu ops and spans from pre / post | `_profiler_enabled()` on workers | python frames |
| --- | --- | --- | --- |
| default (omni today) | none / none | False | none |
| with_stack | none / none | False | pre only |
| profile_all_threads | 103 + 20 spans each | False, also False on main | none |
| profile_all_threads + with_stack | 103 + 20 spans each | False everywhere | pre only |

- Kernels and launch events arrive from every thread in every config (CUPTI is global).
- Under profile_all_threads, eager kernels resolve through `External id` to the cpu op
  of the right thread exactly: 81 per worker, 41 on main.
- Graph replay kernels carry no `External id` in any config (480 = 3 threads x 20 x 8).
  Their only link is the `cudaGraphLaunch` event's tid. That tid was right for threads
  alive before the start and wrong for the thread created after it: raw pthread values
  in default mode, a dead thread's native id under profile_all_threads.

Consequences for omni:

1. Today's traces hold no CPU ops, spans or frames from the scheduler, vocoder or
   reference threads. The step span SGLang opens in `ModelRunner.forward` is gated on
   `_profiler_enabled()` and is never emitted on the scheduler thread in any mode.
2. The capture needs `profile_all_threads=True`, started after warmup so every worker
   thread already exists, and omni spans gated on omni's own armed flag.
3. Owner of an eager kernel = tid of its cpu op. Owner of a graph kernel = tid of its
   `cudaGraphLaunch`, trusted only when that tid owns cpu ops in the same trace and,
   for eager kernels of that tid, launch tid equals cpu op tid (checked, count 0).
4. The BBuf fallback (launch tid, then frames on that tid) and our recovered
   `attribute.py` (runtime call tid, then frames) inherit the stale-tid risk for
   threads created after the start.

## 3. Workload contract for TTS

The talker is an AR model over codec frames, so the LLM contract carries over with
TTS units:

| workload | request | length control | batch | captured steps |
| --- | --- | --- | --- | --- |
| prefill-b1 | longest corpus text + 256-frame reference | max_new_tokens 1 | serial, 1 in flight | every EXTEND forward of W+N requests, last N kept |
| prefill-b16 | same, 16 sent together | max_new_tokens 1 | coalesced EXTEND | N forwards |
| decode-b1 | short text, no reuse of prompt cache | max_new_tokens 2048, text long enough to exceed W+N steps | 1 | N DECODE forwards after W |
| decode-b16 | 16 such requests started together | same | 16 | N forwards while bs stays 16 |

- There is no ignore_eos for Qwen3-TTS, so decode length comes from the text. The
  driver picks texts whose measured frame count exceeds W+N and records it.
- Distinct text per request (prefix cache off the table), reference audio distinct per
  request in prefill runs (reference cache off the table), both recorded.
- Model: `/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base` on moss, streaming on (the
  vocoder is part of decode). No CustomVoice checkpoint on moss; its breakable prefill
  graphs are out of scope until one is fetched.

## 4. Design

```
driver (tasks script, box)                 server (omni, one process)
 warmup W ---------------------------->   scheduler thread: forward_ct
 POST /start_profile {num_steps N,          |  _stamp_batch_launch: +1
   start_step?, with_stack, shapes} --->    |  armed? start at next forward
 send workload ----------------------->     |  omni.step span per iteration
                                            |  count N, stop, export
 wait for trace file <-------------------  trace_<stage>_pid<p>_rank0.trace.json.gz
 analyse on the box:
   BBuf triage (unchanged) -> kernel / overlap / fuse tables per stage dir
   step ledger (ours)      -> per iteration wall, busy, idle, launches, syncs, owner
 return text only
```

### 4a. Runtime change (one omni PR, profiler only, no model code)

1. `ProfilerStartMessage` and `StartReq` carry `num_steps`, `start_step`, `with_stack`,
   `record_shapes` (the dropped `config` is replaced by named fields).
2. `TorchProfiler` gains an armed state. `OmniScheduler._stamp_batch_launch`, after its
   `forward_ct += 1`, calls one hook: start on the first armed forward, stop and export
   when N forwards have run. The count lives on the thread that runs the forwards, so the
   window is exact in both loops; the HTTP stop stays for unbounded captures.
3. The profile passes `profile_all_threads=True` (torch 2.13 `_ExperimentalConfig`),
   per C0. It starts on the scheduler thread at a forward boundary, after the driver's
   warmup, so every worker thread predates the start.
4. One span per scheduler iteration, emitted only while omni's profiler is armed (its
   own flag; `_profiler_enabled()` is False on every thread under profile_all_threads):
   `omni.step prefill bs=B toks=T` / `omni.step decode bs=B` around `_run_batch`, and
   around launch and resolve separately in the lookahead loop with the forward_ct in the
   name so the ledger pairs them. The words `prefill` and `decode` make the BBuf stage
   split correct inside mixed traces.
5. Fix the `rank` use before assignment.

Costs when no profiler runs: one `_profiler_enabled()` call and one armed check per
forward. The PR states this and C7 measures it.

### 4b. Driver (tasks script, not runtime)

`profile_workloads.py --url --model --workload {prefill-b1,prefill-b16,decode-b1,decode-b16}
--warmup W --steps N --out DIR`: builds the requests of section 3 from the seed-tts
corpus, sends warmup, arms with `num_steps`, sends the workload, waits for the trace, moves
it under `DIR/<prefill|decode>/`, writes `server_args.json` from `/server_info` if exposed
and `workload.json` (texts, reference ids, frame counts, bs per step).

### 4c. Step ledger (tasks script)

Reads one trace. For each `omni.step` span on the scheduler thread:

- wall = start of this span to start of the next (the scheduler loop is serial);
- kernels owned by the step: launch event (`correlation`) on the scheduler thread inside
  the span, graph kernels via their `cudaGraphLaunch` correlation;
- device busy union of those kernels, busy union of all kernels in the window, idle =
  wall minus busy union;
- launches (`cudaLaunchKernel`, `cudaGraphLaunch`), memcpy by direction, sync calls
  (`cudaStreamSynchronize`, `cudaEventSynchronize`, blocking `cudaMemcpy`), per thread;
- owner split of device time: talker backbone (inside SGLang's step span), predictor
  (graph launch from `code_predictor_forward`), sampling, vocoder thread, reference
  thread, by launching thread and frame.

Output: p50 / p95 per steady step at the workload's bs, churn steps listed separately,
no share cut, the tail below 1 percent shown as one row with its count and time. The
kernel totals must equal the BBuf kernel table's totals for the same trace (C5).

### 4d. SM issue stays in Nsight

The torch trace has no SM counters. The same driver and workload, one Nsight boot per
arm, `--gpu-metrics-set=ad10x` on moss (gh100 on the H100), metrics averaged over the
window the driver logs, read with `analysis/.../scripts/nsys_gpu_metrics.py`.
`RmProfilingAdminOnly=1` on moss; whether root in the container can read the counters is
check C6.

## 5. Checks before any number is used

| id | question | pass |
| --- | --- | --- |
| C0 | which threads a profiler started on one thread records (cpu ops, annotations, python frames, launch events) | done 2026-09-18, section 2 |
| C1 | step window exact | trace holds N `omni.step` spans of the workload's mode, forward_ct contiguous |
| C2 | graph kernels complete | kernels per predictor replay in the torch trace equal the node count from `nsys --cuda-graph-trace=node` for the same bs |
| C3 | all owners present | kernels launched from the vocoder and reference threads appear with their thread |
| C4 | profiler overhead | steady step wall in the formal trace against the unprofiled step cadence of the same workload (event recorder), two unprofiled runs give the spread; if the gap exceeds the spread, timings come from Nsight and the torch trace only counts |
| C5 | ledger equals BBuf | per-kernel totals identical on the same trace |
| C6 | Nsight metrics readable in the container | one 10 s ad10x capture returns nonzero SM issue |
| C7 | tooling costs nothing when off | c16 streaming decode-b16 cadence with and without the PR, profiler off |

## 6. What the 4090 numbers mean

moss is 8 x RTX 4090 D (sm89), shared; GPUs 6 and 7 were free on 2026-09-18.
Kernel counts, launches, syncs, graph node counts and owner splits transfer to the
H100. Wall times, SM issue levels and attention kernels do not: sm89 has no FA3, so the
talker's attention kernels differ from the H100's. Decisions that rest on time are
confirmed on the H100 before a PR claims them.

## 7. After the profile

The ledger ranks per-step cost by owner. Candidate slices already known, each to be
confirmed or dropped by the ledger, one PR each:

- predictor graph node count (1,062 kernels per replay at c16 on H100, about 1 ms of
  in-graph gaps);
- talker thread eager launches outside graphs (about 81 per step on H100);
- `_zero_slot` in `codec_state_arena.py` (about 35 of 47 codec launches);
- vocoder host syncs in `_run_initial_batch` / `_run_followup_batch`;
- preprocessing launches (speaker embedding, filterbank, prompt build).

Order comes from the ledger, not from this list.
