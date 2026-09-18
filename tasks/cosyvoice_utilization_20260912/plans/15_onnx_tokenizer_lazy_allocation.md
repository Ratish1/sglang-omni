# Plan 15: the speech tokenizer allocates after the KV pool is sized

The request failures on the 24 GB card ("CUBLAS failure 3: the resource allocation failed", 1 of 32
at c4, 9 of 32 at c8, `../slices/11_2_RESULT_20260917.md`), fixed where they happen instead of by
resizing the pool. Written 2026-09-18 from whole file reads of upstream main `7b49bc4b8` and of
ONNX Runtime `v1.30.0` and `v1.22.0`. The technique is NOT chosen yet: section 4 lists the
measurements that choose it, and the box was unreachable when this was written.

## 1. The defect, in order of events

```text
engine build
  before_memory_pool                           scheduling/bootstrap.py:196-199, engine_builder.py:128
    SpeechTokenizerV3(...)  session BUILT      engine_builder.py:164-168, utils.py:67-69
    (no session.run anywhere at startup)
  SGLang profiles free memory, sizes the pool  takes all but the slack: 1.4 to 1.9 GB left on 24 GB
first requests
  preprocessing stage, max_concurrency 8       config.py:141, stages.py:1203-1207
    asyncio.to_thread, default executor        simple_scheduler.py:288  (no max_workers anywhere)
      get_or_encode, single flight per KEY     reference_encoder.py:304-371
        encode_one                             request_builders.py:285-308
          session.run   FIRST RUN, from a pool thread     utils.py:92
            ORT creates a device stream: cudaStreamCreate, cublasCreate, cudnnCreate
            ORT arena grows for the activations
```

| fact | where |
|---|---|
| with our provider options every device stream ORT creates owns a new CUDA stream, a new cuBLAS handle and a new cuDNN handle | ORT `cuda_stream_handle.cc:78-82`, registered at `:266-273`; `own_flag` is true unless `use_ep_level_unified_stream_` |
| that flag is false by default, true only for a user compute stream, an external allocator, CUDA graphs, or the option `use_ep_level_unified_stream` | ORT `cuda_execution_provider.cc:361-375` |
| streams are created at Run, not at session construction; construction creates one handle pair on the constructing thread | `session_state.cc` `AcquireDeviceStreamCollection`; `cuda_execution_provider.cc:3634-3642` |
| how many streams exist depends on the ORT version: v1.22.0 keeps one pool for all threads (count follows peak concurrent runs), v1.30.0 keys the pool by thread and keeps an entry until the thread dies (count follows the distinct threads that ever ran) | v1.22.0 `session_state.cc:1731-1747`; v1.30.0 `session_state.cc:34-40, 1962-2014` |
| omni does not pin ORT: `onnxruntime-gpu>=1.17` | `pyproject.toml:67` |
| cuBLAS allocates device memory in `cublasCreate` (a workspace pool, 4 MiB before Hopper, 32 MiB from Hopper), and ORT never calls `cublasSetWorkspace` | NVIDIA cuBLAS docs 2.4.1, 2.4.8; zero hits in the ORT tree |
| the ORT arena extends by powers of two, has no limit, and is never shrunk unless a run option asks | `bfc_arena.h:57-63`, `inference_session.cc:3576-3581`; our `SessionOptions()` sets neither (`utils.py:51-69`) |
| the 1,050 MiB measured for this session is construction alone; the probe never ran it | `../stage2/vocoder_memory.py:124-133`, `../MEMORY_TRACE_20260917.md` section 3 |
| the tokenizer rejects audio over 30 s, so its largest activation set is bounded | `utils.py:85-88` |
| the speaker encoder is on the CPU provider and takes no device memory | `utils.py:117-121` |

So everything the tokenizer allocates beyond construction (one handle set per stream, the arena's
growth, cuDNN's own allocations) lands after SGLang has taken the free memory, and the amount grows
with concurrency or with thread count. This is a startup ordering defect, not a pool size defect:
any consumer that makes the pool larger or the card smaller brings it back.

## 2. What is not known

| # | unknown | why it matters |
|---|---|---|
| U1 | the ORT version on the box | decides whether streams follow concurrency or thread count |
| U2 | device memory of a first run, of the 30 s run, of 8 runs at once, of 32 threads | sizes the defect: tens of MiB or hundreds |
| U3 | whether `use_ep_level_unified_stream=1` removes the growth from handles, and what it does to 8 concurrent runs (they share one stream and one handle pair) | decides whether the option alone is a fix; cuBLAS documents a shared handle as thread safe but not recommended |
| U4 | S3 tokenizer time per reference on the 4090 under the option | 5 to 7 ms on the H100 (`../ROADMAP_20260915.md:57`); serial runs must stay far below the request rate |

## 3. Candidate techniques, none chosen

| technique | what it does | cost |
|---|---|---|
| T1: run the tokenizer once in `before_memory_pool`, at the 30 s bound | the arena's largest single run and one stream's handles exist before the pool is profiled, so SGLang's own arithmetic accounts for them | one run at startup; covers one stream only |
| T2: `use_ep_level_unified_stream=1` | no stream ever creates handles after construction | concurrent runs serialize on one stream; correctness under 8 threads is U3 |
| T3: `gpu_mem_limit` and `arena_extend_strategy=kSameAsRequested` | bounds the arena and stops power of two overshoot | a limit is a constant that needs a measurement to pin; a run over it fails |

T1 with T2 is the pair that would leave nothing lazy; whether both are needed is what section 4
measures. A lock around `session.run` is not on the list: with the v1.30.0 pool a lock still leaves
one stream per distinct thread.

## 3a. Measured 2026-09-18, RTX 4090 D, onnxruntime-gpu 1.30.0, then parked

Parked by decision: this is a correctness defect of small cards, not a throughput lever, and the
work goes to GPU occupancy first. No code was written. What the probe found, MiB added per step:

| step | default | unified stream | same as requested | serial, warm 30 s first, unified | serial, warm first, unified, same as requested |
|---|---|---|---|---|---|
| session built | 1,036 | 1,052 | 1,044 | 1,044 | 1,044 |
| warm run, 30 s | | | | 1,024 | 138 |
| first run, 3 s | 10 | 0 | 40 | 0 | 0 |
| second run, 3 s | 0 or 1,024 | 0 | 22 | 0 | 0 |
| one run, 30 s | 1,024 or 0 | 1,024 | 116 | 0 | 108 |
| 8 at once, 10 s | 160 | 74 | 342 | 0 | 0 |
| 8 at once, 30 s | 0 or 1,024 | 1,024 | 1,004 | 0 | 0 |
| 32 threads, 3 s | 516 | 244 | 484 | 0 | 0 |
| after the build, total | 1,710 to 2,734 | 2,366 | 2,008 | 1,024, all before the first request | 246, of which 108 after the warm run |

- U1: 1.30.0, the per thread stream pool.
- The handles are the small part, 10 to 20 MiB per stream. The arena is the large part: the
  default strategy extends by 1 GiB once it has grown, whatever the run needs (a 30 s run needs
  116 MiB), and eight concurrent 30 s runs really need about 1 GiB because each holds its own
  activations.
- Concurrency buys no time on this session: warm, eight 10 s runs in a row take 72 to 92 ms, the
  same eight at once 116 to 159 ms.
- Serialized, with the unified stream and one run at the 30 s bound before the first request,
  nothing is allocated afterwards with the default arena. With same as requested the resident
  cost falls from 1,024 to 246 MiB and one later 30 s input still extended by 108 MiB.

If it is unparked, the technique the numbers support is: one run in flight on this session,
`use_ep_level_unified_stream`, and the 30 s warm run inside `before_memory_pool`; the arena
strategy is then a choice between 1 GiB resident with zero growth and 0.25 GiB with a small one.

## 4. The measurement that chooses

`../stage2/onnx_tokenizer_memory.py` (pushed, 8fe1e22a1), one process per option set, on a free
card: device memory after the CUDA context, the session build, a first and second 3 s run, a 30 s
run, 8 runs at once at 10 s and at 30 s, and 32 runs at once from 32 threads. Run with the default
options and with `use_ep_level_unified_stream=1`. It prints the ORT and torch versions.

Then, with the technique chosen: the c8 point of `../slices/11_2_RESULT_20260917.md` on unmodified
main pool sizing (no `kv_cache_bytes`), 32 samples, main against the branch. Pass: zero CUBLAS
lines and zero failed requests on the branch where main fails, and the c1 identity gate, since the
tokenizer's output feeds the prompt.
