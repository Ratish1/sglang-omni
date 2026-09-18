# The vocoder stage, byte by byte, and the four defects in it

Measured 2026-09-17 with `stage2/vocoder_memory.py` on the moss box, card 4,
RTX 4090 D 23.52 GiB, upstream main `27a8293c`, `dtype=bfloat16`. The probe loads
the stage the way the factory does, one step at a time, and reads both torch's
accounting and the device's, so the gap between them names what is allocated
outside torch.

## The ledger

| step | device MiB | torch live | outside torch |
|---|---|---|---|
| CUDA context | 354 | 0 | 354 |
| Flow and HiFT weights | 2,854 | 1,612 | 78 |
| ONNX speech tokenizer | 1,050 | 0 | 1,050 |
| ONNX speaker encoder | 0 | 0 | 0 |
| Flow CUDA graphs, 54 shapes | 5,972 | 249 | 5,862 |

with the AR engine adding 610 MiB of weights, 9,780 MiB of KV pool and 130 MiB of
decode graphs on top. That is 21.5 GiB of a 23.5 GiB card, and four separate
defects account for most of it.

## 1. A captured Flow graph costs 130 MiB, and torch cannot see it

| shapes captured | 1 | 8 | 16 | 24 | 32 | 40 | 54 |
|---|---|---|---|---|---|---|---|
| device MiB | -976 | -36 | 1,034 | 2,076 | 3,136 | 4,178 | 5,972 |
| torch live MiB | 20 | 94 | 147 | 184 | 212 | 232 | 249 |
| torch cached MiB | -68 | -26 | 14 | 46 | 76 | 110 | 110 |

Linear at **130 MiB per shape**: the successive intervals give 133.8, 130.3,
132.5, 130.3 and 128.1 MiB. The first eight look free only because `capture`
ends with `torch.cuda.empty_cache()`, which returns about 1.1 GiB of load time
allocator cache and masks them.

Torch accounts for under 5 MiB of each 130. The rest is driver side memory for
the instantiated graph exec, which no torch level budget can see, ours or
SGLang's.

The cause is the granularity of what is captured. #1861 graphs the **whole ten
step Euler solver** per shape, which is about 19,072 kernel nodes
(stage 1 launch ledger), so 54 shapes instantiate roughly a million graph nodes.
SGLang reserves `len(prefill bs) * 8` MB per captured shape
(`arg_groups/memory_hook.py:321`) because it graphs one forward; at ten forwards
per capture this model costs 16 times that, and nothing in the config says so.

**What follows.** Graphing the DiT forward instead of the solver divides the node
count, and therefore the memory, by ten: about 13 MiB per shape. The Euler loop
stays in Python and pays ten graph launches per call instead of one, against the
19,072 kernel launches it pays today, so at the measured 7.3 microseconds per
launch the loop costs roughly 0.07 ms against a 140 ms launch floor. The same
table then costs 0.7 GiB instead of 7.0, and a table that covers the whole domain
becomes affordable: 17 derived buckets at 13 MiB is 0.2 GiB.

## 2. The AR KV pool is sized by free memory, not by the workload

9,780 MiB, 854,232 tokens, against a measured peak occupancy of 1,725 tokens at
c8 and 929 at c4 (the decode log's `#token` field; `token usage` reads 0.00 on
every line). The configuration's own ceiling, `max_running_requests` 32 times
`context_length` 4096, is 131,072 tokens, so the pool is 6.5 times a bound the
engine already declares and about 500 times what the workload reaches.

Detail in `MEMORY_20260917.md`: `mem_fraction_static` sizes only the slack term,
so the only upper bound is `max_total_tokens`, which the CUDA branch of
`generation_defaults` does not set while the MLX and MPS branches do.

## 3. The ONNX speech tokenizer holds a 1,050 MiB CUDA arena

One session, 1,050 MiB, every byte outside torch. `SessionOptions()` is
constructed with no `gpu_mem_limit` and no `arena_extend_strategy`
(`models/fun_cosyvoice3/utils.py:51-68`), so ONNX Runtime's CUDA arena grows by
powers of two and never returns memory. The speaker encoder costs nothing
because it runs on the CPU provider (`utils.py:120`).

This is also the allocation that fails first under pressure: the c4 and c8
request failures are `cublasCreate` inside this session.

## 4. Flow and HiFT weights are float32 under a bfloat16 autocast

`flow`: 1,267 MiB resident, 332.3M parameters, **all torch.float32**. The DiT
alone is 1,263 MiB. `hift`: 92 MiB, float32 plus a float64 F0 branch.

`load_cosyvoice3_flow_hift` takes `fp16=(dtype == "float16")` (`stages.py:2054`),
so the shipped `dtype="bfloat16"` loads float32 weights and every call runs under
`torch.autocast(bfloat16)`, which casts them on every call. Corrected
2026-09-18: nothing is cached, because torch 2.13.0 excludes inference mode from
the autocast weight cache (`aten/src/ATen/autocast_mode.cpp:129-133`) and every
Flow entry point runs under it. The weights are twice the size they need to be,
and the cast is paid ten times per Flow call rather than once at load.

Corrected 2026-09-18: casting only the Linear and Conv1d parameters moves no
number, since those are the tensors autocast already hands the kernels; a
blanket `.to(bfloat16)` would, because it also takes the rope's `inv_freq`
buffer. `plans/14_dit_weight_precast.md`.

## What this is worth

| defect | now | after | moves a number |
|---|---|---|---|
| graph granularity | 7.0 GiB | 0.2 to 0.7 GiB | no, replay is replay |
| KV pool bound | 9.8 GiB | 1.6 GiB | no, the tokens are never used |
| ONNX arena | 1.05 GiB | bounded, to be measured | no |
| weight dtype | 1.36 GiB | 0.68 GiB | no for the DiT's Linear and Conv1d parameters (plan 14, gated by c1 identity); HiFT not touched |

The first three are about 16 GiB of a 23.5 GiB card and none of them changes an
output. That is the headroom every other optimization in this task has been
waiting for.
