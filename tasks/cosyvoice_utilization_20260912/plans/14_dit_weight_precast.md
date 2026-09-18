# Plan 14: cast the DiT weights once, not on every step

Roadmap row 3.2, the exact half of it. Written 2026-09-18 from whole file reads of upstream main
`7b49bc4b8` (`packed_dit.py`, `stages.py`, `streaming_vocoder.py`), the vendored CosyVoice DiT
(`inputs/external_sources/CosyVoice/cosyvoice/flow/DiT/dit.py`, `modules.py`) and PyTorch at the
pinned tag `v2.13.0`. Nothing here is measured on the 4090 yet; the gates are in section 5.

## 1. The defect

| fact | where |
|---|---|
| omni pins `torch==2.13.0` | `pyproject.toml:22` |
| at that tag autocast caches a cast weight only when inference mode is off: `can_try_cache` ends with `!c10::InferenceMode::is_enabled()`; otherwise every call runs `arg.to(to_type)` | pytorch `aten/src/ATen/autocast_mode.cpp:129-133, :152` |
| every Flow entry point is decorated `@torch.inference_mode()` | `stages.py:629, 689, 771, 783, 802`, capture and replay `:393, :446` |
| the Flow weights are float32 and the shipped vocoder dtype is bfloat16 autocast: `fp16=(dtype == "float16")` only turns on CosyVoice's own autocast, it halves nothing | `stages.py:2045, 2053-2058`; vendored `cli/model.py:403-426`; measured 1,267 MiB, 332.3M parameters, all float32 (`../MEMORY_TRACE_20260917.md` section 4) |
| so each Linear and Conv1d of the DiT casts its weight and its bias to bfloat16 on every call, ten Euler steps per Flow call | follows from the three rows above |
| measured on the H100 at `cc85ddaa9`: 4,588 cast launches per Flow call, 24 percent of 19,072; "autocast weight casts" 19 percent of device time at 1 row; casts 10 to 12 percent at 16 rows | `../ROADMAP_20260915.md:39-45, :82` |

`../MEMORY_TRACE_20260917.md:85` says autocast "caches the casts". That sentence is wrong for this
process and is corrected by this plan: under inference mode nothing is cached.

In the buffered graph the casts are captured and replayed as device work, which is part of the
"copies 22" family of the graphed call (`../ROADMAP_20260915.md:46`).

## 2. Why a one time cast moves no number

Every parameter of the DiT sits in an `nn.Linear` or an `nn.Conv1d`:

| module | parameters | consumer |
|---|---|---|
| `TimestepEmbedding.time_mlp` | two Linear | `F.linear` |
| `InputEmbedding.proj` | Linear | `F.linear` |
| `CausalConvPositionEmbedding.conv1, conv2` | two Conv1d, groups 16, kernel 31 | `F.conv1d` |
| `AdaLayerNormZero.linear`, `AdaLayerNormZero_Final.linear` | Linear | `F.linear` |
| `Attention.to_q, to_k, to_v, to_out[0]` | Linear | `F.linear` |
| `FeedForward` | two Linear | `F.linear` |
| `long_skip_connection`, `proj_out` | Linear | `F.linear` |
| both `LayerNorm` | none, `elementwise_affine=False` | |
| `RotaryEmbedding.inv_freq` | a buffer, not a parameter | autocast disabled rope, reads the buffer's dtype |

(`modules.py:115-127, 230-266, 271-283, 606-616`, `dit.py:124-140`.) `linear` and `conv1d` are on
autocast's lower precision list, so under the bfloat16 context they already run on
`weight.to(bfloat16)` and `bias.to(bfloat16)`. Storing exactly those tensors gives the same operands
to the same kernels. The cast touches parameters only: `inv_freq` stays float32, which the rope
needs because it computes in the buffer's dtype.

The DiT is never called outside the autocast context when one is configured: `hop_batch`,
`leftover_batch`, `decode_batch`, `token2wav_chunk`, the graph capture and replay, and the compile
warmup all enter it (`stages.py:1412, 1508, 1538, 1554, 405, 506, 1161`).

## 3. Code plan

`stages.py`, `create_vocoder_executor`, after `load_cosyvoice3_flow_hift` and before
`compile_dit_backbone` and the graph capture, so neither traces a cast:

```python
    if autocast_dtype is not None:
        # note(ratish): autocast caches no weight cast under inference mode, so
        # each Linear and Conv1d recast its float32 weights on every Euler step.
        for module in flow.decoder.estimator.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                module.to(autocast_dtype)
```

`autocast_dtype` is None for a float32 vocoder and on MPS (`stages.py:2046-2052`), which keeps both
untouched. With the opt in TensorRT estimator the wrapper keeps the PyTorch DiT off its module tree
on purpose (`flow_estimator_trt.py:399-401`), so `modules()` does not reach it and the fallback
stays float32 and behaves as today: no gain there, no change either.

## 4. Decisions to make before code

| # | question | what I found |
|---|---|---|
| D1 | `compile_dit_backbone` takes its warmup dtype from `next(estimator.parameters())` (`stages.py:1142-1143`). After the cast that is bfloat16, while serving feeds float32 `x`, so the warmup would trace a dtype serving never sends. | opt in path, off by default. Options: take the dtype from the Flow's first parameter as `pack_flow_inputs` does (`stages.py:187-188`), or leave it and accept one recompile. I recommend the first. |
| D4 | which devices take the cast. Section 2 rests on autocast's op policy: `linear` and `conv1d` run in the lower precision. That is read for CUDA. The vocoder also runs on CPU (the unit tests build it with `device="cpu"` and the default bfloat16), NPU, XPU and MUSA, whose policies come from their own plugins and are not read. If a plugin kept either op in float32, a bfloat16 weight would meet a float32 input and raise. | I recommend `device_obj.type == "cuda"` next to the `autocast_dtype is not None` check, so no unread hardware path can break; other devices keep today's behaviour and can opt in when someone reads their policy. Cost: the factory tests run on CPU and do not reach the cast, so the contract is carried by the box gates (G-c1, G-launch), not by a CPU unit test. |
| D2 | HiFT | not in scope: its autocast is off by default (`hift_dtype="float32"`), so it casts nothing. |
| D3 | the activation side of row 3.2 (float32 layer norm output feeding three Linear, the AdaLN pointwise in float32) | changes numbers, needs the G0 protocol, stays a separate plan. |

## 5. Gates, 4090

| gate | what | pass |
|---|---|---|
| G-unit | `pytest tests/unit_test/fun_cosyvoice3 -q` on the box | all pass |
| G-c1 | seeded c1, 16 samples, main against the branch, `stage2/g1_stream_identity.sh` | byte identical on every sample the control reproduces. This is the falsifier of section 2. |
| G-launch | one hop call under the stage 1 profiler on both trees | cast launches fall by the weight share, about 3,200 of 4,588 (320 parameter tensors times ten steps, to be counted, V1) |
| G-mem | `stage2/vocoder_memory.py` | Flow weights fall by about half of the DiT's 1,263 MiB |
| G-c8, G-c16 | unseeded, `kv_cache_bytes` 2 GiB on both arms so the 24 GB card does not fail requests, alternating boots | no read worse than main's own spread; continuity reported |

## 6. Validation tasks

| # | unknown | how |
|---|---|---|
| V1 | the count of parameter tensors in the DiT, and so the launches removed | `sum(1 for _ in estimator.parameters())` on the box |
| V2 | the packed estimator and the padded path share one DiT object, so one cast covers both | `PackedDiT(flow.decoder.estimator)` at `stages.py:924`; confirm on the box that `flow.packed_estimator.dit is flow.decoder.estimator` |
| V3 | the box's torch is 2.13.0 | `python -c "import torch; print(torch.__version__)"` in the venv |
| V4 | the time this is worth at 1 row and at 16 rows on the 4090 | G-launch and G-c16; the H100 ledger's 19 percent at 1 row is the only number so far |

## 7. Results, 2026-09-18, RTX 4090 D, branch `slice/cosyvoice-3-2-dit-weight-precast` at 4a57f9a81

Decisions taken: D4 CUDA only, D1 the compile warmup reads the Flow's first parameter.

| check | result |
|---|---|
| V1 | 322 parameter tensors, 322 in a Linear or Conv1d, all float32; the one buffer is `rotary_embed.inv_freq`, float32 |
| V2 | `flow.packed_estimator.dit is flow.decoder.estimator`: True; the Flow's first parameter is `input_embedding.weight`, float32 |
| V3 | torch 2.13.0+cu130 |
| G-unit | 198 passed, 2 skipped |
| G-launch, G-mem and Flow identity, `stage2/dit_precast_hop.py`, one process, serving entry points, hops and finals at 1, 4 and 16 rows | mel bit identical on all 8 calls, each weight state reproduces itself; dtype copies 5,064 to 1,844 per call, kernels 18,218 to 14,998: 3,220 fewer, which is 322 times ten; allocated memory 1,624 to 993 MiB |

The wall times of that probe are not quoted: it profiled between its timings, and a process that
has run the profiler pays its callbacks on every later launch (the one call timed before any
profile read 200 ms, the same call class 257 to 263 ms afterwards). `stage2/dit_precast_time.py`
times one weight state per process with no profiler.

G-c1, end to end audio. Main boots disagree with each other on this box, always on both samples
of one reference voice at once and with different audio lengths, so the difference is upstream of
the Flow (the AR saw a different prompt):

| voice pair | main boots, 5 | branch boots, 3 |
|---|---|---|
| 1205005 | two audios: one boot of 5 has the other | two audios: 2 of 3 have the other |
| 103675 | two audios: one boot of 5 | two audios: 1 of 3 |
| 10933823 | one audio in all 5 | a second audio in 1 of 3 |
| the other 5 voices, 10 samples | one audio | the same audio |

Two of three branch boots produced only audio some main boot also produced. The first produced a
second audio for voice 10933823 that no main boot on record has. The Flow is proven bit identical
in isolation, and the cast touches nothing before the Flow, so this is read as the same per voice
boot instability; it stays an open observation until a main boot shows that second audio or the
source of the instability is found. For a change confined to the Flow, identity of the Flow's
output in one process is the stronger gate, because the end to end gate is confounded upstream.

Open: the source of the per voice boot instability (reference conditioning is the suspect: it is
computed once per voice per boot and cached, which is the pairing seen).
