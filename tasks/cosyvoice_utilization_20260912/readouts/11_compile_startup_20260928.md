# Readout 11: what #2372 (DiT torch.compile default on) costs at startup, and why

Node: radix H100 80GB (host-85-234-79-221), torch 2.13.0+cu130, sglang 0.5.20, main 7dc8909e7.
Raw runs: `artifacts/radix-h100-20260928/` (c1 probe, h4 pair).

## 1. Startup, phase by phase (stage3/c1_compile_startup.py, vocoder factory in one process)

| run | total | load | native compile | graph capture | packed compile | warmup |
|---|---|---|---|---|---|---|
| compile off | 45.2 s | 17.5 s | | 26.9 s | | 0.6 s |
| compile on, cold cache | 216.0 s | 22.3 s | 120.2 s | 11.4 s | 54.9 s | 0.4 s |
| compile on, warm cache | 148.4 s | 11.5 s | 82.4 s | 11.2 s | 42.6 s | 0.4 s |
| compile on, warm cache | 147.7 s | 10.7 s | 82.2 s | 11.3 s | 42.9 s | 0.4 s |

The warm numbers match CI (job 108913098775: native 79 to 81 s, packed 41 s on three boots
with one cache directory). A warm cache removes 68 s of 171 s.

Dynamo's compile table, warm run (seconds, nested): compile_inner 124.5, call_user_compiler 80.5,
create_aot_dispatcher_function 79.8, bytecode_tracing 41.7, aot_collect_metadata 22.7,
fw_compiler_base 14.1 (Inductor, 5 of 6 graphs hit the FX graph cache).
Six graphs: the native DiT four times (streaming True and False, each split by the graph break
at the chunk mask, which is wrapped in torch.compiler.disable), the packed DiT twice (causal and
full contracts, the same function body with a different max_seqlen_q).

## 2. Why the cache saves so little

- AOTAutograd cache: bypassed on all 9 compiles, every boot. Reason logged by
  torch._functorch._aot_autograd.autograd_cache: "Unsupported call_function target
  _enter_autocast". x_transformers 2.28.4 decorates RotaryEmbedding.forward and
  apply_rotary_pos_emb with @autocast('cuda', enabled=False); both compiled paths call it
  (native through DiT.forward, packed through rope_tensor_geometry), so every graph holds an
  autocast region and the AOT cache refuses it. AOT tracing (about 65 s warm) repeats each boot.
- Dynamo bytecode tracing (42 s): no disk cache holds it. Most frames trace twice
  (compile ids 1/0 and 1/0_1: a restart), cause not yet read.
- Six graphs where one block would do: the packed DiT unrolls 22 identical blocks per contract,
  twice.

## 3. Main with compile off crashes at startup (the documented opt out)

`--vocoder.factory.enable_dit_torch_compile false` on main: the buffered Flow CUDA graph capture
fails with cudaErrorStreamCaptureInvalidated. Mechanism, proven on the node: after
load_cosyvoice3_flow_hift returns, two onnxruntime InferenceSessions of CosyVoice's own frontend
(the CUDA speech tokenizer and campplus) are unreachable but alive in a reference cycle; one
gc.collect() frees them (2 alive, 15,036 objects collected, 0 alive). A collection that lands
inside the capture runs the CUDA session's destructor there (cudnnDestroy in the log at the
failure time) and invalidates the graph. With compile on, the 80 s compile collects the cycle
before the capture, which hides it. Fix: one gc.collect() before the capture, as omni's Qwen3-TTS
codec capture does (fix/cosyvoice-flow-capture-gc 8ba709c6b). Confirmed: the compile off arm of
h4 boots on it.

## 4. What the compile buys (h4: one revision 8ba709c6b, compile off against on, same time, full set)

| point | compile off | compile on | delta |
|---|---|---|---|
| streaming c16 req/s | 7.563 | 8.903 | +17.7 % |
| streaming c16 RTF / TTFP mean | 0.478 / 0.999 s | 0.407 / 0.827 s | |
| buffered c16 req/s | 10.184 | 14.695 | +44.3 % |
| buffered c16 RTF / latency p95 | 0.351 / 2.16 s | 0.241 / 1.49 s | |

Seeded c1 streaming 57 of 64 byte identical, the rest AR token changes (lengths differ).
Both compiles pay for themselves: the fix has to keep the kernels and drop the startup.

## 5. First fix attempt: per block regional compile (9505bc641), rejected

Startup 216 s cold and 148 s warm on main, 40 s and 34 s here (2 graphs, no recompile, no break).
But h5 (main+gc 8ba709c6b against it, same time, full set): streaming c16 8.898 against 8.119
(-8.8 %), buffered c16 14.424 against 12.853 (-10.9 %), streaming c1 2.099 against 1.744.
One Flow call (stage3/c2_hop_cost.py): +30 ms of host time per call (220 compiled region
entries per call, about 135 us each, against 10 on main) and +69 kernels per step with 7 to 26 %
more device time (eager prologue and epilogue; on the native path the per block attention mask
work that a whole forward compile shares across the 22 blocks).

## 6. Whole forward compile, repeated block traced once (v5, 4a4318b17)

Packed: PackedDiT.forward compiled whole with the block as torch.compiler.nested_compile_region,
inlined back into one flat graph (dynamo.config.inline_invoke_subgraph); RoPE computed once per
Flow call outside the graph (the AOT cache then hits); one forward, the conv position embed,
the norms and Mish written once with custom ops chosen under torch.compiler.is_compiling(); one
contract for hops and finals; automatic dynamic, warmed on two batch shapes. Native: DiT.forward
compiled whole with DiTBlock.forward as one nested region (bound per instance), the chunk mask
compiled (no graph break, no restart), automatic dynamic, warmed on two batch and frame sizes.

Dead ends on the way, torch 2.13: nested regions under dynamic=True fail (KeyError on the
symbolic hidden size; the layer norm eps becomes SymFloat); a nested region with a graph break
inside fails (KeyError); a nested region left as a called subgraph costs about 1.7 ms of host time
per step (v3).

Startup, vocoder build (stage3/c1_compile_startup.py):

| build | warm | cold |
|---|---|---|
| main | 148 s | 216 s |
| v5 | 70.4 s (native 29.5, packed 17.6, capture 11.4) | 141.3 s |
| compile off | 45 s | 45 s |

One Flow call (stage3/c2_hop_cost.py), v5 against main: packed hop and final equal at 1 row
(53.5 against 53.5 ms) and 1.5 to 2 % faster at 16 rows; buffered 150.0 against 169.7 ms at
16 x 356, and 245.7 against 283.5 ms (16 x 576), 80.4 against 91.6 ms (5 x 544, replayed).

h6, main 7dc8909e7 against v5, same time, full set, compile on both:

| point | main | v5 |
|---|---|---|
| streaming c16 req/s | 8.605 | 8.780 (+2.0 %), latency p95 2.38 against 2.65 s |
| buffered c16 req/s | 14.353 | 13.290 (-7.4 %) |
| streaming c1 req/s | 2.080 | 1.973 (-5.1 %) |
| buffered c1 req/s | 2.245 | 2.229 |

Identity: streaming c1 61 of 64 (two AR length changes, one sample with a whole file difference of
at most 0.0015 on near silence); buffered c1 0 of 64, as expected: the native compile is inexact
against eager on both trees (max abs 1.7 on main, the same order between main and v5), so that
path needs WER and similarity, not byte identity. The packed GPU parity test (torch.equal against
eager, both modes) passes on v5; 211 unit tests pass on the node.

Open: buffered c16 and streaming c1 regress in serving while every isolated call is as fast or
faster. h7 swaps the cards and logs recompiles in both servers.

## 7. v5 and v6 rejected; step 1 on main's compile structure

v5/v6 regressed in serving because every nested region guards on the size of
torch.utils._pytree.SUPPORTED_NODES, which the engine grows after the vocoder compiles (h8: buffered
c16 -5.3 %, streaming c1 -6.8 %). Dropped. The plan, SGLang's compile mechanics read whole at
v0.5.20, and the step 1 gates (d04f4400f to 36f14d351: the dynamic=True copy before FA3, found in
Inductor's generated code; automatic dynamic; the eager FA3 dispatch) are in
`tasks/cosyvoice-perf/COMPILE_2372.md`. At 36f14d351: warm startup 123.9 s against 149.1 s, one
call equal or faster at every size compile on and off, h9 streaming c16 8.787 against 8.550 req/s
with no recompile while serving.

## 8. 2026-09-29: step 1 serving complete, startup root causes

New lease, same H100 host. Raw: `artifacts/radix-h100-20260929/` (h10 to h14, probes c12 to c20).

Step 1 (36f14d351) against main 7dc8909e7, same time pairs, full set at c16, 64 seeded at c1:

| point | main | step 1 |
|---|---|---|
| streaming c16, h10 (swapped) / h11 (ledger) | 8.595 / 8.726 | 8.819 / 9.135 req/s |
| buffered c16, h12 / h13 (swapped) | 14.249 / 14.868 | 13.983 / 15.221 req/s |
| streaming c1, h14 | 1.962 | 2.024 req/s |
| buffered c1, h14 | 2.190 | 2.202 req/s |

Buffered runs identical code on both trees; the card 0 arm was about 2 % faster in both
orientations. Streaming inter chunk p99 (h9 +7 %): higher from p95 up in h9 and h10, lower in
h11; per final call device time (ledger) step 1 is faster in every frame bucket (-11 % to
-1.6 %), per step too; the tail moves with batch composition. Identity: streaming c1 60 / 64,
every difference a length change (AR tokens); buffered 55 / 64 with identical Flow code.

Startup causes, each fix emulated on the probe (stage3/c7, c8, c9, c10, c11):
- libdevice: torch 2.13 sets TRITON_LIBDEVICE_PATH lazily on the first Triton compile
  (runtime/compile_tasks.py:74-106, async_compile.py:453-455) and every FX entry records the
  libdevice hash (codecache.py:2086-2094). Warm native graphs hit without compiling, so the
  first packed lookup sees Triton's bundled file while its entry holds CUDA's: miss, recompile,
  one more entry per boot. Pinned before the first compile: packed warm 16 -> 7 s, all hits.
- chunk mask: the "NaN compare" is Inductor's range analysis of a trunc division by a symbolic
  chunk size; DiT.forward passes a module attribute, which Dynamo specializes, and then the mask
  compiles exactly (0 of 360 mismatches). No graph break, native warm 83 -> 58 s, forward 3 to
  11 % faster, distance to eager unchanged.
- RoPE: x_transformers' @autocast(enabled=False) regions (rotary forward, apply_rotary_pos_emb)
  bypass the AOTAutograd cache; replacements without the region are bit identical in eager (12
  shapes). Native warm 58 -> 35 s.
- All three: warm 66.5 s (main 149, step 1 123), cold 158.7 s (main 196). The one remaining
  miss is a Dynamo restart of native frame 0/0: under dynamic=True the LayerNorm eps become
  SymFloats that the tensorify pass cannot handle (_tensorify_python_scalars.py:458).
- Native on automatic dynamic with maybe_mark_dynamic hints (stage3/c11), with the three
  above: warm 48.4 / 51.0 s, cold 129.7 s, all 5 AOT hits, no restart; one call within about
  3 % of the mask only variant and faster than main at 6 of 8 shapes; distance to eager as
  main's. Serving and WER / SIM not yet measured.

One row regression and a variance source (c21 to c25, h15):
- At one row both trees are host bound (wall about 64 ms, device 30 to 53 ms). Step 1 launched
  230 more kernels per call (6,305 against 6,075): its in-place RoPE write becomes a full copy
  of q and k before the FA3 custom op when compiled. Wall slower at 20 of 30 c1 shapes (up to
  +1.6 %) with device faster at all 30.
- Fix (exp/cosyvoice-rope-where 841fd8c1f): the compiled path rotates with one pointwise
  torch.where over every channel (a cat did not fuse: 440 copy launches, rejected). Exact
  (torch.equal parity test). 5,864 kernels; c1 shapes wall -5.6 % mean against main, 30 of 30.
- Triton's pointwise autotune picks the GELU kernel's config per process: 6.5 or 10.0 ms per
  16 row hop whatever the code (gate1 main 10.0, gate4 step 1 6.5, c25 main 6.6, c25 where
  10.1), a few percent of a run decided at warmup. h15 (streaming c16 9.076 against 9.074,
  c1 1.963 against 1.995) is not a clean code comparison until that is pinned.

Implemented on fix/cosyvoice-dit-compile-startup 0b354a67c (user chose pointwise autotune off):
per compile triton.autotune_pointwise=False for both DiT compiles; the native changes (libdevice
set before any lookup, RoPE without autocast regions, chunk mask compiled, automatic dynamic);
the compile default yields to an explicit TensorRT choice. Tests 220 passed. Startup (fresh
cache, alone): warm 49.0 / 57.7 s, cold 147.4 s (main 149 / 196). Serving against main, both
orientations, shared cache: streaming c16 8.375 / 8.669 and 9.082 / 9.374 req/s, buffered c16
14.883 / 15.362 and 15.069 / 15.721, streaming c1 1.997 / 2.096, buffered c1 2.193 / 2.225.
WER / SIM (full set): buffered 1.725 / 1.708 % and 69.99 / 69.86 (h18), 1.348 / 1.532 % and
70.16 / 70.08 (h19); identical code differs by up to 0.38 WER points and 0.22 SIM points across
runs. Raw: artifacts/radix-h100-20260929/h18, h19, probes3.

History rewritten as four tested commits on ce7ccc015 (96fe1949a head). TensorRT measured (h20,
same code, compile against TensorRT 11 fp32 as the cookbook installs it, same time): streaming
c16 9.271 / 2.162 req/s, buffered c16 14.917 / 6.026, streaming c1 2.063 / 1.571, buffered c1
2.195 / 2.032; WER / SIM equal within run spread. Upstream recommends fp32 TensorRT for its one
request at a time serving (Flow batch 1 everywhere) and never compared torch.compile.

TensorRT (#2402): the exclusion is forced only for the estimator slot; #2372's default made an
explicit TRT opt-in fail, which #1969's review had prevented. The engine is batch 2 (N requests
are N serial calls per step), full attention (streaming loses the chunk mask; the hub ONNX is
exported without streaming), fp32 under TRT 11. The cookbook numbers predate CUDA graphs and
PackedDiT; untested in CI. Details in tasks/cosyvoice-perf/COMPILE_2372.md.

## 9. Not a problem

OMP_NUM_THREADS=1 at spawn: Dynamo's GLOBAL_STATE guard includes num_threads (a 4 to 1 change
recompiles, checked), and sglang's load_model sets one thread after the vocoder is built. The pin
makes the compile see the serving thread count.
