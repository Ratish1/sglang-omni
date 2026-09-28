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

## 6. Not a problem

OMP_NUM_THREADS=1 at spawn: Dynamo's GLOBAL_STATE guard includes num_threads (a 4 to 1 change
recompiles, checked), and sglang's load_model sets one thread after the vocoder is built. The pin
makes the compile see the serving thread count.
