# Persistent predictor layer kernel: stage 1 design (11-14)

## Scope and go/no-go

- **What.** One Qwen3-TTS code predictor decoder layer as one persistent CUDA C++ kernel, for up to 16 rows of one token each at one cache slot `pos`.
- **What it replaces.** The 5 Triton launches of `FusedPredictorLayers` for one layer: `norm_qkv_rope_store`, `decode_attention`, `o_proj` `gemv_add`, `norm_gate_up_silu` and `down` `gemv_add`.
- **Pass bar.** Measured in `predictor_bench.py`:
  - **Speed:** at most about 12 us a layer at 1 and 16 rows, against 28.5 us for the Triton layer (142.5 / 5) and a 9.4 us HBM floor.
  - **Numerics:** fp32-reference error no larger than the Triton path's.

  If speed misses, stop and re-analyse before building the 5-layer, lm_head and sampler stages.

## Why this shape (from the study)

Each Triton operator spends 4 to 6 us on fixed latencies: the weight tile load, MMA on padded rows, and a global split-K handshake. The kernel boundaries also force a full drain each time. The measured fixes are:

1. **Weights stream into shared memory ahead of need,** by a producer warp that never waits on activations.
   - Precedent: `dsv3_fused_a_gemm.cuh` (`GmemLoaderA` issues weight loads without `cudaGridDependencySynchronize`).
   - Precedent: Hazy's Llama-1B megakernel.
2. **Swap-AB tensor-core MMA.** Weights are A (16 output features an MMA) and tokens are B (8 a tile), so batch 1 wastes nothing.
   - Precedent: `dsv3_fused_a_gemm.cuh` `hmma_16_8_16_f32acc_bf16ab`, 2.47 us for the 8.4 MB qkv projection at one row.
3. **No split-K across CTAs.** A tile's K is split across the CTA's warps and reduced in shared memory, as `MmaComputer::epi` does. Weights are already resident when the input arrives, so the K split serves only the compute, which is small.
4. **One handoff a phase through a global counter.**
   - CUTLASS `GenericBarrier::arrive_inc` / `wait_eq` (`cutlass/barrier.h`).
   - SGLang `device::atomic::Event` (`sgl_kernel/atomic.cuh`).

## Phases of one layer launch

`x` is the layer input (M x 1024 bf16), also the attention residual. The rounding at each step matches the Triton path (`sglang_omni/utils/predictor_layers.py`).

| Phase | Work items | Weight chunks (16 rows x 1024 k, 32 KB) | Input | Output |
|---|---|---|---|---|
| P1 qkv | 256 tiles of 16 qkv rows | 1 per tile | x, after PDL wait | `qkv_raw[n][4096]` = bf16(acc * rstd_in[n]) |
| P2 attention | M x 8 (token, kv group) | none | qkv_raw, after E1 | k and v into the cache at `pos`; `attn[n][2048]` |
| P3 o_proj | 64 tiles of 16 hidden rows | 2 per tile (K 2048) | attn, after E2 | `mid[n][1024]` = bf16(bf16(acc) + x) |
| P4 gate_up | 384 tiles: 8 gate rows + the same 8 up rows | 1 per tile | mid, after E3 | `act[n][3072]` = bf16(silu(g) * u), g = bf16(acc_g * rstd_post[n]), u likewise |
| P5 down | 64 tiles of 16 hidden rows | 3 per tile (K 3072) | act, after E4 | `out[n][1024]` = bf16(bf16(acc) + mid) |

P2, for one (token n, kv group g):
- q heads 2g and 2g+1, k head g and v head g are read from qkv_raw.
- The q and k heads are RMS-normed over 128 (fp32 sum of squares of the bf16 values, times the weight, then bf16), then given neox rope from the fp32 cos/sin cache (then bf16).
- k and v are written to `cache[n][pos][g]`.
- Scores are q . k_s * 128^-0.5 over slots s <= pos, in fp32; weights are exp(s - max).
- out = (sum_s bf16(w_s) v_s) / sum_s w_s, then bf16.

Counters E1 to E4 are the four phase handoffs. Each consumer waits until all of the previous phase's producers have arrived.

## Weight packing (once, at load)

A chunk is the shared-memory image of 16 weight rows by 1024 k, so one `cp.async.bulk` of 32 KB moves it. It is laid out `[kb 64][half 2][row 16][j 8]`, element = `W[row0 + row][k0 + 16 kb + 8 half + j]`.

Each 8x8 bf16 matrix is 128 contiguous bytes, so the `ldmatrix.x4` a warp issues for one A k-step reads four contiguous 128 B runs, with no bank conflicts. Lane l points at matrix m = l / 8, row (l % 8) + 8 (m % 2), half m / 2.

The registers it returns are a0a1 (rows 0-7, k 0-7), a2a3 (rows 8-15, k 0-7), a4a5 (rows 0-7, k 8-15) and a6a7 (rows 8-15, k 8-15). That is the A fragment order of `mma.m16n8k16.row.col` in PTX ISA §9.7.14.5.8. `gate_up` tiles interleave 8 gate rows with the same 8 up rows, so one thread's C fragment holds gate (row g) and up (row g + 8) of the same feature and token.

## Activations (B operand) in shared memory

Tokens are padded to 8 (M <= 8) or 16 with zero rows, laid out `[n_tile][kb][half][n 8][j 8]`.

One 16-byte global vector (8 k of one token) is exactly one row of one 8x8 matrix, so staging is one `ld.global.cg` 16 B load and one 16 B shared store per vector. For P1 and P4, staging also:
- multiplies by the norm weight in fp32 and rounds to bf16, as `scaled_tile` does;
- accumulates the token's fp32 sum of squares for its rstd.

`ldmatrix.x4` on `[n 8][j 8]` gives b0b1 (k 0-7) and b2b3 (k 8-15) for one k-step and the next pair for the following one. That is the B fragment order for `.col`, the same mapping `MmaComputer` uses.

## CTA layout

- One CTA per SM, grid = SM count (132 on H100), `__launch_bounds__(288, 1)`.
- **Warps 0-7, compute.** For a tile, warp w takes k-steps w, w + 8, and so on. The 8 warps then reduce their 16 x M fp32 partials through shared memory, and the epilogue writes the outputs.
- **Warp 8, producer.** Lane 0 walks the CTA's chunk list in schedule order. For each chunk it waits for the slot to be empty (mbarrier `empty[s]`), arms `full[s]` with `arrive.expect_tx(32768)` and issues `cp.async.bulk`.
  - It starts before `griddepcontrol.wait`: weights do not depend on the previous kernel.
  - Primitives: `cp_async_bulk_1d_load`, `mbar_arrive_expect_tx`, `mbar_wait_parity` from `kda_fused_decode.cuh` and `sgl_kernel/mbarrier.cuh`.
- **Ring.** 3 stages of 32 KB. `full[s]` counts 1 arrival plus tx bytes; `empty[s]` counts 8 arrivals, one per compute warp. Parity follows `sgl_kernel/mbarrier.cuh`: consumer-first waits start at 0, producer-first at 1.

Shared memory:

| Part | Bytes |
|---|---|
| ring | 96 KB |
| B | 96 KB (16 tokens x 3072 x 2) |
| warp partials | 8 x 16 x 16 x 4 = 8 KB |
| rstd and attention scratch | 2 KB |
| mbarriers | less than 1 KB |
| **total** | **about 203 KB of the 227 KB opt-in limit** |

## Schedule (host, deterministic)

For each phase in order, each tile goes to the CTA with the fewest weight chunks so far, ties to the lowest index. 960 chunks over 132 CTAs gives 7 or 8 each. P2 items go to CTAs in order.

The device table is int16 `[cta][item]` of (phase, tile), with a count per CTA. The producer and the compute warps walk the same list, so they agree on chunk order.

## Synchronization contract

- **Producer arrive** (CUTLASS `arrive_inc`): `__syncthreads()` across the compute warps (named barrier 1, 256 threads), then thread 0 runs `fence.acq_rel.gpu` and `red.release.gpu.global.add.u32`. Per the CUTLASS comment, the fence releases data that other threads wrote before the barrier.
- **Consumer wait** (CUTLASS `wait_eq`): thread 0 spins on `ld.acquire.gpu`, then the named barrier. Cross-CTA data is then read with `ld.global.cg`, at L2, never a stale L1.
- **Counter reset.** Counters are SGLang `Event`s with `wait_multi(num_producers, num_consumers)`: the last consumer subtracts the word, so no host re-zero is needed. The host passes the consumer count of each phase from the schedule.
- **Across launches.** Every counter arrive happens after `griddepcontrol.wait`, which waits for the previous grid to complete. So the next launch's producers can never arrive before every consumer of this launch has been released.
- **Residency.** Grid = SM count and one CTA an SM. A CTA that waits on producers not yet resident only delays: other kernels on the card (the vocoder) do not wait on us, so their SMs free up. That is the same assumption `PackedPostAttention` makes in serving.

## Test plan (`predictor_bench.py`, real weights)

1. One layer at 1, 8 and 16 rows. Max abs and mean abs against the fp32 reference layer, next to the Triton layer's.
2. Timing: one layer back to back and 5 layers in a chain, in a CUDA graph, against Triton (142.5 / 146.4 us for 5 layers).
3. In-kernel `globaltimer` stamps per phase in a debug build, since Nsight Compute cannot replay a kernel whose CTAs wait on each other.
4. `compute-sanitizer --tool racecheck` and `--tool synccheck` on one launch.
