# Branch results: persistent predictor layer (11-14)

Branch `dev/qwen3-tts-predictor-persistent-20261114`, measured on host-62 (H100) with `persistent_layer_check.py` and `persistent_stamps.py`. One layer is layer 0 back to back (L2-warm). Five layers is a chain over layers 0 to 4 in one CUDA graph. Triton is `FusedPredictorLayers`, measured in the same run.

| Head | sglang_omni tree | 1 row: one layer / 5 layers / Triton 5 (us) | 8 rows | 16 rows | Numerics vs fp32 | Read |
|---|---|---|---|---|---|---|
| `c70fc4c61` stage 1 | `9219d448` | 28.7 / 181.8 / 139.1 | 30.8 / 187.0 / 145.6 | 40.8 / 229.1 / 146.6 | max abs equal to Triton's at 1, 8 and 16 rows. Bit-equal to Triton at 1 row | Latency chain. Staging and attention loads ran one after another, and the signal had two `MEMBAR.ALL.GPU` |
| `0bf8e35be` loads at once, one fence | `e2bf5f45` | 31.6 / 163.3 / 138.9 | 34.5 / 176.2 / 145.3 | 39.2 / 198.4 / 146.6 | same as stage 1 | 584 B stack, 168 registers. Unset `packed` arrays were carried across loop iterations |
| `a6900d3de` zeroed staged vectors | `54784df1` | 18.8 / 107.5 / 139.1 | 20.0 / 111.9 / 145.2 | 22.9 / 125.9 / 146.5 | same as stage 1 | No stack, 125 registers. Faster than Triton at every row count. Misses 12 us a layer |
| `1881c0af2` norms and rope in phase loads, raw residual | `ea038190` | not run: the lease ended | | | | |

## Stamp timeline, 1 row, L2-warm (us from the first CTA entry)

| Point | stage 1 | `a6900d3de` |
|---|---|---|
| Kernel span (stamped) | 28.9 | 21.0 |
| Operand constants before the PDL wait | 0.03 | 0.73 |
| qkv staging | 2.84 | 1.81 |
| qkv tiles (2 a CTA) | 1.35 | 1.61 |
| Signal (fence and add) | 0.83 to 1.15 | 0.52 to 0.76 |
| Handoff, last signal to first release | at most 0.26 (timer step) | at most 0.38 |
| Attention item | 6.63 | 2.23 |
| o_proj, gate_up, down staging | 1.17, 1.42, 1.74 | 0.44, 0.69, 0.59 |
| o_proj, gate_up, down tiles | 1.26, 2.66, 1.48 | 1.33, 2.84, 1.73 |

- The global timer steps by 256 ns. Stamps add about 2 us to a 19 us launch.
- In `a6900d3de` the o_proj and down tiles convert the residual right after its load, at a branch join, so the MMAs wait one L2 round trip. `1881c0af2` keeps the residual raw until the epilogue.
- `1881c0af2` also stops loading the norm weights and rope row up front. That load cost 0.73 us, queued behind each SM's 96 KB weight burst.
