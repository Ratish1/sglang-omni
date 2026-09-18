# The conv position embed on the packed sequence: measured, parked (2026-09-18)

RTX 4090 D, torch 2.13.0+cu130, cuDNN 9.20, bfloat16 autocast, inference mode, card 7 alone. Raw
outputs on the Mac under `artifacts/cosyvoice-4090-20260918/conv/` (`conv-20260918`,
`convlayouts-20260918`, `convlayouts2` to `4`, `convsweep-20260918`). Scripts:
`stage2/packed_conv_probe.py`, `stage2/conv_layouts_bench.py`, `stage2/conv_width_sweep.py`.
Branch `slice/cosyvoice-packed-conv` at e7b37b74e holds the first attempt (the long layout) and
is **not** to be merged.

## What was tried

The DiT's causal conv position embed (two Conv1d, 1,024 channels, 16 groups, kernel 31, each left
padded by 30, Mish) is the one module of the packed DiT that still pads every row to the widest.

| layout | what it is |
|---|---|
| padded | rows scattered to (rows, widest); main |
| long | one sequence, 30 zero frames before each row, gaps zeroed again between the two convs |
| tiled | that sequence cut into as many equal tiles as rows, each with the 30 frames before it |
| gathered | tiled, but every move is one gather from an index table built once per Flow call |

All four are the same arithmetic: float64 agrees to 1e-12, and in bfloat16 every layout is 51.3 dB
from the float32 conv on the same activations, padded included. The 31 to 36 dB seen on the final
mel between layouts is that rounding amplified by ten bfloat16 Euler steps, not a lossier conv;
the runaway shape, where cuDNN picks the same algorithm for both, is bit identical.

## The conv alone, ms, mean of two runs

| shape (CFG doubled) | rows x widest | total | padded | long | tiled | gathered | gathered vs padded |
|---|---|---|---|---|---|---|---|
| 1 row of 550 | 1,100 | 1,100 | 0.72 | 0.70 | 0.74 | 0.71 | -2 % |
| 2 rows, equal | 2,200 | 2,200 | 0.71 | 1.10 | 0.79 | 0.69 | -3 % |
| 4 rows, 400 to 950 | 7,600 | 5,300 | 1.49 | 2.41 | 1.53 | 1.30 | -13 % |
| 8 rows, 400 to 1,100 | 17,600 | 12,000 | 3.13 | 5.18 | 2.96 | 2.47 | -21 % |
| 16 rows, 400 to 1,150 | 36,800 | 24,800 | 6.99 | 10.86 | 5.52 | 4.17 | -40 % |
| 16 rows, equal 550 | 17,600 | 17,600 | 3.15 | 7.80 | 4.75 | 3.84 | **+22 %** |
| 16 rows, 500 to 650 | 20,800 | 18,400 | 2.74 | 8.09 | 4.83 | 3.95 | **+44 %** |
| 8 rows, 500 to 640 | 10,240 | 9,120 | 1.86 | 4.07 | 2.40 | 1.99 | **+7 %** |
| 4 rows, 500 to 560 | 4,480 | 4,240 | 0.95 | 1.96 | 1.26 | 1.07 | **+13 %** |
| 16 rows, one runaway of 4,300 | 137,600 | 31,100 | 52.85 | 13.57 | 8.09 | 6.51 | -88 % |
| 8 rows, one runaway of 4,300 | 68,800 | 18,400 | 26.43 | 7.96 | 4.83 | 3.92 | -85 % |

Whole Flow call with the long layout (`packed_conv_probe.py`): -1 to -2 % at 1 to 2 rows, +2 to
+3 % at 4 to 16 ragged rows, -13 % with a runaway, and the runaway call's peak memory 1,269 to
877 MiB. Also measured and dropped: fixed tile widths of 250 to 2,000 (all the same cost as equal
tiles), and `torch.backends.cudnn.flags(benchmark=True)` (every layout 2 to 4 times slower).

## Why no layout wins everywhere

cuDNN chooses its algorithm by heuristic on (batch, width), and the cost per frame moves in steps
with no alignment pattern (width mod 8, 16, 64 all flat). First conv alone, ns per frame:

| batch | regimes by width |
|---|---|
| 32 | 320 to 664: 67; 668 to 1,020: 44 to 47; 1,024 to 1,288: 65 to 72; from 1,292: **172** |
| 8 | sawtooth 66 to 122, steps at 480, 704, 928 |
| 2 | 498 falling to 107 as width grows: a fixed cost of about 0.3 ms dominates |

The padded layout's near equal batches happen to land in the 44 ns regime and the tiles beside it
in the 67 ns one; a runaway pushes padded over the 172 ns cliff, which with the 4.4 times frame
count is its 53 ms. These regimes belong to this cuDNN on this card. A rule that picks the layout
by them would be a hardware constant, and the constant free rule (fewer frames through the conv)
still loses 0.2 to 1.2 ms per step on near equal batches.

SGLang has no kernel for this: every causal conv in v0.5.19 is depthwise, widths 2 to 4 except one
JIT kernel that is still depthwise (`kernels/aot/csrc/mamba/causal_conv1d.cu:132, :233`,
`jit/csrc/inkling/causal_conv1d.cuh:181`). Its audio encoders batch ragged conv input as fixed
windows with a padded tail and a mask (`models/qwen3_omni_moe.py:281-335`,
`dots3_common/dots_omni_audio.py:545-572`, `voxtral.py:307-334`), which is the tiled layout.

## Decision

Parked. The conv is 3 to 4 percent of a Flow call; the change is worth about 1.5 percent of a 16
row ragged call and costs up to 0.7 percent on near equal ones, so by the no regression rule it
does not go in. Its one large effect, runaway batches, is addressed at the source by the hop
prefix cache: a cached hop runs the conv over each row's new frames plus a 60 frame tail, so rows
are short and near equal, which is the padded layout's best case. If finals, which stay whole
sequence, still show the runaway cost after the cache lands, the gathered layout is the one to
take up, for finals only, and it needs the per call object that plan 12 also wants for the
step invariant work.
