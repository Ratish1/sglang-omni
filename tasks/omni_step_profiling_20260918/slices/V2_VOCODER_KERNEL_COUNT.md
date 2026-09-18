# Slice V2 (research): vocoder launches that are not convs

The run01 prefill b1 trace shows 1,120 to 1,146 kernels per vocoder graph replay. Conv
calls are 37 per decode. The rest are candidates, each with its source in the code:

| source | where | kernels per decode (from the code) |
| --- | --- | --- |
| SnakeBeta, eager qwen-tts arithmetic | 29 activations (4 blocks x (1 + 3 x 2) + 1 final), eager chain of 8 ops each per `vocoder_kernels.py:10` | up to 232 |
| arena gather and scatter | `codec_state_arena.py:187-238`, one index op per buffer | 51 + 51 |
| conv history clones | `incremental_codec.py:127-131`, one per conv | 29 |
| quantizer decode | qwen-tts `ResidualVectorQuantization.decode`, 16 lookups and adds | about 32 |
| transformer | 8 layers of manual attention (`incremental_codec.py:205`) and MLP | unmeasured |

Only the steady width 8 is compiled (audit A2), which fuses the elementwise work there;
every other width replays the eager kernels. The fused SnakeBeta exists but is off and
limited to one checkpoint's channels and B <= 8 (audit A1).

Questions, answered from V1-e1's per-key kernel lists before any design:

1. The kernel count per key and its split by source (table above).
2. The device time of the non-conv kernels per key.
3. Compile coverage: every captured width compiled, or one compile with a dynamic width,
   against today's split: startup time, graph memory, and replay time per key.
4. Cohort fragmentation (audit A3): vocoder replays per decode step and rows per
   replay, from the L1 ledger captures at c4, c16, c32.

No design until these are measured.
