# Readout 05: the default launch on a 24 GB card, main, the tokenizer fix, and the weight cast (2026-09-18)

Moss box, RTX 4090 D 24 GB, card 7 alone, one process at a time, host load average 45 to 49 (other
users on cards 0 to 5). Every run is the default launch (`python -m sglang_omni.cli serve
--model-path ...`, no serve arguments) and the benchmark's default request (no seed, no token
limit, warmup 1), streaming, the whole English split of 1,088. Runner `stage2/g1_stream_identity.sh`
at 785a18ad5 with `SEED=` and `SAMPLES=` empty. Run directories `.tmp/out/g1-{main,tokfix,stack}-20260918T08*` and `T09*`.

Arms: main `144bd6399`; tokfix `dc39da67f` (branch `fix/cosyvoice-speech-tokenizer-late-allocation`);
stack `3d3c7e38a` (tokfix plus the plan 14 weight cast, branch `stack/cosyvoice-tokfix-precast`).

## Results

| run | arm | c | completed / failed | req/s | audio s/s | RTF mean / p99 | first audio mean / p95 s | inter chunk mean s | C50 / C100 / C200 | CUBLAS lines | pool tokens | free after pool |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | main | 8 | 10 / 1,078 | | | | | | | 2,126 | 854,232 | |
| 2 | tokfix | 8 | 1,088 / 0 | 2.717 | 13.26 | 0.647 / 1.608 | 1.491 / 2.555 | 0.724 | 70.7 / 74.3 / 82.5 | 0 | 766,509 | 1.97 GB |
| 3 | stack | 8 | 1,088 / 0 | 2.686 | 13.36 | 0.661 / 1.638 | 1.500 / 2.432 | 0.720 | 70.7 / 75.7 / 85.3 | 0 | 902,923 | |
| 4 | main | 16 | 4 / 1,084 | | | | | | | 2,308 | 854,232 | |
| 5 | tokfix | 16 | 1,072 / 16 | 2.895 | 14.05 | 1.204 / 3.037 | 2.568 / 5.016 | 1.412 | 19.1 / 22.3 / 37.5 | 0 | 766,509 | 1.97 GB, 1.38 after decode graphs |
| 6 | stack | 16 | 1,072 / 16 | 2.978 | 14.75 | 1.145 / 2.477 | 2.477 / 3.777 | 1.328 | 20.9 / 23.9 / 39.1 | 0 | 904,288 | |
| 7 | main | 16 | 1 / 1,087, the stage process died | | | | | | | 3 | 852,866 | |
| 8 | tokfix | 16 | 1,035 / 53 | 2.778 | 12.90 | 1.201 / 2.620 | 2.476 / 4.738 | 1.436 | 20.8 / 23.8 / 38.7 | 0 | 766,509 | |
| 9 | stack | 16 | 1,073 / 15 | 2.947 | 14.52 | 1.187 / 2.842 | 2.540 / 4.676 | 1.401 | 18.4 / 22.7 / 32.7 | 0 | 902,752 | |

Main's timing columns are left empty: they describe the handful of requests that survived.

## What it says

1. **Main does not serve this corpus on a 24 GB card with the default launch.** 10 of 1,088 at c8,
   4 and 1 of 1,088 at c16, one boot lost its stage process. Every failure is
   `cublasCreate` inside the ONNX speech tokenizer (`CUBLAS failure 3`), then a torch OOM with 10
   to 30 MiB free. Yesterday's 32 sample runs hid the size of it (9 of 32).
2. **The tokenizer fix removes that failure completely**: zero CUBLAS lines in six runs, and c8
   serves 1,088 of 1,088. Its warm run is accounted by SGLang as designed: the pool shrank from
   854,232 to 766,509 tokens, 1.0 GB.
3. **c16 still fails 15 to 53 requests, and it is a second late consumer, in the Flow.** The
   traceback ends in `packed_dit.py:259 _conv_pos_embed` -> `CausalConvPositionEmbedding.conv1`
   -> `mish`, "tried to allocate 254 MiB" with 1.38 GB of slack left after the decode graphs.
   254 MiB is one bfloat16 tensor of `rows x widest x 1024`: the conv position embed is the one
   module of the packed DiT that still pads every row to the widest (plan 12 amendment), so one
   runaway row of about 4,000 frames pads 32 CFG rows to its width, and the module holds several
   such tensors at once. This is a layout defect, not a pool size defect.
4. **The weight cast frees memory but the pool takes it**: free memory at load 14.93 GB against
   13.41, pool 902,923 tokens against 766,509. The slack the late consumers live in does not grow.
5. **Stack against tokfix, one boot per cell, unpaired, unseeded**: c16 req/s +2.9 and +6.1
   percent, RTF mean -4.9 and -1.1 percent; c8 req/s -1.1 percent, RTF mean +2.1 percent. The c8
   pair is one boot each and inside the spread seen between boots of one tree, so it is neither a
   gain nor cleared of a regression; it needs a second pair before the cast ships.
6. Continuity at c8 is 70 percent on both arms and about 20 percent at c16: this card is over
   capacity for the model at these concurrencies on either tree.

## Next

The conv position embed on the packed sequence (30 zero frames before each row, zeroed again
between the two convs): removes the c16 OOM at its source, removes padded conv work that a runaway
multiplies, and is the prerequisite of plan 12's one number graph key. Gate: SNR of the Flow's mel
against the padded conv on the same inputs in one process, then this readout's runs again.
