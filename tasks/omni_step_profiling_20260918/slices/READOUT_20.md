# Run 20 readout: the vocoder decode, module by module (V-e10)

moss card 5, RTX 4090 D, main `ebd577ea0` + the fused snake branch `6574bc743`. Script
`vocoder_module_map.py`: the incremental decoder's functions and modules wrapped in
profiler ranges, one eager decode per key, each kernel attributed to the innermost range
that launched it. Kernel durations only (what a graph replay executes). Outputs:
`artifacts/moss_omni_step_profiling/omni_step_profiling/run20`.

## 1. Device time per module group, ms (kernels)

| group | w1 b1 eager | w1 b1 fused snake | w8 b8 eager | w8 b8 fused snake | w32 b4 fused | w64 b1 fused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conv (23 causal convs, with their history cat and clone) | 0.984 (131) | 0.971 | 6.161 (136) | 6.195 | 14.636 | 7.296 |
| transconv (6) | 0.403 (37) | 0.403 | 0.991 (39) | 0.993 | 1.737 | 1.005 |
| snake (29) | 0.402 (290) | 0.050 (29) | 1.800 (290) | 0.369 (29) | 0.865 | 0.341 |
| attention (8 layers, manual) | 0.436 (248) | 0.435 | 0.688 (272) | 0.688 | 0.696 | 0.611 |
| transformer norm (17, manual RMSNorm) | 0.190 (136) | 0.189 | 0.224 (136) | 0.221 | 0.226 | 0.223 |
| mlp | 0.100 (40) | 0.100 | 0.150 (48) | 0.148 | 0.167 | 0.150 |
| quantizer decode | 0.131 (69) | 0.130 | 0.164 (91) | 0.163 | 0.149 | 0.127 |
| residual adds | 0.014 (12) | 0.014 | 0.202 (12) | 0.201 | 0.620 | 0.174 |
| convnext rest, transformer rest, layer scale, rotary | 0.163 (79) | 0.162 | 0.242 (84) | 0.239 | 0.296 | 0.230 |
| total | 2.825 (1043) | 2.457 (782) | 10.622 (1109) | 9.220 (848) | 19.394 | 10.159 |

Inside the conv group at w8 b8: cuDNN conv kernels 4.60 ms, history cat and clone 0.45
(46 kernels), GEMM 0.41, elementwise 0.37, cuDNN layout transposes 0.32; the transconv
group adds 0.26 ms of transposes.

## 2. What the map says

1. After the fused snake, a small decode (w1 b1, the ramp chunks and everything at c1)
   spends 33 percent of its time and 487 of its 782 kernels in the 8-layer transformer:
   manual attention 0.435 ms and 248 kernels (KV cat, repeat_kv copies, matmul, scale,
   mask build, masked_fill, fp32 softmax, matmul), manual RMSNorm 0.189 ms and 136
   kernels, plus MLP, layer scale, rotary.
2. A large decode (w8 b8 and wider) is convolution: conv plus transconv are 78 percent
   at w8 b8 and 84 percent at w32 b4, and 4.6 of the 6.2 ms are the cuDNN kernels
   themselves. Around them sit about 1.5 ms of copies, elementwise and layout transposes.
3. The quantizer decode is 69 to 91 kernels for 16 codebook lookups and sums.

## 3. Startup (run 18 clean boots, timestamps in serve.log)

Of main's 120 to 130 s: process and preprocessing about 15 s, vocoder graph capture 89 s
(warm runner 47 s, second warm runner 18 s, cold 5 s, window 18 s), the talker engine
17 s. A warm runner's 32 shapes capture in 14.5 s without torch.compile (run 11), so
about 30 s of the first warm runner is the static compile of the 8-frame shape.
