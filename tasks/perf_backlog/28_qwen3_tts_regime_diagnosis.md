# 28. The half speed regime: a forced torch fused op backend, 2026-09-12

Archive `e5-regime-diagnosis-0323acc0c.tar.gz`, read against the E5 archives and the
11 Sept census traces.

## 1. What the diagnosis run measured

- Fresh shell: no MPS, DCGM or Nsight process; no CUDA, CUPTI, preload or Nsight
  variable in the environment (preflight.txt). Two external Python processes held GPUs 0
  and 1 (75 GB each, 82 percent) during the microbenchmark and had exited before the E4
  head boot.
- Graph replay microbenchmark, 1000 tiny dependent nodes: 1.745 ms per replay on GPU 0
  with the tenant co resident, 1.042 ms on GPU 1, eager launch 4.4 to 4.8 us. The device
  front end is not slowed.
- E4 head 04b62c255 from the root checkout, GPU 0 empty, only our server on it, P0,
  1980 MHz: 9.695 req/s, 1.163 s between decode log lines, 505 tokens per second. The
  slow regime again, so it is not the worktree layout and not the device.

## 2. The cause

The server command of every E5 session (`e4_root/server_command.txt`, and the GPU 2
protocol's "torch fused-op backend") is

```text
env PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 SGLANG_FORCE_FUSED_OP_BACKEND=torch \
  SGLANG_OMNI_QTTS_PREDICTOR_GRAPH=1 .venv/bin/python -m sglang_omni.cli serve ...
```

Every fast session (pair 1 at 13:19, pair 2, E4, the E3 seeded pair, the c16 followup,
the seeded c1 boots) ran the plain command with no variable
(`serve_command.json` in those archives). `SGLANG_FORCE_FUSED_OP_BACKEND` is the pinned
sglang's process wide debug switch that forces every fused op onto one backend
(sglang/kernels/fused_op.py:26 and 223); `torch` means RMSNorm, fused add RMSNorm, the
gated activation and the rest run as their pure torch compositions, several elementwise
kernels each, instead of one flashinfer or sgl_kernel kernel. `SGLANG_OMNI_QTTS_PREDICTOR_GRAPH=1`
is the default and changes nothing.

Kernel level proof. The 11 Sept census on main (default command) has 42k flashinfer
RMSNorm kernels, 13k fused add RMSNorm kernels and 26k `sglang::act_and_mul_kernel`
launches in its first 600 MB and 328 torch silu kernels (the vocoder). The E5 GPU 0
control window has no flashinfer RMSNorm and no act_and_mul kernel at all and 6248 torch
silu kernels. The talker's graphs therefore carry about three times the nodes, which is
the 13.6 ms predictor replay against 4.3 (doc 27 section 1) and the 2.3 times longer
decode step, at full clocks, with the tiny node microbenchmark unaffected.

The variable is in no runbook of this series; the one place the repository mentions it is
`tasks/pr_2057_review_20260909/review.md`, where it is named as the dependency's bisect
mode. It entered the E5 server command on the box.

## 3. What this does to the E5 conclusions

- Docs 26 and 27 measured control against early ids on the torch backend. Each pair is
  internally valid and both reproduce the same direction and stage attribution, but the
  talker's step there is a different object (three times the graph nodes, a 17 ms host
  tail, the predictor finishing before the next step's host work) and none of the E5
  numbers describes the production kernel path. The bounded run ahead result of doc 27
  is a no op only on that backend and is unmeasured on the default one.
- The E4 session, the pairs and every PR number were on the default backend and stand.
- The ordering of the mechanism findings (lock handoff, launch calls, readiness waits,
  the serial initial worker) rests on both the census traces (default backend) and E5
  (torch backend), so the mechanism is not backend specific, but its magnitudes on the
  default backend at 17 req/s are still those of E4 and pair 1.

## 4. Rule

Before any pair is quoted: the server command is the plain one, and the decode log line
gap is about 0.6 s per 40 steps at c16 with about 850 tokens per second. Any environment
variable on the server command is a protocol change and is recorded in the protocol and
in the readout. A session whose gap reads 1.2 s stops before its first pair.

## 5. Next

Repeat E5 with the plain server command, GPUs 1 to 3 recorded, in one session:

1. Step 1, control against early ids, event recorder in pass 2, `nvidia-smi dmon -i 0 -s
   pucv -d 1` logging on both boots. Expected regime: 0.6 s per 40 steps, about 17 req/s.
2. The clean Nsight pair with `--gpu-metrics-devices=0 --gpu-metrics-frequency=20000`
   added, the exact command saved, SQLite exported, read with
   `tasks/perf_backlog/scripts/nsys_threads.py`.
3. The bounded run ahead on early ids, same protocol as step 1.
4. #2126 stacked on early ids, same protocol.
