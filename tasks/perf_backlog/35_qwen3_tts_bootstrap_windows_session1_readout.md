# 35. Bootstrap windows, session 1 readout, 2026-09-13: invalid, all four arms ran main

Archive `bw-session-results-no-wavs.tar.gz`. Heads recorded per arm: A 3060470a8, B
f73586369, both from worktrees under `tmp/`. GPU 1 for every boot (GPU 0 held 80 GB of
another tenant, recorded in every `gpus_before.txt`). Step 1 passed on the box: 36, 22 and
242 tests.

## 1. What actually ran

Every server imported `sglang_omni` from the main checkout, not from its worktree, so the
four measured boots and the two Nsight boots are the same code, and the early ids patch,
applied in the worktree, never ran either.

Evidence, all from the serve logs:

- The branch changes the graph shapes line to carry `window_frames=` and captures a third
  runner with widths 1 to 64. No serve log has `window_frames`, every log has exactly the
  cold and warm capture lists, and no codec state line has a `window` key.
- The Nsight boots print Python source paths: `/sgl-workspace/sglang-omni/sglang_omni/...`,
  the main checkout. No log mentions `tmp/bw` or `tmp/main`.
- The "early ids" arms show main's bootstrap segment, 29.8 and 28.0 ms at ahead 0, not
  the 52 ms doc 32 measured for early ids, and the same chunks before audio (p50 2).

Cause: the session steps I wrote gave the server command as `sgl-omni serve`, the console
script. It resolves the package through the venv's editable install, which points at the
main checkout, whatever directory it is started from. The unit tests passed on the branch
because `python -m pytest` puts the working directory first on `sys.path`. Earlier
sessions used `python -m sglang_omni.cli serve` from the worktree, which does the same
for the server, and archived the import path. This session did neither.

## 2. What the four boots are worth: a repeatability sample of main on GPU 1

Same code, same corpus, unseeded, warmup 1, streaming c16, pass 2 (event recorder on).

| boot | req/s | TTFC mean ms | TTFC p99 ms | inter chunk ms | ahead 0 p50 ms | bootstrap mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s2a | 15.48 | 123 | 306 | 109 | 29.8 | 38.9 |
| s2b | 15.54 | 123 | 340 | 109 | 28.0 | 37.1 |
| s3a | 14.89 | 132 | 375 | 116 | 29.7 | 36.5 |
| s3b | 15.17 | 129 | 338 | 114 | 28.4 | 35.1 |

Pass 1 req/s: 15.80, 15.85, 14.70, 15.58. So the noise band on this GPU and day is about
4 percent on req/s and about 2 ms on the ahead 0 bootstrap segment. s2a and s2b pass 2
both contain a 163.84 s runaway output and are not quotable by the session rule; s3a and
s3b are clean. Quality on s3b pass 2: WER 1.00 percent, similarity 71.26, both inside the
bands, and both measured on main.

Seeded streaming c1, two boots of the same code: 38 of 1088 WAVs byte identical. The
streaming path is not reproducible across boots even on identical code, so the identity
count cannot serve as a gate for this slice; it stays reported. The s3b pass 3 was retried
on a second server on port 58107 while the first still held 31001; both were main.

The two Nsight windows (s2a-nsys, s2b-nsys) are main under the profiler, 20 s each, with
`threads.json` and the SQLite exports kept. They are a main only dataset, unused here.

## 3. What the next session must do differently

- Server command from the worktree: `python -m sglang_omni.cli serve ...`, never the
  console script.
- Before every boot, in the worktree, archive `python -c "import sglang_omni; print(sglang_omni.__file__)"`
  as `import_path.txt`; it must be under that worktree.
- After the B boot reaches ready, the serve log must contain `window_frames=(1, 2, 4, 8,
  16, 32, 64)` on the graph shapes line and a third capture list of widths 1 to 64. If
  not, the boot is void, stop.
- Runbook 33 carries these as gates.
