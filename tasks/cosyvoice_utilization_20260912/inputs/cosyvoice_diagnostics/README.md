# CosyVoice utilization diagnostics

Read `PLAYBOOK.md` for the complete research-backed procedure, model-specific pitfalls, commands, decision tables, and source references. This kit assumes an already working SGLang-Omni CosyVoice deployment on an approved NVIDIA GPU test instance.

## Files

| File | Purpose |
|---|---|
| `PLAYBOOK.md` | Detailed profiling and optimization guide with primary-source references |
| `collect_env.py` | Standard-library, read-only environment/source-default snapshot; does not load the model |
| `run_sweep.sh` | Native SeedTTS screening runs against an existing server; never starts or stops it |
| `analyze_nsys_sqlite.py` | Explicit-window union of captured CUDA-kernel intervals; reports overlapping work correctly |
| `test_analyze_nsys_sqlite.py` | Ten synthetic tests for union math, clipping, schema handling, and read-only database access |
| `experiment_template.json` | Record of workload, effective configuration, evidence, and acceptance criteria |

## Setup

Extract this directory anywhere accessible to the server environment. No extra Python packages are required for the collector, analyzer, or tests. The sweep uses the dependencies and CLI of the SGLang-Omni checkout; it is not a standalone HTTP load generator. Nsight Systems and DCGM are separate NVIDIA tools and must already be installed/configured to use their commands.

Run the collector inside the serving container with the server's Python executable:

```bash
python /path/to/cosyvoice_diagnostics/collect_env.py --output results/environment.json
```

Attach the actual resolved server configuration separately. AST-extracted factory signatures are source defaults, not proof of live settings. Review any diagnostic output before sharing: GPU names, paths, process information, and source identities can be operationally sensitive. The collector does not dump arbitrary environment variables, authentication tokens, or Git diffs.

After starting an explicitly configured candidate server, run the native sweep from the SGLang-Omni repository root:

```bash
CONCURRENCIES="1 2 4 8 16" REPEATS=3 MAX_SAMPLES=256 \
  bash /path/to/cosyvoice_diagnostics/run_sweep.sh compiled streaming
```

Use `buffered` instead of `streaming` for a buffered-client comparison. Available environment settings are `PYTHON`, `MODEL`, `PORT`, `CONCURRENCIES`, `REPEATS`, `MAX_SAMPLES`, `WARMUP`, and `OUT`. Choose a new label/output location for every independent run. The script refuses to reuse result directories. Its default 256 samples are for screening, not tail-latency qualification. Do not send saturation traffic to an unapproved production endpoint.

Export a captured Nsight report to SQLite, then inspect its devices and time bounds:

```bash
nsys export --type sqlite -o traces/c16.sqlite traces/c16.nsys-rep
python /path/to/cosyvoice_diagnostics/analyze_nsys_sqlite.py \
  traces/c16.sqlite --describe
```

The parser requires an explicit selected device and analysis window. Specify `--start-ns` and `--end-ns` using the SQLite export clock, with the steady-state boundaries identified in the trace. Example only:

```bash
python /path/to/cosyvoice_diagnostics/analyze_nsys_sqlite.py \
  traces/c16.sqlite --device 0 \
  --start-ns 10000000000 --end-ns 30000000000 > traces/c16.coverage.json
```

The returned percentage is **captured CUDA-kernel timeline coverage**, not SM activity, occupancy, GPU FLOP efficiency, or total physical-device utilization. A gap can contain copies, CPU work, dependencies, or work from untraced processes. Missing CUDA graph/child-process events can invalidate coverage. Never infer a root cause from this one number.

## Validation and limitations

```bash
cd /path/to/cosyvoice_diagnostics
python -m unittest -v
python -m py_compile *.py
bash -n run_sweep.sh
```

Ten unit tests passed on synthetic data. The analyzer's CLI and environment collector were smoke-tested without GPU software. Shell syntax was checked. GPU serving, native benchmark execution, profiling, model correctness, and performance gains were **not** tested on the target deployment. Nsight export schemas and SGLang-Omni APIs can change; unsupported schemas produce an explicit error rather than fabricated metrics.

No universal stage replay is included: it must preserve the target commit's real conditioning, masks, CFG pairing, streaming state, and shape contracts. Section 10 of the playbook defines the replay experiments and measurement boundaries needed to implement that adapter correctly.
