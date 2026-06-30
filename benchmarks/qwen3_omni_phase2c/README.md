# Qwen3-Omni Phase 2C Torch Trace Analysis

Phase 2C uses torch-profiler traces only after request-level profiling has
identified the expensive stage. The analyzer ranks trace events locally so
Perfetto is used for visual inspection, not for manual counting.

## Analyze One Run

```bash
python -m benchmarks.qwen3_omni_phase2c.analyze_torch_traces \
  /tmp/qwen3_phase2c_profiles/phase2c-thinker_videomme-* \
  --label thinker_videomme \
  --output-dir results/qwen3_phase2c_thinker_trace_summary
```

```bash
python -m benchmarks.qwen3_omni_phase2c.analyze_torch_traces \
  /tmp/qwen3_phase2c_profiles/phase2c-talker_seedtts-* \
  --label talker_seedtts \
  --output-dir results/qwen3_phase2c_talker_trace_summary
```

The analyzer writes:

- `phase2c_trace_summary.json`
- `phase2c_trace_summary.md`

## How To Use The Output

Use `Top CPU Ops`, `Top CUDA Kernels`, and `Top CUDA Runtime Calls` to pick the
actual owner. Then open the same trace in Perfetto only for timeline questions:

- long blank regions between kernels;
- CPU thread waiting before GPU work starts;
- many tiny kernels that may need fusion or CUDA graph coverage;
- repeated runtime calls or synchronizations;
- trace regions that do not match request-profile stage timings.

Do not implement an optimization until the trace summary identifies the owner
file/function and the request-profile report proves the same stage is a
latency bottleneck.
