# MOSS-TTS Local Codec Batch-Sensitivity Debug Plan

## Scope

This debug branch investigates why the upstream MOSS-TTS Local processor can
produce different reference audio codes when the same waveform is encoded alone
versus inside a batch.

This is intentionally separate from the codec-ownership PR. The PR invariant is:

```text
candidate-single == upstream-single
candidate-batch == upstream-batch
```

The remaining question is upstream behavior:

```text
upstream-single != upstream-batch
```

Do not fix this by reducing serving batch size. That changes runtime policy and
does not explain the codec behavior.

## Source Of Truth

For `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`, the HF processor implements:

- `encode_audios_from_path(paths, n_vq=...)`
  - loads paths with `torchaudio.load`;
  - resamples path audio to `model_config.sampling_rate`;
  - delegates to `encode_audios_from_wav`.
- `encode_audios_from_wav(wavs, sampling_rate, n_vq=...)`
  - enforces `n_vq == model_config.n_vq`;
  - folds mono/multichannel audio to stereo;
  - loudness-normalizes each waveform;
  - calls `audio_tokenizer.batch_encode(prepared, num_quantizers=n_vq)`.

Therefore the debug path must call these processor methods directly.

## Hypotheses To Test

1. Codec input prep differs between single and batch.
   - Prepared waveforms should match before `batch_encode`.
   - If they do not, the bug is in path/wav loading, channel handling, resample,
     loudness normalization, or dtype/device conversion.

2. `batch_encode` is shape-sensitive.
   - The same prepared waveform can produce different codes when padded with
     other waveforms in the batch.
   - Likely surfaces: encoder padding masks, convolution boundary handling,
     sequence lengths, quantizer tie-breaking, or batch-shape-dependent CUDA
     kernels.

3. Same-mode candidate parity is already correct.
   - If candidate-single or candidate-batch differs from the matching upstream
     mode, fix the codec-ownership PR.
   - If only upstream-single vs upstream-batch differs, keep the PR unchanged
     and continue this branch as an upstream codec investigation.

## Debug Procedure

Run on H100 from repo root.

```bash
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

python tmp/moss_local_codec_debug/compare_reference_codes.py \
  --sample-ids \
    common_voice_en_628642-common_voice_en_628644 \
    common_voice_en_25748530-common_voice_en_25748532 \
    common_voice_en_20005340-common_voice_en_20005341 \
    common_voice_en_19599129-common_voice_en_19599130 \
  --out /data/moss_local_codec_batch_sensitivity_probe
```

If the host hits the known cuDNN SDPA execution-plan issue while loading/running
the codec, rerun with:

```bash
python - <<'PY'
import runpy
import sys
import torch

torch.backends.cuda.enable_cudnn_sdp(False)
sys.argv = [
    "compare_reference_codes.py",
    "--sample-ids",
    "common_voice_en_628642-common_voice_en_628644",
    "common_voice_en_25748530-common_voice_en_25748532",
    "common_voice_en_20005340-common_voice_en_20005341",
    "common_voice_en_19599129-common_voice_en_19599130",
    "--out",
    "/data/moss_local_codec_batch_sensitivity_probe",
]
runpy.run_path("tmp/moss_local_codec_debug/compare_reference_codes.py", run_name="__main__")
PY
```

## Decision Rules

- `upstream-single == candidate-single` and
  `upstream-batch-all == candidate-batch-all`:
  the codec-ownership PR is mechanically correct.

- `upstream-single != upstream-batch-all`:
  upstream codec batch behavior is shape-sensitive. Do not change PR serving
  policy unless maintainers explicitly choose the throughput/conditioning tradeoff.

- Candidate mismatch in same mode:
  fix the PR before any further quality/performance work.

## Next Instrumentation If Needed

If code comparisons confirm upstream batch sensitivity and we need root cause:

1. Hook the HF processor around waveform preparation and verify each prepared
   waveform is identical in single and batch modes.
2. Hook `audio_tokenizer.batch_encode` inputs and outputs.
3. Add module-level hooks inside the codec encoder and locate the first module
   where a row from the batch diverges from the single-run tensor.
4. Only after first mismatch localization, decide whether this is:
   - valid floating-point/kernel variance;
   - missing/incorrect padding-mask handling;
   - quantizer tie-breaking sensitivity;
   - a real upstream codec bug.

The first invalid transition, not the final generated-audio symptom, determines
the fix boundary.

Run the first-mismatch trace for one mismatching sample:

```bash
python tmp/moss_local_codec_debug/trace_batch_encode_first_mismatch.py \
  --sample-ids \
    common_voice_en_628642-common_voice_en_628644 \
    common_voice_en_25748530-common_voice_en_25748532 \
    common_voice_en_20005340-common_voice_en_20005341 \
    common_voice_en_19599129-common_voice_en_19599130 \
  --trace-sample-id common_voice_en_25748530-common_voice_en_25748532 \
  --out /data/moss_local_codec_batch_sensitivity_trace
```

If cuDNN SDPA fails on the host, use the same in-process
`torch.backends.cuda.enable_cudnn_sdp(False)` wrapper as above.

Then run the fp32 precision-control variant:

```bash
python tmp/moss_local_codec_debug/trace_batch_encode_first_mismatch.py \
  --sample-ids \
    common_voice_en_628642-common_voice_en_628644 \
    common_voice_en_25748530-common_voice_en_25748532 \
    common_voice_en_20005340-common_voice_en_20005341 \
    common_voice_en_19599129-common_voice_en_19599130 \
  --trace-sample-id common_voice_en_25748530-common_voice_en_25748532 \
  --codec-weight-dtype fp32 \
  --compute-dtype fp32 \
  --disable-tf32 \
  --out /data/moss_local_codec_batch_sensitivity_trace_fp32
```

Decision:

- If the default trace diverges at a linear projection but the fp32 trace
  collapses or greatly shrinks, the mechanism is batch-shape-dependent BF16 GEMM
  variance amplified by hard quantization.
- If the fp32 trace still diverges at comparable scale, continue into padding
  mask, transformer, or quantizer semantics.
