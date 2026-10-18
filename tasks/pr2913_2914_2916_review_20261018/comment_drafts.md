# Local comment drafts

These are local drafts only. Nothing has been staged or posted to GitHub.

## 2913: optional docstring cleanup

Update these two docstrings too. The feedback buffer is removed; the prompt builder now reads device and dtype from the codec embedding weight.

## 2914: code finding

Gate the fused KV store on the RoPE backend too. After this change, forcing `SGLANG_FORCE_FUSED_OP_BACKEND=torch` reaches text RoPE's native implementation with a fused-store argument and raises. The previous MRoPE path works. Use the backend check already used by the predictor, and leave the separate KV store active on the native path. Add a CUDA case for that fallback beside the normal fused path.

## 2914: optional reporting correction

The -0.32 similarity change is outside the stated 0.25-point main/main difference. Remove the within-noise claim or support it with repeated main/main runs.

## 2916: optional reporting correction

Report full-corpus WER first, including the two main-arm outliers. Keep filtered WER and the outlier counts alongside it.
