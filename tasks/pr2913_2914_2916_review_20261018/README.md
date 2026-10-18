# Qwen3-TTS PRs 2913, 2914 and 2916 review

Reviewed on 18 October 2026. The three existing author worktrees were clean and already matched the fetched PR heads. Descriptions, issue comments and inline reviews were retrieved directly from GitHub. There were no earlier review replies on these heads to resolve. No GitHub review was staged or submitted, and no author source branch was changed.

| PR | Reviewed head | Conclusion |
| --- | --- | --- |
| [2913](https://github.com/sgl-project/sglang-omni/pull/2913) | `4fa680cb1f273673635d7bd2042597af8551b757` | The embedding invariant and removal are sound. Only two stale buffer descriptions need a small optional cleanup. |
| [2914](https://github.com/sgl-project/sglang-omni/pull/2914) | `9f1840ee4322921b3f10016634ce32a449b7b7e6` | The text-RoPE design is sound. One reproduced CUDA native-backend compatibility regression needs a small fix. |
| [2916](https://github.com/sgl-project/sglang-omni/pull/2916) | `0e608409ce7284ba0ca01838ee371faeb6991533` | The stacking, checkpoint mapping and reduction order are sound. No actionable defect found. |

The substantive finding is limited to #2914's diagnostic native-backend path. Default CUDA RoPE and fused KV-store execution passed the independent numerical/replay probe. This does not justify rejecting the optimization itself.

There is also an optional wording correction in #2914: a reported speaker-similarity change of -0.32 is not inside a stated 0.25-point main/main difference. This review does not infer a quality regression from that sentence or claim to have established statistical equivalence.

Independent H200 validation: **173 unit cases passed**, one unrelated Ascend case skipped, real Base serving passed with full and breakable prefill graphs, and #2916's checkpoint-width ICL outputs matched the pre-PR builder bit-for-bit in three alignment cases. These targeted checks do not claim a fresh full-corpus quality or throughput benchmark.

- [2913 review](pr2913.md)
- [2914 review and exact failure mechanism](pr2914.md)
- [2916 review](pr2916.md)
- [High-level and low-level mechanics](mechanics.md)
- [Verification, test admission and limits](verification.md)

The raw PR metadata and diffs are retained beside these documents. Source references are pinned to the reviewed heads. SGLang source hashes and the local SGLang revision are in `sglang_sources.json`.
