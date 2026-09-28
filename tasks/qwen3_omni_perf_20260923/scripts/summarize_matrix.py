"""Per-arm values of a pulled benchmark round: python summarize_matrix.py <round dir> [<round dir> ...]"""

import glob
import json
import os
import sys


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


for root in sys.argv[1:]:
    print(f"== {root}")
    for arm in sorted(
        d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
    ):
        name = os.path.basename(arm)
        s = load(os.path.join(arm, "seedtts_en", "speed_results.json"))
        if s:
            m = s["summary"]
            w = load(os.path.join(arm, "seedtts_en", "score_summary.json"))
            sim = load(os.path.join(arm, "seedtts_en", "sim_summary.json"))
            print(
                f"{name:10s} seedtts req/s {m['throughput_qps']:.3f} lat {m['latency_mean_s']:.3f}/{m['latency_p99_s']:.3f} first_audio {m['audio_ttfp_mean_s']:.3f} first_token {m['text_ttft_mean_s']:.4f} audio_s/s {m['audio_throughput_s_per_s']:.2f} rtf {m['rtf_mean']:.4f} inter_chunk {m['inter_chunk_mean_s']:.4f}/{m['inter_chunk_p99_s']:.4f} fail {m['failed_requests']}"
                + (
                    f" wer {100*w['wer']['wer_corpus']:.2f}% (>50%: {w['wer']['n_above_50_pct_wer']})"
                    if w
                    else ""
                )
                + (
                    f" sim {sim['similarity']['speaker_similarity_mean']:.3f}"
                    if sim
                    else ""
                )
            )
        u = load(os.path.join(arm, "mmsu_talker", "mmsu_results.json"))
        if u:
            m = u["speed_metrics"]
            w = load(os.path.join(arm, "mmsu_talker", "score_summary.json"))
            print(
                f"{name:10s} mmsu    req/s {m['throughput_qps']:.3f} lat {m['latency_mean_s']:.3f}/{m['latency_p99_s']:.3f} audio_s/s {m['audio_throughput_s_per_s']:.2f} rtf {m['rtf_mean']:.4f} tokens {m['output_tokens_total']} acc {u['summary']['overall_accuracy']:.4f} fail {m['failed_requests']}"
                + (f" wer {100*w['wer']['wer_corpus']:.2f}%" if w else "")
            )
        v = load(os.path.join(arm, "mmmu_talker", "mmmu_results.json"))
        if v:
            m = v["speed"]
            w = load(os.path.join(arm, "mmmu_talker", "score_summary.json"))
            print(
                f"{name:10s} mmmu    req/s {m['throughput_qps']:.3f} lat {m['latency_mean_s']:.3f}/{m['latency_p99_s']:.3f} audio_s/s {m['audio_throughput_s_per_s']:.2f} rtf {m['rtf_mean']:.4f} tokens {m['output_tokens_total']} audio_mean {m['audio_duration_mean_s']:.2f} acc {v['summary']['accuracy']:.4f} ({v['summary']['correct']}) fail {m['failed_requests']}"
                + (f" wer {100*w['wer']['wer_corpus']:.2f}%" if w else "")
            )
