# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3

from scripts.debug.nsys_moss_codec_summary import summarize


def _write_fake_nsys_sqlite(path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO StringIds (id, value) VALUES (?, ?)",
            [
                (1, "moss_tts_local.vocoder.processor.decode_audio_codes"),
                (
                    2,
                    "moss_tts_local.vocoder.audio_tokenizer.decoder.0."
                    "ProjectedTransformer.layer.0.self_attn.forward",
                ),
                (
                    3,
                    "moss_tts_local.vocoder.audio_tokenizer.decoder.0."
                    "ProjectedTransformer.layer.0.self_attn._forward_streaming_sdpa",
                ),
                (4, "cudaLaunchKernel"),
                (5, "cudaMemcpyAsync"),
                (6, "cudnn_generated_fort_native_sdpa_sm90_flash_kernel"),
            ],
        )
        conn.execute(
            "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, textId INTEGER, "
            "globalTid INTEGER)"
        )
        conn.executemany(
            "INSERT INTO NVTX_EVENTS (start, end, textId, globalTid) "
            "VALUES (?, ?, ?, ?)",
            [
                (0, 1_000_000, 1, 11),
                (100_000, 600_000, 2, 11),
                (200_000, 400_000, 3, 11),
            ],
        )
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
            "(start INTEGER, end INTEGER, nameId INTEGER, globalTid INTEGER)"
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
            "(start, end, nameId, globalTid) VALUES (?, ?, ?, ?)",
            [
                (250_000, 300_000, 4, 11),
                (700_000, 800_000, 5, 11),
                (1_500_000, 1_600_000, 4, 11),
            ],
        )
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, shortName INTEGER, globalTid INTEGER)"
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "(start, end, shortName, globalTid) VALUES (?, ?, ?, ?)",
            [
                (260_000, 290_000, 6, 11),
                (1_200_000, 1_300_000, 6, 11),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_nsys_moss_codec_summary_overlaps_processor_scope(tmp_path) -> None:
    db_path = tmp_path / "trace.sqlite"
    _write_fake_nsys_sqlite(db_path)

    report = summarize(db_path)

    assert report["counts"]["processor_ranges"] == 1
    label_totals = {row["label"]: row for row in report["label_totals"]}
    processor = "moss_tts_local.vocoder.processor.decode_audio_codes"
    assert label_totals[processor]["count"] == 1
    assert label_totals[processor]["total_ms"] == 1.0

    runtime = {row["name"]: row for row in report["runtime_overlap"]}
    assert runtime["cudaLaunchKernel"]["overlap_ms"] == 0.05
    assert runtime["cudaMemcpyAsync"]["overlap_ms"] == 0.1

    kernels = {row["name"]: row for row in report["kernel_overlap"]}
    assert (
        kernels["cudnn_generated_fort_native_sdpa_sm90_flash_kernel"]["overlap_ms"]
        == 0.03
    )
    assert report["kernel_category_overlap"][0]["category"] == (
        "sdpa_or_flash_attention"
    )

    decoder_labels = {row["label"]: row for row in report["decoder_subscope_overlap"]}
    assert decoder_labels[".self_attn.forward"]["overlap_ms"] == 0.5
    assert decoder_labels[".self_attn._forward_streaming_sdpa"]["overlap_ms"] == 0.2

    hot_scope_kernels = {
        (row["scope"], row["name"]): row for row in report["hot_scope_kernel_overlap"]
    }
    hot_scope_key = (
        "moss_tts_local.vocoder.audio_tokenizer.decoder.0."
        "ProjectedTransformer.layer.0.self_attn._forward_streaming_sdpa",
        "cudnn_generated_fort_native_sdpa_sm90_flash_kernel",
    )
    assert hot_scope_kernels[hot_scope_key]["overlap_ms"] == 0.03
