# SPDX-License-Identifier: Apache-2.0
"""Check whether matched WER changes survive an independent codec renderer."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    apply_wer,
    load_qwen3_asr,
    transcribe,
)
from sglang_omni.client.audio import encode_wav
from sglang_omni.utils import find_available_port

SERVER_START_TIMEOUT_SECONDS = 600


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--asr-path", required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir()
    rows = [
        json.loads(line)
        for line in (arguments.population / "scored.jsonl").read_text().splitlines()
    ]
    selected: dict[str, set[str]] = {}
    for scope in ("first", "decode"):
        candidates = []
        for row in rows:
            pair = row["pairs"][scope]
            baseline = row["arms"]["none"]
            candidate = row["arms"][scope]
            if (
                pair["opening_state_equal"]
                and pair["future_text_equal"]
                and "spoken_wer" in baseline
                and "spoken_wer" in candidate
                and baseline["returned_text"] == candidate["returned_text"]
            ):
                change = candidate["word_errors"] - baseline["word_errors"]
                candidates.append((change, row["directory"]))
            else:
                pass
        if candidates:
            for delta, case_directory in (min(candidates), max(candidates)):
                if delta:
                    selected.setdefault(case_directory, {"none"}).add(scope)
                else:
                    pass
        else:
            pass
    if not selected:
        (arguments.output / "evidence.json").write_text("[]\n")
        return
    else:
        pass

    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path)
    reference, loading = Qwen3OmniMoeCode2Wav.from_pretrained(
        arguments.model_path,
        config=config.code2wav_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        key_mapping={r"^code2wav\.": ""},
        attn_implementation="sdpa",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
    reference.eval()
    codec_eos = config.talker_config.codec_eos_token_id
    reports = []
    for row in rows:
        directory = row["directory"]
        if directory not in selected:
            continue
        else:
            pass
        for scope in sorted(selected[directory]):
            native_directory = arguments.population / directory / scope
            frames = [
                torch.load(path, weights_only=True)["codec_ids"].reshape(-1)
                for path in sorted(native_directory.glob("codec-*.pt"))
            ]
            finish = torch.load(
                native_directory / "backbone-finish-000.pt", weights_only=True
            )
            assert len(frames) == finish["codec_frames_seen"]
            if not frames or int(frames[-1][0]) != codec_eos:
                continue
            else:
                pass
            assert all(int(frame[0]) != codec_eos for frame in frames[:-1])
            codes = torch.stack(frames[:-1]).T.unsqueeze(0).to(reference.device)
            waveform = reference.chunked_decode(codes).float().reshape(-1).cpu()
            assert waveform.isfinite().all()
            native_audio = native_directory / "audio.wav"
            with wave.open(str(native_audio)) as native_wave:
                sample_rate = native_wave.getframerate()
            rendered_audio = arguments.output / f"{directory}-{scope}.wav"
            rendered_audio.write_bytes(encode_wav(waveform.numpy(), sample_rate))
            reports.append(
                {
                    "sample_id": row["sample_id"],
                    "arm": scope,
                    "returned_text": row["arms"][scope]["returned_text"],
                    "native_previously_scored_wer": row["arms"][scope]["spoken_wer"],
                    "codec_frames": len(frames) - 1,
                    "native_audio": str(native_audio),
                    "checkpoint_audio": str(rendered_audio),
                }
            )
    del reference
    torch.cuda.empty_cache()

    asr_port = find_available_port()
    asr_server = start_server_from_cmd(
        [
            sys.executable,
            "-P",
            "-m",
            "sglang_omni.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(asr_port),
            "--model-path",
            arguments.asr_path,
            "--model-name",
            QWEN3_ASR_MODEL_PATH,
            "--asr.gpu",
            "0",
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        arguments.output / "asr.log",
        asr_port,
        timeout=SERVER_START_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    try:
        asr = load_qwen3_asr(asr_port)
        for report in reports:
            for source in ("native", "checkpoint"):
                transcript = transcribe(asr, report[f"{source}_audio"], "en", "cuda:0")
                result = apply_wer(
                    SampleOutput(
                        sample_id=report["sample_id"],
                        target_text=report["returned_text"],
                    ),
                    transcript,
                    "en",
                )
                assert result.is_success
                report[f"{source}_transcript"] = transcript
                report[f"{source}_spoken_wer"] = result.wer
            assert report["native_previously_scored_wer"] == report["native_spoken_wer"]
    finally:
        stop_server(asr_server)
    (arguments.output / "evidence.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
else:
    pass
