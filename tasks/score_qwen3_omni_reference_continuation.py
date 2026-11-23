# SPDX-License-Identifier: Apache-2.0
"""Score native and checkpoint codec histories through one checkpoint decoder."""

import argparse
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from qwen3_omni_corpus_support import SERVER_TIMEOUT_SECONDS, server_command
from replay_qwen3_omni_reference_continuation import audit_weights
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


@dataclass(kw_only=True)
class RenderedClip:
    sample_id: str
    arm: str
    waveform_path: str
    waveform_sha256: str
    audio_duration_seconds: float


@torch.inference_mode()
def render(
    checkpoint: Path,
    reference_root: Path,
    sample_ids: list[str],
    output: Path,
    gpu_index: int,
    sample_rate: int,
) -> list[RenderedClip]:
    config = Qwen3OmniMoeConfig.from_pretrained(checkpoint)
    model, loading = Qwen3OmniMoeCode2Wav.from_pretrained(
        checkpoint,
        config=config.code2wav_config,
        dtype=torch.bfloat16,
        device_map={"": gpu_index},
        key_mapping={r"^code2wav\.": ""},
        attn_implementation="sdpa",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
    model.eval()
    audit_weights(model, checkpoint, output)
    clips: list[RenderedClip] = []
    for sample_id in sample_ids:
        sample_directory = output / sample_id
        sample_directory.mkdir()
        for arm, filename in (
            ("native_codes", "conditioning.pt"),
            ("reference_0", "reference-0.pt"),
            ("reference_1", "reference-1.pt"),
        ):
            codes = torch.load(
                reference_root / sample_id / filename,
                weights_only=True,
                map_location="cpu",
            )["codec_ids"]
            assert int(codes[-1, 0]) == config.talker_config.codec_eos_token_id
            speech_codes = codes[:-1].T.unsqueeze(0).to(model.device)
            assert speech_codes.shape[1] == config.code2wav_config.num_quantizers
            assert int(speech_codes.min()) >= 0
            assert int(speech_codes.max()) < config.code2wav_config.codebook_size
            waveform = model.chunked_decode(speech_codes).float().reshape(-1).cpu()
            assert torch.isfinite(waveform).all()
            wave_bytes = encode_wav(waveform.numpy(), sample_rate)
            waveform_path = sample_directory / f"{arm}.wav"
            waveform_path.write_bytes(wave_bytes)
            clip = RenderedClip(
                sample_id=sample_id,
                arm=arm,
                waveform_path=str(waveform_path),
                waveform_sha256=sha256(wave_bytes).hexdigest(),
                audio_duration_seconds=waveform.numel() / sample_rate,
            )
            clips.append(clip)
            print(json.dumps(asdict(clip)), flush=True)
    (output / "rendered.json").write_text(
        json.dumps([asdict(clip) for clip in clips], indent=2)
    )
    return clips


def main(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir()
    torch.cuda.set_device(arguments.gpu)
    results = json.loads((arguments.reference_root / "results.json").read_text())
    sample_ids = sorted({row["sample_id"] for row in results})
    assert len(results) == 2 * len(sample_ids)
    for sample_id in sample_ids:
        assert sorted(
            row["repeat"] for row in results if row["sample_id"] == sample_id
        ) == [0, 1]
    clips = render(
        arguments.checkpoint,
        arguments.reference_root,
        sample_ids,
        arguments.output,
        arguments.gpu,
        arguments.sample_rate,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    native = json.loads((arguments.native_root / "mmsu/mmsu_results.json").read_text())
    responses = {row["sample_id"]: row for row in native["per_sample"]}
    previous_scores = {row["id"]: row for row in native["wer"]["per_sample"]}
    hashes = json.loads((arguments.native_root / "waveform-hashes.json").read_text())
    for sample_id in sample_ids:
        relative_path = f"mmsu/audio/{sample_id}.wav"
        path = arguments.native_root / relative_path
        assert sha256(path.read_bytes()).hexdigest() == hashes[relative_path]
        clips.append(
            RenderedClip(
                sample_id=sample_id,
                arm="native_waveform",
                waveform_path=str(path),
                waveform_sha256=hashes[relative_path],
                audio_duration_seconds=previous_scores[sample_id]["audio_duration_s"],
            )
        )
    port = find_available_port()
    server = start_server_from_cmd(
        [
            *server_command(),
            "--port",
            str(port),
            "--model-path",
            str(arguments.asr_checkpoint),
            "--model-name",
            QWEN3_ASR_MODEL_PATH,
            "--asr.gpu",
            str(arguments.gpu),
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        arguments.output / "asr.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    scores = []
    try:
        asr = load_qwen3_asr(port)
        for clip in clips:
            assert (
                sha256(Path(clip.waveform_path).read_bytes()).hexdigest()
                == clip.waveform_sha256
            )
            transcript = transcribe(
                asr, clip.waveform_path, "en", f"cuda:{arguments.gpu}"
            )
            score = apply_wer(
                SampleOutput(
                    sample_id=clip.sample_id,
                    target_text=responses[clip.sample_id]["raw_response"],
                    audio_duration_s=clip.audio_duration_seconds,
                ),
                transcript,
                "en",
            )
            assert score.is_success and score.is_transcribed
            report = {
                **asdict(clip),
                "score": asdict(score),
                "matches_previous_native_transcript": (
                    score.hyp_norm == previous_scores[clip.sample_id]["hyp_norm"]
                    if clip.arm == "native_waveform"
                    else None
                ),
            }
            scores.append(report)
            (arguments.output / "scores.json").write_text(json.dumps(scores, indent=2))
            print(json.dumps(report), flush=True)
    finally:
        stop_server(server)
    assert len(scores) == 4 * len(sample_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("asr_checkpoint", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("native_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    main(parser.parse_args())
else:
    pass
