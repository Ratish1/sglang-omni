# SPDX-License-Identifier: Apache-2.0
"""Measure speaker similarity when the native waveform batch changes but codes do not."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from scripts.debug_qwen3_omni_codec_waveform import load_native_scheduler, replay
from sglang_omni.client.audio import encode_wav


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--witness-directory", type=Path, required=True)
    parser.add_argument("--waveform-replay-report", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    arguments = parser.parse_args()

    scheduler = load_native_scheduler(
        arguments.server_config, arguments.model_path, arguments.gpu_id
    )
    directory = arguments.witness_directory
    manifests = json.loads((directory / "witnesses.json").read_bytes())
    frames_by_request = json.loads((directory / "codec-histories.json").read_bytes())
    expected = json.loads(arguments.waveform_replay_report.read_bytes())["comparisons"]
    sample_by_request = {
        row["directory"]: row["sample_id"]
        for row in manifests
        if row["request_count"] == 1
    }
    reference_by_sample = {
        sample.sample_id: sample.ref_audio
        for sample in load_seedtts_samples(SEEDTTS_DATASET_ID)
    }
    histories = {
        request_id: torch.tensor(
            frames_by_request[request_id], device=scheduler.device, dtype=torch.long
        )
        for request_id in sample_by_request
    }
    reference_paths = [
        reference_by_sample[sample_id] for sample_id in sample_by_request.values()
    ]
    with TemporaryDirectory(
        prefix="held-code-waveforms-", dir=arguments.waveform_replay_report.parent
    ) as temporary_directory:
        output = Path(temporary_directory)
        waveforms_by_mode: dict[str, list[str]] = {}
        for name, batch_windows in (("graph_solo", False), ("graph_peers", True)):
            windows_by_request, _ = replay(
                scheduler, histories, batch_windows=batch_windows
            )
            waveform_paths = []
            for request_id, windows in windows_by_request.items():
                waveform = encode_wav(np.concatenate(windows), scheduler.sample_rate)
                assert (
                    hashlib.sha256(waveform).hexdigest()
                    == expected[request_id]["replay_sha256"][name]
                ), (request_id, name)
                wav_path = output / f"{name}-{request_id}.wav"
                wav_path.write_bytes(waveform)
                waveform_paths.append(str(wav_path))
            waveforms_by_mode[name] = waveform_paths
        scheduler.on_serving_stop()
        assets = ensure_speaker_similarity_assets()
        scorer = WavLMSpeakerSimilarity(
            finetune_checkpoint=assets.finetune_checkpoint,
            wavlm_base=assets.wavlm_base,
            device=str(scheduler.device),
        )
        solo_paths = waveforms_by_mode["graph_solo"]
        peer_paths = waveforms_by_mode["graph_peers"]
        solo = scorer.score_batch(reference_paths, solo_paths)
        peers = scorer.score_batch(reference_paths, peer_paths)
        repeat = scorer.score_batch(reference_paths, solo_paths)
        assert solo == repeat
        paired = scorer.score_batch(solo_paths, peer_paths)
    scorer_hashes = {}
    for asset in (assets.finetune_checkpoint, assets.wavlm_base):
        with asset.open("rb") as checkpoint:
            scorer_hashes[asset.name] = hashlib.file_digest(
                checkpoint, "sha256"
            ).hexdigest()
    print(
        json.dumps(
            {
                "scorer_sha256": scorer_hashes,
                "samples": {
                    sample_id: {
                        "reference_to_solo": solo_score,
                        "reference_to_peers": peer_score,
                        "solo_to_peers": paired_score,
                    }
                    for sample_id, solo_score, peer_score, paired_score in zip(
                        sample_by_request.values(), solo, peers, paired, strict=True
                    )
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
