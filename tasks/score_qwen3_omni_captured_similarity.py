# SPDX-License-Identifier: Apache-2.0
"""Score retained native captures against the same pinned reference waveforms."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import benchmarks.metrics.speaker_similarity as similarity_module
from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)

SIMILARITY_SHA256 = "c089056ff6e5023961a961f001a1a716cb71031e8b10a805a63a5115cc62f577"


def main(root: Path, output: Path, device: str) -> None:
    assert not output.exists()
    scorer_source = Path(similarity_module.__file__)
    assert sha256(scorer_source.read_bytes()).hexdigest() == SIMILARITY_SHA256
    arms = (root / "first", root / "second")
    sample_ids = sorted(path.name for path in (arms[0] / "captures").iterdir())
    assert sample_ids
    assert sample_ids == sorted(path.name for path in (arms[1] / "captures").iterdir())
    samples = {
        sample.sample_id: sample for sample in load_seedtts_samples(SEEDTTS_DATASET_ID)
    }
    original_hashes = json.loads((root / "waveform-hashes.json").read_text())
    reference_paths = [samples[sample_id].ref_audio for sample_id in sample_ids]
    assets = ensure_speaker_similarity_assets()
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device=device,
    )
    scores_by_arm = {}
    for arm in arms:
        waveform_paths = [
            arm / "seedtts/audio" / f"{sample_id}.wav" for sample_id in sample_ids
        ]
        for waveform in waveform_paths:
            assert (
                sha256(waveform.read_bytes()).hexdigest()
                == original_hashes[str(waveform.relative_to(root))]
            )
        scores = scorer.score_batch(reference_paths, list(map(str, waveform_paths)))
        repeated = scorer.score_batch(reference_paths, list(map(str, waveform_paths)))
        assert scores == repeated
        scores_by_arm[arm.name] = dict(zip(sample_ids, scores, strict=True))
    result = {
        "source": str(root),
        "scorer_sha256": SIMILARITY_SHA256,
        "reference_hashes": {
            sample_id: sha256(Path(reference).read_bytes()).hexdigest()
            for sample_id, reference in zip(sample_ids, reference_paths, strict=True)
        },
        "waveform_hashes": {
            str(waveform.relative_to(root)): original_hashes[
                str(waveform.relative_to(root))
            ]
            for arm in arms
            for waveform in (
                arm / "seedtts/audio" / f"{sample_id}.wav" for sample_id in sample_ids
            )
        },
        "scores": scores_by_arm,
        "repeat_exact": True,
    }
    with output.open("x") as receipt:
        json.dump(result, receipt, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", required=True)
    arguments = parser.parse_args()
    main(arguments.root, arguments.output, arguments.device)
else:
    pass
