# SPDX-License-Identifier: Apache-2.0
"""Verify actual media scatter and extracted-video audio across the flow matrix."""

from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--flow-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    else:
        pass
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path
    ).thinker_config.text_config
    expected_cases = {
        (modalities, outputs, stream, extraction)
        for count in range(1, 5)
        for modalities in combinations(("text", "audio", "image", "video"), count)
        for outputs, stream, extraction in product(
            (("text",), ("audio",), ("text", "audio")), (False, True), (False, True)
        )
    }
    records = {}
    for summary_file in sorted(arguments.flow_root.glob("flow-*/summary.json")):
        summary = json.loads(summary_file.read_text())
        assert summary["successes"] == summary["cases"]
        for result in summary["results"]:
            request_id = result["request_id"]
            request = json.loads(
                (summary_file.parent / f"{request_id}.request.json").read_text()
            )
            modalities = tuple(result["input_modalities"])
            extraction = "video-audio-True" in result["case"]
            case = (
                modalities,
                tuple(request["modalities"]),
                request["stream"],
                extraction,
            )
            assert case not in records
            captured = torch.load(
                arguments.trace_root / request_id / "thinker-prefill.pt",
                weights_only=True,
            )
            token_count = int(captured["live_tokens"])
            input_ids = captured["input_ids"][:token_count]
            embeddings = captured["input_embeds"][:token_count]
            assert embeddings.shape == (token_count, config.hidden_size)
            assert torch.isfinite(embeddings).all()
            assert captured["positions"].shape[-1] >= token_count
            media_slots = int((input_ids >= config.vocab_size).sum())
            has_vision = "image" in modalities or "video" in modalities
            assert ("deepstack" in captured) == has_vision
            if has_vision:
                visual_mask = captured["visual_masks"][:token_count].bool()
                visual_slots = int(visual_mask.sum())
                assert (
                    torch.count_nonzero(
                        captured["deepstack"][:token_count][~visual_mask]
                    )
                    == 0
                )
            else:
                visual_slots = 0
            audio_slots = media_slots - visual_slots
            assert (visual_slots > 0) == has_vision
            assert audio_slots >= 0
            assert (audio_slots > 0) == (
                "audio" in modalities or ("video" in modalities and extraction)
            )
            records[case] = {
                "request_id": request_id,
                "modalities": modalities,
                "outputs": request["modalities"],
                "stream": request["stream"],
                "video_audio_extraction": extraction,
                "tokens": token_count,
                "visual_slots": visual_slots,
                "audio_slots": audio_slots,
            }
    assert set(records) == expected_cases
    extraction_pairs = 0
    for case, extracted in records.items():
        modalities, outputs, stream, extraction = case
        if "video" in modalities and extraction:
            without_audio = records[(modalities, outputs, stream, False)]
            assert extracted["visual_slots"] == without_audio["visual_slots"]
            assert extracted["audio_slots"] > without_audio["audio_slots"]
            extraction_pairs += 1
        else:
            pass
    report = {
        "cases": len(records),
        "video_extraction_pairs": extraction_pairs,
        "results": list(records.values()),
        "scope": "The cached-media request builder replaces placeholder IDs with out-of-vocabulary cache pads. Real slots and dense deepstack masks, not raw checkpoint token IDs, establish that each requested modality reached Thinker. This proves scatter presence and branch activation, not embedding ordering or semantic accuracy.",
    }
    arguments.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps({"cases": len(records), "video_extraction_pairs": extraction_pairs})
    )


if __name__ == "__main__":
    main()
