# SPDX-License-Identifier: Apache-2.0
"""Check native conditioning handoffs before comparing their producers."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import difference, validate


def audit(directory: Path, sample_id: str) -> dict[str, torch.Tensor]:
    captures = directory / "captures"
    provenance = captures / "provenance" / sample_id
    encoder = torch.load(
        provenance / "audio-encoder.pt", map_location="cpu", weights_only=True
    )
    encoder_metadata = json.loads((provenance / "audio-encoder.json").read_text())
    prompt = torch.load(
        provenance / "talker-prompt.pt", map_location="cpu", weights_only=True
    )
    prompt_metadata = json.loads((provenance / "talker-prompt.json").read_text())
    frames = validate(captures / sample_id)
    positions = torch.tensor(prompt_metadata["captured_positions"], dtype=torch.long)
    assert positions.numel() > 0 and torch.all(positions[1:] > positions[:-1])
    assert torch.equal(prompt["input_embeds"], frames[0]["prefill_embeddings"])
    assert torch.equal(prompt["input_ids"], frames[0]["input_ids"])
    received_hidden = prompt["prompt_hidden"][positions]
    exported_hidden = torch.empty_like(received_hidden)
    inserted_embeddings = torch.empty_like(received_hidden)
    hidden_visits = torch.zeros(positions.numel(), dtype=torch.long)
    input_visits = torch.zeros_like(hidden_visits)
    for path in sorted(provenance.glob("thinker-hidden-*.pt")):
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        selected = torch.searchsorted(positions, snapshot["prompt_positions"])
        assert torch.equal(positions[selected], snapshot["prompt_positions"])
        assert torch.all(hidden_visits[selected] == 0)
        exported_hidden[selected] = snapshot["hidden_states"]
        hidden_visits[selected] += 1
    assert torch.all(
        hidden_visits == 1
    ), "Uncaptured prefix-cache history cannot establish provenance"
    assert torch.equal(
        received_hidden, exported_hidden
    ), "Prompt hidden handoff differs"
    for path in sorted(provenance.glob("thinker-input-*.pt")):
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        metadata = json.loads(path.with_suffix(".json").read_text())
        prefix = metadata["prefix_tokens"]
        length = metadata["extend_tokens"]
        selected = (positions >= prefix) & (positions < prefix + length)
        assert torch.all(input_visits[selected] == 0)
        inserted_embeddings[selected] = snapshot["input_embeds"][
            positions[selected] - prefix
        ]
        input_visits[selected] += 1
    assert torch.all(input_visits == 1)
    encoded_embeddings = encoder["audio_embeds"][
        encoder_metadata["token_start"] : encoder_metadata["token_end"]
    ]
    assert torch.equal(
        inserted_embeddings, encoded_embeddings
    ), "Audio embedding insertion differs"
    row_start = encoder_metadata["row_start"]
    row_count = encoder_metadata["row_count"]
    lengths = encoder["audio_feature_lengths"][row_start : row_start + row_count]
    features = encoder["input_features"][row_start : row_start + row_count]
    mask = encoder["feature_attention_mask"][row_start : row_start + row_count]
    assert torch.equal(
        mask.bool(), torch.arange(mask.shape[-1]).unsqueeze(0) < lengths.unsqueeze(1)
    )
    packed_features = torch.cat(
        [row[:, : int(length)] for row, length in zip(features, lengths, strict=True)],
        dim=-1,
    )
    generated = json.loads((directory / "seedtts/generated.json").read_text())
    assert len(generated) == 1088 and all(row["is_success"] for row in generated)
    return {
        "packed_audio_features": packed_features,
        "audio_feature_lengths": lengths,
        "encoder_output": encoded_embeddings,
        "thinker_inserted_audio": inserted_embeddings,
        "thinker_exported_hidden": exported_hidden,
        "talker_received_hidden": received_hidden,
        "talker_projected_prompt": prompt["input_embeds"],
        "decode_text_rows": torch.stack(
            [frame["text_embedding"] for frame in frames[1:]]
        ),
        "prompt_token_ids": prompt["prompt_ids"],
        "assistant_token_ids": prompt["assistant_token_ids"],
        "codec_history": torch.stack([frame["codec_ids"] for frame in frames]),
    }


def main(root: Path, sample_id: str) -> None:
    torch.set_num_threads(1)
    tensors = [audit(root / arm, sample_id) for arm in ("first", "second")]
    identities = {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for arm in ("first", "second")
        for path in (root / arm / "captures" / "provenance" / sample_id).iterdir()
    }
    report = {
        "sample_id": sample_id,
        "both_handoffs_exact": True,
        "both_peer_loads_successful": 1088,
        "differences": {
            name: difference(tensors[0][name], tensors[1][name]) for name in tensors[0]
        },
        "capture_sha256": identities,
    }
    with (root / "conditioning-provenance.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report["differences"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("sample_id")
    arguments = parser.parse_args()
    main(arguments.root, arguments.sample_id)
else:
    pass
