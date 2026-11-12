"""Join actual multimodal hidden writers, virtual-slot reads and Talker inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-directory", type=Path, required=True)
    parser.add_argument("--conditioning-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    write_files = sorted((arguments.delivery_directory / "writes").glob("*.pt"))
    assert write_files
    assert [path.stem for path in write_files] == [
        f"{index:06d}" for index in range(1, len(write_files) + 1)
    ]
    reports = []
    for request_id in arguments.request_id:
        handoff = torch.load(
            arguments.delivery_directory / request_id / "handoff.pt", weights_only=True
        )
        identity = json.loads(
            (arguments.delivery_directory / request_id / "identity.json").read_text()
        )
        assert identity["request_id"] == request_id
        positions = handoff["prompt_positions"]
        slots = handoff["virtual_slots"]
        outgoing = handoff["outgoing_hidden"]
        assert len(positions) == len(slots) == len(outgoing) > 0
        assert handoff["writers"].shape == (len(positions), 2)
        assert positions.dtype == slots.dtype == torch.int64
        assert outgoing.dtype == torch.bfloat16 and outgoing.isfinite().all()
        assert len(slots.unique()) == len(slots)
        assert 0 < handoff["write_count"] <= len(write_files)
        latest: dict[int, tuple[int, int]] = {}
        for path in write_files[: handoff["write_count"]]:
            packet = torch.load(path, weights_only=True)
            assert packet["write_id"] == int(path.stem)
            assert packet["forward_step"] > 0
            producer = packet["producer_hidden"]
            assert producer.dtype == torch.bfloat16 and producer.isfinite().all()
            assert torch.equal(producer, packet["auxiliary_hidden"])
            prefix = packet["prefix_length"]
            length = packet["extend_length"]
            assert torch.equal(
                packet["actual_input_ids"],
                packet["scheduled_token_ids"][prefix : prefix + length],
            )
            assert (
                len(packet["virtual_slots"])
                == len(packet["physical_slots"])
                == len(producer)
            )
            assert len(packet["prompt_positions"]) == len(producer)
            assert (
                (packet["prompt_positions"] >= prefix)
                & (packet["prompt_positions"] < prefix + length)
            ).all()
            assert torch.equal(
                packet["packed_rows"],
                packet["prompt_positions"] - prefix + packet["batch_offset"],
            )
            assert (packet["positions"].reshape(len(producer), -1) >= 0).all()
            for row, slot in enumerate(packet["virtual_slots"].tolist()):
                latest[slot] = (packet["write_id"], row)
        lineage = []
        for row, (position, slot) in enumerate(
            zip(positions.tolist(), slots.tolist(), strict=True)
        ):
            writer_id, writer_row = handoff["writers"][row].tolist()
            assert 0 < writer_id <= handoff["write_count"]
            assert latest[slot] == (writer_id, writer_row)
            source = torch.load(write_files[writer_id - 1], weights_only=True)
            assert 0 <= writer_row < len(source["producer_hidden"])
            assert int(source["virtual_slots"][writer_row]) == slot
            assert int(source["prompt_positions"][writer_row]) == position
            assert torch.equal(
                source["causal_token_ids"][: position + 1],
                handoff["causal_token_ids"][: position + 1],
            )
            assert torch.equal(source["producer_hidden"][writer_row], outgoing[row])
            if handoff["mrope_positions"].numel():
                assert torch.equal(
                    source["positions"][writer_row], handoff["mrope_positions"][:, row]
                )
            else:
                pass
            lineage.append(
                {
                    "prompt_position": position,
                    "virtual_slot": slot,
                    "writer_request_id": source["request_id"],
                    "writer_id": writer_id,
                    "writer_row": writer_row,
                    "source_physical_slot": int(source["physical_slots"][writer_row]),
                }
            )
        received = torch.load(
            arguments.conditioning_directory / request_id / "prompt.pt",
            weights_only=True,
        )
        assert bool(received["has_prompt_hidden"])
        original_prompt = received["prompt_input_ids"]
        multimodal = torch.isin(original_prompt, handoff["multimodal_token_ids"])
        assert torch.equal(positions, multimodal.nonzero(as_tuple=True)[0])
        assert original_prompt.shape == handoff["causal_token_ids"].shape
        assert torch.equal(
            original_prompt[~multimodal], handoff["causal_token_ids"][~multimodal]
        )
        assert torch.equal(received["prompt_hidden"][positions], outgoing)
        report = {
            "request_id": request_id,
            "multimodal_rows": len(positions),
            "source_write_count_at_read": handoff["write_count"],
            "cached_owner_rows": sum(
                row["writer_request_id"] != request_id for row in lineage
            ),
            "producer_auxiliary_equal": True,
            "producer_outgoing_equal": True,
            "outgoing_received_equal": True,
            "virtual_slot_lineage": lineage,
            "scope": "Actual captured producer, request causal-token prefix, virtual-slot latest-writer lineage and received Talker rows. Source semantics and projected-prefix construction are separate checks, not inferred from these equalities.",
        }
        reports.append(report)
        (arguments.output_directory / f"{request_id}.json").write_text(
            json.dumps(report, indent=2)
        )
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(reports, indent=2)
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
else:
    pass
