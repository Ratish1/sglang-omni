"""Recover cached audio inputs only from their observed original hidden rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pydantic import BaseModel, TypeAdapter
from transformers import Qwen3OmniMoeConfig


class PrefixOwner(BaseModel):
    owner_request_id: str
    prefix: int


class PrefixProof(BaseModel):
    request_id: str
    owners: list[PrefixOwner]


def recover_audio_prefix(
    producer_directory: Path,
    proof_file: Path,
    request_id: str,
    audio_positions: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    input_embeddings: torch.Tensor,
    covered_positions: torch.Tensor,
    expected_positions: torch.Tensor,
) -> list[PrefixOwner]:
    proofs = TypeAdapter(list[PrefixProof]).validate_json(proof_file.read_bytes())
    proof = next(proof for proof in proofs if proof.request_id == request_id)
    actual_audio = torch.load(
        producer_directory / request_id / "audio-conditioning.pt", weights_only=True
    )["audio_embeds"]
    first_current_forward = min(
        filename.stat().st_mtime_ns
        for filename in (producer_directory / request_id).glob("thinker-input-*.pt")
    )
    recovered: list[PrefixOwner] = []
    for owner in proof.owners:
        source = producer_directory / owner.owner_request_id
        inputs = torch.load(
            source / f"thinker-input-{owner.prefix}.pt", weights_only=True
        )
        hidden_file = source / f"thinker-hidden-{owner.prefix}.pt"
        hidden = torch.load(hidden_file, weights_only=True)
        within = (audio_positions >= owner.prefix) & (
            audio_positions < owner.prefix + len(inputs["positions"])
        )
        indices = audio_positions[within] - owner.prefix
        assert torch.equal(inputs["input_embeds"][indices], actual_audio[within])
        assert torch.equal(
            inputs["mrope_positions"][:, indices],
            expected_positions[:, 0, audio_positions[within]].cpu(),
        )
        exact = (hidden[indices] == prompt_hidden_states[audio_positions[within]]).all(
            dim=-1
        )
        positions = audio_positions[within][exact]
        missing = ~covered_positions[positions]
        if bool(missing.any()):
            assert hidden_file.stat().st_mtime_ns <= first_current_forward
            positions = positions[missing]
            input_embeddings[positions] = inputs["input_embeds"][
                positions - owner.prefix
            ].to(input_embeddings.device)
            covered_positions[positions] = True
            recovered.append(owner)
        else:
            pass
    assert bool(
        covered_positions[audio_positions].all()
    ), "cached audio has no verified source"
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--producer-directory", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    assert not arguments.output_file.exists()
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    proofs = []
    candidates = sorted(
        arguments.producer_directory.glob("*/thinker-input-*.pt"),
        key=lambda filename: filename.stat().st_mtime_ns,
    )
    for request_id in arguments.request_id:
        directory = arguments.producer_directory / request_id
        prompt = torch.load(
            arguments.capture_directory / "conditioning" / request_id / "prompt.pt",
            map_location="cpu",
            weights_only=True,
        )
        audio = torch.load(
            directory / "audio-conditioning.pt", map_location="cpu", weights_only=True
        )["audio_embeds"]
        audio_positions = (
            prompt["prompt_input_ids"] == config.thinker_config.audio_token_id
        ).nonzero(as_tuple=True)[0]
        assert len(audio_positions) == len(audio)
        audio_row_indices = {
            int(position): row for row, position in enumerate(audio_positions)
        }
        missing_positions = set(audio_row_indices)
        current_inputs = list(directory.glob("thinker-input-*.pt"))
        first_current_forward = min(
            filename.stat().st_mtime_ns for filename in current_inputs
        )
        for filename in current_inputs:
            prefix = int(filename.stem.removeprefix("thinker-input-"))
            inputs = torch.load(filename, map_location="cpu", weights_only=True)
            missing_positions.difference_update(
                range(prefix, prefix + len(inputs["positions"]))
            )
        proof = PrefixProof(request_id=request_id, owners=[])
        for filename in candidates:
            if not missing_positions:
                break
            else:
                pass
            prefix = int(filename.stem.removeprefix("thinker-input-"))
            hidden_file = filename.with_name(f"thinker-hidden-{prefix}.pt")
            if (
                prefix > max(missing_positions)
                or not hidden_file.exists()
                or hidden_file.stat().st_mtime_ns > first_current_forward
            ):
                continue
            else:
                pass
            inputs = torch.load(filename, map_location="cpu", weights_only=True)
            hidden_states = torch.load(
                hidden_file, map_location="cpu", weights_only=True
            )
            positions = sorted(
                missing_positions.intersection(
                    range(prefix, prefix + len(inputs["positions"]))
                )
            )
            matches = [
                position
                for position in positions
                if torch.equal(
                    inputs["input_embeds"][position - prefix],
                    audio[audio_row_indices[position]],
                )
                and torch.equal(
                    hidden_states[position - prefix], prompt["prompt_hidden"][position]
                )
            ]
            if matches:
                proof.owners.append(
                    PrefixOwner(owner_request_id=filename.parent.name, prefix=prefix)
                )
                missing_positions.difference_update(matches)
            else:
                pass
        assert not missing_positions, (request_id, sorted(missing_positions))
        proofs.append(proof)
        print(proof.model_dump_json(), flush=True)
    arguments.output_file.write_bytes(
        TypeAdapter(list[PrefixProof]).dump_json(proofs, indent=2)
    )


if __name__ == "__main__":
    main()
