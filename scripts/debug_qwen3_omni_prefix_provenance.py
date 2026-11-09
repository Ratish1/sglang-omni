"""Recover cached audio inputs only from their observed original hidden rows."""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel, TypeAdapter


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
