"""Apply a guarded producer substitution without changing the cached source rows."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from sglang_omni.scheduling.sglang_backend.prompt_hidden import (
    PromptHiddenRequest,
    PromptHiddenStore,
)


def substitute_prompt_hidden(
    request: PromptHiddenRequest,
    store: PromptHiddenStore,
    outgoing_hidden_states: torch.Tensor,
    destination: Path,
) -> None:
    filename = os.environ.get("SGLANG_OMNI_DIAGNOSTIC_PRODUCER_CONTROL")
    if filename is None:
        return
    else:
        pass
    control = torch.load(filename, map_location="cpu", weights_only=True)
    assert control["request_id"] == request.rid
    assert torch.equal(control["prompt_positions"], request.prompt_hidden_positions)
    assert torch.equal(
        control["causal_token_ids"], torch.tensor(request.origin_input_ids)
    )
    native_hidden_states = control["native_hidden"]
    replacement_hidden_states = control["replacement_hidden"]
    assert isinstance(native_hidden_states, torch.Tensor)
    assert isinstance(replacement_hidden_states, torch.Tensor)
    assert torch.equal(outgoing_hidden_states, native_hidden_states)
    assert replacement_hidden_states.shape == outgoing_hidden_states.shape
    assert replacement_hidden_states.dtype == outgoing_hidden_states.dtype
    assert replacement_hidden_states.device == outgoing_hidden_states.device
    assert bool(torch.isfinite(replacement_hidden_states).all())
    outgoing_hidden_states.copy_(replacement_hidden_states)
    assert torch.equal(outgoing_hidden_states, replacement_hidden_states)
    assert torch.equal(
        store.read(request.kv.req_pool_idx, request.prompt_hidden_positions),
        native_hidden_states,
    )
    torch.save(control, destination / "producer-control.pt")
