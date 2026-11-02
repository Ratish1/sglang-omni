# SPDX-License-Identifier: Apache-2.0
"""The served code predictor, the module Qwen3-Omni shares with Qwen3-TTS, on the checkpoint's weights against HF's.

Real talker hidden states and codes come from HF's own FP32 talker over real prefill rows. Omni's served
predictor_forward_tokens runs them teacher-forced at every batch size the talker serves, fused up to the fused rows and
on the plain path above them, with an opening pair and then one token per pass as served. HF's predictor in FP32 is
the truth and its BF16 copy the floor. Measured per pass and batch size: the final-norm hidden state's relative L2, the
logits' relative L2, argmax agreement (omni serves the argmax) and the overlap of HF's top-k 50, top-p 0.8 draw.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration

from tests.test_model.qwen3_omni_component_accuracy.comparison import (
    CaseRecord,
    deviation,
    floor_violations,
    geometric_mean_ratio,
    reference_copy,
)
from tests.unit_test.qwen3_omni.test_predictor_kernels import (
    build_talker,
    fuse,
    reset_caches,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

BATCH_SIZES = (1, 2, 12, 31, 32, 33, 64, 65, 128)
FRAMES_PER_REQUEST = 40
REQUESTS = 8
THETA_POSITIONS = 64


def real_frames(
    truth: torch.nn.Module, rows_path: str, device: str, eos: int, pad: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Final hidden states and 16 codes per frame from HF's FP32 talker sampling over real prefill rows."""
    rows = torch.load(rows_path, map_location="cpu")
    torch.manual_seed(20261102)
    hidden_rows = []
    code_rows = []
    vocab = truth.config.text_config.vocab_size
    suppressed = [index for index in range(vocab - 1024, vocab) if index != eos]
    with torch.inference_mode():
        for row in list(rows.values())[:REQUESTS]:
            prefill = row["prefill"].to(device).float()[None]
            result = truth.generate(
                inputs_embeds=prefill,
                trailing_text_hidden=row["trailing"].to(device).float()[None],
                tts_pad_embed=row["pad"].to(device).float()[None, None],
                talker_input_ids=torch.full((1, prefill.shape[1]), pad, device=device),
                attention_mask=torch.ones(1, prefill.shape[1], device=device, dtype=torch.long),
                max_new_tokens=FRAMES_PER_REQUEST,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                eos_token_id=eos,
                repetition_penalty=1.05,
                suppress_tokens=suppressed,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            steps = result.hidden_states
            for step_index in range(1, len(steps)):
                codes = steps[step_index][-1]
                if codes is None:
                    continue
                else:
                    pass
                hidden_rows.append(steps[step_index - 1][0][-1][:, -1].float())
                code_rows.append(codes[:, :, 0] if codes.ndim == 3 else codes)
    return torch.cat(hidden_rows), torch.cat(code_rows)


def load_real_weights(
    talker, predictor: torch.nn.Module, theta_table: torch.Tensor
) -> None:
    """Puts the checkpoint's predictor weights in the served layer shape, q, k and v stacked, gate and up stacked."""
    served = talker.code_predictor.model
    with torch.no_grad():
        for layer, source in zip(served.layers, predictor.model.layers):
            attention = layer.self_attn
            reference = source.self_attn
            attention.qkv_proj.weight.copy_(
                torch.cat(
                    [
                        reference.q_proj.weight,
                        reference.k_proj.weight,
                        reference.v_proj.weight,
                    ]
                )
            )
            attention.o_proj.weight.copy_(reference.o_proj.weight)
            attention.q_norm.weight.copy_(reference.q_norm.weight)
            attention.k_norm.weight.copy_(reference.k_norm.weight)
            attention.rotary_emb.cos_sin_cache.copy_(theta_table)
            layer.mlp.gate_up_proj.weight.copy_(
                torch.cat([source.mlp.gate_proj.weight, source.mlp.up_proj.weight])
            )
            layer.mlp.down_proj.weight.copy_(source.mlp.down_proj.weight)
            layer.input_layernorm.weight.copy_(source.input_layernorm.weight)
            layer.post_attention_layernorm.weight.copy_(
                source.post_attention_layernorm.weight
            )
        served.norm.weight.copy_(predictor.model.norm.weight)


def reference_passes(
    owner: torch.nn.Module, hidden: torch.Tensor, codes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF's final-norm states at the rows the heads read and its logits: [batch, groups - 1, ...]."""
    tables = owner.code_predictor.get_input_embeddings()
    inputs = torch.stack(
        [hidden, owner.model.get_input_embeddings()(codes[:, 0])]
        + [tables[group](codes[:, group + 1]) for group in range(len(tables) - 1)],
        dim=1,
    )
    states = owner.code_predictor.model(inputs_embeds=inputs).last_hidden_state[:, 1:]
    logits = torch.stack(
        [
            head(states[:, group])
            for group, head in enumerate(owner.code_predictor.lm_head)
        ],
        dim=1,
    )
    return states, logits


def served_passes(
    talker,
    floor: torch.nn.Module,
    hidden: torch.Tensor,
    codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Omni's served forward, teacher-forced: the opening pair, then one token per pass."""
    tables = floor.code_predictor.get_input_embeddings()
    batch_size = hidden.shape[0]
    reset_caches(talker)
    states = []
    logits = []
    pair = torch.stack(
        [hidden, floor.model.get_input_embeddings()(codes[:, 0])], dim=1
    ).clone()
    state = talker.predictor_forward_tokens(
        token_embeds=pair, batch_size=batch_size, cache_len=0
    )[:, 1:2].clone()
    cache_len = 2
    for group, head in enumerate(floor.code_predictor.lm_head):
        states.append(state[:, 0])
        logits.append(head(state[:, 0]))
        if group < len(floor.code_predictor.lm_head) - 1:
            token = tables[group](codes[:, group + 1])[:, None].clone()
            state = talker.predictor_forward_tokens(
                token_embeds=token, batch_size=batch_size, cache_len=cache_len
            ).clone()
            cache_len += 1
        else:
            pass
    return torch.stack(states, dim=1), torch.stack(logits, dim=1)


def draw_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    kept = probabilities.topk(50, dim=-1)
    shares = kept.values / kept.values.sum(-1, keepdim=True)
    beyond = shares.cumsum(-1) > 0.8
    beyond[..., 1:] = beyond[..., :-1].clone()
    beyond[..., 0] = False
    shares = shares.masked_fill(beyond, 0.0)
    filtered = torch.zeros_like(probabilities).scatter_(-1, kept.indices, shares)
    return filtered / filtered.sum(-1, keepdim=True)


def test_code_predictor_follows_reference(
    reference_model: Qwen3OmniMoeForConditionalGeneration,
    component_device: str,
    report_dir: Path | None,
) -> None:
    rows_path = os.environ.get("QWEN3_OMNI_ACCURACY_TALKER_ROWS")
    if rows_path is None:
        pytest.skip("Set QWEN3_OMNI_ACCURACY_TALKER_ROWS to the prefill rows file")
    else:
        pass
    config = reference_model.config
    truth = reference_copy(
        reference_model.talker, device=component_device, dtype=torch.float32
    )
    floor = reference_copy(
        reference_model.talker, device=component_device, dtype=torch.bfloat16
    )
    hidden, codes = real_frames(
        truth,
        rows_path,
        component_device,
        config.talker_config.codec_eos_token_id,
        config.tts_pad_token_id,
    )
    device = torch.device(component_device)
    talker = build_talker(device, seed=0, max_bs=max(BATCH_SIZES))
    predictor_config = truth.config.code_predictor_config
    rotary = truth.code_predictor.model.rotary_emb
    positions = torch.arange(THETA_POSITIONS, device=device, dtype=torch.float32)
    angles = torch.outer(positions, rotary.inv_freq.float().to(device))
    theta_table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    assert predictor_config.head_dim == talker.code_predictor.model.layers[0].self_attn.head_dim
    load_real_weights(talker, floor.code_predictor, theta_table)
    talker = fuse(talker)
    records: list[CaseRecord] = []
    reports = []
    frames = hidden.shape[0]
    with torch.inference_mode():
        for batch_size in BATCH_SIZES:
            rows = torch.arange(batch_size, device=device) % frames
            batch_hidden = hidden[rows]
            batch_codes = codes[rows]
            truth_states, truth_logits = reference_passes(truth, batch_hidden, batch_codes)
            floor_states, floor_logits = reference_passes(
                floor, batch_hidden.bfloat16(), batch_codes
            )
            omni_states, omni_logits = served_passes(
                talker, floor, batch_hidden.bfloat16(), batch_codes
            )
            for name, omni_value, floor_value, truth_value in (
                ("states", omni_states, floor_states, truth_states),
                ("logits", omni_logits, floor_logits, truth_logits),
            ):
                records.append(
                    CaseRecord(
                        case=f"{name}/{batch_size}",
                        omni=deviation(omni_value, truth_value),
                        floor=deviation(floor_value, truth_value),
                    )
                )
            p_truth = draw_probabilities(truth_logits)
            choice = truth_logits.argmax(-1)
            reports.append(
                {
                    "batch_size": batch_size,
                    "path": "fused" if talker.predictor_fused_layers.covers(batch_size) else "plain",
                    "omni_argmax_agreement": (omni_logits.argmax(-1) == choice).float().mean().item(),
                    "floor_argmax_agreement": (floor_logits.argmax(-1) == choice).float().mean().item(),
                    "omni_overlap": (
                        1 - 0.5 * (p_truth - draw_probabilities(omni_logits)).abs().sum(-1)
                    ).mean().item(),
                    "floor_overlap": (
                        1 - 0.5 * (p_truth - draw_probabilities(floor_logits)).abs().sum(-1)
                    ).mean().item(),
                    "omni_vs_floor_logits_relative_l2": deviation(omni_logits, floor_logits).relative_l2,
                }
            )
    if report_dir is not None:
        (report_dir / "code_predictor.json").write_text(
            json.dumps(
                {
                    "records": [record.as_json() for record in records],
                    "requests": reports,
                },
                indent=1,
            )
        )
    else:
        pass
    print(
        f"\n[code_predictor] floor ratio geometric mean {geometric_mean_ratio(records):.3f}, {frames} real frames\n"
        + "\n".join(json.dumps(report) for report in reports)
    )
    violations = floor_violations(records)
    assert not violations, "the code predictor strays past the BF16 floor:\n" + "\n".join(violations)
