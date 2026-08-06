# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
from dots_tts.models.dots_tts.config import _DiTConfig, _EncoderConfig
from dots_tts.modules.backbone.dit import DiT
from dots_tts.modules.backbone.encoder import VAESemanticEncoder

from sglang_omni.models.dots_tts import tail
from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead

FM_HIDDEN = 32
LATENT_DIM = 6
PATCH_SIZE = 2
NFE = 2


def test_batched_tail_mask_hides_padding_and_preserves_causality() -> None:
    from sglang_omni.models.dots_tts.tail import batched_causal_update_mask

    mask = batched_causal_update_mask(
        capacity_tokens=4,
        valid_persistent=torch.tensor([1, 3]),
        prev_len=2,
        current_len=2,
    )

    assert mask.shape == (2, 1, 4, 8)
    assert mask[0, 0].tolist() == [
        [True, False, False, False, True, False, False, False],
        [True, False, False, False, True, True, False, False],
        [True, False, False, False, True, True, True, True],
        [True, False, False, False, True, True, True, True],
    ]
    assert mask[1, 0].tolist() == [
        [True, True, True, False, True, False, False, False],
        [True, True, True, False, True, True, False, False],
        [True, True, True, False, True, True, True, True],
        [True, True, True, False, True, True, True, True],
    ]


class _TailModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity_field_predictor = DiT(
            in_dim=FM_HIDDEN,
            out_dim=LATENT_DIM,
            transformer_config=_DiTConfig(
                num_layers=2,
                num_heads=2,
                hidden_size=FM_HIDDEN,
                ffn_hidden_size=64,
                modulation=True,
                qk_norm=True,
                rotary_bias=True,
            ),
            mode="meanflow",
        )
        self.coordinate_proj = torch.nn.Linear(LATENT_DIM, FM_HIDDEN)
        self.latent_proj = torch.nn.Linear(LATENT_DIM, FM_HIDDEN)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.normal_(0.0, 0.2)


def _patch_encoder() -> VAESemanticEncoder:
    encoder_config = _EncoderConfig(
        num_layers=1,
        num_heads=2,
        hidden_size=FM_HIDDEN,
        ffn_hidden_size=64,
        causal=True,
    )
    config = type(
        "_EncoderConfigStub",
        (),
        {"patch_size": PATCH_SIZE, "PatchEncoder": encoder_config},
    )()
    return VAESemanticEncoder(in_dim=LATENT_DIM, out_dim=FM_HIDDEN, config=config)


def _build_tail(
    model: _TailModel,
    *,
    slots: int,
    encoder: VAESemanticEncoder | None = None,
):
    encoder = _patch_encoder() if encoder is None else encoder
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.normal_(0.0, 0.2)
    return tail.DotsTtsAcousticTail(
        dit=tail.fuse_dit_for_inference(model),
        coordinate_proj=model.coordinate_proj,
        patch_encoder=encoder,
        spec=tail.DotsTtsTailSpec(
            nfe=NFE,
            patch_capacity=8,
            num_slots=slots,
            hidden_patch_size=1,
            latent_patch_size=PATCH_SIZE,
            latent_dim=LATENT_DIM,
            fm_hidden_size=FM_HIDDEN,
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class _IdentityIO:
    @staticmethod
    def normalize(value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def denormalize(value: torch.Tensor) -> torch.Tensor:
        return value


def _build_flow_head(model: _TailModel, *, slots: int) -> DotsTTSFlowHead:
    flow = object.__new__(DotsTTSFlowHead)
    torch.nn.Module.__init__(flow)
    flow.fm_hidden_size = FM_HIDDEN
    flow.hidden_patch_size = 1
    flow.latent_patch_size = PATCH_SIZE
    flow.latent_dim = LATENT_DIM
    flow.optimize = False
    flow.mode = "meanflow"
    flow.patch_encoder = _patch_encoder()
    flow.hidden_proj = torch.nn.Linear(FM_HIDDEN, FM_HIDDEN)
    flow.latent_proj = model.latent_proj
    flow.coordinate_proj = model.coordinate_proj
    flow.eos_proj = torch.nn.Linear(FM_HIDDEN, 2)
    flow.io = _IdentityIO()
    flow._patch_inference = None
    flow._dit_solver = None
    flow._tail = _build_tail(model, slots=slots, encoder=flow.patch_encoder)
    flow._batched_nfe = NFE
    return flow


def _reference_meanflow(
    dit: torch.nn.Module,
    coordinate_proj: torch.nn.Module,
    sequence: torch.Tensor,
    fm_seq_len: int,
    g_cond: torch.Tensor,
) -> torch.Tensor:
    total = fm_seq_len + PATCH_SIZE
    x_base = sequence.new_zeros(1, total, FM_HIDDEN)
    x_base[:, :fm_seq_len] = sequence[:, :fm_seq_len]
    mask = torch.zeros((1, total, total), dtype=torch.bool)
    block_start = fm_seq_len - 1
    if block_start:
        mask[:, :block_start, :block_start] = torch.ones(
            block_start, block_start, dtype=torch.bool
        ).tril()
    mask[:, block_start:fm_seq_len, :fm_seq_len] = True
    mask[:, block_start:fm_seq_len, fm_seq_len:] = True
    mask[:, fm_seq_len:, :] = True
    positions = torch.arange(total, dtype=torch.float32).reshape(1, total)
    latent = torch.randn(1, PATCH_SIZE, LATENT_DIM)
    times = torch.linspace(0.0, 1.0, NFE + 1)
    for step in range(NFE):
        value = x_base.clone()
        value[:, fm_seq_len:] = coordinate_proj(latent)
        duration = (times[step + 1] - times[step]).expand(1)
        velocity = dit(
            x=value,
            timesteps=times[step].expand(1),
            duration=duration,
            attn_mask=mask,
            pos_ids=positions,
            g_cond=g_cond,
        )[:, fm_seq_len:]
        latent = (latent + duration.reshape(1, 1, 1) * velocity).clone()
    return latent


def test_kv_cached_tail_matches_full_recompute() -> None:
    torch.manual_seed(1234)
    model = _TailModel().eval()
    acoustic_tail = _build_tail(model, slots=1)
    unit = acoustic_tail.spec.unit_len
    g_cond = torch.randn(1, FM_HIDDEN)
    grid = torch.linspace(0.0, 1.0, NFE + 1)
    mods = acoustic_tail.dit.build_mods(
        grid[:-1], duration=grid[1:] - grid[:-1], g_cond=g_cond
    )
    prompt_rows = torch.randn(3 * unit, FM_HIDDEN)
    slot = acoustic_tail.acquire_slot()
    acoustic_tail.initialize_slot_rng(slot, 9)
    acoustic_tail.seed_fm_history(slot, fm_rows=prompt_rows, all_mods=mods)
    sequence = torch.zeros(1, acoustic_tail.spec.dit_cache_tokens + unit, FM_HIDDEN)
    sequence[0, : prompt_rows.size(0)] = prompt_rows
    sequence_len = prompt_rows.size(0)

    hidden = torch.randn(1, FM_HIDDEN)
    sequence[0, sequence_len] = hidden[0]
    sequence_len += 1
    torch.manual_seed(9)
    expected = _reference_meanflow(
        acoustic_tail.dit,
        model.coordinate_proj,
        sequence,
        sequence_len,
        g_cond,
    )
    actual = acoustic_tail.sample_patches(
        [slot], fm_hidden_rows=hidden, latent_proj=model.latent_proj
    )

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_multi_slot_tail_matches_isolated_reordered_recurrence() -> None:
    torch.manual_seed(2026)
    model = _TailModel().eval()
    batched = _build_tail(model, slots=2)
    isolated = copy.deepcopy(batched)
    batched_slots = [batched.acquire_slot(), batched.acquire_slot()]
    isolated_slots = [isolated.acquire_slot(), isolated.acquire_slot()]
    unit = batched.spec.unit_len
    prompt_rows = [
        torch.randn(2 * unit, FM_HIDDEN),
        torch.randn(4 * unit, FM_HIDDEN),
    ]
    conditions = [torch.randn(1, FM_HIDDEN), torch.randn(1, FM_HIDDEN)]
    grid = torch.linspace(0.0, 1.0, NFE + 1)

    for row in range(2):
        mods = batched.dit.build_mods(
            grid[:-1],
            duration=grid[1:] - grid[:-1],
            g_cond=conditions[row],
        )
        batched.initialize_slot_rng(batched_slots[row], 100 + row)
        isolated.initialize_slot_rng(isolated_slots[row], 100 + row)
        batched.seed_fm_history(
            batched_slots[row],
            fm_rows=prompt_rows[row],
            all_mods=mods,
        )
        isolated.seed_fm_history(
            isolated_slots[row],
            fm_rows=prompt_rows[row],
            all_mods=mods,
        )

    first_hidden = torch.randn(2, FM_HIDDEN)
    batched_first = batched.sample_patches(
        batched_slots,
        fm_hidden_rows=first_hidden,
        latent_proj=model.latent_proj,
    )
    isolated_first = torch.cat(
        [
            isolated.sample_patches(
                [isolated_slots[row]],
                fm_hidden_rows=first_hidden[row : row + 1],
                latent_proj=model.latent_proj,
            )
            for row in range(2)
        ]
    )
    torch.testing.assert_close(
        batched_first,
        isolated_first,
        rtol=3e-4,
        atol=3e-4,
    )

    batched_feedback = batched.encode_feedback(batched_slots, batched_first)
    isolated_feedback = torch.cat(
        [
            isolated.encode_feedback(
                [isolated_slots[row]],
                isolated_first[row : row + 1],
            )
            for row in range(2)
        ]
    )
    torch.testing.assert_close(
        batched_feedback,
        isolated_feedback,
        rtol=3e-4,
        atol=3e-4,
    )

    second_hidden = torch.randn(2, FM_HIDDEN)
    reordered = [1, 0]
    batched_second = batched.sample_patches(
        [batched_slots[row] for row in reordered],
        fm_hidden_rows=second_hidden[reordered],
        latent_proj=model.latent_proj,
    )
    isolated_second = torch.cat(
        [
            isolated.sample_patches(
                [isolated_slots[row]],
                fm_hidden_rows=second_hidden[row : row + 1],
                latent_proj=model.latent_proj,
            )
            for row in reordered
        ]
    )
    torch.testing.assert_close(
        batched_second,
        isolated_second,
        rtol=3e-4,
        atol=3e-4,
    )


def test_slot_rng_resumes_exactly_after_release_and_reacquire() -> None:
    acoustic_tail = _build_tail(_TailModel().eval(), slots=1)
    slot = acoustic_tail.acquire_slot()
    acoustic_tail.initialize_slot_rng(slot, 314)
    acoustic_tail._sample_noise([slot])
    saved_state = acoustic_tail.slot_rng_state(slot)
    expected_next = acoustic_tail._sample_noise([slot])

    acoustic_tail.release_slot(slot)
    resumed_slot = acoustic_tail.acquire_slot()
    acoustic_tail.initialize_slot_rng(resumed_slot, saved_state)
    resumed_next = acoustic_tail._sample_noise([resumed_slot])

    torch.testing.assert_close(resumed_next, expected_next, rtol=0, atol=0)


def test_flow_rematerialization_matches_uninterrupted_next_step() -> None:
    torch.manual_seed(1618)
    flow = _build_flow_head(_TailModel().eval(), slots=2)
    prompt_latents = torch.randn(1, 2 * PATCH_SIZE, LATENT_DIM)
    prefill_hidden = torch.randn(1, 3, FM_HIDDEN)
    prompt_positions = torch.tensor([1, 2])
    schedule = torch.tensor([[0, 1, 1, 1]])

    uninterrupted, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=prompt_latents,
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=41,
    )
    retracted, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=prompt_latents,
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=41,
    )
    for state in (uninterrupted, retracted):
        flow.initialize_flow_history(
            state,
            hidden_states=prefill_hidden,
            prompt_span_positions=prompt_positions,
            audio_span_token_ids={1},
            generation_schedule=schedule,
            prefill_end=3,
            decoded_latent_patches=[],
        )

    first_steps = flow.decode_batch(
        [uninterrupted, retracted],
        hidden_states=prefill_hidden[:, -1].expand(2, -1),
        num_steps=[NFE, NFE],
        ode_methods=["euler", "euler"],
        guidance_scales=[1.0, 1.0],
        eos_thresholds=[2.0, 2.0],
        append_hidden=False,
    )
    torch.testing.assert_close(
        first_steps[0].latent_patch,
        first_steps[1].latent_patch,
        rtol=3e-4,
        atol=3e-4,
    )

    rng_state = flow.suspend_request(retracted)
    rematerialized, _ = flow.new_request(
        max_audio_patch_count=6,
        prompt_latents=prompt_latents,
        speaker_embedding=None,
        speaker_scale=1.0,
        rng=rng_state,
    )
    replayed_feedback = flow.replay_feedback(
        rematerialized,
        [first_steps[1].latent_patch],
    )
    torch.testing.assert_close(
        replayed_feedback[0],
        first_steps[1].feedback_embedding,
        rtol=3e-4,
        atol=3e-4,
    )

    next_hidden = torch.randn(1, FM_HIDDEN)
    flow.initialize_flow_history(
        rematerialized,
        hidden_states=torch.cat([prefill_hidden, next_hidden.unsqueeze(1)], dim=1),
        prompt_span_positions=prompt_positions,
        audio_span_token_ids={1},
        generation_schedule=schedule,
        prefill_end=3,
        decoded_latent_patches=[first_steps[1].latent_patch],
    )
    [expected] = flow.decode_batch(
        [uninterrupted],
        hidden_states=next_hidden,
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[2.0],
        append_hidden=True,
    )
    [actual] = flow.decode_batch(
        [rematerialized],
        hidden_states=next_hidden,
        num_steps=[NFE],
        ode_methods=["euler"],
        guidance_scales=[1.0],
        eos_thresholds=[2.0],
        append_hidden=False,
    )

    torch.testing.assert_close(
        actual.latent_patch,
        expected.latent_patch,
        rtol=3e-4,
        atol=3e-4,
    )
    torch.testing.assert_close(
        actual.feedback_embedding,
        expected.feedback_embedding,
        rtol=3e-4,
        atol=3e-4,
    )


def test_reused_slot_matches_fresh_tail_after_longer_request_history() -> None:
    torch.manual_seed(2718)
    model = _TailModel().eval()
    reused = _build_tail(model, slots=1)
    fresh = copy.deepcopy(reused)
    unit = reused.spec.unit_len
    grid = torch.linspace(0.0, 1.0, NFE + 1)

    old_slot = reused.acquire_slot()
    old_condition = torch.randn(1, FM_HIDDEN)
    old_mods = reused.dit.build_mods(
        grid[:-1],
        duration=grid[1:] - grid[:-1],
        g_cond=old_condition,
    )
    reused.initialize_slot_rng(old_slot, 17)
    reused.seed_fm_history(
        old_slot,
        fm_rows=torch.randn(5 * unit, FM_HIDDEN),
        all_mods=old_mods,
    )
    reused.encode_prompt_patches(old_slot, torch.randn(1, 8, LATENT_DIM))
    old_patch = reused.sample_patches(
        [old_slot],
        fm_hidden_rows=torch.randn(1, FM_HIDDEN),
        latent_proj=model.latent_proj,
    )
    reused.encode_feedback([old_slot], old_patch)
    reused.release_slot(old_slot)

    reused_slot = reused.acquire_slot()
    fresh_slot = fresh.acquire_slot()
    new_condition = torch.randn(1, FM_HIDDEN)
    new_mods = reused.dit.build_mods(
        grid[:-1],
        duration=grid[1:] - grid[:-1],
        g_cond=new_condition,
    )
    new_prompt_rows = torch.randn(2 * unit, FM_HIDDEN)
    new_prompt_latents = torch.randn(1, 4, LATENT_DIM)
    new_hidden = torch.randn(1, FM_HIDDEN)
    for acoustic_tail, slot in ((reused, reused_slot), (fresh, fresh_slot)):
        acoustic_tail.initialize_slot_rng(slot, 23)
        acoustic_tail.seed_fm_history(
            slot,
            fm_rows=new_prompt_rows,
            all_mods=new_mods,
        )

    reused_prompt = reused.encode_prompt_patches(
        reused_slot,
        new_prompt_latents,
    )
    fresh_prompt = fresh.encode_prompt_patches(
        fresh_slot,
        new_prompt_latents,
    )
    reused_patch = reused.sample_patches(
        [reused_slot],
        fm_hidden_rows=new_hidden,
        latent_proj=model.latent_proj,
    )
    fresh_patch = fresh.sample_patches(
        [fresh_slot],
        fm_hidden_rows=new_hidden,
        latent_proj=model.latent_proj,
    )
    reused_feedback = reused.encode_feedback([reused_slot], reused_patch)
    fresh_feedback = fresh.encode_feedback([fresh_slot], fresh_patch)

    torch.testing.assert_close(reused_prompt, fresh_prompt, rtol=0, atol=0)
    torch.testing.assert_close(reused_patch, fresh_patch, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(
        reused_feedback,
        fresh_feedback,
        rtol=3e-4,
        atol=3e-4,
    )


def test_tail_slots_are_bounded_and_reusable() -> None:
    acoustic_tail = _build_tail(_TailModel().eval(), slots=2)
    first = acoustic_tail.acquire_slot()
    acoustic_tail.acquire_slot()
    try:
        acoustic_tail.acquire_slot()
    except RuntimeError as error:
        assert "ran out of slots" in str(error)
    else:
        raise AssertionError("slot exhaustion must fail")
    acoustic_tail.release_slot(first)
    assert acoustic_tail.acquire_slot() == first


def test_request_release_forgets_slot_before_it_can_be_reused() -> None:
    from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead

    released = []
    flow = SimpleNamespace(_tail=SimpleNamespace(release_slot=released.append))
    state = SimpleNamespace(slot=3)

    DotsTTSFlowHead.release_request(flow, state)
    DotsTTSFlowHead.release_request(flow, state)

    assert state.slot is None
    assert released == [3]


def test_fused_dit_builds_modulations_with_bfloat16_weights() -> None:
    model = _TailModel().eval().to(torch.bfloat16)
    dit = tail.fuse_dit_for_inference(model)
    steps = torch.tensor([0.0, 0.5], dtype=torch.bfloat16)

    mods = dit.build_mods(steps, duration=torch.full_like(steps, 0.5))

    assert mods.dtype == torch.bfloat16
