"""Review-only checks for Qwen prompt-row ownership and video decode positions."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.runtime_context import get_context
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.stage_io import (
    read_stream_chunk,
    serialize_inline_stream_chunk,
    write_stream_chunk,
)
from sglang_omni.relay.shm import ShmRelay
from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.thinker_session_runner import (
    QwenSessionThinkerModelRunner,
)
from sglang_omni.models.qwen3_omni.request_builders import compute_mrope_positions
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend.prompt_hidden import (
    PromptHiddenRequest,
    PromptHiddenStore,
)
from tests.unit_test.qwen3_omni.test_mrope_positions import (
    audio_in_video_span,
    oracle_and_fast,
    thinker_config_ns,
)


@contextmanager
def small_store(*, hidden_size: int = 8):
    request_to_token = torch.zeros((2, 16), dtype=torch.int32, device="cuda")
    store = PromptHiddenStore(
        slot_count=64,
        hidden_size=hidden_size,
        dtype=torch.bfloat16,
        req_to_token_pool=SimpleNamespace(req_to_token=request_to_token),
        device="cuda",
    )
    try:
        yield store, request_to_token
    finally:
        store.destroy()


@pytest.mark.asyncio
async def test_real_width_bf16_prompt_rows_cross_the_shm_stream_path() -> None:
    """Media prompt rows exceed the inline limit and must retain bytes and metadata over SHM."""
    with small_store(hidden_size=4096) as (store, request_to_token):
        rows = torch.arange(3 * 4096, device="cuda", dtype=torch.bfloat16).reshape(
            3, 4096
        )
        store.stage(torch.tensor([10, 20, 30], device="cuda"), rows)
        request_to_token[0, :3] = torch.tensor(
            [10, 20, 30], device="cuda", dtype=torch.int32
        )
        host_rows = store.read(0, torch.arange(3))
    metadata = {"token_id": 20, "prompt_hidden_positions": [1, 4, 9]}
    assert serialize_inline_stream_chunk(host_rows, metadata) is None
    relay = ShmRelay(engine_id="review2861")
    try:
        data_reference, operations = await write_stream_chunk(
            relay,
            request_id="review2861",
            data=host_rows,
            target_stage="talker_ar",
            from_stage="thinker",
            chunk_id=0,
            metadata=metadata,
            transport=TransportKind.SHM,
        )
        received_rows, received_metadata = await read_stream_chunk(
            relay, data_reference
        )
        for operation in operations:
            operation.mark_receiver_done()
            await operation.wait_for_completion()
        torch.testing.assert_close(received_rows, host_rows, rtol=0, atol=0)
        assert received_metadata == metadata
    finally:
        relay.close()


def prompt_request(positions: list[int]) -> PromptHiddenRequest:
    request = object.__new__(PromptHiddenRequest)
    request.prompt_hidden_positions = torch.tensor(positions, dtype=torch.int64)
    return request


def test_translated_writes_and_cached_prefix_keep_the_virtual_slot_identity() -> None:
    """Physical IDs must not index the host store that cached prefixes read virtually."""
    runner = object.__new__(ThinkerModelRunner)
    hidden = torch.arange(32, device="cuda", dtype=torch.bfloat16).reshape(4, 8)
    with small_store() as (store, request_to_token):
        runner.prompt_hidden_store = store
        request_to_token[0, 1] = 30
        request_to_token[0, 4] = 40
        store.stage(torch.tensor([30], device="cuda"), hidden[:1])
        schedule_batch = SimpleNamespace(
            reqs=[SimpleNamespace(), prompt_request([1, 4])],
            prefix_lens=[0, 4],
            extend_lens=[2, 2],
        )
        runner.post_prefill(
            SimpleNamespace(logits_output=SimpleNamespace(hidden_states=hidden)),
            SimpleNamespace(
                capture_hidden_mode=CaptureHiddenMode.FULL,
                out_cache_loc=torch.tensor([1, 2, 3, 4], device="cuda"),
                out_cache_loc_virtual=torch.tensor([20, 21, 40, 41], device="cuda"),
            ),
            schedule_batch,
            [],
        )
        torch.testing.assert_close(
            store.read(0, torch.tensor([1, 4])),
            hidden[[0, 2]].cpu(),
            rtol=0,
            atol=0,
        )


def test_graph_replays_cannot_overwrite_an_earlier_staged_snapshot() -> None:
    """The next replay reuses the graph output; every staged row keeps its own forward."""
    source = torch.ones((3, 8), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output.copy_(source * 2)
    with small_store() as (store, request_to_token):
        graph.replay()
        store.stage(torch.tensor([10, 11, 12], device="cuda"), output)
        source.fill_(7)
        graph.replay()
        store.stage(torch.tensor([20, 21, 22], device="cuda"), output)
        source.fill_(9)
        graph.replay()
        request_to_token[0, :3] = torch.tensor(
            [10, 11, 12], device="cuda", dtype=torch.int32
        )
        request_to_token[1, :3] = torch.tensor(
            [20, 21, 22], device="cuda", dtype=torch.int32
        )
        torch.testing.assert_close(
            store.read(0, torch.arange(3)),
            torch.full((3, 8), 2, dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            store.read(1, torch.arange(3)),
            torch.full((3, 8), 14, dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("grid", [[4, 4, 4], [8, 16, 16]])
def test_fractional_audio_in_video_decode_uses_the_served_prompt_floor(
    grid: list[int],
) -> None:
    """Flooring must retain the float-timed interleave and reach the real decode consumer."""
    audio_length = 300
    tokens = [7, 8] + audio_in_video_span(grid, audio_length) + [9, 10]
    video_grid = torch.tensor([grid])
    seconds = torch.tensor([1.1611])
    audio_lengths = torch.tensor([audio_length])
    (oracle_positions, _), _ = oracle_and_fast(
        torch.tensor([tokens]),
        video_grid_thw=video_grid,
        second_per_grid_ts=seconds,
        audio_seqlens=audio_lengths,
        use_audio_in_video=True,
    )
    positions, delta = compute_mrope_positions(
        torch.tensor(tokens),
        {
            "video_grid_thw": video_grid,
            "video_second_per_grid": seconds,
            "audio_feature_lengths": audio_lengths,
            "use_audio_in_video": True,
        },
        thinker_config_ns(),
    )
    torch.testing.assert_close(
        positions, oracle_positions.squeeze(1).floor(), rtol=0, atol=0
    )
    multimodal_inputs = MultimodalInputs(mm_items=[])
    multimodal_inputs.mrope_positions = positions
    multimodal_inputs.mrope_position_delta = delta
    first_decode = ForwardBatch._expand_mrope_from_input(
        None, multimodal_inputs, len(tokens) + 1
    ).to(torch.int64)
    assert torch.equal(
        first_decode, torch.full((3, 1), int(positions.max()) + 1, dtype=torch.int64)
    )
    decoded_batch = SimpleNamespace(
        seq_lens_cpu=torch.tensor([len(tokens) + 1]),
        seq_lens=torch.tensor([len(tokens) + 1], device="cuda"),
    )
    with get_context().override_server_args():
        ForwardBatch._compute_mrope_positions_decode(
            decoded_batch,
            SimpleNamespace(device="cuda"),
            SimpleNamespace(multimodal_inputs=[multimodal_inputs]),
        )
    torch.testing.assert_close(
        decoded_batch.mrope_positions.cpu(), first_decode, rtol=0, atol=0
    )


def cuda_prompt_builder(directory: Path) -> TalkerPrefillBuilder:
    embedding = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    save_file(
        {"thinker.model.embed_tokens.weight": embedding},
        str(directory / "model.safetensors"),
    )
    torch.manual_seed(0)
    codec = torch.nn.Embedding(32, 8, device="cuda", dtype=torch.bfloat16)
    model = SimpleNamespace(
        model=SimpleNamespace(codec_embedding=codec),
        activation_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        text_projection=torch.nn.Linear(8, 8, device="cuda", dtype=torch.bfloat16),
        hidden_projection=torch.nn.Linear(8, 8, device="cuda", dtype=torch.bfloat16),
        get_input_embeddings=lambda: codec,
    )
    return TalkerPrefillBuilder(
        model=model,
        model_path=str(directory),
        audio_token_id=3,
        image_token_id=4,
        video_token_id=5,
        tts_bos_token_id=10,
        tts_eos_token_id=11,
        tts_pad_token_id=12,
        im_start_token_id=1,
        im_end_token_id=9,
        system_token_id=19,
        user_token_id=2,
        assistant_token_id=6,
        codec_bos_id=13,
        codec_nothink_id=14,
        codec_think_bos_id=15,
        codec_think_eos_id=16,
        codec_pad_id=17,
    )


def test_cuda_prompt_projection_matches_the_installed_hf_user_path(
    tmp_path: Path,
) -> None:
    builder = cuda_prompt_builder(tmp_path)
    token_ids = torch.tensor([1, 2, 3, 4, 5, 8, 9, 1, 6, 7])
    hidden_rows = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8) / 8
    payload = StagePayload(
        request_id="projection",
        request=OmniRequest(inputs=[], params={}),
        data={"prompt": {"input_ids": token_ids}},
    )
    captured = StreamItem(
        chunk_id=0,
        from_stage="thinker",
        data=hidden_rows,
        metadata={"token_id": 20, "prompt_hidden_positions": [2, 3, 4]},
    )
    plain = StreamItem(
        chunk_id=0,
        from_stage="thinker",
        data=torch.tensor([20]),
        metadata={"token_id": 20},
    )
    with torch.inference_mode():
        actual = builder.build_prompt_prefill(payload, [captured], thinker_done=False)
        embeddings = builder.load_prompt_token_embeddings(token_ids)
        hidden = torch.zeros_like(embeddings)
        hidden[2:5] = hidden_rows.to(hidden)
        mask = builder.build_multimodal_mask(token_ids)
        reference_context = SimpleNamespace(
            config=SimpleNamespace(
                talker_config=SimpleNamespace(
                    text_config=SimpleNamespace(hidden_size=8)
                )
            ),
            talker=builder.model,
        )
        reference = Qwen3OmniMoeForConditionalGeneration._get_talker_user_parts(
            reference_context,
            0,
            7,
            mask.unsqueeze(0),
            hidden.unsqueeze(0),
            embeddings.unsqueeze(0),
        )
        torch.testing.assert_close(
            actual["input_embeds"][:7], reference[0], rtol=0, atol=0
        )
        profile_counts = {}
        for label, chunk in (
            ("without_prompt_rows", plain),
            ("with_prompt_rows", captured),
        ):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                builder.build_prompt_prefill(payload, [chunk], thinker_done=False)
                torch.cuda.synchronize()
            profile_counts[label] = {
                entry.key: entry.count
                for entry in profile.key_averages()
                if entry.key
                in ("aten::nonzero", "cudaStreamSynchronize", "cudaDeviceSynchronize")
            }
        print("ALIGNMENT_PROFILE", json.dumps(profile_counts, sort_keys=True))


def test_failed_scheduler_construction_releases_registered_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-allocation tokenizer failure must not leave an unowned registered buffer."""
    from sglang.srt.arg_groups import model_override_base
    from sglang.srt.utils import hf_transformers_utils
    from sglang_omni.model_runner import thinker_model_runner
    from sglang_omni.models.qwen3_omni import bootstrap
    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend

    arguments = SimpleNamespace(
        disable_cuda_graph=True,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=CudaGraphBackend.DISABLED)
        ),
    )
    model_config = SimpleNamespace(
        model_path="dummy",
        vocab_size=32,
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(
            thinker_config=SimpleNamespace(
                image_token_id=4,
                video_token_id=5,
                audio_token_id=3,
                text_config=SimpleNamespace(hidden_size=8),
            )
        ),
    )
    thinker = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(32, 8))
    )
    worker = SimpleNamespace(
        gpu_id=0,
        model_runner=SimpleNamespace(
            model=SimpleNamespace(
                thinker=thinker, set_talker_layers_to_capture=lambda: None
            ),
            model_config=model_config,
            device="cuda",
            max_token_pool_size=64,
            kv_index_translator=SimpleNamespace(
                capture_token_capacity=lambda capacity: capacity
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((2, 16), dtype=torch.int32, device="cuda")
            ),
        ),
    )
    monkeypatch.setattr(model_override_base, "resolved_view", lambda args: args)
    monkeypatch.setattr(
        scheduling_bootstrap,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (worker, None, None, None, model_config),
    )
    allocated_stores: list[PromptHiddenStore] = []

    def record_store(**options) -> PromptHiddenStore:
        store = PromptHiddenStore(**options)
        allocated_stores.append(store)
        return store

    monkeypatch.setattr(thinker_model_runner, "PromptHiddenStore", record_store)
    try:

        def fail_tokenizer(*args, **kwargs):
            raise OSError("review tokenizer setup failure")

        monkeypatch.setattr(hf_transformers_utils, "get_tokenizer", fail_tokenizer)
        with pytest.raises(OSError, match="review tokenizer setup failure"):
            bootstrap.create_thinker_scheduler(arguments, speech_enabled=True)
        assert len(allocated_stores) == 1
        assert allocated_stores[0].hidden_states is None, (
            "scheduler construction left the store registered without a shutdown owner"
        )
    finally:
        for store in allocated_stores:
            store.destroy()
