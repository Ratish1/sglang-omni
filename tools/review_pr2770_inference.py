"""Exercise the TTS sampler using its construction-time semantic seed buffer."""

from types import SimpleNamespace

import torch
from sglang.srt.layers.sampler import (
    multinomial_with_seed,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.utils import DynamicGradMode
from torch._dynamo.utils import counters

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker


class CodecHead(torch.nn.Linear):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return super().forward(hidden_states), None


def check_semantic_seed_buffer(inference_mode: bool, warm_normal_seed: bool) -> None:
    DynamicGradMode.set_inference_mode(inference_mode)
    talker = object.__new__(Qwen3TTSTalker)
    torch.nn.Module.__init__(talker)
    talker.config = SimpleNamespace(vocab_size=8)
    talker.codec_head = CodecHead(16, 8, bias=False).to("cuda", torch.bfloat16)
    talker.model = (
        lambda input_ids, positions, forward_batch, input_embeds: input_embeds
    )
    talker.semantic_sampling_seed_tensor = torch.arange(64, device="cuda")
    runner = SimpleNamespace(model=talker)
    torch.compiler.reset()
    compiled_before_warmup = counters["stats"]["unique_graphs"]
    talker.precompile_kernels_after_loading()
    if warm_normal_seed:
        for rows in (1, 2):
            seeds = torch.zeros(rows, device="cuda", dtype=torch.long)
            with DynamicGradMode():
                multinomial_with_seed(
                    torch.zeros(rows, 8, device="cuda", dtype=torch.float64),
                    seeds,
                    torch.zeros(rows, device="cuda", dtype=torch.long),
                )
    else:
        pass
    compiled_before_requests = counters["stats"]["unique_graphs"]
    with DynamicGradMode():
        for rows in (1, 3, 16):
            forward_batch = SimpleNamespace(
                mrope_positions=None,
                forward_mode=SimpleNamespace(is_extend=lambda: True),
                extend_seq_lens=torch.full((rows,), 2, device="cuda"),
                sampling_info=SimpleNamespace(sampling_seed=None),
            )
            Qwen3TTSModelRunner.install_semantic_sampling_seeds(
                runner, forward_batch, [None] * rows
            )
            logits = talker(
                None,
                None,
                forward_batch,
                input_embeds=torch.ones(
                    2 * rows, 16, device="cuda", dtype=torch.bfloat16
                ),
            ).next_token_logits
            apply_scaling_penalties(
                logits, torch.full(logits.shape, 1.05, device="cuda")
            )
            top_k_top_p_min_p_sampling_from_probs_torch(
                torch.softmax(logits, dim=-1),
                torch.full((rows,), 50, device="cuda"),
                torch.ones(rows, device="cuda"),
                torch.zeros(rows, device="cuda"),
                False,
                forward_batch.sampling_info.sampling_seed,
                torch.full((rows,), 7, device="cuda"),
            )
    print(
        f"inference_mode={inference_mode} "
        f"warm_normal_seed={warm_normal_seed} "
        f"seed_buffer_is_inference={talker.semantic_sampling_seed_tensor.is_inference()} "
        f"warmup_compiles={compiled_before_requests - compiled_before_warmup} "
        f"request_compiles={counters['stats']['unique_graphs'] - compiled_before_requests}",
        flush=True,
    )
    DynamicGradMode.set_inference_mode(False)


if __name__ == "__main__":
    check_semantic_seed_buffer(False, False)
    check_semantic_seed_buffer(True, False)
    check_semantic_seed_buffer(True, True)
