"""Compare the stacked tables and ICL prompts to the actual pre-PR implementation."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess

import torch
from torch import nn

from sglang_omni.models.qwen3_tts.prompt_frontend import (
    iter_checkpoint_tensors,
    load_qwen3_tts_prompt_frontend,
)
from sglang_omni.models.qwen3_tts.stages import register_qwen3_tts_hf_config
from sglang_omni.utils.checkpoint import resolve_checkpoint

BASE_REVISION = "76de3ff8f2ee98cb96c989e7d89527dc3d7d263f"


def original_icl_builder():
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{BASE_REVISION}:sglang_omni/models/qwen3_tts/sglang_model.py",
        ],
        text=True,
    )
    tree = ast.parse(source)
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3TTSPromptBuilderMixin"
    )
    method = next(
        node
        for node in builder.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_icl_prompt"
    )
    namespace = {"torch": torch}
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(method),
            "pre_pr_icl_builder",
            "exec",
        ),
        namespace,
    )
    return namespace["generate_icl_prompt"]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    options = parser.parse_args()
    register_qwen3_tts_hf_config()
    checkpoint = resolve_checkpoint(options.model)
    frontend = load_qwen3_tts_prompt_frontend(
        checkpoint, device="cuda", dtype=torch.bfloat16
    )
    tables = frontend.code_predictor.model.codec_embedding
    count, vocabulary, width = tables.weight.shape
    assert count == frontend.config.num_code_groups - 1
    checkpoint_tables = {
        f"talker.code_predictor.model.codec_embedding.{index}.weight"
        for index in range(count)
    }
    for name, tensor in iter_checkpoint_tensors(checkpoint, checkpoint_tables):
        codebook = int(name.split(".")[-2])
        assert torch.equal(tables.weight[codebook], tensor.to("cuda", torch.bfloat16))
    assert tables.row_offsets.device.type == "cuda"
    assert tables.row_offsets.dtype == torch.long
    generator = torch.Generator(device="cpu").manual_seed(2916)
    codes = torch.randint(vocabulary, (37, count), generator=generator).cuda()
    codes[0] = 0
    codes[1] = vocabulary - 1
    expected = torch.stack(
        [
            torch.nn.functional.embedding(codes[:, index], tables.weight[index])
            for index in range(count)
        ],
        dim=1,
    )
    assert torch.equal(tables(codes), expected)
    # note (ratish): IPC rebinds Parameter.data before capture; future lookups
    # must read that allocation rather than a slice saved during construction.
    original_weight = tables.weight.detach()
    tables.weight.data = tables.weight.detach().clone() + 0.25
    expected = torch.stack(
        [
            torch.nn.functional.embedding(codes[:, index], tables.weight[index])
            for index in range(count)
        ],
        dim=1,
    )
    assert torch.equal(tables(codes), expected)
    tables.weight.data = original_weight
    old_tables = nn.ModuleList()
    for table in tables.weight:
        embedding = nn.Embedding(vocabulary, width, device="meta", dtype=table.dtype)
        embedding.weight = nn.Parameter(table, requires_grad=False)
        old_tables.append(embedding)
    old_builder = original_icl_builder()
    constants = frontend.build_prompt_constants()
    outcomes: list[dict[str, bool | int]] = []
    for frame_count, text_count, non_streaming in (
        (3, 29, False),
        (37, 3, False),
        (37, 29, True),
    ):
        reference_codes = torch.randint(
            vocabulary, (frame_count, count + 1), generator=generator
        ).cuda()
        reference_codes[0] = 0
        reference_codes[-1] = vocabulary - 1
        text_ids = torch.arange(10, 10 + text_count, device="cuda")[None]
        reference_ids = torch.arange(80, 87, device="cuda")[None]
        actual, actual_tail = frontend.generate_icl_prompt(
            text_ids,
            reference_ids,
            reference_codes,
            constants,
            non_streaming,
        )
        frontend.code_predictor.model.codec_embedding = old_tables
        try:
            expected, expected_tail = old_builder(
                frontend,
                text_ids,
                reference_ids,
                reference_codes,
                constants,
                non_streaming,
            )
        finally:
            frontend.code_predictor.model.codec_embedding = tables
        assert torch.equal(actual, expected)
        assert torch.equal(actual_tail, expected_tail)
        outcomes.append(
            {
                "frames": frame_count,
                "text_tokens": text_count,
                "non_streaming": non_streaming,
                "bit_identical": True,
            }
        )
    print(
        json.dumps(
            {
                "checkpoint_tables": count,
                "vocabulary": vocabulary,
                "width": width,
                "prompt_comparisons": outcomes,
                "storage_rebind": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
