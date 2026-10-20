# Run FunCosyVoice3SGLangModel.forward's speech head of the tree on PYTHONPATH on fixed bf16 inputs of the served
# shapes (hidden 896, 6,761 speech ids) at 1, 16 and 64 decode rows and save the logits, so the heads of two trees
# can be compared bit for bit.
# usage: python3 head_identity_c20261012.py <out.pt>   |   python3 ... --compare <a.pt> <b.pt> [<c.pt> ...]
import sys
from types import SimpleNamespace

import torch

HIDDEN = 896


def run(out: str) -> None:
    from sglang_omni.models.fun_cosyvoice3.sglang_model import (
        TOTAL_VOCAB_SIZE,
        FunCosyVoice3SGLangModel,
    )

    generator = torch.Generator().manual_seed(0)
    weight = (torch.randn(TOTAL_VOCAB_SIZE, HIDDEN, generator=generator) * 0.05).to(
        "cuda", torch.bfloat16
    )
    saved = {}
    for rows in (1, 16, 64):
        hidden_states = (torch.randn(rows, HIDDEN, generator=generator) * 4).to(
            "cuda", torch.bfloat16
        )
        model = object.__new__(FunCosyVoice3SGLangModel)
        torch.nn.Module.__init__(model)
        model.speech_embedding = torch.nn.Embedding(TOTAL_VOCAB_SIZE, HIDDEN).to(
            "cuda", torch.bfloat16
        )
        model.llm_decoder = torch.nn.Linear(HIDDEN, TOTAL_VOCAB_SIZE, bias=False).to(
            "cuda", torch.bfloat16
        )
        with torch.no_grad():
            model.llm_decoder.weight.copy_(weight)
        model.model = lambda *args, **kwargs: hidden_states
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True, is_extend=lambda: False)
        )
        token_ids = torch.zeros(rows, dtype=torch.long, device="cuda")
        logits = model(token_ids, token_ids, forward_batch).next_token_logits
        saved[rows] = logits.cpu()
        print(rows, logits.dtype, tuple(logits.shape))
    torch.save(saved, out)


def compare(paths: list[str]) -> None:
    first = torch.load(paths[0])
    for path in paths[1:]:
        other = torch.load(path)
        for rows in first:
            same = first[rows].dtype == other[rows].dtype and torch.equal(first[rows], other[rows])
            print(f"{paths[0]} vs {path} rows {rows}: {'IDENTICAL' if same else 'DIFFERENT'}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2:])
    else:
        run(sys.argv[1])
