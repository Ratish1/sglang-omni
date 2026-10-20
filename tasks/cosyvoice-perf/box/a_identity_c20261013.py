# A's prompt construction against main's on the real checkpoint's embedding tables (bf16, as served). Every long
# form text with 8 seeded references, in the ref text, instruct and cross-lingual modes, goes through
# prepare_cosyvoice3_request twice: on CUDA (A's branch, under sync debug mode "error", on a side stream with an event)
# and on the CPU (main's construction, kept byte for byte off CUDA). Checked: the prompt rows bit identical, the
# prompt lengths equal, and for every pair of prompts the old (blake2b) and new (host id) radix keys giving the same
# longest common prefix.
# usage: python3 a_identity_c20261013.py <longform meta.lst>
import itertools
import random
import sys
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download

from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3ReferenceArtifact,
    prepare_cosyvoice3_request,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import VOCAB_SIZE
from sglang_omni.models.fun_cosyvoice3.utils import CosyVoice3Tokenizer


class Tables(torch.nn.Module):
    def __init__(self, text_weight: torch.Tensor, speech_weight: torch.Tensor) -> None:
        super().__init__()
        self.text_embed_tokens = torch.nn.Embedding.from_pretrained(text_weight, freeze=True)
        self.speech_embedding = torch.nn.Embedding.from_pretrained(speech_weight, freeze=True)
        self.config = SimpleNamespace(hidden_size=int(text_weight.shape[1]))


def common_prefix(a: list[int], b: list[int]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def main() -> None:
    root = snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    weights = torch.load(f"{root}/llm.pt", map_location="cpu", weights_only=True)
    text_weight = weights["llm.model.model.embed_tokens.weight"].to(torch.bfloat16)
    speech_weight = weights["speech_embedding.weight"].to(torch.bfloat16)
    cpu_model = Tables(text_weight, speech_weight)
    cuda_model = Tables(text_weight, speech_weight).cuda()
    tokenizer = CosyVoice3Tokenizer(f"{root}/CosyVoice-BlankEN")

    lines = [line.rstrip("\n").split("|") for line in open(sys.argv[1]) if line.strip()]
    generator = random.Random(0)
    references = [
        (lines[index % len(lines)][1], [generator.randrange(VOCAB_SIZE) for _ in range(generator.randrange(0, 300))])
        for index in range(8)
    ]
    cases = []
    for (_, _, _, target), (reference, (prompt_text, speech_ids)) in itertools.product(
        lines, enumerate(references)
    ):
        for mode in ("ref_text", "instruct", "cross_lingual"):
            cases.append((target, reference, prompt_text, speech_ids, mode))

    side_stream = torch.cuda.Stream()
    old_keys, new_keys, identical, lengths_equal, total = [], [], 0, 0, 0
    for target, reference, prompt_text, speech_ids, mode in cases:
        state = FunCosyVoice3State(
            text=target,
            ref_audio=f"ref_{reference}.wav",
            ref_text=prompt_text if mode == "ref_text" else None,
            instructions="speak warmly" if mode == "instruct" else None,
        )
        speech = torch.tensor([speech_ids], dtype=torch.int32).reshape(1, -1)
        artifact = CosyVoice3ReferenceArtifact(
            llm_prompt_speech_token=speech,
            flow_prompt_speech_token=speech,
            flow_prompt_speech_feat=torch.zeros(1, 2 * speech.shape[1], 80),
            flow_embedding=torch.zeros(1, 192),
        )
        old = prepare_cosyvoice3_request(model=cpu_model, tokenizer=tokenizer, state=state, reference_artifact=artifact)
        torch.cuda.set_sync_debug_mode("error")
        try:
            with torch.cuda.stream(side_stream):
                new = prepare_cosyvoice3_request(
                    model=cuda_model, tokenizer=tokenizer, state=state, reference_artifact=artifact
                )
                ready = torch.cuda.Event()
                ready.record(side_stream)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        ready.synchronize()
        total += 1
        identical += int(torch.equal(new.prompt_input_embeds.cpu(), old.prompt_input_embeds))
        lengths_equal += int(len(new.input_ids_list) == len(old.input_ids_list) == old.prompt_input_embeds.shape[0])
        old_keys.append(old.input_ids_list)
        new_keys.append(new.input_ids_list)

    prefix_mismatches = 0
    pairs = 0
    for i, j in itertools.combinations(range(total), 2):
        pairs += 1
        if common_prefix(old_keys[i], old_keys[j]) != common_prefix(new_keys[i], new_keys[j]):
            prefix_mismatches += 1
    print(f"prompts {total}, rows identical {identical}, lengths equal {lengths_equal}")
    print(f"pairs {pairs}, longest common prefix differs in {prefix_mismatches}")
    print("sync debug: no host wait in A's branch" if total else "no case ran")


if __name__ == "__main__":
    main()
