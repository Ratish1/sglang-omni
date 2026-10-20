# Run FunCosyVoice3ModelRunner.redraw_repeated_tokens of the tree on PYTHONPATH on fixed inputs and
# save the ids, so two trees can be compared bit for bit: pytorch sampler with bf16 and fp32 probs,
# the Ascend branch with tempered fp32 logits, each seeded and unseeded (torch.manual_seed before
# the call). Rows mix repeats, new tokens, greedy rows and control ids.
# usage: python3 ras_fold_identity_c20261012.py <out.pt>   |   python3 ... --compare <a.pt> <b.pt>
import sys
from array import array
from types import SimpleNamespace

import torch

ROWS = 64


def inputs(dtype: torch.dtype, use_ascend_backend: bool, seeded: bool):
    from sglang_omni.models.fun_cosyvoice3.sglang_model import EOS_ID, VOCAB_SIZE

    gen = torch.Generator().manual_seed(7)
    logits = torch.randn((ROWS, EOS_ID + 1), generator=gen) * 4
    probs = torch.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, 1, generator=gen).view(-1)
    sampled[::9] = EOS_ID
    histories, top_ks = [], []
    for row in range(ROWS):
        token = int(sampled[row])
        history = torch.randint(0, VOCAB_SIZE, (12,), generator=gen).tolist()
        if row % 3 != 2:
            history[-1 - row % 10] = token
        histories.append(history)
        top_ks.append(1 if row % 7 == 0 else 20)
    device_logits = (logits / 0.7).cuda() if use_ascend_backend else probs.to(dtype).cuda()
    seeds = torch.randint(0, 2**31 - 1, (ROWS,), generator=gen).cuda() if seeded else None
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    output_ids=array("q", histories[row]),
                    sampling_params=SimpleNamespace(top_k=top_ks[row]),
                )
            )
        )
        for row in range(ROWS)
    ]
    result = SimpleNamespace(
        next_token_ids=sampled.to(torch.int32).cuda(),
        logits_output=SimpleNamespace(next_token_logits=device_logits),
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        sampling_info=SimpleNamespace(sampling_seed=seeds),
        positions=torch.randint(100, 2000, (ROWS,), generator=gen).cuda(),
    )
    return result, forward_batch, requests, sampled.tolist()


def run(out: str) -> None:
    from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner

    saved = {}
    for name, dtype, ascend in [
        ("bf16", torch.bfloat16, False),
        ("fp32", torch.float32, False),
        ("ascend", torch.float32, True),
    ]:
        for seeded in (False, True):
            runner = object.__new__(FunCosyVoice3ModelRunner)
            runner.token_id_host_bufs = None
            runner.token_id_host_slot = 0
            runner.tp_worker = SimpleNamespace(
                model_runner=SimpleNamespace(
                    sampler=SimpleNamespace(use_ascend_backend=ascend)
                )
            )
            result, forward_batch, requests, host_ids = inputs(dtype, ascend, seeded)
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            returned = runner.redraw_repeated_tokens(
                result, forward_batch, requests, host_ids
            )
            torch.cuda.synchronize()
            key = f"{name}_{'seeded' if seeded else 'unseeded'}"
            saved[key] = {
                "device": result.next_token_ids.cpu(),
                "host": torch.tensor(returned),
                "changed": int((torch.tensor(returned) != torch.tensor(host_ids)).sum()),
            }
            print(key, "rows changed", saved[key]["changed"])
    torch.save(saved, out)


def compare(a: str, b: str) -> None:
    left, right = torch.load(a), torch.load(b)
    assert left.keys() == right.keys()
    for key in left:
        same = torch.equal(left[key]["device"], right[key]["device"]) and torch.equal(
            left[key]["host"], right[key]["host"]
        )
        print(f"{key}: {'IDENTICAL' if same else 'DIFFERENT'}, rows changed {left[key]['changed']} / {right[key]['changed']}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1])
