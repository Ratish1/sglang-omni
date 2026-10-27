"""Whether two trees load the same speech tokenizer: `dump` (run with PYTHONPATH set to the tree) saves every encoder
and decoder parameter and buffer, persistent or not, as the tree's loader leaves them for a bf16 server; `compare A B
--part encoder|decoder` reports the names whose dtype or bits differ."""

import argparse
import sys

import torch

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def dump(out: str) -> None:
    from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

    tokenizer = load_qwen3_tts_tokenizer(MODEL, device="cuda:0", dtype="bfloat16", attn_implementation=None)
    tensors = {}
    for part in ("encoder", "decoder"):
        module = getattr(tokenizer.model, part)
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            tensors[f"{part}.{name}"] = tensor.detach().cpu()
        cached = [name for name, child in module.named_modules() if getattr(child, "_embed", None) is not None]
        print(f"{part}: {sum(1 for key in tensors if key.startswith(part + '.'))} tensors, cached codebooks {len(cached)}")
    torch.save(tensors, out)
    print(f"dtypes {sorted({str(t.dtype) for t in tensors.values()})} saved {out}")


def compare(first: str, second: str, part: str) -> int:
    a = {k: v for k, v in torch.load(first).items() if k.startswith(part + ".")}
    b = {k: v for k, v in torch.load(second).items() if k.startswith(part + ".")}
    only = sorted(set(a) ^ set(b))
    differ = [k for k in sorted(set(a) & set(b)) if a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k])]
    print(f"{part}: {len(a)} vs {len(b)} tensors, {len(only)} unmatched names, {len(differ)} differ")
    for key in only[:10]:
        print(f"  only in one: {key}")
    for key in differ[:20]:
        print(f"  differs: {key} {a[key].dtype} vs {b[key].dtype}")
    return 1 if only or differ else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    dump_parser = sub.add_parser("dump")
    dump_parser.add_argument("out")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("first")
    compare_parser.add_argument("second")
    compare_parser.add_argument("--part", choices=("encoder", "decoder"), required=True)
    args = parser.parse_args()
    if args.command == "dump":
        dump(args.out)
        return 0
    else:
        return compare(args.first, args.second, args.part)


if __name__ == "__main__":
    sys.exit(main())
