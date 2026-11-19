"""Every sampled step of chosen thinker requests: a bit-exact fingerprint of the request's logits row, its top two tokens,
and the forward it ran in, in the serving process.

Enabled by OMNI_NUMERICS_PROBE=<output prefix>; OMNI_NUMERICS_TARGET_LENS=<prompt token counts, comma separated> picks
the requests (every request when unset). Wraps ModelRunner.sample_next_token_ids (prefill, mixed chunk and synchronous decode) and
ThinkerModelRunner.sample_lookahead (overlapped decode) read-only, before they apply any suppress or sampling step.
One JSON line per target request per step to <prefix>.<pid>.jsonl: rid, prompt length, tokens generated so far, forward
mode, rows in the forward, requests in the forward and their prompt lengths, the logits fingerprint (a weighted sum of
the float32 bit patterns, equal only for equal rows in practice), top-2 ids and logits.
"""

import json
import os
import sys
import threading
import time

PREFIX = os.environ.get("OMNI_NUMERICS_PROBE")
TARGET_LENS = {int(item) for item in os.environ.get("OMNI_NUMERICS_TARGET_LENS", "").split(",") if item}
BASE = "sglang_omni.model_runner.base"
THINKER = "sglang_omni.model_runner.thinker_model_runner"


def install(base, thinker) -> None:
    torch = sys.modules["torch"]
    lock = threading.Lock()
    path = f"{PREFIX}.{os.getpid()}.jsonl"
    weights: dict[tuple[int, str], object] = {}

    def fingerprint(row):
        key = (row.shape[-1], str(row.device))
        if key not in weights:
            generator = torch.Generator(device="cpu").manual_seed(1)
            weights[key] = torch.randint(1, 2**20, (row.shape[-1],), generator=generator, dtype=torch.int64).to(row.device)
        else:
            pass
        bits = row.float().contiguous().view(torch.int32).to(torch.int64)
        return int((bits * weights[key]).sum())

    def record(logits_output, forward_batch, requests, source):
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2 or logits.shape[0] != len(requests):
            return
        else:
            pass
        lengths = [len(sched.data.req.origin_input_ids) for sched in requests]
        seq_lens = getattr(forward_batch, "seq_lens_cpu", None) if forward_batch is not None else None
        if seq_lens is None and forward_batch is not None:
            seq_lens = forward_batch.seq_lens.cpu()
        else:
            pass
        lines = []
        for index, sched in enumerate(requests):
            req = sched.data.req
            if TARGET_LENS and lengths[index] not in TARGET_LENS:
                continue
            else:
                pass
            row = logits[index]
            top = torch.topk(row.float(), 2)
            position = int(seq_lens[index]) if seq_lens is not None else None
            if TARGET_LENS and position is not None and position >= lengths[index]:
                torch.save(row.float().cpu(), f"{path[:-6]}.logits.{req.rid}.{position}.pt")
            else:
                pass
            lines.append(
                json.dumps(
                    {
                        "time": time.time(),
                        "source": source,
                        "rid": req.rid,
                        "prompt_len": lengths[index],
                        "position": position,
                        "sampled": position is not None and position >= lengths[index],
                        "generated": len(req.output_ids),
                        "mode": str(getattr(forward_batch, "forward_mode", None)),
                        "forward_rows": int(forward_batch.input_ids.shape[0]) if forward_batch is not None and forward_batch.input_ids is not None else None,
                        "batch_reqs": len(requests),
                        "batch_prompt_lens": lengths,
                        "fingerprint": fingerprint(row),
                        "top_ids": top.indices.tolist(),
                        "top_logits": top.values.tolist(),
                    }
                )
            )
        if lines:
            with lock:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")
        else:
            pass

    original_sample = base.ModelRunner.sample_next_token_ids

    def sample_next_token_ids(self, logits_output, forward_batch, schedule_batch, requests):
        record(logits_output, forward_batch, requests, "sample")
        return original_sample(self, logits_output, forward_batch, schedule_batch, requests)

    base.ModelRunner.sample_next_token_ids = sample_next_token_ids

    original_lookahead = thinker.ThinkerModelRunner.sample_lookahead

    def sample_lookahead(self, logits_output, forward_batch, requests):
        record(logits_output, forward_batch, requests, "lookahead")
        return original_lookahead(self, logits_output, forward_batch, requests)

    thinker.ThinkerModelRunner.sample_lookahead = sample_lookahead

    if os.environ.get("OMNI_NUMERICS_LAYERS") == "1":
        install_layers(base, torch, lock, path)
    else:
        pass


def install_layers(base, torch, lock, path) -> None:
    """Per forward with a target request (eager forwards only): a fingerprint of the target's rows after every thinker
    layer's attention, router GEMM, experts and layer output, one JSON line per forward per target, and its last row
    at each of those points saved to <prefix>.<pid>.<rid>.<first position>.pt for its first forwards."""
    plan: dict[str, object] = {"items": None, "rows": None, "records": None, "saved": None}
    hooked: set[int] = set()
    saved_forwards: dict[str, int] = {}

    def row_fingerprint(rows):
        flat = rows.reshape(rows.shape[0], -1)
        bits = flat.contiguous().view(torch.int16 if flat.element_size() == 2 else torch.int32).to(torch.int64)
        columns = torch.arange(1, bits.shape[1] + 1, device=bits.device, dtype=torch.int64) * 2654435761 % 1000003
        per_row = (bits * columns).sum(dim=1)
        row_weights = torch.arange(1, bits.shape[0] + 1, device=bits.device, dtype=torch.int64) * 40503 % 65521 + 1
        return int((per_row * row_weights).sum())

    def hook(name):
        def capture(module, args, output):
            items = plan["items"]
            if items is None:
                return
            else:
                pass
            parts = [(name, output[0])] if isinstance(output, tuple) else [(name, output)]
            if isinstance(output, tuple) and len(output) > 1 and name.endswith(".out"):
                parts.append((name[:-4] + ".res", output[1]))
            else:
                pass
            for part_name, tensor in parts:
                if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != plan["rows"]:
                    continue
                else:
                    pass
                for item in items:
                    rows = tensor[item["start"] : item["start"] + item["length"]]
                    item["fingerprints"][part_name] = row_fingerprint(rows)
                    if item["save"]:
                        item["last_rows"][part_name] = rows[-1].detach().float().cpu()
                    else:
                        pass

        return capture

    def input_hook(name):
        def capture(module, args, output):
            items = plan["items"]
            if items is None:
                return
            else:
                pass
            inputs = [(f"{name}{index}", value) for index, value in enumerate(args) if isinstance(value, torch.Tensor)]
            outputs = [output[0] if isinstance(output, tuple) else output]
            inputs += [(name[:-3] + ".out", value) for value in outputs if isinstance(value, torch.Tensor)]
            for part_name, tensor in inputs:
                if tensor.shape[0] != plan["rows"]:
                    continue
                else:
                    pass
                for item in items:
                    rows = tensor[item["start"] : item["start"] + item["length"]]
                    item["fingerprints"][part_name] = row_fingerprint(rows)
                    if item["save"]:
                        item["last_rows"][part_name] = rows[-1].detach().float().cpu()
                    else:
                        pass

        return capture

    def has_path(module, path) -> bool:
        try:
            module.get_submodule(path)
            return True
        except AttributeError:
            return False

    def attach(model) -> None:
        for module_name, module in model.named_modules():
            if type(module).__name__ not in ("Qwen3MoeDecoderLayer", "Qwen3OmniMoeThinkerTextDecoderLayer") or id(module) in hooked:
                continue
            else:
                pass
            hooked.add(id(module))
            index = int(module_name.rsplit(".", 1)[-1])
            module.register_forward_hook(hook(f"L{index:02d}.out"))
            module.self_attn.register_forward_hook(hook(f"L{index:02d}.attn"))
            module.mlp.register_forward_hook(hook(f"L{index:02d}.moe"))
            if hasattr(module.mlp, "gate"):
                module.mlp.gate.register_forward_hook(hook(f"L{index:02d}.gate"))
            else:
                pass
            if hasattr(module.mlp, "experts"):
                module.mlp.experts.register_forward_hook(hook(f"L{index:02d}.experts"))
            else:
                pass
            if index < 2:
                for path in ("input_layernorm", "self_attn.qkv_proj", "self_attn.attn", "self_attn.o_proj", "post_attention_layernorm", "mlp.gate", "mlp.experts"):
                    sub = module.get_submodule(path) if has_path(module, path) else None
                    if sub is not None:
                        sub.register_forward_hook(input_hook(f"L{index:02d}.{path}.in"))
                    else:
                        pass
            else:
                pass

    original = base.ModelRunner.prepare_and_forward

    def prepare_and_forward(self, forward_batch, schedule_batch, requests, is_prefill, **kwargs):
        if forward_batch is None or not hooked:
            attach(self.tp_worker.model_runner.model)
        else:
            pass
        items = []
        if forward_batch is not None:
            reqs = list(schedule_batch.reqs)
            if is_prefill:
                lengths = forward_batch.extend_seq_lens_cpu
                lengths = lengths.tolist() if isinstance(lengths, torch.Tensor) else list(lengths)
                prefixes = forward_batch.extend_prefix_lens_cpu
                prefixes = prefixes.tolist() if isinstance(prefixes, torch.Tensor) else list(prefixes)
            else:
                lengths = [1] * len(reqs)
                prefixes = [len(req.origin_input_ids) + len(req.output_ids) - 1 for req in reqs]
            start = 0
            for req, length, prefix in zip(reqs, lengths, prefixes):
                if not TARGET_LENS or len(req.origin_input_ids) in TARGET_LENS:
                    count = saved_forwards.get(req.rid, 0)
                    items.append(
                        {
                            "rid": req.rid,
                            "prompt_len": len(req.origin_input_ids),
                            "start": start,
                            "length": int(length),
                            "prefix": int(prefix),
                            "save": count < 8,
                            "fingerprints": {},
                            "last_rows": {},
                        }
                    )
                    saved_forwards[req.rid] = count + 1
                else:
                    pass
                start += int(length)
            plan["rows"] = start
        else:
            pass
        plan["items"] = items or None
        try:
            return original(self, forward_batch, schedule_batch, requests, is_prefill, **kwargs)
        finally:
            plan["items"] = None
            lines = []
            for item in items:
                if not item["fingerprints"]:
                    continue
                else:
                    pass
                lines.append(
                    json.dumps(
                        {
                            "time": time.time(),
                            "layers": True,
                            "rid": item["rid"],
                            "prompt_len": item["prompt_len"],
                            "prefill": bool(is_prefill),
                            "prefix": item["prefix"],
                            "length": item["length"],
                            "forward_rows": plan["rows"],
                            "batch_reqs": len(schedule_batch.reqs),
                            "fingerprints": item["fingerprints"],
                        }
                    )
                )
                if item["save"] and item["last_rows"]:
                    torch.save(item["last_rows"], f"{path[:-6]}.{item['rid']}.{item['prefix']}.pt")
                else:
                    pass
            if lines:
                with lock:
                    with open(path, "a", encoding="utf-8") as handle:
                        handle.write("\n".join(lines) + "\n")
            else:
                pass

    base.ModelRunner.prepare_and_forward = prepare_and_forward


def watch() -> None:
    while not (
        BASE in sys.modules
        and THINKER in sys.modules
        and hasattr(sys.modules[THINKER], "ThinkerModelRunner")
        and hasattr(sys.modules[BASE], "ModelRunner")
    ):
        time.sleep(0.2)
    install(sys.modules[BASE], sys.modules[THINKER])


def pin_attention_splits(splits: int) -> None:
    """Causal control only: fixes FlashAttention's split-KV count for prefill and decode at `splits`, whatever the batch."""
    module_name = "sglang.srt.layers.attention.flashattention_backend"
    while module_name not in sys.modules or not hasattr(sys.modules[module_name], "FlashAttentionBackend"):
        time.sleep(0.05)
    backend = sys.modules[module_name].FlashAttentionBackend
    original_init = backend.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.num_splits = splits
        self.decode_num_splits = splits

    backend.__init__ = init
    module = sys.modules[module_name]
    original_kvcache = module.flash_attn_with_kvcache
    calls = {"count": 0}

    def flash_attn_with_kvcache(*args, **kwargs):
        kwargs["num_splits"] = splits
        if "scheduler_metadata" in kwargs:
            kwargs["scheduler_metadata"] = None
        else:
            pass
        calls["count"] += 1
        if calls["count"] == 1 and PREFIX:
            with open(f"{PREFIX}.{os.getpid()}.splits", "w", encoding="utf-8") as handle:
                handle.write(f"flash_attn_with_kvcache forced num_splits={splits}, scheduler_metadata dropped\n")
        else:
            pass
        return original_kvcache(*args, **kwargs)

    module.flash_attn_with_kvcache = flash_attn_with_kvcache


if PREFIX:
    threading.Thread(target=watch, name="thinker-numerics-probe", daemon=True).start()
    if os.environ.get("OMNI_FA_SPLITS"):
        threading.Thread(
            target=pin_attention_splits, args=(int(os.environ["OMNI_FA_SPLITS"]),), name="fa-splits-control", daemon=True
        ).start()
    else:
        pass
else:
    pass
