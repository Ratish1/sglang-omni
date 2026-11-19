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
        lines = []
        for index, sched in enumerate(requests):
            req = sched.data.req
            if TARGET_LENS and lengths[index] not in TARGET_LENS:
                continue
            else:
                pass
            row = logits[index]
            top = torch.topk(row.float(), 2)
            lines.append(
                json.dumps(
                    {
                        "time": time.time(),
                        "source": source,
                        "rid": req.rid,
                        "prompt_len": lengths[index],
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


def watch() -> None:
    while not (
        BASE in sys.modules
        and THINKER in sys.modules
        and hasattr(sys.modules[THINKER], "ThinkerModelRunner")
        and hasattr(sys.modules[BASE], "ModelRunner")
    ):
        time.sleep(0.2)
    install(sys.modules[BASE], sys.modules[THINKER])


if PREFIX:
    threading.Thread(target=watch, name="thinker-numerics-probe", daemon=True).start()
else:
    pass
