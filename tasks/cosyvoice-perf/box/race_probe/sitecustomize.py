"""A's race check, local only: wraps FunCosyVoice3ModelRunner.build_prefill_input_embeds so every prefill batch's
input rows, as the AR step reads them on its own stream, are compared bit for bit against rows rebuilt from the
host ids on the Req (SOS, text, TASK, prompt speech), and logs every new CUDA stream (handle, priority, caller) to
check that the preprocessing stream shares no pool slot with the vocoder's. Lines start with "race_probe".
Put this directory first on PYTHONPATH of the server only, with RACE_PROBE=1."""

import importlib.abc
import importlib.util
import os
import sys

RUNNER = "sglang_omni.models.fun_cosyvoice3.model_runner"
BUILDERS = "sglang_omni.models.fun_cosyvoice3.request_builders"


def log(message: str) -> None:
    print(f"race_probe {message}", file=sys.stderr, flush=True)


def patch_streams() -> None:
    import torch

    original = torch.cuda.Stream.__new__

    def logged_new(cls, device=None, priority=0, **kwargs):
        stream = original(cls, device, priority, **kwargs)
        if "stream_id" not in kwargs:
            caller = sys._getframe(1)
            log(
                f"stream handle={stream.cuda_stream:#x} priority={stream.priority} device={stream.device} "
                f"at {caller.f_code.co_filename}:{caller.f_lineno}"
            )
        return stream

    torch.cuda.Stream.__new__ = staticmethod(logged_new)


def patch_runner(module) -> None:
    import torch

    from sglang_omni.models.fun_cosyvoice3.sglang_model import SOS_ID, TASK_ID

    runner = module.FunCosyVoice3ModelRunner
    build = runner.build_prefill_input_embeds
    totals = {"batches": 0, "requests": 0, "rows": 0, "mismatched_requests": 0}

    def checked_build(self, forward_batch, requests):
        rows = build(self, forward_batch, requests)
        device = rows.device
        expected_pieces = []
        spans = []
        with torch.no_grad():
            for sched_req in requests:
                req = sched_req.data.req
                text_ids = torch.tensor(req._cosyvoice3_text_token_ids, dtype=torch.long, device=device)
                speech_ids = torch.tensor(
                    [SOS_ID, TASK_ID] + list(req._cosyvoice3_prompt_speech_token_ids),
                    dtype=torch.long,
                    device=device,
                )
                text_rows = self.model.text_embed_tokens(text_ids)
                speech_rows = self.model.speech_embedding(speech_ids)
                full = torch.cat([speech_rows[:1], text_rows, speech_rows[1:]])
                prefix_len = len(req.prefix_indices)
                req_len = int(req.extend_range.length)
                expected_pieces.append(full[prefix_len : prefix_len + req_len].to(rows.dtype))
                spans.append((sched_req.request_id, req_len))
            expected = torch.cat(expected_pieces)
            start = 0
            for request_id, req_len in spans:
                got = rows[start : start + req_len]
                want = expected[start : start + req_len]
                if not torch.equal(got, want):
                    differing = (got != want).any(dim=1).nonzero().flatten()
                    totals["mismatched_requests"] += 1
                    log(
                        f"MISMATCH request={request_id} rows={req_len} differing_rows={differing.numel()} "
                        f"first={int(differing[0])} max_abs={float((got.float() - want.float()).abs().max())}"
                    )
                start += req_len
        totals["batches"] += 1
        totals["requests"] += len(spans)
        totals["rows"] += int(rows.shape[0])
        log(
            f"batch {totals['batches']} requests={len(spans)} rows={rows.shape[0]} totals requests={totals['requests']} "
            f"rows={totals['rows']} mismatched_requests={totals['mismatched_requests']}"
        )
        return rows

    runner.build_prefill_input_embeds = checked_build
    log("runner patched")


class PatchOnImport(importlib.abc.MetaPathFinder):
    done: set[str] = set()

    def find_spec(self, name, path, target=None):
        if name not in (RUNNER, BUILDERS) or name in self.done:
            return None
        self.done.add(name)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            if name == BUILDERS:
                patch_streams()
                exec_module(module)
            else:
                exec_module(module)
                patch_runner(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("RACE_PROBE") == "1":
    sys.meta_path.insert(0, PatchOnImport())
    log("on")
