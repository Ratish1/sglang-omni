"""Measurement only: a captured decode replay skips ModelRunner.forward's wrappers and the shared experts override."""

import contextlib
import importlib.abc
import importlib.util
import sys

RUNNER = "sglang_omni.model_runner.sglang_model_runner"
DECODE = "sglang.srt.model_executor.runner.decode_cuda_graph_runner"


def patch_runner(module) -> None:
    upstream = sys.modules["sglang.srt.model_executor.model_runner"]
    cls = module.SGLModelRunner
    original = cls.forward

    def forward(self, forward_batch, skip_attn_backend_init=None, pp_proxy_tensors=None, reinit_attn_backend=False, split_forward_count=1):
        graph_runner = self.decode_cuda_graph_runner
        if forward_batch.forward_mode.is_cuda_graph() and graph_runner and graph_runner.can_run_graph(forward_batch):
            self.forward_pass_id += 1
            context = upstream.ForwardContext(attn_backend=self.attn_backend)
            with upstream.forward_context(context, forward_batch=forward_batch):
                logits_output = graph_runner.execute(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
            return upstream.ModelRunnerOutput(logits_output=logits_output, can_run_graph=True)
        return original(self, forward_batch, skip_attn_backend_init, pp_proxy_tensors, reinit_attn_backend, split_forward_count)

    cls.forward = forward
    print("decode_fast runner patched", file=sys.stderr, flush=True)


def patch_decode(module) -> None:
    module.DecodeCudaGraphRunner.model_context = lambda self: contextlib.nullcontext()
    print("decode_fast decode runner patched", file=sys.stderr, flush=True)


PATCHES = {RUNNER: patch_runner, DECODE: patch_decode}


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen: set[str] = set()

    def find_spec(self, name, path, target=None):
        if name not in PATCHES or name in self.seen:
            return None
        self.seen.add(name)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(loaded):
            exec_module(loaded)
            PATCHES[name](loaded)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, PatchOnImport())
