"""Synthetic profile probe: forced codec frame counts and labeled step ranges.

QWEN3_TTS_SYNTHETIC=1 turns it on in every process that imports it:
- the talker's codec EOS is masked on every row, so max_new_tokens sets each request's frames;
- each talker step gets an NVTX range named with its mode, batch size, new tokens and KV tokens;
- each vocoder batch gets an NVTX range named with its stream count.
"""

import importlib.abc
import os
import sys

ENABLED = os.environ.get("QWEN3_TTS_SYNTHETIC") == "1"


def label_range(message, domain):
    import nvtx

    return nvtx.annotate(message, domain=domain)


def mask_codec_eos(module):
    runner_class = module.Qwen3TTSModelRunner
    original = runner_class.apply_codec_suppress_tokens

    def apply_codec_suppress_tokens(self, logits_output, requests):
        original(self, logits_output, requests)
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2 or not requests:
            return
        else:
            pass
        codec_eos = int(self.model.config.codec_eos_token_id)
        logits[: len(requests), codec_eos : codec_eos + 1] = float("-inf")

    runner_class.apply_codec_suppress_tokens = apply_codec_suppress_tokens


def label_talker_steps(module):
    scheduler_class = module.OmniScheduler
    original = scheduler_class.run_batch

    def run_batch(self, batch, pp_proxy_tensors=None):
        mode = batch.forward_mode.name
        request_count = len(batch.reqs)
        if batch.forward_mode.is_extend():
            new_tokens = int(batch.extend_num_tokens or 0)
        else:
            new_tokens = request_count
        message = (
            f"step[{mode} bs={request_count} toks={new_tokens} "
            f"kv={int(batch.seq_lens_sum or 0)}]"
        )
        with label_range(message, "talker"):
            return original(self, batch, pp_proxy_tensors)

    scheduler_class.run_batch = run_batch


def label_vocoder_batches(module):
    vocoder_class = module.Qwen3TTSStreamingVocoderScheduler
    for name, kind in (
        ("run_initial_batch", "initial"),
        ("run_followup_batch", "followup"),
    ):
        original = getattr(vocoder_class, name)

        def labeled(self, batch, original=original, kind=kind):
            with label_range(f"vocoder.{kind}[streams={len(batch)}]", "vocoder"):
                return original(self, batch)

        setattr(vocoder_class, name, labeled)


class PatchAfterImport(importlib.abc.MetaPathFinder):
    """Runs a patch on a module right after its first import."""

    def __init__(self, patches):
        self.patches = patches

    def find_spec(self, name, path, target=None):
        patch = self.patches.get(name)
        if patch is None:
            return None
        else:
            pass
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            else:
                pass
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
            else:
                pass
        else:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if ENABLED:
    sys.meta_path.insert(
        0,
        PatchAfterImport(
            {
                "sglang_omni.models.qwen3_tts.model_runner": mask_codec_eos,
                "sglang_omni.scheduling.omni_scheduler": label_talker_steps,
                "sglang_omni.models.qwen3_tts.streaming_vocoder": label_vocoder_batches,
            }
        ),
    )
else:
    pass
