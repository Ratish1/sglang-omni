# SPDX-License-Identifier: Apache-2.0
"""Component check of #2625 on the GPU: FunCosyVoice3ModelRunner.collect_tokens given a CUDA tensor of ids
(the served case; the PR's unit test runs only the CPU path, where nothing is staged) leaves pinned host ids
on the result that equal the device ids, collects the same speech tokens as a plain .tolist() would, and over
three steps alternates the two pinned buffers without the earlier step's ids changing before it is consumed.

    python3 staged_ids_component_c20261012.py
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.request_builders import CosyVoice3SGLangRequestData
from sglang_omni.models.fun_cosyvoice3.sglang_model import EOS_ID, VOCAB_SIZE


def main() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.outbox = None
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    steps = [[7, EOS_ID, 13], [8, 9, 10], [11, 12, 6000]]
    seen = []
    for ids in steps:
        requests = [SimpleNamespace(data=CosyVoice3SGLangRequestData()) for _ in ids]
        device_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
        torch.cuda._sleep(1_000_000)
        result = SimpleNamespace(next_token_ids=device_ids)
        runner.collect_tokens(result, None, None, requests)
        host = runner.resolve_host_token_ids(result)
        assert host.device.type == "cpu" and host.is_pinned(), "host ids are not a pinned copy"
        assert host.tolist() == device_ids.tolist(), "host ids differ from the device ids"
        collected = [[int(code) for code in request.data.output_codes] for request in requests]
        assert collected == [[i] if i < VOCAB_SIZE else [] for i in ids], collected
        seen.append((host, list(ids)))
        assert seen[-2][0].tolist() == seen[-2][1] if len(seen) > 1 else True, "previous step's ids changed"
    print("staged ids: pinned, equal to the device ids, tokens collected as before, buffers alternate safely")


if __name__ == "__main__":
    main()
