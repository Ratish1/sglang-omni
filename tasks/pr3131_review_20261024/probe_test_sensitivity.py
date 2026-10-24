"""Check whether the added vision integration test detects discarded attention."""

import json
from unittest.mock import patch

import torch

from tests.unit_test.qwen3_omni import test_vision_compat as test


class DiscardAttention(torch.nn.Module):
    def forward(self, query, key, value, cu_seqlens, **kwargs):
        return torch.zeros_like(value)


def main():
    encoder = test.vision_encoder.__wrapped__()
    with patch.object(test, 'resolve_packed_attention', return_value=DiscardAttention()):
        try:
            test.test_packed_attention_matches_the_fp32_transformers_encoder(encoder)
        except AssertionError as error:
            print(json.dumps(dict(discarded_attention_detected=True, error=str(error))), flush=True)
        else:
            print(json.dumps(dict(discarded_attention_detected=False)), flush=True)


if __name__ == '__main__':
    main()
