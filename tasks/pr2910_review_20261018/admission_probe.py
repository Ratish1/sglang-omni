"""Check whether the new Snake test detects dropped learned alpha factors."""

import copy
import json

import pytest
import torch
from cosyvoice.transformer.activation import Snake

from sglang_omni.models.fun_cosyvoice3 import stages
from tests.unit_test.fun_cosyvoice3.test_vocoder import (
    test_vocoder_hift_snake_is_bit_identical_to_cosyvoice,
)


@torch.inference_mode()
def main() -> None:
    real_snake = stages.snake

    def ignore_alpha(
        x: torch.Tensor, alpha: torch.Tensor, inverse_alpha: torch.Tensor
    ) -> torch.Tensor:
        return real_snake(x, torch.ones_like(alpha), torch.ones_like(inverse_alpha))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stages, "snake", ignore_alpha)
        test_vocoder_hift_snake_is_bit_identical_to_cosyvoice()
    module = Snake(17).cuda().eval()
    module.alpha.copy_(torch.linspace(0.125, 4.0, 17, device="cuda"))
    reference = copy.deepcopy(module)
    stages.patch_hift_snake(module)
    torch.manual_seed(2910)
    x = torch.randn(3, 17, 2049, device="cuda") * 30
    expected = reference(x)
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stages, "snake", ignore_alpha)
        broken = module(x)
    assert not torch.equal(broken, expected)
    print(
        json.dumps(
            dict(
                existing_test_passes_ignored_alpha=True,
                nonunit_alpha_passes_correct_kernel=True,
                nonunit_alpha_detects_ignored_alpha=True,
                broken_max_abs=(broken - expected).abs().max().item(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
