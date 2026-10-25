"""Check whether the new allocation regression test preserves allocator policy."""

import json

import pytest
import torch


def main():
    device = torch.device("cuda", 0)
    original_fraction = torch.cuda.get_per_process_memory_fraction(device)
    try:
        torch.cuda.set_per_process_memory_fraction(0.75, device)
        before = torch.cuda.get_per_process_memory_fraction(device)
        status = pytest.main(
            [
                "-q",
                "tests/unit_test/fun_cosyvoice3/test_prefix_cache_gpu.py::"
                "test_prefix_pool_leaves_its_reserve_under_the_allocator_cap",
            ]
        )
        after = torch.cuda.get_per_process_memory_fraction(device)
        assert int(status) == 0
        assert after == before, (before, after)
        print(
            json.dumps(
                dict(
                    case="test_allocator_cleanup",
                    pytest_status=int(status),
                    before_fraction=before,
                    after_fraction=after,
                )
            ),
            flush=True,
        )
    finally:
        torch.cuda.set_per_process_memory_fraction(original_fraction, device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
