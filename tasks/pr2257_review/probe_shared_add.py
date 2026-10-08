import json

import torch


def main() -> None:
    torch.manual_seed(17)
    for dtype in (torch.bfloat16, torch.float16):
        routed = torch.randn(96, 2048, dtype=dtype, device="cuda")
        shared = torch.randn_like(routed)
        plain = routed + shared
        promoted = (routed.float() + shared.float()).to(dtype)
        assert torch.equal(plain, promoted)
        for name in ("plain", "promoted"):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ) as trace:
                if name == "plain":
                    output = routed + shared
                else:
                    output = (routed.float() + shared.float()).to(dtype)
                torch.cuda.synchronize()
            kernels = [
                event.name
                for event in trace.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
            print(
                json.dumps(
                    {
                        "dtype": str(dtype),
                        "path": name,
                        "exact_parity": torch.equal(output, plain),
                        "kernels": kernels,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
