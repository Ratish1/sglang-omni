import torch, time
d = torch.device("cuda")
dst = torch.zeros(4, dtype=torch.float32, device=d)
torch.cuda.synchronize()
torch.cuda._sleep(600_000_000)
t0 = time.perf_counter()
dst.copy_(torch.tensor([1.,2.,3.,4.]), non_blocking=True)
unpinned = (time.perf_counter()-t0)*1000
torch.cuda.synchronize()
torch.cuda._sleep(600_000_000)
t0 = time.perf_counter()
dst.copy_(torch.tensor([5.,6.,7.,8.], pin_memory=True), non_blocking=True)
pinned = (time.perf_counter()-t0)*1000
torch.cuda.synchronize()
print(f"unpinned non_blocking returned after {unpinned:8.2f} ms -> {'BLOCKED' if unpinned>50 else 'returned early'}")
print(f"pinned   non_blocking returned after {pinned:8.2f} ms -> {'BLOCKED' if pinned>50 else 'returned early'}")
