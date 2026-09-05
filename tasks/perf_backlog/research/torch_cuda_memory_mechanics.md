# torch 2.13.0 CUDA memory and cuDNN SDPA mechanics

Mechanics report. Every statement below is anchored to a file and line that was
read in full at the stated revision. Claims that could not be established from a
source are listed in the final Unverified section and appear nowhere else.

## Summary

The PyTorch CUDA caching allocator is a per-process C++ singleton
(`c10/cuda/CUDACachingAllocator.cpp:5303`). It reserves device memory from the
driver in segments through `cudaMalloc` and hands out blocks carved from those
segments. A freed tensor returns its block to an in-process pool and does not
call `cudaFree` (`free_block`, `c10/cuda/CUDACachingAllocator.cpp:3505-3579`).
The warning text the log carried is emitted at
`c10/cuda/CUDACachingAllocator.cpp:3933-3937`, inside `alloc_block`, immediately
after a `cudaMalloc` returned `cudaErrorMemoryAllocation`. It is a log line, not
a throw. `alloc_block` then clears CUDA's sticky error state at line 3947 and
returns false. The caller `malloc` runs a retry chain
(`c10/cuda/CUDACachingAllocator.cpp:1778-1792`) that releases cached blocks back
to the driver and calls `alloc_block` a second time. Only if that second attempt
also fails does `malloc` raise, and it raises `c10::OutOfMemoryError` with the
"CUDA out of memory. Tried to allocate ..." text
(`c10/cuda/CUDACachingAllocator.cpp:1928-1953`). The reported failure carried
neither that text nor that exception type, so the failing call was not a torch
allocation.

The reported exception text comes from
`aten/src/ATen/native/cudnn/MHA.cpp`, where four call sites wrap
`mha_graph.execute(...)` in a bare `TORCH_CHECK` with no message
(lines 1453-1454, 1575-1576, 1736-1737, 1880-1881). `TORCH_CHECK` with a bare
condition stringifies the expression into
"Expected ... to be true, but got false." (`c10/util/Exception.h:524-527`), and
`c10::Error` surfaces in Python as `RuntimeError`. The cuDNN frontend's own
message is discarded by that form of the check. The workspace buffer for that
call is sized by cuDNN and allocated through the torch caching allocator
(`aten/src/ATen/native/cudnn/MHA.cpp:1450-1452`), and a failure there would have
thrown `OutOfMemoryError` rather than produced a false `is_good()`.

On sm90 with a recent enough cuDNN, torch reorders the SDPA backend priority so
cuDNN is tried first (`aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:80-122`),
overriding the built-in default order in `aten/src/ATen/Context.h:480-485`.

## Provenance

All PyTorch paths were fetched from
`https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/<path>`. That tag's
`version.txt` reads `2.13.0a0`. Line numbers below are line numbers in those
files as fetched.

The reported warning line number, 3933, matches
`c10/cuda/CUDACachingAllocator.cpp:3933` in this tag exactly, which corroborates
that the running build is this source revision.

cuDNN frontend paths were fetched from
`https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/c4a97621eca52fa0c3a1862a411a16be580b25c6/<path>`.
That commit is the value of the `third_party/cudnn_frontend` submodule entry at
pytorch tag v2.13.0, read from the GitHub contents API.

NVIDIA documentation was fetched live from docs.nvidia.com. The CUDA Programming
Guide page served is labelled v13.3 and "Last updated on May 27, 2026". The cuDNN
backend API page served is the "latest" channel.

## 1. The CUDA caching allocator

### 1.1 Reserved versus allocated

Two distinct counters exist, both maintained inside the process.

`reserved_bytes` is incremented only in `alloc_block` after a successful
`cudaMalloc`, at `c10/cuda/CUDACachingAllocator.cpp:3967-3975`, together with
`total_allocated_memory += size` at line 3962. It is decremented only in
`release_block` after `cudaFree`, at lines 4143 and 4152-4160. Reserved therefore
tracks what the process holds from the driver.

`allocated_bytes` tracks what the program currently holds in live tensors. The
allocator's own comment block states the definitions plainly at
`c10/cuda/CUDACachingAllocator.cpp:1910-1923`:

```
      // "total capacity": total global memory on GPU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the CUDA API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
```

`torch.cuda.memory_reserved` reads `reserved_bytes.all.current`
(`torch/cuda/memory.py:571-583`).

Segment sizing is fixed by `get_allocation_size`
(`c10/cuda/CUDACachingAllocator.cpp:3700-3708`). A request at or below
`kSmallSize` reserves `kSmallBuffer`. A request below `kMinLargeAlloc` reserves
`large_segment_size()`. Anything larger is rounded up to a multiple of
`kRoundLarge`. The constants are in `c10/core/AllocatorConfig.h`, at lines 16, 20,
22 and 24: `kSmallBuffer` is 2097152, `kSmallSize` is 1048576, `kMinLargeAlloc` is
10485760, `kRoundLarge` is 2097152. `large_segment_size_` defaults to 20971520
(`c10/core/AllocatorConfig.h:339`).

Pool selection is by size at `c10/cuda/CUDACachingAllocator.cpp:3665-3669`, small
pool at or below `kSmallSize` and large pool above.

The byte counts in the reported warning are the `p.alloc_size` values, since the
warning prints the local `size` that was assigned from `p.alloc_size` at
`c10/cuda/CUDACachingAllocator.cpp:3851`. 612368384 is exactly 584 MiB and is a
multiple of `kRoundLarge`, consistent with the third branch of
`get_allocation_size`. 2097152 equals `kSmallBuffer` exactly. Under default
config the only branch of `get_allocation_size` that can produce 2097152 is the
small branch, so that request was for at most 1 MiB of tensor.

### 1.2 When freed blocks go back to the driver

They normally do not. `free` takes the lock and calls `free_locked`
(`c10/cuda/CUDACachingAllocator.cpp:2474-2479`), which reaches `free_block`
(`c10/cuda/CUDACachingAllocator.cpp:3505-3579`). `free_block` merges the block
with its neighbours and inserts it back into the pool at line 3542. There is no
`cudaFree` anywhere in that function. `cudaFree` appears only in `release_block`
(`c10/cuda/CUDACachingAllocator.cpp:4141`).

`release_block` is reached from exactly three places:

- `garbage_collect_cached_blocks` (line 3832), described below.
- `release_available_cached_blocks` (lines 4036 and 4045).
- `release_blocks` (line 4266), which is what `release_cached_blocks` and
  `emptyCache` use.

`release_blocks` (`c10/cuda/CUDACachingAllocator.cpp:4251-4275`) frees only
blocks with no neighbours, gated at line 4265 by `!block->prev && !block->next`.
A block that was split off a larger segment still in partial use is not returned.
Expandable-segment blocks take the `unmap_block` path at lines 4269-4274 instead.

### 1.3 What happens at and around line 3933

`alloc_block` spans `c10/cuda/CUDACachingAllocator.cpp:3843-3997`. The relevant
failure branch, verbatim, lines 3927-3954:

```
      if (p.err != cudaSuccess) {
        if (p.err == cudaErrorMemoryAllocation) {
          {
            size_t device_free = 0;
            size_t device_total = 0;
            (void)cudaMemGetInfo(&device_free, &device_total);
            LOG(WARNING) << "memory allocation failed with OOM on device "
                         << static_cast<int>(device_id)
                         << " while trying to allocate " << size
                         << " bytes (free: " << device_free
                         << ", total: " << device_total << ").";
          }
          // If this is the first attempt (!isRetry), we can forgive and clear
          // CUDA's internal error state.
          //
          // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
          // will take over to throw a helpful exception. The user can choose
          // to catch the exception, free some stuff in their script, and
          // attempt the allocation again. In this case, we can also forgive and
          // clear CUDA's internal error state.
          (void)cudaGetLastError();
        } else {
          // If the error's unrelated to memory allocation, we should throw
          // immediately.
          C10_CUDA_CHECK(p.err);
        }
        return false;
      }
```

Three mechanics follow from this block.

First, the free and total numbers in the warning are read by `cudaMemGetInfo`
after the `cudaMalloc` has already failed, at line 3932. They are a device-wide
snapshot, not a torch counter.

Second, line 3947 swallows the sticky CUDA error. A later CUDA call in the same
process will not see it.

Third, `alloc_block` never throws on `cudaErrorMemoryAllocation`. It returns
false. Any non-memory error goes to `C10_CUDA_CHECK` at line 3951 and throws
immediately.

The retry logic lives in the caller, `malloc`
(`c10/cuda/CUDACachingAllocator.cpp:1722-1960`). Lines 1778-1792, verbatim:

```
        block_found =
            // Try to use memory pools that have opted in as overflow before
            // expensive memory freeing operations.
            try_mempool_fallback(
                params, size, stream, device_id, alloc_size, stats)
            // Free enough available cached blocks to satisfy alloc and retry
            // alloc.
            || (release_available_cached_blocks(params, context) &&
                alloc_block(params, false, context, lock))
            // Free all non-split cached blocks and retry alloc.
            // Only skip this during actual graph capture; user mempools
            // should be able to reclaim cached memory.
            || (C10_LIKELY(!is_capture_context()) &&
                release_cached_blocks(context, {0, 0}) &&
                alloc_block(params, true, context, lock));
```

Under default configuration the middle step is a no-op.
`release_available_cached_blocks` returns false at its first statement,
`c10/cuda/CUDACachingAllocator.cpp:4005-4007`, when `max_split_size()` is
`SIZE_MAX`, which is its default (`c10/core/AllocatorConfig.h:341`). So the
default sequence is: first `alloc_block(isRetry=false)`, then
`try_mempool_fallback` (which only searches already-reserved opt-in pools,
`c10/cuda/CUDACachingAllocator.cpp:2052-2082`), then `release_cached_blocks`
followed by `alloc_block(isRetry=true)`.

`release_cached_blocks` (`c10/cuda/CUDACachingAllocator.cpp:4059-4103`) calls
`synchronize_and_free_events` at line 4069 and then `release_blocks` on the large
and small pools at lines 4072-4073. It always returns true (line 4102), which is
why the `&&` chain always proceeds to the retry.

A consequence worth stating: a single failing request that reaches the exception
logs the line at 3933 twice, with the same byte count both times, because
`p.alloc_size` does not change between attempts. A request that logs the warning
once and no exception is one whose retry succeeded after cached blocks were
returned to the driver.

The exception, when it comes, is raised at
`c10/cuda/CUDACachingAllocator.cpp:1928-1953`:

```
      TORCH_CHECK_WITH(
          OutOfMemoryError,
          false,
          "CUDA out of memory. Tried to allocate ",
```

The type is `c10::OutOfMemoryError`. Before throwing, `malloc` asserts at line
1799 that `params.err == cudaErrorMemoryAllocation`, re-reads `cudaMemGetInfo` at
line 1840, increments `stats.num_ooms` at line 1858, records an OOM trace entry
at lines 1850-1857, and notifies registered OOM observers at lines 1903-1908.

The entry point that a caller such as cuDNN's workspace allocation uses is
`NativeCachingAllocator::allocate` (`c10/cuda/CUDACachingAllocator.cpp:4958-4984`),
which delegates to `this->malloc` at line 4975 for non-zero sizes. There is no
nullptr-on-failure path there. Failure means `OutOfMemoryError` propagates out.

### 1.4 ASCII flow of the allocation failure path

```
CUDACachingAllocator::allocate(N)                          CCA.cpp:4958
        |  N != 0
        v
DeviceCachingAllocator::malloc(N, stream)                  CCA.cpp:1722
        |
        +-> round_size(N) -> size                          CCA.cpp:1731,3062
        +-> get_pool(size) -> small_blocks | large_blocks   CCA.cpp:1732,3644
        +-> get_allocation_size(size) -> alloc_size         CCA.cpp:1733,3700
        |
        v
   get_free_block(params)  ---- hit ----> split? --> return block
        |  miss                          CCA.cpp:1749-1753,1956
        v
   [gc, only if setMemoryFraction was called AND
        garbage_collection_threshold > 0]                   CCA.cpp:1758-1763
        |
        v
   alloc_block(params, isRetry=false)                       CCA.cpp:1770,3843
        |
        +-> cudaMallocMaybeCapturing(&ptr, alloc_size)      CCA.cpp:3918/3920
        |        |
        |        +-- cudaSuccess --> reserved_bytes += alloc_size --> true
        |        |                                          CCA.cpp:3962-3975
        |        +-- cudaErrorMemoryAllocation
        |                 |
        |                 +-> cudaMemGetInfo(&free,&total)  CCA.cpp:3932
        |                 +-> LOG(WARNING) "memory allocation failed with
        |                     OOM on device 0 while trying to allocate
        |                     <alloc_size> bytes (free: F, total: T)."
        |                                                   CCA.cpp:3933-3937
        |                 +-> cudaGetLastError()  [clears sticky error]
        |                                                   CCA.cpp:3947
        |                 +-> return false                   CCA.cpp:3953
        v
   retry chain                                              CCA.cpp:1778-1792
        |
        +-> try_mempool_fallback(...)      [opt-in pools only]  CCA.cpp:2052
        |        |  false
        +-> release_available_cached_blocks(...)                CCA.cpp:4002
        |        |  returns false immediately when
        |        |  max_split_size == SIZE_MAX (the default)    CCA.cpp:4005
        +-> release_cached_blocks(ctx, {0,0})                   CCA.cpp:4059
        |        +-> synchronize_and_free_events                CCA.cpp:4069
        |        +-> release_blocks(large_blocks)               CCA.cpp:4072
        |        +-> release_blocks(small_blocks)               CCA.cpp:4073
        |               +-> only !prev && !next blocks          CCA.cpp:4265
        |                      +-> release_block -> cudaFree    CCA.cpp:4141
        |                             reserved_bytes -= size    CCA.cpp:4154
        |        (always returns true)                          CCA.cpp:4102
        |
        +-> alloc_block(params, isRetry=true)                   CCA.cpp:1792
                 |
                 +-- success --> return block
                 +-- cudaErrorMemoryAllocation
                          +-> num_alloc_retries += 1            CCA.cpp:3855
                          +-> SAME warning logged again,
                              SAME byte count                   CCA.cpp:3933
                          +-> return false
        |
        v
   !block_found                                             CCA.cpp:1796
        +-> TORCH_INTERNAL_ASSERT(err == cudaErrorMemoryAllocation)  :1799
        +-> cudaMemGetInfo again                                     :1840
        +-> stats.num_ooms += 1                                      :1858
        +-> oom observers                                            :1903
        +-> throw c10::OutOfMemoryError
            "CUDA out of memory. Tried to allocate ..."              :1928
```

### 1.5 garbage_collection_threshold

Parsed by `AcceleratorAllocatorConfig::parseGarbageCollectionThreshold`
(`c10/core/AllocatorConfig.cpp:146-157`), which requires a value strictly inside
(0.0, 1.0) at lines 151-153. The default is 0
(`c10/core/AllocatorConfig.h:350`).

It gates `garbage_collect_cached_blocks`
(`c10/cuda/CUDACachingAllocator.cpp:3779-3836`), which computes a threshold as
the fraction multiplied by `allowed_memory_maximum.value()` at lines 3785-3787,
returns early if `total_allocated_memory` is at or under that threshold at lines
3789-3791, and otherwise frees non-split, non-expandable large blocks whose
`gc_count` is at or above the running average age, in a loop at lines 3812-3835.
Unlike `release_cached_blocks` it does not synchronize on events, as its own
comment at lines 3781-3783 says.

Two gating conditions matter. The call site is
`c10/cuda/CUDACachingAllocator.cpp:1758-1763`, which requires both
`allowed_memory_maximum.has_value()` and a non-zero threshold.
`allowed_memory_maximum` is set only in `setMemoryFraction`
(`c10/cuda/CUDACachingAllocator.cpp:2550-2561`), and only when the fraction is
strictly below 1.0 (line 2557). Setting `garbage_collection_threshold` without
also setting a memory fraction therefore changes nothing on the allocation path.
The same pairing gates the block-age bookkeeping in `get_free_block` at
`c10/cuda/CUDACachingAllocator.cpp:3713-3718`.

### 1.6 expandable_segments

Parsed by `AcceleratorAllocatorConfig::parseExpandableSegments`
(`c10/core/AllocatorConfig.cpp:227-234`), default false
(`c10/core/AllocatorConfig.h:352`). The CUDA-side accessor
`CUDAAllocatorConfig::expandable_segments` at
`c10/cuda/CUDAAllocatorConfig.h:34-46` returns false unconditionally when the
driver API is not compiled in.

The design note at `c10/cuda/CUDACachingAllocator.cpp:325-388` states the
mechanism. Address space is reserved once with `cuMemAddressReserve` and physical
memory is created separately with `cuMemCreate` and attached with `cuMemMap` and
`cuMemSetAccess` (lines 336-339). The comment at lines 341-346 says the allocator
reserves address space for essentially the whole device but maps only the
physical memory currently needed, growing at 2 MiB page granularity. Lines
348-350 state that on OOM the allocator can unmap the pages of its segment
corresponding to empty physical pages and return them to CUDA. Lines 358-360 say
2 MiB pages are used for the small pool and 20 MiB for the large pool. Lines
381-385 record the limitations: CUDA IPC of such tensors is not supported, and
`cudaDeviceEnablePeerAccess` does not work for memory allocated with `cuMemMap`.

The driver calls are in `struct ExpandableSegment`
(`c10/cuda/CUDACachingAllocator.cpp:390`), with `cuMemAddressReserve_` at line
415, `cuMemCreate_` at line 500, `cuMemRelease_` at lines 527, 710 and 939,
`cuMemSetAccess_` at line 873, `cuMemMap_` at line 891 and `cuMemUnmap_` at line
928.

In `alloc_block`, the expandable branch is taken at
`c10/cuda/CUDACachingAllocator.cpp:3896-3911`. On failure it sets
`p.err = cudaErrorMemoryAllocation` at line 3909 and returns without logging the
line-3933 warning, because that warning is in the `else` branch that calls
`cudaMallocMaybeCapturing`. Expandable segments are disabled for user mempools by
the `active_user_pool` test at `c10/cuda/CUDACachingAllocator.cpp:1734-1738`.

Freeing an expandable block goes through `unmap_block`
(`c10/cuda/CUDACachingAllocator.cpp:4173-4230`), which decrements
`total_allocated_memory` and `reserved_bytes` by the unmapped size at lines 4218
and 4221.

Expandable segments change how physical pages are grouped and returned. They do
not change how much physical memory the device has, and nothing in the design
note claims otherwise.

### 1.7 What torch.cuda.empty_cache releases

`torch.cuda.empty_cache` (`torch/cuda/memory.py:216-228`) calls
`torch._C._cuda_emptyCache`, which reaches
`DeviceCachingAllocator::emptyCache` (`c10/cuda/CUDACachingAllocator.cpp:2580-2584`):

```
  void emptyCache(MempoolId_t mempool_id) {
    auto context = maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks(context, mempool_id);
  }
```

So it releases exactly what the OOM retry path releases: after synchronizing
outstanding events, every cached block in the default pools that has no
neighbours (`c10/cuda/CUDACachingAllocator.cpp:4265`), by `cudaFree`
(line 4141). Blocks that are splits of a segment still partly in use stay. Live
allocations are untouched. The Python docstring at `torch/cuda/memory.py:217-225`
says the released memory becomes usable by other GPU applications and visible in
nvidia-smi, and that the call does not increase the memory available to PyTorch.

The malloc comment at `c10/cuda/CUDACachingAllocator.cpp:1925-1927` makes the
same point about the state at OOM time: by then all releasable cached memory has
already been returned, and what remains cached is split from a partly in-use
block.

## 2. Why several processes cannot share one caching allocator, and what "free" measures

### 2.1 The allocator is a process-local C++ object

The allocator is a file-scope static:
`static NativeCachingAllocator allocator;` at
`c10/cuda/CUDACachingAllocator.cpp:5303`. Its per-device state is a vector of
owned `DeviceCachingAllocator` objects,
`std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator` at
`c10/cuda/CUDACachingAllocator.cpp:4562`, created in-process at lines 4581-4583.
All of the state that decides whether a request can be served without touching
the driver lives in that object: the `large_blocks` and `small_blocks` pools,
`total_allocated_memory`, `allowed_memory_maximum`
(`c10/cuda/CUDACachingAllocator.cpp:1506`), and the `stats` counters. Mutual
exclusion is a `std::recursive_mutex` taken at
`c10/cuda/CUDACachingAllocator.cpp:1727`, which is a process-local lock.

Nothing in this data structure is placed in shared memory and no cross-process
handshake exists on the allocation path. The only cross-process facility present
is `shareIpcHandle` (`c10/cuda/CUDACachingAllocator.cpp:2501-2520`), which calls
`cudaIpcGetMemHandle` at line 2515 to export one specific base allocation, and
which explicitly refuses expandable-segment blocks at lines 2483-2485. That
exports a buffer, not the allocator.

The consequence for three CUDA processes on one card is direct. Each process has
its own pools and its own `total_allocated_memory`. A block cached in the
tts_engine process is invisible to the preprocessing process's `get_free_block`
at `c10/cuda/CUDACachingAllocator.cpp:3710`, and the preprocessing process's
`release_cached_blocks` at line 4059 can only free segments its own
`DeviceCachingAllocator` owns. Cached-but-unused memory in one process is
unreachable by the other two, yet it is reserved from the driver and so it is
absent from every process's `cudaMemGetInfo` free figure.

### 2.2 What "free" in the warning measures

The warning's free and total come from `cudaMemGetInfo` at
`c10/cuda/CUDACachingAllocator.cpp:3932`. The NVIDIA CUDA Runtime API reference
for `cudaMemGetInfo`, fetched from
`https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html`,
states verbatim:

> Returns in *total the total amount of memory available to the the current
> context. Returns in *free the amount of memory on the device that is free
> according to the OS. CUDA is not guaranteed to be able to allocate all of the
> memory that the OS reports as free. In a multi-tenet situation, free estimate
> returned is prone to race condition where a new allocation/free done by a
> different process or a different thread in the same process between the time
> when free memory was estimated and reported, will result in deviation in free
> value reported and actual free memory.

Three properties follow, all stated by that text. The figure is device-wide and
not per-process, so it already accounts for every other process's reservations.
It is an OS-level estimate, and CUDA is explicitly not guaranteed to be able to
allocate all of it. And in a multi-process setting it is racy by construction.

This is why the second reported warning is internally consistent: free is
2228224 and the request is 2097152, that is 2.125 MiB free against a 2 MiB
request, and `cudaMalloc` still failed. The documentation above covers exactly
that case.

The first reported warning reads free 442630144 against a request of 612368384,
that is 422.125 MiB free against 584 MiB requested, an ordinary shortfall. The
total, 85017493504 bytes or 79.18 GiB, is what `cudaMemGetInfo` reports as
available to the context.

## 3. The cuDNN SDPA path

### 3.1 Route into cuDNN

`_scaled_dot_product_cudnn_attention_cuda`
(`aten/src/ATen/native/transformers/cuda/attention.cu:1150-1166`) forwards to
`at::_cudnn_attention_forward`, whose implementation is at lines 986-1148 of the
same file. For the dense, non-nested case it calls `run_cudnn_SDP_fprop` at lines
1052-1069.

`run_cudnn_SDP_fprop` is at `aten/src/ATen/native/cudnn/MHA.cpp:1310-1455`. It
returns immediately for empty inputs (lines 1330-1332), allocates the output if
undefined (line 1338), allocates the softmax statistics tensor if requested
(lines 1341-1349), fetches the handle at line 1379, builds a cache key at lines
1384-1398, and looks up or builds the graph at lines 1399-1421.

The graph cache is thread-local, `getMHAGraphCache_`
(`aten/src/ATen/native/cudnn/MHA.cpp:392-395`). A cache miss calls `build_graph`
(line 447), whose plan-preparation steps are at lines 640-644:
`validate`, `build_operation_graph(handle)`,
`create_execution_plans({fe::HeurMode_t::A})` and `check_support(handle)`, each
wrapped in `AT_CUDNN_FRONTEND_CHECK`.

### 3.2 Workspace sizing and allocation

`aten/src/ATen/native/cudnn/MHA.cpp:1450-1454`, verbatim:

```
  auto workspace_size = mha_graph.get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(
      mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good());
```

The size comes from cuDNN. The buffer comes from the torch caching allocator, so
it counts toward `allocated_bytes` and can trigger a segment reservation and the
line-3933 warning. Because `allocate` reaches `malloc`
(`c10/cuda/CUDACachingAllocator.cpp:4975`), a workspace that cannot be satisfied
raises `OutOfMemoryError` at `c10/cuda/CUDACachingAllocator.cpp:1928`. It cannot
produce a false `is_good()`.

The other three call sites are identical in shape. Two of them add a defensive
null check that the fprop dense path lacks:
`TORCH_CHECK(!workspace_size || workspace_ptr.get())` at
`aten/src/ATen/native/cudnn/MHA.cpp:1735` and 1879.

### 3.3 What the mha_graph.execute check does

The check is `TORCH_CHECK(cond)` with no message. `TORCH_CHECK_MSG`
(`c10/util/Exception.h:524-527`) builds the string:

```
#define TORCH_CHECK_MSG(cond, type, ...)                   \
  (::c10::detail::torchCheckMsgImpl(                       \
      "Expected " #cond                                    \
      " to be true, but got false.  "                      \
```

The `#cond` stringification of the condition at
`aten/src/ATen/native/cudnn/MHA.cpp:1454` is exactly
`mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good()`, which
reproduces the reported message character for character. The same stringification
is produced at lines 1576, 1737 and 1881, because the expression text is
identical at all four sites. The message alone therefore cannot distinguish dense
fprop, nested fprop, dense bprop and nested bprop.

Two consequences. First, the cuDNN frontend's own diagnostic is thrown away. The
`error_object` returned by `execute` carries a message accessible through
`get_message()` (`cudnn-frontend include/cudnn_frontend/graph_helpers.h:66-69`),
and the codebase has a macro that would have printed it,
`AT_CUDNN_FRONTEND_CHECK` (`aten/src/ATen/cuda/Exceptions.h:30-37`), which
formats `"cuDNN Frontend error: ", error_object.get_message()`. That macro is used
for the build-time steps at `aten/src/ATen/native/cudnn/MHA.cpp:640-644` but not
for `execute`.

Second, the exception type is `c10::Error`, which is what plain `TORCH_CHECK`
raises, not `c10::OutOfMemoryError`. That is consistent with the reported
`RuntimeError` and inconsistent with a torch allocation failure.

### 3.4 Which conditions make is_good() false

`is_good` is defined at
`cudnn-frontend include/cudnn_frontend/graph_helpers.h:71-74` as
`code == error_code_t::OK`. The full enum is at lines 36-53:

```
enum class [[nodiscard]] error_code_t {
    OK,
    ATTRIBUTE_NOT_SET,
    SHAPE_DEDUCTION_FAILED,
    INVALID_TENSOR_NAME,
    INVALID_VARIANT_PACK,
    GRAPH_NOT_SUPPORTED,
    GRAPH_EXECUTION_PLAN_CREATION_FAILED,
    GRAPH_EXECUTION_FAILED,
    HEURISTIC_QUERY_FAILED,
    UNSUPPORTED_GRAPH_FORMAT,
    CUDA_API_FAILED,
    CUDNN_BACKEND_API_FAILED,
    INVALID_CUDA_DEVICE,
    HANDLE_ERROR,
    INVALID_VALUE,
    NVRTC_COMPILATION_FAILED
};
```

The `execute` overload torch calls is at
`cudnn-frontend include/cudnn_frontend/graph_interface.h:1288-1293`, which
forwards to `execute_plan_at_index`. The implementation is at lines 1361-1463.
Reading it end to end, the code paths that can return a non-OK code are:

- `prepare_variant_pack_template` failing on the lazy-init path, lines 1371-1373.
- `plans.is_plan_index_executable(plan_index)` failing, line 1375.
- Variant-pack shape mismatches, `INVALID_VARIANT_PACK` at lines 1378-1384 and at
  lines 1350-1352 in the uid-map overload.
- `run_auxiliary_kernels` failing, line 1416. That helper is at lines 244-268 and
  issues `cudaMemcpyAsync` and `cudaMemsetAsync` into the workspace. Both are
  wrapped by `_CUDNN_CHECK_CUDA_ERROR`
  (`graph_helpers.h:136-145`), which maps any non-`cudaSuccess` return to
  `CUDA_API_FAILED`. Because CUDA runtime calls report previously deferred
  asynchronous errors, a fault raised by an earlier launch in the same stream can
  surface here.
- The OSS SDPA engine dispatch at lines 1423-1436, which calls
  `plans.execute_oss_sdpa_engine`
  (`cudnn-frontend include/cudnn_frontend/plans.h:828`).
- The backend dispatch at lines 1448-1461, `detail::execute`, whose cuDNN status
  is mapped by `_CUDNN_CHECK_CUDNN_ERROR` (`graph_helpers.h:124-134`) to
  `CUDNN_BACKEND_API_FAILED`.

For the backend path, the underlying call is `cudnnBackendExecute`. The NVIDIA
cuDNN Backend API reference, fetched from
`https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html`,
gives its returns as `CUDNN_STATUS_SUCCESS`, `CUDNN_STATUS_BAD_PARAM`
("An incorrect or inconsistent value is encountered. For example, a required data
pointer is invalid."), `CUDNN_STATUS_INTERNAL_ERROR` ("Some internal errors were
encountered.") and `CUDNN_STATUS_EXECUTION_FAILED` ("An error was encountered
executing the plan with the variant pack."). The same page describes the variant
pack as containing a "Pointer to user-allocated workspace in global memory at
least as large as the size queried".

### 3.5 Whether cuDNN allocates device memory outside torch's allocator

Yes, and the library documents error codes for exactly that. The same cuDNN
Backend API page lists, among the `CUDNN_STATUS_INTERNAL_ERROR` sub-codes:

> CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED A runtime kernel has failed to
> be compiled.
>
> CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED An internal host memory
> allocation failed inside the cuDNN library.
>
> CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED Resource allocation failed
> inside the cuDNN library.

The existence of `CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED`
establishes that cuDNN performs device resource allocation internally, outside
the workspace pointer the caller supplies, and reports failure of it.

The same page documents the handle: "cudnnHandle_t is a pointer to an opaque
structure holding the cuDNN library context. The cuDNN library context must be
created using cudnnCreate() ... The context is associated with only one GPU
device, the current device at the time of the call to cudnnCreate()."

The cuDNN frontend vendored at this pytorch tag also compiles and loads kernels
itself on its OSS engine path. `sm90_sdpa_prefill_engine.h:191-203` is labelled
"Phase 2: NVRTC compilation + module loading" and calls
`compile_and_load_kernel`. That function
(`include/cudnn_frontend/experimental/oss_engine_interface.h:164-256`) runs NVRTC
on an embedded source, then at line 243 calls `cuda_library_load_data`, mapping a
failure to `CUDA_API_FAILED` with the message
"cudaLibraryLoadData failed (cubin_size=...)". Note that this happens in
`build()`, not in `execute()`. On the sm90 engine, `execute` starts by refusing
to run if `built_` is false (`sm90_sdpa_prefill_engine.h:232`) and then builds TMA
descriptors (lines 279, 295, 311, 327). Its static workspace requirement is 16
bytes (`sm90_sdpa_prefill_engine.h:186-189`).

Whether the running cuDNN chose the OSS engine or the backend engine for this
graph is not determinable from the reported message, since torch discards the
frontend message.

### 3.6 How the dispatcher picks cuDNN over flash and efficient on sm90

The selection function is `select_sdp_backend`
(`aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:1049-1112`). It returns
`SDPBackend::error` if all four user-enable switches are off (lines 1055-1058),
then walks the ordering returned by `priority_order` and returns the first
backend whose `can_use_*` predicate passes (lines 1065-1095). It is reached from
`_fused_sdp_choice_cuda`
(`aten/src/ATen/native/transformers/cuda/attention.cu:1287-1298`), which is
registered as the CUDA dispatch for `_fused_sdp_choice_stub` at line 1866 of the
same file.

The built-in default order is in `aten/src/ATen/Context.h:480-485`:

```
  std::array<at::SDPBackend, at::num_sdp_backends> sdp_priority_order = {
      at::SDPBackend::flash_attention,
      at::SDPBackend::efficient_attention,
      at::SDPBackend::math,
      at::SDPBackend::cudnn_attention,
      at::SDPBackend::overrideable};
```

cuDNN is last in that default. It is moved to the front by `priority_order`
(`aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:110-122`), which on its
first ever call consults `check_prefer_cudnn_attention` and, if that returns
true, installs the order cuDNN, flash, efficient, math through
`at::globalContext().setSDPPriorityOrder` at lines 114-118. The one-shot guard is
the file-scope `priority_order_init_` at line 76.

`check_prefer_cudnn_attention` is at lines 80-107. Verbatim, lines 81 and 92-97:

```
  static const bool prefer_cudnn = c10::utils::check_env("TORCH_CUDNN_SDPA_DEPRIORITIZED") != true;
```

```
#if defined(CUDA_VERSION) && (CUDA_VERSION < 13000)
    auto minor = dprops->minor;
    return cudnn_version > 91500 && (major == 9 || major == 10) && (!minor || minor == 3);
#else
    return cudnn_version > 91500 && (major == 9 || major == 10);
#endif
```

For a build against CUDA 13.0, the `#else` arm applies. On an H100, `major` is 9.
So the predicate reduces to a runtime cuDNN version strictly greater than 91500,
which the comment at lines 85-86 explains as "cuDNN 9.15.1 required for seq_len
not divisible by 128 fix". The runtime cuDNN version is read once through
`at::detail::getCUDAHooks().versionRuntimeCuDNN()` at line 88.

Two knobs change the outcome.

The environment variable `TORCH_CUDNN_SDPA_DEPRIORITIZED`, read once at line 81,
makes `check_prefer_cudnn_attention` return false at lines 82-84, leaving the
default order in which cuDNN is last.

The API `at::globalContext().setSDPPriorityOrder`
(`aten/src/ATen/Context.h:275`) sets the order directly. The comment at
`sdp_utils.cpp:66-75` describes the interaction with a user-supplied order set
through a context manager.

Independently of ordering, `check_runtime_disabled_cudnn`
(`sdp_utils.cpp:826-836`) rejects the cuDNN backend when
`userEnabledCuDNNSDP()` is false. That flag defaults to true
(`aten/src/ATen/Context.h:490`). It is the first of the general constraints in
`can_use_cudnn_attention` (`sdp_utils.cpp:853`, constraint list at lines 877-888,
dense constraints at lines 894-900).

## 4. CUDA lazy module loading

The NVIDIA CUDA Programming Guide section 4.7, fetched from
`https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html`
(page labelled v13.3), states:

> Lazy loading reduces program initialization time by waiting to load CUDA
> modules until they are needed. ... As of CUDA 12.3 lazy Loading is enabled by
> default on all platforms, but can be controlled via the CUDA_MODULE_LOADING
> environment variable.

Its change history table records "12.2 Lazy loading enabled by default for Linux"
and "11.7 Lazy loading first introduced, disabled by default". Driver version 515
or newer is required.

On enabling and disabling:

> Lazy loading is enabled by setting the CUDA_MODULE_LOADING environment variable
> to LAZY . Lazy loading can be disabled by setting the CUDA_MODULE_LOADING
> environment variable to EAGER . As of CUDA 12.3, lazy loading is enabled by
> default on all platforms.

On whether loading a module allocates device memory, the guide answers directly
in two places. Section 4.7.4.3, "Forcing a Module to Load Eagerly at Runtime":

> The cuModuleGetFunction() function will cause a module to be loaded into device
> memory
>
> The cudaFuncGetAttributes() function will cause a kernel to be loaded into
> device memory

And section 4.7.5.2, "Large Memory Allocations":

> Lazy loading delays memory allocation for CUDA modules from program
> initialization until closer to execution time. If an application allocates the
> entire VRAM on startup, CUDA can fail to allocate memory for modules at
> runtime. Possible solutions:
>
> use cudaMallocAsync() instead of an allocator that allocates the entire VRAM on
> startup
>
> add some buffer to compensate for the delayed loading of kernels
>
> preload all kernels that will be used in the program before trying to
> initialize the allocator

So loading a module does allocate device memory, that allocation is deferred to
first use under lazy loading, and NVIDIA documents the failure mode where a
process that has taken the device's memory leaves nothing for a module loaded
later. That allocation is made by the driver, not by torch's caching allocator,
so it does not appear in `reserved_bytes` or `allocated_bytes` and is not
released by `torch.cuda.empty_cache`.

The guide also notes that section 4.7.4.2 gives `cuModuleGetLoadingMode` as the
runtime query for whether lazy loading is active, and that
"cuModuleLoad() does not guarantee that a module will be loaded immediately."

Section 4.7.3.4 records one exception: "Lazy loading does not affect modules
containing managed variables, which will still be loaded eagerly."

PyTorch v2.13.0 does not set `CUDA_MODULE_LOADING` in `torch/cuda/__init__.py`.
A case-insensitive grep of that file at this tag returns no match.

## Unverified

The following were not established from a source and are recorded here rather
than asserted above.

1. Which of the four `mha_graph.execute` call sites in
   `aten/src/ATen/native/cudnn/MHA.cpp` produced the reported exception. The
   `TORCH_CHECK` stringification is byte-identical at lines 1454, 1576, 1737 and
   1881, so the message cannot distinguish them.
2. Which `error_code_t` value the failing `execute` actually returned. The bare
   `TORCH_CHECK` discards `get_message()` and the code, so no evidence remains in
   the log.
3. Whether the running cuDNN dispatched this graph through the OSS SDPA engine
   path (`graph_interface.h:1423-1436`) or the backend path (lines 1448-1461).
4. Whether the two reported warnings came from two distinct allocation requests
   or from retries of one. The mechanics say a retried request logs the same byte
   count twice, and the two reported lines carry different byte counts, but the
   log excerpt available may not be complete.
5. Whether either reported warning was followed by a torch `OutOfMemoryError` in
   the same process. Not present in the excerpt provided.
6. The runtime cuDNN version, the compiled `CUDA_VERSION`, and the value of
   `TORCH_CUDNN_SDPA_DEPRIORITIZED` in the failing environment. Whether
   `check_prefer_cudnn_attention` returned true, and therefore whether cuDNN was
   actually first in the priority order, depends on these.
7. Whether `torch.cuda.set_per_process_memory_fraction` was called in any of the
   three processes. Without it, `garbage_collection_threshold` is inert per
   `c10/cuda/CUDACachingAllocator.cpp:1758-1763` and 2550-2561.
8. The value of `PYTORCH_CUDA_ALLOC_CONF` in the failing processes.
9. How much device memory cuDNN allocates internally per handle or per loaded
   kernel. The cuDNN documentation read establishes that such allocation exists
   and can fail, but gives no sizes.
10. Whether `CUDA_MODULE_LOADING` was set in the failing environment, and whether
    any module load actually failed there. The NVIDIA documentation establishes
    the mechanism and the documented failure mode, not that it occurred.
11. Whether the CUDA Programming Guide text quoted, served as v13.3, differs from
    the CUDA 13.0 revision of the same section. The archive was not consulted.
12. Whether the vendored cuDNN frontend at commit
    `c4a97621eca52fa0c3a1862a411a16be580b25c6` is the one actually compiled into
    the installed torch wheel, as opposed to a system cuDNN frontend.
13. Whether the preprocessing process's nvidia-smi figure equalling total was
    caused by that process, by the other two, by driver reservations, or by some
    combination. `cudaMemGetInfo` does not attribute memory to a process.
