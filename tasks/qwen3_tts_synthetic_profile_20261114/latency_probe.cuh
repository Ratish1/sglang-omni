// SPDX-License-Identifier: Apache-2.0
// Cycle counts of the primitives the persistent predictor layer is built from,
// each measured by one warp or one CTA with clock64 around a dependent chain.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace latency_probe {

inline constexpr int kRepeats = 256;
inline constexpr int kChaseSteps = 2048;
inline constexpr int kBarrierThreads = 256;
// Result slots, in cycles per operation.
inline constexpr int kMmaDependent = 0;
inline constexpr int kMmaIndependent = 1;
inline constexpr int kLdmatrixDependent = 2;
inline constexpr int kL2Chase = 3;
inline constexpr int kDramChase = 4;
inline constexpr int kStoreFence = 5;
inline constexpr int kStoreRelease = 6;
inline constexpr int kNamedBarrier = 7;
inline constexpr int kClockRead = 8;
inline constexpr int kGlobalTimerRead = 9;
inline constexpr int kSlots = 10;

__device__ __forceinline__ void mma_bf16(float (&accumulator)[4],
                                         const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, "
               "%2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(accumulator[0]), "+f"(accumulator[1]),
                 "+f"(accumulator[2]), "+f"(accumulator[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0),
                 "r"(b1));
}

__global__ void mma_kernel(int64_t *cycles, float *sink) {
  const uint32_t lane = threadIdx.x;
  const uint32_t a[4] = {lane, lane + 1, lane + 2, lane + 3};
  float chain[4] = {};
  float independent[4][4] = {};
  long long start = clock64();
#pragma unroll 16
  for (int i = 0; i < kRepeats; ++i) {
    mma_bf16(chain, a, lane, lane + 7);
  }
  long long end = clock64();
  if (lane == 0)
    cycles[kMmaDependent] = (end - start) / kRepeats;
  start = clock64();
#pragma unroll 16
  for (int i = 0; i < kRepeats; ++i) {
    mma_bf16(independent[i % 4], a, lane, lane + 7);
  }
  end = clock64();
  if (lane == 0)
    cycles[kMmaIndependent] = (end - start) / kRepeats;
  sink[lane] = chain[0] + independent[0][0] + independent[1][1] +
               independent[2][2] + independent[3][3];
}

// Each ldmatrix address depends on the previous result, which is zero.
__global__ void ldmatrix_kernel(int64_t *cycles, uint32_t *sink) {
  __shared__ __align__(128) uint8_t tile[4096];
  const uint32_t lane = threadIdx.x;
  for (int offset = lane * 16; offset < 4096; offset += 32 * 16) {
    *reinterpret_cast<uint4 *>(tile + offset) = make_uint4(0, 0, 0, 0);
  }
  __syncwarp();
  uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(tile)) + lane * 16;
  uint32_t registers[4] = {};
  const long long start = clock64();
#pragma unroll 16
  for (int i = 0; i < kRepeats; ++i) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]),
          "=r"(registers[3])
        : "r"(address + registers[0]));
  }
  const long long end = clock64();
  if (lane == 0)
    cycles[kLdmatrixDependent] = (end - start) / kRepeats;
  sink[lane] = registers[0] + registers[1] + registers[2] + registers[3];
}

// Pointer chase over `chain`, whose element i holds the index of the next,
// with ld.global.cg as the kernel's staging and attention loads use.
__global__ void chase_kernel(int64_t *cycles, const uint32_t *chain,
                             uint32_t *sink, int slot) {
  uint32_t index = 0;
  const long long start = clock64();
  for (int i = 0; i < kChaseSteps; ++i) {
    index = __ldcg(chain + index);
  }
  const long long end = clock64();
  cycles[slot] = (end - start) / kChaseSteps;
  sink[0] = index;
}

__global__ void fence_kernel(int64_t *cycles, uint32_t *flag,
                             uint32_t *data) {
  long long start = clock64();
  for (int i = 0; i < kRepeats; ++i) {
    data[i] = i;
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
  }
  long long end = clock64();
  cycles[kStoreFence] = (end - start) / kRepeats;
  start = clock64();
  for (int i = 0; i < kRepeats; ++i) {
    data[i] = i;
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(flag)
                 : "memory");
  }
  end = clock64();
  cycles[kStoreRelease] = (end - start) / kRepeats;
}

__global__ void barrier_kernel(int64_t *cycles) {
  long long start = clock64();
  for (int i = 0; i < kRepeats; ++i) {
    asm volatile("bar.sync 1, %0;" ::"r"(kBarrierThreads) : "memory");
  }
  long long end = clock64();
  if (threadIdx.x == 0)
    cycles[kNamedBarrier] = (end - start) / kRepeats;
  int64_t clock_sum = 0;
  start = clock64();
  for (int i = 0; i < kRepeats; ++i) {
    clock_sum += clock64();
  }
  end = clock64();
  if (threadIdx.x == 0)
    cycles[kClockRead] = (end - start) / kRepeats + (clock_sum & 1);
  uint64_t timer_sum = 0;
  start = clock64();
  for (int i = 0; i < kRepeats; ++i) {
    uint64_t global_time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(global_time));
    timer_sum += global_time;
  }
  end = clock64();
  if (threadIdx.x == 0)
    cycles[kGlobalTimerRead] = (end - start) / kRepeats + (timer_sum & 1);
}

struct LatencyProbe {
  // cycles: int64 [kSlots] on the device. chain: uint32 pointer-chase ring
  // that fits in L2; dram_chain: one larger than L2.
  static void run(tvm::ffi::TensorView cycles, tvm::ffi::TensorView chain,
                  tvm::ffi::TensorView dram_chain, tvm::ffi::TensorView scratch) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto chain_length = SymbolicSize{"chain_length"};
    auto dram_length = SymbolicSize{"dram_length"};
    TensorMatcher({kSlots}).with_dtype<int64_t>().with_device(device).verify(
        cycles);
    TensorMatcher({chain_length})
        .with_dtype<uint32_t>()
        .with_device(device)
        .verify(chain);
    TensorMatcher({dram_length})
        .with_dtype<uint32_t>()
        .with_device(device)
        .verify(dram_chain);
    TensorMatcher({kRepeats + 64})
        .with_dtype<uint32_t>()
        .with_device(device)
        .verify(scratch);
    const DLDevice resolved = device.unwrap();
    auto *cycle_data = static_cast<int64_t *>(cycles.data_ptr());
    auto *scratch_data = static_cast<uint32_t *>(scratch.data_ptr());
    LaunchKernel(1, 32, resolved)(
        mma_kernel, cycle_data, reinterpret_cast<float *>(scratch_data));
    LaunchKernel(1, 32, resolved)(ldmatrix_kernel, cycle_data, scratch_data);
    LaunchKernel(1, 1, resolved)(
        chase_kernel, cycle_data,
        static_cast<const uint32_t *>(chain.data_ptr()), scratch_data,
        kL2Chase);
    LaunchKernel(1, 1, resolved)(
        chase_kernel, cycle_data,
        static_cast<const uint32_t *>(dram_chain.data_ptr()), scratch_data,
        kDramChase);
    LaunchKernel(1, 1, resolved)(fence_kernel, cycle_data, scratch_data,
                                 scratch_data + 64);
    LaunchKernel(1, kBarrierThreads, resolved)(barrier_kernel, cycle_data);
  }
};

} // namespace latency_probe

} // namespace sglang
