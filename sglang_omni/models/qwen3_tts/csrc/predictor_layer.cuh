// SPDX-License-Identifier: Apache-2.0
// One Qwen3-TTS code predictor decoder layer as one persistent kernel: one
// token a row at one cache slot, for up to 16 rows. Every CTA owns a fixed list
// of work items; a producer warp streams each item's weight chunks into a
// shared-memory ring ahead of need, and phases hand their outputs to the next
// through global arrival counters.

#pragma once

#include <sgl_kernel/atomic.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {

namespace qwen3_tts_predictor {

inline constexpr int kHiddenSize = 1024;
inline constexpr int kHeadDim = 128;
inline constexpr int kHalfHeadDim = kHeadDim / 2;
inline constexpr int kQueryHeads = 16;
inline constexpr int kKeyValueHeads = 8;
inline constexpr int kQkvWidth = (kQueryHeads + 2 * kKeyValueHeads) * kHeadDim;
inline constexpr int kKeyOffset = kQueryHeads * kHeadDim;
inline constexpr int kValueOffset = kKeyOffset + kKeyValueHeads * kHeadDim;
inline constexpr int kAttentionWidth = kQueryHeads * kHeadDim;
inline constexpr int kIntermediateSize = 3072;
inline constexpr int kCacheSlots = 17;

// A weight chunk is 16 output features by 1024 input features, the
// shared-memory image one bulk copy fills; an MMA k-step covers 16 input
// features.
inline constexpr int kTileFeatures = 16;
inline constexpr int kChunkDepth = 1024;
inline constexpr int kChunkElements = kTileFeatures * kChunkDepth;
inline constexpr int kChunkBytes = kChunkElements * 2;
inline constexpr int kKStepDepth = 16;
inline constexpr int kKStepsPerChunk = kChunkDepth / kKStepDepth;
inline constexpr int kKStepBytes = kTileFeatures * kKStepDepth * 2;

inline constexpr int kComputeWarps = 8;
inline constexpr int kComputeThreads = kComputeWarps * 32;
inline constexpr int kThreads = kComputeThreads + 32;
inline constexpr int kKStepsPerWarp = kKStepsPerChunk / kComputeWarps;
inline constexpr int kRingStages = 3;

inline constexpr int kMaxRows = 16;
inline constexpr int kTokenTileRows = 8;
inline constexpr int kMaxTokenTiles = kMaxRows / kTokenTileRows;
inline constexpr int kRowsPerWarp = kMaxRows / kComputeWarps;
// The MMA B operand keeps each token's row contiguous; 16 bytes of padding a
// row put the 8 rows one ldmatrix reads in different banks.
inline constexpr int kOperandRowPadding = 16;
inline constexpr int kOperandBytes =
    kMaxRows * (kIntermediateSize * 2 + kOperandRowPadding);
inline constexpr int kPartialFloats =
    kComputeWarps * kMaxTokenTiles * kTileFeatures * kTokenTileRows;
inline constexpr int kPartialBuffers = 2;

// Weights every launch reads in full, staged once: both layer norms, the query
// and key norms, and the rope cos/sin row at the position.
inline constexpr int kInputNormOffset = 0;
inline constexpr int kPostNormOffset = kInputNormOffset + kHiddenSize * 2;
inline constexpr int kQueryNormOffset = kPostNormOffset + kHiddenSize * 2;
inline constexpr int kKeyNormOffset = kQueryNormOffset + kHeadDim * 2;
inline constexpr int kRopeOffset = kKeyNormOffset + kHeadDim * 2;
inline constexpr int kConstantBytes = kRopeOffset + kHeadDim * 4;

inline constexpr int kSharedBytes = kRingStages * kChunkBytes + kOperandBytes +
                                    kPartialBuffers * kPartialFloats * 4 +
                                    kConstantBytes + kMaxRows * 4 +
                                    2 * kRingStages * 8;

// One attention item's shared memory, inside the operand buffer: the cached
// keys and values in 272-byte rows so lanes reading different slots hit
// different banks, the raw query, key and value heads of the group, and each
// query warp's rotated key and fp32 query.
inline constexpr int kHeadVectors = kHeadDim / 8;
inline constexpr int kHeadRowBytes = kHeadDim * 2 + 16;
inline constexpr int kAttentionHeads = 4;
inline constexpr int kAttentionKeysOffset = 0;
inline constexpr int kAttentionValuesOffset = kCacheSlots * kHeadRowBytes;
inline constexpr int kAttentionHeadsOffset = 2 * kCacheSlots * kHeadRowBytes;
inline constexpr int kAttentionRotatedKeysOffset =
    kAttentionHeadsOffset + kAttentionHeads * kHeadDim * 2;
inline constexpr int kAttentionQueriesOffset =
    kAttentionRotatedKeysOffset + 2 * kHeadRowBytes;
inline constexpr int kAttentionBytes =
    kAttentionQueriesOffset + 2 * kHeadDim * 4;
inline constexpr int kAttentionVectors =
    (2 * (kCacheSlots - 1) + kAttentionHeads) * kHeadVectors;
inline constexpr int kAttentionVectorsPerThread =
    (kAttentionVectors + kComputeThreads - 1) / kComputeThreads;
static_assert(kAttentionBytes <= kOperandBytes);

inline constexpr int kPhaseQkv = 0;
inline constexpr int kPhaseAttention = 1;
inline constexpr int kPhaseOutputProjection = 2;
inline constexpr int kPhaseGateUp = 3;
inline constexpr int kPhaseDown = 4;
inline constexpr int kPhaseEvents = 4;
inline constexpr int kScheduleTileBits = 12;
inline constexpr int kScheduleTileMask = (1 << kScheduleTileBits) - 1;
inline constexpr int kQkvTiles = kQkvWidth / kTileFeatures;
inline constexpr int kOutputProjectionTiles = kHiddenSize / kTileFeatures;
inline constexpr int kGateUpTiles = kIntermediateSize / (kTileFeatures / 2);
inline constexpr int kDownTiles = kHiddenSize / kTileFeatures;
inline constexpr unsigned kComputeBarrierId = 1;

// Stamp slots a CTA records in a QWEN3_TTS_PREDICTOR_STAMPS build: launch
// points, then four points a phase, then lists indexed by item or chunk.
inline constexpr int kStampCtas = 160;
inline constexpr int kStampEntry = 0;
inline constexpr int kStampPrimaryWaited = 1;
inline constexpr int kStampExit = 2;
inline constexpr int kStampPhaseBase = 4;
inline constexpr int kStampPhaseReached = 0;
inline constexpr int kStampPhaseSignaled = 1;
inline constexpr int kStampPhaseWaited = 2;
inline constexpr int kStampPhaseStaged = 3;
inline constexpr int kStampPhasePoints = 4;
inline constexpr int kStampListLength = 16;
inline constexpr int kStampItemEnd = 24;
inline constexpr int kStampChunkWait = kStampItemEnd + kStampListLength;
inline constexpr int kStampChunkReady = kStampChunkWait + kStampListLength;
inline constexpr int kStampChunkIssue = kStampChunkReady + kStampListLength;
inline constexpr int kStampSlots = kStampChunkIssue + kStampListLength;

struct LayerParams {
  const bf16_t *input;
  bf16_t *output;
  const bf16_t *qkv_chunks;
  const bf16_t *output_projection_chunks;
  const bf16_t *gate_up_chunks;
  const bf16_t *down_chunks;
  const bf16_t *input_norm_weight;
  const bf16_t *post_norm_weight;
  const bf16_t *query_norm_weight;
  const bf16_t *key_norm_weight;
  const float *rope_cos_sin;
  bf16_t *key_cache;
  bf16_t *value_cache;
  int64_t cache_row_stride;
  bf16_t *qkv_scratch;
  bf16_t *attention_scratch;
  bf16_t *middle_scratch;
  bf16_t *activation_scratch;
  uint32_t *phase_events;
  const int16_t *schedule;
  const int32_t *schedule_lengths;
  int32_t schedule_width;
  int32_t phase_consumers[kPhaseEvents];
  int32_t rows;
  int32_t position;
  float norm_epsilon;
  float attention_scale;
};

SGL_DEVICE int chunks_per_tile(int phase) {
  if (phase == kPhaseOutputProjection)
    return kAttentionWidth / kChunkDepth;
  if (phase == kPhaseDown)
    return kIntermediateSize / kChunkDepth;
  if (phase == kPhaseAttention)
    return 0;
  return 1;
}

SGL_DEVICE int operand_row_bytes(int phase) {
  if (phase == kPhaseOutputProjection)
    return kAttentionWidth * 2 + kOperandRowPadding;
  if (phase == kPhaseDown)
    return kIntermediateSize * 2 + kOperandRowPadding;
  return kHiddenSize * 2 + kOperandRowPadding;
}

SGL_DEVICE const bf16_t *chunk_pointer(const LayerParams &params, int phase,
                                       int tile, int chunk) {
  if (phase == kPhaseQkv)
    return params.qkv_chunks + static_cast<int64_t>(tile) * kChunkElements;
  if (phase == kPhaseOutputProjection) {
    return params.output_projection_chunks +
           static_cast<int64_t>(tile * 2 + chunk) * kChunkElements;
  }
  if (phase == kPhaseGateUp)
    return params.gate_up_chunks + static_cast<int64_t>(tile) * kChunkElements;
  return params.down_chunks +
         static_cast<int64_t>(tile * 3 + chunk) * kChunkElements;
}

SGL_DEVICE int phase_producers(const LayerParams &params, int phase) {
  if (phase == kPhaseQkv)
    return kQkvTiles;
  if (phase == kPhaseAttention)
    return params.rows * kKeyValueHeads;
  if (phase == kPhaseOutputProjection)
    return kOutputProjectionTiles;
  return kGateUpTiles;
}

SGL_DEVICE uint32_t shared_address(const void *pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

SGL_DEVICE void compute_barrier() {
  asm volatile("bar.sync %0, %1;" ::"r"(kComputeBarrierId), "r"(kComputeThreads)
               : "memory");
}

SGL_DEVICE void load_matrix_x4(uint32_t address, uint32_t (&registers)[4]) {
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]),
        "=r"(registers[3])
      : "r"(address));
}

SGL_DEVICE void mma_bf16(float (&accumulator)[4], const uint32_t (&weights)[4],
                         uint32_t tokens_low, uint32_t tokens_high) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, "
               "%2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(accumulator[0]), "+f"(accumulator[1]),
                 "+f"(accumulator[2]), "+f"(accumulator[3])
               : "r"(weights[0]), "r"(weights[1]), "r"(weights[2]),
                 "r"(weights[3]), "r"(tokens_low), "r"(tokens_high));
}

SGL_DEVICE void bulk_copy_to_shared(void *shared_destination,
                                    const void *global_source, uint32_t bytes,
                                    uint64_t *barrier) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];" ::"r"(shared_address(shared_destination)),
      "l"(global_source), "r"(bytes), "r"(shared_address(barrier))
      : "memory");
}

SGL_DEVICE float round_bf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

#ifdef QWEN3_TTS_PREDICTOR_STAMPS
// note (ratish): Nsight Compute cannot replay a launch whose CTAs wait on each
// other, so a debug build records the global timer and the SM clock instead.
__device__ int64_t predictor_stamps[kStampCtas * kStampSlots * 2];

SGL_DEVICE void record_stamp(int slot) {
  if (blockIdx.x >= kStampCtas)
    return;
  uint64_t global_time;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(global_time));
  int64_t *stamp = predictor_stamps + (blockIdx.x * kStampSlots + slot) * 2;
  stamp[0] = static_cast<int64_t>(global_time);
  stamp[1] = clock64();
}
#else
SGL_DEVICE void record_stamp(int) {}
#endif

SGL_DEVICE void record_listed_stamp(int list, int index) {
  if (index < kStampListLength)
    record_stamp(list + index);
}

SGL_DEVICE void record_phase_stamp(int phase, int point) {
  record_stamp(kStampPhaseBase + phase * kStampPhasePoints + point);
}

// Event::arrive is a red.release.gpu: one fence and add, as CUTLASS arrive_inc
// does, after the compute barrier that orders every compute thread's stores.
SGL_DEVICE void signal_phase(const LayerParams &params, int phase,
                             uint32_t items) {
  reinterpret_cast<device::atomic::Event *>(params.phase_events + phase)
      ->arrive(items);
}

SGL_DEVICE void wait_phase(const LayerParams &params, int phase) {
  reinterpret_cast<device::atomic::Event *>(params.phase_events + phase)
      ->wait_multi(phase_producers(params, phase),
                   params.phase_consumers[phase]);
}

// Norm weights and the rope row never depend on the previous kernel, so they
// are staged before the PDL wait.
SGL_DEVICE void stage_constants(const LayerParams &params, uint8_t *constants) {
  for (int offset = threadIdx.x * 16; offset < kConstantBytes;
       offset += kComputeThreads * 16) {
    const uint8_t *source;
    if (offset < kPostNormOffset) {
      source = reinterpret_cast<const uint8_t *>(params.input_norm_weight) +
               (offset - kInputNormOffset);
    } else if (offset < kQueryNormOffset) {
      source = reinterpret_cast<const uint8_t *>(params.post_norm_weight) +
               (offset - kPostNormOffset);
    } else if (offset < kKeyNormOffset) {
      source = reinterpret_cast<const uint8_t *>(params.query_norm_weight) +
               (offset - kQueryNormOffset);
    } else if (offset < kRopeOffset) {
      source = reinterpret_cast<const uint8_t *>(params.key_norm_weight) +
               (offset - kKeyNormOffset);
    } else {
      source = reinterpret_cast<const uint8_t *>(params.rope_cos_sin +
                                                 params.position * kHeadDim) +
               (offset - kRopeOffset);
    }
    *reinterpret_cast<uint4 *>(constants + offset) =
        __ldg(reinterpret_cast<const uint4 *>(source));
  }
}

// Stages the phase's input rows as the MMA B operand, one row a warp, with all
// of a warp's loads issued before any is used so they cost one round trip. A
// normalized phase multiplies by the norm weight and rounds to bf16 before the
// product and keeps each row's rstd for the epilogue, the order of the fused
// Triton layer. Rows past params.rows keep stale values: their columns are
// never stored.
template <int kDepth, bool kNormalized>
SGL_DEVICE void stage_operand(const LayerParams &params, const bf16_t *source,
                              const bf16_t *norm_weight, uint8_t *operand,
                              float *row_scales, int warp_index,
                              int lane_index) {
  constexpr int kVectorsPerLane = kDepth / 8 / 32;
  constexpr int kRowBytes = kDepth * 2 + kOperandRowPadding;
  uint4 packed[kRowsPerWarp][kVectorsPerLane];
#pragma unroll
  for (int row_slot = 0; row_slot < kRowsPerWarp; ++row_slot) {
    const int token = warp_index + row_slot * kComputeWarps;
    if (token < params.rows) {
      const uint4 *row = reinterpret_cast<const uint4 *>(
          source + static_cast<int64_t>(token) * kDepth);
#pragma unroll
      for (int i = 0; i < kVectorsPerLane; ++i) {
        packed[row_slot][i] = __ldcg(row + lane_index + 32 * i);
      }
    }
  }
#pragma unroll
  for (int row_slot = 0; row_slot < kRowsPerWarp; ++row_slot) {
    const int token = warp_index + row_slot * kComputeWarps;
    if (token >= params.rows)
      continue;
    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVectorsPerLane; ++i) {
      const int vector = lane_index + 32 * i;
      if constexpr (kNormalized) {
        const uint4 weight_packed =
            *reinterpret_cast<const uint4 *>(norm_weight + vector * 8);
        bf16_t *values = reinterpret_cast<bf16_t *>(&packed[row_slot][i]);
        const bf16_t *weights =
            reinterpret_cast<const bf16_t *>(&weight_packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float value = __bfloat162float(values[j]);
          sum_of_squares += value * value;
          values[j] = __float2bfloat16(value * __bfloat162float(weights[j]));
        }
      }
      *reinterpret_cast<uint4 *>(operand + token * kRowBytes + vector * 16) =
          packed[row_slot][i];
    }
    if constexpr (kNormalized) {
      sum_of_squares = device::warp::reduce_sum(sum_of_squares);
      if (lane_index == 0)
        row_scales[token] =
            rsqrtf(sum_of_squares / kDepth + params.norm_epsilon);
    }
  }
}

SGL_DEVICE void stage_phase_operand(const LayerParams &params, int phase,
                                    const uint8_t *constants, uint8_t *operand,
                                    float *row_scales, int warp_index,
                                    int lane_index) {
  const auto *input_norm =
      reinterpret_cast<const bf16_t *>(constants + kInputNormOffset);
  const auto *post_norm =
      reinterpret_cast<const bf16_t *>(constants + kPostNormOffset);
  if (phase == kPhaseQkv) {
    stage_operand<kHiddenSize, true>(params, params.input, input_norm, operand,
                                     row_scales, warp_index, lane_index);
  } else if (phase == kPhaseOutputProjection) {
    stage_operand<kAttentionWidth, false>(params, params.attention_scratch,
                                          nullptr, operand, row_scales,
                                          warp_index, lane_index);
  } else if (phase == kPhaseGateUp) {
    stage_operand<kHiddenSize, true>(params, params.middle_scratch, post_norm,
                                     operand, row_scales, warp_index,
                                     lane_index);
  } else if (phase == kPhaseDown) {
    stage_operand<kIntermediateSize, false>(params, params.activation_scratch,
                                            nullptr, operand, row_scales,
                                            warp_index, lane_index);
  }
}

// Multiplies one 16-feature weight tile, streamed through the ring, by the
// staged rows. Warp w takes k-steps [8w, 8w + 8) of every chunk and alternates
// two accumulators, so consecutive MMAs do not wait on each other; the warps'
// partial sums land in `partials` as [warp][token tile][feature 16][token 8].
SGL_DEVICE void multiply_tile(int chunks, int row_bytes, int token_tiles,
                              uint8_t *ring, const uint8_t *operand,
                              float *partials, uint64_t *full_barriers,
                              uint64_t *empty_barriers, int &stage,
                              uint32_t &full_parity, int first_chunk,
                              int warp_index, int lane_index) {
  float accumulator[2][kMaxTokenTiles][4] = {};
  // ldmatrix lane l gives the address of token row l % 8 of matrix l / 8:
  // matrices 0 and 1 are the two 8-value halves of a k-step, 2 and 3 the next.
  const uint32_t operand_address = shared_address(operand) +
                                   (lane_index % 8) * row_bytes +
                                   (lane_index / 8) * 16;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    if (threadIdx.x == 0)
      record_listed_stamp(kStampChunkWait, first_chunk + chunk);
    device::ptx::mbar_wait_parity(full_barriers + stage, full_parity);
    if (threadIdx.x == 0)
      record_listed_stamp(kStampChunkReady, first_chunk + chunk);
    const uint32_t chunk_address =
        shared_address(ring + stage * kChunkBytes) + lane_index * 16;
#pragma unroll
    for (int pair = 0; pair < kKStepsPerWarp / 2; ++pair) {
      const int kstep = warp_index * kKStepsPerWarp + pair * 2;
      const int operand_kstep = chunk * kKStepsPerChunk + kstep;
      uint32_t first_weights[4];
      uint32_t second_weights[4];
      load_matrix_x4(chunk_address + kstep * kKStepBytes, first_weights);
      load_matrix_x4(chunk_address + (kstep + 1) * kKStepBytes, second_weights);
#pragma unroll
      for (int token_tile = 0; token_tile < kMaxTokenTiles; ++token_tile) {
        if (token_tile < token_tiles) {
          uint32_t tokens[4];
          load_matrix_x4(operand_address +
                             token_tile * kTokenTileRows * row_bytes +
                             operand_kstep * kKStepDepth * 2,
                         tokens);
          mma_bf16(accumulator[0][token_tile], first_weights, tokens[0],
                   tokens[1]);
          mma_bf16(accumulator[1][token_tile], second_weights, tokens[2],
                   tokens[3]);
        }
      }
    }
    __syncwarp();
    if (lane_index == 0)
      device::ptx::mbar_arrive(empty_barriers + stage);
    stage += 1;
    if (stage == kRingStages) {
      stage = 0;
      full_parity ^= 1u;
    }
  }
  // C fragment of m16n8k16: features lane / 4 and lane / 4 + 8, tokens 2 (lane
  // % 4) and the next.
  const int feature = lane_index / 4;
  const int token = 2 * (lane_index % 4);
#pragma unroll
  for (int token_tile = 0; token_tile < kMaxTokenTiles; ++token_tile) {
    const float(&even)[4] = accumulator[0][token_tile];
    const float(&odd)[4] = accumulator[1][token_tile];
    float *block = partials + ((warp_index * kMaxTokenTiles + token_tile) *
                               kTileFeatures) *
                                  kTokenTileRows;
    block[feature * kTokenTileRows + token] = even[0] + odd[0];
    block[feature * kTokenTileRows + token + 1] = even[1] + odd[1];
    block[(feature + 8) * kTokenTileRows + token] = even[2] + odd[2];
    block[(feature + 8) * kTokenTileRows + token + 1] = even[3] + odd[3];
  }
}

SGL_DEVICE float reduce_partials(const float *partials, int token_tile,
                                 int feature, int token) {
  float total = 0.0f;
#pragma unroll
  for (int warp = 0; warp < kComputeWarps; ++warp) {
    total += partials[((warp * kMaxTokenTiles + token_tile) * kTileFeatures +
                       feature) *
                          kTokenTileRows +
                      token];
  }
  return total;
}

// In a 16-feature tile, compute thread t finishes token t % 8 of token tile
// t / 128 at feature (t / 8) % 16. Threads whose token is past params.rows
// finish nothing.
SGL_DEVICE int epilogue_token(int thread) {
  return (thread / (kTileFeatures * kTokenTileRows)) * kTokenTileRows +
         thread % kTokenTileRows;
}

SGL_DEVICE int epilogue_feature(int thread) {
  return (thread / kTokenTileRows) % kTileFeatures;
}

// The residual this thread's epilogue adds, read before the tile's MMAs so its
// latency hides behind them.
SGL_DEVICE float load_residual(const LayerParams &params, int phase, int tile) {
  const int token = epilogue_token(threadIdx.x);
  if (token >= params.rows)
    return 0.0f;
  const int64_t hidden_index = static_cast<int64_t>(token) * kHiddenSize +
                               tile * kTileFeatures +
                               epilogue_feature(threadIdx.x);
  if (phase == kPhaseOutputProjection)
    return __bfloat162float(params.input[hidden_index]);
  if (phase == kPhaseDown)
    return __bfloat162float(__ldcg(params.middle_scratch + hidden_index));
  return 0.0f;
}

SGL_DEVICE void finish_tile(const LayerParams &params, int phase, int tile,
                            const float *partials, const float *row_scales,
                            float residual) {
  const int thread = threadIdx.x;
  if (phase == kPhaseGateUp) {
    // Rows 0-7 of a gate/up tile are gate features, rows 8-15 the up features
    // of the same index.
    constexpr int kHalfFeatures = kTileFeatures / 2;
    const int token_tile = thread / (kHalfFeatures * kTokenTileRows);
    const int token_in_tile = thread % kTokenTileRows;
    const int token = token_tile * kTokenTileRows + token_in_tile;
    if (token >= params.rows)
      return;
    const int feature = (thread / kTokenTileRows) % kHalfFeatures;
    const float scale = row_scales[token];
    const float gate = round_bf16(
        reduce_partials(partials, token_tile, feature, token_in_tile) * scale);
    const float up =
        round_bf16(reduce_partials(partials, token_tile,
                                   feature + kHalfFeatures, token_in_tile) *
                   scale);
    params.activation_scratch[static_cast<int64_t>(token) * kIntermediateSize +
                              tile * kHalfFeatures + feature] =
        __float2bfloat16(gate / (1.0f + expf(-gate)) * up);
    return;
  }
  const int token = epilogue_token(thread);
  if (token >= params.rows)
    return;
  const int feature = epilogue_feature(thread);
  const float total = reduce_partials(partials, token / kTokenTileRows, feature,
                                      token % kTokenTileRows);
  const int output_feature = tile * kTileFeatures + feature;
  const int64_t hidden_index =
      static_cast<int64_t>(token) * kHiddenSize + output_feature;
  if (phase == kPhaseQkv) {
    params
        .qkv_scratch[static_cast<int64_t>(token) * kQkvWidth + output_feature] =
        __float2bfloat16(total * row_scales[token]);
  } else if (phase == kPhaseOutputProjection) {
    params.middle_scratch[hidden_index] =
        __float2bfloat16(round_bf16(total) + residual);
  } else {
    params.output[hidden_index] =
        __float2bfloat16(round_bf16(total) + residual);
  }
}

// RMS norm over one head held four values a lane (dims lane + 32 i), then neox
// rope at the cache position, each step rounded to bf16 where the fused Triton
// layer rounds.
SGL_DEVICE void norm_and_rotate_head(float (&values)[4],
                                     const bf16_t *norm_weight,
                                     const float *angles, float norm_epsilon,
                                     int lane_index) {
  float sum_of_squares = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sum_of_squares += values[i] * values[i];
  }
  const float scale = rsqrtf(
      device::warp::reduce_sum(sum_of_squares) / kHeadDim + norm_epsilon);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = round_bf16(values[i] * scale *
                           __bfloat162float(norm_weight[lane_index + 32 * i]));
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int dim = lane_index + 32 * i;
    const float cosine = angles[dim];
    const float sine = angles[kHalfHeadDim + dim];
    const float low = values[i];
    const float high = values[i + 2];
    values[i] = round_bf16(low * cosine - high * sine);
    values[i + 2] = round_bf16(low * sine + high * cosine);
  }
}

// Global source of attention vector `vector` (16 bytes): the cached keys of
// slots below the position, then their values, then the query heads 2g and
// 2g + 1, the key head g and the value head g of the new token.
SGL_DEVICE const uint4 *attention_source(const LayerParams &params,
                                         const bf16_t *qkv_row,
                                         int64_t cache_base, int group,
                                         int cached_vectors, int vector) {
  constexpr int64_t kSlotStride = kKeyValueHeads * kHeadDim;
  if (vector < 2 * cached_vectors) {
    const bool is_key = vector < cached_vectors;
    const int cached = is_key ? vector : vector - cached_vectors;
    const bf16_t *cache = is_key ? params.key_cache : params.value_cache;
    return reinterpret_cast<const uint4 *>(
        cache + cache_base + (cached / kHeadVectors) * kSlotStride +
        (cached % kHeadVectors) * 8);
  }
  const int head_vector = vector - 2 * cached_vectors;
  const int head = head_vector / kHeadVectors;
  int head_offset;
  if (head < 2) {
    head_offset = (group * 2 + head) * kHeadDim;
  } else if (head == 2) {
    head_offset = kKeyOffset + group * kHeadDim;
  } else {
    head_offset = kValueOffset + group * kHeadDim;
  }
  return reinterpret_cast<const uint4 *>(qkv_row + head_offset +
                                         (head_vector % kHeadVectors) * 8);
}

SGL_DEVICE int attention_offset(int cached_vectors, int vector) {
  if (vector < cached_vectors) {
    return kAttentionKeysOffset + (vector / kHeadVectors) * kHeadRowBytes +
           (vector % kHeadVectors) * 16;
  }
  if (vector < 2 * cached_vectors) {
    const int cached = vector - cached_vectors;
    return kAttentionValuesOffset + (cached / kHeadVectors) * kHeadRowBytes +
           (cached % kHeadVectors) * 16;
  }
  return kAttentionHeadsOffset + (vector - 2 * cached_vectors) * 16;
}

// q . k over one head, q fp32 in shared memory and k one bf16 row, in four
// partial sums.
SGL_DEVICE float dot_head(const float *query, const uint8_t *key_row) {
  float partial[4] = {};
#pragma unroll
  for (int vector = 0; vector < kHeadVectors; ++vector) {
    const uint4 packed =
        *reinterpret_cast<const uint4 *>(key_row + vector * 16);
    const bf16_t *keys = reinterpret_cast<const bf16_t *>(&packed);
    const float4 low = *reinterpret_cast<const float4 *>(query + vector * 8);
    const float4 high =
        *reinterpret_cast<const float4 *>(query + vector * 8 + 4);
    partial[0] += low.x * __bfloat162float(keys[0]);
    partial[1] += low.y * __bfloat162float(keys[1]);
    partial[2] += low.z * __bfloat162float(keys[2]);
    partial[3] += low.w * __bfloat162float(keys[3]);
    partial[0] += high.x * __bfloat162float(keys[4]);
    partial[1] += high.y * __bfloat162float(keys[5]);
    partial[2] += high.z * __bfloat162float(keys[6]);
    partial[3] += high.w * __bfloat162float(keys[7]);
  }
  return (partial[0] + partial[1]) + (partial[2] + partial[3]);
}

// One (token, kv group) of attention. All compute threads load the group's
// cached keys and values and the new token's heads into shared memory at once;
// then warps 0 and 1 each take one query head: they rotate it and the new key,
// score every slot with one lane a slot, and sum the values with four head
// dims a lane. Warp 0 writes the new key and warp 1 the new value into the
// caches at the position. The caller's barrier ends the item.
SGL_DEVICE void attend(const LayerParams &params, int item, uint8_t *workspace,
                       const uint8_t *constants, int warp_index,
                       int lane_index) {
  const int token = item / kKeyValueHeads;
  const int group = item % kKeyValueHeads;
  const int position = params.position;
  const bf16_t *qkv_row =
      params.qkv_scratch + static_cast<int64_t>(token) * kQkvWidth;
  const int64_t cache_base =
      static_cast<int64_t>(token) * params.cache_row_stride + group * kHeadDim;
  const int cached_vectors = position * kHeadVectors;
  const int vector_count = 2 * cached_vectors + kAttentionHeads * kHeadVectors;
  uint4 packed[kAttentionVectorsPerThread];
#pragma unroll
  for (int i = 0; i < kAttentionVectorsPerThread; ++i) {
    const int vector = threadIdx.x + i * kComputeThreads;
    if (vector < vector_count) {
      packed[i] = __ldcg(attention_source(params, qkv_row, cache_base, group,
                                          cached_vectors, vector));
    }
  }
#pragma unroll
  for (int i = 0; i < kAttentionVectorsPerThread; ++i) {
    const int vector = threadIdx.x + i * kComputeThreads;
    if (vector < vector_count) {
      *reinterpret_cast<uint4 *>(
          workspace + attention_offset(cached_vectors, vector)) = packed[i];
    }
  }
  compute_barrier();
  if (warp_index >= 2)
    return;

  const auto *heads =
      reinterpret_cast<const bf16_t *>(workspace + kAttentionHeadsOffset);
  const auto *angles = reinterpret_cast<const float *>(constants + kRopeOffset);
  float query[4];
  float key[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int dim = lane_index + 32 * i;
    query[i] = __bfloat162float(heads[warp_index * kHeadDim + dim]);
    key[i] = __bfloat162float(heads[2 * kHeadDim + dim]);
  }
  norm_and_rotate_head(
      query, reinterpret_cast<const bf16_t *>(constants + kQueryNormOffset),
      angles, params.norm_epsilon, lane_index);
  norm_and_rotate_head(
      key, reinterpret_cast<const bf16_t *>(constants + kKeyNormOffset), angles,
      params.norm_epsilon, lane_index);
  uint8_t *rotated_key_row =
      workspace + kAttentionRotatedKeysOffset + warp_index * kHeadRowBytes;
  auto *rotated_key = reinterpret_cast<bf16_t *>(rotated_key_row);
  float *rotated_query =
      reinterpret_cast<float *>(workspace + kAttentionQueriesOffset) +
      warp_index * kHeadDim;
  const int64_t current_slot =
      cache_base + static_cast<int64_t>(position) * kKeyValueHeads * kHeadDim;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int dim = lane_index + 32 * i;
    rotated_key[dim] = __float2bfloat16(key[i]);
    rotated_query[dim] = query[i];
    if (warp_index == 0) {
      params.key_cache[current_slot + dim] = __float2bfloat16(key[i]);
    } else {
      params.value_cache[current_slot + dim] = heads[3 * kHeadDim + dim];
    }
  }
  __syncwarp();

  float score = -INFINITY;
  if (lane_index <= position) {
    const uint8_t *key_row =
        lane_index < position
            ? workspace + kAttentionKeysOffset + lane_index * kHeadRowBytes
            : rotated_key_row;
    score = dot_head(rotated_query, key_row) * params.attention_scale;
  }
  const float best = device::warp::reduce_max(score);
  const float weight = lane_index <= position ? expf(score - best) : 0.0f;
  const float weight_sum = device::warp::reduce_sum(weight);
  const float rounded_weight = round_bf16(weight);
  float attended[4] = {};
  for (int slot = 0; slot <= position; ++slot) {
    const float slot_weight = __shfl_sync(0xffffffffu, rounded_weight, slot);
    const bf16_t *value_row =
        slot < position
            ? reinterpret_cast<const bf16_t *>(
                  workspace + kAttentionValuesOffset + slot * kHeadRowBytes)
            : heads + 3 * kHeadDim;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      attended[i] +=
          slot_weight * __bfloat162float(value_row[lane_index + 32 * i]);
    }
  }
  const int head = group * 2 + warp_index;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    params.attention_scratch[static_cast<int64_t>(token) * kAttentionWidth +
                             head * kHeadDim + lane_index + 32 * i] =
        __float2bfloat16(attended[i] / weight_sum);
  }
}

template <bool kUsePDL>
__global__ __launch_bounds__(kThreads, 1) void predictor_layer_kernel(
    const __grid_constant__ LayerParams params) {
  extern __shared__ __align__(128) uint8_t shared_memory[];
  uint8_t *ring = shared_memory;
  uint8_t *operand = ring + kRingStages * kChunkBytes;
  float *partials = reinterpret_cast<float *>(operand + kOperandBytes);
  uint8_t *constants =
      reinterpret_cast<uint8_t *>(partials + kPartialBuffers * kPartialFloats);
  float *row_scales = reinterpret_cast<float *>(constants + kConstantBytes);
  uint64_t *full_barriers = reinterpret_cast<uint64_t *>(row_scales + kMaxRows);
  uint64_t *empty_barriers = full_barriers + kRingStages;

  const int warp_index = threadIdx.x / 32;
  const int lane_index = threadIdx.x % 32;
  const int16_t *schedule = params.schedule + static_cast<int64_t>(blockIdx.x) *
                                                  params.schedule_width;
  const int schedule_length = params.schedule_lengths[blockIdx.x];

  if (threadIdx.x == 0) {
    for (int stage = 0; stage < kRingStages; ++stage) {
      device::ptx::mbar_init(full_barriers + stage, 1);
      device::ptx::mbar_init(empty_barriers + stage, kComputeWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0)
    record_stamp(kStampEntry);

  if (warp_index == kComputeWarps) {
    // Weights never depend on the previous kernel, so the producer streams
    // before the PDL wait.
    if (lane_index == 0) {
      int stage = 0;
      uint32_t empty_parity = 1u;
      int issued_chunks = 0;
      for (int index = 0; index < schedule_length; ++index) {
        const int phase = schedule[index] >> kScheduleTileBits;
        const int tile = schedule[index] & kScheduleTileMask;
        for (int chunk = 0; chunk < chunks_per_tile(phase); ++chunk) {
          device::ptx::mbar_wait_parity(empty_barriers + stage, empty_parity);
          record_listed_stamp(kStampChunkIssue, issued_chunks);
          issued_chunks += 1;
          device::ptx::mbar_arrive_expect_tx(full_barriers + stage,
                                             kChunkBytes);
          bulk_copy_to_shared(ring + stage * kChunkBytes,
                              chunk_pointer(params, phase, tile, chunk),
                              kChunkBytes, full_barriers + stage);
          stage += 1;
          if (stage == kRingStages) {
            stage = 0;
            empty_parity ^= 1u;
          }
        }
      }
    }
  } else {
    stage_constants(params, constants);
    device::PDLWaitPrimary<kUsePDL>();
    if (threadIdx.x == 0)
      record_stamp(kStampPrimaryWaited);
    const int token_tiles = (params.rows + kTokenTileRows - 1) / kTokenTileRows;
    int stage = 0;
    uint32_t full_parity = 0u;
    int current_phase = -1;
    uint32_t phase_items = 0u;
    int consumed_chunks = 0;
    int partial_buffer = 0;
    for (int index = 0; index < schedule_length; ++index) {
      const int phase = schedule[index] >> kScheduleTileBits;
      const int tile = schedule[index] & kScheduleTileMask;
      if (phase != current_phase) {
        if (threadIdx.x == 0)
          record_phase_stamp(phase, kStampPhaseReached);
        // Orders the previous phase's stores (the constants on the first
        // pass) before the signal and the staging below.
        compute_barrier();
        if (threadIdx.x == 0) {
          if (current_phase >= 0) {
            signal_phase(params, current_phase, phase_items);
            record_phase_stamp(current_phase, kStampPhaseSignaled);
          }
          if (phase > kPhaseQkv)
            wait_phase(params, phase - 1);
        }
        if (phase > kPhaseQkv) {
          compute_barrier();
          if (threadIdx.x == 0)
            record_phase_stamp(phase, kStampPhaseWaited);
        }
        stage_phase_operand(params, phase, constants, operand, row_scales,
                            warp_index, lane_index);
        compute_barrier();
        if (threadIdx.x == 0)
          record_phase_stamp(phase, kStampPhaseStaged);
        current_phase = phase;
        phase_items = 0u;
      }
      if (phase == kPhaseAttention) {
        attend(params, tile, operand, constants, warp_index, lane_index);
        compute_barrier();
      } else {
        const float residual = load_residual(params, phase, tile);
        float *tile_partials = partials + partial_buffer * kPartialFloats;
        multiply_tile(chunks_per_tile(phase), operand_row_bytes(phase),
                      token_tiles, ring, operand, tile_partials, full_barriers,
                      empty_barriers, stage, full_parity, consumed_chunks,
                      warp_index, lane_index);
        consumed_chunks += chunks_per_tile(phase);
        compute_barrier();
        finish_tile(params, phase, tile, tile_partials, row_scales, residual);
        // note (ratish): alternating the partial buffers lets the next tile's
        // MMAs start without a barrier; its own barrier comes after every
        // thread has finished reading this buffer.
        partial_buffer ^= 1;
      }
      if (threadIdx.x == 0)
        record_listed_stamp(kStampItemEnd, index);
      phase_items += 1u;
    }
    if (current_phase >= 0 && current_phase < kPhaseDown) {
      compute_barrier();
      if (threadIdx.x == 0) {
        signal_phase(params, current_phase, phase_items);
        record_phase_stamp(current_phase, kStampPhaseSignaled);
      }
    }
    if (threadIdx.x == 0)
      record_stamp(kStampExit);
  }
  // note (ratish): triggered only at exit, so a dependent grid cannot take an
  // SM that one of this grid's waiting producers still needs.
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL> struct PredictorLayer {
  static void
  run(tvm::ffi::TensorView input, tvm::ffi::TensorView output,
      tvm::ffi::TensorView qkv_chunks,
      tvm::ffi::TensorView output_projection_chunks,
      tvm::ffi::TensorView gate_up_chunks, tvm::ffi::TensorView down_chunks,
      tvm::ffi::TensorView input_norm_weight,
      tvm::ffi::TensorView post_norm_weight,
      tvm::ffi::TensorView query_norm_weight,
      tvm::ffi::TensorView key_norm_weight, tvm::ffi::TensorView rope_cos_sin,
      tvm::ffi::TensorView key_cache, tvm::ffi::TensorView value_cache,
      tvm::ffi::TensorView qkv_scratch, tvm::ffi::TensorView attention_scratch,
      tvm::ffi::TensorView middle_scratch,
      tvm::ffi::TensorView activation_scratch,
      tvm::ffi::TensorView phase_events, tvm::ffi::TensorView schedule,
      tvm::ffi::TensorView schedule_lengths,
      tvm::ffi::TensorView phase_consumers, int64_t position,
      double norm_epsilon) {
    using namespace host;

    auto rows = SymbolicSize{"rows"};
    auto cache_rows = SymbolicSize{"cache_rows"};
    auto cache_row_stride = SymbolicSize{"cache_row_stride"};
    auto ctas = SymbolicSize{"ctas"};
    auto schedule_width = SymbolicSize{"schedule_width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    // The kernel reads the input, the norm weights, the rope row and the
    // caches in 16-byte vectors.
    TensorMatcher({rows, kHiddenSize})
        .with_dtype<bf16_t>()
        .ensure_alignment(16)
        .with_device(device)
        .verify(input);
    TensorMatcher({rows, kHiddenSize})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output);
    TensorMatcher({kQkvTiles, kChunkElements})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(qkv_chunks);
    TensorMatcher({kOutputProjectionTiles * 2, kChunkElements})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(output_projection_chunks);
    TensorMatcher({kGateUpTiles, kChunkElements})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(gate_up_chunks);
    TensorMatcher({kDownTiles * 3, kChunkElements})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(down_chunks);
    TensorMatcher({kHiddenSize})
        .with_dtype<bf16_t>()
        .ensure_alignment(16)
        .with_device(device)
        .verify(input_norm_weight)
        .verify(post_norm_weight);
    TensorMatcher({kHeadDim})
        .with_dtype<bf16_t>()
        .ensure_alignment(16)
        .with_device(device)
        .verify(query_norm_weight)
        .verify(key_norm_weight);
    TensorMatcher({kCacheSlots, kHeadDim})
        .with_dtype<float>()
        .ensure_alignment(16)
        .with_device(device)
        .verify(rope_cos_sin);
    TensorMatcher({cache_rows, kCacheSlots, kKeyValueHeads, kHeadDim})
        .with_strides(
            {cache_row_stride, kKeyValueHeads * kHeadDim, kHeadDim, 1})
        .with_dtype<bf16_t>()
        .ensure_alignment(16)
        .with_device(device)
        .verify(key_cache)
        .verify(value_cache);
    TensorMatcher({kMaxRows, kQkvWidth})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(qkv_scratch);
    TensorMatcher({kMaxRows, kAttentionWidth})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(attention_scratch);
    TensorMatcher({kMaxRows, kHiddenSize})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(middle_scratch);
    TensorMatcher({kMaxRows, kIntermediateSize})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(activation_scratch);
    TensorMatcher({kPhaseEvents})
        .with_dtype<uint32_t>()
        .with_device(device)
        .verify(phase_events);
    TensorMatcher({ctas, schedule_width})
        .with_dtype<int16_t>()
        .with_device(device)
        .verify(schedule);
    TensorMatcher({ctas}).with_dtype<int32_t>().with_device(device).verify(
        schedule_lengths);
    TensorMatcher({kPhaseEvents})
        .with_dtype<int32_t>()
        .with_device<kDLCPU>()
        .verify(phase_consumers);

    const int row_count = static_cast<int>(rows.unwrap());
    RuntimeCheck(row_count >= 1 && row_count <= kMaxRows,
                 "predictor layer rows must be in [1, 16], got ", row_count);
    RuntimeCheck(row_count <= cache_rows.unwrap(),
                 "the caches hold fewer rows than the input");
    RuntimeCheck(position >= 0 && position < kCacheSlots,
                 "position must be a cache slot, got ", position);

    LayerParams params{};
    params.input = static_cast<const bf16_t *>(input.data_ptr());
    params.output = static_cast<bf16_t *>(output.data_ptr());
    params.qkv_chunks = static_cast<const bf16_t *>(qkv_chunks.data_ptr());
    params.output_projection_chunks =
        static_cast<const bf16_t *>(output_projection_chunks.data_ptr());
    params.gate_up_chunks =
        static_cast<const bf16_t *>(gate_up_chunks.data_ptr());
    params.down_chunks = static_cast<const bf16_t *>(down_chunks.data_ptr());
    params.input_norm_weight =
        static_cast<const bf16_t *>(input_norm_weight.data_ptr());
    params.post_norm_weight =
        static_cast<const bf16_t *>(post_norm_weight.data_ptr());
    params.query_norm_weight =
        static_cast<const bf16_t *>(query_norm_weight.data_ptr());
    params.key_norm_weight =
        static_cast<const bf16_t *>(key_norm_weight.data_ptr());
    params.rope_cos_sin = static_cast<const float *>(rope_cos_sin.data_ptr());
    params.key_cache = static_cast<bf16_t *>(key_cache.data_ptr());
    params.value_cache = static_cast<bf16_t *>(value_cache.data_ptr());
    params.cache_row_stride = cache_row_stride.unwrap();
    params.qkv_scratch = static_cast<bf16_t *>(qkv_scratch.data_ptr());
    params.attention_scratch =
        static_cast<bf16_t *>(attention_scratch.data_ptr());
    params.middle_scratch = static_cast<bf16_t *>(middle_scratch.data_ptr());
    params.activation_scratch =
        static_cast<bf16_t *>(activation_scratch.data_ptr());
    params.phase_events = static_cast<uint32_t *>(phase_events.data_ptr());
    params.schedule = static_cast<const int16_t *>(schedule.data_ptr());
    params.schedule_lengths =
        static_cast<const int32_t *>(schedule_lengths.data_ptr());
    params.schedule_width = static_cast<int32_t>(schedule_width.unwrap());
    const auto *consumers =
        static_cast<const int32_t *>(phase_consumers.data_ptr());
    for (int phase = 0; phase < kPhaseEvents; ++phase) {
      params.phase_consumers[phase] = consumers[phase];
    }
    params.rows = row_count;
    params.position = static_cast<int32_t>(position);
    params.norm_epsilon = static_cast<float>(norm_epsilon);
    params.attention_scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));

    const DLDevice resolved_device = device.unwrap();
    constexpr auto kernel = predictor_layer_kernel<kUsePDL>;
    runtime::set_max_dynamic_smem_per_device<kernel>(resolved_device.device_id,
                                                     kSharedBytes);
    LaunchKernel(static_cast<uint32_t>(ctas.unwrap()), kThreads,
                 resolved_device, kSharedBytes)
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

#ifdef QWEN3_TTS_PREDICTOR_STAMPS
struct PredictorStamps {
  // Copies the stamps into a host int64 (ctas, slots, 2) tensor of global timer
  // and SM clock pairs, then zeroes them for the next launch.
  static void read(tvm::ffi::TensorView stamps) {
    using namespace host;
    TensorMatcher({kStampCtas, kStampSlots, 2})
        .with_dtype<int64_t>()
        .with_device<kDLCPU>()
        .verify(stamps);
    RuntimeDeviceCheck(cudaMemcpyFromSymbol(stamps.data_ptr(), predictor_stamps,
                                            sizeof(predictor_stamps)));
    void *device_stamps = nullptr;
    RuntimeDeviceCheck(cudaGetSymbolAddress(&device_stamps, predictor_stamps));
    RuntimeDeviceCheck(cudaMemset(device_stamps, 0, sizeof(predictor_stamps)));
  }
};
#endif

} // namespace qwen3_tts_predictor

} // namespace sglang
