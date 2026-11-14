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
inline constexpr int kOperandKStepBytes = kTokenTileRows * kKStepDepth * 2;
inline constexpr int kMaxOperandKSteps = kIntermediateSize / kKStepDepth;
inline constexpr int kOperandBytes =
    kMaxTokenTiles * kMaxOperandKSteps * kOperandKStepBytes;
inline constexpr int kPartialFloats =
    kComputeWarps * kMaxTokenTiles * kTileFeatures * kTokenTileRows;
inline constexpr int kSharedBytes = kRingStages * kChunkBytes + kOperandBytes +
                                    kPartialFloats * 4 + kMaxRows * 4 +
                                    2 * kHeadDim * 4 + 2 * kRingStages * 8;

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

SGL_DEVICE int operand_depth(int phase) {
  if (phase == kPhaseOutputProjection)
    return kAttentionWidth;
  if (phase == kPhaseDown)
    return kIntermediateSize;
  return kHiddenSize;
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

SGL_DEVICE float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
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

// note (ratish): the fence publishes what the other compute threads wrote
// before the barrier that precedes this call, as a split-K arrive does.
SGL_DEVICE void signal_phase(const LayerParams &params, int phase,
                             uint32_t items) {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
  reinterpret_cast<device::atomic::Event *>(params.phase_events + phase)
      ->arrive(items);
}

SGL_DEVICE void wait_phase(const LayerParams &params, int phase) {
  reinterpret_cast<device::atomic::Event *>(params.phase_events + phase)
      ->wait_multi(phase_producers(params, phase),
                   params.phase_consumers[phase]);
}

// Byte offset of one token's 8 values in the operand buffer, laid out
// [token tile][k-step][half][token 8][8 values] so every 8x8 matrix is 128
// contiguous bytes.
SGL_DEVICE int operand_offset(int operand_ksteps, int token, int value_vector) {
  const int token_tile = token / kTokenTileRows;
  const int kstep = value_vector / 2;
  const int half = value_vector % 2;
  return ((token_tile * operand_ksteps + kstep) * 2 + half) * 128 +
         (token % kTokenTileRows) * 16;
}

// Stages the phase's input rows as the MMA B operand. A normalized phase
// multiplies by the norm weight and rounds to bf16 before the product and keeps
// each row's rstd for the epilogue, the order of the fused Triton layer.
SGL_DEVICE void stage_operand(const LayerParams &params, const bf16_t *source,
                              int depth, const bf16_t *norm_weight,
                              uint8_t *operand, float *row_scales,
                              int token_tiles, int warp_index, int lane_index) {
  const int vectors_per_row = depth / 8;
  const int operand_ksteps = depth / kKStepDepth;
  for (int token = warp_index; token < token_tiles * kTokenTileRows;
       token += kComputeWarps) {
    float sum_of_squares = 0.0f;
    for (int vector = lane_index; vector < vectors_per_row; vector += 32) {
      uint4 packed = make_uint4(0u, 0u, 0u, 0u);
      if (token < params.rows) {
        packed = __ldcg(reinterpret_cast<const uint4 *>(
                            source + static_cast<int64_t>(token) * depth) +
                        vector);
        if (norm_weight != nullptr) {
          const uint4 weight_packed =
              __ldg(reinterpret_cast<const uint4 *>(norm_weight) + vector);
          bf16_t *values = reinterpret_cast<bf16_t *>(&packed);
          const bf16_t *weights =
              reinterpret_cast<const bf16_t *>(&weight_packed);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float value = __bfloat162float(values[j]);
            sum_of_squares += value * value;
            values[j] = __float2bfloat16(value * __bfloat162float(weights[j]));
          }
        }
      }
      *reinterpret_cast<uint4 *>(
          operand + operand_offset(operand_ksteps, token, vector)) = packed;
    }
    if (norm_weight != nullptr) {
      sum_of_squares = warp_sum(sum_of_squares);
      if (lane_index == 0)
        row_scales[token] =
            rsqrtf(sum_of_squares / depth + params.norm_epsilon);
    }
  }
}

// Multiplies one 16-feature weight tile, streamed through the ring, by the
// staged rows. Warp w takes k-steps [8w, 8w + 8) of every chunk; the warps'
// partial sums land in `partials` as [warp][token tile][feature 16][token 8].
SGL_DEVICE void multiply_tile(int chunks, int operand_ksteps, int token_tiles,
                              uint8_t *ring, const uint8_t *operand,
                              float *partials, uint64_t *full_barriers,
                              uint64_t *empty_barriers, int &stage,
                              uint32_t &full_parity, int first_chunk,
                              int warp_index, int lane_index) {
  float accumulator[kMaxTokenTiles][4] = {};
  const uint32_t operand_address = shared_address(operand) + lane_index * 16;
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
      uint32_t first_weights[4];
      uint32_t second_weights[4];
      load_matrix_x4(chunk_address + kstep * kKStepBytes, first_weights);
      load_matrix_x4(chunk_address + (kstep + 1) * kKStepBytes, second_weights);
#pragma unroll
      for (int token_tile = 0; token_tile < kMaxTokenTiles; ++token_tile) {
        if (token_tile < token_tiles) {
          uint32_t tokens[4];
          const int operand_kstep = chunk * kKStepsPerChunk + kstep;
          load_matrix_x4(operand_address +
                             (token_tile * operand_ksteps + operand_kstep) *
                                 kOperandKStepBytes,
                         tokens);
          mma_bf16(accumulator[token_tile], first_weights, tokens[0],
                   tokens[1]);
          mma_bf16(accumulator[token_tile], second_weights, tokens[2],
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
    float *block = partials + ((warp_index * kMaxTokenTiles + token_tile) *
                               kTileFeatures) *
                                  kTokenTileRows;
    block[feature * kTokenTileRows + token] = accumulator[token_tile][0];
    block[feature * kTokenTileRows + token + 1] = accumulator[token_tile][1];
    block[(feature + 8) * kTokenTileRows + token] = accumulator[token_tile][2];
    block[(feature + 8) * kTokenTileRows + token + 1] =
        accumulator[token_tile][3];
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

SGL_DEVICE void finish_tile(const LayerParams &params, int phase, int tile,
                            int token_tiles, const float *partials,
                            const float *row_scales) {
  const int thread = threadIdx.x;
  if (phase == kPhaseGateUp) {
    // Rows 0-7 of a gate/up tile are gate features, rows 8-15 the up features
    // of the same index.
    if (thread >= token_tiles * (kTileFeatures / 2) * kTokenTileRows)
      return;
    const int token_tile = thread / ((kTileFeatures / 2) * kTokenTileRows);
    const int feature = (thread / kTokenTileRows) % (kTileFeatures / 2);
    const int token_in_tile = thread % kTokenTileRows;
    const int token = token_tile * kTokenTileRows + token_in_tile;
    if (token >= params.rows)
      return;
    const float scale = row_scales[token];
    const float gate = round_bf16(
        reduce_partials(partials, token_tile, feature, token_in_tile) * scale);
    const float up = round_bf16(
        reduce_partials(partials, token_tile, feature + 8, token_in_tile) *
        scale);
    params.activation_scratch[static_cast<int64_t>(token) * kIntermediateSize +
                              tile * 8 + feature] =
        __float2bfloat16(gate / (1.0f + expf(-gate)) * up);
    return;
  }
  if (thread >= token_tiles * kTileFeatures * kTokenTileRows)
    return;
  const int token_tile = thread / (kTileFeatures * kTokenTileRows);
  const int feature = (thread / kTokenTileRows) % kTileFeatures;
  const int token_in_tile = thread % kTokenTileRows;
  const int token = token_tile * kTokenTileRows + token_in_tile;
  if (token >= params.rows)
    return;
  const float total =
      reduce_partials(partials, token_tile, feature, token_in_tile);
  const int output_feature = tile * kTileFeatures + feature;
  const int64_t hidden_index =
      static_cast<int64_t>(token) * kHiddenSize + output_feature;
  if (phase == kPhaseQkv) {
    params
        .qkv_scratch[static_cast<int64_t>(token) * kQkvWidth + output_feature] =
        __float2bfloat16(total * row_scales[token]);
  } else if (phase == kPhaseOutputProjection) {
    params.middle_scratch[hidden_index] = __float2bfloat16(
        round_bf16(total) + __bfloat162float(params.input[hidden_index]));
  } else {
    params.output[hidden_index] = __float2bfloat16(
        round_bf16(total) +
        __bfloat162float(__ldcg(params.middle_scratch + hidden_index)));
  }
}

// RMS norm over one head held four values a lane (dims lane + 32 i), then neox
// rope at the cache position, each step rounded to bf16 where the fused Triton
// layer rounds.
SGL_DEVICE void norm_and_rotate_head(const LayerParams &params,
                                     float (&values)[4],
                                     const bf16_t *norm_weight,
                                     int lane_index) {
  float sum_of_squares = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sum_of_squares += values[i] * values[i];
  }
  const float scale =
      rsqrtf(warp_sum(sum_of_squares) / kHeadDim + params.norm_epsilon);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = round_bf16(values[i] * scale *
                           __bfloat162float(norm_weight[lane_index + 32 * i]));
  }
  const float *angles = params.rope_cos_sin + params.position * kHeadDim;
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

// One (token, kv group) of attention: warps 0 and 1 take the group's two query
// heads, warp 2 the key and warp 3 the value of the new token, which also go
// into the cache at the position.
SGL_DEVICE void attend(const LayerParams &params, int item, float *key_current,
                       float *value_current, int warp_index, int lane_index) {
  const int token = item / kKeyValueHeads;
  const int group = item % kKeyValueHeads;
  const bf16_t *qkv_row =
      params.qkv_scratch + static_cast<int64_t>(token) * kQkvWidth;
  const int64_t slot_stride = kKeyValueHeads * kHeadDim;
  const int64_t cache_base =
      static_cast<int64_t>(token) * params.cache_row_stride + group * kHeadDim;
  const int64_t current_slot = cache_base + params.position * slot_stride;
  float query[4] = {};
  if (warp_index == 2) {
    float key[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      key[i] = __bfloat162float(__ldcg(qkv_row + kKeyOffset + group * kHeadDim +
                                       lane_index + 32 * i));
    }
    norm_and_rotate_head(params, key, params.key_norm_weight, lane_index);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      params.key_cache[current_slot + lane_index + 32 * i] =
          __float2bfloat16(key[i]);
      key_current[lane_index + 32 * i] = key[i];
    }
  } else if (warp_index == 3) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const bf16_t value = __ldcg(qkv_row + kValueOffset + group * kHeadDim +
                                  lane_index + 32 * i);
      params.value_cache[current_slot + lane_index + 32 * i] = value;
      value_current[lane_index + 32 * i] = __bfloat162float(value);
    }
  } else if (warp_index < 2) {
    const int head = group * 2 + warp_index;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      query[i] = __bfloat162float(
          __ldcg(qkv_row + head * kHeadDim + lane_index + 32 * i));
    }
    norm_and_rotate_head(params, query, params.query_norm_weight, lane_index);
  }
  compute_barrier();
  if (warp_index >= 2)
    return;
  const int head = group * 2 + warp_index;
  float scores[kCacheSlots];
  float best = -INFINITY;
  for (int slot = 0; slot <= params.position; ++slot) {
    float partial = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float key =
          slot == params.position
              ? key_current[lane_index + 32 * i]
              : __bfloat162float(
                    params.key_cache[cache_base + slot * slot_stride +
                                     lane_index + 32 * i]);
      partial += query[i] * key;
    }
    scores[slot] = warp_sum(partial) * params.attention_scale;
    best = fmaxf(best, scores[slot]);
  }
  float weight_sum = 0.0f;
  float attended[4] = {};
  for (int slot = 0; slot <= params.position; ++slot) {
    const float weight = expf(scores[slot] - best);
    weight_sum += weight;
    const float rounded_weight = round_bf16(weight);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float value =
          slot == params.position
              ? value_current[lane_index + 32 * i]
              : __bfloat162float(
                    params.value_cache[cache_base + slot * slot_stride +
                                       lane_index + 32 * i]);
      attended[i] += rounded_weight * value;
    }
  }
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
  float *row_scales = partials + kPartialFloats;
  float *key_current = row_scales + kMaxRows;
  float *value_current = key_current + kHeadDim;
  uint64_t *full_barriers =
      reinterpret_cast<uint64_t *>(value_current + kHeadDim);
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
    device::PDLWaitPrimary<kUsePDL>();
    if (threadIdx.x == 0)
      record_stamp(kStampPrimaryWaited);
    const int token_tiles = (params.rows + kTokenTileRows - 1) / kTokenTileRows;
    int stage = 0;
    uint32_t full_parity = 0u;
    int current_phase = -1;
    uint32_t phase_items = 0u;
    int consumed_chunks = 0;
    for (int index = 0; index < schedule_length; ++index) {
      const int phase = schedule[index] >> kScheduleTileBits;
      const int tile = schedule[index] & kScheduleTileMask;
      if (phase != current_phase) {
        if (threadIdx.x == 0)
          record_phase_stamp(phase, kStampPhaseReached);
        if (current_phase >= 0 && threadIdx.x == 0) {
          signal_phase(params, current_phase, phase_items);
          record_phase_stamp(current_phase, kStampPhaseSignaled);
        }
        if (phase > kPhaseQkv) {
          if (threadIdx.x == 0)
            wait_phase(params, phase - 1);
          compute_barrier();
          if (threadIdx.x == 0)
            record_phase_stamp(phase, kStampPhaseWaited);
        }
        if (phase == kPhaseQkv) {
          stage_operand(params, params.input, kHiddenSize,
                        params.input_norm_weight, operand, row_scales,
                        token_tiles, warp_index, lane_index);
        } else if (phase == kPhaseOutputProjection) {
          stage_operand(params, params.attention_scratch, kAttentionWidth,
                        nullptr, operand, row_scales, token_tiles, warp_index,
                        lane_index);
        } else if (phase == kPhaseGateUp) {
          stage_operand(params, params.middle_scratch, kHiddenSize,
                        params.post_norm_weight, operand, row_scales,
                        token_tiles, warp_index, lane_index);
        } else if (phase == kPhaseDown) {
          stage_operand(params, params.activation_scratch, kIntermediateSize,
                        nullptr, operand, row_scales, token_tiles, warp_index,
                        lane_index);
        }
        compute_barrier();
        if (threadIdx.x == 0)
          record_phase_stamp(phase, kStampPhaseStaged);
        current_phase = phase;
        phase_items = 0u;
      }
      if (phase == kPhaseAttention) {
        attend(params, tile, key_current, value_current, warp_index,
               lane_index);
      } else {
        multiply_tile(chunks_per_tile(phase),
                      operand_depth(phase) / kKStepDepth, token_tiles, ring,
                      operand, partials, full_barriers, empty_barriers, stage,
                      full_parity, consumed_chunks, warp_index, lane_index);
        consumed_chunks += chunks_per_tile(phase);
        compute_barrier();
        finish_tile(params, phase, tile, token_tiles, partials, row_scales);
      }
      // Every item ends on this barrier, so a signal after it covers all of the
      // item's stores.
      compute_barrier();
      if (threadIdx.x == 0)
        record_listed_stamp(kStampItemEnd, index);
      phase_items += 1u;
    }
    if (current_phase >= 0 && current_phase < kPhaseDown && threadIdx.x == 0) {
      signal_phase(params, current_phase, phase_items);
      record_phase_stamp(current_phase, kStampPhaseSignaled);
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

    TensorMatcher({rows, kHiddenSize})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input)
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
        .with_device(device)
        .verify(input_norm_weight)
        .verify(post_norm_weight);
    TensorMatcher({kHeadDim})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(query_norm_weight)
        .verify(key_norm_weight);
    TensorMatcher({kCacheSlots, kHeadDim})
        .with_dtype<float>()
        .with_device(device)
        .verify(rope_cos_sin);
    TensorMatcher({cache_rows, kCacheSlots, kKeyValueHeads, kHeadDim})
        .with_strides(
            {cache_row_stride, kKeyValueHeads * kHeadDim, kHeadDim, 1})
        .with_dtype<bf16_t>()
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
