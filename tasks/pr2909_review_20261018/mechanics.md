# PR #2909 mechanics review

Reviewed head: cefab51fc709735b12b3d25793229a8fed89722a. Merge base: d07df0b4339cf6efcb6cec1d0da7e7ecd4b80ddf. The author worktree was clean and already at the fetched PR head. GitHub had no issue comments, submitted reviews or inline review comments at the review snapshot.

## Change decomposition

| Part | Production files | Mechanical responsibility | Assessment |
|---|---|---|---|
| State allocation | components/codec_state_pool.py | Fixed stream slots, separate capture slots, conv history/overlap tensors, per-layer KV rings, absolute frame cursors | Appropriate model-local ownership; no unbounded per-window cache |
| Transformer | components/code2wav.py | Shared fused layer math, absolute RoPE, ring writes and local-attention reads | Restores the actual model context rather than approximating it with overlapping input windows |
| Convolutions | components/incremental_codec.py | Save causal input tails and unfinished transposed-convolution contributions | State is taken from actual module padding/stride/channel dimensions |
| CUDA graphs | components/code2wav_cuda_graph.py | Separate first/next graphs, dynamic slot indices, scratch slots for capture, eager stateful fallback | Necessary distinction: first and next have different state-read and padding contracts |
| Scheduling | components/code2wav_scheduler.py | Claim once, group compatible windows, advance state, finish tails, release on lifecycle exits | Continues the existing serialized decode and D2H ownership model |
| Capacity/configuration | config.py, stages.py | Move the talker cap to pipeline policy and derive decoder slot count from the effective cap and replica count | Pipeline → factory → runtime flow is coherent; engine override wins over factory cap |
| Tests | Eight changed test modules | Numerical parity, ring math, reuse, fallback, cleanup and configuration wiring | Most tests exercise derived properties or real lifecycle bookkeeping; multi-row graph ownership needs a stronger integration case |

This decomposition is an analysis of separate mechanisms, not a claim that intermediate commits were benchmarked with git bisect.

## High-level serving flow

1. Preprocessing and media encoders build the thinker/talker inputs. The thinker produces text and feedback consumed by the talker. This PR does not change those payload formats or sampling.
2. The talker delivers codec frames to code2wav. A new stream claims a pool slot on its first frame, before its first decode. If no slot is free, that stream stays on the old stateless path for its lifetime.
3. The code2wav serving loop ingests queued messages before choosing a ready group. It groups first versus later windows, stateful versus stateless windows, and equal window lengths separately. First audio retains its existing scheduling priority.
4. A stateful decode processes only new codec frames. The transformer and convolution stack read/write the row's slot. Graph replay receives current slot IDs through a pinned nonblocking upload.
5. First audio is copied synchronously as before. Later audio can use the existing pinned D2H slots and completion events. The graph's output stays borrowed until its dependent reads/copies have been enqueued on the same stream.
6. On stream completion, complete threshold windows drain, then the pending copy, then the final shorter window. The final result either carries metadata for streaming requests or concatenated audio for buffered requests. The decoder slot returns to the pool.
7. Abort, step failure and shutdown use the existing release hook. State slots return without a device-wide wait. Pending D2H buffers and input chunks retain their existing event-based retirement.

The real scheduler interface here is handle_stream_chunk followed by run_ready_step when queued input has drained. Calling the generic on_stream_chunk_batch pump directly is not its serving path. The independent probes use the former interface.

## Low-level invariants

### Attention and ring wrap

Let W be the model's sliding window, C the largest new-frame chunk, and p the first new absolute position. Each slot has W−1+C KV rows per layer. The current checkpoint has W=72 and the default C=10, so each ring has 81 rows.

Position t is stored at slot×81 + t mod 81. At a mature step, the retained/new positions are p−71 through p+C−1. Their addresses are distinct. Writing the new rows only overwrites positions older than the first retained row.

For query p+i, the boolean SDPA mask admits keys with 0 ≤ query−key < W. It excludes future positions and uninitialized/stale ring rows. When p is still below W−1, the query offset uses the actual retained length. Final chunks shorter than C use the same derivation.

The ring stores each layer's projected, rotated K and V, rather than recomputing old hidden states from a truncated prompt. Absolute RoPE positions therefore matter. The table now uses the talker checkpoint's context bound; ordinary serving cannot generate more codec positions than the talker context permits. No CPU read of the frame cursor is introduced.

A cold decode attends only to its own newly projected frames. A warm decode reads the retained ring. Slot reuse does not need to clear old KV: the first decode writes current rows and ignores old rows; later masks only admit fresh valid positions.

### Causal convolution

For each convolution, the pool retains exactly module.padding input samples. The next input is history followed by new samples. Its last padding samples replace the saved history. When a chunk is shorter than the history, this retains both the necessary old suffix and the new samples.

The dilated path preserves the existing channels-last phase decomposition. Extra right padding is determined by the current length modulo dilation and discarded after convolution. It is not retained as real history.

The depthwise ConvNeXt convolution uses its actual PyTorch depthwise operator with the retained input prefix; its normalization, GELU, projections and residual retain the original order.

### Transposed convolution

The raw convolution runs without bias. A warm call adds the previous unfinished contributions to its beginning. The tail past input_length×stride becomes the next saved overlap. Bias is added once to the emitted region after overlap addition.

Only the first call trims the module's left padding. Every call withholds the right overlap. Consequently, the initial output deficit occurs once, matching the whole-sequence decoder. Later calls produce the next full span. Adding two biased overlap regions would be wrong; the implementation avoids that.

The equivalence claim is numerical, not bit identity with the previous stateless windows. Chunk boundaries can change bf16 convolution accumulation, and retaining transformer context intentionally changes the waveform toward whole-sequence decoding.

### Ownership and concurrency

The scheduler's state_lock serializes slot allocation/release against launch. GPU decode work is enqueued on the serving thread's one stream; a colocation-specific stream is selected at serving start. An abort from another thread cannot insert a new owner's decode between an old owner's state operations.

Returning a slot before its previous GPU operations complete is therefore safe under this existing single-stream contract: a new cold decode is queued after them. Introducing parallel decode streams in a future change would require revisiting this contract.

The codec state pool and pinned output pool are different resources. A returned codec slot can be reused while old D2H work finishes because D2H consumes the decoded output tensor, not live codec state. Input chunks and transfer buffers still use completion-event retirement.

Failed cohorts abort all participants. The implementation does not retry a partially executed stateful replay as though no state advanced. Graph-key misses and a disabled graph runner execute the stateful eager path, preserving the same slot ownership.

### Capture and memory

Each captured row gets a distinct scratch slot outside the stream free list. Next-window equivalence checks reset scratch state before eager and replay executions. Cold graphs overwrite their required state and do not need that reset.

The graph key now includes decode mode as well as row/frame counts. The published batch-size lookup uses the same mode, so a stateless or first-window graph cannot stand in for a next-window replay merely because its dimensions match.

The pool's substantive tensor footprint is charged to the stage's existing graph budget. Missing a few bytes of integer cursors in the estimated per-slot count does not create a meaningful budget defect; it is not a review finding.

The capture plan currently misses one available bound: a stateful group cannot exceed the number of stream slots. The factory still sizes scratch state and stateful graph row variants from max_replay_rows alone. With one stream slot and the default eight-row cap, fourteen multi-row graphs are unreachable. The trained factory probe confirmed that clamping this bound preserves audio and avoids approximately 394 MiB of capture/pool memory in that H200 configuration. This is the conditional resource finding in the main review. No capture scratch state is needed when graphs are explicitly disabled.

The pool ring and convolution lengths derive from model mechanics. This PR does not add a GPU-name rule, reference-duration heuristic, dataset-specific threshold or arbitrary deadline constant.

## SGLang comparison and reuse

Local SGLang revision: e00930c5489053f26d86b179cee0d087f846acbb.

Read in full for the relevant mechanics:

- python/sglang/kernels/ops/attention/rope.py and kernels/jit/csrc/elementwise/rope.cuh: flattened token positions, fp32 cosine/sine tables, half-precision Q/K views, runtime strides, in-place rotation and optional fused KV store.
- python/sglang/kernels/ops/mamba/causal_conv1d.py and kernels/jit/csrc/mamba/causal_conv1d.cuh: per-sequence state indices, cold-state handling, retained input taps, circular history updates and CUDA launch contracts.
- python/sglang/srt/mem_cache/base_swa_memory_pool.py, allocator/swa.py and allocation_sizing.py: request-owned sliding-window capacity, ring lifetime versus token/page allocation, and capacity derived from scheduler bounds.

Also traced the complete cached-convolution helper, causal-convolution class and resampling cache transitions in multimodal_gen/runtime/models/vaes/autoencoder_kl_qwenimage.py. Those use retained feature tails and distinguish first/later chunks. The unrelated spatial-parallel VAE implementation is not a code2wav substitute.

The mechanisms agree, but the components are not interchangeable. SGLang's Mamba kernel supports depthwise widths 2–4; this codec has dense width-7/dilated convolutions and transposed convolutions. The SRT allocator owns AR token/page storage, not a complete audio decoder's convolution histories. Requiring either component here would add adapters without removing the model-specific state logic.

The PR already uses the existing SGLang rotary kernel. Its description explicitly defers fused rotary/KV storage and faster ring attention to follow-up PRs. Their absence is not a correctness defect in this prerequisite PR.

## Source audit scope

Read all seven changed production files, the complete new pool/convolution modules, the changed test hunks and complete code2wav parity test module, the channels-last convolution utility, weight loader, typed factory argument resolution, streaming vocoder lifecycle and streaming simple scheduler. Traced the complete Hugging Face code2wav transformer, convolution, upsampler and decoder classes. Checked all callers of the changed fused-transformer signature and the new required talker cap.

The source review is scoped to supported serving contracts. Invalid internal slot IDs, concurrent external calls to an unowned graph runner and a future multi-stream scheduler are not presented as current PR bugs.
