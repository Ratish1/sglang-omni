# PR #869 per-commit H200 and Nsight execution plan

## 1. Revisions and attribution

The product revision under test is:

```text
8adc63a695a7a4b896a455c6afc7d315dd0e9177
```

The debug branch tip may contain this execution plan as a later documentation-only
commit. Use `8adc63a6` for product A/B results so the tested source revision is
unambiguous.

Use these immutable revisions:

| Symbol | Revision | Change under test |
| --- | --- | --- |
| P0 | `58f14089871bba3e55b65c3273e9e6fc60c5b294` | untouched PR baseline |
| D1 | `68a937ce` | raw stream refs preserve producer device |
| D2 | `b974365f` | valid direct-ineligible streams select pooled relay |
| D3 | `5fca76a9` | valid direct-ineligible payloads select pooled relay |
| C1 | `c3d0bc91` | unused `CommEngine` facade removed |
| C2 | `8d5d65ae` | unused non-Mooncake data-ref variants removed |
| C3 | `7ccc7e01` | dead send timing state removed |
| C4 | `9973fef6` | unused SHM state removed |
| D4 | `42f135aa` | received CUDA stream values select pooled relay |
| DOC | `8adc63a6` | communication lifecycle documentation |

The tests are grouped by changed mechanism, not repeated per Git commit:

| Shared gate | Revisions | Commits proven |
| --- | --- | --- |
| M1 pooled raw-ref device | P0, D1 | D1 |
| M2 stream direct admission | D1, D2, D4 | D2 and D4 |
| M3 payload direct admission | D2, D3 | D3 |
| M4 static/API cleanup | each cleanup parent and child | C1, C2, C3, C4, DOC |
| M5 final Qwen integration | P0, DOC | combined production result |

No Mooncake file, route, option, or lifecycle is under test. Preserve the
Mooncake seam but do not install or activate it for these runs.

## 2. Worktrees and immutable harness

Create worktrees for the exact revisions. Do not repeatedly switch one source
tree while children, profiler launchers, or Python bytecode may still exist.

```bash
ROOT=/sgl-workspace/pr869-final
mkdir -p "$ROOT/worktrees" "$ROOT/artifacts" "$ROOT/tmp"

git worktree add "$ROOT/worktrees/p0" 58f14089871bba3e55b65c3273e9e6fc60c5b294
git worktree add "$ROOT/worktrees/d1" 68a937ce
git worktree add "$ROOT/worktrees/d2" b974365f
git worktree add "$ROOT/worktrees/d3" 5fca76a9
git worktree add "$ROOT/worktrees/c3" 7ccc7e01
git worktree add "$ROOT/worktrees/c4" 9973fef6
git worktree add "$ROOT/worktrees/d4" 42f135aa
git worktree add "$ROOT/worktrees/final" 8adc63a6
```

Use one harness stored outside all worktrees. Record its SHA256 and copy or
symlink the identical file into each invocation. Every child must import product
code from the selected worktree, verified by printing
`sglang_omni.__file__` and `git rev-parse HEAD` into the result JSON.

Use a repository-local short temp root such as `$ROOT/tmp/r` for IPC sockets.
Do not use a long pytest base path or mutate the harness between revisions.

## 3. Environment gate

Run once before M1, then only recheck GPU ownership before each launch:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.used \
  --format=csv,noheader
nvidia-smi topo -m
nvidia-smi -q -d MIG
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.nccl.version())'
nsys --version
taskset -pc $$
env | sort | grep -E 'CUDA|NCCL|TORCH|PYTORCH|SGLANG'
```

Select one genuinely idle NV18 H200 pair. Keep physical GPU UUIDs, visible
ordinal mapping, NUMA binding, CPU affinity, allocator variables, power mode,
clocks, container, driver, Torch, and CUDA identical for all accepted launches.
Reject a launch if another process appears on either GPU.

Use an immutable source ring equal to the outstanding window. A ring entry may
not be overwritten until its receiver checksum event and sender ownership
completion have both finished. This prevents a benchmark race from being
misclassified as a transport bug.

## 4. Harness boundary

Every correctness and latency case must traverse:

```text
scheduler/output producer
  -> real scheduler.outbox
  -> Stage drain and routing
  -> direct serializer OR CommEngine target queue
  -> production control publication
  -> real receiver Stage
  -> direct import OR relay get/copy
  -> device reconstruction
  -> GPU checksum event
  -> DataAck when relay-backed
  -> sender completion and slot release
```

Do not call `CudaIpcRelay.put_async()` directly for primary evidence. A direct
relay probe may be retained only to localize a product-path failure.

Each result JSON must contain:

- revision, harness hash, process PIDs and exit codes;
- source/destination GPU UUID and local ordinal;
- case, tensor shape/dtype/device/stride/storage offset/logical bytes;
- wire type and control bytes;
- relay put/get count, logical ACK count, pending before/after;
- pool slots before/peak/after and acquire wait rounds;
- GPU checksum and receiver device for every tensor leaf;
- end-to-end, admission, serialize, queue, write, control publication,
  import/get, reconstruction, ACK, and checksum timings;
- event-loop lag for sender and receiver;
- allocated/reserved/NVML before, peak, after;
- warnings and child stderr.

Use GPU reductions for CUDA correctness. Scalar D2H checksum reads must happen
after the measured transfer interval or be identified separately in Nsight.

## 5. M1: D1 pooled stream device semantics

### 5.1 Baseline proof on P0

Use cross-GPU pooled streaming with one CUDA primary and one CPU metadata leaf.
The primary forces `cuda_ipc`; the metadata therefore passes through
`write_tensor()` on the sender relay device.

Run one warmup plus five measured envelopes for each leaf:

1. `float32[1]`, contiguous;
2. `float32[4096]`, contiguous;
3. `int64[17]`, noncontiguous source view with exact logical values;
4. one-element `float32` view backed by a 4 MiB CPU allocation;
5. nested metadata containing a dictionary, list, and tuple with CPU and CUDA
   tensor leaves.

Expected P0 failure: CPU leaves reconstruct on receiver CUDA. Values should be
exact. If values fail, localize that separately; D1 is specifically a device
semantic change.

### 5.2 D1 correctness

Repeat the exact five cases on D1. Require:

- primary and original CUDA metadata on destination CUDA;
- every original CPU leaf on CPU with exact dtype, shape, and values;
- the one-element view transfers four logical bytes and does not retain its
  4 MiB source backing;
- raw `DataRef.device` equals the producer device string;
- put/get counts equal tensor-ref count; one logical ACK per envelope;
- pending zero and every pool slot returned;
- no CUDA IPC producer-lifetime warning or memory slope.

### 5.3 D1 performance isolation

The device field and CUDA-device reconstruction branch also execute for normal
CUDA stream refs. Compare P0 and D1 using a cross-GPU Qwen-shaped envelope:

```text
primary CUDA tensor: 4096 bytes
metadata.layer_hidden CUDA tensor: 4096 bytes
window: 16
warmup/measured: 12/1000
launch order: P0,D1,D1,P0 repeated three blocks
```

Keep COMM_TRACE and Nsight disabled. Record per-launch percentiles rather than
pooling samples first. Require identical wire, copy cardinality, operation
counts, ACK counts, and memory. D1 launch-median p50/throughput must be within
2% and p95 within 5%; any repeated difference must be localized to control
serialization, reconstruction, or event-loop lag.

This is the only latency test for D1.

## 6. M2: D2 and D4 stream admission

Use a three-revision sequence so the same eligible-stream workload measures
both stream changes without running two independent matrices:

```text
R0 = D1 68a937ce        before stream admission
R1 = D2 b974365f        static ineligibility admission
R2 = D4 42f135aa        received-storage admission
order per block: R0,R1,R2,R2,R1,R0
blocks: 3
```

### 6.1 D2 failure proof and correctness

Through the same-GPU, different-process production path, use a CUDA primary and:

1. CPU `float32[1]` metadata;
2. CPU `float32[4096]` metadata;
3. a 128 KiB ordinary metadata string;
4. nested CPU metadata from M1.

R0 must fail direct admission before publication. R1 and R2 must select the
pooled CUDA-IPC wire and satisfy the M1 device, operation, ACK, slot, memory, and
exit invariants. Do not assign R0 latency to these cases because R0 has no
successful transfer.

Add one delivered-abort and one caller-cancellation case on R1 only. They prove
that the newly selected pooled route connects to the already-tested drain and
ownership lifecycle. Do not repeat the broad abort/cancellation matrices.

### 6.2 D4 re-export correctness

Use three processes A, B, C on one physical GPU:

```text
A creates CUDA tensor
  -> direct CUDA IPC to B
B receives imported PyTorch CUDA storage
  -> forwards through real Stage route to C
C validates GPU checksum
```

Run payload once only as a control; payload already had a known re-export route.
Run streams with 4 KiB and 16 MiB tensors at windows 1 and 8.

R1 must reproduce PyTorch's pinned
`Attempted to send CUDA tensor received from another process` failure for the
stream. R2 must select pooled CUDA IPC from B to C, produce exact values, use
one logical ACK, return B's slots, and exit without invalid-resource-handle or
producer-lifetime warnings.

Verify ownership mechanically: C's relay import must refer to B's pool export,
not A's direct storage token.

### 6.3 Shared D2/D4 unchanged-path performance

Use a same-GPU direct Qwen-shaped stream with two CUDA tensors totaling 8 KiB,
window 16, 12 warmups, and 1,000 measured envelopes in the three-revision order
above.

Required for R0/R1/R2:

- wire always `TorchCudaIpcStreamChunk`;
- relay puts/gets/ACKs always zero;
- exact checksums and device placement;
- no payload-sized D2D/H2D/D2H;
- pending and owned pool memory zero;
- each adjacent revision p50/throughput within 2%, p95 within 5%;
- no directionally repeated event-loop or control-size regression.

This one matrix is the only eligible-stream latency test for D2 and D4.

## 7. M3: D3 payload admission

Compare D2 and D3 in `D2,D3,D3,D2`, repeated three blocks.

### 7.1 Correctness cases

Use a same-GPU cross-process payload with one CUDA primary and:

1. exact 128 KiB ordinary data/header value;
2. a small CPU request tensor plus CUDA payload data;
3. a small CPU data tensor plus CUDA data tensor;
4. received CUDA storage forwarded from an intermediate process;
5. an unexpected injected serialization `RuntimeError` whose text is not the
   pinned PyTorch re-export error.

Expected results:

- D2 fails cases 1 and 2 at direct admission;
- D3 uses pooled CUDA IPC for cases 1 and 2 and preserves all values/devices;
- case 3 remains direct when its complete header fits the inline budget;
- case 4 uses pooled on both revisions, proving D3 preserved the existing
  re-export behavior while moving ownership of the decision into `stage_io`;
- case 5 fails on D3 and must not silently select pooled.

For pooled results require exact put/get/ACK, pending zero, slot return, child
exit zero, and no warning. Do not add a threshold based on payload size.

### 7.2 D3 performance

Use an eligible same-GPU direct payload containing one contiguous 16 MiB CUDA
tensor. Run windows 1 and 16, 12 warmups, 1,000 measured transfers, in the
matched order above.

Record direct admission/serialization separately. D3 removed a preliminary
whole-payload CUDA traversal, so it must not regress eligible serialization.
Require direct wire, zero relay/ACK, unchanged control bytes/copy topology,
p50/throughput within 2%, and p95 within 5%.

This is the only latency matrix for D3.

## 8. M4: cleanup commits without redundant GPU profiling

### C1 `c3d0bc91`: unused engine facade

No H200 or Nsight run. Prove absence of callers on the parent and absence of
symbols on C1:

```bash
rg -n '_comm\.(outbound|outbound_stream|inbound_relay|write_payload)\(' \
  sglang_omni tests
rg -n 'def (outbound|outbound_stream|inbound_relay|write_payload)\(' \
  sglang_omni/comm/engine.py
```

Run import/AST/pre-commit checks only. M5 exercises the surviving engine API.

### C2 `8d5d65ae`: speculative data-ref variants

No H200 or Nsight run. Search for the six removed names across code, tests,
docs, examples, configs, and serialized fixtures. Round-trip one instance of
each live combination through `to_dict()`/`from_dict()`:

- stage payload + packed tensors;
- stream chunk + raw tensor;
- stream metadata tensor + raw tensor;
- transports CUDA IPC, SHM, and preserved Mooncake vocabulary.

The removed values had no producer or consumer; a GPU run cannot provide more
evidence than reference and codec inspection.

### C3 `7ccc7e01`: dead send timing state

No dedicated launch. Inspect control flow to prove every successful trace assigns
`write_ms` and `control_ms`, while every exception exits before the trace. M1 and
M5 naturally execute `_run_stream_send`; M3 naturally executes payload sending.
Do not create a benchmark for four removed Python assignments.

### C4 `9973fef6`: unused SHM state

No CUDA or Nsight run. Run one CPU-only production-path payload and one CPU-only
stream with 0 and 17 logical bytes. Require exact data, SHM unlink, credit return,
and no `/dev/shm` object after completion. Do not repeat the old SHM size or
timeout matrices.

### DOC `8adc63a6`

No runtime test. Run Markdown/pre-commit checks and manually verify that every
named class, method, message, transport, ACK boundary, and direct/pooled claim
exists on the final source tree.

## 9. Nsight Systems: four reports only

Nsight is not repeated for every commit. These four short reports prove every
changed physical mechanism:

| Report | Revision | Cases in one process-tree capture | Purpose |
| --- | --- | --- | --- |
| T0 | P0 | eligible direct stream; cross-GPU pooled Qwen stream | physical baseline |
| T1 | D1 | cross-GPU pooled Qwen stream; pooled CPU metadata | device-field and restoration copies |
| T2 | D2 | eligible direct stream; same-GPU CPU-metadata pooled stream | static admission, unchanged direct path |
| T3 | D4 | eligible direct stream; A-to-B-to-C re-export stream | re-export ownership and final direct path |

Each case uses one warmup plus five measured envelopes at window 1. Profiling
is for operation/copy/synchronization proof, not latency. Workload JSON must be
written independently before Nsight symbol processing begins.

### 9.1 Profiling-only NVTX ranges

Apply the same uncommitted profiling patch or harness wrappers to all four
revisions. Record its patch SHA256. Use asynchronous NVTX range IDs for asyncio
tasks; do not use thread-stack push/pop across `await` boundaries.

Required range names and owners:

| Range | Owner boundary |
| --- | --- |
| `producer.outbox_put` | scheduler producer to Stage ownership transfer |
| `stage.route` | Stage target selection |
| `direct.admission` | direct representability traversal |
| `direct.serialize` | ForkingPickler handle creation |
| `comm.queue_put` | target queue backpressure |
| `comm.worker` | dequeue through publication |
| `codec.write_tensor` | contiguous/device conversion and raw-ref construction |
| `relay.slot_acquire` | contiguous CUDA pool allocation |
| `relay.pool_copy_submit` | source or staged tensor into sender pool |
| `control.serialize` | msgpack control encoding |
| `control.publish` | ZMQ send await |
| `control.receive` | ZMQ receive and decode |
| `direct.import` | PyTorch CUDA storage reconstruction |
| `relay.pool_import` | receiver pool handle import/cache lookup |
| `relay.peer_copy_submit` | sender pool to receiver-private storage |
| `codec.restore_device` | CPU restoration or CUDA no-op |
| `control.ack` | receiver ACK publication |
| `sender.release` | ACK observation and slot/source release |
| `wire_commit_wait` | per-lane scheduler-visibility predecessor wait |
| `validation.checksum` | profiling-only correctness work |

Each range label includes compact fields only:

```text
seq, request_id_short, from, to, kind, transport, logical_bytes,
tensor_count, metadata_tensor_count, window, object_id_short
```

Do not include full metadata, tensor values, IPC handles, or shapes in NVTX
strings.

### 9.2 Nsight command

Use the proven process-tree form:

```bash
CUDA_VISIBLE_DEVICES=<physical-a>,<physical-b> \
SGLANG_OMNI_COMM_TRACE=0 \
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --trace-fork-before-exec=true \
  --cuda-memory-usage=true \
  --wait=all \
  --output="$OUT/<report-name>" \
  python "$HARNESS" --suite <trace-suite> --warmups 1 --iterations 5 --window 1
```

Use the same bounded TERM workaround previously required by Nsight 2026.2 only
after the workload JSON reports success. A profiler wrapper timeout during symbol
download is not a workload result. Preserve the valid `.nsys-rep` and note the
wrapper status separately.

### 9.3 Export and inspection

```bash
nsys stats \
  --report=nvtx_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_kern_sum,osrt_sum \
  --format=csv \
  --output="$BASE-stats" \
  "$BASE.nsys-rep"

nsys export --type=sqlite --force-overwrite=true \
  --output="$BASE.sqlite" "$BASE.nsys-rep"
```

Inspect the 2026.2 SQLite schema before querying table names. Produce derived
tables keyed by process, stream, correlation ID, NVTX range, copy kind, copy
bytes, kernel, and CUDA API.

### 9.4 Physical acceptance

Eligible direct stream in T0/T2/T3:

- zero payload-sized D2D, H2D, and D2H;
- no relay pool copy, peer copy, or ACK;
- no `cudaDeviceSynchronize`, context synchronization, or new host-blocking
  stream synchronization;
- only checksum scalar D2H identified as validation.

Unchanged cross-GPU pooled Qwen stream in T0/T1:

- identical source-to-pool and peer-copy count/bytes per tensor ref;
- identical destination fill behavior and CUDA event ordering;
- no payload-sized host staging;
- identical logical ACK cardinality and final slot state.

CPU metadata in T1/T2:

- H2D bytes equal the logical CPU metadata when staged to the CUDA relay;
- D2H bytes equal the logical CPU metadata when restored, plus separately
  identified checksum reads;
- no copy of the 4 MiB backing allocation for the tiny-view case;
- CUDA primary remains entirely device-to-device.

Re-export in T3:

- direct import A to B;
- B pool copy from imported source into B-owned pool;
- B-to-C private destination copy;
- C never imports A's token as a B-produced direct ref;
- no invalid-resource-handle or producer-lifetime warning.

## 10. COMM_TRACE diagnostic, once per changed owner

JSON tracing is perturbative and must not be enabled during latency runs. Run
only six envelopes for each of these revisions:

- D1 for raw-ref construction/device restoration;
- D2 for stream admission and pooled ACK lifecycle;
- D3 for payload admission;
- D4 for re-export admission.

Extract queue wait, stage write, control send, pool acquire, pool copy, receiver
copy, ACK wait/resume, and event-loop lag. Do not compare trace-on end-to-end
latency against trace-off results.

## 11. M5 minimal real Qwen integration

Run only P0 and final DOC. This is the one combined integration comparison; do
not rerun it for intermediate commits.

Use the same model, prompt, seed, dtype, graph settings, stage placement, client
concurrency, CPU affinity, physical GPUs, and allocator environment. Warm the
compiler/kernel cache before accepted measurements. Run two fresh launches per
revision in `P0,FINAL,FINAL,P0` order.

Each launch performs:

1. one non-streaming request;
2. one streaming request;
3. four concurrent streaming clients;
4. disconnect after first audio chunk;
5. one subsequent reuse request.

Record model compute timings separately from communication. Require HTTP 200 for
completed requests, ordered audio/EOS, healthy post-cancel reuse, exact wire and
put/get/ACK counts, pending zero, and memory return after shutdown.

Performance gates:

- no directionally repeated TTFT or requests/s regression above 5%;
- ITL median within 3%;
- control bytes may increase only by the expected raw-ref `device` fields;
- no new relay operation for existing Qwen traffic;
- no communication queue, slot wait, event-loop lag, or GPU memory slope.

One outlier does not fail the branch. A repeated block-level regression does.

## 12. Stop and rollback rules

Stop at the first red correctness or ownership gate. Do not continue into
performance and average away a semantic failure.

Rollback ownership:

| Failure | Owning commit |
| --- | --- |
| CPU metadata reconstructs on CUDA or CUDA moves to CPU | D1 |
| static ineligible stream fails or selects wrong wire | D2 |
| valid ineligible payload fails or unexpected error is hidden | D3 |
| live protocol enum cannot decode | C2 |
| SHM block/credit remains | C4 |
| received CUDA stream fails or wrong producer owns C's storage | D4 |

Mechanical cleanup is not reverted for a noisy H200 percentile. Admission or
device changes are not retained merely because they fix correctness if unchanged
production traffic shows a repeated matched regression outside the gates.

Do not respond to regression by adding size thresholds, broad exception catches,
extra synchronization, retry layers, or more transport states. First identify
the exact added traversal, serialization bytes, copy, CUDA API, queue delay, or
event-loop stall.

## 13. Tests explicitly not repeated

- full prior Phase A--F matrix;
- 70,000-transfer stress and 20,000-transfer direct matrices;
- 1 MiB/16 MiB/256 MiB general size sweep;
- empty tensor and slot-size boundary matrices;
- fan-out receiver close-order matrix;
- receive ordering/error/abort matrix;
- ACK publication clock, cancellation, and idle-retention matrices except the
  one new-route integration checks named above;
- 21-case CPU metadata size/backing matrix;
- `torch.zeros` versus `torch.empty` A/B;
- producer-readiness experiments;
- twelve fresh Qwen launch study;
- generic unit-test expansion or parametrized tests.

## 14. Final report

Report five independent decisions:

1. D1 device correctness and pooled-path cost.
2. D2 stream admission correctness and eligible-direct cost.
3. D3 payload admission correctness and eligible-direct cost.
4. D4 re-export ownership correctness and eligible-direct cost.
5. P0 versus final real-Qwen result.

For every excluded launch state the exact exclusion reason and retain its raw
artifact. Do not manufacture a latency comparison for a baseline case that
failed. End with a keep/revert decision for D1--D4 and a separate statement that
C1--C4 are static cleanups, not measured optimizations.
