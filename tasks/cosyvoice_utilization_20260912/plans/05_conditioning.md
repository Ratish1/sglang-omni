# Conditioning and CPU preparation

Open PR #1693 batches reference S3 tokenization and changes the tokenizer from ONNX/Whisper execution to `s3tokenizer==0.3.0` PyTorch. This overlaps the conditioning owner and merits a frozen-reference/backend comparison, but does not implement the duplicate decode/resample removal proposed below. Reconcile dependency, reference-token parity and lifecycle changes before extending that path. See [report 17](../reports/17_open_prs.md).

Owners: reference normalization/cache/service, `request_builders.py`, ONNX utilities, prepared-state map, and public speech validator. Evidence: reports 01, 04, 07, 13.

Trigger: validation/reference/finalization spans leave AR underfed or dominate TTFA. Distinguish cold unique references from repeated references and single-flight followers. Warmup repeats one reference; it does not warm the full SeedTTS corpus.

First run a controlled ONNX-thread/preprocessing-concurrency experiment against the H100 container's **CPU quota and affinity**. Two per-session pools sized from host CPU count plus preprocessing executor threads may oversubscribe the allocation, but quota/CPU traces must establish this. Document chosen deployment values in a small config PR only after workload results support them.

P1 can remove duplicate audio decoding/resampling by retaining one decoded source with explicitly derived 16 kHz and 24 kHz views. Preserve mono handling, resampler/kernel choices, trim limits, amplitude and sample rounding. A nominal sample-rate equivalence is insufficient to preserve tokenizer/conditioning output. This is independent of cache policy and must be reviewed as such.

A separate P2 may optimize embedding-cache-key preparation. The present key is a hash of actual float32 embedding bytes, which ties prefix identity to weights and conditioning. A replacement must preserve equality/inequality of prefixes for all supported text/reference/task/model/weight-update cases. Prefer reducing repeated work while retaining the exact byte-hash contract; any semantic-key scheme requires a formal injective identity argument and epoch handling. Moving computation outside the finalize lock also requires proving tokenizer/model/thread safety and one-shot prepared-map ownership. Do not simply remove the lock or hash only text.

The local prep→AR edge serializes Flow state into CPU lists despite colocated delivery. A typed local conditioning payload can be a later separate PR if measured conversion cost matters. Retain a compatible wire representation for remote transport and explicit tensor lifetime/device ownership; first stream metadata and final payload must agree. Never expose process-global prepared state across processes.

Remote gates: exact reference tokens/features/speaker embedding and cache keys on cold/hit/follower paths; changed content at a reused path; differing transcript/instruction/mode; abort while encoding and late publish cleanup; weight epoch; non-default CPU quota. Compare c1/c16 unique-reference and repeated-reference cohorts, then full English. Stop if the apparent improvement is just a cache hit-rate change. Roll back each independent preparation change to its original implementation; cache schema changes need an epoch/reset, not reuse of stale entries.
