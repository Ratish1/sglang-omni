# Talker decode-input helper scope, and the Qwen3-Omni benchmark protocol

Mechanics only. No recommendations. Every claim below is anchored to a line I
read in one of the two trees named next.

Code tree: `/Users/ratish/sglang-omni/.worktrees/qwen3-tts-predictor-chain`,
branch `perf/qwen3-tts-predictor-chain`, head `0a88253c6`, base main
`7989a5ed2`. Analysis tree for Part B.3:
`/Users/ratish/sglang-omni/.worktrees/qwen3-omni-0518-numerics/tasks/`.

Diff under study, `git diff 7989a5ed2..HEAD --stat`:

```
 .../models/qwen3_omni/talker_model_runner.py       |  76 +++++-----
 sglang_omni/models/qwen3_tts/model_runner.py       |  55 +++++--
 sglang_omni/models/qwen3_tts/sglang_model.py       | 163 +++++++++++----------
 sglang_omni/scheduling/omni_scheduler.py           |  15 ++
 tests/unit_test/pipeline/test_scheduler.py         |  73 +++++++++
 tests/unit_test/qwen3_omni/test_talker.py          |  27 ++++
 .../qwen3_omni/test_talker_feedback_write.py       |   1 +
 .../qwen3_omni/test_talker_row_ownership.py        |   1 +
 tests/unit_test/qwen3_tts/test_pipeline.py         | 138 ++++++++-------
 .../qwen3_tts/test_predictor_cuda_graph.py         | 130 ++++++++++----
 tests/unit_test/qwen3_tts/test_retract_prefill.py  |  47 ++++++
 tests/unit_test/qwen3_tts/test_sampling_kernels.py |  12 +-
 12 files changed, 528 insertions(+), 210 deletions(-)
```

The `sglang_model.py`, `test_predictor_cuda_graph.py` and
`test_sampling_kernels.py` parts of that diff are the Qwen3-TTS predictor CUDA
graph and subtalker sampling work. They do not touch the decode-input helpers
and are outside the scope of this report.

---

## Part A, helper scope

### A.0 Where the helpers live and what they read

All six named helpers are static methods on `QwenTalkerModelRunner` in
`/Users/ratish/sglang-omni/.worktrees/qwen3-tts-predictor-chain/sglang_omni/models/qwen3_omni/talker_model_runner.py`:

| helper | definition |
|---|---|
| `_data_has_next_decode_input` | talker_model_runner.py:388-396 |
| `_decode_input_history` | talker_model_runner.py:423-429 |
| `_append_decode_input_history` | talker_model_runner.py:431-435 |
| `_peek_next_decode_inputs` | talker_model_runner.py:453-469 (new on this branch) |
| `_pop_next_decode_inputs` | talker_model_runner.py:471-474 (new on this branch) |
| `_combine_feedback_with_next_text` | talker_model_runner.py:476-495 |

Three supporting statics in the same class are load bearing for the six above
and are cited throughout: `_pop_left` (403-411), `_peek_left` (413-421) and
`_decode_row` (437-451). `_decode_row` raises if a row is not already on the
target device and dtype (talker_model_runner.py:445-450), so no helper does an
implicit transfer.

The request-data fields the helpers read and write are declared on
`SGLangARRequestData` in
`sglang_omni/scheduling/sglang_backend/request_data.py`:
`decode_input_embeds` is `list[torch.Tensor]` with `default_factory=list`
(request_data.py:26), `pending_feedback_queue` and `pending_text_queue` are
`collections.deque` by default (request_data.py:29-30), `tts_pad_embed` defaults
to `None` (request_data.py:31) and `thinker_chunks_done` defaults to `True`
(request_data.py:33).

Producers of those fields, read for context:

- Qwen3-Omni: `request_builders.py:865-867` sets `thinker_chunks_done`,
  `pending_text_queue` (coerced to `PendingTextTensorQueue`) and
  `tts_pad_embed` at request build. `talker_prefill.py:264-280` appends later
  text chunks and appends the EOS row once at `mark_thinker_done`.
  `talker_scheduler.py:148-156` is the fallback stream-chunk append.
- Qwen3-TTS: `request_builders.py:1451-1454` sets `pending_text_queue` from a
  tensor and `tts_pad_embed`.
- Feedback rows are appended after each forward:
  `talker_model_runner.py:157` for Qwen3-Omni and
  `qwen3_tts/model_runner.py:262` for Qwen3-TTS.

`decode_input_embeds` is cleared to `None` at request finish in
`sglang_omni/scheduling/omni_scheduler.py:1677`, which is why
`_decode_input_history` still handles a `None` history even though the dataclass
default is a list.

### A.1 Caller table

Every call site of the six helpers across `sglang_omni` (grep over the package,
then whole-file reads of the calling functions). No module outside these two
files imports them. `dots_tts`, `moss_tts`, `moss_tts_local` and `voxtral_tts`
have their own `pending_feedback_queue` on their own request data
(`moss_tts/request_builders.py:84,96`, `voxtral_tts/request_builders.py:25`) and
never reach `QwenTalkerModelRunner`.

| helper | caller function | file:line | model |
|---|---|---|---|
| `_data_has_next_decode_input` | `QwenTalkerModelRunner.is_decode_batch_ready` | talker_model_runner.py:175 | Qwen3-Omni talker |
| `_data_has_next_decode_input` | `QwenTalkerModelRunner._requests_ready_for_decode` | talker_model_runner.py:400 | Qwen3-Omni talker |
| `_decode_input_history` | `QwenTalkerModelRunner._generated_prefill_slice` | talker_model_runner.py:331 | Qwen3-Omni talker and Qwen3-TTS (shared static) |
| `_decode_input_history` | `QwenTalkerModelRunner._append_decode_input_history` | talker_model_runner.py:435 | both |
| `_append_decode_input_history` | `QwenTalkerModelRunner._generated_prefill_slice` | talker_model_runner.py:343 | both |
| `_append_decode_input_history` | `QwenTalkerModelRunner._write_feedback_buffers` | talker_model_runner.py:372 | Qwen3-Omni talker |
| `_append_decode_input_history` | `Qwen3TTSModelRunner._write_feedback_buffers` | qwen3_tts/model_runner.py:344 | Qwen3-TTS |
| `_peek_next_decode_inputs` | `QwenTalkerModelRunner._combine_feedback_with_next_text` | talker_model_runner.py:483 | Qwen3-Omni talker (and Qwen3-TTS through the prefill replay) |
| `_peek_next_decode_inputs` | `Qwen3TTSModelRunner._write_feedback_buffers` | qwen3_tts/model_runner.py:312 | Qwen3-TTS |
| `_pop_next_decode_inputs` | `QwenTalkerModelRunner._take_next_decode_input_embed` | talker_model_runner.py:512 | both |
| `_pop_next_decode_inputs` | `Qwen3TTSModelRunner._write_feedback_buffers` | qwen3_tts/model_runner.py:325 | Qwen3-TTS |
| `_combine_feedback_with_next_text` | `QwenTalkerModelRunner._take_next_decode_input_embed` | talker_model_runner.py:505 | both |

Indirect reach, for completeness. `_take_next_decode_input_embed`
(talker_model_runner.py:497-513) is the only public-ish entry that combines and
pops in one call. Its callers are `_generated_prefill_slice`
(talker_model_runner.py:333) and `QwenTalkerModelRunner._write_feedback_buffers`
(talker_model_runner.py:365). `_generated_prefill_slice` is reached from
`_projected_prefill_slice` (talker_model_runner.py:271), which Qwen3-TTS calls
directly at `qwen3_tts/model_runner.py:381` inside
`Qwen3TTSModelRunner._build_prefill_input_embeds`. That is how the Qwen3-TTS
prefill path uses `_decode_input_history`, `_append_decode_input_history`,
`_combine_feedback_with_next_text` and `_pop_next_decode_inputs` without naming
them.

One more writer of `decode_input_embeds` that does not go through the helpers,
added by this branch: `_compact_decode_input_history` in
`sglang_omni/scheduling/omni_scheduler.py:87-94`, called from
`OmniScheduler._add_request_to_queue` at omni_scheduler.py:2199. Both AR stages
run `OmniScheduler` (`engine_factory.py:419` and `engine_factory.py:492`), and
the Qwen3-Omni talker runs `QwenTalkerScheduler`, which subclasses it.

### A.2 Per decode step flow, per caller

#### Qwen3-Omni talker

Stage wiring: `talker_ar` in `sglang_omni/models/qwen3_omni/config.py:194-227`,
runner constructed in `qwen3_omni/bootstrap.py:260`.

Gate before the batch runs. `QwenTalkerScheduler._is_batch_ready_to_run`
(talker_scheduler.py:96-108) calls `is_decode_batch_ready`
(talker_model_runner.py:171-177), which walks `schedule_batch.reqs` and reads
`getattr(req, "_omni_data", None)` per request, then
`_data_has_next_decode_input`. That helper returns `False` when
`pending_feedback_queue` is empty, `True` when `pending_text_queue` is non-empty,
and otherwise `bool(thinker_chunks_done and tts_pad_embed is not None)`
(talker_model_runner.py:392-396). A `False` makes `get_next_batch_to_run`
(talker_scheduler.py:110-115) roll the decode prep back through
`_rollback_decode_prep_after_skip` (talker_scheduler.py:117-138) and return
`None`, so no forward runs that iteration. Nothing is written by this path.

Decode step. `ModelRunner._prepare_and_forward` calls `before_decode`
(model_runner/base.py:443). `QwenTalkerModelRunner.before_decode`
(talker_model_runner.py:59-79) re-checks readiness through
`_requests_ready_for_decode` (talker_model_runner.py:398-401), raises if a row is
not ready, then calls `self.model.prepare_decode_buffers(requests)` and
`_write_feedback_buffers(requests)` (talker_model_runner.py:78-79).

`QwenTalkerModelRunner._write_feedback_buffers`
(talker_model_runner.py:353-386) does, per row:

- reads `self.model._feedback_buffer` and `self.model._feedback_mask`
  (talker_model_runner.py:358-359) and clears the mask for the batch
  (talker_model_runner.py:360),
- calls `_take_next_decode_input_embed` with the buffer device and dtype
  (talker_model_runner.py:365-369), which peeks the head of
  `pending_feedback_queue` and the head of `pending_text_queue` (or
  `tts_pad_embed` after the text stream closed), adds the two rows, then pops
  both queues,
- appends the summed row to `decode_input_embeds` through
  `_append_decode_input_history` (talker_model_runner.py:372),
- collects the row index and the tensor.

If every row produced a value the buffer is slice-assigned and the whole mask is
set (talker_model_runner.py:378-383), otherwise the written rows are scattered by
an index tensor (talker_model_runner.py:384-386) and the starved rows stay
unwritten with `mask=False`.

After the forward, `post_decode` (talker_model_runner.py:108-124) reads
`self.model._sampled_token_ids`, stages the ids, then
`_emit_code_chunks_and_feedback` (talker_model_runner.py:126-157) clones
`_output_codes` and `_output_embeds` once per batch and appends
`embeds_snap[idx]` back onto `pending_feedback_queue`
(talker_model_runner.py:157). That append is what makes the next step's
`_data_has_next_decode_input` true again. `post_prefill`
(talker_model_runner.py:81-106) runs the same emit path after the prefill
forward.

Retract replay. On re-prefill, `before_prefill` (talker_model_runner.py:40-57)
calls `_compose_prefill_embeds` (talker_model_runner.py:179-230), which per
request calls `_projected_prefill_slice` (talker_model_runner.py:232-283). When
the extend range runs past the prompt, `_generated_prefill_slice`
(talker_model_runner.py:318-351) takes `_decode_input_history(data)` and, while
the history is shorter than the needed end index, calls
`_take_next_decode_input_embed` and appends the result
(talker_model_runner.py:332-343). A `None` there raises
`"Cannot replay retracted talker decode tokens"` (talker_model_runner.py:338-342).
The rows in `[gen_start:gen_end]` are then re-validated through `_decode_row` and
stacked (talker_model_runner.py:345-351).

#### Qwen3-TTS

Stage wiring: `Qwen3TtsEngineBuilder.make_model_runner`
(`qwen3_tts/engine_builder.py:171-176`) constructs `Qwen3TTSModelRunner`, and the
scheduler is a plain `OmniScheduler` (`engine_factory.py:492`). There is no
`is_decode_batch_ready` gate on this stage, so `_data_has_next_decode_input` and
`_requests_ready_for_decode` never run for Qwen3-TTS.

Decode step. `Qwen3TTSModelRunner.before_decode`
(`qwen3_tts/model_runner.py:59-70`) calls `prepare_decode_buffers` then
`_write_feedback_buffers(forward_batch, requests)`.

`Qwen3TTSModelRunner._write_feedback_buffers`
(`qwen3_tts/model_runner.py:288-348`) targets
`self.model._decode_feedback_embedding.weight[:batch_size]`, not a separate
feedback buffer, and rewrites `forward_batch.input_ids` to the staged row ids at
the end (`qwen3_tts/model_runner.py:348`). Per row:

- `_peek_next_decode_inputs(data)` (qwen3_tts/model_runner.py:312). When it
  returns `None` the row falls back to the ordinary token embedding of
  `input_ids[row_idx]` (qwen3_tts/model_runner.py:314-316) and no queue is
  touched.
- Otherwise the feedback row and the text row are each validated by
  `_decode_row` and appended to two parallel lists
  (qwen3_tts/model_runner.py:318-323), the row index is recorded, and
  `_pop_next_decode_inputs(data)` pops both queues
  (qwen3_tts/model_runner.py:325).

The write itself is one batched path when every row was staged
(`torch.stack(feedback_rows, out=target)` then `target.add_(stack(text_rows))`,
qwen3_tts/model_runner.py:329-331) and a mixed path otherwise, which sums the
staged rows in one stack pair, scatters them into the `rows` list, and stacks the
whole list into `target` (qwen3_tts/model_runner.py:332-339). The buffer is then
cloned once into `history` (qwen3_tts/model_runner.py:342) and every request gets
`history[row_idx]` appended through `_append_decode_input_history`
(qwen3_tts/model_runner.py:343-346), including rows that fell back to the token
embedding.

After the forward, `post_process_outputs` (`qwen3_tts/model_runner.py:239-262`)
clones `_output_codes` and `_output_embeds` for the batch and appends
`embeds_snap[row_idx]` to `pending_feedback_queue`
(qwen3_tts/model_runner.py:262), skipping rows whose sampled id is the codec EOS.

Retract replay. `before_prefill` (`qwen3_tts/model_runner.py:41-57`) calls
`_build_prefill_input_embeds` (`qwen3_tts/model_runner.py:366-397`), which
delegates each request's slice to `QwenTalkerModelRunner._projected_prefill_slice`
(`qwen3_tts/model_runner.py:381`) and therefore to the same
`_generated_prefill_slice` replay described above.

#### ASCII flow of the changed path

```
Qwen3-Omni talker                                Qwen3-TTS
-----------------                                ---------
QwenTalkerScheduler._is_batch_ready_to_run        (no gate)
  talker_scheduler.py:96-108
    is_decode_batch_ready  tmr.py:171-177
      _data_has_next_decode_input tmr.py:388-396
        reads pending_feedback_queue / pending_text_queue
              thinker_chunks_done / tts_pad_embed
              |
              v
ModelRunner._prepare_and_forward  base.py:443
  before_decode tmr.py:59-79                      before_decode qtts.py:59-70
    _requests_ready_for_decode tmr.py:398-401       prepare_decode_buffers
    prepare_decode_buffers                          _write_feedback_buffers qtts.py:288
    _write_feedback_buffers tmr.py:353-386            per row:
      per row:                                         _peek_next_decode_inputs tmr.py:453  <-- new
        _take_next_decode_input_embed tmr.py:497         _peek_left(feedback) tmr.py:461
          _combine_feedback_with_next_text tmr.py:476    _peek_left(text) or tts_pad_embed 464-468
            _peek_next_decode_inputs tmr.py:453  <-- new  _decode_row x2  qtts.py:318-323
            _decode_row(feedback)+_decode_row(text)       _pop_next_decode_inputs tmr.py:471 <-- new
          _pop_next_decode_inputs tmr.py:471   <-- new  batched write qtts.py:327-339
        _append_decode_input_history tmr.py:431         history = target.clone() qtts.py:342
          _decode_input_history tmr.py:423              _append_decode_input_history qtts.py:344
      dense slice-assign or index scatter 378-386       input_ids[:bs].copy_(row_ids) qtts.py:348
              |                                                 |
              v                                                 v
        forward + sample                                  forward + sample
              |                                                 |
              v                                                 v
  post_decode tmr.py:108-124                        post_process_outputs qtts.py:239-262
    _emit_code_chunks_and_feedback tmr.py:126           pending_feedback_queue.append qtts.py:262
      pending_feedback_queue.append tmr.py:157

retract, both stages:
  OmniScheduler._add_request_to_queue omni_scheduler.py:2197-2200   <-- new
    _compact_decode_input_history omni_scheduler.py:87-94           <-- new
      data.decode_input_embeds = list(torch.stack(history).unbind(0))
  before_prefill -> _projected_prefill_slice tmr.py:232
    _generated_prefill_slice tmr.py:318-351
      _decode_input_history tmr.py:331
      _take_next_decode_input_embed tmr.py:333 while history is short
      _append_decode_input_history tmr.py:343

tmr.py  = sglang_omni/models/qwen3_omni/talker_model_runner.py
qtts.py = sglang_omni/models/qwen3_tts/model_runner.py
```

### A.3 What the branch changed, before and after

Base line numbers below come from `git show 7989a5ed2:<path>`.

**1. `_data_has_next_decode_input` (base talker_model_runner.py:386, now 388-396).**
Before, the function read every field with `getattr(data, name, None)` and
compared the result. After, it reads `data.pending_feedback_queue`,
`data.pending_text_queue`, `data.thinker_chunks_done` and `data.tts_pad_embed` as
attributes, and the parameter is typed `SGLangARRequestData | None`. The
`data is None` early return is unchanged. Behaviour differs only for an object
that lacks one of those attributes, which previously read as absent and now
raises `AttributeError`. Executed at serving time by the Qwen3-Omni talker only
(talker_model_runner.py:175 and 400).

**2. `_decode_input_history` (base 426, now 423-429) and
`_append_decode_input_history` (base 434, now 431-435).** Before,
`_decode_input_history` read `getattr(data, "decode_input_embeds", None)`. After,
it reads `data.decode_input_embeds`. The `None` to `[]` initialisation and the
write back to `data.decode_input_embeds` are unchanged, and
`_append_decode_input_history` still appends `row.detach()`. Both signatures are
now typed `SGLangARRequestData`. Executed by both models.

**3. `_peek_next_decode_inputs` and `_pop_next_decode_inputs` are new**
(talker_model_runner.py:453-474). They factor out the peek half and the pop half
of what `_combine_feedback_with_next_text` and
`_take_next_decode_input_embed` previously did inline. `_peek_next_decode_inputs`
returns `(feedback, next_text)` or `None`, with the pad fallback applied when
`pending_text_queue` is empty and `thinker_chunks_done` is set.
`_pop_next_decode_inputs` pops both queues unconditionally through `_pop_left`,
which itself returns early on an empty queue (talker_model_runner.py:404-406).

**4. `_combine_feedback_with_next_text` (base 454, now 476-495).** Before, it
peeked `pending_feedback_queue` through `getattr`, converted the feedback row,
then peeked `pending_text_queue` through `getattr`, applied the pad fallback, and
returned the sum. After, it calls `_peek_next_decode_inputs` and returns the sum
of the two `_decode_row` results. The evaluation order changed: before, the
feedback row was passed through `_decode_row` before the text row was even
peeked, so a device or dtype mismatch on the feedback row raised even when the
text row was missing. After, both `_decode_row` calls happen only once both rows
are known. The pad-fallback rule itself is unchanged.

**5. `_take_next_decode_input_embed` (base 485, now 497-513).** Before, on a
successful combine it popped `pending_feedback_queue` through
`_pop_left(getattr(data, "pending_feedback_queue", None))` and popped
`pending_text_queue` only `if getattr(data, "pending_text_queue", None)`. After,
it calls `_pop_next_decode_inputs`, which pops both. The observable pop set is
the same, because `_pop_left` no-ops on an empty queue.

**6. `Qwen3TTSModelRunner._write_feedback_buffers` (base
qwen3_tts/model_runner.py:288-321, now 288-348).** Before, the loop called
`_take_next_decode_input_embed` per row, fell back to the token embedding when it
returned `None`, called `_append_decode_input_history` with that per-row tensor,
and appended it to a `rows` list. One `torch.stack(rows, out=weight[:batch_size])`
wrote the buffer. Each history entry was therefore the per-row tensor object the
loop produced, which for a staged row was the freshly allocated sum of the two
queue rows.

After, the loop calls `_peek_next_decode_inputs` and, for a staged row, pushes
the feedback row and the text row into two separate lists and records the row
index, then pops both queues. Rows without staged inputs still fall back to the
token embedding, into a preallocated `rows` list of `None`. The sum is then done
in one of two batched forms. When every row is staged,
`torch.stack(feedback_rows, out=target)` followed by
`target.add_(torch.stack(text_rows))` writes the buffer with no per-row addition
(qwen3_tts/model_runner.py:329-331). When only some rows are staged, the staged
rows are summed in one `torch.stack(...) + torch.stack(...)`, scattered into
`rows`, and the whole list is stacked into `target`
(qwen3_tts/model_runner.py:332-339). The history is then taken as one
`target.detach().clone()` and each request gets `history[row_idx]`
(qwen3_tts/model_runner.py:342-346). The in-code note at
qwen3_tts/model_runner.py:340-341 states the reason for the clone: the history
outlives the buffer because a retracted request replays it in its re-prefill.

Two mechanical consequences that follow from the code as written. First, the
history rows of one decode step are now views into one shared clone rather than
independent per-row tensors. Second, every history row is now a copy of what
landed in the embedding weight, including rows that took the token-embedding
fallback, whereas before the fallback row was appended as the tensor the
embedding lookup returned. The values written to the weight are the same in both
versions for the same inputs, subject to floating point ordering, which I did not
test.

**7. `Qwen3TTSModelRunner._write_feedback_buffers` local hoists.** `weight`,
`device` and `dtype` are read once at qwen3_tts/model_runner.py:303-305 instead
of via `decode_feedback_embedding.weight.device` and `.dtype` inside the loop.

**8. `OmniScheduler._add_request_to_queue` and `_compact_decode_input_history`
are new** (omni_scheduler.py:87-94 and 2197-2200). Before, the branch inherited
upstream's `_add_request_to_queue` unchanged. After, when `req.is_retracted` is
true the scheduler calls `_compact_decode_input_history(req._omni_data)`, which
rebuilds `data.decode_input_embeds` as
`list(torch.stack(history).unbind(0))` and returns early on an empty or falsy
history. The docstring at omni_scheduler.py:88-90 states the reason: a decode
input row is a view of the batch snapshot it was written in, so a request that
leaves the running batch would keep every snapshot of its run alive while it
waits. Both AR stages reach this line, because both run `OmniScheduler` and
`QwenTalkerScheduler` inherits it. Note that only the Qwen3-TTS runner produces
rows that are views of a shared clone (item 6). The Qwen3-Omni talker's
`_write_feedback_buffers` still appends the independent per-row tensor it built
(talker_model_runner.py:372), so for that model the compaction copies rows that
were already independent. Whether the Qwen3-Omni talker's history rows alias
anything else at serving time is not established from the code I read, since the
rows come out of `_combine_feedback_with_next_text`, which allocates a fresh sum.

**Which models execute the changed lines at serving time.** The typing changes
(items 1, 2, 4, 5) and the new peek and pop helpers run in both the Qwen3-Omni
talker stage and the Qwen3-TTS AR stage, since both go through
`_take_next_decode_input_embed` or the new helpers directly. Item 6 runs in the
Qwen3-TTS AR stage only. Item 8 runs in every stage that uses `OmniScheduler`,
which is both of these and every other SGLang AR stage in the repo, but it only
does work when `req._omni_data.decode_input_embeds` is non-empty, which is the
two stages that populate it.

### A.4 Test coverage on this branch

Unit tests, `tests/unit_test/qwen3_omni/test_talker.py`. The file imports
`QwenTalkerModelRunner` at test_talker.py:30 and drives the helpers through the
local `_take_decode_input` wrapper at test_talker.py:76-81.

| test | file:line | asserts |
|---|---|---|
| `test_qwen_talker_decode_input_consumes_feedback_and_text_or_pad` | test_talker.py:112-136 | the sum of the feedback row and the text row is returned, both queues drain, and with an empty text queue and `thinker_chunks_done=True` the pad row is used instead |
| `test_qwen_talker_decode_input_consumes_device_text_queue` | test_talker.py:139-155 | FIFO order over a `PendingTextTensorQueue` backed by a 2D tensor, with the queue length dropping by one per take |
| `test_qwen_talker_decode_input_rejects_implicit_row_transfer` | test_talker.py:158-168 | a float64 text row raises `RuntimeError` matching "must already match" |
| `test_qwen_talker_decode_input_preserves_feedback_until_text_arrives` | test_talker.py:171-191 | with no text and `thinker_chunks_done=False` the take returns `None` and both queued feedback rows survive, then the first take after a text append pops exactly one feedback row |
| `test_qwen_talker_decode_readiness_requires_feedback_and_text_or_pad` | test_talker.py:194-217 | `_data_has_next_decode_input` is false with no text and no pad, true with text, true with pad plus `thinker_chunks_done` |
| `test_qwen_talker_decode_inputs_read_the_request_data_as_built` (new on this branch) | test_talker.py:220-244 | on a real `SGLangARRequestData`: readiness and `_peek_next_decode_inputs` are false and `None` on a fresh object, the pad row is returned by identity after a feedback append, `_pop_next_decode_inputs` empties the feedback queue, `_append_decode_input_history` grows `decode_input_embeds`, and `_decode_input_history` re-initialises a `None` history to `[]` and writes it back onto the data |
| `test_write_feedback_buffers_records_decode_input_history` | test_talker.py:2042-2066 | `QwenTalkerModelRunner._write_feedback_buffers` sets the mask, writes the summed row into `_feedback_buffer`, and records the same row in `decode_input_embeds` |
| `test_projected_prefill_retract_replays_generated_decode_inputs` | test_talker.py:2069-2110 | the retract replay concatenates the prompt tail with two history rows plus one row drained from the queues, and both queues end empty with a 3-entry history |
| `test_post_prefill_preserves_prefill_embeds_for_retract` | test_talker.py:1979-1999 | `post_prefill` leaves `prefill_input_embeds` in place |
| `test_projected_prefill_survives_decode_retract` | test_talker.py:2002-2039 | a second prefill after `post_prefill` still yields the full projected embeds |

`tests/unit_test/qwen3_omni/test_talker_feedback_write.py` covers the
Qwen3-Omni write path directly. `test_dense_write_skips_index_tensor`
(test_talker_feedback_write.py:41-64) monkeypatches
`talker_model_runner.torch.tensor` and asserts it is never called on the dense
path, that the mask is all true, and that each buffer row equals feedback plus
text. `test_sparse_write_leaves_starved_row_unwritten`
(test_talker_feedback_write.py:67-85) asserts the mask is
`[True, False, True]` and the starved row stays zero. This branch added
`decode_input_embeds=[]` to the fake request data at
test_talker_feedback_write.py:31, which is the only change to that file.

`tests/unit_test/qwen3_omni/test_talker_row_ownership.py` covers row identity
across prep and emit. `test_row_ownership_survives_prep_then_emit`
(test_talker_row_ownership.py:65-93) asserts the buffer rows and then the emitted
code chunks and the re-appended feedback rows line up by request index.
`test_sparse_feedback_row_stays_unwritten` (96-114) and
`test_stale_mask_cannot_leak_into_reused_slot` (117-135) assert the mask and
buffer state for starved rows and for a pre-set stale mask.
`test_row_ownership_tracks_current_batch_order_across_steps` (138-246) drives
three steps with reordered and shrinking batches and asserts, per step, the
buffer contents, the emitted message order, and that each request's
`pending_feedback_queue` holds exactly its own new row. This branch added
`decode_input_embeds=[]` at test_talker_row_ownership.py:50, its only change.

`tests/unit_test/qwen3_omni/test_talker_emit_snapshot.py` asserts the emit-side
snapshot contract: `test_emitted_rows_survive_next_step_inplace_write`
(test_talker_emit_snapshot.py:48-67) mutates `_output_codes` and `_output_embeds`
in place after the emit and asserts the emitted rows are unchanged, and
`test_two_batched_clones_rows_share_storage` (70-100) counts `torch.Tensor.clone`
calls (exactly 2) and asserts all emitted code rows share one storage and all
feedback rows share one storage. Unchanged by this branch.

`tests/unit_test/qwen3_omni/test_talker_prefill_sidecar.py` covers
`before_prefill` and the projected-slice composition (176 lines, seven tests
including `test_sidecar_composes_the_logical_rows_for_each_request` at
test_talker_prefill_sidecar.py:113-128). It does not reach the six helpers.
Unchanged by this branch.

`tests/unit_test/qwen3_tts/test_retract_prefill.py` covers the Qwen3-TTS side.

| test | file:line | asserts |
|---|---|---|
| `test_write_feedback_buffers_records_decode_input_history` | test_retract_prefill.py:57-85 | one staged row lands as `[21.0, 32.0]` in both `decode_input_embeds[0]` and `embedding.weight[0]`, `input_ids` becomes `[0]`, and both queues are drained |
| `test_write_feedback_buffers_batches_staged_rows_and_embeds_the_rest` (new on this branch) | test_retract_prefill.py:88-132 | a three-row batch mixing a staged row, a pad-fallback row and a first-step row with no feedback produces `[[21,32],[3.5,4.5],[6,7]]` in the weight, one history row per request equal to the matching weight row, `input_ids` rewritten to `[0,1,2]`, and the un-consumed text row left on the first-step request's queue |
| `test_reprefill_after_retract_replays_prompt_plus_generated` | test_retract_prefill.py:135-162 | the 460 plus 134 replay shape from issue #1555, with the history rows in the middle and the drained leftover feedback row last |
| `test_reprefill_replays_prompt_tail_and_generated_tail` | test_retract_prefill.py:234-247 | prompt tail plus three history rows |
| `test_reprefill_drains_leftover_feedback_when_history_is_short` | test_retract_prefill.py:250-274 | the replay drains one more decode input from the queues and grows the history to 2 |
| `test_reprefill_without_generated_history_fails_loudly` | test_retract_prefill.py:277-283 | `RuntimeError` matching "missing feedback/text input embeds" |
| `test_decode_then_retract_reprefill_roundtrip` | test_retract_prefill.py:286-335 | after N decodes the history is N and one feedback row is queued, and the re-prefill yields prompt plus N history rows plus the leftover |
| `test_fresh_prefill_still_uses_prompt_only_buffer` | test_retract_prefill.py:338-345 | a fresh prefill slices the prompt only |
| `test_reprefill_restores_retained_repetition_penalty_history` | test_retract_prefill.py:165-231 | the penalizer seeding path, unrelated to the helpers |

`tests/unit_test/pipeline/test_scheduler.py` covers the new scheduler hook, both
tests added by this branch.
`test_retracted_request_history_gets_its_own_storage_before_requeue`
(test_scheduler.py:429-478) builds three snapshot tensors, gives a retracted
request rows sliced out of them, calls
`OmniScheduler._add_request_to_queue(scheduler, retracted, is_retracted=True)`
and then the same for a non-retracted request, then asserts the retracted
request's history values are preserved, that all its rows now share exactly one
storage, that this storage is disjoint from the snapshot storages, that the
storage is exactly the size of the stacked history, and that the non-retracted
request's rows still point at the original snapshot storages.
`test_retracted_request_without_history_is_requeued_untouched`
(test_scheduler.py:481-489) asserts an empty history is left as `[]` and the
request is still queued.

Accelerator tests. `pytest.mark.accelerator` is registered at
`pyproject.toml:147`. In `tests/unit_test/qwen3_omni/` the marks sit at
test_talker.py:606, 674, 716, 754 and 2291, at
test_talker_token_readback.py:163, test_talker_attention.py:29,
test_audio_feature_packing.py:67 and 78, test_audio_layer_graph.py:235-297,
test_code2wav_cuda_graph.py:411-487, test_code2wav_overlap.py:896-1060 and
test_thinker_fused_rope.py:280 and 346. The talker-runner-adjacent ones are:

- `test_qwen_predictor_decode_graph_uses_configured_batch_buckets`
  (test_talker.py:606-671), asserts a live batch of 3 replays the bucket-4 graph
  and that no exact key for 3 is created.
- `test_qwen_predictor_decode_graph_matches_eager` (test_talker.py:674-713),
  asserts graph replay matches the eager incremental predictor for codes and
  embeds and that the key `(2, torch.int)` exists.
- `test_qwen_predictor_decode_graph_covers_real_incremental_step`
  (test_talker.py:716-751), same equality against the real step and KV cache
  path.
- `test_qwen_predictor_decode_graph_uses_tensor_device_when_current_device_differs`
  (test_talker.py:754-794), two-GPU test asserting capture follows the tensor
  device and leaves the process-current device at 0.
- `test_talker_prepare_decode_buffers_cuda_matches_fresh_rebuild`
  (test_talker.py:2291-2373), asserts ten sampling and mask buffers on a reused
  fake match a fresh rebuild after two batch turnovers.
- `test_stage_token_ids_cuda_matches_reference`
  (test_talker_token_readback.py:163-174), asserts the staged host copy of a CUDA
  token tensor is on host and equal.

None of these accelerator tests exercise the six decode-input helpers. I found no
accelerator-marked test anywhere in `tests/unit_test/qwen3_omni/` or
`tests/unit_test/qwen3_tts/` that calls `_peek_next_decode_inputs`,
`_pop_next_decode_inputs`, `_combine_feedback_with_next_text`,
`_take_next_decode_input_embed`, `_decode_input_history` or
`_append_decode_input_history`. The helpers are covered by CPU unit tests only.

The end-to-end model tests that exercise the Qwen3-Omni talker at serving time
live in `tests/test_model/`: `test_qwen3_omni_tts_ci.py`,
`test_qwen3_omni_mmmu_talker_ci.py`, `test_qwen3_omni_mmsu_talker_ci.py`,
`test_qwen3_omni_videomme_talker_ci.py`,
`test_qwen3_omni_videoamme_talker_tp2_ci.py`, plus the text-path
`test_qwen3_omni_mmmu_ci.py`, `test_qwen3_omni_mmsu_ci.py`,
`test_qwen3_omni_videomme_ci.py`, `test_qwen3_omni_videoamme_ci.py`,
`test_qwen3_omni_thinker_length.py`, `test_qwen3_omni_realtime.py`,
`test_qwen3_omni_realtime_audio.py` and
`test_qwen3_omni_process_replicas.py`. I listed these by filename and by the
stage table quoted in Part B.3 and did not read their assertions in this pass.
UNVERIFIED: what each of those files asserts.

---

## Part B, benchmark protocol

### B.1 Benchmark scripts under `benchmarks/`

The Qwen3-Omni entry points, from `benchmarks/README.md:178-191` and from the
files themselves:

| script | task | input to output | API |
|---|---|---|---|
| `benchmarks/eval/benchmark_omni_seedtts.py` | SeedTTS speed plus WER | text (optional reference audio) to speech | `/v1/chat/completions` |
| `benchmarks/eval/benchmark_omni_mmsu.py` | MMSU audio comprehension | text or text plus audio to text | `/v1/chat/completions` |
| `benchmarks/eval/benchmark_omni_mmau.py` | MMAU audio comprehension | audio to text | `/v1/chat/completions` |
| `benchmarks/eval/benchmark_omni_mmar.py` | MMAR audio reasoning | audio to text | `/v1/chat/completions` |
| `benchmarks/eval/benchmark_omni_mmmu.py` | MMMU accuracy and speed | image plus text to text | `/v1/chat/completions` |
| `benchmarks/eval/benchmark_omni_videomme.py` | Video-MME | video to text (audio optional) | `/v1/chat/completions` |
| `benchmarks/eval/benchmark_omni_videoamme.py` | Video-AMME | video plus spoken question to text (audio optional) | `/v1/chat/completions` |
| `benchmarks/eval/benchmark_omni_streaming_ttft.py` | streaming time to first audio chunk | text to text plus audio, streamed | `/v1/chat/completions` |
| `benchmarks/eval/benchmark_omni_rollout_stress.py` | closed-loop concurrency sweep on one repeated prompt | multimodal MMMU sample to text plus audio by default | `/v1/chat/completions` |

**Speech in, speech out.** `benchmark_omni_seedtts.py` is the one that drives the
talker end to end. Its config dataclass is `OmniSeedttsBenchmarkConfig`
(benchmark_omni_seedtts.py:186-215) and its flags are declared in
`_build_arg_parser` (benchmark_omni_seedtts.py:445-616):

`--base-url`, `--host`, `--port` (default 8000), `--model` (default
`qwen3-omni`), `--meta` with alias `--testset` (default
`zhaochenyang20/seed-tts-eval-arrow`), `--lang` (`en` or `zh`), `--speaker`
(`Ethan`, `Chelsie`, `Aiden`), `--voice-clone` and its legacy complement
`--no-ref-audio`, `--output-dir` (default `results/omni_seedtts`),
`--max-samples`, `--max-new-tokens` (default 256), `--temperature` (default 0.7),
`--stream`, `--warmup` (default `None`, which resolves to the configured
concurrency), `--max-concurrency` (default from env
`TTS_BENCHMARK_CONCURRENCY`, else 16, benchmark_omni_seedtts.py:183),
`--request-rate` (default infinity), `--save-audio` (a documented no-op),
`--disable-tqdm`, `--device` and its alias `--asr-device`, `--asr-model-path`,
`--asr-concurrency`, `--similarity-checkpoint`, `--server-timeout` (default
1200), `--system-prompt`, `--with-similarity`, and a mutually exclusive mode
group of `--generate-only`, `--transcribe-only`, `--similarity-only` and
`--utmos-only`. There is no `--seed` flag on this script. The docstring's own
usage block is benchmark_omni_seedtts.py:13-45.

`main` (benchmark_omni_seedtts.py:619-664) waits for the service, runs the
generation phase, then unless `--generate-only` runs the ASR transcription phase
and writes `eval_results.json` with `generation.speed`, `generation.config`,
`generation.per_request`, `accuracy.wer` and `asr.speed`, plus `similarity` when
`--with-similarity` is set.

**Speech in, text out.** `benchmark_omni_mmsu.py` posts audio plus text and reads
text. Its flags are in `main` (benchmark_omni_mmsu.py:223-267): `--base-url`,
`--host`, `--port`, `--model`, `--modalities` (`text` or `text+audio`),
`--output-dir` (default `results/mmsu`), `--max-samples`, `--task-names`,
`--categories`, `--prompt`, `--max-tokens` (default 32), `--temperature`
(default 0.0), `--warmup`, `--max-concurrency` (default 32), `--request-rate`,
`--timeout-s` (default 300), `--save-audio`, `--disable-tqdm`, `--seed`,
`--repo-id`, `--lang`, `--asr-device`, `--asr-concurrency`. Its output dict is
`accuracy`, `speed`, `per_sample`, plus `wer` when the audio modality is on
(benchmark_omni_mmsu.py:183-196).

**Streaming TTFT.** `benchmark_omni_streaming_ttft.py` measures wall time from
request submission to the first audio delta with `modalities=["text","audio"]`
and `stream=true` (benchmark_omni_streaming_ttft.py:96-140). Flags:
`--base-url` (default `http://localhost:8000`), `--model` (default
`qwen3-omni`), `--label` (required), `--output` (default
`results/ttft_<label>_<run-id>.json`), `--warmup` (default 2), `--repeats`
(default 5), `--timeout-s` (default 300). Seeds are fixed in the script:
`9000 + warm` for warmups and `1000 + repeat` for measured runs
(benchmark_omni_streaming_ttft.py:150 and 176). Two fixed prompts, `short` and
`medium` (benchmark_omni_streaming_ttft.py:52-60). Per prompt it emits
`ttft_mean`, `ttft_min`, `ttft_max`, `ttft_stdev` and `total_mean`
(benchmark_omni_streaming_ttft.py:210-216), plus every run's `ttft_seconds`,
`total_seconds`, `audio_chunks` and `status_code`
(benchmark_omni_streaming_ttft.py:63-72). Its own docstring documents an A/B
form against `--no-enable-partial-start` and `--enable-partial-start
--partial-start-min-chunks 5` (benchmark_omni_streaming_ttft.py:11-25).

**Concurrency sweep.** `benchmark_omni_rollout_stress.py` reuses one prompt
across a sweep of concurrency levels and reports per-level requests per second,
output tokens per second, prompt token counts and p50, p95 and p99 latency, plus
server profiler events (benchmark_omni_rollout_stress.py:3-23). Flags at
benchmark_omni_rollout_stress.py:300-324: `--base-url`, `--host`, `--port`
(8000), `--model` (`qwen3-omni`), `--repo-id` (default
`zhaochenyang20/mmmu-ci-50`), `--sample-index`, `--prompt-override`,
`--rollout-counts`, `--rollout-group-id`, `--max-tokens` (256), `--temperature`
(0.8), `--text-only`, `--talker-max-new-tokens`, `--timeout-s` (300),
`--output-dir` (default `results/rollout_stress`), `--profile-run-id`,
`--profile-event-dir`, `--no-profile` and `--disable-tqdm`. It writes
`<output-dir>/rollout_stress_results.json`. There is no `--seed` flag.

**Runner semantics shared by all of the above except the TTFT script.**
`BenchmarkRunner` in `benchmarks/benchmarker/runner.py:45-148`. Warmup resolves
`None` to the configured concurrency (`resolve_warmup`, runner.py:23-29) and
repeats `samples[0]` for the warmup cohort rather than distinct samples
(runner.py:99-113, with the reason in the note at runner.py:105-109). Concurrency
is an `asyncio.Semaphore` (runner.py:121-125) and `max_concurrency=0` means
open loop with `aiohttp.TCPConnector(limit=0)` (runner.py:66-71).
`wall_clock_s` covers the measured dispatch only (runner.py:80-82).

**Metrics emitted.** `benchmarks/metrics/performance.py:114-225` builds the
summary. Always present on a successful run: `total_requests`,
`completed_requests`, `failed_requests`, `latency_mean_s`, `latency_median_s`,
`latency_p95_s`, `latency_p99_s`, `audio_duration_mean_s`, `rtf_mean`,
`rtf_median`, `rtf_p95`, `rtf_p99`, `throughput_qps`, plus the token metrics
`output_tok_per_req_s`, `output_throughput`, `output_tokens_mean`,
`output_tokens_total`, `prompt_tokens_mean` and `prompt_tokens_total` when the
underlying counts are non-zero (performance.py:80-111). Conditionally present:
`audio_throughput_s_per_s`, the TTFC block `audio_ttfp_mean_s` and its median,
p95 and p99, the TTFT block `text_ttft_mean_s` and its median, p95 and p99, the
ITL block `inter_chunk_mean_s`, `inter_chunk_p95_s` and `inter_chunk_p99_s`,
`audio_chunks_mean` and `audio_chunks_p95`,
`first_audio_payload_bytes_mean` and its p95, and the playback continuity block
`max_playback_underrun_*`, `c50`, `c100`, `c200`,
`playback_continuity_requests` and `playback_continuity_na_requests`
(performance.py:180-224). The precise definitions are in the module docstring at
performance.py:5-59. Per request, `_request_result_to_dict`
(performance.py:391-422) writes `id`, `text`, `is_success`, `latency_s`,
`audio_duration_s`, `rtf`, `prompt_tokens`, `completion_tokens`,
`output_token_rate`, `wav_path`, `error`, `audio_ttfp_s`, `text_ttft_s`,
`inter_chunk_s`, `chunk_audio_duration_s`, `max_playback_underrun_s`,
`audio_chunk_count` and `first_audio_payload_bytes`.

**Dataset preparation.** `python -m benchmarks.dataset.prepare --dataset
<name>` with the names listed at benchmarks/README.md:339-353, including
`seedtts`, `seedtts-mini`, `seedtts-50`, `mmsu`, `mmmu`, `mmmu-ci-50`,
`videomme`, `videomme-ci-50` and `videoamme-ci-50`.

**Docs.** `docs/basic_usage/qwen3_omni.md` documents the serve forms and the
speech-stage placement experiment. `docs/cookbook/qwen3_omni.md` exists (112
lines) and was not read in this pass. UNVERIFIED: its contents.

### B.2 Starting a Qwen3-Omni server

**Default speech pipeline, no config file.** `docs/basic_usage/qwen3_omni.md:228-233`:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008
```

The shipped default profile is `Qwen3OmniSpeechPipelineConfig`
(`sglang_omni/models/qwen3_omni/config.py:341-380`), which the module exports as
`EntryClass` (config.py:407) and registers as the `speech` variant
(config.py:409-413). Its stage list is `_speech_stages(thinker_gpu=0,
talker_gpu=1, process_by_stage=_SPEECH_DEFAULT_PROCESSES,
enable_partial_start=True)` (config.py:361-368).

Seven stages, seven processes (`_SPEECH_DEFAULT_PROCESSES`, config.py:290-298):

| stage | process | gpu | definition |
|---|---|---|---|
| `preprocessing` | `preprocessing` | CPU stage, no gpu field | config.py:42-72, 261-264 |
| `image_encoder` | `image_encoder` | `thinker_gpu` = 0 | config.py:93-102, 265-269 |
| `audio_encoder` | `audio_encoder` | `thinker_gpu` = 0 | config.py:105-116, 270-274 |
| `thinker` | `thinker` | 0 | config.py:133-172, 275-279 |
| `decode` | `decode` | CPU stage, terminal | config.py:175-182, 280 |
| `talker_ar` | `talker_ar` | `talker_gpu` = 1 | config.py:194-227, 281-285 |
| `code2wav` | `code2wav` | `thinker_gpu` = 0 | config.py:230-239, 286 |

GPU count for the default speech profile: two, GPU 0 carrying image encoder,
audio encoder, thinker and code2wav, and GPU 1 carrying the talker. The
docstring at config.py:342 calls it a 7-stage speech pipeline. The talker stage
sets `max_seq_len=32768`, `enable_partial_start=True` and
`partial_start_min_chunks=5` (config.py:215-219), and `code2wav` carries
`gpu_memory_fraction=0.02` (config.py:236). The talker's runner is constructed
with `speech_enabled=True` and `feedback_enabled=True` (config.py:370-375), which
is what turns on the feedback path in `QwenTalkerModelRunner.before_decode`
(talker_model_runner.py:70-71).

`docs/basic_usage/qwen3_omni.md:296-312` states the default topology in prose:
thinker alone, talker alone, code2wav on the thinker's GPU, and records that
code2wav sharing the thinker's GPU costs the thinker about 4.3 GiB of
auto-sized KV on H200.

**Colocated one-GPU profile.** `Qwen3OmniSpeechColocatedPipelineConfig`
(config.py:383-404) places thinker and talker both on GPU 0 and sets
`enable_partial_start=False`. It is selected by config file, for example
`examples/configs/qwen3_omni_colocated_h200.yaml`, which sets
`config_cls: Qwen3OmniSpeechColocatedPipelineConfig` and the per-stage budgets
`image_encoder 0.017`, `audio_encoder 0.017`, `thinker 0.769`, `talker_ar 0.123`,
`code2wav 0.014`. The documented launch, `docs/basic_usage/qwen3_omni.md:167-172`:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --port 8008
```

One GPU. The other colocated configs present are
`qwen3_omni_colocated_h100_bf16.yaml`, `qwen3_omni_colocated_h100_fp8.yaml`,
`qwen3_omni_colocated_h20.yaml`, `qwen3_omni_colocated_gfx950_bf16.yaml`,
`qwen3_omni_colocated_h200.yaml` and `qwen3_omni_fp8_colocated.yaml`.

**Text-only.** `--text-only` selects the 6-stage
`Qwen3OmniPipelineConfig` (config.py:328-338) with every stage on GPU 0
(`_text_stages`, config.py:242-250). One GPU. Documented at
`docs/basic_usage/qwen3_omni.md:15-19`, with the fused MMSU variant at
`docs/basic_usage/qwen3_omni.md:25-31` using
`examples/configs/qwen3_omni_mmsu.yaml`.

**Three-GPU replica profile.** `examples/configs/qwen3_omni_speech_replica2.yaml`
documents its own layout in the header: GPU 0 thinker, GPU 1
`talker_ar@r0` plus `code2wav@r0`, GPU 2 `talker_ar@r1` plus `code2wav@r1`, with
`num_replicas: 2` and `replica_devices: [1, 2]` for both stages. Its serve line
is `sgl-omni serve --config
examples/configs/qwen3_omni_speech_replica2.yaml --port 8091`.

**Explicit per-stage placement.** `examples/run_qwen3_omni_speech_server.py`
(19 lines) forwards to the `qwen3-speech-server` preset in
`examples/launchers/qwen3_omni.py:539-544`. The documented invocation,
`docs/basic_usage/qwen3_omni.md:218-226`:

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 0 \
  --port 8008
```

The preset description at `examples/launchers/qwen3_omni.py:32-46` shows a TP2
form with `--thinker-tp-size 2 --gpu-thinker-tp 0,1 --gpu-talker 2
--gpu-code2wav 0`.

Memory flags: `--mem-fraction-static` applies to every SGLang engine stage, and
dotted per-stage paths such as `--thinker.engine.mem_fraction_static` and
`--talker_ar.engine.mem_fraction_static` override it
(`docs/basic_usage/qwen3_omni.md:240-268`). The thinker admits 64 running
requests by default and is changed by
`--thinker.engine.max_running_requests` (`docs/basic_usage/qwen3_omni.md:277-284`).

### B.3 Earlier A/B protocol recorded in the analysis worktree

All paths below are under
`/Users/ratish/sglang-omni/.worktrees/qwen3-omni-0518-numerics/tasks/`.

**The slim A/B driver.**
`qwen3_omni_0518_numerics/scripts/slim_ab.sh:1-24` states the shape and the
invocation:

```
# The slim A/B: three workloads on a manually started server, no router, no
# pytest. Per arm, one fp8 colocated boot on one GPU (the H100 serving
# profile, examples/configs/qwen3_omni_colocated_h100_fp8.yaml) takes the
# voice clone bench at c1, c16 and c32, MMSU at c16 and Video-MME with the
# talker at c16, in that order, so the finished output window carries
# across workloads as it does on a production server. Then `score` runs
# the CI's WER and UTMOS scorers on the voice clone outputs against a
# Qwen3-ASR server. Results land in $OUT/<stage>/<arm>/ so
# full_ab_compare.py reads them.
#
# PROFILE=bf16 runs the CI's bf16 topologies instead: bf16 colocated for
# the voice clone, bf16 thinker only for MMSU, bf16 disagg (two GPUs) for
# Video-MME, three boots per arm. SLIM_STAGES="seedtts" limits either
# profile to the voice clone stage:
#   PROFILE=bf16 SLIM_STAGES=seedtts OMNI_ROOT=... OUT=... GPU=0 bash "$OUT/scripts/slim_ab.sh"
#
# Required: OMNI_ROOT (clean tree, installed editable), OUT (outside the
# tree), GPU (default 0, two GPUs for PROFILE=bf16). Optional: A_SHA, B_SHA.
# Run from a copy outside the tree:
#   cp -r "$OMNI_ROOT/tasks/qwen3_omni_0518_numerics/scripts" "$OUT/scripts"
#   OMNI_ROOT=... OUT=... GPU=0 bash "$OUT/scripts/slim_ab.sh"
#   OMNI_ROOT=... OUT=... GPU=0 bash "$OUT/scripts/slim_ab.sh" score
#   python "$OUT/scripts/full_ab_compare.py" "$OUT" --md "$OUT/readout.md"
```

The fp8 arm body, slim_ab.sh:85-95:

```bash
  if [ "$PROFILE" = fp8 ]; then
    gpus_idle || return 1
    GPU=$GPU_ONE serve_fp8_colocated 31000 || return 1
    if wants seedtts; then for c in 1 16 32; do bench "seedtts_c$c" "$arm" seedtts-vc 31000 "$c"; done; fi
    if wants mmsu; then bench mmsu_c16 "$arm" mmsu 31000 16; fi
    if wants videomme; then bench videomme_talker_c16 "$arm" videomme-talker 31000 16; fi
    stop_server 31000
```

The bf16 arm body, slim_ab.sh:98-118, boots three servers per arm:
`serve_bf16_colocated 31000` for the three seedtts points,
`serve_bf16_thinker 31001` for MMSU and `serve_bf16_disagg 31002` for
Video-MME with the talker.

The per-run invocation, slim_ab.sh:63-73:

```bash
bench() {
  # $1 stage dir name, $2 arm, $3 run_bench task, $4 port, $5 concurrency
  local dir="$OUT/$1/$2"
  mkdir -p "$dir"
  echo "A B" > "$OUT/$1/order.txt"
  git -C "$OMNI_ROOT" rev-parse HEAD > "$dir/git_head.txt"
  (cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" "$3" \
    --port "$4" --out "$dir" --concurrency "$5" --top-logprobs 0) > "$dir/bench.log" 2>&1
  echo $? > "$dir/exit_code"
  log "$1 $2 exit $(cat "$dir/exit_code")"
}
```

The scoring pass, slim_ab.sh:121-136:

```bash
run_score() {
  local port=31011 dir
  gpus_idle || return 1
  GPU=$GPU_ONE _launch_server $port "serve_asr_$port.log" sgl-omni serve \
    --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $port || return 1
  for dir in "$OUT"/seedtts_c*/[AB]/seedtts_vc; do
    [ -f "$dir/speed_results.json" ] || continue
    log "score $dir"
    (cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python -m benchmarks.eval.benchmark_omni_seedtts \
      --transcribe-only --meta zhaochenyang20/seed-tts-eval-50-arrow --output-dir "$dir" \
      --model qwen3-omni --lang en --port $port) > "$dir/../transcribe.log" 2>&1
    (cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python -m benchmarks.eval.benchmark_omni_seedtts \
      --utmos-only --output-dir "$dir" --device "cuda:0") > "$dir/../utmos.log" 2>&1
  done
  stop_server $port
}
```

**The serve functions.**
`qwen3_omni_0518_numerics/scripts/h100_runs.sh:46-53` fixes the models and
configs:

```bash
FP8_MODEL=marksverdhei/Qwen3-Omni-30B-A3B-FP8
BF16_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
FP8_CONFIG=examples/configs/qwen3_omni_colocated_h100_fp8.yaml
BF16_THINKER_CONFIG=examples/configs/qwen3_omni_mmmu_h100.yaml
BF16_COLOCATED_CONFIG=examples/configs/qwen3_omni_colocated_h100_bf16.yaml
THINKER_MAX_SEQ_LEN=32768
```

h100_runs.sh:120-161:

```bash
serve_fp8_colocated() {
  local port=$1
  shift
  _launch_server "$port" "serve_fp8_$port.log" sgl-omni serve \
      --model-path "$FP8_MODEL" --host 127.0.0.1 --port "$port" --model-name qwen3-omni \
      --config "$FP8_CONFIG" --colocate \
      --preprocessing.factory.max_seq_len "$THINKER_MAX_SEQ_LEN" \
      --thinker.factory.max_seq_len "$THINKER_MAX_SEQ_LEN" \
      "$@"
}

serve_bf16_thinker() {
  ... --config "$BF16_THINKER_CONFIG" ...
}

# Voice clone stage server: the bf16 colocated speech config of
# test_qwen3_omni_tts_ci.py on one GPU (CI runs two of these behind a router).
serve_bf16_colocated() {
  ... --config "$BF16_COLOCATED_CONFIG" --colocate ...
}

serve_bf16_disagg() {
  local port=$1
  shift
  _launch_server "$port" "serve_bf16_disagg_$port.log" python examples/run_qwen3_omni_speech_server.py \
      --model-path "$BF16_MODEL" --port "$port" --model-name qwen3-omni \
      --thinker-max-seq-len "$THINKER_MAX_SEQ_LEN" \
      --gpu-thinker 0 --gpu-image-encoder 0 --gpu-audio-encoder 0 --gpu-talker 1 --gpu-code2wav 1 \
      --thinker-mem-fraction-static 0.82 --talker-mem-fraction-static 0.40 \
      "$@"
}
```

and h100_runs.sh:166-176 for the FP8 thinker TP2 stage:

```bash
serve_fp8_tp2_disagg() {
  ... python examples/run_qwen3_omni_speech_server.py \
      --model-path "$FP8_MODEL" ... \
      --thinker-tp-size 2 --gpu-thinker-tp 0,1 --gpu-talker 1 --gpu-code2wav 1 \
      --thinker-mem-fraction-static 0.40 --talker-mem-fraction-static 0.21 ...
}
```

Server readiness is a `/health` poll every 5 s up to `WAIT_READY_TRIES`
(default 180) at h100_runs.sh:176-190, and `stop_server` ends the whole process
group and waits for the port to release (h100_runs.sh:192 onward). `slim_ab.sh`
refuses to run when `OMNI_ROOT` has uncommitted changes and restores the
original ref on exit (slim_ab.sh:143-149).

**The per-workload benchmark wrapper.**
`qwen3_omni_0518_numerics/scripts/run_bench.py:1-24` states what each task
mirrors:

```
Usage:
    python run_bench.py videoamme --port P --out DIR [--concurrency 16] [--top-logprobs 5]
    python run_bench.py videomme-talker --port P --out DIR [--concurrency 16] [--max-samples 20]
    python run_bench.py mmsu --port P --out DIR [--concurrency 16] [--max-samples N]
    python run_bench.py seedtts-vc --port P --out DIR [--concurrency 16] [--max-samples 50]

videoamme mirrors tests/test_model/test_qwen3_omni_videoamme_ci.py (stage 9),
videomme-talker mirrors test_qwen3_omni_videomme_talker_ci.py (stage 8) with
the same short-answer prompt and speech output but without the inline WER
pass ..., and mmsu
mirrors test_qwen3_omni_mmsu_ci.py (stage 5, text only), and seedtts-vc
mirrors the voice clone speed benchmark of test_qwen3_omni_tts_ci.py (50
SeedTTS-50 samples, one warmup request, speech output) without its WER, UTMOS
and similarity passes, so it exercises the talker and code2wav with a short
text prompt only.
```

The seedtts-vc call it builds (run_bench.py:150-173) uses
`DATASETS["seedtts-50"]`, `max_samples=50`, `voice_clone=True`,
`disable_tqdm=True` and the passed concurrency, and prints
`qps`, `latency_mean_s`, `rtf_mean` and `failed`. The mmsu call
(run_bench.py:114-147) pins `modalities="text"`, `max_tokens=32`,
`temperature=0.0`, `warmup=0`, `repo_id=DATASETS["mmsu-ci-2000"]` and
`timeout_s=300`. The videomme-talker call (run_bench.py:68-111) pins
`max_tokens=256`, `enable_audio=True`, `video_fps=2`, `video_max_frames=128`,
`video_max_pixels=401408`, `timeout_s=500` and adds the short-answer prompt at
run_bench.py:73-76.

**The CI stage table this protocol mirrors.**
`qwen3_omni_0518_numerics/09_reservation_flow_full_ab.md:192-203`:

| stage | test file | server | input to output | samples at c16 | what the test asserts |
|---|---|---|---|---|---|
| thinker_length | test_qwen3_omni_thinker_length.py | bf16 thinker TP2, max_seq_len 128 | text | contract posts | finish_reason and context length HTTP contract |
| tts | test_qwen3_omni_tts_ci.py | bf16 colocated, router with two workers | ref audio and text to speech | 50 | speed P95 gates, WER, UTMOS, similarity (disabled, issue #483) |
| mmmu | test_qwen3_omni_mmmu_ci.py | fp8 colocated, two workers | image and text to text | 50 | accuracy 0.6, speed P95 gates |
| mmmu_talker | test_qwen3_omni_mmmu_talker_ci.py | bf16 disagg, two GPUs | image and text to text and speech | 20 | accuracy 0.7, WER, speed and rtf gates |
| mmsu | test_qwen3_omni_mmsu_ci.py | bf16 thinker only, two workers | audio and text to text | 2000 | accuracy 0.7035, speed P95 gates |
| mmsu_talker | test_qwen3_omni_mmsu_talker_ci.py | fp8 thinker TP2 | audio and text to text and speech | 40 | accuracy 0.625, WER, speed and rtf gates |
| videomme | test_qwen3_omni_videomme_ci.py | bf16 disagg | video to text | 50 | accuracy 0.58, speed P95 gates |
| videomme_talker | test_qwen3_omni_videomme_talker_ci.py | bf16 disagg | video to text and speech | 20 | accuracy 0.6, WER, speed and rtf gates |
| videoamme | test_qwen3_omni_videoamme_ci.py | fp8 colocated, two workers | video with audio to text | 50 | accuracy 0.62, speed P95 gates |
| videoamme_talker_tp2 | test_qwen3_omni_videoamme_talker_tp2_ci.py | fp8 thinker TP2 | video with audio to text and speech | 10 | accuracy 0.5, WER, speed and rtf gates |

The same doc records the slim pass composition
(09_reservation_flow_full_ab.md:251-256):

| stage | concurrency | what it reads |
|---|---|---|
| seedtts_c1, c16, c32 | 1, 16, 32 | speed, then WER and UTMOS from the score pass |
| mmsu_c16 | 16, the 2000 clips | accuracy, speed |
| videomme_talker_c16 | 16, 20 clips, speech output | thinker text accuracy, rtf, latency, qps |

and its own wall-time note at 09_reservation_flow_full_ab.md:266: "Wall time on
fp8 is about 35 minutes per arm plus 15 minutes of scoring."

**Recorded readouts.** `qwen3_omni_0518_numerics/09_reservation_flow_full_ab.md:283-289`,
the fp8 slim run:

| run | metric | A | B |
|---|---|---|---|
| seedtts c1 | qps, latency p95 s, WER, UTMOS | 1.530, 1.001, 0.0142, 4.445 | 1.519, 1.064, 0.0160, 4.471 |
| seedtts c16 | qps, latency p95 s, WER, UTMOS | 8.976, 2.439, 0.0089, 4.462 | 8.272, 2.869, 0.0106, 4.471 |
| seedtts c32 | qps, latency p95 s, WER, UTMOS | 11.535, 3.059, 0.0177, 4.441 | 11.455, 3.154, 0.0089, 4.463 |
| mmsu c16 | accuracy, qps, latency p99 s | 0.7125, 60.8, 2.109 | 0.7105, 76.8, 0.464 |
| videomme talker c16 | accuracy, qps, latency p95 s | 11/20, 0.773, 21.7 | 11/20, 0.776, 23.1 |

`qwen3_omni_0518_numerics/08_ab_reservation.md:1-27`, the bf16 run and its
protocol line:

```
Bundle artifacts/ab-reservation.tar.gz. A = upstream/main 68c88dae6, B =
perf/scheduler-observed-reservation adc09ff5d ... bf16 colocated config, voice
clone (SeedTTS 50, warmup), one boot per arm and workload, GPU idle before
each boot, talker decode log at every step, request profiler events at c16
and c32.

| run | qps | latency p50 s | latency p95 s | RTF mean | RTF p95 | WER |
|---|---|---|---|---|---|---|
| A c1 | 2.237 | 0.422 | 0.640 | 0.141 | 0.200 | 1.60 |
| B c1 | 2.214 | 0.428 | 0.684 | 0.138 | 0.195 | 0.89 |
| A c16 | 6.737 | 2.297 | 3.400 | 0.695 | 1.033 | 1.42 |
| B c16 | 8.387 | 1.812 | 2.952 | 0.519 | 0.698 | 1.06 |
| A c32 | 7.595 | 3.712 | 5.192 | 1.160 | 2.024 | 1.24 |
| B c32 | 10.852 | 2.391 | 4.474 | 0.757 | 1.104 | 1.42 |
```

That doc also records the talker admission and event table at
08_ab_reservation.md:34-40, and at 08_ab_reservation.md:96-100 it names the code
path this report's Part A covers:

```
The log confirms the attribution without a stage name on the line: every
retraction sits between Decode batch lines at token usage 0.07 for about
1.5k tokens, which is the talker's 21373 token pool ..., and each is followed by a Prefill batch of
`new_tokens_gained + 1` tokens with 0 cached tokens, which is the replay of
the retracted request from `_decode_input_history`
(talker_model_runner.py:328-340).
```

**Step ledger protocol, for per-step numbers rather than A/B.**
`qwen3_omni_0518_numerics/15_step_ledger_runbook.md:33-56` states one fresh
server per model and per concurrency point, the benchmark's default warmup, and
the profile window:

```
One fresh server per model and per concurrency point, one profiled pass
with the benchmark's default warmup. Boot the server, run the point,
stop the server, then the next point. ... A fresh server per point is
what CI does, so the numbers match CI's cold and warm mix exactly (C
warm of 50 at concurrency C).

curl -s -X POST http://127.0.0.1:$PORT/start_request_profile \
  -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"${MODEL}_c${C}\",\"event_dir\":\"$OUT/${MODEL}_c${C}\"}"
<the benchmark at C, warmup left at its default>
curl -s -X POST http://127.0.0.1:$PORT/stop_request_profile \
  -H 'Content-Type: application/json' -d "{\"run_id\":\"${MODEL}_c${C}\"}"
```

**Earlier talker-step measurement.**
`qwen3_omni_0518_numerics/06_e0_talker_step.md:1-10` records the boot and
workload used for the talker step census: checkout 2f5d2ab5, sglang 0.5.18,
torch 2.13.0+cu130, H100 80GB, `examples/configs/qwen3_omni_colocated_h100_bf16.yaml`
on one GPU, voice clone workload `seed-tts-eval-50-arrow` en, three boots
(events at c16, events at c16 with the decode log at interval 1, and torch
profiler traces for 8 requests at c1 then 16 at c16 in one window). Its
noise-floor statement is at 06_e0_talker_step.md:31-34: "The two c16 event runs
differ by 9 percent in qps with the same code and the same 50 requests, so
single c16 runs of 50 requests carry at least that much noise."

I found no A/B protocol in those docs that targets the six decode-input helpers
or the Qwen3-TTS `_write_feedback_buffers` batching specifically. The recorded
Qwen3-Omni protocols target scheduler admission, predictor CUDA graph capture and
the talker step census.
