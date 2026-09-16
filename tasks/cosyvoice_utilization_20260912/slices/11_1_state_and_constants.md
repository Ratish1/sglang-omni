# Slice 1.1: one owner for the hop contract

Plan row: `../plans/11_hop_prefix_cache.md` section 7, slice 1.1. Tree read:
upstream main `27a8293c`. Runtime change with no numeric change; gate G1 is byte
identity against main on a seeded c1 stream.

## What the read changed about the plan

The plan listed four items. One of them is already done on main and one costs
more than the plan assumed.

- **The state owner already exists.** `release_stream_resources` is overridden
  at `models/fun_cosyvoice3/streaming_vocoder.py:486` and already clears the
  tokens, the HiFT mel and the three prompt tensors. The base calls it from
  `clear_stream_state` (`scheduling/streaming_vocoder.py:259-263`), which is the
  single path for completion, abort, step failure and scheduler stop
  (`scheduling/streaming_vocoder.py:18-24`). Slice 1.2 adds one line to that
  override for the cache handle. Nothing to do here.
- **H3 cannot read the chunk size in every configuration.** With TensorRT,
  `attach_flow_estimator_trt` replaces `flow.decoder.estimator` with
  `FlowEstimatorTRTModule` and sets `flow.packed_estimator = None`
  (`stages.py:862-864`). The wrapper keeps the DiT as a private `_fallback`
  (`flow_estimator_trt.py:401`) and forwards no attributes, so
  `estimator.static_chunk_size` raises there. The validation needs the wrapper
  to carry the value.

## The changes

### H5, dead code (`models/fun_cosyvoice3/streaming.py`)

Three functions and one argument, all provably unused at runtime. Checked with
`git grep` over the whole tree at `27a8293c`:

| symbol | line | runtime callers | note |
|---|---|---|---|
| `stream_hop_len` | :39 | none. Its only caller is `tokens_needed_for_causal_chunk` :82, which itself has none | delete |
| `tokens_needed_for_causal_chunk` | :74 | none, tests only | delete |
| `prompt_token_len` | :19 | `model_runner.py:308, :495`, only to feed an argument that is discarded | delete once the argument goes |
| `first_ar_flush_tokens(prompt_len, ...)` | :87 | `model_runner.py:307, :494` | the body is `del prompt_len`; make the signature keyword only |

`first_ar_flush_tokens` becomes `first_ar_flush_tokens(*, hop_len=TOKEN_HOP_LEN)`
and the two call sites drop their first argument and the `prompt_token_len`
import (`model_runner.py:15-19`). The MLX vocoder has locals of the same name
(`mlx/vocoder/flow.py:100`, `loader.py:251`); those are unrelated and untouched.

### H4, constants from the flow (`models/fun_cosyvoice3/streaming_vocoder.py`)

`PRE_LOOKAHEAD_LEN` and `TOKEN_MEL_RATIO` are module constants in the scheduler
(:25-28) while `stages.py` already reads the same two from the checkpoint
(`flow.pre_lookahead_len` :546, :815; `flow.token_mel_ratio` :171, :196, :581,
:596, :727). They agree for this checkpoint and diverge by construction.

The scheduler holds the vocoder, so the four uses on scheduler methods read from
it directly: the warmup shapes (:139, :143) and `run_step` (:366, :380, :395).
The fifth is on the state, `next_decode` :64, which has no vocoder; the state
gains a `lookahead` field that `create_stream_state` :130 sets. That keeps all
nine `next_decode()` call sites (:257, :270, :282, :292, :307, :323, :330, :337,
:412) unchanged.

### H3, chunk alignment (`streaming_vocoder.py` and `flow_estimator_trt.py`)

The constructor checks only positive and ordered (:96-104). A hop that is not a
chunk multiple silently changes frames already emitted, and would silently
corrupt the 1.2 cache. Chunk tokens are `static_chunk_size / token_mel_ratio`,
50 / 2 = 25 for this checkpoint, so the shipped 25 and 100 pass and only a
misconfiguration is rejected.

`FlowEstimatorTRTModule` carries `static_chunk_size` from the DiT it wraps, so
`flow.decoder.estimator.static_chunk_size` reads the same value with or without
TensorRT, and the scheduler needs no branch.

## The changed path

```text
build the vocoder                                stages.py:872 load_cosyvoice3_flow_hift
  flow.decoder.estimator = DiT                   static_chunk_size 50
  optional: attach_flow_estimator_trt            stages.py:831-869
    wrapper.static_chunk_size = fallback's       NEW, so the read below is configuration free

construct the scheduler                          streaming_vocoder.py:83-120
  self.lookahead   = flow.pre_lookahead_len      NEW, was PRE_LOOKAHEAD_LEN :25
  self.mel_ratio   = flow.token_mel_ratio        NEW, was TOKEN_MEL_RATIO :28
  chunk_tokens     = estimator.static_chunk_size // mel_ratio      NEW
  require hop and max hop are multiples of it    NEW, was positive and ordered only :96-104

create_stream_state                              streaming_vocoder.py:129-130
  CosyVoice3StreamState(hop_len=..., lookahead=self.lookahead)     NEW field

state.next_decode()                              streaming_vocoder.py:63-72
  token_offset + hop_len + self.lookahead        was PRE_LOOKAHEAD_LEN :64

run_step, causal_window                          streaming_vocoder.py:376-416
  tokens[: offset + hop + self.lookahead]        was PRE_LOOKAHEAD_LEN :380
  mel[:, :, offset * self.mel_ratio :]           was TOKEN_MEL_RATIO :366, :395

release_stream_resources                         streaming_vocoder.py:486-493
  unchanged; 1.2 adds the cache handle here
```

## Tests that move with it

`tests/unit_test/fun_cosyvoice3/test_streaming.py`: the imports (:18-31) lose
three names, `test_stream_hop_math_matches_cosyvoice3` (:42-61) loses the four
assertions on the deleted functions and keeps the rest, and the two
`first_ar_flush_tokens` call sites (:57-60, :333) drop their argument. A new case
covers the chunk multiple rejection, which is the only behaviour this slice adds.

## Gate

G1: seeded c1 stream, same boot shape as main, emitted audio byte identical to
main. Nothing here changes a number, so byte identity is the whole gate, and it
is the one gate a 4090 settles as well as an H100 because it compares one build
against another on one card.
