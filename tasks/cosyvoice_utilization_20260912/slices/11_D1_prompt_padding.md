# Slice D1: the padded prompt

Plan row: `../plans/11_hop_prefix_cache.md` section 4 H1 and section 7. Decision
P4 is open and this slice does not start before it is taken. Trees read: upstream
main `27a8293c` and slice 1.1.

## What the code does today

The streaming path pads the prompt up to a 25 token multiple by repeating the
last prompt token and the last prompt mel frame:

```text
latch_prompts                              streaming_vocoder.py:219-250
  pad_flow_prompt_to_hop(token, feat, hop_len=self.token_hop_len)   :234
    prompt_token_pad: next hop multiple minus the length   streaming.py:19-26
    token_fill = prompt_token[:, -1:].repeat(1, pad)       streaming.py:87
    feat_fill  = prompt_feat[:, -1:, :].repeat(1, pad * 2, 1)       :92
first_ar_flush_tokens(hop_len=25) = 28     streaming.py:47-57
  so the first hop always waits hop + lookahead generated tokens, whatever the
  prompt length                            model_runner.py:303-309, :492-496
```

Up to 24 fabricated tokens and 48 fabricated mel frames therefore condition every
streaming request. The buffered and fallback paths do not pad: they build their
Flow input straight from the request (`stages.py:1578-1606`), so the same
reference produces different conditioning depending on whether the caller asked
for a stream.

The reference implementation does neither. It keeps the prompt real and lets the
first chunk wait until enough generated tokens exist (`model.py:345-350`).

## What D1 changes

The prompt stays as the caller sent it. The first flush waits for the tokens the
first chunk actually needs, so `first_ar_flush_tokens` becomes prompt aware
again, this time with a body:

```text
first_ar_flush_tokens(prompt_len, *, hop_len=25)
  pad = prompt_token_pad(prompt_len, hop_len=hop_len)
  return hop_len + PRE_LOOKAHEAD_LEN + pad
```

and `latch_prompts` stops calling `pad_flow_prompt_to_hop`. The two call sites in
`model_runner.py` pass the prompt length again, which is the argument slice 1.1
deleted, so D1 restores it rather than inventing one: 1.1 removed it because
nothing used it, and D1 is the change that gives it a use.

Chunk alignment still has to hold, because slice 1.1 requires hops to be whole
attention chunks and slice 1.2's cache requires the chunk boundaries of emitted
frames never to move. Waiting `hop + pad` generated tokens is exactly what keeps
`prompt + first hop` on a chunk boundary without fabricating anything, so the
invariant is preserved by waiting rather than by padding.

## What it costs and what it buys

Cost, derived: at most 24 extra AR decode steps before the first hop, about 60 ms
at the stage 0 measured 2.53 ms per decode step. It is paid once per request and
only by requests whose prompt is not already a hop multiple.

Buy, not derived, which is why this slice has its own A/B: the conditioning stops
carrying up to 0.55 s of repeated audio frames that the speaker never produced,
and streaming and buffered stop conditioning differently for the same reference.
Whether that shows up in WER or SIM is the question the run answers. It is the
only slice in this plan that changes output on purpose.

## Gate

Its own A/B, not G2: c1 and c16, full English corpus, one boot per arm against
the head D1 sits on.

- WER and SIM reported as paired deltas, with the sign that matters stated first:
  D1 is expected to help or do nothing, and a regression beyond the G2 bound of
  0.3 absolute WER or 0.005 SIM ends the slice.
- First audio mean and p95, where the cost lands. The derived 60 ms bound is the
  number to check the measurement against.
- Streaming against buffered on the same reference, which is the asymmetry this
  slice removes: with D1 the two paths build the same Flow prompt, so their
  outputs should stop diverging for reasons that have nothing to do with
  streaming.

## Ordering

The plan sequences D1 after 1.2 and P4 may choose otherwise. What the read says
about ordering: D1 and 1.2 touch disjoint code, D1 changes output and 1.2 must
not, so running D1 first makes 1.2's G2 arms harder to compare against the
existing ledger. After 1.2 is the cheaper order.
