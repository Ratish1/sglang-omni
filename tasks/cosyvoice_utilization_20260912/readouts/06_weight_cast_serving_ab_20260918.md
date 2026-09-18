# Readout 06: the DiT weight cast in serving, and what the c16 benchmark can resolve (2026-09-18)

Moss box, RTX 4090 D, card 7 alone, strictly one process at a time. Both arms:
`--tts_engine.engine.mem_fraction_static 0.3` (the test rig setting for this 24 GB card: 9.3 GB free
after a pool of about 209,000 tokens, above the 131,072 the engine can address), no seed, no token
limit from us, warmup 1, streaming, the whole English split. main `b5c3b44aa`, cast `4a57f9a81`
(main plus one commit). Raw run directories, without the whole split wavs, are in
`artifacts/cosyvoice-4090-20260918/runs/` on the Mac; probe outputs in `.../probes/`.

## Serving

| c | arm | req/s | audio s/s | RTF mean / p99 | first audio mean / p95 s | inter chunk mean s | C50 / C100 / C200 | runaways | audio total s |
|---|---|---|---|---|---|---|---|---|---|
| 8 | main | 2.758 | 13.31 | 0.643 / 1.676 | 1.495 / 2.196 | 0.705 | 74.7 / 76.1 / 83.2 | 2 | 5,251 |
| 8 | cast | 2.768 | 13.76 | 0.618 / 1.432 | 1.429 / 2.402 | 0.700 | 73.3 / 77.9 / 85.3 | 4 | 5,410 |
| 8 | main | 2.611 | 12.94 | 0.669 / 1.838 | 1.501 / 2.780 | 0.771 | 70.5 / 72.4 / 79.2 | 4 | 5,393 |
| 8 | cast | 2.785 | 13.71 | 0.637 / 1.745 | 1.438 / 2.455 | 0.711 | 72.1 / 78.1 / 87.2 | 4 | 5,355 |
| 16 | main | 3.301 | 15.65 | 1.085 / 2.225 | 2.251 / 3.453 | 1.321 | 22.6 / 27.3 / 41.2 | 1 | 5,158 |
| 16 | cast | 2.805 | 14.24 | 1.227 / 3.021 | 2.595 / 4.978 | 1.449 | 14.9 / 27.2 / 33.7 | 6 | 5,521 |
| 16 | main | 2.939 | 14.62 | 1.200 / 2.828 | 2.525 / 4.139 | 1.432 | 10.7 / 21.3 / 39.5 | 3 | 5,411 |
| 16 | cast | 2.771 | 14.09 | 1.293 / 3.161 | 2.560 / 4.544 | 1.537 | 18.1 / 22.5 / 29.2 | 5 | 5,533 |

Zero failed requests, zero CUBLAS and zero out of memory lines in all eight. A runaway is a request
whose audio is 60 s or longer (2,048 tokens, 81.9 s).

c8: the cast is ahead on every read in both pairs. c16: the cast reads below both main boots, so by
the pass line set before the run it is **not cleared**.

## Why the c16 rows do not compare

The arms did not get the same work: 6 and 5 runaways on the cast, 1 and 3 on main, 7 percent more
audio. The cast touches only the DiT, which runs after the AR that decides lengths, and nothing is
seeded, so the count is sampling. Main against itself prices a runaway: 1 runaway 3.301 req/s,
3 runaways 2.939, about 6 percent each, a 12 percent swing between boots of one tree. A 2 percent
effect cannot be read through that.

Time between vocoder steps, from the timestamps of the step log lines in `serve.log`, by batch size:

| step | main, two boots | cast, two boots |
|---|---|---|
| hop, 16 rows, median | 1,165 / 1,252 ms | 1,175 / 1,229 ms |
| final, 16 rows, median | 1,753 / 1,709 ms | 1,680 / 1,744 ms |
| hop, 1 row, median | 753 / 675 ms | 663 / 609 ms |

Equal at 16 rows, faster at 1 row, which is what the isolated probe measured (`plans/14`, section 7:
bit identical mel, 3,220 fewer launches, minus 2 percent at 16 rows to minus 10 percent at 1 row).

## Noted for later: the benchmark disables the model's own length bound

The engine derives `max_new_tokens` from the text length when the caller sends none
(`request_builders.py:610-619`, the CosyVoice contract of 20 tokens per text token, capped at
2,048). The benchmark always sends one: every run's config carries `'max_new_tokens': 2048`. So an
82 s runaway for a ten word sentence, which a caller that sends no limit cannot get, is something
the benchmark manufactures, and it is what dominates the c16 numbers above (0.1 to 0.5 percent of
requests, about 6 percent of throughput each). To analyse later: whether the benchmark should leave
the field unset for this model, and what the runaway rate is under the model's own bound.

## Also seen, not chased

A 16 row step is about 1.2 s and runs HiFT once per request inside it, sixteen sequential calls
(roadmap rank 4). A 1 row hop step is 600 to 750 ms in serving against 196 ms for the Flow call
alone; part of that is waiting for AR tokens, not separated here.
