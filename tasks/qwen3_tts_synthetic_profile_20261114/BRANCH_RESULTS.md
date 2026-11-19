# Branch results: codec slot wait, #4077 (11-16, host-221, two H100s)

Each head against main `ea8216220` at the same time on the two cards, orders swapped. RTF is fix against main.

| Head | sglang_omni tree | Overload (60 / 70 / 80 req/s succeeded, fix) | Normal load | Read |
|---|---|---|---|---|
| `d2507351c` FIFO waiting queue | `d1460e887` | not run | streaming c16 RTF +1.5 %, c64 +2.3 %, whole c16 +1.5 % | the queue took `state_lock` on every stream end and release while empty |
| `e528b9d9a` no lock while the queue is empty | `7d3df299e` | 100 / 96.6-98.7 / 85-86.9 %, three seeds | killed before it finished (head moved) | |
| `f9bc7891b` planner owns the queue, failure exit, free-list count | `4bc1fa14d` | 97.3-100 / 97.9-99.5 / 83.2-93.0 %, three seeds | inside the A/A band in every cell on warm rounds; first audio equal or better | opened, ready for review |
| `038a071bd` initial worker owns admission (review 1) | `bc5372bbf` | 100 / 100 / 92.6-92.8 %, two seeds | c16 RTF +0.1 %, c64 +0.3 % on warm passes | second review: F1 last row retired strands waiters, F2 fatal drain gap |
| `dade67035` fallback once every slot retires, fatal drain under the lock (review 2) | `780d0d283` | 80 req/s against `038a071bd`: 87.4 → 87.5 %, underrun requests 8 → 5 | failure paths only; normal path takes no new lock | pushed to #4077 (11-18); new tests fail on `038a071bd`; reviewer probe completes |
| main `2a66cb083` with `codec_state_slots 128` (CLI), against `dade67035` | `110bc32b4` | 100 / 100 / 90.4-90.5 % against 100 / 99.4-100 / 88.1-88.4 %, seeds 0 and 1; underruns 23 against 28; first byte p50 at 70 req/s 62-86 against 174-216 ms | | the waiting queue is not needed: the arena was undersized |
| `33bde3fa9` arena sized to admitted requests (`fix/qwen3-tts-codec-arena-holds-admitted-streams`) | `ce22bfb43` | against main, seed 0: 1.1 → 100 %, 85.6 → 100 %, 4.0 → 91.5 %; underruns 2,284 / 13 / 2,229 → 4 / 5 / 0; 0 fallbacks | SeedTTS Base both orders: c1 and c16 inside the band, c64 first audio mean -0.3 ms, p95 -17 ms, RTF +1.0 % (two readings), which warm rounds do not repeat (six readings: RTF -0.4 %, req/s +0.6 %, first audio mean +0.2 ms, p95 -16 ms); WER 0.95 / 1.67 → 1.04 / 1.09 % (in the pulled files); SIM and main's fallback count were not pulled before the lease ended | Fable review (11-19, `tasks/qwen3_tts_stack_review_20261119/REVIEW.md`) approved replacing #4077 with two fixes first |
| `95a524770` arena from the coordinator's own cap (`resolve_coordinator_max_in_flight`), factory requires the slots and the cap, refuses an uncapped server without slots | `0aa406b93` | 11-19, host-221 card 3, each arm alone, seeds 1 / 2 against main: 1.0 / 2.1 → 100 / 100 %, 88.0 / 89.9 → 100 / 100 %, 76.5 / 77.5 → 95.4 / 95.1 %; underrun requests 2,269 / 2,381 → 1 / 0, 23 / 15 → 4 / 4, 31 / 8 → 0 / 2; main's left-context decodes 1,183 / 1,397, 343 / 358, 351 / 345, branch 0; branch peak held rows (2 s samples) 61 / 57, 65 / 65, 66 / 66 | 54 seeded requests (27 whole, 27 streamed; Base, CustomVoice, VoiceDesign) byte-identical to main; KV pool 551,255 tokens, 58.88 GiB (demand 524,288) | at `51bd5606c` CPU 2,940 passed and 7 failed (4 that main shares on a one-card box, 3 replica rows without replica_devices, fixed in `8020b64a6`); accelerator 266 passed; null queue exits naming the cap, with slots set it serves and warns; 128 clients dropping after the first chunk and streaming again, two rounds: 0 exhausted, 37 / 40 HTTP 503 at the cap |
| `758506aef` the cap functions move to `config/topology.py` (Fable delta review approved the push) | `5d2386ad9` | same behaviour as `0aa406b93` (a move) | | CPU 3,583 passed, 5 failures that main shares here (4 one-card topology tests, Drama3 CPU decode without triton); the new test on main: 8 of 8 fail with 64 against the cap; pushed to #4077 as `ed1700380` (fast-forward), title and body replaced (11-19) |

# Branch results: coordinator waiting (11-18, host-221 GPUs [4 5])

Against the slot fix at the same time on the two cards. Nari's client, 180 s measured.

| Head | sglang_omni tree | 100 / 80 req/s: completed, underrun requests, first byte p50 | Normal load | Read |
|---|---|---|---|---|
| `42c1bd93b` unbounded, polled disconnect | `08fd7ff2c` | 100 / 100 %, 556 / 110 against 3 / 0 | not run | API loop saturated by the 50 ms disconnect poll |
| `636861be0` event-driven disconnect watch | `6de866337` | 100 / 100 %, 128-145 / 40 against 7-21 / 1 | not run | full collections 270-470 ms with the queue: per-waiter objects |
| same, `--admission.max_waiting_requests 0` | `6de866337` | 70.4 %, 2 against 4 (100 only) | | the residual is the waiting itself |
| same, bound 512 / 256 (CLI) | `6de866337` | 512: 70.1 / 89.9 %, 3 / 4 against 4 / 12; 256: 69.7 / 88.2 %, 7 / 2 against 4 / 5 | | 256 chosen for stall margin |
| `ddd52d694` default 256 on `dade67035` | `f6d2c8198` | seeds 0 / 1: 69.7 / 69.8 % and 85.8 / 87.3 %, underruns 4, 3 / 3, 7 against 2, 4 / 7, 6; first byte 4.6 s against 0.9 s | SeedTTS Base both orders inside the A/A band, WER 1.03 / 1.07 %, SIM 71.39 / 71.36; whole c64 RTF -0.4 % | ready to open on the user's word; bound trade-off for the user |
| `3e07f76e4` rebuilt on the sizing branch `33bde3fa9` (`fix/qwen3-tts-admission-waits-on-arena`), then `a941cd93a` on `95a524770`, then `23fa9ea59` on `758506aef` | `7dab90093`, `75e445466`, `88ac8e2c6` | one card, against the sizing head: running | | 11-19 one card: CPU 2,745 passed (4 failures main shares), accelerator 286 passed; 384 streams at once two rounds and the mixed round times 8 three rounds both models complete, 0 refused; `a941cd93a`: 128 clients dropping after the first chunk and streaming again, two rounds, all complete, 0 refused, 0 exhausted |

# Branch results: persistent predictor layer (11-14)

Branch `dev/qwen3-tts-predictor-persistent-20261114`, measured on host-62 (H100) with `persistent_layer_check.py` and `persistent_stamps.py`. One layer is layer 0 back to back (L2-warm). Five layers is a chain over layers 0 to 4 in one CUDA graph. Triton is `FusedPredictorLayers`, measured in the same run.

| Head | sglang_omni tree | 1 row: one layer / 5 layers / Triton 5 (us) | 8 rows | 16 rows | Numerics vs fp32 | Read |
|---|---|---|---|---|---|---|
| `c70fc4c61` stage 1 | `9219d448` | 28.7 / 181.8 / 139.1 | 30.8 / 187.0 / 145.6 | 40.8 / 229.1 / 146.6 | max abs equal to Triton's at 1, 8 and 16 rows. Bit-equal to Triton at 1 row | Latency chain. Staging and attention loads ran one after another, and the signal had two `MEMBAR.ALL.GPU` |
| `0bf8e35be` loads at once, one fence | `e2bf5f45` | 31.6 / 163.3 / 138.9 | 34.5 / 176.2 / 145.3 | 39.2 / 198.4 / 146.6 | same as stage 1 | 584 B stack, 168 registers. Unset `packed` arrays were carried across loop iterations |
| `a6900d3de` zeroed staged vectors | `54784df1` | 18.8 / 107.5 / 139.1 | 20.0 / 111.9 / 145.2 | 22.9 / 125.9 / 146.5 | same as stage 1 | No stack, 125 registers. Faster than Triton at every row count. Misses 12 us a layer |
| `1881c0af2` norms and rope in phase loads, raw residual | `ea038190` | not run: the lease ended | | | | |

## H200 (node-radixark-16-0001), 11-14 and 11-15

Same check, with Triton timed in the same run as a contention reference (129 to 131 us at 1 row when the card is ours alone). Runs whose Triton reference was 154 to 166 us overlapped another session's jobs and are not listed. From `c9e61be0e` on, "5 layers" is one launch over the stacked layers; "5 launches" runs the same kernel once a layer.

| Head | 1 row: one layer / 5 layers / 5 launches | 8 rows | 16 rows | Read |
|---|---|---|---|---|
| `1881c0af2` | 17.79 / 99.1 | 19.38 / 107.3 | 22.33 / 120.8 | baseline on H200 |
| `c9eede580` finer stamps | 17.80 / 99.1 | 19.41 / 107.4 | 22.34 / 120.9 | repeat run: same within 0.1 us |
| `3f0cd624f` per-phase balance, schedule in shared memory | 16.70 / 97.9 | 18.13 / 105.0 | 20.93 / 117.5 | gate_up max 3.79 to 2.57 us |
| `0f02a0203` attention value loop unrolled | 17.15 / 99.3 | 18.30 / 105.9 | 21.24 / 118.7 | attention compute 1.60 to 1.85 us: regression |
| `1e8c34fca` phase tiles batched | 17.31 / 104.1 | 18.52 / 110.2 | 21.29 / 121.4 | gate_up items faster, every fixed cost slower: regression |
| `42093246a` back to `3f0cd624f` plus attention stamps | 16.77 / 97.9 | 18.16 / 105.4 | 20.95 / 117.7 | |
| `c9e61be0e` 5 layers one launch, next-layer L2 prefetch | 16.66 / 116.1 / 94.6 | 17.98 / 123.7 / 105.7 | 21.12 / 136.0 / 118.9 | MMA-phase signals 1.4 to 3.0 us under the prefetch traffic |
| `2a28ee342` no prefetch | - / 98.2 / 95.5 | - / 107.7 / 105.7 | - / 119.6 / 118.9 | signals still 1.1 to 1.4 us beside ring refills; 0.61 us where none |
| `9b6ef40cc` flagged-word handoff, no fences | - / 84.9 / 79.2 | - / 104.4 / 100.3 | - / 138.8 / 131.7 | numerics unchanged; flags double staged bytes |
| `83b9060d8` 256 ns poll back-off | - / 84.4 / 84.4 | - / 102.5 / 99.7 | - / 135.4 / 130.9 | no gain: polling pressure is not the 16-row cost |
| `94c7595fa` flagged up to 8 rows, counted above | 16.6 / 86.6 / 82.6 | - / 106.4 / 102.9 | - / 120.6 / 119.4 | each mode where it measured best; numerics unchanged |

Codebook step (5 layers, final norm, lm_head, seeded sampler), `predictor_bench.py`, us:

| | 1 row | 16 rows |
|---|---|---|
| Triton | 147.0 | 155.2 |
| CuTe DSL | 128.3 | 132.0 |
| `2a28ee342` one launch / per-layer launches | 112.4 / 109.6 | 132.1 / 132.5 |
| `9b6ef40cc` one launch / per-layer launches | 104.9 / 101.5 | 154.5 / 148.5 |
| `94c7595fa` one launch / per-layer launches | 105.3 / 107.7 | 134.1 / 133.7 |

Error of the step against fp32 at 1 row: persistent 0.162, Triton and CuTe 0.193.

## H100 (host-62, cards 4 and 5), 11-14

`94c7595fa`, check and step bench, each run alone on its card:

| | 1 row | 8 rows | 16 rows |
|---|---|---|---|
| 5 layers one launch / 5 launches / Triton | 100.3 / 90.7 / 138.9 | 115.5 / 108.5 / 144.7 | 129.7 / 125.3 / 146.4 |
| Step: Triton / CuTe DSL / persistent one launch / per-layer | 172.7 / 148.8 / 126.2 / 128.4 | | 178.4 / 151.3 / 156.5 / 154.3 |

- The step bench gave the same numbers alone and beside the check, so the two cards did not disturb each other.
- The 11-13 study measured Triton's step at 157.4 us and CuTe's at 137.4 us on this host's other cards. Today's cards run every backend about 10 % slower, so compare backends within one run only.

Primitive latencies on this H200 (`latency_probe.py`, cycles): chained MMA 24, independent MMA 7, chained ldmatrix 34, L2 round trip 288, store and gpu fence 415, 256-thread barrier 27, clock64 and globaltimer reads 24 each.

## Stamp timeline, 1 row, L2-warm (us from the first CTA entry)

| Point | stage 1 | `a6900d3de` |
|---|---|---|
| Kernel span (stamped) | 28.9 | 21.0 |
| Operand constants before the PDL wait | 0.03 | 0.73 |
| qkv staging | 2.84 | 1.81 |
| qkv tiles (2 a CTA) | 1.35 | 1.61 |
| Signal (fence and add) | 0.83 to 1.15 | 0.52 to 0.76 |
| Handoff, last signal to first release | at most 0.26 (timer step) | at most 0.38 |
| Attention item | 6.63 | 2.23 |
| o_proj, gate_up, down staging | 1.17, 1.42, 1.74 | 0.44, 0.69, 0.59 |
| o_proj, gate_up, down tiles | 1.26, 2.66, 1.48 | 1.33, 2.84, 1.73 |

- The global timer steps by 256 ns. Stamps add about 2 us to a 19 us launch.
- In `a6900d3de` the o_proj and down tiles convert the residual right after its load, at a branch join, so the MMAs wait one L2 round trip. `1881c0af2` keeps the residual raw until the epilogue.
- `1881c0af2` also stops loading the norm weights and rope row up front. That load cost 0.73 us, queued behind each SM's 96 KB weight burst.
