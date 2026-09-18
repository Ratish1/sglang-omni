# Recovered Qwen3-TTS scripts

Fetched 2026-09-18 from container sglang-omni-ratish on moss. Contents unmodified, md5 verified on both sides. Mtimes are container local time.

| container path | size | mtime | md5 | first docstring/comment line |
|---|---|---|---|---|
| /workspace/sglang-omni/.tmp/attribute.py | 4507 | 2026-09-16 19:50:12 | 92a267f6c8991dc040f53208c96db679 | """Kernel and host-block attribution for an sglang-omni torch trace. |
| /workspace/sglang-omni/.tmp/micro.py | 1826 | 2026-09-17 06:47:38 | 2a76f52e73d230292e17fa74a9221d4a | """Isolate the two restage patterns and count the CUDA runtime calls each issues. |
| /workspace/sglang-omni/.tmp/runtime.py | 1759 | 2026-09-17 07:00:43 | fe67e4b75ce811e887d112e48abc5670 | # aggregate CUDA runtime + memcpy events by name: count, total us, mean, p90 |
| /workspace/sglang-omni/.tmp/frames.py | 1312 | 2026-09-16 19:11:41 | 3cb4cd3f5ab5dbf5eb6224f322b0103c | (none) |
| /workspace/sglang-omni/.tmp/frames2.py | 2766 | 2026-09-16 19:13:24 | 66875b36c0865cfe8f8eae35f13a34ec | # self time = dur - sum(direct children dur) |
| /workspace/sglang-omni/.tmp/prof.py | 1175 | 2026-09-17 09:40:57 | 76c5743513bd9045888970a77dec5ec9 | (none) |
| /workspace/sglang-omni/.tmp/steady.py | 1127 | 2026-09-17 09:42:17 | cef32ce749395a8e67d5bfcc08908f5c | (none) |
| /workspace/sglang-omni/.tmp/seg16.py | 1980 | 2026-09-17 08:57:01 | 9a34bf3e8fa003fa4c4ee073024147ae | (none) |
| /workspace/sglang-omni/.tmp/analyze.py | 5322 | 2026-09-16 18:15:55 | b64507114a6b438c9abf5e476e21104c | # ---- drift bound and paired deltas ---- (mid-file, no header comment) |
| /workspace/sglang-omni/.tmp/pageable.py | 728 | 2026-09-17 11:22:47 | 3491d80f8da63994dc772ca72f321e40 | (none) |
| /workspace/sglang-omni/.tmp/an/pair.py | 3225 | 2026-09-17 16:59:14 | afd15c6a495893b9980b16ee8b16d318 | (none) |
| /workspace/sglang-omni/.tmp/an/long.py | 652 | 2026-09-17 17:00:32 | bf739498929fae80cde2233cfa5f4d08 | (none) |
| /workspace/sglang-omni/.tmp/census.sh | 3472 | 2026-09-16 17:55:51 | f7c97949242b769a8acfa6fb2fd105aa | (none) |
| /workspace/sglang-omni/.tmp/run_census.sh | 3419 | 2026-09-16 17:37:25 | cab4f267e6ce8900fdcd66914e5fa7f9 | # A/B census for #2126 (non-blocking copies) on moss, GPU 4, one card for every boot. |
| /workspace/sglang-omni/.tmp/c16.sh | 2592 | 2026-09-16 19:37:15 | 1ea37e5c6c09cba43f163dc4366a3ddd | (none) |
| /workspace/sglang-omni/.tmp/ab16.sh | 2888 | 2026-09-17 09:08:15 | 91c997e1f322b8575f2a246da45062d8 | (none) |
| /workspace/sglang-omni/.tmp/ab16b.sh | 2775 | 2026-09-17 09:13:41 | 4b840a0c6be6f9062e2307c6319b3183 | (none) |
| /workspace/sglang-omni/.tmp/bnorec.sh | 2387 | 2026-09-17 10:00:10 | 6ef145250c8a510d55f97bda3ef698ab | # B only. Identical to the ab16b B boots except the event recorder is never started. |
| /workspace/sglang-omni/.tmp/bfinal.sh | 2399 | 2026-09-17 10:22:55 | 62d84f6cbf7319251b665c03fef3610e | # B only. Identical to the ab16b B boots except the event recorder is never started. |
| /workspace/sglang-omni/.tmp/ns16.sh | 2565 | 2026-09-17 12:03:56 | d80ed2073cb42834770ac7d18d4bab9b | # Non-streaming c16, A = upstream/main, B = PR head. Interleaved, one pass per boot. |
| /workspace/sglang-omni/.tmp/boot.sh | 2167 | 2026-09-17 07:27:48 | 9e47901c68ff35412c2b2fa40e3448b8 | # boot_server <worktree> <outdir> <srv_gpu> <port> |
