## [Fix][Qwen3-TTS] Hold streams for a codec state slot instead of the left-context decoder

### Problem

The streaming vocoder keeps each stream's incremental codec state in an arena of 64 slots, the engine's running bound.

A stream takes its slot at its first decode and returns it at its final decode, after its engine slot is already free, so under load more than 64 streams are live in the vocoder.

Each stream past the 64th took the left-context decoder, which costs more per chunk than the incremental one, so the vocoder fell further behind and more streams stayed live.

Under an open-loop load just past the knee the server collapsed: on Pipeshift's bar (Nari Labs' client, 1.7B CustomVoice) 60 req/s served 4.5 % of requests, every one with playback underruns.

### Change

With async decoding, a stream that finds the arena full now waits for a slot and keeps the incremental decoder.

- `build_incremental_plan` marks the stream `is_waiting_for_codec_slot` instead of setting the fallback, and a new stream queues behind any stream already waiting.
- `plan_stream_decode` keeps the waiting streams oldest first; every slot returned to the arena schedules the first decode of the oldest waiting streams, one per free slot.
- A waiting stream is not rescheduled by its new chunks, and one whose generation ends keeps the async finish instead of a synchronous whole decode on the message thread.
- The synchronous decoder keeps the left-context fallback; the codec stats gain `slot_waits`.

### Results

TBD: same-time SeedTTS EN pairs (c1, c16, c64 streaming; c1, c16 whole requests) with WER and SIM, and the bar at 60, 70 and 80 req/s over three seeds.
