# 12. The talker KV pool is sized for 48 layers, the talker has 20 (2026-09-02)

## 1. The measurement

Every boot log gives the talker pool's bytes per token as K size divided
by tokens:

| boot | K size | tokens | K bytes per token | K plus V per token |
|---|---|---|---|---|
| bf16 colocated H100 (doc 08 run) | 0.46 GB | 20136 | 24.5 KB | 49 KB |
| fp8 colocated H100 (doc 09 section 6 run) | 2.76 GB | 120769 | 24.5 KB | 49 KB |

The talker's shape from the checkpoint's config.json
(talker_config.text_config) is 20 layers, 2 KV heads, head_dim 128. In
bf16 that is 20 x 2 x 128 x 2 bytes = 10 KB for K per token, 20 KB for K
plus V. The pool holds 24.5 KB of K per token, which is 96 head layers of
256 bytes: 48 layers x 2 heads. The thinker's pool is exact for its
shape (48 layers x 4 heads x 128 x 2 bytes = 49 KB for K, measured 4.99 GB
over 109029 tokens = 49.1 KB).

So the talker's KV pool is allocated for 48 layers and uses 20. 2.4 times
the memory per token, 58 percent of the talker's KV budget never touched.

## 2. The mechanism

sglang builds the talker engine's ModelConfig from the whole Qwen3-Omni
checkpoint. get_hf_text_config (sglang utils/hf_transformers/common.py:400)
prefers thinker_config.text_config for a config that has a thinker_config,
so the engine's text config is the thinker's, and ModelConfig sets
num_hidden_layers = 48 and num_attention_layers = num_hidden_layers = 48
at construction (configs/model_config.py:1039-1040).

Omni then repoints the engine at the talker: ModelWorker._apply_arch_override
(sglang_omni/model_runner/model_worker.py:126-155) maps Qwen3OmniTalker to
talker_config.text_config and rewrites hf_text_config, num_attention_heads,
num_key_value_heads (2), hidden_size and num_hidden_layers (20). It does
not rewrite num_attention_layers, which stays 48. The Whisper branch of
the same function does rewrite it (:137).

sglang sizes the pool from resolve_layer_indices
(model_runner_components/layer_setup.py:117-150): the layer count is
max(model_config.num_hidden_layers, model_config.num_attention_layers)
unless the model object carries start_layer and end_layer, and the top
level Qwen3OmniTalker does not (only its inner text model does). So the
pool gets max(20, 48) = 48 layers, with the overridden head count of 2.
The same max is in the 0.5.17 pin (layer_setup.py:172) and the 0.5.16 pin (:167-168).

The code predictor is not involved: its five layers keep their own K and
V tensors (components/talker.py:904-913, 1706-1707), outside the pool.

## 3. What it costs, per profile

| profile | talker pool today | pool with 20 layers | rows at which sglang's guess binds, today and fixed |
|---|---|---|---|
| bf16 colocated H100 | 20136 tokens, 0.92 GB | about 48k tokens | 6, then about 14 |
| fp8 colocated H100 | 120769 tokens, 5.52 GB | about 290k tokens, or 3.2 GB back to the thinker | 38, then about 95 |

On bf16 H100 the talker still starves under sglang's guess at c16 and
c32 after the fix (14 rows), so the reservation change keeps its value
there. On fp8 H100 the fix alone frees 3.2 GB that the thinker could take
(about 35k thinker tokens at 98 KB per token), which is the profile that
runs out of thinker pool on video prompts.

## 4. The fix and the proof

One line at the omni seam, no upstream patch: in _apply_arch_override,
after num_hidden_layers, set num_attention_layers to the sub model's
num_hidden_layers for the mapped architectures (the Whisper branch already
sets its own). Every consumer of num_attention_layers in sglang (the pool
layer count, the attention backend's per layer buffers) then sees the
talker's 20 layers, which are the layer ids the model registers (0 to 19,
components/talker.py:453-464).

Other mapped architectures to check for the same gap, by comparing the
pool line against the sub model's shape at boot: Qwen3TTSTalker
(talker_config against the root config sglang picks), the llm_config and
language_config models are the ones sglang's own priority list already
selects, so their num_attention_layers should already match. Task: read
the pool line for Qwen3-TTS, MOSS-TTS and Dots.

Proof: the boot line on the two Qwen3-Omni profiles (tokens x 2.4), then
the voice clone bench at c32 on bf16 with WER and UTMOS, and a video
request on fp8 to show the thinker unaffected. The codes cannot change:
the pool holds the same bytes per token for the 20 layers the model
writes, only the unused 28 layers of cells disappear.

The 0.5.16 pin sizes the pool with the same max (its
layer_setup.py:167-168), so the gap is as old as the override itself
(model_worker.py, from the V0 retirement of 2026-05-14), not a pin bump.
