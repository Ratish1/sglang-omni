"""Box only: every talker request's inputs and outputs, step by step, saved when it finishes, for talker_parity.py.

Each stage process imports sitecustomize at start while this directory is on PYTHONPATH. With TALKER_TRACE_DIR set,
the talker's prefill builder and model runner are wrapped once their modules load; each call then runs unchanged:
- TalkerPrefillBuilder.build_prompt_prefill: the thinker tokens the prefill holds and whether the thinker had finished;
- TalkerPrefillBuilder.append_text_chunk, mark_thinker_done: each later thinker token, the thinker's end;
- QwenTalkerModelRunner.combine_feedback_with_next_text: each decode step's feedback row and text row, apart;
- QwenTalkerModelRunner.before_prefill, before_decode: the KV slots each prefill writes; at each decode step the rope
  positions, sequence length and KV slot the forward uses, the batch size and row, and the feedback buffer row and
  mask the model reads;
- QwenTalkerModelRunner.sample_next_token_ids: the prefill's logits before SGLang's sampler rewrites them in place;
- QwenTalkerModelRunner.post_prefill, post_decode: the codec logits a step samples from, before the repetition
  penalty, the talker hidden state, the sampled code and the step's 16 codes; at prefill also the request's
  projected prompt rows, its talker input ids and positions;
- QwenTalkerModelRunner.on_request_finished: everything for the request to <TALKER_TRACE_DIR>/<request id>.pt.
"""

import importlib.abc
import importlib.machinery
import os
import sys

PREFILL_MODULE = "sglang_omni.models.qwen3_omni.components.talker_prefill"
RUNNER_MODULE = "sglang_omni.models.qwen3_omni.talker_model_runner"
TRACES = {}


def trace(request_id):
    return TRACES.setdefault(
        request_id,
        {"prefill_tokens": None, "thinker_done_at_prefill": None, "appended_tokens": [], "thinker_done": False,
         "feedback_rows": [], "text_rows": [], "text_is_pad": [], "logits": [], "sampled": [], "codes": []},
    )


def install_prefill(module):
    builder = module.TalkerPrefillBuilder
    build_prompt_prefill = builder.build_prompt_prefill
    append_text_chunk = builder.append_text_chunk
    mark_thinker_done = builder.mark_thinker_done

    def traced_build_prompt_prefill(self, payload, thinker_chunks, *, thinker_done):
        record = trace(payload.request_id)
        record["prefill_tokens"] = self.extract_chunk_token_ids(thinker_chunks).tolist()
        record["thinker_done_at_prefill"] = bool(thinker_done)
        return build_prompt_prefill(self, payload, thinker_chunks, thinker_done=thinker_done)

    def traced_append_text_chunk(self, req_data, chunk):
        if not req_data.thinker_chunks_done:
            token_id = (chunk.metadata or {}).get("token_id")
            trace(req_data.req.rid)["appended_tokens"].append(None if token_id is None else int(token_id))
        else:
            pass
        return append_text_chunk(self, req_data, chunk)

    def traced_mark_thinker_done(self, req_data):
        trace(req_data.req.rid)["thinker_done"] = True
        return mark_thinker_done(self, req_data)

    builder.build_prompt_prefill = traced_build_prompt_prefill
    builder.append_text_chunk = traced_append_text_chunk
    builder.mark_thinker_done = traced_mark_thinker_done


def install_runner(module):
    import torch

    runner = module.QwenTalkerModelRunner
    combine = runner.combine_feedback_with_next_text
    before_prefill = runner.before_prefill
    before_decode = runner.before_decode
    sample_next_token_ids = runner.sample_next_token_ids
    post_prefill = runner.post_prefill
    post_decode = runner.post_decode
    on_request_finished = runner.on_request_finished
    # note (ratish): SGLang's sampler turns the prefill logits into probabilities in place, so they are kept first.
    prefill_logits = {}
    forward_index = {"value": 0}

    def traced_before_prefill(self, forward_batch, schedule_batch, requests):
        output = before_prefill(self, forward_batch, schedule_batch, requests)
        forward_index["value"] += 1
        slots = forward_batch.out_cache_loc.detach().to("cpu")
        start = 0
        for sched_req, extend_len in zip(requests, forward_batch.extend_seq_lens_cpu):
            record = trace(sched_req.data.req.rid)
            record["prefill_slots"] = slots[start:start + extend_len]
            record["prefill_forward"] = forward_index["value"]
            start += extend_len
        return output

    def traced_before_decode(self, forward_batch, schedule_batch, requests, *, is_lookahead=False):
        output = before_decode(self, forward_batch, schedule_batch, requests, is_lookahead=is_lookahead)
        forward_index["value"] += 1
        batch_size = len(requests)
        positions = (forward_batch.mrope_positions if forward_batch.mrope_positions is not None
                     else forward_batch.positions.view(1, -1))[:, :batch_size].detach().to("cpu")
        seq_lens = forward_batch.seq_lens[:batch_size].detach().to("cpu")
        slots = forward_batch.out_cache_loc[:batch_size].detach().to("cpu")
        buffer_rows = self.model.feedback_buffer[:batch_size].detach().to("cpu", torch.float32)
        mask = self.model.feedback_mask[:batch_size].detach().to("cpu")
        for row, sched_req in enumerate(requests):
            record = trace(sched_req.data.req.rid)
            record.setdefault("decode_steps", []).append(dict(
                positions=positions[:, row].tolist(), seq_len=int(seq_lens[row]), slot=int(slots[row]),
                batch_size=batch_size, row=row, mask=bool(mask[row]), buffer_row=buffer_rows[row],
                forward=forward_index["value"]))
        return output

    def traced_sample_next_token_ids(self, logits_output, forward_batch, schedule_batch, requests):
        prefill_logits["value"] = logits_output.next_token_logits.detach().to("cpu", torch.float32)
        return sample_next_token_ids(self, logits_output, forward_batch, schedule_batch, requests)

    def traced_combine(*, data, device, dtype):
        inputs = runner.peek_next_decode_inputs(data)
        if inputs is not None:
            feedback, next_text = inputs
            record = trace(data.req.rid)
            record["feedback_rows"].append(feedback.detach().to("cpu", torch.float32))
            record["text_rows"].append(next_text.detach().to("cpu", torch.float32))
            record["text_is_pad"].append(not data.pending_text_queue)
        else:
            pass
        return combine(data=data, device=device, dtype=dtype)

    def record_step(self, result, requests, logits):
        batch_size = len(requests)
        sampled = result.next_token_ids[:batch_size].detach().to("cpu").view(-1)
        codes = self.model.output_codes[:batch_size].detach().to("cpu")
        # a decode graph replay may return no hidden states; the talker's code predictor read them in the forward
        hidden_states = result.logits_output.hidden_states
        hidden = None if hidden_states is None else hidden_states[:batch_size].detach().to("cpu", torch.float32)
        for row, sched_req in enumerate(requests):
            record = trace(sched_req.data.req.rid)
            record["logits"].append(logits[row])
            record["sampled"].append(int(sampled[row]))
            record["codes"].append(codes[row])
            record.setdefault("hidden", []).append(None if hidden is None else hidden[row].reshape(-1))

    def traced_post_prefill(self, result, forward_batch, schedule_batch, requests):
        output = post_prefill(self, result, forward_batch, schedule_batch, requests)
        if result.next_token_ids is not None:
            for sched_req in requests:
                data = sched_req.data
                record = trace(data.req.rid)
                record["prompt_rows"] = data.prefill_input_embeds.detach().to("cpu", torch.float32)
                record["input_ids"] = [int(token_id) for token_id in data.req.origin_input_ids]
                mm_inputs = data.req.multimodal_inputs
                if mm_inputs is not None and mm_inputs.mrope_positions is not None:
                    record["mrope_positions"] = mm_inputs.mrope_positions.detach().to("cpu")
                    record["mrope_delta"] = int(torch.as_tensor(mm_inputs.mrope_position_delta).view(-1)[0])
                else:
                    record["mrope_positions"] = None
                    record["mrope_delta"] = 0
            record_step(self, result, requests, prefill_logits.pop("value"))
        else:
            pass
        return output

    def traced_post_decode(self, result, forward_batch, schedule_batch, requests):
        output = post_decode(self, result, forward_batch, schedule_batch, requests)
        if self.feedback_enabled:
            logits = result.logits_output.next_token_logits[: len(requests)].detach().to("cpu", torch.float32)
            record_step(self, result, requests, logits)
        else:
            pass
        return output

    def traced_on_request_finished(self, request_id, req_data):
        record = TRACES.pop(request_id, None)
        if record is not None:
            record["request_id"] = request_id
            record["finish_reason"] = req_data.finish_reason
            torch.save(record, f"{os.environ['TALKER_TRACE_DIR']}/{request_id}.pt")
        else:
            pass
        return on_request_finished(self, request_id, req_data)

    runner.combine_feedback_with_next_text = staticmethod(traced_combine)
    runner.sample_next_token_ids = traced_sample_next_token_ids
    runner.before_prefill = traced_before_prefill
    runner.before_decode = traced_before_decode
    runner.post_prefill = traced_post_prefill
    runner.post_decode = traced_post_decode
    runner.on_request_finished = traced_on_request_finished


INSTALLERS = {PREFILL_MODULE: install_prefill, RUNNER_MODULE: install_runner}


class TraceAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        install = INSTALLERS.get(name)
        if install is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def exec_and_install(module):
            exec_module(module)
            install(module)

        spec.loader.exec_module = exec_and_install
        return spec


if os.environ.get("TALKER_TRACE_DIR"):
    os.makedirs(os.environ["TALKER_TRACE_DIR"], exist_ok=True)
    sys.meta_path.insert(0, TraceAfterLoad())
else:
    pass
