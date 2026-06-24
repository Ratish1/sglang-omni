from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from loguru import logger

from sglang_omni.models.dots_tts.native.utils.tokenizer import (
    AUDIO_GEN_SPAN_TOKEN,
    AUDIO_GEN_START_TOKEN,
    TEXT_COND_END_TOKEN,
    require_token_id,
)

TEMPLATE_PATTERN = re.compile(r"\{text\}|\{audio\}|\{interleave\}|[^\{]+")


@dataclass(frozen=True)
class ParsedTemplate:
    parts: tuple[str, ...]
    has_audio_placeholder: bool
    has_interleave_placeholder: bool


@dataclass(frozen=True)
class TokenizedTemplatePart:
    kind: str
    token_ids: tuple[int, ...] = ()
    raw_text: str | None = None


def parse_template(template: str) -> ParsedTemplate:
    parts = tuple(re.findall(TEMPLATE_PATTERN, template))
    has_audio_placeholder = "{audio}" in parts
    interleave_count = parts.count("{interleave}")
    if has_audio_placeholder and interleave_count:
        raise ValueError("Template cannot mix audio and interleave placeholders.")
    if interleave_count > 1:
        raise ValueError(
            "Interleave generation template must contain exactly one interleave placeholder."
        )
    return ParsedTemplate(
        parts=parts,
        has_audio_placeholder=has_audio_placeholder,
        has_interleave_placeholder=interleave_count == 1,
    )


def _prepare_template_tokens(
    *, text: str, tokenizer, template: str
) -> tuple[ParsedTemplate, list[int]]:
    return parse_template(template), tokenizer.encode(text, add_special_tokens=False)


def _iter_tokenized_template_parts(
    *,
    parsed_template: ParsedTemplate,
    tokenizer,
    text_tokens: list[int],
):
    for part in parsed_template.parts:
        if part == "{text}":
            yield TokenizedTemplatePart(kind="text", token_ids=tuple(text_tokens))
            continue
        if part == "{audio}":
            yield TokenizedTemplatePart(kind="audio")
            continue
        if part == "{interleave}":
            yield TokenizedTemplatePart(kind="interleave")
            continue
        yield TokenizedTemplatePart(
            kind="literal",
            token_ids=tuple(tokenizer.encode(part, add_special_tokens=False)),
            raw_text=part,
        )


def build_generation_schedule(
    *,
    text: str,
    tokenizer,
    template: str,
    max_audio_tokens: int,
) -> dict[str, Any]:
    if max_audio_tokens <= 0:
        raise ValueError("max_audio_tokens must be positive for generation.")

    parsed_template, text_tokens = _prepare_template_tokens(
        text=text,
        tokenizer=tokenizer,
        template=template,
    )
    schedule_ids: list[int] = []
    audio_gen_start_id = require_token_id(tokenizer, AUDIO_GEN_START_TOKEN)
    audio_gen_span_id = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)

    if parsed_template.has_audio_placeholder:
        for part in _iter_tokenized_template_parts(
            parsed_template=parsed_template,
            tokenizer=tokenizer,
            text_tokens=text_tokens,
        ):
            if part.kind == "audio":
                schedule_ids.append(audio_gen_start_id)
                schedule_ids.extend([audio_gen_span_id] * max_audio_tokens)
                continue
            schedule_ids.extend(part.token_ids)
        visible_schedule_ids = [
            token_id for token_id in schedule_ids if token_id != audio_gen_span_id
        ]
        decoded_schedule = (
            tokenizer.decode(
                visible_schedule_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if hasattr(tokenizer, "decode")
            else repr(visible_schedule_ids)
        )
        logger.info(
            "Built generation schedule: interleave={} max_audio_tokens={} sequence={!r}",
            False,
            int(max_audio_tokens),
            decoded_schedule,
        )
        return {
            "schedule_ids": schedule_ids,
            "interleave": False,
        }

    if not parsed_template.has_interleave_placeholder:
        raise ValueError(
            "Generation template must contain either {audio} or {interleave}."
        )
    text_cond_end_id = require_token_id(tokenizer, TEXT_COND_END_TOKEN)
    if max_audio_tokens < len(text_tokens):
        raise ValueError(
            "Interleave generation requires at least one audio span per text token: "
            f"text_token_count={len(text_tokens)} "
            f"max_audio_patch_count={max_audio_tokens}."
        )

    interleave_started = False
    for part in _iter_tokenized_template_parts(
        parsed_template=parsed_template,
        tokenizer=tokenizer,
        text_tokens=text_tokens,
    ):
        if part.kind == "interleave":
            _append_interleave_schedule_tokens(
                schedule_ids=schedule_ids,
                text_tokens=text_tokens,
                max_audio_tokens=max_audio_tokens,
                audio_span_id=audio_gen_span_id,
                text_cond_end_id=text_cond_end_id,
            )
            interleave_started = True
            continue
        if part.kind == "text":
            raise ValueError(
                "Generation schedule does not support {text} inside an interleave template."
            )
        if part.kind == "audio":
            raise ValueError(
                "Generation schedule does not support {audio} inside an interleave template."
            )
        if interleave_started:
            if (part.raw_text or "").strip():
                raise ValueError(
                    "Generation schedule does not support non-empty suffix text after the interleave placeholder."
                )
            continue
        schedule_ids.extend(part.token_ids)

    visible_schedule_ids = [
        token_id for token_id in schedule_ids if token_id != audio_gen_span_id
    ]
    decoded_schedule = (
        tokenizer.decode(
            visible_schedule_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if hasattr(tokenizer, "decode")
        else repr(visible_schedule_ids)
    )
    logger.info(
        "Built generation schedule: interleave={} max_audio_tokens={} sequence={!r}",
        True,
        int(max_audio_tokens),
        decoded_schedule,
    )
    return {
        "schedule_ids": schedule_ids,
        "interleave": True,
    }


def _append_interleave_schedule_tokens(
    *,
    schedule_ids: list[int],
    text_tokens: list[int],
    max_audio_tokens: int,
    audio_span_id: int,
    text_cond_end_id: int,
) -> None:
    for token_id in text_tokens:
        schedule_ids.append(token_id)
        schedule_ids.append(audio_span_id)
    schedule_ids.append(text_cond_end_id)
    remaining_audio_tokens = max_audio_tokens - len(text_tokens)
    if remaining_audio_tokens > 0:
        schedule_ids.extend([audio_span_id] * remaining_audio_tokens)
