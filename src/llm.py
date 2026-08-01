"""Provider abstraction: Gemini (google-genai) + Groq, async, structured JSON
output, safety-refusal handling, per-provider concurrency caps.

Model IDs and rate limits below are pinned and dated - see MODELS and
"Reproducibility" note. Verify both against the provider's live dashboard
before trusting old numbers; neither provider publishes a permanently fixed
free-tier table (Google's rate-limits page is now dynamic per AI Studio
account; Groq's model catalog changes with deprecations).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date

from google import genai
from google.genai import types as gtypes
from groq import AsyncGroq

# --- Pinned models -----------------------------------------------------
# Resolved 2026-08-01 against primary docs (ai.google.dev/gemini-api/docs/models,
# console.groq.com/docs/models). gemini-2.5-* was rejected: Google announced
# its shutdown for 2026-10-16, an 11-week runway from the resolution date -
# too short for a portfolio piece meant to outlive the build. Groq's
# llama-3.3-70b-versatile was rejected: deprecated 2026-06-17, already dead
# by resolution date.
MODEL_RESOLUTION_DATE = "2026-08-01"

GEMINI_FLASH_LITE = "gemini-3.5-flash-lite"  # A1 reviewer
GEMINI_FLASH = "gemini-3.5-flash"            # A2 aggregator, A3 fallback
GROQ_PRIMARY = "openai/gpt-oss-120b"         # A3 validator - genuinely different
                                              # family from Gemini (OpenAI-arch
                                              # open-weight, not Google), confirmed
                                              # free-tier via console.groq.com/docs/rate-limits

MODELS = {"a1": GEMINI_FLASH_LITE, "a2": GEMINI_FLASH, "a3_primary": GROQ_PRIMARY, "a3_fallback": GEMINI_FLASH}

# --- Concurrency caps ----------------------------------------------------
# Soft pre-emptive limits based on last-verified free-tier figures. Treated
# as a starting point, not gospel: Gemini's rate-limits page no longer
# publishes a static table (dynamic per account in AI Studio), so 429s are
# the authoritative signal - these semaphores exist to avoid hammering the
# API into 429s constantly, not to guarantee we never see one.
RATE_LIMITS = {
    "gemini_flash_lite_rpm": 15,
    "gemini_flash_rpm": 10,
    "groq_gpt_oss_120b_rpm": 30,
}

SAFETY_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class SafetyRefusal(Exception):
    """Terminal, not retryable - identical content refuses identically.
    Callers switch provider once; if the fallback also refuses, drop the
    pass and log a `safety_refusal` degradation. See plan: a PR that fixes
    an SQL injection *contains* an SQL injection - refusals are routine.
    """


class ProviderError(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str


@dataclass
class TokenUsage:
    """Shared accumulator for A2/A3 (and anything else that isn't A1's tool
    loop, which has its own PassUsage in passes.py for the extra
    tool_calls_used field). Retries burn real tokens too - callers must add
    every attempt's usage, not just the attempt that ultimately validated.
    """
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response: LLMResponse) -> None:
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens


async def call_gemini(
    client: genai.Client,
    model: str,
    system_prompt: str,
    user_content: str,
    response_schema: dict,
    semaphore: asyncio.Semaphore,
) -> LLMResponse:
    async with semaphore:
        response = await client.aio.models.generate_content(
            model=model,
            contents=user_content,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_json_schema=response_schema,
            ),
        )

    candidates = response.candidates or []
    if not candidates:
        raise SafetyRefusal(f"gemini:{model} returned no candidates (blocked)")

    finish_reason = getattr(candidates[0], "finish_reason", None)
    finish_value = getattr(finish_reason, "value", finish_reason)
    if finish_value in SAFETY_FINISH_REASONS:
        raise SafetyRefusal(f"gemini:{model} finish_reason={finish_value}")

    text = response.text
    if text is None:
        raise SafetyRefusal(f"gemini:{model} returned empty text (likely blocked)")

    usage = response.usage_metadata
    # candidates_token_count is what the live API actually populates.
    # response_token_count exists in the type stub's field list but was
    # empirically None on real responses - caught by a live call, not by
    # inspecting model_fields, which lists both without saying which one
    # the server actually fills in.
    output_tokens = getattr(usage, "candidates_token_count", None) or getattr(usage, "response_token_count", None) or 0
    return LLMResponse(
        text=text,
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=output_tokens,
        model=model,
        provider="gemini",
    )


async def call_groq(
    client: AsyncGroq,
    model: str,
    system_prompt: str,
    user_content: str,
    response_schema: dict,
    schema_name: str,
    semaphore: asyncio.Semaphore,
) -> LLMResponse:
    async with semaphore:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            },
        )

    choice = completion.choices[0]
    finish_reason = choice.finish_reason
    if finish_reason == "content_filter":
        raise SafetyRefusal(f"groq:{model} finish_reason=content_filter")

    text = choice.message.content
    if not text:
        raise SafetyRefusal(f"groq:{model} returned empty content (likely refused)")

    usage = completion.usage
    return LLMResponse(
        text=text,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        model=model,
        provider="groq",
    )


async def call_with_provider_fallback(primary_call, fallback_call) -> LLMResponse:
    """Runs primary_call(); on SafetyRefusal, switches provider once via
    fallback_call(). Any other exception propagates immediately - refusals
    are the only case where "try someone else" makes sense, since a
    malformed-JSON or network error isn't provider-specific.
    """
    try:
        return await primary_call()
    except SafetyRefusal:
        return await fallback_call()


def parse_json_response(raw_text: str) -> dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON response despite structured-output mode: {exc}") from exc
