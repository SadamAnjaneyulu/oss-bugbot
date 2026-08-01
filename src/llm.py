"""Provider abstraction: any OpenAI-compatible endpoint (base_url + api_key +
model), async, structured JSON output, safety-refusal handling.

Was Gemini-native (google-genai) + Groq-native (groq SDK) until the pipeline
went provider-agnostic - call_groq's shape was already the OpenAI wire
format (Groq's SDK is a compatible fork of openai's), so this module is a
generalization of that existing, already-tested shape, not new logic.
Gemini is reached through its own documented OpenAI-compat layer
(generativelanguage.googleapis.com/v1beta/openai/), live-verified to
support both tool-calling and strict json_schema mode.

Model IDs below are pinned and dated - see MODEL_RESOLUTION_DATE. Verify
against the provider's live dashboard before trusting old numbers; neither
Gemini nor Groq publishes a permanently fixed free-tier table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import AsyncOpenAI

# --- Pinned models -----------------------------------------------------
# Resolved 2026-08-01 against primary docs (ai.google.dev/gemini-api/docs/models,
# console.groq.com/docs/models). gemini-2.5-* was rejected: Google announced
# its shutdown for 2026-10-16, an 11-week runway from the resolution date -
# too short for a portfolio piece meant to outlive the build. Groq's
# llama-3.3-70b-versatile was rejected: deprecated 2026-06-17, already dead
# by resolution date.
MODEL_RESOLUTION_DATE = "2026-08-01"

GEMINI_FLASH_LITE = "gemini-3.5-flash-lite"  # default A1 reviewer
GEMINI_FLASH = "gemini-3.5-flash"            # default A2 aggregator, A3 fallback
GROQ_PRIMARY = "openai/gpt-oss-120b"         # default A3 validator - genuinely different
                                              # family from Gemini (OpenAI-arch
                                              # open-weight, not Google), confirmed
                                              # free-tier via console.groq.com/docs/rate-limits

# OpenAI-compat base URLs for the two providers this project defaults to -
# used by main.py's default_provider_configs_from_env() so review.yml keeps
# working unchanged. Any other OpenAI-compatible provider (OpenAI, Together,
# Fireworks, DeepSeek, OpenRouter, local Ollama, ...) just needs its own
# base_url - nothing here is Gemini/Groq-specific beyond these two defaults.
GEMINI_OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_OPENAI_COMPAT_BASE_URL = "https://api.groq.com/openai/v1"

MODELS = {"a1": GEMINI_FLASH_LITE, "a2": GEMINI_FLASH, "a3_primary": GROQ_PRIMARY, "a3_fallback": GEMINI_FLASH}


@dataclass(frozen=True)
class ProviderConfig:
    """base_url + api_key + model - the whole "any OpenAI-compatible
    provider" contract. Four independent instances thread through the
    pipeline (a1, a2, a3_primary, a3_fallback) rather than one shared
    config, since A3 being a genuinely different provider/model family from
    A1/A2 is a deliberate adversarial-independence design choice, not an
    accident of how Gemini+Groq happened to get picked originally.
    """
    base_url: str
    api_key: str
    model: str


def make_client(config: ProviderConfig) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)


# --- Concurrency caps ----------------------------------------------------
# Soft pre-emptive limits for the two DEFAULT providers (default_provider_
# configs_from_env in main.py) - not gospel, 429s are the authoritative
# signal. A visitor-supplied arbitrary provider has no known rate limit up
# front, so main.py picks a conservative generic default for those instead
# of looking it up here.
RATE_LIMITS = {
    "gemini_flash_lite_rpm": 15,
    "gemini_flash_rpm": 10,
    "groq_gpt_oss_120b_rpm": 30,
}


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


async def call_llm(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    response_schema: dict,
    schema_name: str,
    semaphore,
) -> LLMResponse:
    """Generic call for any OpenAI-compatible provider - a generalization of
    call_groq's body (already the OpenAI wire shape) to an arbitrary
    base_url'd client instead of a Groq-specific one. No tool-calling here;
    passes.py's A1 loop has its own request-building (tools and strict
    response_format can't always be combined in one call - see passes.py).
    """
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
    if choice.finish_reason == "content_filter":
        raise SafetyRefusal(f"{model}: finish_reason=content_filter")

    text = choice.message.content
    if not text:
        raise SafetyRefusal(f"{model}: returned empty content (likely refused)")

    usage = completion.usage
    return LLMResponse(
        text=text,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        model=model,
        provider="openai-compatible",
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
