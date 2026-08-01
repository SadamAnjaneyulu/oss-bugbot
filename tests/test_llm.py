import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm import (  # noqa: E402
    LLMResponse,
    ProviderError,
    SafetyRefusal,
    call_gemini,
    call_groq,
    call_with_provider_fallback,
    parse_json_response,
)


def fake_gemini_client(text, finish_reason="STOP", prompt_tokens=100, response_tokens=50, candidates_present=True):
    client = MagicMock()
    if not candidates_present:
        response = SimpleNamespace(candidates=[], text=None, usage_metadata=None)
    else:
        candidate = SimpleNamespace(finish_reason=SimpleNamespace(value=finish_reason))
        # candidates_token_count is the field the real API actually populates
        # (response_token_count exists in the stub but is empirically unused).
        usage = SimpleNamespace(prompt_token_count=prompt_tokens, candidates_token_count=response_tokens)
        response = SimpleNamespace(candidates=[candidate], text=text, usage_metadata=usage)
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


def fake_groq_client(content, finish_reason="stop", prompt_tokens=80, completion_tokens=40):
    client = MagicMock()
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    completion = SimpleNamespace(choices=[choice], usage=usage)
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


class TestCallGemini(unittest.TestCase):
    def test_success(self):
        client = fake_gemini_client('{"pass_id": "p1", "findings": []}')
        sem = asyncio.Semaphore(1)
        result = asyncio.run(call_gemini(client, "gemini-3.5-flash-lite", "sys", "user", {}, sem))
        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.input_tokens, 100)
        self.assertEqual(result.output_tokens, 50)

    def test_safety_finish_reason_raises(self):
        client = fake_gemini_client("blocked", finish_reason="SAFETY")
        sem = asyncio.Semaphore(1)
        with self.assertRaises(SafetyRefusal):
            asyncio.run(call_gemini(client, "gemini-3.5-flash-lite", "sys", "user", {}, sem))

    def test_prohibited_content_raises(self):
        client = fake_gemini_client("blocked", finish_reason="PROHIBITED_CONTENT")
        sem = asyncio.Semaphore(1)
        with self.assertRaises(SafetyRefusal):
            asyncio.run(call_gemini(client, "gemini-3.5-flash-lite", "sys", "user", {}, sem))

    def test_empty_candidates_raises(self):
        client = fake_gemini_client(None, candidates_present=False)
        sem = asyncio.Semaphore(1)
        with self.assertRaises(SafetyRefusal):
            asyncio.run(call_gemini(client, "gemini-3.5-flash-lite", "sys", "user", {}, sem))


class TestCallGroq(unittest.TestCase):
    def test_success(self):
        client = fake_groq_client('{"cluster_id": "c1"}')
        sem = asyncio.Semaphore(1)
        result = asyncio.run(call_groq(client, "openai/gpt-oss-120b", "sys", "user", {}, "schema_name", sem))
        self.assertEqual(result.provider, "groq")
        self.assertEqual(result.input_tokens, 80)

    def test_content_filter_raises(self):
        client = fake_groq_client(None, finish_reason="content_filter")
        sem = asyncio.Semaphore(1)
        with self.assertRaises(SafetyRefusal):
            asyncio.run(call_groq(client, "openai/gpt-oss-120b", "sys", "user", {}, "schema_name", sem))

    def test_empty_content_raises(self):
        client = fake_groq_client("")
        sem = asyncio.Semaphore(1)
        with self.assertRaises(SafetyRefusal):
            asyncio.run(call_groq(client, "openai/gpt-oss-120b", "sys", "user", {}, "schema_name", sem))


class TestProviderFallback(unittest.TestCase):
    def test_primary_success_skips_fallback(self):
        fallback_called = {"n": 0}

        async def primary():
            return LLMResponse("ok", 1, 1, "m", "gemini")

        async def fallback():
            fallback_called["n"] += 1
            return LLMResponse("fallback", 1, 1, "m", "groq")

        result = asyncio.run(call_with_provider_fallback(primary, fallback))
        self.assertEqual(result.text, "ok")
        self.assertEqual(fallback_called["n"], 0)

    def test_refusal_triggers_fallback(self):
        async def primary():
            raise SafetyRefusal("blocked")

        async def fallback():
            return LLMResponse("fallback worked", 1, 1, "m", "groq")

        result = asyncio.run(call_with_provider_fallback(primary, fallback))
        self.assertEqual(result.text, "fallback worked")

    def test_non_refusal_error_does_not_trigger_fallback(self):
        fallback_called = {"n": 0}

        async def primary():
            raise ProviderError("network blew up")

        async def fallback():
            fallback_called["n"] += 1
            return LLMResponse("should not run", 1, 1, "m", "groq")

        with self.assertRaises(ProviderError):
            asyncio.run(call_with_provider_fallback(primary, fallback))
        self.assertEqual(fallback_called["n"], 0)


class TestParseJsonResponse(unittest.TestCase):
    def test_valid_json(self):
        self.assertEqual(parse_json_response('{"a": 1}'), {"a": 1})

    def test_invalid_json_raises_provider_error(self):
        with self.assertRaises(ProviderError):
            parse_json_response("not json at all")


class TestSemaphoreActuallyLimits(unittest.TestCase):
    def test_concurrency_capped_at_one(self):
        max_concurrent = {"n": 0, "peak": 0}

        async def slow_call(**kwargs):
            max_concurrent["n"] += 1
            max_concurrent["peak"] = max(max_concurrent["peak"], max_concurrent["n"])
            await asyncio.sleep(0.01)
            max_concurrent["n"] -= 1
            return SimpleNamespace(
                candidates=[SimpleNamespace(finish_reason=SimpleNamespace(value="STOP"))],
                text="{}",
                usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
            )

        client = MagicMock()
        client.aio.models.generate_content = slow_call
        sem = asyncio.Semaphore(1)

        async def run_both():
            await asyncio.gather(
                call_gemini(client, "m", "sys", "u1", {}, sem),
                call_gemini(client, "m", "sys", "u2", {}, sem),
            )

        asyncio.run(run_both())
        self.assertEqual(max_concurrent["peak"], 1)


if __name__ == "__main__":
    unittest.main()
