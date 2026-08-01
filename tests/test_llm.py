import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm import (  # noqa: E402
    LLMResponse,
    ProviderConfig,
    ProviderError,
    SafetyRefusal,
    call_llm,
    call_with_provider_fallback,
    make_client,
    parse_json_response,
)


def fake_openai_client(content, finish_reason="stop", prompt_tokens=80, completion_tokens=40):
    client = MagicMock()
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    completion = SimpleNamespace(choices=[choice], usage=usage)
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


class TestMakeClient(unittest.TestCase):
    def test_builds_client_with_base_url_and_key(self):
        config = ProviderConfig(base_url="https://example.com/v1", api_key="sk-test", model="m")
        client = make_client(config)
        self.assertEqual(str(client.base_url), "https://example.com/v1/")
        self.assertEqual(client.api_key, "sk-test")


class TestCallLlm(unittest.TestCase):
    def test_success(self):
        client = fake_openai_client('{"cluster_id": "c1"}')
        sem = asyncio.Semaphore(1)
        result = asyncio.run(call_llm(client, "openai/gpt-oss-120b", "sys", "user", {}, "schema_name", sem))
        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.provider, "openai-compatible")
        self.assertEqual(result.input_tokens, 80)
        self.assertEqual(result.output_tokens, 40)

    def test_content_filter_raises(self):
        client = fake_openai_client(None, finish_reason="content_filter")
        sem = asyncio.Semaphore(1)
        with self.assertRaises(SafetyRefusal):
            asyncio.run(call_llm(client, "openai/gpt-oss-120b", "sys", "user", {}, "schema_name", sem))

    def test_empty_content_raises(self):
        client = fake_openai_client("")
        sem = asyncio.Semaphore(1)
        with self.assertRaises(SafetyRefusal):
            asyncio.run(call_llm(client, "openai/gpt-oss-120b", "sys", "user", {}, "schema_name", sem))


class TestProviderFallback(unittest.TestCase):
    def test_primary_success_skips_fallback(self):
        fallback_called = {"n": 0}

        async def primary():
            return LLMResponse("ok", 1, 1, "m", "openai-compatible")

        async def fallback():
            fallback_called["n"] += 1
            return LLMResponse("fallback", 1, 1, "m", "openai-compatible")

        result = asyncio.run(call_with_provider_fallback(primary, fallback))
        self.assertEqual(result.text, "ok")
        self.assertEqual(fallback_called["n"], 0)

    def test_refusal_triggers_fallback(self):
        async def primary():
            raise SafetyRefusal("blocked")

        async def fallback():
            return LLMResponse("fallback worked", 1, 1, "m", "openai-compatible")

        result = asyncio.run(call_with_provider_fallback(primary, fallback))
        self.assertEqual(result.text, "fallback worked")

    def test_non_refusal_error_does_not_trigger_fallback(self):
        fallback_called = {"n": 0}

        async def primary():
            raise ProviderError("network blew up")

        async def fallback():
            fallback_called["n"] += 1
            return LLMResponse("should not run", 1, 1, "m", "openai-compatible")

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
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        client = MagicMock()
        client.chat.completions.create = slow_call
        sem = asyncio.Semaphore(1)

        async def run_both():
            await asyncio.gather(
                call_llm(client, "m", "sys", "u1", {}, "schema_name", sem),
                call_llm(client, "m", "sys", "u2", {}, "schema_name", sem),
            )

        asyncio.run(run_both())
        self.assertEqual(max_concurrent["peak"], 1)


if __name__ == "__main__":
    unittest.main()
