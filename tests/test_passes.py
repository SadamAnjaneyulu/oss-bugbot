import asyncio
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import passes  # noqa: E402
from passes import MAX_TOOL_CALLS, _dispatch_tool, run_pass  # noqa: E402
from schemas import ReviewerOutput  # noqa: E402

VALID_FINDING_JSON = json.dumps({
    "pass_id": "p1",
    "findings": [{
        "file": "app.py", "line": 3, "category": "logic", "severity": "high",
        "title": "null deref", "reasoning": "db.get returns None",
        "evidence_lines": [3], "semgrep_corroborated": False, "self_confidence": 0.8,
    }],
})


def tool_call_response(name, args, finish_reason="tool_calls"):
    """OpenAI wire shape: tool_calls[].function.arguments is a JSON STRING
    (not a dict, unlike Gemini's native fc.args) - passes.py's loop
    json.loads's it before dispatch, so the fixture must produce a string
    here too or it wouldn't be testing the real shape. tc needs a real
    .model_dump() too - live-verified that passes.py round-trips the SDK's
    own serialization of each tool call (not a hand-picked subset) because
    Gemini's compat layer attaches a provider extension field that must be
    echoed back on the next turn or the request 400s.
    """
    tc = SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    tc.model_dump = lambda: {"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
    message = SimpleNamespace(content=None, tool_calls=[tc])
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


def final_response(text, finish_reason="stop"):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
    return SimpleNamespace(choices=[choice], usage=usage)


def make_client(responses):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=responses)
    return client


class TestDispatchTool(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "app.py").write_text("x = 1\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_dispatches_read_file(self):
        result = _dispatch_tool("read_file", {"path": "app.py"}, self.root)
        self.assertTrue(result["ok"])

    def test_unknown_tool_returns_structured_error(self):
        result = _dispatch_tool("delete_everything", {}, self.root)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "unknown_tool")

    def test_missing_required_arg_returns_structured_error_not_exception(self):
        result = _dispatch_tool("read_file", {}, self.root)  # no "path"
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "bad_arguments")


class TestRunPass(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "app.py").write_text("def get_user(uid):\n    return db.get(uid).name\n")
        self.changed_lines = {"app.py": {1, 2, 3}}
        self.deleted = set()
        self.sem = asyncio.Semaphore(1)

    def tearDown(self):
        self._tmp.cleanup()

    def test_direct_final_answer_no_tools(self):
        # Even a "no tool calls at all" pass costs 2 requests under the
        # split-phase design: the first turn (tools still offered) isn't
        # schema-enforced, so its content is provisional - a second,
        # tools-stripped + schema-enforced request gets the real answer.
        client = make_client([final_response(VALID_FINDING_JSON), final_response(VALID_FINDING_JSON)])
        result, usage = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsInstance(result, ReviewerOutput)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(client.chat.completions.create.await_count, 2)
        self.assertEqual(usage.input_tokens, 20)
        self.assertEqual(usage.output_tokens, 40)

    def test_one_tool_call_then_final_answer(self):
        client = make_client([
            tool_call_response("read_file", {"path": "app.py"}),
            final_response(VALID_FINDING_JSON),  # model stops calling tools, but not yet schema-enforced
            final_response(VALID_FINDING_JSON),  # re-ask, schema-enforced - the real final answer
        ])
        result, usage = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsInstance(result, ReviewerOutput)
        self.assertEqual(client.chat.completions.create.await_count, 3)

    def test_tool_cap_forces_final_answer(self):
        # MAX_TOOL_CALLS tool-call turns, then the loop must strip tools and
        # force a final answer on the next request - never loop forever.
        responses = [tool_call_response("list_dir", {"path": "."}) for _ in range(MAX_TOOL_CALLS)]
        responses.append(final_response(VALID_FINDING_JSON))
        client = make_client(responses)
        result, usage = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsInstance(result, ReviewerOutput)
        # MAX_TOOL_CALLS tool turns + 1 forced-final turn
        self.assertEqual(client.chat.completions.create.await_count, MAX_TOOL_CALLS + 1)
        self.assertEqual(usage.tool_calls_used, MAX_TOOL_CALLS)

    def test_never_combines_tools_and_response_format_in_one_request(self):
        # Real bug this guards against, caught by this exact test during
        # development: Groq 400s if `tools` and strict `response_format`
        # are combined in one request (live-verified). No single request
        # may carry both keys; the last request (the schema-enforced
        # final answer) must carry response_format and never tools.
        client = make_client([
            tool_call_response("read_file", {"path": "app.py"}),
            final_response(VALID_FINDING_JSON),  # model stops calling tools, not yet schema-enforced
            final_response(VALID_FINDING_JSON),  # re-ask, schema-enforced
        ])
        asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        for call in client.chat.completions.create.await_args_list:
            kwargs = call.kwargs
            self.assertFalse(
                "tools" in kwargs and "response_format" in kwargs,
                f"a single request combined tools and response_format: {kwargs}",
            )
        last_kwargs = client.chat.completions.create.await_args_list[-1].kwargs
        self.assertIn("response_format", last_kwargs)
        self.assertNotIn("tools", last_kwargs)

    def test_safety_refusal_returns_none_not_exception(self):
        client = make_client([final_response(None, finish_reason="content_filter")])
        result, usage = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsNone(result)

    def test_empty_candidates_returns_none(self):
        response = SimpleNamespace(choices=[], usage=None)
        client = make_client([response])
        result, usage = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsNone(result)

    def test_g1_failure_retries_and_recovers(self):
        # Every G1-retry attempt is itself a fresh _run_tool_loop call, and
        # under the split-phase design each attempt costs 2 requests
        # (unenforced direct answer, then the schema-enforced re-ask) even
        # with zero tool calls - so 2 G1 attempts here means 4 total requests.
        bad_json = json.dumps({
            "pass_id": "p1",
            "findings": [{
                "file": "app.py", "line": 999, "category": "logic", "severity": "high",  # line not in diff
                "title": "bad", "reasoning": "x", "evidence_lines": [999],
                "semgrep_corroborated": False, "self_confidence": 0.5,
            }],
        })
        client = make_client([
            final_response(bad_json), final_response(bad_json),          # attempt 1: fails G1
            final_response(VALID_FINDING_JSON), final_response(VALID_FINDING_JSON),  # attempt 2: succeeds
        ])
        result, usage = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsInstance(result, ReviewerOutput)
        self.assertEqual(client.chat.completions.create.await_count, 4)
        # Usage sums every request across both attempts, not just the one
        # that finally validated - every rejected attempt spent real tokens.
        self.assertEqual(usage.input_tokens, 40)
        self.assertEqual(usage.output_tokens, 80)

    def test_g1_failure_exhausted_returns_none(self):
        bad_json = json.dumps({
            "pass_id": "p1",
            "findings": [{
                "file": "app.py", "line": 999, "category": "logic", "severity": "high",
                "title": "bad", "reasoning": "x", "evidence_lines": [999],
                "semgrep_corroborated": False, "self_confidence": 0.5,
            }],
        })
        client = make_client([final_response(bad_json)] * 10)  # always fails G1
        result, usage = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
