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


def tool_call_response(name, args, finish_reason="STOP"):
    part = SimpleNamespace(function_call=SimpleNamespace(name=name, args=args), text=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason=SimpleNamespace(value=finish_reason))
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
    return SimpleNamespace(candidates=[candidate], text=None, usage_metadata=usage)


def final_response(text, finish_reason="STOP"):
    part = SimpleNamespace(function_call=None, text=text)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason=SimpleNamespace(value=finish_reason))
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=20)
    return SimpleNamespace(candidates=[candidate], text=text, usage_metadata=usage)


def make_client(responses):
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=responses)
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
        client = make_client([final_response(VALID_FINDING_JSON)])
        result = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsInstance(result, ReviewerOutput)
        self.assertEqual(len(result.findings), 1)

    def test_one_tool_call_then_final_answer(self):
        client = make_client([
            tool_call_response("read_file", {"path": "app.py"}),
            final_response(VALID_FINDING_JSON),
        ])
        result = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsInstance(result, ReviewerOutput)
        self.assertEqual(client.aio.models.generate_content.await_count, 2)

    def test_tool_cap_forces_final_answer(self):
        # MAX_TOOL_CALLS tool-call turns, then the loop must strip tools and
        # force a final answer on the next request - never loop forever.
        responses = [tool_call_response("list_dir", {"path": "."}) for _ in range(MAX_TOOL_CALLS)]
        responses.append(final_response(VALID_FINDING_JSON))
        client = make_client(responses)
        result = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsInstance(result, ReviewerOutput)
        # MAX_TOOL_CALLS tool turns + 1 forced-final turn
        self.assertEqual(client.aio.models.generate_content.await_count, MAX_TOOL_CALLS + 1)

    def test_safety_refusal_returns_none_not_exception(self):
        client = make_client([final_response(None, finish_reason="SAFETY")])
        result = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsNone(result)

    def test_empty_candidates_returns_none(self):
        response = SimpleNamespace(candidates=[], text=None, usage_metadata=None)
        client = make_client([response])
        result = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsNone(result)

    def test_g1_failure_retries_and_recovers(self):
        bad_json = json.dumps({
            "pass_id": "p1",
            "findings": [{
                "file": "app.py", "line": 999, "category": "logic", "severity": "high",  # line not in diff
                "title": "bad", "reasoning": "x", "evidence_lines": [999],
                "semgrep_corroborated": False, "self_confidence": 0.5,
            }],
        })
        client = make_client([final_response(bad_json), final_response(VALID_FINDING_JSON)])
        result = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsInstance(result, ReviewerOutput)
        self.assertEqual(client.aio.models.generate_content.await_count, 2)

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
        result = asyncio.run(run_pass(client, "m", self.sem, self.root, "diff", "p1", self.changed_lines, self.deleted))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
