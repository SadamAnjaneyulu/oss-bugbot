import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from starlette.testclient import TestClient  # noqa: E402

import app as backend_app  # noqa: E402


class TestHealth(unittest.TestCase):
    def test_health_lists_stages(self):
        client = TestClient(backend_app.app)
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")
        self.assertIn("size_gate", resp.json()["stages"])


class TestWsReview(unittest.TestCase):
    def test_bad_url_sends_error_and_closes(self):
        client = TestClient(backend_app.app)
        with client.websocket_connect("/ws/review") as ws:
            ws.send_json({"url": "not-a-url", "github_token": "t"})
            msg = ws.receive_json()
        self.assertEqual(msg["type"], "error")
        self.assertEqual(msg["stage"], "resolving")

    def test_happy_path_streams_stages_then_result_no_keys_leaked(self):
        async def fake_resolve(*a, **kw):
            return {"head_sha": "abcdef1234", "head_clone_url": "x", "head_ref": "main"}  # pragma: allowlist secret

        def fake_clone(*a, **kw):
            return None

        canned_result = {
            "schema_version": "1.0", "pr": 2, "findings": [],
            "post_result": {"posted": False, "reason": "dry_run", "count": 0, "fallback": False},
            "skipped_files": [], "degradations": [],
            "token_usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "calls": {"a1": 4, "a2": 1, "a3": 0}},
        }

        async def fake_run_review(*a, **kw):
            on_progress = kw.get("on_progress")
            for stage in ("size_gate", "diff_fetched", "semgrep_done", "a1_pass_done", "a2_done", "a3_done", "posted"):
                if on_progress:
                    on_progress(stage, "ok")
            return canned_result

        secret_token = "ghp_supersecrettoken123"  # pragma: allowlist secret
        secret_key = "sk-secret-abc"  # pragma: allowlist secret

        with patch("app.cli.resolve_pr_info", side_effect=fake_resolve), \
             patch("app.cli.clone_pr_branch", side_effect=fake_clone), \
             patch("app.run_review", side_effect=fake_run_review):
            client = TestClient(backend_app.app)
            with client.websocket_connect("/ws/review") as ws:
                ws.send_json({
                    "url": "https://github.com/o/r/pull/2",
                    "github_token": secret_token,
                    "a1": {"base_url": "https://example.com/v1", "api_key": secret_key, "model": "m1"},
                    "a2": {"base_url": "https://example.com/v1", "api_key": secret_key, "model": "m2"},
                    "a3_primary": {"base_url": "https://example.com/v1", "api_key": secret_key, "model": "m3"},
                    "a3_fallback": {"base_url": "https://example.com/v1", "api_key": secret_key, "model": "m4"},
                    "post": False,
                })
                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg["type"] in ("result", "error"):
                        break

        types_and_stages = [(m["type"], m.get("stage")) for m in messages]
        self.assertIn(("stage", "resolving"), types_and_stages)
        self.assertIn(("stage", "cloning"), types_and_stages)
        self.assertEqual(sum(1 for m in messages if m.get("stage") == "a1_pass_done"), 1)
        self.assertEqual(messages[-1]["type"], "result")
        self.assertEqual(messages[-1]["data"]["pr"], 2)

        # No secret value anywhere in what was actually sent to the client.
        serialized = str(messages)
        self.assertNotIn(secret_token, serialized)
        self.assertNotIn(secret_key, serialized)


if __name__ == "__main__":
    unittest.main()
