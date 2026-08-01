import asyncio
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import harvest  # noqa: E402

REAL_REVERSED_DIFF = """\
diff --git a/src/requests/sessions.py b/src/requests/sessions.py
index 74029c8..87c87a2 100644
--- a/src/requests/sessions.py
+++ b/src/requests/sessions.py
@@ -567,7 +567,7 @@ class Session(SessionRedirectMixin):
         timeout: _t.TimeoutType = None,
         allow_redirects: bool = True,
         proxies: dict[str, str] | None = None,
-        hooks: _t.HooksType = None,
+        hooks: _t.HooksInputType | None = None,
         stream: bool | None = None,
         verify: _t.VerifyType | None = None,
         cert: _t.CertType = None,
"""


class TestClassifyCategory(unittest.TestCase):
    def test_security_keyword_in_title(self):
        self.assertEqual(harvest.classify_category("Fix SQL injection", "", ""), "security")

    def test_concurrency_keyword_in_body(self):
        self.assertEqual(harvest.classify_category("Fix bug", "fixes a race condition", ""), "concurrency")

    def test_resource_keyword_in_diff(self):
        self.assertEqual(harvest.classify_category("Fix bug", "", "forgot to call close() on the file handle"), "resource")

    def test_defaults_to_logic(self):
        self.assertEqual(harvest.classify_category("Fix off-by-one", "", ""), "logic")

    def test_case_insensitive(self):
        self.assertEqual(harvest.classify_category("Fix XSS vulnerability", "", ""), "security")


class TestExtractGroundTruth(unittest.TestCase):
    def test_real_reversed_diff_produces_ground_truth(self):
        gt = harvest.extract_ground_truth(REAL_REVERSED_DIFF)
        self.assertIn("src/requests/sessions.py", gt)
        self.assertTrue(len(gt["src/requests/sessions.py"]) > 0)

    def test_test_files_excluded(self):
        diff_text = REAL_REVERSED_DIFF.replace("src/requests/sessions.py", "tests/test_sessions.py")
        gt = harvest.extract_ground_truth(diff_text)
        self.assertEqual(gt, {})

    def test_empty_diff_yields_no_ground_truth(self):
        self.assertEqual(harvest.extract_ground_truth(""), {})


class TestTestFileRegex(unittest.TestCase):
    def test_matches_common_test_patterns(self):
        for path in ["tests/test_foo.py", "test_foo.py", "src/test_bar.py",
                     "app.test.tsx", "app.spec.ts", "foo_test.py"]:
            with self.subTest(path=path):
                self.assertTrue(harvest.TEST_FILE_RE.search(path), path)

    def test_does_not_match_normal_source(self):
        for path in ["src/app.py", "lib/contest.py", "attestation.py"]:
            with self.subTest(path=path):
                # "contest.py"/"attestation.py" contain "test" as a substring
                # but not as a path/filename token - must not false-positive.
                self.assertFalse(harvest.TEST_FILE_RE.search(path), path)


class TestFetchPrShas(unittest.TestCase):
    def _client(self, json_body, status=200):
        def handler(request):
            return httpx.Response(status, json=json_body)
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def test_oversize_by_files_rejected(self):
        candidate = harvest.Candidate("o", "r", 1, "Fix bug")
        body = {"changed_files": 10, "additions": 5, "deletions": 5,
                "base": {"repo": {"full_name": "o/r"}}, "head": {"repo": {"full_name": "o/r"}}}
        client = self._client(body)
        result = asyncio.run(harvest.fetch_pr_shas(client, candidate, "tok"))
        self.assertIsNone(result)

    def test_oversize_by_lines_rejected(self):
        candidate = harvest.Candidate("o", "r", 1, "Fix bug")
        body = {"changed_files": 1, "additions": 40, "deletions": 40,
                "base": {"repo": {"full_name": "o/r"}}, "head": {"repo": {"full_name": "o/r"}}}
        client = self._client(body)
        result = asyncio.run(harvest.fetch_pr_shas(client, candidate, "tok"))
        self.assertIsNone(result)

    def test_deleted_fork_head_repo_null_does_not_crash(self):
        # GitHub returns head.repo: null once the PR's source fork/branch
        # has been deleted - common for old merged PRs. Caught by a live
        # run against real historical PRs, not anticipated up front.
        candidate = harvest.Candidate("o", "r", 1, "Fix bug")
        body = {"changed_files": 1, "additions": 5, "deletions": 5,
                "base": {"sha": "a", "repo": {"full_name": "o/r"}},
                "head": {"sha": "b", "repo": None}}
        client = self._client(body)
        result = asyncio.run(harvest.fetch_pr_shas(client, candidate, "tok"))
        self.assertIsNone(result)

    def test_fork_pr_rejected(self):
        candidate = harvest.Candidate("o", "r", 1, "Fix bug")
        body = {"changed_files": 1, "additions": 5, "deletions": 5,
                "base": {"sha": "a", "repo": {"full_name": "o/r"}},
                "head": {"sha": "b", "repo": {"full_name": "someone-else/r"}}}
        client = self._client(body)
        result = asyncio.run(harvest.fetch_pr_shas(client, candidate, "tok"))
        self.assertIsNone(result)

    def test_valid_small_same_repo_pr_accepted(self):
        candidate = harvest.Candidate("o", "r", 1, "Fix bug")
        body = {"changed_files": 1, "additions": 2, "deletions": 2, "title": "Fix bug", "body": "",
                "base": {"sha": "aaa", "repo": {"full_name": "o/r"}},
                "head": {"sha": "bbb", "repo": {"full_name": "o/r"}}}
        client = self._client(body)
        result = asyncio.run(harvest.fetch_pr_shas(client, candidate, "tok"))
        self.assertIsNotNone(result)
        self.assertEqual(result["base_sha"], "aaa")
        self.assertEqual(result["head_sha"], "bbb")


class TestSearchBugfixPrs(unittest.TestCase):
    def test_filters_to_titles_matching_bugfix_pattern(self):
        def handler(request):
            return httpx.Response(200, json={"items": [
                {"number": 1, "title": "Fix null deref"},
                {"number": 2, "title": "Add new feature"},  # should be filtered out
                {"number": 3, "title": "Bug: crash on empty input"},
            ]})
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        results = asyncio.run(harvest.search_bugfix_prs(client, "o", "r", "tok"))
        numbers = {c.pr_number for c in results}
        self.assertEqual(numbers, {1, 3})


class TestEnsureRepoClone(unittest.TestCase):
    def test_reuses_existing_clone_directory(self):
        with TemporaryDirectory() as tmp:
            clone_root = Path(tmp)
            # Don't actually hit the network - just verify idempotent dir creation.
            import subprocess
            first = clone_root / "o__r"
            first.mkdir()
            (first / ".git").mkdir()
            result = harvest.ensure_repo_clone(clone_root, "o", "r")
            self.assertEqual(result, first)
            self.assertTrue((first / ".git").exists())  # untouched, not recreated


if __name__ == "__main__":
    unittest.main()
