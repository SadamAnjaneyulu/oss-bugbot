import asyncio
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cli  # noqa: E402


def _test_console() -> tuple[Console, io.StringIO]:
    """Rich's Console binds its output file at construction time, so
    contextlib.redirect_stdout (which patches sys.stdout) never reaches it -
    tests must build a Console(file=buf) and pass it via print_findings's
    `out` parameter instead. force_terminal=False, no_color=True strips
    ANSI styling so assertions can match plain substrings.
    """
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, no_color=True, width=120), buf


class TestParsePrUrl(unittest.TestCase):
    def test_standard_url(self):
        owner, repo, num = cli.parse_pr_url("https://github.com/tensorflow/tensorflow/pull/12345")
        self.assertEqual((owner, repo, num), ("tensorflow", "tensorflow", 12345))

    def test_trailing_slash_and_whitespace_tolerated(self):
        owner, repo, num = cli.parse_pr_url("  https://github.com/owner/repo/pull/1/  \n")
        self.assertEqual((owner, repo, num), ("owner", "repo", 1))

    def test_http_not_just_https(self):
        owner, repo, num = cli.parse_pr_url("http://github.com/owner/repo/pull/7")
        self.assertEqual((owner, repo, num), ("owner", "repo", 7))

    def test_rejects_non_pr_url(self):
        with self.assertRaises(ValueError):
            cli.parse_pr_url("https://github.com/owner/repo/issues/5")

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            cli.parse_pr_url("not a url at all")

    def test_repo_names_with_dots_and_dashes(self):
        owner, repo, num = cli.parse_pr_url("https://github.com/my-org/my.repo-name/pull/42")
        self.assertEqual((owner, repo, num), ("my-org", "my.repo-name", 42))


class TestPrintFindings(unittest.TestCase):
    def test_skipped_prints_reason(self):
        test_console, buf = _test_console()
        cli.print_findings({"skipped": True, "skip_reason": "too many lines"}, posted_for_real=False, out=test_console)
        self.assertIn("too many lines", buf.getvalue())

    def test_no_findings_says_so(self):
        test_console, buf = _test_console()
        cli.print_findings({"findings": [], "post_result": {"reason": "dry_run", "count": 0}},
                            posted_for_real=False, out=test_console)
        self.assertIn("No confirmed findings", buf.getvalue())

    def test_dry_run_tells_user_to_pass_post_flag(self):
        result = {
            "findings": [{"file": "a.py", "line": 1, "category": "logic", "severity": "high",
                          "title": "bug", "verdict": "confirmed", "score": 0.8,
                          "vote_count": 3, "passes_surviving": 4}],
            "post_result": {"reason": "dry_run", "count": 1, "posted": False},
            "degradations": [],
        }
        test_console, buf = _test_console()
        cli.print_findings(result, posted_for_real=False, out=test_console)
        out = buf.getvalue()
        self.assertIn("Dry run", out)
        self.assertIn("--post", out)
        self.assertIn("a.py:1", out)

    def test_real_post_says_posted(self):
        result = {
            "findings": [],
            "post_result": {"posted": True, "count": 2, "reason": None},
            "degradations": [],
        }
        test_console, buf = _test_console()
        cli.print_findings(result, posted_for_real=True, out=test_console)
        self.assertIn("Posted 2 comment(s) to the PR for real", buf.getvalue())

    def test_degradations_surfaced(self):
        result = {
            "findings": [],
            "post_result": {"reason": "dry_run", "count": 0, "posted": False},
            "degradations": [{"node": "a1", "action": "dropped"}],
        }
        test_console, buf = _test_console()
        cli.print_findings(result, posted_for_real=False, out=test_console)
        self.assertIn("1 degradation", buf.getvalue())


class TestMainMissingEnvVars(unittest.TestCase):
    def test_missing_all_keys_reports_all_and_exits_nonzero(self):
        test_args = ["cli.py", "--pr", "https://github.com/o/r/pull/1"]
        with patch.object(sys, "argv", test_args), patch.dict("os.environ", {}, clear=True):
            code = cli.main()
        self.assertEqual(code, 1)

    def test_invalid_url_exits_before_touching_env(self):
        test_args = ["cli.py", "--pr", "not-a-url"]
        with patch.object(sys, "argv", test_args):
            code = cli.main()
        self.assertEqual(code, 1)


class TestResolvePrInfo(unittest.TestCase):
    def test_deleted_fork_raises_clear_value_error(self):
        # GitHub returns head.repo: null when the PR's source fork/branch
        # was deleted - a normal occurrence, not exotic. Must fail with a
        # clear message, not a raw TypeError on data["head"]["repo"]["clone_url"].
        import httpx as httpx_module

        def handler(request):
            return httpx_module.Response(200, json={"head": {"sha": "abc123", "ref": "feature", "repo": None}})

        with patch("cli.httpx.AsyncClient", return_value=httpx_module.AsyncClient(transport=httpx_module.MockTransport(handler))):
            with self.assertRaises(ValueError) as ctx:
                asyncio.run(cli.resolve_pr_info("o", "r", 1, "tok"))
        self.assertIn("deleted", str(ctx.exception))


class TestMainErrorPaths(unittest.TestCase):
    def _env(self):
        return {"GITHUB_TOKEN": "t", "GEMINI_API_KEY": "g", "GROQ_API_KEY": "q"}

    def test_deleted_fork_exits_cleanly_not_traceback(self):
        test_args = ["cli.py", "--pr", "https://github.com/o/r/pull/1"]

        async def fake_resolve(*a, **kw):
            raise ValueError("PR #1's source repository has been deleted - nothing left to clone.")

        with patch.object(sys, "argv", test_args), \
             patch.dict("os.environ", self._env(), clear=True), \
             patch("cli.resolve_pr_info", side_effect=fake_resolve):
            code = cli.main()
        self.assertEqual(code, 1)

    def test_network_error_during_resolve_exits_cleanly(self):
        import httpx as httpx_module
        test_args = ["cli.py", "--pr", "https://github.com/o/r/pull/1"]

        async def fake_resolve(*a, **kw):
            raise httpx_module.ConnectError("connection failed")

        with patch.object(sys, "argv", test_args), \
             patch.dict("os.environ", self._env(), clear=True), \
             patch("cli.resolve_pr_info", side_effect=fake_resolve):
            code = cli.main()
        self.assertEqual(code, 1)

    def test_missing_git_binary_exits_cleanly(self):
        test_args = ["cli.py", "--pr", "https://github.com/o/r/pull/1"]

        async def fake_resolve(*a, **kw):
            return {"head_sha": "abc", "head_clone_url": "https://x/y.git", "head_ref": "main"}

        with patch.object(sys, "argv", test_args), \
             patch.dict("os.environ", self._env(), clear=True), \
             patch("cli.resolve_pr_info", side_effect=fake_resolve), \
             patch("cli.clone_pr_branch", side_effect=FileNotFoundError("git not found")):
            code = cli.main()
        self.assertEqual(code, 1)


class TestClonePrBranch(unittest.TestCase):
    def test_clone_timeout_reported_not_raised_uncaught(self):
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="git", timeout=1)):
            with self.assertRaises(sp.TimeoutExpired):
                cli.clone_pr_branch("https://example.invalid/x.git", "main", Path("/tmp/nope"))
        # main() itself is what catches this and prints a clean message -
        # clone_pr_branch's job is only to raise, not swallow.


if __name__ == "__main__":
    unittest.main()
