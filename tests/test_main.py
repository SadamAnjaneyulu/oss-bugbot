import asyncio
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main  # noqa: E402
from diff import FileMeta  # noqa: E402
from llm import TokenUsage  # noqa: E402
from passes import PassUsage  # noqa: E402
from schemas import Cluster, Finding, ValidatorOutput  # noqa: E402


class TestComputeScore(unittest.TestCase):
    def test_confirmed_scores_above_zero(self):
        score = main.compute_score(vote_count=3, passes_surviving=4, verdict="confirmed")
        self.assertGreater(score, 0.0)

    def test_false_positive_scores_zero(self):
        score = main.compute_score(vote_count=3, passes_surviving=4, verdict="false_positive")
        self.assertEqual(score, 0.0)

    def test_uncertain_scores_between(self):
        confirmed = main.compute_score(3, 4, "confirmed")
        uncertain = main.compute_score(3, 4, "uncertain")
        self.assertGreater(confirmed, uncertain)
        self.assertGreater(uncertain, 0.0)

    def test_zero_passes_surviving_returns_zero(self):
        self.assertEqual(main.compute_score(0, 0, "confirmed"), 0.0)

    def test_unanimous_small_sample_beats_split_larger_sample(self):
        # Mirrors the stats.py property this formula depends on end to end.
        small_unanimous = main.compute_score(2, 2, "confirmed")
        larger_split = main.compute_score(2, 4, "confirmed")
        self.assertGreater(small_unanimous, larger_split)


class TestRunReviewOversizeGate(unittest.TestCase):
    def test_oversize_pr_skips_before_diff_fetch(self):
        big_files = [FileMeta(f"f{i}.py", "modified", 100, 0, 100) for i in range(10)]  # 1000 lines total

        async def fake_fetch_pr_files(*a, **kw):
            return big_files

        with patch("main.fetch_pr_files", side_effect=fake_fetch_pr_files), \
             patch("main.fetch_diff", new_callable=AsyncMock) as mock_fetch_diff:
            result = asyncio.run(main.run_review("o", "r", 1, "sha", "tok", "gk", "qk", Path(".")))

        self.assertTrue(result["skipped"])
        self.assertIn("changed lines", result["skip_reason"])
        mock_fetch_diff.assert_not_called()  # the whole point of metadata-first gating
        # Skipped before any LLM call - token_usage must still be present
        # (same schema key on every return path) and all zero.
        self.assertEqual(result["token_usage"]["total_tokens"], 0)


class TestRunReviewHappyPath(unittest.TestCase):
    def test_full_pipeline_wiring_confirmed_finding_gets_posted(self):
        files = [FileMeta("app.py", "modified", 2, 1, 3)]
        diff_text = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1,1 +1,2 @@\n def f():\n+    return None\n"
        )
        finding = Finding(
            file="app.py", line=2, category="logic", severity="high", title="bug",
            reasoning="x", evidence_lines=[2], self_confidence=0.8,
        )
        pass_result = type("R", (), {"findings": [finding]})()
        cluster = Cluster(cluster_id="c1", vote_count=4, supporting_pass_ids=["p1", "p2", "p3", "p4"], merged=finding)
        verdict = ValidatorOutput(
            cluster_id="c1", verdict="confirmed", refutation="none",
            validator_family="llama", validator_confidence=0.9, comment_markdown="**Bug**: real",
        )

        async def fake_fetch_pr_files(*a, **kw):
            return files

        async def fake_fetch_diff(*a, **kw):
            return diff_text

        async def fake_run_pass(*a, **kw):
            return pass_result, PassUsage(input_tokens=10, output_tokens=20, tool_calls_used=1)

        async def fake_run_aggregation(*a, **kw):
            return [cluster], False, TokenUsage(input_tokens=5, output_tokens=15)

        async def fake_run_all_validators(*a, **kw):
            return [verdict], TokenUsage(input_tokens=8, output_tokens=12)

        async def fake_fetch_existing_markers(*a, **kw):
            return set()

        async def fake_post_review(*a, **kw):
            # Capture what was actually posted so we can assert on it.
            fake_post_review.called_with = a
            return {"posted": True, "fallback": False, "count": len(a[-1])}

        with TemporaryDirectory() as tmp, \
             patch("main.fetch_pr_files", side_effect=fake_fetch_pr_files), \
             patch("main.fetch_diff", side_effect=fake_fetch_diff), \
             patch("main.run_semgrep", return_value={}), \
             patch("main.run_pass", side_effect=fake_run_pass), \
             patch("main.run_aggregation", side_effect=fake_run_aggregation), \
             patch("main.run_all_validators", side_effect=fake_run_all_validators), \
             patch("main.fetch_existing_markers", side_effect=fake_fetch_existing_markers), \
             patch("main.post_review", side_effect=fake_post_review), \
             patch("main.genai.Client"), patch("main.AsyncGroq"):
            result = asyncio.run(main.run_review("o", "r", 1, "sha", "tok", "gk", "qk", Path(tmp)))

        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["verdict"], "confirmed")
        self.assertGreater(result["findings"][0]["score"], 0.0)
        self.assertEqual(result["post_result"]["posted"], True)
        # One comment was built and handed to post_review for the confirmed finding
        posted_comments = fake_post_review.called_with[-1]
        self.assertEqual(len(posted_comments), 1)
        self.assertEqual(posted_comments[0]["path"], "app.py")

        # Token accounting: 4 A1 passes (10/20 each) + 1 A2 call (5/15) +
        # 1 A3 call (8/12), summed across every stage into one total.
        usage = result["token_usage"]
        self.assertEqual(usage["input_tokens"], 4 * 10 + 5 + 8)
        self.assertEqual(usage["output_tokens"], 4 * 20 + 15 + 12)
        self.assertEqual(usage["total_tokens"], usage["input_tokens"] + usage["output_tokens"])
        self.assertEqual(usage["calls"], {"a1": 4, "a2": 1, "a3": 1})

    def test_dry_run_never_calls_post_review_but_still_reports_findings(self):
        # cli.py's default (--post not passed) must never hit the write
        # endpoint - this is the actual safety property, not just a flag
        # that exists. Assert the real post_review function is never
        # invoked, not just that the result dict looks right.
        files = [FileMeta("app.py", "modified", 2, 1, 3)]
        diff_text = (
            "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
            "@@ -1,1 +1,2 @@\n def f():\n+    return None\n"
        )
        finding = Finding(
            file="app.py", line=2, category="logic", severity="high", title="bug",
            reasoning="x", evidence_lines=[2], self_confidence=0.8,
        )
        pass_result = type("R", (), {"findings": [finding]})()
        cluster = Cluster(cluster_id="c1", vote_count=4, supporting_pass_ids=["p1", "p2", "p3", "p4"], merged=finding)
        verdict = ValidatorOutput(
            cluster_id="c1", verdict="confirmed", refutation="none",
            validator_family="llama", validator_confidence=0.9, comment_markdown="**Bug**: real",
        )

        async def fake_fetch_pr_files(*a, **kw):
            return files

        async def fake_fetch_diff(*a, **kw):
            return diff_text

        async def fake_run_pass(*a, **kw):
            return pass_result, PassUsage(input_tokens=10, output_tokens=20, tool_calls_used=1)

        async def fake_run_aggregation(*a, **kw):
            return [cluster], False, TokenUsage(input_tokens=5, output_tokens=15)

        async def fake_run_all_validators(*a, **kw):
            return [verdict], TokenUsage(input_tokens=8, output_tokens=12)

        async def fake_fetch_existing_markers(*a, **kw):
            return set()

        with TemporaryDirectory() as tmp, \
             patch("main.fetch_pr_files", side_effect=fake_fetch_pr_files), \
             patch("main.fetch_diff", side_effect=fake_fetch_diff), \
             patch("main.run_semgrep", return_value={}), \
             patch("main.run_pass", side_effect=fake_run_pass), \
             patch("main.run_aggregation", side_effect=fake_run_aggregation), \
             patch("main.run_all_validators", side_effect=fake_run_all_validators), \
             patch("main.fetch_existing_markers", side_effect=fake_fetch_existing_markers), \
             patch("main.post_review") as mock_post_review, \
             patch("main.genai.Client"), patch("main.AsyncGroq"):
            result = asyncio.run(main.run_review("o", "r", 1, "sha", "tok", "gk", "qk", Path(tmp), post=False))

        mock_post_review.assert_not_called()
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["verdict"], "confirmed")
        self.assertEqual(result["post_result"]["reason"], "dry_run")
        self.assertEqual(result["post_result"]["count"], 1)
        self.assertFalse(result["post_result"]["posted"])

    def test_false_positive_never_reaches_post(self):
        files = [FileMeta("app.py", "modified", 1, 0, 1)]
        diff_text = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1,0 +1,1 @@\n+x = 1\n"
        finding = Finding(
            file="app.py", line=1, category="logic", severity="low", title="nit",
            reasoning="x", evidence_lines=[1], self_confidence=0.3,
        )
        pass_result = type("R", (), {"findings": [finding]})()
        cluster = Cluster(cluster_id="c1", vote_count=1, supporting_pass_ids=["p1"], merged=finding)
        verdict = ValidatorOutput(
            cluster_id="c1", verdict="false_positive", refutation="not a real bug",
            validator_family="llama", validator_confidence=0.9, comment_markdown="",
        )

        async def fake_fetch_pr_files(*a, **kw):
            return files

        async def fake_fetch_diff(*a, **kw):
            return diff_text

        async def fake_run_pass(*a, **kw):
            return pass_result, PassUsage(input_tokens=10, output_tokens=20, tool_calls_used=1)

        async def fake_run_aggregation(*a, **kw):
            return [cluster], False, TokenUsage(input_tokens=5, output_tokens=15)

        async def fake_run_all_validators(*a, **kw):
            return [verdict], TokenUsage(input_tokens=8, output_tokens=12)

        async def fake_fetch_existing_markers(*a, **kw):
            return set()

        async def fake_post_review(*a, **kw):
            fake_post_review.called_with = a
            return {"posted": False, "reason": "nothing_new_to_post", "count": 0, "fallback": False}

        with TemporaryDirectory() as tmp, \
             patch("main.fetch_pr_files", side_effect=fake_fetch_pr_files), \
             patch("main.fetch_diff", side_effect=fake_fetch_diff), \
             patch("main.run_semgrep", return_value={}), \
             patch("main.run_pass", side_effect=fake_run_pass), \
             patch("main.run_aggregation", side_effect=fake_run_aggregation), \
             patch("main.run_all_validators", side_effect=fake_run_all_validators), \
             patch("main.fetch_existing_markers", side_effect=fake_fetch_existing_markers), \
             patch("main.post_review", side_effect=fake_post_review), \
             patch("main.genai.Client"), patch("main.AsyncGroq"):
            result = asyncio.run(main.run_review("o", "r", 1, "sha", "tok", "gk", "qk", Path(tmp)))

        posted_comments = fake_post_review.called_with[-1]
        self.assertEqual(posted_comments, [])  # false_positive finding never becomes a comment
        self.assertFalse(result["post_result"]["posted"])


if __name__ == "__main__":
    unittest.main()
