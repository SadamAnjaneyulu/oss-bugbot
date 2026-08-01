import asyncio
import sys
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import post  # noqa: E402
from schemas import Finding, ValidatorOutput  # noqa: E402


def make_pair(file="app.py", line=10, category="logic", title="null deref", verdict="confirmed", comment="**Bug**: x"):
    f = Finding(file=file, line=line, category=category, severity="medium",
                title=title, reasoning="x", evidence_lines=[line], self_confidence=0.8)
    v = ValidatorOutput(cluster_id="c1", verdict=verdict, refutation="none",
                         validator_family="llama", validator_confidence=0.9, comment_markdown=comment)
    return f, v


class TestMarkerHash(unittest.TestCase):
    def test_deterministic(self):
        h1 = post.marker_hash("app.py", "logic", "Null Deref")
        h2 = post.marker_hash("app.py", "logic", "Null Deref")
        self.assertEqual(h1, h2)

    def test_case_and_whitespace_insensitive(self):
        h1 = post.marker_hash("app.py", "logic", "Null   Deref")
        h2 = post.marker_hash("app.py", "logic", "null deref")
        self.assertEqual(h1, h2)

    def test_different_files_differ(self):
        h1 = post.marker_hash("app.py", "logic", "title")
        h2 = post.marker_hash("other.py", "logic", "title")
        self.assertNotEqual(h1, h2)

    def test_line_number_not_part_of_hash(self):
        # marker_hash takes no line argument at all - this test documents why.
        import inspect
        params = list(inspect.signature(post.marker_hash).parameters)
        self.assertNotIn("line", params)


class TestSanitizeComment(unittest.TestCase):
    def test_strips_html_tags(self):
        result = post.sanitize_comment("<script>alert(1)</script>real content")
        self.assertNotIn("<script>", result)
        self.assertIn("real content", result)

    def test_breaks_markdown_autolinks(self):
        result = post.sanitize_comment("[click here](http://evil.example.com)")
        self.assertNotIn("](http", result)

    def test_truncates_long_comments(self):
        long_text = "x" * 5000
        result = post.sanitize_comment(long_text, max_length=100)
        self.assertLessEqual(len(result), 130)
        self.assertIn("truncated", result)

    def test_short_comment_unmodified_content(self):
        result = post.sanitize_comment("**Bug**: null deref at line 10")
        self.assertIn("null deref", result)


class TestBuildReviewComments(unittest.TestCase):
    def test_builds_comment_for_new_finding(self):
        pair = make_pair()
        result = post.build_review_comments([pair], existing_markers=set())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["path"], "app.py")
        self.assertEqual(result[0]["line"], 10)

    def test_dedupes_against_existing_marker(self):
        pair = make_pair()
        h = post.marker_hash("app.py", "logic", "null deref")
        result = post.build_review_comments([pair], existing_markers={h})
        self.assertEqual(result, [])

    def test_line_shift_does_not_repost(self):
        pair_old_line = make_pair(line=10)
        h_old = post.marker_hash("app.py", "logic", "null deref")
        pair_new_line = make_pair(line=15)  # same bug, line shifted by an earlier insertion
        result = post.build_review_comments([pair_new_line], existing_markers={h_old})
        self.assertEqual(result, [])  # same file/category/title -> same hash -> deduped despite line change

    def test_marker_embedded_in_body(self):
        pair = make_pair()
        result = post.build_review_comments([pair], existing_markers=set())
        h = post.marker_hash("app.py", "logic", "null deref")
        self.assertIn(post.marker_comment(h), result[0]["body"])


class TestFetchExistingMarkers(unittest.TestCase):
    def test_extracts_markers_from_comments(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[
                {"body": "some text\n<!-- bugbot:v1:abc123456789 -->"},
                {"body": "human comment, no marker"},
            ])

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        markers = asyncio.run(post.fetch_existing_markers(client, "o", "r", 1, "tok"))
        self.assertEqual(markers, {"abc123456789"})

    def test_paginates(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(
                    200, json=[{"body": "<!-- bugbot:v1:111111111111 -->"}],
                    headers={"Link": '<https://api.github.com/next>; rel="next"'},
                )
            return httpx.Response(200, json=[{"body": "<!-- bugbot:v1:222222222222 -->"}])

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        markers = asyncio.run(post.fetch_existing_markers(client, "o", "r", 1, "tok"))
        self.assertEqual(markers, {"111111111111", "222222222222"})


class TestPostReview(unittest.TestCase):
    def test_empty_to_post_makes_no_api_call(self):
        called = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            called["n"] += 1
            return httpx.Response(200, json={})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = asyncio.run(post.post_review(client, "o", "r", 1, "tok", "sha", []))
        self.assertFalse(result["posted"])
        self.assertEqual(called["n"], 0)

    def test_single_review_call_for_multiple_comments(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"id": 1})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        to_post = [
            {"path": "a.py", "line": 1, "body": "finding 1", "_hash": "h1"},
            {"path": "b.py", "line": 2, "body": "finding 2", "_hash": "h2"},
        ]
        result = asyncio.run(post.post_review(client, "o", "r", 1, "tok", "sha", to_post))
        self.assertTrue(result["posted"])
        self.assertFalse(result["fallback"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(calls), 1)  # exactly one API call regardless of finding count

    def test_422_falls_back_to_body_only(self):
        call_log = []

        def handler(request: httpx.Request) -> httpx.Response:
            call_log.append(request)
            if len(call_log) == 1:
                return httpx.Response(422, json={"message": "invalid line"})
            return httpx.Response(200, json={"id": 2})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        to_post = [{"path": "a.py", "line": 999, "body": "finding 1", "_hash": "h1"}]
        result = asyncio.run(post.post_review(client, "o", "r", 1, "tok", "sha", to_post))
        self.assertTrue(result["posted"])
        self.assertTrue(result["fallback"])
        self.assertEqual(len(call_log), 2)


if __name__ == "__main__":
    unittest.main()
