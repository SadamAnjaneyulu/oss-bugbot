import asyncio
import sys
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import diff  # noqa: E402


class TestIsSkippableFile(unittest.TestCase):
    def test_binary_extension_skipped(self):
        self.assertTrue(diff.is_skippable_file("assets/logo.png"))

    def test_lockfile_skipped(self):
        self.assertTrue(diff.is_skippable_file("package-lock.json"))

    def test_minified_skipped(self):
        self.assertTrue(diff.is_skippable_file("dist/bundle.min.js"))

    def test_vendored_dir_skipped(self):
        self.assertTrue(diff.is_skippable_file("node_modules/foo/index.js"))

    def test_normal_source_not_skipped(self):
        self.assertFalse(diff.is_skippable_file("src/auth.py"))


class TestSizeGate(unittest.TestCase):
    def test_under_limits_passes(self):
        files = [diff.FileMeta("a.py", "modified", 5, 2, 7)]
        result = diff.size_gate(files, max_lines=500, max_files=30)
        self.assertTrue(result.ok)

    def test_over_line_limit_rejected(self):
        files = [diff.FileMeta("a.py", "modified", 600, 0, 600)]
        result = diff.size_gate(files, max_lines=500, max_files=30)
        self.assertFalse(result.ok)
        self.assertIn("changed lines", result.reason)

    def test_over_file_limit_rejected(self):
        files = [diff.FileMeta(f"f{i}.py", "modified", 1, 0, 1) for i in range(31)]
        result = diff.size_gate(files, max_lines=500, max_files=30)
        self.assertFalse(result.ok)
        self.assertIn("reviewable files", result.reason)

    def test_binary_and_lockfiles_excluded_from_count(self):
        files = [
            diff.FileMeta("package-lock.json", "modified", 10000, 10000, 20000),
            diff.FileMeta("src/app.py", "modified", 5, 0, 5),
        ]
        result = diff.size_gate(files, max_lines=500, max_files=30)
        self.assertTrue(result.ok)
        self.assertEqual(result.total_lines, 5)
        self.assertEqual(result.total_files, 1)


class TestDeletedFilenames(unittest.TestCase):
    def test_identifies_removed_files(self):
        files = [
            diff.FileMeta("a.py", "modified", 1, 1, 2),
            diff.FileMeta("b.py", "removed", 0, 10, 10),
        ]
        self.assertEqual(diff.deleted_filenames(files), {"b.py"})


SAMPLE_DIFF = """\
diff --git a/src/app.py b/src/app.py
index 1234567..89abcde 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,4 @@
 def handler(req):
-    return None
+    if req.user is None:
+        return None
+    return req.user.name

"""


class TestParseChangedLines(unittest.TestCase):
    def test_parses_added_line_numbers(self):
        result = diff.parse_changed_lines(SAMPLE_DIFF)
        self.assertIn("src/app.py", result)
        # three added lines in the hunk above
        self.assertEqual(len(result["src/app.py"]), 3)

    def test_empty_diff_returns_empty(self):
        self.assertEqual(diff.parse_changed_lines(""), {})


TWO_FILE_DIFF = """\
diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -1,1 +1,1 @@
-old_a
+new_a
diff --git a/b.py b/b.py
index 3333333..4444444 100644
--- a/b.py
+++ b/b.py
@@ -1,1 +1,1 @@
-old_b
+new_b
"""


class TestShuffleHunks(unittest.TestCase):
    def test_empty_diff_returns_empty(self):
        self.assertEqual(diff.shuffle_hunks("", 42), "")

    def test_single_file_diff_unchanged(self):
        result = diff.shuffle_hunks(SAMPLE_DIFF, 42)
        self.assertIn("src/app.py", result)

    def test_all_files_preserved_after_shuffle(self):
        result = diff.shuffle_hunks(TWO_FILE_DIFF, seed=1)
        self.assertIn("a.py", result)
        self.assertIn("b.py", result)
        self.assertIn("new_a", result)
        self.assertIn("new_b", result)

    def test_same_seed_is_deterministic(self):
        r1 = diff.shuffle_hunks(TWO_FILE_DIFF, seed=7)
        r2 = diff.shuffle_hunks(TWO_FILE_DIFF, seed=7)
        self.assertEqual(r1, r2)

    def test_different_seeds_can_produce_different_order(self):
        # Not a strict guarantee for n=2 (50% chance of matching order per
        # seed pair), so try a spread of seeds and require at least one
        # observed reordering rather than asserting on a single pair.
        orders = {diff.shuffle_hunks(TWO_FILE_DIFF, seed=s) for s in range(10)}
        self.assertGreater(len(orders), 1, "expected at least one differently-ordered result across 10 seeds")


class TestFetchPRFiles(unittest.TestCase):
    def test_single_page(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[
                {"filename": "a.py", "status": "modified", "additions": 1, "deletions": 0, "changes": 1, "patch": "..."},
            ])

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        files = asyncio.run(diff.fetch_pr_files(client, "o", "r", 1, "tok"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].filename, "a.py")

    def test_pagination_follows_link_header(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(
                    200,
                    json=[{"filename": "a.py", "status": "modified", "additions": 1, "deletions": 0, "changes": 1, "patch": None}],
                    headers={"Link": '<https://api.github.com/next>; rel="next"'},
                )
            return httpx.Response(
                200,
                json=[{"filename": "b.py", "status": "modified", "additions": 1, "deletions": 0, "changes": 1, "patch": None}],
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        files = asyncio.run(diff.fetch_pr_files(client, "o", "r", 1, "tok"))
        self.assertEqual([f.filename for f in files], ["a.py", "b.py"])
        self.assertEqual(calls["n"], 2)


class TestFetchDiff(unittest.TestCase):
    def test_normal_diff_returned(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=SAMPLE_DIFF)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text = asyncio.run(diff.fetch_diff(client, "o", "r", 1, "tok"))
        self.assertEqual(text, SAMPLE_DIFF)

    def test_406_falls_back_to_per_file_patches(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(406)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        fallback = [
            diff.FileMeta("a.py", "modified", 1, 0, 1, patch="@@ -1 +1 @@\n-x\n+y"),
            diff.FileMeta("b.py", "modified", 1, 0, 1, patch=None),
        ]
        text = asyncio.run(diff.fetch_diff(client, "o", "r", 1, "tok", files_fallback=fallback))
        self.assertIn("@@ -1 +1 @@", text)

    def test_406_with_no_fallback_returns_empty(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(406)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        text = asyncio.run(diff.fetch_diff(client, "o", "r", 1, "tok"))
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
