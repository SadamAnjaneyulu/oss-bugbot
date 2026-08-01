import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import static  # noqa: E402

SEMGREP_AVAILABLE = shutil.which("semgrep") is not None


@unittest.skipUnless(SEMGREP_AVAILABLE, "semgrep not installed in this environment")
class TestRunSemgrep(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_finds_planted_vendored_rule_violation(self):
        (self.root / "bad.py").write_text(
            "import subprocess\n"
            "def run(cmd):\n"
            "    subprocess.call(cmd, shell=True)\n"
        )
        hits = static.run_semgrep(self.root, ["bad.py"])
        self.assertIn("bad.py", hits)
        check_ids = [h["check_id"] for h in hits["bad.py"]]
        self.assertTrue(any("shell-true" in c for c in check_ids))

    def test_clean_file_produces_no_hits(self):
        (self.root / "clean.py").write_text("def add(a, b):\n    return a + b\n")
        hits = static.run_semgrep(self.root, ["clean.py"])
        self.assertEqual(hits.get("clean.py", []), [])

    def test_empty_file_list_returns_empty_without_running(self):
        hits = static.run_semgrep(self.root, [])
        self.assertEqual(hits, {})

    def test_scoped_to_given_files_only(self):
        (self.root / "bad.py").write_text("eval(x)\n")
        (self.root / "also_bad.py").write_text("eval(y)\n")
        hits = static.run_semgrep(self.root, ["bad.py"])  # only bad.py passed
        self.assertIn("bad.py", hits)
        self.assertNotIn("also_bad.py", hits)


class TestRunSemgrepFailureModes(unittest.TestCase):
    def test_missing_binary_degrades_to_empty_not_crash(self):
        root = Path(".")
        original_cmd_name = "semgrep"
        # Simulate a missing binary by pointing at a nonexistent one via
        # monkeypatching subprocess indirectly: easiest is to just trust the
        # try/except FileNotFoundError path, exercised for real if semgrep
        # genuinely isn't installed (see class skip above for the has-it path).
        import subprocess
        import unittest.mock as mock
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("no semgrep")):
            hits = static.run_semgrep(root, ["some_file.py"])
        self.assertEqual(hits, {})

    def test_timeout_degrades_to_empty_not_crash(self):
        import subprocess
        import unittest.mock as mock
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="semgrep", timeout=60)):
            hits = static.run_semgrep(Path("."), ["some_file.py"])
        self.assertEqual(hits, {})

    def test_malformed_json_output_degrades_to_empty(self):
        import unittest.mock as mock
        fake_proc = mock.MagicMock(returncode=0, stdout="not json{{{")
        with mock.patch("subprocess.run", return_value=fake_proc):
            hits = static.run_semgrep(Path("."), ["some_file.py"])
        self.assertEqual(hits, {})


class TestIsCorroborated(unittest.TestCase):
    def test_exact_line_match(self):
        hits = {"a.py": [{"check_id": "x", "line": 10, "message": ""}]}
        self.assertTrue(static.is_corroborated(hits, "a.py", 10))

    def test_within_tolerance(self):
        hits = {"a.py": [{"check_id": "x", "line": 10, "message": ""}]}
        self.assertTrue(static.is_corroborated(hits, "a.py", 12, tolerance=2))

    def test_outside_tolerance(self):
        hits = {"a.py": [{"check_id": "x", "line": 10, "message": ""}]}
        self.assertFalse(static.is_corroborated(hits, "a.py", 20, tolerance=2))

    def test_different_file_not_corroborated(self):
        hits = {"a.py": [{"check_id": "x", "line": 10, "message": ""}]}
        self.assertFalse(static.is_corroborated(hits, "b.py", 10))

    def test_no_hits_at_all(self):
        self.assertFalse(static.is_corroborated({}, "a.py", 10))


if __name__ == "__main__":
    unittest.main()
