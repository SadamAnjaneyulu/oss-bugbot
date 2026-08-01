import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tools  # noqa: E402


class TestReadFile(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "app.py").write_text("line1\nline2\nline3\nline4\nline5\n")
        (self.root / "image.png").write_bytes(b"\x89PNG\x00\x01")

    def tearDown(self):
        self._tmp.cleanup()

    def test_full_read(self):
        result = tools.read_file(self.root, "app.py")
        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "line1\nline2\nline3\nline4\nline5")

    def test_range_read(self):
        result = tools.read_file(self.root, "app.py", start=2, end=3)
        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "line2\nline3")

    def test_binary_rejected(self):
        result = tools.read_file(self.root, "image.png")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "binary_file")

    def test_missing_file(self):
        result = tools.read_file(self.root, "nope.py")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "not_found")

    def test_traversal_denied(self):
        result = tools.read_file(self.root, "../../../etc/passwd")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "path_denied")


class TestListDir(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("x = 1\n")
        (self.root / ".git").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_lists_entries_and_skips_git(self):
        result = tools.list_dir(self.root, ".")
        self.assertTrue(result["ok"])
        self.assertIn("src/", result["entries"])
        self.assertNotIn(".git/", result["entries"])

    def test_not_a_directory(self):
        (self.root / "file.txt").write_text("x\n")
        result = tools.list_dir(self.root, "file.txt")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "not_a_directory")


class TestFindReferences(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_finds_matches(self):
        (self.root / "a.py").write_text("getUser()\nx = getUser()\n")
        (self.root / "b.py").write_text("def getUser(): pass\n")
        result = tools.find_references(self.root, "getUser")
        self.assertTrue(result["ok"])
        self.assertEqual(result["total"], 3)

    def test_no_matches(self):
        (self.root / "a.py").write_text("print('hi')\n")
        result = tools.find_references(self.root, "nonexistentSymbol")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "no_references")

    def test_word_boundary_not_substring(self):
        (self.root / "a.py").write_text("getUserById()\n")
        result = tools.find_references(self.root, "getUser")
        self.assertFalse(result["ok"])

    def test_capped_at_50(self):
        content = "\n".join(f"call_target()  # line {i}" for i in range(60))
        (self.root / "big.py").write_text(content)
        result = tools.find_references(self.root, "call_target")
        self.assertTrue(result["ok"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["shown"], 50)
        self.assertEqual(result["total"], 60)


class TestFindDefinition(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_python_def(self):
        (self.root / "a.py").write_text("def getUser(id):\n    pass\n")
        result = tools.find_definition(self.root, "getUser")
        self.assertTrue(result["ok"])
        self.assertEqual(result["results"][0]["file"], "a.py")

    def test_python_class(self):
        (self.root / "a.py").write_text("class UserModel:\n    pass\n")
        result = tools.find_definition(self.root, "UserModel")
        self.assertTrue(result["ok"])

    def test_typescript_function(self):
        (self.root / "a.ts").write_text("export function getUser(id: string) {}\n")
        result = tools.find_definition(self.root, "getUser")
        self.assertTrue(result["ok"])

    def test_typescript_const_arrow(self):
        (self.root / "a.ts").write_text("export const getUser = (id: string) => {}\n")
        result = tools.find_definition(self.root, "getUser")
        self.assertTrue(result["ok"])

    def test_not_found(self):
        (self.root / "a.py").write_text("x = 1\n")
        result = tools.find_definition(self.root, "getUser")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "symbol_not_found")

    def test_ignores_non_py_ts_files(self):
        (self.root / "a.go").write_text("func getUser() {}\n")
        result = tools.find_definition(self.root, "getUser")
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
