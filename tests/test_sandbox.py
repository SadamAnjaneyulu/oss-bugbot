import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sandbox import SandboxViolation, is_binary, resolve_within_root  # noqa: E402


class TestResolveWithinRoot(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("print('hi')\n")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("[core]\n")
        (self.root / ".env").write_text("SECRET=x\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_normal_file_resolves(self):
        result = resolve_within_root(self.root, "src/app.py")
        self.assertEqual(result, (self.root / "src" / "app.py").resolve())

    def test_dot_dot_traversal_rejected(self):
        with self.assertRaises(SandboxViolation):
            resolve_within_root(self.root, "../../../etc/passwd")

    def test_absolute_path_override_rejected(self):
        # Path(root) / "/proc/self/environ" discards root entirely at the
        # pathlib join step - must still be caught by the commonpath check.
        with self.assertRaises(SandboxViolation):
            resolve_within_root(self.root, "/proc/self/environ")

    def test_proc_prefix_rejected(self):
        with self.assertRaises(SandboxViolation):
            resolve_within_root(self.root, "../../../../proc/self/environ")

    def test_git_dir_rejected(self):
        with self.assertRaises(SandboxViolation):
            resolve_within_root(self.root, ".git/config")

    def test_env_file_rejected(self):
        with self.assertRaises(SandboxViolation):
            resolve_within_root(self.root, ".env")

    def test_symlink_escape_rejected(self):
        outside = TemporaryDirectory()
        try:
            secret = Path(outside.name) / "secret.txt"
            secret.write_text("nope\n")
            link = self.root / "escape_link"
            try:
                os.symlink(secret, link)
            except OSError:
                self.skipTest("symlinks unsupported in this environment")
            with self.assertRaises(SandboxViolation):
                resolve_within_root(self.root, "escape_link")
        finally:
            outside.cleanup()

    def test_symlink_to_git_rejected(self):
        link = self.root / "safe_looking_name"
        try:
            os.symlink(self.root / ".git" / "config", link)
        except OSError:
            self.skipTest("symlinks unsupported in this environment")
        with self.assertRaises(SandboxViolation):
            resolve_within_root(self.root, "safe_looking_name")

    # --- red-team round: adversarial payloads beyond the basic cases above.
    # Production runs on ubuntu-latest; POSIX-relevant attacks get real
    # assertions, Windows-only quirks (trailing dots, case-insensitive FS)
    # are noted, not chased - we don't deploy there.

    def test_percent_encoded_traversal_does_not_escape(self):
        # No URL-decoding happens anywhere in this path - "%2e%2e" is a
        # literal, nonexistent filename, not a decoded "..". It must resolve
        # to a (nonexistent) location still inside root, not escape it.
        result = resolve_within_root(self.root, "%2e%2e/%2e%2e/etc/passwd")
        self.assertTrue(str(result).startswith(str(self.root.resolve())))

    def test_absolute_escape_beyond_proc_sys_rejected(self):
        # The denylist only names /proc and /sys explicitly - anything
        # outside root must be caught by the commonpath check regardless of
        # whether it's on that list. This is the actually sensitive one on
        # a real runner (SSH keys, runner temp secrets), not /proc.
        for target in ("/etc/passwd", "/home/runner/.ssh/id_rsa", "/etc/shadow"):
            with self.subTest(target=target):
                with self.assertRaises(SandboxViolation):
                    resolve_within_root(self.root, target)

    def test_null_byte_in_path_raises_cleanly(self):
        # Python's own filesystem layer rejects embedded NUL (ValueError),
        # not SandboxViolation - assert it fails closed, not that a specific
        # exception type is used, since Python raises before we get a say.
        with self.assertRaises((SandboxViolation, ValueError)):
            resolve_within_root(self.root, "app.py\x00.png")

    def test_deeply_nested_traversal_rejected(self):
        with self.assertRaises(SandboxViolation):
            resolve_within_root(self.root, "a/b/c/d/e/f/../../../../../../../../etc/passwd")

    def test_current_dir_noise_does_not_bypass_denylist(self):
        with self.assertRaises(SandboxViolation):
            resolve_within_root(self.root, "./src/../.git/./config")


class TestIsBinary(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_text_file_not_binary(self):
        p = self.root / "app.py"
        p.write_text("print('hi')\n")
        self.assertFalse(is_binary(p))

    def test_null_byte_is_binary(self):
        p = self.root / "image.png"
        p.write_bytes(b"\x89PNG\x00\x01\x02")
        self.assertTrue(is_binary(p))

    def test_missing_file_treated_as_binary(self):
        self.assertTrue(is_binary(self.root / "does_not_exist"))


if __name__ == "__main__":
    unittest.main()
