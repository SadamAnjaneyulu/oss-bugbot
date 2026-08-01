import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tui  # noqa: E402


class TestBugbotTUI(unittest.IsolatedAsyncioTestCase):
    async def test_mounts_without_crashing(self):
        app = tui.BugbotTUI(github_token="fake", gemini_api_key="fake", groq_api_key="fake")
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertIsNotNone(app.query_one("#log"))
            self.assertIsNotNone(app.query_one("#url-input"))

    async def test_bad_url_shows_error_and_does_not_start_a_review(self):
        app = tui.BugbotTUI(github_token="fake", gemini_api_key="fake", groq_api_key="fake")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#url-input").value = "not-a-url"
            await pilot.press("enter")
            await pilot.pause()
            self.assertFalse(app.running)

    async def test_exit_command_closes_the_app(self):
        app = tui.BugbotTUI(github_token="fake", gemini_api_key="fake", groq_api_key="fake")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#url-input").value = "exit"
            await pilot.press("enter")
            await pilot.pause()
        # run_test's context manager unwinding without hanging or raising
        # is the actual assertion - 'exit' must cleanly stop the app.

    async def test_on_progress_updates_checklist_and_log(self):
        app = tui.BugbotTUI(github_token="fake", gemini_api_key="fake", groq_api_key="fake")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.on_progress("size_gate", "ok")
            app.on_progress("a1_pass_done", "p1")
            self.assertEqual(len(app.completed_stages), 2)
            self.assertIn("Size gate", app.completed_stages[0])
            self.assertIn("A1 review pass", app.completed_stages[1])

    async def test_confirm_post_screen_dismisses_true_on_yes(self):
        app = tui.BugbotTUI(github_token="fake", gemini_api_key="fake", groq_api_key="fake", post=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            result = {}
            worker = app.run_worker(self._push_and_capture(app, tui.ConfirmPostScreen("o", "r", 1), result))
            await pilot.pause()
            await pilot.click("#yes")
            await worker.wait()
            self.assertTrue(result["value"])

    async def test_confirm_post_screen_dismisses_false_on_no(self):
        app = tui.BugbotTUI(github_token="fake", gemini_api_key="fake", groq_api_key="fake", post=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            result = {}
            worker = app.run_worker(self._push_and_capture(app, tui.ConfirmPostScreen("o", "r", 1), result))
            await pilot.pause()
            await pilot.click("#no")
            await worker.wait()
            self.assertFalse(result["value"])

    @staticmethod
    async def _push_and_capture(app, screen, result):
        result["value"] = await app.push_screen_wait(screen)

    async def test_bare_repo_with_no_open_prs_shows_message(self):
        # patch() auto-detects cli.list_open_prs is `async def` and installs
        # an AsyncMock - side_effect must return the plain value, NOT a
        # coroutine, or the await in tui.py unwraps to a coroutine object
        # instead of the list (AsyncMock does not double-await side_effect).
        app = tui.BugbotTUI(github_token="fake", gemini_api_key="fake", groq_api_key="fake")
        with patch("tui.cli.list_open_prs", side_effect=lambda *a, **kw: []):
            async with app.run_test() as pilot:
                await pilot.pause()
                app.query_one("#url-input").value = "https://github.com/owner/emptyrepo"
                await pilot.press("enter")
                await pilot.pause()
                self.assertFalse(app.running)


if __name__ == "__main__":
    unittest.main()
