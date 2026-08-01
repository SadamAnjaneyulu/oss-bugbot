"""Persistent textual TUI - the OpenCode/Claude Code style entrypoint the
README's "Local CLI mode" describes as an alternative to cli.py's
print-and-scroll interactive loop. Same pipeline underneath (main.run_review,
identical A1/A2/A3/gates logic) - this file is presentation only, plus the
on_progress wiring that main.py now exposes for exactly this purpose.

Reuses cli.py's already-tested resolve/clone/parse logic rather than
duplicating it - the only new code here is the widget layout and the
progress-event -> UI mapping.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from rich import box
from rich.panel import Panel
from rich.table import Table
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

import cli
from llm import GEMINI_FLASH, GEMINI_FLASH_LITE, GROQ_PRIMARY
from main import run_review

DEMO_OWNER, DEMO_REPO, DEMO_PR = "SadamAnjaneyulu", "oss-bugbot", 2

STAGE_LABELS = {
    "size_gate": "Size gate",
    "diff_fetched": "Diff fetched",
    "semgrep_done": "Semgrep scan",
    "a1_pass_done": "A1 review pass",
    "a2_done": "A2 cluster + vote",
    "a3_done": "A3 adversarial validate",
    "posted": "Post",
}
ALL_STAGES = ["size_gate", "diff_fetched", "semgrep_done"] + ["a1_pass_done"] * 4 + [
    "a2_done", "a3_done", "posted",
]


def _findings_renderables(result: dict) -> list:
    """Same fields cli.print_findings shows, as Rich renderables instead of
    Console.print calls - RichLog.write() takes a renderable directly, so
    this is the TUI-native form of the same presentation logic.
    """
    out: list = []
    if result.get("skipped"):
        out.append(Panel(f"[yellow]Skipped:[/] {result.get('skip_reason')}", border_style="yellow"))
        return out

    findings = result.get("findings", [])
    if not findings:
        out.append(Panel("[bold green]No confirmed findings.[/]", border_style="green"))
    else:
        table = Table(title=f"{len(findings)} finding(s)", box=box.ROUNDED, show_lines=True, title_style="bold")
        table.add_column("File:Line", style="bold")
        table.add_column("Category/Severity")
        table.add_column("Title")
        table.add_column("Verdict")
        table.add_column("Score", justify="right")
        table.add_column("Votes", justify="right")
        for f in findings:
            sev_style = cli.SEVERITY_STYLE.get(f["severity"], "white")
            verdict_style = cli.VERDICT_STYLE.get(f["verdict"], "white")
            table.add_row(
                f"{f['file']}:{f['line']}",
                f"[{sev_style}]{f['category']}/{f['severity']}[/]",
                f["title"],
                f"[{verdict_style}]{f['verdict']}[/]",
                f"{f['score']:.2f}",
                f"{f['vote_count']}/{f['passes_surviving']}",
            )
        out.append(table)

    post = result.get("post_result", {})
    if post.get("reason") == "dry_run":
        out.append(f"[dim]Dry run[/] - would post {post['count']} comment(s). Pass --post at launch to post for real.")
    elif post.get("posted"):
        out.append(f"[bold green]Posted[/] {post['count']} comment(s) to the PR for real.")
    else:
        out.append(f"Nothing posted ([dim]{post.get('reason', 'unknown')}[/]).")

    if result.get("degradations"):
        out.append(f"[yellow]{len(result['degradations'])} degradation(s)[/] - see findings.json for detail.")

    tokens = result.get("token_usage", {})
    if tokens:
        out.append(f"[dim]{tokens.get('total_tokens', 0)} tokens total - $0.00 (free tier)[/]")

    return out


class ConfirmPostScreen(ModalScreen[bool]):
    """Asked once per review, only when the TUI was launched with --post.
    Declining falls back to dry-run for that one review - --post enables
    posting, it doesn't force it every time.
    """
    CSS = """
    ConfirmPostScreen { align: center middle; }
    #dialog { width: 64; height: auto; border: thick $accent; padding: 1 2; background: $surface; }
    #buttons { align: center middle; margin-top: 1; height: auto; }
    Button { margin: 0 1; }
    """

    def __init__(self, owner: str, repo: str, pr_number: int):
        super().__init__()
        self.owner, self.repo, self.pr_number = owner, repo, pr_number

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(f"Post this review for real to {self.owner}/{self.repo}#{self.pr_number}?")
            with Horizontal(id="buttons"):
                yield Button("Post for real", id="yes", variant="error")
                yield Button("Dry run instead", id="no", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class BugbotTUI(App):
    TITLE = "oss-bugbot"
    SUB_TITLE = "AI code review, $0/month"

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #log {
        width: 3fr;
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        padding: 0 1;
    }
    #sidebar {
        width: 1fr;
        min-width: 32;
        border: round $secondary;
        border-title-color: $secondary;
        border-title-style: bold;
        padding: 1;
    }
    #sidebar .section-label { color: $secondary; text-style: bold; margin-top: 1; }
    #sidebar .section-label:first-of-type { margin-top: 0; }
    #stage { color: $text; margin-bottom: 1; }
    #checklist { color: $text-muted; }
    #models { color: $text-muted; }
    #tokens { color: $success; }
    #url-input { dock: bottom; }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, github_token: str, gemini_api_key: str, groq_api_key: str,
                 post: bool = False, demo: bool = False):
        super().__init__()
        self.github_token = github_token
        self.gemini_api_key = gemini_api_key
        self.groq_api_key = groq_api_key
        self.post = post
        self.demo = demo
        self.completed_stages: list[str] = []
        self.running = False
        self._stage_label = "[dim]Idle - paste a repo or PR link below[/]"
        self._run_start: float | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield RichLog(id="log", markup=True, wrap=True, highlight=False)
            with Vertical(id="sidebar"):
                yield Static("Status", classes="section-label")
                yield Static("[dim]Idle - paste a repo or PR link below[/]", id="stage")
                yield Static("Pipeline", classes="section-label")
                yield Static("[dim]No review running yet.[/]", id="checklist")
                yield Static("Models", classes="section-label")
                yield Static(
                    f"[dim]A1[/] {GEMINI_FLASH_LITE}\n[dim]A2[/] {GEMINI_FLASH}\n[dim]A3[/] {GROQ_PRIMARY}",
                    id="models",
                )
                yield Static("Usage", classes="section-label")
                yield Static("[dim]No tokens spent yet.[/]", id="tokens")
        yield Input(placeholder="github.com/owner/repo or .../pull/N  (or 'exit')", id="url-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#log", RichLog).border_title = "Review log"
        self.query_one("#sidebar", Vertical).border_title = "oss-bugbot"
        log = self.query_one("#log", RichLog)
        demo_url = f"github.com/{DEMO_OWNER}/{DEMO_REPO}/pull/{DEMO_PR}"
        log.write(
            "[dim]4x Gemini review . vote . Groq adversarial validate[/]\n"
            "[dim]A 'pull request' (PR) is a proposed code change on GitHub, open for review.[/]\n"
            "[dim]This reviews one PR's changes and flags real bugs in them. Ctrl+C to quit.[/]\n\n"
            f"[dim]New here? Try this PR - it usually catches a real planted bug:[/] [bold cyan]{demo_url}[/]\n"
            "[dim](it's a live LLM ensemble, not scripted - occasionally 0 findings on a rerun is normal)[/]"
        )
        self.query_one(Input).focus()
        if self.demo:
            log.write("[dim]--demo: running the example above automatically...[/]")
            self.run_worker(self.do_review(DEMO_OWNER, DEMO_REPO, DEMO_PR), exclusive=True)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        url = event.value.strip()
        event.input.value = ""
        if not url:
            return
        if url.lower() in ("exit", "quit", "q"):
            self.exit()
            return
        if self.running:
            self.query_one("#log", RichLog).write("[yellow]A review is already in progress - please wait.[/]")
            return

        log = self.query_one("#log", RichLog)
        try:
            owner, repo, pr_number = cli.parse_github_url(url)
        except ValueError as exc:
            log.write(f"[bold red]Error:[/] {exc}")
            return

        if pr_number is None:
            log.write(f"[dim]Looking up open pull requests on {owner}/{repo}...[/]")
            try:
                prs = await cli.list_open_prs(owner, repo, self.github_token)
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                log.write(f"[bold red]Error:[/] could not look up {owner}/{repo}: {exc}")
                return
            if not prs:
                log.write(f"[yellow]{owner}/{repo} has no open pull requests right now.[/] "
                           f"Paste a direct link to a specific PR instead.")
                return
            pr_number = prs[0]["number"]
            log.write(f"[dim]{len(prs)} open PR(s) - reviewing the most recently updated: "
                       f"#{pr_number} {prs[0]['title']}[/]")

        self.run_worker(self.do_review(owner, repo, pr_number), exclusive=True)

    def _set_stage(self, text: str) -> None:
        # Only the label changes here - _render_stage appends the live
        # elapsed-seconds suffix on top of whatever label is current, so a
        # long-running stage (a slow clone, a slow LLM call) still visibly
        # ticks instead of sitting static for 60-120s looking frozen.
        self._stage_label = text
        self._render_stage()

    def _render_stage(self) -> None:
        if self.running and self._run_start is not None:
            elapsed = int(time.monotonic() - self._run_start)
            text = f"{self._stage_label} [dim]({elapsed}s)[/]"
        else:
            text = self._stage_label
        self.query_one("#stage", Static).update(text)

    def on_progress(self, stage: str, detail: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        line = f"[green][OK][/] {label}" + (f" [dim]({detail})[/]" if detail else "")
        self.completed_stages.append(line)
        self.query_one("#checklist", Static).update("\n".join(self.completed_stages))
        self._set_stage(f"[bold cyan]Running:[/] {label}...")
        self.query_one("#log", RichLog).write(line)

    async def do_review(self, owner: str, repo: str, pr_number: int) -> None:
        self.running = True
        self.completed_stages = []
        self.query_one("#checklist", Static).update("[dim]Starting...[/]")
        self._run_start = time.monotonic()
        ticker = self.set_interval(1.0, self._render_stage)
        log = self.query_one("#log", RichLog)

        try:
            self._set_stage(f"[bold cyan]Resolving[/] {owner}/{repo}#{pr_number}...")
            try:
                info = await cli.resolve_pr_info(owner, repo, pr_number, self.github_token)
            except httpx.HTTPStatusError as exc:
                log.write(f"[bold red]Error:[/] could not resolve PR ({exc.response.status_code}).")
                return
            except (httpx.RequestError, ValueError) as exc:
                log.write(f"[bold red]Error:[/] {exc}")
                return
            log.write(f"[green][OK][/] Resolved [bold]{info['head_ref']}[/] @ [dim]{info['head_sha'][:7]}[/]")

            with TemporaryDirectory(prefix="oss-bugbot-tui-") as tmp:
                checkout = Path(tmp)
                self._set_stage(
                    f"[bold cyan]Cloning[/] {info['head_ref']} "
                    f"[dim](up to {cli.CLONE_TIMEOUT_SECONDS}s for a large repo)[/]..."
                )
                try:
                    await asyncio.to_thread(cli.clone_pr_branch, info["head_clone_url"], info["head_ref"], checkout)
                except subprocess.CalledProcessError as exc:
                    log.write(f"[bold red]Error:[/] clone failed:\n{exc.stderr}")
                    return
                except subprocess.TimeoutExpired:
                    log.write(f"[bold red]Error:[/] clone exceeded {cli.CLONE_TIMEOUT_SECONDS}s. "
                              f"This repo/branch is too large for local CLI mode - try a smaller PR.")
                    return
                except FileNotFoundError:
                    log.write("[bold red]Error:[/] git executable not found on PATH.")
                    return
                log.write("[green][OK][/] Cloned")

                post_this_run = False
                if self.post:
                    post_this_run = await self.push_screen_wait(ConfirmPostScreen(owner, repo, pr_number))
                    log.write("[dim]Posting for real this run[/]" if post_this_run else "[dim]Dry run (declined)[/]")

                result = await run_review(
                    owner, repo, pr_number, info["head_sha"], self.github_token,
                    self.gemini_api_key, self.groq_api_key, checkout, post=post_this_run,
                    on_progress=self.on_progress,
                )

            Path("findings.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            for renderable in _findings_renderables(result):
                log.write(renderable)

            tokens = result.get("token_usage", {})
            self.query_one("#tokens", Static).update(
                f"[bold]{tokens.get('total_tokens', 0)}[/] tokens\n[dim]$0.00 (free tier)[/]"
            )
        finally:
            ticker.stop()
            self.running = False
            self._set_stage("[dim]Idle - paste another link, or 'exit'[/]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run oss-bugbot locally with a persistent split-pane TUI.")
    parser.add_argument("--post", action="store_true",
                         help="Enable posting real reviews - still asks to confirm before each individual post. "
                              "Default is dry-run only (findings printed and written to findings.json, never posted).")
    parser.add_argument("--demo", action="store_true",
                         help="Skip typing anything - automatically reviews oss-bugbot PR #2 (a real planted SQL "
                              "injection) the moment the app starts. It's a live LLM ensemble, not scripted, so an "
                              "occasional 0-findings rerun is expected variance, not a bug.")
    args = parser.parse_args()

    missing = cli.check_env_vars()
    if missing:
        print(f"Error: missing required environment variable(s): {', '.join(missing)}")
        print("Set them in your current shell before running this (see README, Local CLI mode).")
        return 1

    app = BugbotTUI(
        github_token=os.environ["GITHUB_TOKEN"],
        gemini_api_key=os.environ["GEMINI_API_KEY"],
        groq_api_key=os.environ["GROQ_API_KEY"],
        post=args.post,
        demo=args.demo,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
