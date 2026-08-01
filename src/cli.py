"""Local dev/demo/eval wrapper - NOT the production runtime. Points the same
pipeline (main.run_review, identical A1/A2/A3/gates logic) at any public PR
by URL, run from the user's own machine with their own token.

Different security model from review.yml, documented separately rather than
folded into the S1-S8 fork-PR threat model: there is no pull_request_target
sandbox here because there is no untrusted-context write-token exposure to
defend against - this is a user running software they control, on hardware
they control, reviewing a PR they chose to point it at. The invariant that
DOES carry over unchanged: clone, never execute. This script only clones
and reads; it never installs, builds, or runs anything from the cloned PR.

Usage:
    python src/cli.py                                     # interactive session
    python src/cli.py --pr https://github.com/owner/repo/pull/123   # one-shot
    python src/cli.py --pr https://github.com/owner/repo/pull/123 --post

Auth note: a fine-grained PAT only works on repos you own or a repo owner
explicitly approved it for - it will not work against someone else's
random public repo. For that, use a classic PAT with the public_repo scope
(github.com/settings/tokens -> Generate new token (classic)).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import box

from main import default_provider_configs_from_env, run_review

PR_URL_RE = re.compile(r"github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)")
REPO_URL_RE = re.compile(r"github\.com/([^/\s]+)/([^/\s.]+?)(?:\.git)?/?$")
GITHUB_API = "https://api.github.com"
CLONE_TIMEOUT_SECONDS = 120

# Presentation only - every function below this line is display, not logic.
# parse_pr_url/resolve_pr_info/clone_pr_branch (further down) are untouched.
console = Console()
err_console = Console(stderr=True)

SEVERITY_STYLE = {"high": "bold red", "medium": "yellow", "low": "cyan"}
VERDICT_STYLE = {"confirmed": "bold green", "uncertain": "yellow", "false_positive": "dim"}


def print_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]oss-bugbot[/]  [dim]local CLI[/]\n"
        "[dim]4x Gemini review - Gemini vote - Groq adversarial validate - $0/month[/]",
        border_style="cyan",
    ))


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = PR_URL_RE.search(url.strip())
    if not m:
        raise ValueError(f"not a GitHub PR URL (expected .../owner/repo/pull/N): {url}")
    owner, repo, number = m.groups()
    return owner, repo, int(number)


def parse_github_url(url: str) -> tuple[str, str, int | None]:
    """Interactive mode's entry point - accepts either a full PR link or a
    bare repo link. pr_number is None for a bare repo link; the caller
    resolves which PR to review via list_open_prs. Most people pasting a
    link into this will paste the repo, not a specific PR - this is the
    fix for that, not a workaround for it.
    """
    url = url.strip()
    m = PR_URL_RE.search(url)
    if m:
        owner, repo, number = m.groups()
        return owner, repo, int(number)
    m = REPO_URL_RE.search(url)
    if m:
        owner, repo = m.groups()
        return owner, repo, None
    raise ValueError(
        f"that doesn't look like a GitHub link. Paste a repo "
        f"(github.com/owner/repo) or a specific pull request "
        f"(github.com/owner/repo/pull/N): {url}"
    )


async def list_open_prs(owner: str, repo: str, token: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            params={"state": "open", "per_page": 25, "sort": "updated", "direction": "desc"},
        )
        resp.raise_for_status()
        return resp.json()


async def resolve_pr_info(owner: str, repo: str, pr_number: int, token: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        data = resp.json()

    # GitHub returns head.repo: null whenever the PR's source fork/branch has
    # since been deleted - a normal occurrence, not exotic. Fail with a clear
    # message here rather than a raw TypeError three lines down.
    if data["head"]["repo"] is None:
        raise ValueError(
            f"PR #{pr_number}'s source repository has been deleted - nothing left to clone."
        )

    return {
        "head_sha": data["head"]["sha"],
        "head_clone_url": data["head"]["repo"]["clone_url"],
        "head_ref": data["head"]["ref"],
    }


def clone_pr_branch(clone_url: str, ref: str, dest: Path) -> None:
    """--depth 1: this is a read-only demo/eval tool, full history is never
    needed. If the clone alone takes over CLONE_TIMEOUT_SECONDS, the repo is
    a poor fit for local ad-hoc review - the Actions runtime doesn't have
    this problem since GitHub-hosted runners aren't bandwidth/disk limited
    the way a laptop can be.
    """
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, "--single-branch", clone_url, str(dest)],
        check=True, capture_output=True, text=True, timeout=CLONE_TIMEOUT_SECONDS,
    )


def print_findings(result: dict, posted_for_real: bool, out: Console | None = None) -> None:
    """out: injectable Console, defaults to the module-level one. Rich's
    Console captures its own stdout reference at construction time, so
    contextlib.redirect_stdout (which patches the sys.stdout global) does
    NOT capture its output the way it does plain print() - tests must pass
    a Console(file=io.StringIO()) here instead. This is the standard way
    to test a rich-based CLI, not a workaround.
    """
    c = out or console
    if result.get("skipped"):
        c.print(Panel(f"[yellow]Skipped:[/] {result.get('skip_reason')}", border_style="yellow"))
        return

    findings = result.get("findings", [])
    if not findings:
        c.print(Panel("[bold green]No confirmed findings.[/]", border_style="green"))
    else:
        table = Table(title=f"{len(findings)} finding(s)", box=box.ROUNDED, show_lines=True,
                       title_style="bold")
        table.add_column("File:Line", style="bold")
        table.add_column("Category/Severity")
        table.add_column("Title")
        table.add_column("Verdict")
        table.add_column("Score", justify="right")
        table.add_column("Votes", justify="right")
        for f in findings:
            sev_style = SEVERITY_STYLE.get(f["severity"], "white")
            verdict_style = VERDICT_STYLE.get(f["verdict"], "white")
            table.add_row(
                f"{f['file']}:{f['line']}",
                f"[{sev_style}]{f['category']}/{f['severity']}[/]",
                f["title"],
                f"[{verdict_style}]{f['verdict']}[/]",
                f"{f['score']:.2f}",
                f"{f['vote_count']}/{f['passes_surviving']}",
            )
        c.print(table)

    post = result.get("post_result", {})
    if post.get("reason") == "dry_run":
        c.print(f"[dim]Dry run[/] - would post {post['count']} comment(s). "
                f"Re-run with [bold]--post[/] to actually post to GitHub.")
    elif post.get("posted"):
        c.print(f"[bold green]Posted[/] {post['count']} comment(s) to the PR for real.")
    else:
        c.print(f"Nothing posted ([dim]{post.get('reason', 'unknown')}[/]).")

    if result.get("degradations"):
        c.print(f"[yellow]{len(result['degradations'])} degradation(s)[/] - see findings.json for detail.")


def check_env_vars() -> list[str]:
    return [name for name, val in [
        ("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN")),
        ("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY")),
        ("GROQ_API_KEY", os.environ.get("GROQ_API_KEY")),
    ] if not val]


def run_one_review(owner: str, repo: str, pr_number: int, github_token: str, post: bool) -> dict | None:
    """Resolve -> clone -> review for exactly one PR. Returns the result
    dict, or None if a handled error occurred (already printed). Shared by
    both the --pr single-shot flag and the interactive session below - one
    implementation, two entry points, not a duplicated copy of the logic.
    """
    try:
        with console.status(f"[bold cyan]Resolving[/] {owner}/{repo}#{pr_number}...", spinner="dots"):
            info = asyncio.run(resolve_pr_info(owner, repo, pr_number, github_token))
    except httpx.HTTPStatusError as exc:
        err_console.print(f"[bold red]Error:[/] could not resolve PR ({exc.response.status_code}). "
                           f"Check the URL and that your token can read this repo.")
        return None
    except httpx.RequestError as exc:
        err_console.print(f"[bold red]Error:[/] network request to GitHub failed: {exc}")
        return None
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/] {exc}")
        return None
    console.print(f"[bold green][OK][/] Resolved [bold]{info['head_ref']}[/] @ [dim]{info['head_sha'][:7]}[/]")

    with TemporaryDirectory(prefix="oss-bugbot-cli-") as tmp:
        checkout = Path(tmp)
        try:
            with console.status(f"[bold cyan]Cloning[/] {info['head_ref']} "
                                 f"(depth 1, read-only - never executed)...", spinner="dots"):
                clone_pr_branch(info["head_clone_url"], info["head_ref"], checkout)
        except subprocess.CalledProcessError as exc:
            err_console.print(f"[bold red]Error:[/] clone failed:\n{exc.stderr}")
            return None
        except subprocess.TimeoutExpired:
            err_console.print(f"[bold red]Error:[/] clone exceeded {CLONE_TIMEOUT_SECONDS}s. This repo/branch is "
                               f"a poor fit for local CLI mode - use the Actions runtime instead.")
            return None
        except FileNotFoundError:
            err_console.print("[bold red]Error:[/] git executable not found on PATH. Install git and try again.")
            return None
        console.print("[bold green][OK][/] Cloned")

        # Provider-agnostic pipeline; the CLI still just uses the zero-config
        # Gemini+Groq default (env vars already validated by check_env_vars) -
        # bring-your-own-arbitrary-provider is web-UI-only, see README.
        a1_config, a2_config, a3_primary_config, a3_fallback_config = default_provider_configs_from_env()
        with console.status("[bold cyan]Running review[/] - 4x Gemini, vote, Groq adversarial validate "
                             "[dim](typically 20-60s)[/]...", spinner="dots"):
            result = asyncio.run(run_review(
                owner, repo, pr_number, info["head_sha"], github_token,
                a1_config, a2_config, a3_primary_config, a3_fallback_config, checkout, post=post,
            ))
        console.print("[bold green][OK][/] Review complete\n")

    Path("findings.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def resolve_pr_from_repo(owner: str, repo: str, github_token: str) -> int | None:
    """A bare repo link has no diff to review - there has to be an actual
    open PR. Lists them and picks one (auto-picks if there's exactly one).
    Returns None if there's nothing to review or the lookup failed - both
    cases are already explained to the user, caller just continues the loop.
    """
    try:
        with console.status(f"[bold cyan]Looking up open pull requests[/] on {owner}/{repo}...", spinner="dots"):
            prs = asyncio.run(list_open_prs(owner, repo, github_token))
    except httpx.HTTPStatusError as exc:
        err_console.print(f"[bold red]Error:[/] could not look up {owner}/{repo} ({exc.response.status_code}). "
                           f"Check the repo name is right and it's public.\n")
        return None
    except httpx.RequestError as exc:
        err_console.print(f"[bold red]Error:[/] network request to GitHub failed: {exc}\n")
        return None

    if not prs:
        console.print(f"[yellow]{owner}/{repo} doesn't have any open pull requests right now[/] - "
                       f"there's nothing to review. Try a repo with an active PR, or paste a "
                       f"direct link to a specific pull request instead.\n")
        return None

    if len(prs) == 1:
        console.print(f"[dim]One open PR - reviewing #{prs[0]['number']}: {prs[0]['title']}[/]")
        return prs[0]["number"]

    table = Table(title=f"{len(prs)} open pull requests on {owner}/{repo}", box=box.ROUNDED)
    table.add_column("#", style="bold")
    table.add_column("Title")
    for pr in prs:
        table.add_row(str(pr["number"]), pr["title"])
    console.print(table)

    try:
        choice = Prompt.ask("Which one? (number)", choices=[str(pr["number"]) for pr in prs], show_choices=False)
    except (EOFError, KeyboardInterrupt):
        return None
    return int(choice)


def run_interactive(github_token: str) -> int:
    """Claude Code / OpenCode style: launch once, paste PR URLs into the
    running session, keep going until you quit. Reuses run_one_review for
    every iteration - the interactive loop is presentation, not new logic.
    """
    print_banner()
    console.print("[dim]Paste a GitHub repo or pull request link. Type 'exit' or press Ctrl+C to quit.[/]\n")

    while True:
        try:
            url = Prompt.ask("[bold cyan]Repo or PR URL[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            return 0

        url = url.strip()
        if url.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/]")
            return 0
        if not url:
            continue

        try:
            owner, repo, pr_number = parse_github_url(url)
        except ValueError as exc:
            err_console.print(f"[bold red]Error:[/] {exc}\n")
            continue

        if pr_number is None:
            pr_number = resolve_pr_from_repo(owner, repo, github_token)
            if pr_number is None:
                continue  # already explained why - no open PRs, or lookup failed

        try:
            post = Confirm.ask("Actually post the review to GitHub?", default=False)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            return 0

        result = run_one_review(owner, repo, pr_number, github_token, post)
        if result is not None:
            print_findings(result, posted_for_real=post)
        console.print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run oss-bugbot locally against any public GitHub PR.")
    parser.add_argument("--pr", required=False, default=None,
                         help="Full PR URL. If omitted, starts an interactive session instead of exiting after one PR.")
    parser.add_argument("--post", action="store_true",
                         help="Actually post the review to GitHub. Default is dry-run (print only, no write call). "
                              "Only applies in --pr mode - interactive mode asks each time.")
    args = parser.parse_args()

    missing = check_env_vars()
    if missing:
        print_banner()
        err_console.print(f"[bold red]Error:[/] missing required environment variable(s): {', '.join(missing)}")
        err_console.print("[dim]Set them in your current shell before running this (see README, Local CLI mode).[/]")
        return 1

    github_token = os.environ["GITHUB_TOKEN"]

    if args.pr is None:
        return run_interactive(github_token)

    print_banner()
    try:
        owner, repo, pr_number = parse_pr_url(args.pr)
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/] {exc}")
        return 1

    result = run_one_review(owner, repo, pr_number, github_token, args.post)
    if result is None:
        return 1
    print_findings(result, posted_for_real=args.post)
    return 0


if __name__ == "__main__":
    sys.exit(main())
