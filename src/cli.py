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
    python src/cli.py --pr https://github.com/owner/repo/pull/123
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
from rich.table import Table
from rich import box

from main import run_review

PR_URL_RE = re.compile(r"github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)")
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
        c.print(f"[dim]Dry run[/] — would post {post['count']} comment(s). "
                f"Re-run with [bold]--post[/] to actually post to GitHub.")
    elif post.get("posted"):
        c.print(f"[bold green]Posted[/] {post['count']} comment(s) to the PR for real.")
    else:
        c.print(f"Nothing posted ([dim]{post.get('reason', 'unknown')}[/]).")

    if result.get("degradations"):
        c.print(f"[yellow]{len(result['degradations'])} degradation(s)[/] — see findings.json for detail.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run oss-bugbot locally against any public GitHub PR.")
    parser.add_argument("--pr", required=True, help="Full PR URL, e.g. https://github.com/owner/repo/pull/123")
    parser.add_argument("--post", action="store_true",
                         help="Actually post the review to GitHub. Default is dry-run (print only, no write call).")
    args = parser.parse_args()

    print_banner()

    try:
        owner, repo, pr_number = parse_pr_url(args.pr)
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/] {exc}")
        return 1

    github_token = os.environ.get("GITHUB_TOKEN")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    missing = [name for name, val in
               [("GITHUB_TOKEN", github_token), ("GEMINI_API_KEY", gemini_api_key), ("GROQ_API_KEY", groq_api_key)]
               if not val]
    if missing:
        err_console.print(f"[bold red]Error:[/] missing required environment variable(s): {', '.join(missing)}")
        err_console.print("[dim]Set them in your current shell before running this (see README, Local CLI mode).[/]")
        return 1

    try:
        with console.status(f"[bold cyan]Resolving[/] {owner}/{repo}#{pr_number}...", spinner="dots"):
            info = asyncio.run(resolve_pr_info(owner, repo, pr_number, github_token))
    except httpx.HTTPStatusError as exc:
        err_console.print(f"[bold red]Error:[/] could not resolve PR ({exc.response.status_code}). "
                           f"Check the URL and that your token can read this repo.")
        return 1
    except httpx.RequestError as exc:
        err_console.print(f"[bold red]Error:[/] network request to GitHub failed: {exc}")
        return 1
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/] {exc}")
        return 1
    console.print(f"[bold green][OK][/] Resolved [bold]{info['head_ref']}[/] @ [dim]{info['head_sha'][:7]}[/]")

    with TemporaryDirectory(prefix="oss-bugbot-cli-") as tmp:
        checkout = Path(tmp)
        try:
            with console.status(f"[bold cyan]Cloning[/] {info['head_ref']} "
                                 f"(depth 1, read-only — never executed)...", spinner="dots"):
                clone_pr_branch(info["head_clone_url"], info["head_ref"], checkout)
        except subprocess.CalledProcessError as exc:
            err_console.print(f"[bold red]Error:[/] clone failed:\n{exc.stderr}")
            return 1
        except subprocess.TimeoutExpired:
            err_console.print(f"[bold red]Error:[/] clone exceeded {CLONE_TIMEOUT_SECONDS}s. This repo/branch is "
                               f"a poor fit for local CLI mode — use the Actions runtime instead.")
            return 1
        except FileNotFoundError:
            err_console.print("[bold red]Error:[/] git executable not found on PATH. Install git and try again.")
            return 1
        console.print("[bold green][OK][/] Cloned")

        with console.status("[bold cyan]Running review[/] — 4x Gemini, vote, Groq adversarial validate "
                             "[dim](typically 20-60s)[/]...", spinner="dots"):
            result = asyncio.run(run_review(
                owner, repo, pr_number, info["head_sha"], github_token,
                gemini_api_key, groq_api_key, checkout, post=args.post,
            ))
        console.print("[bold green][OK][/] Review complete\n")

    print_findings(result, posted_for_real=args.post)
    Path("findings.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
