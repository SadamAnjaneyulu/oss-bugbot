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

from main import run_review

PR_URL_RE = re.compile(r"github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)")
CLONE_TIMEOUT_SECONDS = 120


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


def print_findings(result: dict, posted_for_real: bool) -> None:
    if result.get("skipped"):
        print(f"Skipped: {result.get('skip_reason')}")
        return

    findings = result.get("findings", [])
    if not findings:
        print("No confirmed findings.")
    else:
        print(f"{len(findings)} finding(s):\n")
        for f in findings:
            print(f"  {f['file']}:{f['line']}  [{f['category']}/{f['severity']}]  {f['title']}")
            print(f"    verdict={f['verdict']}  score={f['score']:.2f}  votes={f['vote_count']}/{f['passes_surviving']}")
        print()

    post = result.get("post_result", {})
    if post.get("reason") == "dry_run":
        print(f"Dry run - would post {post['count']} comment(s). Re-run with --post to actually post to GitHub.")
    elif post.get("posted"):
        print(f"Posted {post['count']} comment(s) to the PR for real.")
    else:
        print(f"Nothing posted ({post.get('reason', 'unknown')}).")

    if result.get("degradations"):
        print(f"\n{len(result['degradations'])} degradation(s) - see findings.json for detail.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run oss-bugbot locally against any public GitHub PR.")
    parser.add_argument("--pr", required=True, help="Full PR URL, e.g. https://github.com/owner/repo/pull/123")
    parser.add_argument("--post", action="store_true",
                         help="Actually post the review to GitHub. Default is dry-run (print only, no write call).")
    args = parser.parse_args()

    try:
        owner, repo, pr_number = parse_pr_url(args.pr)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    github_token = os.environ.get("GITHUB_TOKEN")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    missing = [name for name, val in
               [("GITHUB_TOKEN", github_token), ("GEMINI_API_KEY", gemini_api_key), ("GROQ_API_KEY", groq_api_key)]
               if not val]
    if missing:
        print(f"Error: missing required environment variable(s): {', '.join(missing)}", file=sys.stderr)
        print("Set them in your current shell before running this (see README, Local CLI mode).", file=sys.stderr)
        return 1

    print(f"Resolving {owner}/{repo}#{pr_number}...")
    try:
        info = asyncio.run(resolve_pr_info(owner, repo, pr_number, github_token))
    except httpx.HTTPStatusError as exc:
        print(f"Error: could not resolve PR ({exc.response.status_code}). "
              f"Check the URL and that your token can read this repo.", file=sys.stderr)
        return 1
    except httpx.RequestError as exc:
        print(f"Error: network request to GitHub failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    with TemporaryDirectory(prefix="oss-bugbot-cli-") as tmp:
        checkout = Path(tmp)
        print(f"Cloning {info['head_ref']} (depth 1, read-only - never executed)...")
        try:
            clone_pr_branch(info["head_clone_url"], info["head_ref"], checkout)
        except subprocess.CalledProcessError as exc:
            print(f"Error: clone failed:\n{exc.stderr}", file=sys.stderr)
            return 1
        except subprocess.TimeoutExpired:
            print(f"Error: clone exceeded {CLONE_TIMEOUT_SECONDS}s. This repo/branch is a poor fit "
                  f"for local CLI mode - use the Actions runtime instead.", file=sys.stderr)
            return 1
        except FileNotFoundError:
            print("Error: git executable not found on PATH. Install git and try again.", file=sys.stderr)
            return 1

        print("Running review (calls Gemini and Groq - typically 20-60s)...")
        result = asyncio.run(run_review(
            owner, repo, pr_number, info["head_sha"], github_token,
            gemini_api_key, groq_api_key, checkout, post=args.post,
        ))

    print_findings(result, posted_for_real=args.post)
    Path("findings.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
