"""Mines real merged bug-fix PRs into eval cases with ground truth that
costs zero human annotation.

The trick: a merged PR that fixes a bug has a diff showing buggy->fixed.
Reverse that diff (git diff <head> <base>, SHAs swapped) and you get a
synthetic fixed->buggy diff - i.e. a PR that REINTRODUCES the bug. Feed
that reversed diff to the pipeline as "the PR under review." Ground truth
is exactly the lines the reversed diff adds, because that is exactly
where the original fix - and therefore the original bug - lived.

Verified against a real PR (psf/requests#7425) before writing this file:
confirmed `git diff <head> <base>` genuinely produces the opposite of
`git diff <base> <head>`, not a guess about git's argument order.

No LLM calls happen here. This only talks to the GitHub API and git.
score.py (separate module) is what feeds a case's diff into the pipeline -
via a plain function argument, not by patching main.py's internals. An
earlier version of this file did that (patch("diff.fetch_diff", ...)) and
the mock silently never fired: main.py does `from diff import fetch_diff`,
which binds the name into main's own namespace at import time, so
patching diff.fetch_diff doesn't touch what main.run_review actually
calls. Confirmed empirically, not assumed - see commit history.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

GITHUB_API = "https://api.github.com"

# Deliberately conservative bounds - a case with a huge diff makes a poor
# eval case (the "1 bug" signal drowns in unrelated noise) even though
# main.py's own size gate is looser (500 lines/30 files) for real PRs.
MAX_CASE_LINES = 60
MAX_CASE_FILES = 3

BUGFIX_TITLE_RE = re.compile(r"\b(fix|bug|regression|crash|error|patch)\b", re.IGNORECASE)
TEST_FILE_RE = re.compile(r"(^|/)(test_|tests?/|.*_test\.py$|.*\.test\.[jt]sx?$|.*\.spec\.[jt]sx?$)")

CATEGORY_KEYWORDS = {
    "security": ["inject", "sanitiz", "escape", "xss", "csrf", "auth", "credential",
                 "secret", "password", "vulnerab", "cve", "exploit", "unsafe"],
    "concurrency": ["race", "lock", "deadlock", "thread", "async", "concurren",
                    "atomic", "mutex", "synchroniz"],
    "resource": ["leak", "close()", "resource", "memory", "file handle",
                 "connection pool", "cleanup", "dispose"],
    "api-misuse": ["deprecat", "argument", "parameter", "signature", "type error",
                   "wrong type", "incorrect usage"],
}
DEFAULT_CATEGORY = "logic"


@dataclass
class Candidate:
    owner: str
    repo: str
    pr_number: int
    title: str


@dataclass
class HarvestResult:
    built: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


async def search_bugfix_prs(client: httpx.AsyncClient, owner: str, repo: str, token: str, limit: int = 30) -> list[Candidate]:
    """Merged PRs whose title looks like a bug fix. Search API, not the
    pulls list endpoint - lets GitHub do the title filtering server-side.
    """
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    query = f"repo:{owner}/{repo} is:pr is:merged fix in:title"
    resp = await client.get(
        f"{GITHUB_API}/search/issues",
        headers=headers,
        params={"q": query, "sort": "updated", "order": "desc", "per_page": limit},
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        Candidate(owner=owner, repo=repo, pr_number=item["number"], title=item["title"])
        for item in data.get("items", [])
        if BUGFIX_TITLE_RE.search(item["title"])
    ]


async def fetch_pr_shas(client: httpx.AsyncClient, candidate: Candidate, token: str) -> dict | None:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    resp = await client.get(
        f"{GITHUB_API}/repos/{candidate.owner}/{candidate.repo}/pulls/{candidate.pr_number}",
        headers=headers,
    )
    if resp.status_code != 200:
        return None
    data = resp.json()

    if data.get("changed_files", 0) > MAX_CASE_FILES:
        return None
    if (data.get("additions", 0) + data.get("deletions", 0)) > MAX_CASE_LINES:
        return None
    # GitHub returns head.repo: null once the PR's source fork/branch has
    # been deleted - common for old merged PRs, not exotic. Same defect
    # already fixed in cli.py's resolve_pr_info; missed applying it here
    # until a live run against real historical PRs hit it.
    if data["base"]["repo"] is None or data["head"]["repo"] is None:
        return None
    if data["base"]["repo"]["full_name"] != data["head"]["repo"]["full_name"]:
        return None  # fork PR - base/head SHAs may not share ancestry in a shallow local clone

    return {
        "base_sha": data["base"]["sha"],
        "head_sha": data["head"]["sha"],
        "title": data["title"],
        "body": data.get("body") or "",
    }


def ensure_repo_clone(clone_root: Path, owner: str, repo: str) -> Path:
    """One clone per repo, reused across every PR harvested from it -
    cloning fresh per PR would be needlessly slow for a multi-case harvest.
    """
    dest = clone_root / f"{owner}__{repo}"
    if not dest.exists():
        subprocess.run(
            ["git", "init", "-q", str(dest)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["git", "-C", str(dest), "remote", "add", "origin", f"https://github.com/{owner}/{repo}.git"],
            check=True, capture_output=True, text=True, timeout=30,
        )
    return dest


def fetch_commits(repo_dir: Path, shas: list[str], timeout: int = 60) -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", *shas],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def reversed_diff(repo_dir: Path, base_sha: str, head_sha: str) -> str:
    """SHAs swapped: base<->head reversed gives fixed->buggy instead of
    buggy->fixed. This is the whole mechanism - see module docstring.
    """
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "diff", head_sha, base_sha],
        capture_output=True, text=True, timeout=30,
    )
    return proc.stdout


def classify_category(title: str, body: str, diff_text: str) -> str:
    """Crude keyword heuristic, not a claim of precision - the eval report
    must say so. Category is a stratification axis, not ground truth
    itself; a mislabeled category costs coverage-balance, not correctness.
    """
    haystack = f"{title} {body} {diff_text}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in haystack for kw in keywords):
            return category
    return DEFAULT_CATEGORY


def extract_ground_truth(diff_text: str) -> dict[str, list[int]]:
    """Reuses diff.parse_changed_lines - the added lines in the REVERSED
    diff are exactly where the original bug was reintroduced. Test files
    excluded: a fix that only touched test assertions has no ground truth
    for the reviewer to find.
    """
    from diff import parse_changed_lines  # local import: eval/ isn't on sys.path by default

    changed = parse_changed_lines(diff_text)
    return {
        f: sorted(lines)
        for f, lines in changed.items()
        if lines and not TEST_FILE_RE.search(f)
    }


async def build_case(client: httpx.AsyncClient, candidate: Candidate, token: str, clone_root: Path) -> dict | None:
    pr_info = await fetch_pr_shas(client, candidate, token)
    if pr_info is None:
        return {"_skipped": True, "reason": "oversize_or_metadata_fetch_failed", "pr": candidate.pr_number}

    repo_dir = ensure_repo_clone(clone_root, candidate.owner, candidate.repo)
    if not fetch_commits(repo_dir, [pr_info["base_sha"], pr_info["head_sha"]]):
        return {"_skipped": True, "reason": "commit_fetch_failed", "pr": candidate.pr_number}

    diff_text = reversed_diff(repo_dir, pr_info["base_sha"], pr_info["head_sha"])
    if not diff_text.strip():
        return {"_skipped": True, "reason": "empty_reversed_diff", "pr": candidate.pr_number}

    ground_truth = extract_ground_truth(diff_text)
    if not ground_truth:
        return {"_skipped": True, "reason": "no_ground_truth_after_test_file_exclusion", "pr": candidate.pr_number}

    category = classify_category(pr_info["title"], pr_info["body"], diff_text)

    return {
        "case_id": f"{candidate.owner}-{candidate.repo}-{candidate.pr_number}",
        "source_pr": f"https://github.com/{candidate.owner}/{candidate.repo}/pull/{candidate.pr_number}",
        "source_title": pr_info["title"],
        "category": category,
        "base_sha": pr_info["base_sha"],
        "head_sha": pr_info["head_sha"],
        "diff": diff_text,
        "ground_truth": ground_truth,
    }


async def harvest(repos: list[tuple[str, str]], token: str, per_repo: int, output_dir: Path) -> HarvestResult:
    result = HarvestResult()
    output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="harvest-clones-") as clone_root_str:
        clone_root = Path(clone_root_str)
        async with httpx.AsyncClient(timeout=30.0) as client:
            for owner, repo in repos:
                candidates = await search_bugfix_prs(client, owner, repo, token, limit=per_repo * 3)
                built_for_repo = 0
                for candidate in candidates:
                    if built_for_repo >= per_repo:
                        break
                    case = await build_case(client, candidate, token, clone_root)
                    if case is None:
                        continue
                    if case.get("_skipped"):
                        result.skipped.append({"repo": f"{owner}/{repo}", **case})
                        continue
                    case_path = output_dir / f"{case['case_id']}.json"
                    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
                    result.built.append(case)
                    built_for_repo += 1

    return result
