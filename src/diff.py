"""PR diff acquisition: metadata-first sizing gate, then diff fetch, then hunk parse.

Order matters. GitHub returns 406 for Accept: application/vnd.github.diff once
a diff exceeds its size cap. Fetching the diff before gating guarantees every
oversize PR -- exactly the ones the gate exists to reject -- hits that 406 on
the expensive path. So: /files metadata first, gate on it, only then request
the diff. See plan "Diff acquisition: metadata first, then diff".
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from unidiff import PatchSet

GITHUB_API = "https://api.github.com"

MAX_LINES = 500
MAX_FILES = 30

BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".pdf", ".zip", ".gz",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".wasm", ".so", ".dylib",
    ".dll", ".exe", ".pyc", ".class", ".jar",
}
GENERATED_SUFFIXES = (".min.js", ".min.css", "-lock.json", ".lock")
GENERATED_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "poetry.lock"}
VENDORED_DIR_MARKERS = ("vendor/", "node_modules/", "dist/", "build/", "third_party/")


@dataclass
class FileMeta:
    filename: str
    status: str  # added, removed, modified, renamed
    additions: int
    deletions: int
    changes: int
    patch: str | None = None  # absent for binary files or diffs GitHub omits


@dataclass
class SizeGateResult:
    ok: bool
    reason: str | None
    total_lines: int
    total_files: int


def is_skippable_file(filename: str) -> bool:
    lower = filename.lower()
    if any(lower.endswith(ext) for ext in BINARY_EXT):
        return True
    if any(lower.endswith(suf) for suf in GENERATED_SUFFIXES):
        return True
    if filename.rsplit("/", 1)[-1] in GENERATED_NAMES:
        return True
    if any(marker in lower for marker in VENDORED_DIR_MARKERS):
        return True
    return False


def size_gate(files: list[FileMeta], max_lines: int = MAX_LINES, max_files: int = MAX_FILES) -> SizeGateResult:
    reviewable = [f for f in files if not is_skippable_file(f.filename)]
    total_lines = sum(f.changes for f in reviewable)
    total_files = len(reviewable)

    if total_files > max_files:
        return SizeGateResult(False, f"{total_files} reviewable files exceeds limit of {max_files}", total_lines, total_files)
    if total_lines > max_lines:
        return SizeGateResult(False, f"{total_lines} changed lines exceeds limit of {max_lines}", total_lines, total_files)
    return SizeGateResult(True, None, total_lines, total_files)


def deleted_filenames(files: list[FileMeta]) -> set[str]:
    return {f.filename for f in files if f.status == "removed"}


async def fetch_pr_files(client: httpx.AsyncClient, owner: str, repo: str, pr_number: int, token: str) -> list[FileMeta]:
    """GET /pulls/{n}/files, paginated. Metadata only - no diff text requested here."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    files: list[FileMeta] = []
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/files"
    params = {"per_page": 100}
    while url:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        for item in resp.json():
            files.append(FileMeta(
                filename=item["filename"],
                status=item["status"],
                additions=item["additions"],
                deletions=item["deletions"],
                changes=item["changes"],
                patch=item.get("patch"),
            ))
        url = resp.links.get("next", {}).get("url")
        params = None  # pagination params are baked into the "next" url
    return files


async def fetch_diff(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    pr_number: int,
    token: str,
    files_fallback: list[FileMeta] | None = None,
) -> str:
    """GET /pulls/{n} with the diff media type. Only called after size_gate passes.

    Residual 406 (a PR under the line limit containing one enormous single
    file) falls back to joining the per-file `patch` fields already fetched
    in fetch_pr_files -- never crashes, degrades to whatever GitHub gave us.
    """
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.diff"}
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
    resp = await client.get(url, headers=headers)

    if resp.status_code == 406:
        if not files_fallback:
            return ""
        parts = [f.patch for f in files_fallback if f.patch]
        return "\n".join(parts)

    resp.raise_for_status()
    return resp.text


def parse_changed_lines(diff_text: str) -> dict[str, set[int]]:
    """Returns {filename: {added line numbers in the post-image}}.

    Used by G1 to assert a finding's line actually falls inside a changed
    hunk. Context lines are deliberately excluded - a finding must point at
    a line the PR actually added, not a line it merely displays around.
    """
    if not diff_text.strip():
        return {}
    patch = PatchSet(diff_text)
    result: dict[str, set[int]] = {}
    for pfile in patch:
        if pfile.is_removed_file:
            continue
        filename = pfile.path
        lines: set[int] = set()
        for hunk in pfile:
            for line in hunk:
                if line.is_added and line.target_line_no is not None:
                    lines.add(line.target_line_no)
        result[filename] = lines
    return result
