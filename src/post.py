"""post.py: dedupe against existing comments, sanitize (S7), batch every
confirmed finding into ONE Review API call. 422 (a comment anchors to a line
GitHub doesn't consider commentable - can legitimately differ from our own
G1 hunk parse) falls back to folding everything into the review body rather
than losing the whole review. Empty review after dedupe makes no API call
at all - see plan "Posting: one review, not N comments".
"""

from __future__ import annotations

import hashlib
import re

import httpx

GITHUB_API = "https://api.github.com"
MAX_COMMENT_LENGTH = 3000

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MARKER_RE = re.compile(r"<!-- bugbot:v1:([0-9a-f]{12}) -->")


def marker_hash(file: str, category: str, title: str) -> str:
    """Deliberately excludes the line number - a bug whose line shifted
    because code was added above it is the same bug. Hashing the line would
    repost it on every push, which is the fastest way to get a bot muted.
    """
    normalized_title = re.sub(r"\s+", " ", title.strip().lower())
    raw = f"{file}|{category}|{normalized_title}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def marker_comment(hash_: str) -> str:
    return f"<!-- bugbot:v1:{hash_} -->"


def sanitize_comment(text: str, max_length: int = MAX_COMMENT_LENGTH) -> str:
    """S7: strip raw HTML tags (defense in depth - GitHub sanitizes markdown
    itself, but an injected comment_markdown shouldn't rely on that alone),
    break markdown auto-link syntax so injected text can't render a
    clickable link, cap length so a runaway/injected response can't post a
    wall of text.
    """
    text = _HTML_TAG_RE.sub("", text)
    text = text.replace("](", "] (")
    if len(text) > max_length:
        text = text[:max_length].rstrip() + "\n\n*(truncated)*"
    return text


async def fetch_existing_markers(client: httpx.AsyncClient, owner: str, repo: str, pr_number: int, token: str) -> set[str]:
    """Paginated - a long-lived PR can exceed one page, and a truncated
    list silently reposts old findings.
    """
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    markers: set[str] = set()
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    params = {"per_page": 100}
    while url:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        for item in resp.json():
            for m in _MARKER_RE.finditer(item.get("body", "")):
                markers.add(m.group(1))
        url = resp.links.get("next", {}).get("url")
        params = None
    return markers


def build_review_comments(confirmed_findings: list[tuple], existing_markers: set[str]) -> list[dict]:
    """confirmed_findings: list of (Finding, ValidatorOutput) pairs already
    filtered to verdict == "confirmed". Returns the comments to post, each
    annotated with its own _hash for degradation logging by the caller.
    """
    to_post = []
    for finding, verdict in confirmed_findings:
        h = marker_hash(finding.file, finding.category, finding.title)
        if h in existing_markers:
            continue
        body = sanitize_comment(verdict.comment_markdown) + f"\n\n{marker_comment(h)}"
        to_post.append({"path": finding.file, "line": finding.line, "body": body, "_hash": h})
    return to_post


async def post_review(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    pr_number: int,
    token: str,
    head_sha: str,
    to_post: list[dict],
) -> dict:
    """Posts ONE review for everything in to_post. Empty to_post makes no
    API call. 422 retries once with comments folded into the review body.
    Returns a result dict for findings.json - never raises on a 422, only
    on a genuinely unexpected error.
    """
    if not to_post:
        return {"posted": False, "reason": "nothing_new_to_post", "count": 0, "fallback": False}

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    comments = [{"path": c["path"], "line": c["line"], "body": c["body"]} for c in to_post]
    payload = {"commit_id": head_sha, "event": "COMMENT", "comments": comments}

    resp = await client.post(url, headers=headers, json=payload)

    if resp.status_code == 422:
        body = "\n\n---\n\n".join(c["body"] for c in to_post)
        fallback_payload = {"commit_id": head_sha, "event": "COMMENT", "body": body}
        resp2 = await client.post(url, headers=headers, json=fallback_payload)
        resp2.raise_for_status()
        return {"posted": True, "fallback": True, "count": len(to_post), "reason": "422_on_inline_comments"}

    resp.raise_for_status()
    return {"posted": True, "fallback": False, "count": len(to_post)}
