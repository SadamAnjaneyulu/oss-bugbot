"""Path sandbox for A1's file tools (S5).

The single control standing between a fork PR's injected text and a tool call
that reads `.git/config`, `.env`, or `/proc/self/environ` out of the process
that holds GEMINI_API_KEY. Every tool in tools.py must route paths through
resolve_within_root before touching disk.
"""

from __future__ import annotations

import os
from pathlib import Path

DENIED_NAMES = {".git", ".env"}
DENIED_PREFIXES = ("/proc", "/sys")


class SandboxViolation(Exception):
    pass


def resolve_within_root(root: Path, requested: str) -> Path:
    """Resolve `requested` to an absolute, symlink-resolved path guaranteed to
    live under `root`. Raises SandboxViolation on traversal, symlink escape,
    an absolute path that overrides root, or a denied path segment.
    """
    root = root.resolve()
    # Path.__truediv__ discards the left side entirely if `requested` is
    # itself absolute (e.g. "/proc/self/environ") - the commonpath check
    # below still catches this, but don't rely on that alone reading this
    # function; the join is not a sandbox by itself.
    candidate = (root / requested).resolve()

    try:
        common = os.path.commonpath([str(root), str(candidate)])
    except ValueError as exc:
        # Different drives on Windows, or no common path at all.
        raise SandboxViolation(f"path escapes sandbox root: {requested}") from exc

    if common != str(root):
        raise SandboxViolation(f"path escapes sandbox root: {requested}")

    for part in candidate.relative_to(root).parts:
        if part in DENIED_NAMES:
            raise SandboxViolation(f"denied path segment '{part}': {requested}")

    candidate_str = str(candidate)
    for prefix in DENIED_PREFIXES:
        if candidate_str.startswith(prefix):
            raise SandboxViolation(f"denied path prefix '{prefix}': {requested}")

    return candidate


def is_binary(path: Path, sniff_bytes: int = 8192) -> bool:
    """Null-byte sniff of the first `sniff_bytes`. Crude on purpose - see
    plan's 'Binary / generated' gate. Unreadable counts as binary: never
    raise, treat as not reviewable.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(sniff_bytes)
    except OSError:
        return True
    return b"\x00" in chunk
