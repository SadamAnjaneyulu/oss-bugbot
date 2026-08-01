"""A1's context tools: read_file, list_dir, find_references, find_definition.

Every tool routes paths through sandbox.resolve_within_root and never raises -
callers (the async tool loop in passes.py) get a structured {"ok": ...} dict
back regardless of what went wrong. See plan section "Tool error contract".

find_definition uses a definition-shaped regex, not tree-sitter. Deliberately:
this is the fallback the plan specs for when tree-sitter grammars fail to
load in the runner, and it works today with zero new dependencies. Upgrade
to tree-sitter only if the Phase 3 tools-ablation shows regex-based recall
is the bottleneck - not before, that's a guess this code shouldn't make.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from sandbox import SandboxViolation, is_binary, resolve_within_root

MAX_RESULTS = 50
MAX_READ_BYTES = 20_000  # a single read_file call should never pull an entire huge file

SKIP_DIR_NAMES = {
    ".git", "node_modules", "vendor", "dist", "build", "__pycache__",
    ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache",
}

_DEF_PATTERNS = (
    re.compile(r"^\s*(?:async\s+)?def\s+{name}\s*\("),
    re.compile(r"^\s*class\s+{name}\s*[:\(]"),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+{name}\s*\("),
    re.compile(r"^\s*(?:export\s+)?class\s+{name}\s*[{{<]"),
    re.compile(r"^\s*(?:export\s+)?const\s+{name}\s*="),
)

_DEF_LANG_EXT = (".py", ".ts", ".tsx")


def _iter_source_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for fname in filenames:
            yield Path(dirpath) / fname


def read_file(root: Path, path: str, start: int = 1, end: int | None = None) -> dict:
    try:
        resolved = resolve_within_root(root, path)
    except SandboxViolation as exc:
        return {"ok": False, "error": "path_denied", "hint": str(exc)}

    if not resolved.is_file():
        return {"ok": False, "error": "not_found", "hint": f"no such file: {path}"}
    if is_binary(resolved):
        return {"ok": False, "error": "binary_file", "hint": "Not reviewable."}

    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"ok": False, "error": "read_failed", "hint": str(exc)}

    lines = text.splitlines()
    end = end or len(lines)
    start = max(1, start)
    end = min(len(lines), end)
    snippet = "\n".join(lines[start - 1:end])
    if len(snippet.encode("utf-8")) > MAX_READ_BYTES:
        snippet = snippet.encode("utf-8")[:MAX_READ_BYTES].decode("utf-8", errors="ignore")
        return {"ok": True, "truncated": True, "start": start, "end": end, "content": snippet}
    return {"ok": True, "truncated": False, "start": start, "end": end, "content": snippet}


def list_dir(root: Path, path: str = ".") -> dict:
    try:
        resolved = resolve_within_root(root, path)
    except SandboxViolation as exc:
        return {"ok": False, "error": "path_denied", "hint": str(exc)}

    if not resolved.is_dir():
        return {"ok": False, "error": "not_a_directory", "hint": path}

    try:
        entries = sorted(
            e.name + ("/" if e.is_dir() else "")
            for e in os.scandir(resolved)
            if e.name not in SKIP_DIR_NAMES
        )
    except OSError as exc:
        return {"ok": False, "error": "read_failed", "hint": str(exc)}

    return {"ok": True, "entries": entries}


def find_references(root: Path, symbol: str) -> dict:
    if not symbol or not symbol.strip():
        return {"ok": False, "error": "empty_symbol", "hint": "symbol must be non-empty"}

    pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
    results = []
    total = 0
    for f in _iter_source_files(root):
        if is_binary(f):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(f.relative_to(root))
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                total += 1
                if len(results) < MAX_RESULTS:
                    results.append({"file": rel, "line": lineno, "text": line.strip()[:200]})

    if total == 0:
        return {"ok": False, "error": "no_references", "hint": f"'{symbol}' not found in checkout"}

    return {
        "ok": True,
        "truncated": total > MAX_RESULTS,
        "shown": len(results),
        "total": total,
        "results": results,
    }


def find_definition(root: Path, symbol: str) -> dict:
    if not symbol or not symbol.strip():
        return {"ok": False, "error": "empty_symbol", "hint": "symbol must be non-empty"}

    patterns = [p.pattern.format(name=re.escape(symbol)) for p in _DEF_PATTERNS]
    compiled = [re.compile(p) for p in patterns]

    hits = []
    for f in _iter_source_files(root):
        if f.suffix not in _DEF_LANG_EXT:
            continue
        if is_binary(f):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(f.relative_to(root))
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(c.search(line) for c in compiled):
                hits.append({"file": rel, "line": lineno, "text": line.strip()[:200]})
                if len(hits) >= MAX_RESULTS:
                    break
        if len(hits) >= MAX_RESULTS:
            break

    if not hits:
        return {
            "ok": False,
            "error": "symbol_not_found",
            "hint": f"No definition found for '{symbol}' in Python/TS sources. Try find_references.",
        }

    return {"ok": True, "truncated": len(hits) >= MAX_RESULTS, "results": hits}
