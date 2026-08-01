"""Semgrep advisory pass (S2, S3-notation "STATIC" step). Scoped to changed
files only, vendored+pinned rules only. Never gates A1's attention - every
pass sees the full diff regardless of what Semgrep found; a hit only raises
a finding's corroboration signal. Never raises: a Semgrep failure degrades
to "no corroboration available," not a pipeline crash.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

RULES_DIR = Path(__file__).resolve().parent.parent / "rules"
CORROBORATION_LINE_TOLERANCE = 2


def run_semgrep(root: Path, changed_files: list[str], timeout: int = 60) -> dict[str, list[dict]]:
    """Returns {relative_filename: [{"check_id", "line", "message"}]}."""
    if not changed_files:
        return {}

    rule_files = sorted(RULES_DIR.glob("*.yml"))
    if not rule_files:
        return {}

    cmd = ["semgrep", "--json", "--quiet", "--no-git-ignore"]
    for rf in rule_files:
        cmd += ["--config", str(rf)]
    cmd += [str(root / f) for f in changed_files]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(root))
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}

    if proc.returncode not in (0, 1):  # 1 == findings present, still a clean run
        return {}

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}

    hits: dict[str, list[dict]] = {}
    for result in data.get("results", []):
        path = result.get("path", "")
        try:
            rel = str(Path(path).resolve().relative_to(root.resolve()))
        except ValueError:
            rel = path
        rel = rel.replace("\\", "/")
        hits.setdefault(rel, []).append({
            "check_id": result.get("check_id", ""),
            "line": result.get("start", {}).get("line", 0),
            "message": result.get("extra", {}).get("message", ""),
        })
    return hits


def is_corroborated(hits: dict[str, list[dict]], file: str, line: int, tolerance: int = CORROBORATION_LINE_TOLERANCE) -> bool:
    """Within `tolerance` lines, not exact - the two tools often anchor to
    slightly different lines for the same underlying issue.
    """
    return any(abs(h["line"] - line) <= tolerance for h in hits.get(file, []))
