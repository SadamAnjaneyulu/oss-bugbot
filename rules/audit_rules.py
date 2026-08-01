"""S2 enforcement: rejects any vendored rule containing pattern-where-python
or other Python-execution keys, converting "we pin rules" from a promise into
a build failure. Run in CI before every run; also runnable standalone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

DENIED_KEYS = {"pattern-where-python", "r2c-internal-project-depends-on-python"}
RULES_DIR = Path(__file__).resolve().parent


def _walk(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _walk(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def audit_file(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    violations = []
    for key in _walk(data):
        if isinstance(key, str) and key in DENIED_KEYS:
            violations.append(f"{path}: denied key '{key}'")
    return violations


def main() -> int:
    violations = []
    for f in sorted(RULES_DIR.glob("*.yml")):
        violations.extend(audit_file(f))

    if violations:
        print("Semgrep rule audit FAILED:")
        for v in violations:
            print(f"  {v}")
        return 1

    print(f"Semgrep rule audit OK: {len(list(RULES_DIR.glob('*.yml')))} files, no denied keys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
