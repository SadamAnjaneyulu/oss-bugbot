"""Entrypoint the GitHub Action invokes. Wires every already-tested module
into one pipeline run: size-gate -> diff -> static -> A1x4 -> A2 -> A3 ->
post -> findings.json. See docs/prompts.md (once written) for the published
system prompts, and the plan's "Agent + gate flow" diagram for the shape
this mirrors.

Note on naming: "size-gate" above is diff.py's size_gate (the only check
that runs before the diff is fetched at all). gates.py's G1/G2/G3 are a
different thing - they don't run as a standalone stage here, they're
interleaved inside passes.py (G1), vote.py (G2), and validate.py (G3).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from google import genai
from groq import AsyncGroq

from diff import (
    MAX_FILES, MAX_LINES, deleted_filenames, fetch_diff, fetch_pr_files,
    is_skippable_file, parse_changed_lines, shuffle_hunks, size_gate,
)
from llm import GEMINI_FLASH, GEMINI_FLASH_LITE, GROQ_PRIMARY, MODEL_RESOLUTION_DATE
from passes import NUM_PASSES, run_pass
from post import build_review_comments, fetch_existing_markers, post_review
from static import is_corroborated, run_semgrep
from vote import DEFAULT_VOTE_THRESHOLD, run_aggregation
from validate import run_all_validators
from stats import wilson_lower_bound

SCHEMA_VERSION = "1.0"
VALIDATOR_WEIGHT = {"confirmed": 1.0, "uncertain": 0.4, "false_positive": 0.0}

# Soft pre-emptive caps - see llm.py RATE_LIMITS docstring: neither provider
# publishes a permanently fixed free-tier table anymore, these are a
# starting point, not a guarantee. 429s are the authoritative signal.
GEMINI_FLASH_LITE_SEM = asyncio.Semaphore(int(os.environ.get("GEMINI_FLASH_LITE_RPM", 15)))
GEMINI_FLASH_SEM = asyncio.Semaphore(int(os.environ.get("GEMINI_FLASH_RPM", 10)))
GROQ_SEM = asyncio.Semaphore(1)  # plan: TPM ceiling trips under any real concurrency


def compute_score(vote_count: int, passes_surviving: int, verdict: str) -> float:
    if passes_surviving == 0:
        return 0.0
    return wilson_lower_bound(vote_count, passes_surviving) * VALIDATOR_WEIGHT.get(verdict, 0.0)


async def run_review(
    owner: str, repo: str, pr_number: int, head_sha: str, github_token: str,
    gemini_api_key: str, groq_api_key: str, root: Path, post: bool = True,
) -> dict:
    """Returns the findings.json-shaped dict. Never raises for expected
    conditions (oversize PR, refusals, gate exhaustion) - those are all
    degradations logged in the returned dict, per "Degrade, never crash".

    post=True is the library default (matches review.yml's existing
    production behavior unchanged). cli.py, the local ad-hoc wrapper that
    can point at any public PR a token can reach, defaults ITS flag the
    other way - dry-run unless --post is passed explicitly. A tool that can
    be pointed at a stranger's real PR should not post there by default.
    """
    degradations: list[dict] = []
    skipped_files: list[str] = []

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        files = await fetch_pr_files(http_client, owner, repo, pr_number, github_token)
        gate = size_gate(files, MAX_LINES, MAX_FILES)

        if not gate.ok:
            return {
                "schema_version": SCHEMA_VERSION, "pr": pr_number,
                "skipped": True, "skip_reason": gate.reason,
                "findings": [], "degradations": [], "skipped_files": [],
            }

        deleted = deleted_filenames(files)
        skippable = {f.filename for f in files if is_skippable_file(f.filename)}
        skipped_files.extend(sorted(skippable))

        diff_text = await fetch_diff(http_client, owner, repo, pr_number, github_token, files_fallback=files)
        if not diff_text.strip():
            return {
                "schema_version": SCHEMA_VERSION, "pr": pr_number,
                "skipped": True, "skip_reason": "empty_diff",
                "findings": [], "degradations": [], "skipped_files": skipped_files,
            }

        changed_lines_all = parse_changed_lines(diff_text)
        changed_lines = {f: lines for f, lines in changed_lines_all.items() if f not in skippable}
        reviewable_files = [f for f in changed_lines if f not in deleted]

        semgrep_hits = run_semgrep(root, reviewable_files)

    gemini_client = genai.Client(api_key=gemini_api_key)
    groq_client = AsyncGroq(api_key=groq_api_key)

    # --- A1: NUM_PASSES concurrent passes, each a differently-shuffled diff
    pass_diffs = [shuffle_hunks(diff_text, seed=i) for i in range(NUM_PASSES)]
    pass_results = await asyncio.gather(*[
        run_pass(gemini_client, GEMINI_FLASH_LITE, GEMINI_FLASH_LITE_SEM, root, pass_diffs[i], f"p{i + 1}", changed_lines, deleted)
        for i in range(NUM_PASSES)
    ])

    findings_by_pass = {}
    for i, result in enumerate(pass_results):
        pass_id = f"p{i + 1}"
        if result is None:
            degradations.append({"node": "a1", "pass_id": pass_id, "action": "dropped"})
            continue
        findings_by_pass[pass_id] = result.findings

    # Deterministic corroboration cross-check - never trust the model's own
    # self-reported semgrep_corroborated flag, verify it against real hits.
    for findings in findings_by_pass.values():
        for f in findings:
            f.semgrep_corroborated = is_corroborated(semgrep_hits, f.file, f.line)

    passes_surviving = len(findings_by_pass)

    # --- A2: cluster + vote
    threshold = int(os.environ.get("VOTE_THRESHOLD", DEFAULT_VOTE_THRESHOLD))
    clusters, used_deterministic_clustering = await run_aggregation(
        gemini_client, GEMINI_FLASH, GEMINI_FLASH_SEM, findings_by_pass, threshold=threshold,
    )
    if used_deterministic_clustering:
        degradations.append({"node": "a2", "action": "deterministic_clustering_fallback"})

    # --- A3: adversarial validation, fanned out
    verdicts = await run_all_validators(
        groq_client, gemini_client, GROQ_PRIMARY, GEMINI_FLASH, GROQ_SEM, GEMINI_FLASH_SEM, clusters,
    )
    for v in verdicts:
        if v.validator_family != "llama":
            degradations.append({"node": "a3", "cluster_id": v.cluster_id, "action": "same_family_fallback_used"})
        if v.refutation == "validator exhausted retries or was refused by both providers":
            degradations.append({"node": "a3", "cluster_id": v.cluster_id, "action": "marked_uncertain"})

    findings_out = []
    confirmed_pairs = []
    for cluster, verdict in zip(clusters, verdicts):
        score = compute_score(cluster.vote_count, passes_surviving, verdict.verdict)
        findings_out.append({
            "cluster_id": cluster.cluster_id, "file": cluster.merged.file, "line": cluster.merged.line,
            "category": cluster.merged.category, "severity": cluster.merged.severity,
            "title": cluster.merged.title, "vote_count": cluster.vote_count,
            "passes_surviving": passes_surviving, "verdict": verdict.verdict,
            "validator_family": verdict.validator_family, "score": score,
            "semgrep_corroborated": cluster.merged.semgrep_corroborated,
        })
        if verdict.verdict == "confirmed":
            confirmed_pairs.append((cluster.merged, verdict))

    # --- Post (dry-run reads existing comments for accurate dedupe-aware
    # preview counts, but never calls the write endpoint - a tool that can
    # be pointed at any public PR should not post there without --post)
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        existing_markers = await fetch_existing_markers(http_client, owner, repo, pr_number, github_token)
        to_post = build_review_comments(confirmed_pairs, existing_markers)
        if post:
            post_result = await post_review(http_client, owner, repo, pr_number, github_token, head_sha, to_post)
        else:
            post_result = {"posted": False, "reason": "dry_run", "count": len(to_post), "fallback": False}

    return {
        "schema_version": SCHEMA_VERSION,
        "pr": pr_number,
        "models": {"a1": GEMINI_FLASH_LITE, "a2": GEMINI_FLASH, "a3_primary": GROQ_PRIMARY, "a3_fallback": GEMINI_FLASH},
        "model_resolution_date": MODEL_RESOLUTION_DATE,
        "vote_threshold": threshold,
        "passes_surviving": passes_surviving,
        "findings": findings_out,
        "post_result": post_result,
        "skipped_files": skipped_files,
        "degradations": degradations,
    }


def main() -> int:
    owner_repo = os.environ["GITHUB_REPOSITORY"]  # "owner/repo", set by Actions
    owner, repo = owner_repo.split("/", 1)
    pr_number = int(os.environ["PR_NUMBER"])
    head_sha = os.environ["HEAD_SHA"]  # workflow passes github.event.pull_request.head.sha
    github_token = os.environ["GITHUB_TOKEN"]
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    groq_api_key = os.environ["GROQ_API_KEY"]
    # PR_CHECKOUT_PATH, not GITHUB_WORKSPACE: the workflow checks out this
    # bugbot's own (trusted, base-branch) code at GITHUB_WORKSPACE and the
    # PR's (untrusted) content into a separate directory. root must point at
    # the PR content - sandbox.py's tools read from here, nothing here is
    # ever executed or installed. See review.yml.
    root = Path(os.environ["PR_CHECKOUT_PATH"])

    result = asyncio.run(run_review(owner, repo, pr_number, head_sha, github_token, gemini_api_key, groq_api_key, root))

    output_path = Path("findings.json")
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {output_path}: {len(result.get('findings', []))} findings, "
          f"{len(result.get('degradations', []))} degradations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
