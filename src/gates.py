"""G1/G2/G3 - semantic assertions the JSON schema can't express, plus the
retry-same-node wrapper. Schema validation (Pydantic, decode-time) already
guarantees shape; these gates check truth: does this line actually exist in
the diff, does this cluster actually trace back to a real finding.

Retry policy: retry the node that failed, not the one before it. If A2's
output fails G2, A1's findings were valid and still in memory - re-running
A1 would spend 4 calls reproducing input that was never the problem.
"""

from __future__ import annotations

from pydantic import ValidationError

from schemas import AggregatorOutput, Finding, ReviewerOutput, ValidatorOutput

MAX_RETRIES = 2
CLUSTER_TRACE_LINE_TOLERANCE = 3  # matches the deterministic-clustering fallback distance


class GateViolation(Exception):
    pass


class NodeExhausted(Exception):
    """Raised after MAX_RETRIES failed attempts. Callers catch this and apply
    the degrade-never-crash fallback for that node (drop pass / deterministic
    clustering / mark uncertain) - never propagate it as a crash.
    """

    def __init__(self, last_error: Exception):
        self.last_error = last_error
        super().__init__(str(last_error))


def check_g1(output: ReviewerOutput, changed_lines: dict[str, set[int]], deleted_files: set[str]) -> None:
    if len(output.findings) > 20:
        raise GateViolation("more than 20 findings in a single pass")

    for f in output.findings:
        if f.file in deleted_files:
            raise GateViolation(f"finding on deleted/renamed-away file: {f.file}")
        if f.file not in changed_lines:
            raise GateViolation(f"finding references a file not present in the diff: {f.file}")
        if f.line not in changed_lines[f.file]:
            raise GateViolation(f"finding line {f.line} is not inside a changed hunk of {f.file}")
        if not set(f.evidence_lines) <= changed_lines[f.file]:
            raise GateViolation(f"evidence_lines {f.evidence_lines} not a subset of changed lines in {f.file}")


def check_g2(
    output: AggregatorOutput,
    valid_pass_ids: set[str],
    findings_by_pass: dict[str, list[Finding]],
) -> None:
    for cluster in output.clusters:
        unknown = set(cluster.supporting_pass_ids) - valid_pass_ids
        if unknown:
            raise GateViolation(f"cluster {cluster.cluster_id} cites unknown pass_ids: {unknown}")

        if cluster.vote_count != len(cluster.supporting_pass_ids):
            raise GateViolation(
                f"cluster {cluster.cluster_id} vote_count={cluster.vote_count} "
                f"!= len(supporting_pass_ids)={len(cluster.supporting_pass_ids)}"
            )

        candidates: list[Finding] = []
        for pid in cluster.supporting_pass_ids:
            candidates.extend(findings_by_pass.get(pid, []))

        merged = cluster.merged
        traced = any(
            c.file == merged.file
            and c.category == merged.category
            and abs(c.line - merged.line) <= CLUSTER_TRACE_LINE_TOLERANCE
            for c in candidates
        )
        if not traced:
            raise GateViolation(
                f"cluster {cluster.cluster_id}: merged finding ({merged.file}:{merged.line}) "
                f"does not trace to any finding from its cited passes - possible hallucinated merge"
            )


def check_g3(output: ValidatorOutput, valid_cluster_ids: set[str]) -> None:
    if output.cluster_id not in valid_cluster_ids:
        raise GateViolation(f"unknown cluster_id: {output.cluster_id}")
    if output.verdict == "confirmed" and not output.comment_markdown.strip():
        raise GateViolation("verdict=confirmed requires a non-empty comment_markdown")


async def run_with_retries(agent_call, validate_fn, max_retries: int = MAX_RETRIES):
    """agent_call(feedback: str | None) -> raw model output for THIS node only.
    validate_fn(raw) -> parsed result; raises ValidationError or GateViolation.

    Retries only this node, feeding the validation error back into the next
    call so the model sees exactly what it broke. Raises NodeExhausted after
    max_retries+1 total attempts - callers apply their own degrade path.
    """
    last_error: Exception | None = None
    feedback: str | None = None

    for _ in range(max_retries + 1):
        raw = await agent_call(feedback)
        try:
            return validate_fn(raw)
        except (ValidationError, GateViolation) as exc:
            last_error = exc
            feedback = str(exc)

    raise NodeExhausted(last_error)
