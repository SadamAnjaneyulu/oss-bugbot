"""A2: LLM semantic clustering + majority vote, wrapped in G2 with
retry-same-node, deterministic-clustering degrade on NodeExhausted.

Note: the published confidence score (wilson_lower_bound(vote_count,
passes_surviving) * validator_weight) needs A3's verdict too, so it isn't
computed here - this module hands off Cluster objects with vote_count and
supporting_pass_ids; the score is assembled downstream once A3 has run.
"""

from __future__ import annotations

import json

from gates import NodeExhausted, check_g2, run_with_retries
from llm import LLMResponse, SafetyRefusal, TokenUsage, call_llm
from schemas import AggregatorOutput, Cluster, Finding, to_response_schema

DEFAULT_VOTE_THRESHOLD = 2
DETERMINISTIC_LINE_TOLERANCE = 3  # same rule as gates.py's trace-check tolerance

SYSTEM_PROMPT = """\
You cluster bug-report findings from multiple independent reviewers of the \
same pull request. Findings describing the same underlying bug - even if \
worded differently or off by a few lines - belong in one cluster. Every \
cluster's supporting_pass_ids must be pass_ids that were actually given to \
you. vote_count must equal the number of supporting_pass_ids. The "merged" \
finding in each cluster must be a real finding from one of the cited \
passes, not one you invent - do not synthesize new findings, only group \
findings you were given. Findings you were not given never happened.
"""


def _findings_prompt(findings_by_pass: dict[str, list[Finding]]) -> str:
    items = []
    for pass_id, findings in findings_by_pass.items():
        for f in findings:
            items.append({"pass_id": pass_id, **f.model_dump()})
    return json.dumps(items, indent=2)


async def _call_a2(client, model: str, semaphore, findings_by_pass: dict[str, list[Finding]], feedback: str | None) -> LLMResponse:
    schema = to_response_schema(AggregatorOutput)
    user_text = _findings_prompt(findings_by_pass)
    if feedback:
        user_text += f"\n\nYour previous attempt was rejected: {feedback}\nFix this and try again."
    return await call_llm(client, model, SYSTEM_PROMPT, user_text, schema, "AggregatorOutput", semaphore)


def _deterministic_cluster(findings_by_pass: dict[str, list[Finding]]) -> list[Cluster]:
    """Fallback when A2 exhausts retries. Crude but never wrong in a way
    that invents bugs: every merged finding IS a real input finding, so
    there is nothing here for G2's hallucinated-merge check to catch,
    because this path never goes through G2 at all - it's the fallback for
    when the LLM path already failed it.
    """
    groups: list[dict] = []
    for pass_id, findings in findings_by_pass.items():
        for f in findings:
            target = None
            for g in groups:
                if g["file"] == f.file and g["category"] == f.category and any(
                    abs(f.line - line) <= DETERMINISTIC_LINE_TOLERANCE for line in g["lines"]
                ):
                    target = g
                    break
            if target is None:
                target = {"file": f.file, "category": f.category, "lines": [], "pass_ids": [], "findings": []}
                groups.append(target)
            target["lines"].append(f.line)
            target["pass_ids"].append(pass_id)
            target["findings"].append(f)

    clusters = []
    for i, g in enumerate(groups):
        representative = max(g["findings"], key=lambda f: f.self_confidence)
        clusters.append(Cluster(
            cluster_id=f"c{i + 1}",
            vote_count=len(g["pass_ids"]),
            supporting_pass_ids=g["pass_ids"],
            merged=representative,
        ))
    return clusters


async def run_aggregation(
    client,
    model: str,
    semaphore,
    findings_by_pass: dict[str, list[Finding]],
    threshold: int = DEFAULT_VOTE_THRESHOLD,
) -> tuple[list[Cluster], bool, TokenUsage]:
    """Returns (surviving_clusters, used_deterministic_fallback, token_usage).
    The bool is for findings.json's degradation log - callers must not
    silently publish deterministic-fallback results as if A2 succeeded.
    token_usage sums every attempt including failed retries - a rejected
    attempt still spent real tokens.
    """
    valid_pass_ids = set(findings_by_pass.keys())
    total_usage = TokenUsage()
    if not any(findings_by_pass.values()):
        return [], False, total_usage

    async def agent_call(feedback: str | None) -> str:
        response = await _call_a2(client, model, semaphore, findings_by_pass, feedback)
        total_usage.add(response)
        return response.text

    def validate_fn(raw_text: str) -> AggregatorOutput:
        output = AggregatorOutput.model_validate_json(raw_text)
        check_g2(output, valid_pass_ids, findings_by_pass)
        return output

    used_fallback = False
    try:
        output = await run_with_retries(agent_call, validate_fn)
        clusters = output.clusters
    except (NodeExhausted, SafetyRefusal):
        clusters = _deterministic_cluster(findings_by_pass)
        used_fallback = True

    surviving = [c for c in clusters if c.vote_count >= threshold]
    return surviving, used_fallback, total_usage
