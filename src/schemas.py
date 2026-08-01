"""S1/S2/S3 contracts -- single source of truth. Exported to each provider's
structured-output mode (Gemini responseSchema / Groq JSON mode) so malformed
JSON is decode-time-impossible; gates.py checks what these schemas can't
express (semantic truth, not shape).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["logic", "security", "resource", "concurrency", "api-misuse"]
Severity = Literal["high", "medium", "low"]
Verdict = Literal["confirmed", "false_positive", "uncertain"]
ValidatorFamily = Literal["llama", "gemini"]


class Finding(BaseModel):
    file: str
    line: int
    category: Category
    severity: Severity
    title: str = Field(max_length=60)
    reasoning: str
    evidence_lines: list[int]
    semgrep_corroborated: bool = False
    # Model self-report, poorly calibrated by construction. Recorded for
    # analysis and as an A2 tiebreak only -- never reaches the published
    # score. See plan "Published confidence - computed, not model-reported".
    self_confidence: float = Field(ge=0.0, le=1.0)


class ReviewerOutput(BaseModel):
    """S1 - A1 -> G1"""
    pass_id: str
    findings: list[Finding] = Field(max_length=20)  # G1 also enforces this; belt+suspenders


class Cluster(BaseModel):
    cluster_id: str
    vote_count: int
    supporting_pass_ids: list[str]
    merged: Finding


class AggregatorOutput(BaseModel):
    """S2 - A2 -> G2"""
    clusters: list[Cluster]


class ValidatorOutput(BaseModel):
    """S3 - A3 -> G3"""
    cluster_id: str
    verdict: Verdict
    refutation: str
    validator_family: ValidatorFamily
    validator_confidence: float = Field(ge=0.0, le=1.0)
    comment_markdown: str = ""


_CONTAINER_KEYS = {"properties", "$defs"}


def to_response_schema(model: type[BaseModel]) -> dict:
    """Strips Pydantic's auto-generated "title" metadata (Gemini's schema
    validator rejects it) down to a plain schema dict for providers'
    structured-output modes.

    _strip must NOT pop "title" from a "properties" or "$defs" container
    dict's own keys - those keys are field/definition NAMES, and a field
    can legitimately be named "title" (Finding.title is). A naive
    `node.pop("title")` on every dict deletes that property's entire
    definition while `required` still lists it, producing a schema Gemini
    rejects with "requires unspecified property 'title'" - caught by a live
    call, not by any mock, because the mocked tests never round-tripped a
    real schema through this function's actual bug.
    """
    schema = model.model_json_schema()

    def _strip(node, is_container: bool = False):
        if isinstance(node, dict):
            if not is_container:
                node.pop("title", None)
            for key, v in node.items():
                _strip(v, is_container=(key in _CONTAINER_KEYS))
        elif isinstance(node, list):
            for v in node:
                _strip(v)

    _strip(schema)
    _forbid_additional_properties(schema)
    _require_all_properties(schema)
    return schema


def _require_all_properties(node) -> None:
    """Groq strict mode also 400s unless every property key appears in
    `required`, even ones Pydantic gave a default (comment_markdown="" is
    still required-to-be-present in strict OpenAI-style schemas - the model
    must emit the key, an empty string satisfies it). Same live-caught
    pattern as _forbid_additional_properties: Pydantic's default
    model_json_schema() doesn't do this, Groq's validator demands it.
    """
    if isinstance(node, dict):
        if "properties" in node and isinstance(node["properties"], dict):
            node["required"] = list(node["properties"].keys())
        for v in node.values():
            _require_all_properties(v)
    elif isinstance(node, list):
        for v in node:
            _require_all_properties(v)


def _forbid_additional_properties(node) -> None:
    """Groq's strict JSON-schema mode 400s with 'additionalProperties:false
    must be set on every object' unless every object node has it - Pydantic
    doesn't emit this by default. Applied universally (not Groq-only): it's
    valid, harmless JSON Schema everywhere, and the Gemini call already
    confirmed live that adding it doesn't break Gemini's acceptance either.
    One schema shape for both providers beats two functions to keep in sync.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
        for v in node.values():
            _forbid_additional_properties(v)
    elif isinstance(node, list):
        for v in node:
            _forbid_additional_properties(v)
