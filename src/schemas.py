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


def to_response_schema(model: type[BaseModel]) -> dict:
    """Strips pydantic/JSON-Schema keys providers' structured-output modes
    reject (title, $defs refs aren't universally supported) down to a plain
    schema dict. Gemini's responseSchema and Groq's json_schema mode both
    want the same shape: this is the one place that assumption lives.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)

    def _strip(node):
        if isinstance(node, dict):
            node.pop("title", None)
            for v in node.values():
                _strip(v)
        elif isinstance(node, list):
            for v in node:
                _strip(v)

    _strip(schema)
    return schema
