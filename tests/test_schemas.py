import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import (  # noqa: E402
    AggregatorOutput,
    Cluster,
    Finding,
    ReviewerOutput,
    ValidatorOutput,
    to_response_schema,
)


def make_finding(**overrides):
    base = dict(
        file="src/app.py", line=10, category="logic", severity="medium",
        title="null deref on missing user", reasoning="getUser() can return None",
        evidence_lines=[9, 10], self_confidence=0.7,
    )
    base.update(overrides)
    return Finding(**base)


class TestReviewerOutput(unittest.TestCase):
    def test_valid_construction(self):
        out = ReviewerOutput(pass_id="p1", findings=[make_finding()])
        self.assertEqual(out.pass_id, "p1")

    def test_rejects_bad_category(self):
        with self.assertRaises(ValidationError):
            make_finding(category="not-a-real-category")

    def test_rejects_confidence_out_of_range(self):
        with self.assertRaises(ValidationError):
            make_finding(self_confidence=1.5)

    def test_rejects_title_over_60_chars(self):
        with self.assertRaises(ValidationError):
            make_finding(title="x" * 61)

    def test_rejects_more_than_20_findings(self):
        with self.assertRaises(ValidationError):
            ReviewerOutput(pass_id="p1", findings=[make_finding() for _ in range(21)])


class TestAggregatorOutput(unittest.TestCase):
    def test_valid_construction(self):
        cluster = Cluster(cluster_id="c1", vote_count=2, supporting_pass_ids=["p1", "p2"], merged=make_finding())
        out = AggregatorOutput(clusters=[cluster])
        self.assertEqual(out.clusters[0].vote_count, 2)


class TestValidatorOutput(unittest.TestCase):
    def test_valid_construction(self):
        out = ValidatorOutput(
            cluster_id="c1", verdict="confirmed", refutation="none found",
            validator_family="llama", validator_confidence=0.8,
            comment_markdown="**Bug found**: ...",
        )
        self.assertEqual(out.verdict, "confirmed")

    def test_rejects_bad_verdict(self):
        with self.assertRaises(ValidationError):
            ValidatorOutput(
                cluster_id="c1", verdict="maybe", refutation="x",
                validator_family="llama", validator_confidence=0.5,
            )


class TestResponseSchema(unittest.TestCase):
    def test_no_title_keys_leak_through(self):
        schema = to_response_schema(ReviewerOutput)
        self.assertNotIn("title", schema)

    def test_has_expected_top_level_property(self):
        schema = to_response_schema(ReviewerOutput)
        self.assertIn("findings", schema["properties"])


if __name__ == "__main__":
    unittest.main()
