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

    def test_field_named_title_survives_stripping(self):
        # Regression test: an earlier _strip() implementation called
        # node.pop("title") on every dict, which deleted the "title"
        # PROPERTY from the Finding schema's properties container (its key
        # is the string "title") while `required` still listed it -
        # producing a schema Gemini's live API rejected with 400
        # INVALID_ARGUMENT: "requires unspecified property 'title'".
        # Only caught by a live call; no mock touches real schema validation.
        schema = to_response_schema(ReviewerOutput)
        # Nested models live under $defs with a $ref from `items`, not inlined.
        finding_props = schema["$defs"]["Finding"]["properties"]
        self.assertIn("title", finding_props, "the 'title' FIELD must survive stripping")
        self.assertIn("title", schema["$defs"]["Finding"]["required"])
        # and the metadata-title INSIDE that field's own definition must be gone
        self.assertNotIn("title", finding_props["title"])


if __name__ == "__main__":
    unittest.main()
