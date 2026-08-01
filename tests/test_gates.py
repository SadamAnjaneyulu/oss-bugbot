import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gates import GateViolation, NodeExhausted, check_g1, check_g2, check_g3, run_with_retries  # noqa: E402
from schemas import AggregatorOutput, Cluster, Finding, ReviewerOutput, ValidatorOutput  # noqa: E402


def make_finding(**overrides):
    base = dict(
        file="src/app.py", line=10, category="logic", severity="medium",
        title="null deref", reasoning="getUser() can return None",
        evidence_lines=[10], self_confidence=0.7,
    )
    base.update(overrides)
    return Finding(**base)


class TestG1(unittest.TestCase):
    def setUp(self):
        self.changed_lines = {"src/app.py": {8, 9, 10, 11}}
        self.deleted = set()

    def test_valid_finding_passes(self):
        out = ReviewerOutput(pass_id="p1", findings=[make_finding()])
        check_g1(out, self.changed_lines, self.deleted)  # no raise

    def test_line_outside_diff_rejected(self):
        out = ReviewerOutput(pass_id="p1", findings=[make_finding(line=999, evidence_lines=[999])])
        with self.assertRaises(GateViolation):
            check_g1(out, self.changed_lines, self.deleted)

    def test_file_not_in_diff_rejected(self):
        out = ReviewerOutput(pass_id="p1", findings=[make_finding(file="other.py")])
        with self.assertRaises(GateViolation):
            check_g1(out, self.changed_lines, self.deleted)

    def test_deleted_file_rejected(self):
        out = ReviewerOutput(pass_id="p1", findings=[make_finding()])
        with self.assertRaises(GateViolation):
            check_g1(out, self.changed_lines, deleted_files={"src/app.py"})

    def test_evidence_lines_outside_diff_rejected(self):
        out = ReviewerOutput(pass_id="p1", findings=[make_finding(evidence_lines=[10, 500])])
        with self.assertRaises(GateViolation):
            check_g1(out, self.changed_lines, self.deleted)


class TestG2(unittest.TestCase):
    def test_valid_cluster_passes(self):
        f1 = make_finding()
        cluster = Cluster(cluster_id="c1", vote_count=1, supporting_pass_ids=["p1"], merged=f1)
        out = AggregatorOutput(clusters=[cluster])
        check_g2(out, valid_pass_ids={"p1", "p2"}, findings_by_pass={"p1": [f1]})  # no raise

    def test_unknown_pass_id_rejected(self):
        f1 = make_finding()
        cluster = Cluster(cluster_id="c1", vote_count=1, supporting_pass_ids=["ghost"], merged=f1)
        out = AggregatorOutput(clusters=[cluster])
        with self.assertRaises(GateViolation):
            check_g2(out, valid_pass_ids={"p1"}, findings_by_pass={"p1": [f1]})

    def test_vote_count_mismatch_rejected(self):
        f1 = make_finding()
        cluster = Cluster(cluster_id="c1", vote_count=3, supporting_pass_ids=["p1"], merged=f1)
        out = AggregatorOutput(clusters=[cluster])
        with self.assertRaises(GateViolation):
            check_g2(out, valid_pass_ids={"p1"}, findings_by_pass={"p1": [f1]})

    def test_hallucinated_merge_rejected(self):
        real = make_finding(file="src/app.py", line=10, category="logic")
        hallucinated = make_finding(file="src/completely_different.py", line=200, category="security")
        cluster = Cluster(cluster_id="c1", vote_count=1, supporting_pass_ids=["p1"], merged=hallucinated)
        out = AggregatorOutput(clusters=[cluster])
        with self.assertRaises(GateViolation):
            check_g2(out, valid_pass_ids={"p1"}, findings_by_pass={"p1": [real]})

    def test_merge_within_line_tolerance_accepted(self):
        real = make_finding(file="src/app.py", line=10, category="logic")
        merged = make_finding(file="src/app.py", line=12, category="logic")  # within tolerance=3
        cluster = Cluster(cluster_id="c1", vote_count=1, supporting_pass_ids=["p1"], merged=merged)
        out = AggregatorOutput(clusters=[cluster])
        check_g2(out, valid_pass_ids={"p1"}, findings_by_pass={"p1": [real]})  # no raise


class TestG3(unittest.TestCase):
    def test_confirmed_with_comment_passes(self):
        out = ValidatorOutput(
            cluster_id="c1", verdict="confirmed", refutation="none",
            validator_family="llama", validator_confidence=0.9,
            comment_markdown="**Bug**: real issue",
        )
        check_g3(out, valid_cluster_ids={"c1"})  # no raise

    def test_unknown_cluster_id_rejected(self):
        out = ValidatorOutput(
            cluster_id="ghost", verdict="confirmed", refutation="none",
            validator_family="llama", validator_confidence=0.9,
            comment_markdown="text",
        )
        with self.assertRaises(GateViolation):
            check_g3(out, valid_cluster_ids={"c1"})

    def test_confirmed_without_comment_rejected(self):
        out = ValidatorOutput(
            cluster_id="c1", verdict="confirmed", refutation="none",
            validator_family="llama", validator_confidence=0.9,
            comment_markdown="",
        )
        with self.assertRaises(GateViolation):
            check_g3(out, valid_cluster_ids={"c1"})

    def test_false_positive_without_comment_ok(self):
        out = ValidatorOutput(
            cluster_id="c1", verdict="false_positive", refutation="not a bug",
            validator_family="llama", validator_confidence=0.9,
            comment_markdown="",
        )
        check_g3(out, valid_cluster_ids={"c1"})  # no raise


class TestRunWithRetries(unittest.TestCase):
    def test_succeeds_first_try(self):
        async def agent_call(feedback):
            return {"n": 1}

        def validate_fn(raw):
            return raw["n"]

        result = asyncio.run(run_with_retries(agent_call, validate_fn))
        self.assertEqual(result, 1)

    def test_retries_same_node_and_recovers(self):
        calls = []

        async def agent_call(feedback):
            calls.append(feedback)
            return {"ok": len(calls) >= 2}

        def validate_fn(raw):
            if not raw["ok"]:
                raise GateViolation("not ready yet")
            return "recovered"

        result = asyncio.run(run_with_retries(agent_call, validate_fn, max_retries=2))
        self.assertEqual(result, "recovered")
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0])  # first call: no feedback yet
        self.assertIn("not ready yet", calls[1])  # second call: prior error fed back

    def test_exhausts_and_raises_node_exhausted(self):
        async def agent_call(feedback):
            return {"bad": True}

        def validate_fn(raw):
            raise GateViolation("always fails")

        with self.assertRaises(NodeExhausted):
            asyncio.run(run_with_retries(agent_call, validate_fn, max_retries=2))

    def test_does_not_call_upstream_agent(self):
        # The wrapper only ever calls the ONE agent_call passed in - proving
        # "retry the failing node, not the one before it" isn't just a
        # docstring claim: there is no code path here that could re-invoke
        # a different node.
        call_count = {"n": 0}

        async def agent_call(feedback):
            call_count["n"] += 1
            return {"n": call_count["n"]}

        def validate_fn(raw):
            if raw["n"] < 3:
                raise GateViolation("retry")
            return raw["n"]

        result = asyncio.run(run_with_retries(agent_call, validate_fn, max_retries=2))
        self.assertEqual(result, 3)
        self.assertEqual(call_count["n"], 3)


if __name__ == "__main__":
    unittest.main()
