import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import Finding  # noqa: E402
from vote import _deterministic_cluster, run_aggregation  # noqa: E402


def make_finding(**overrides):
    base = dict(
        file="app.py", line=10, category="logic", severity="medium",
        title="null deref", reasoning="x", evidence_lines=[10], self_confidence=0.7,
    )
    base.update(overrides)
    return Finding(**base)


def gemini_response(text, finish_reason="STOP"):
    candidate = SimpleNamespace(finish_reason=SimpleNamespace(value=finish_reason))
    return SimpleNamespace(candidates=[candidate], text=text)


def make_client(responses):
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=responses)
    return client


class TestDeterministicCluster(unittest.TestCase):
    def test_merges_nearby_same_file_same_category(self):
        f1 = make_finding(line=10, self_confidence=0.5)
        f2 = make_finding(line=12, self_confidence=0.9)
        clusters = _deterministic_cluster({"p1": [f1], "p2": [f2]})
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].vote_count, 2)
        self.assertEqual(clusters[0].merged.self_confidence, 0.9)  # picked highest

    def test_does_not_merge_far_apart_lines(self):
        f1 = make_finding(line=10)
        f2 = make_finding(line=50)
        clusters = _deterministic_cluster({"p1": [f1], "p2": [f2]})
        self.assertEqual(len(clusters), 2)

    def test_does_not_merge_different_files(self):
        f1 = make_finding(file="a.py", line=10)
        f2 = make_finding(file="b.py", line=10)
        clusters = _deterministic_cluster({"p1": [f1], "p2": [f2]})
        self.assertEqual(len(clusters), 2)

    def test_does_not_merge_different_categories(self):
        f1 = make_finding(category="logic", line=10)
        f2 = make_finding(category="security", line=10)
        clusters = _deterministic_cluster({"p1": [f1], "p2": [f2]})
        self.assertEqual(len(clusters), 2)

    def test_merged_finding_is_always_a_real_input_finding(self):
        f1 = make_finding(line=10, self_confidence=0.9)
        clusters = _deterministic_cluster({"p1": [f1]})
        self.assertIs(clusters[0].merged, f1)


class TestRunAggregation(unittest.TestCase):
    def setUp(self):
        self.sem = asyncio.Semaphore(1)

    def test_empty_findings_returns_empty(self):
        result, used_fallback = asyncio.run(run_aggregation(make_client([]), "m", self.sem, {}))
        self.assertEqual(result, [])
        self.assertFalse(used_fallback)

    def test_llm_success_path(self):
        f1 = make_finding()
        raw = json.dumps({"clusters": [{
            "cluster_id": "c1", "vote_count": 1, "supporting_pass_ids": ["p1"],
            "merged": f1.model_dump(),
        }]})
        client = make_client([gemini_response(raw)])
        clusters, used_fallback = asyncio.run(run_aggregation(client, "m", self.sem, {"p1": [f1]}, threshold=1))
        self.assertEqual(len(clusters), 1)
        self.assertFalse(used_fallback)

    def test_threshold_filters_low_vote_clusters(self):
        f1 = make_finding()
        raw = json.dumps({"clusters": [{
            "cluster_id": "c1", "vote_count": 1, "supporting_pass_ids": ["p1"],
            "merged": f1.model_dump(),
        }]})
        client = make_client([gemini_response(raw)])
        clusters, _ = asyncio.run(run_aggregation(client, "m", self.sem, {"p1": [f1]}, threshold=2))
        self.assertEqual(clusters, [])  # vote_count=1 < threshold=2

    def test_g2_retry_then_success(self):
        f1 = make_finding()
        bad = json.dumps({"clusters": [{
            "cluster_id": "c1", "vote_count": 1, "supporting_pass_ids": ["ghost_pass"],  # unknown pass_id
            "merged": f1.model_dump(),
        }]})
        good = json.dumps({"clusters": [{
            "cluster_id": "c1", "vote_count": 1, "supporting_pass_ids": ["p1"],
            "merged": f1.model_dump(),
        }]})
        client = make_client([gemini_response(bad), gemini_response(good)])
        clusters, used_fallback = asyncio.run(run_aggregation(client, "m", self.sem, {"p1": [f1]}, threshold=1))
        self.assertEqual(len(clusters), 1)
        self.assertFalse(used_fallback)
        self.assertEqual(client.aio.models.generate_content.await_count, 2)

    def test_g2_exhausted_falls_back_to_deterministic(self):
        f1 = make_finding()
        always_bad = json.dumps({"clusters": [{
            "cluster_id": "c1", "vote_count": 1, "supporting_pass_ids": ["ghost"],
            "merged": f1.model_dump(),
        }]})
        client = make_client([gemini_response(always_bad)] * 10)
        clusters, used_fallback = asyncio.run(run_aggregation(client, "m", self.sem, {"p1": [f1]}, threshold=1))
        self.assertTrue(used_fallback)
        self.assertEqual(len(clusters), 1)  # deterministic path still clusters the one finding

    def test_safety_refusal_falls_back_to_deterministic(self):
        f1 = make_finding()
        client = make_client([gemini_response(None, finish_reason="SAFETY")])
        clusters, used_fallback = asyncio.run(run_aggregation(client, "m", self.sem, {"p1": [f1]}, threshold=1))
        self.assertTrue(used_fallback)
        self.assertEqual(len(clusters), 1)


if __name__ == "__main__":
    unittest.main()
