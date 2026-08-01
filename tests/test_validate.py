import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import Cluster, Finding  # noqa: E402
from validate import run_all_validators, run_validator  # noqa: E402


def make_cluster(cluster_id="c1"):
    f = Finding(
        file="app.py", line=10, category="logic", severity="medium",
        title="null deref", reasoning="x", evidence_lines=[10], self_confidence=0.7,
    )
    return Cluster(cluster_id=cluster_id, vote_count=2, supporting_pass_ids=["p1", "p2"], merged=f)


def groq_client_with_responses(contents, finish_reasons=None):
    finish_reasons = finish_reasons or ["stop"] * len(contents)
    responses = []
    for content, fr in zip(contents, finish_reasons):
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason=fr)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=10)
        responses.append(SimpleNamespace(choices=[choice], usage=usage))
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=responses)
    return client


def gemini_client_with_responses(texts, finish_reasons=None):
    finish_reasons = finish_reasons or ["STOP"] * len(texts)
    responses = []
    for text, fr in zip(texts, finish_reasons):
        candidate = SimpleNamespace(finish_reason=SimpleNamespace(value=fr))
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=10)
        responses.append(SimpleNamespace(candidates=[candidate], text=text, usage_metadata=usage))
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=responses)
    return client


def confirmed_json(cluster_id="c1", family="llama"):
    return json.dumps({
        "cluster_id": cluster_id, "verdict": "confirmed", "refutation": "none found",
        "validator_family": family, "validator_confidence": 0.9,
        "comment_markdown": "**Bug**: real issue here",
    })


class TestRunValidator(unittest.TestCase):
    def setUp(self):
        self.groq_sem = asyncio.Semaphore(1)
        self.gemini_sem = asyncio.Semaphore(1)

    def test_groq_success_confirmed(self):
        cluster = make_cluster()
        groq = groq_client_with_responses([confirmed_json()])
        gemini = gemini_client_with_responses([])  # never called
        result, usage = asyncio.run(run_validator(groq, gemini, "gm", "gem", self.groq_sem, self.gemini_sem, cluster))
        self.assertEqual(result.verdict, "confirmed")
        self.assertEqual(result.validator_family, "llama")
        self.assertTrue(result.comment_markdown)
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 10)

    def test_groq_refusal_falls_back_to_gemini(self):
        cluster = make_cluster()
        groq = groq_client_with_responses([None])  # empty content -> refusal
        gemini = gemini_client_with_responses([confirmed_json(family="gemini")])
        result, usage = asyncio.run(run_validator(groq, gemini, "gm", "gem", self.groq_sem, self.gemini_sem, cluster))
        self.assertEqual(result.validator_family, "gemini")
        self.assertEqual(result.verdict, "confirmed")

    def test_g3_retry_then_recover(self):
        cluster = make_cluster(cluster_id="c1")
        bad = json.dumps({
            "cluster_id": "wrong_cluster", "verdict": "confirmed", "refutation": "x",
            "validator_family": "llama", "validator_confidence": 0.5, "comment_markdown": "x",
        })
        good = confirmed_json(cluster_id="c1")
        groq = groq_client_with_responses([bad, good])
        gemini = gemini_client_with_responses([])
        result, usage = asyncio.run(run_validator(groq, gemini, "gm", "gem", self.groq_sem, self.gemini_sem, cluster))
        self.assertEqual(result.cluster_id, "c1")
        self.assertEqual(groq.chat.completions.create.await_count, 2)

    def test_g3_exhausted_degrades_to_uncertain(self):
        cluster = make_cluster()
        bad = json.dumps({
            "cluster_id": "wrong", "verdict": "confirmed", "refutation": "x",
            "validator_family": "llama", "validator_confidence": 0.5, "comment_markdown": "x",
        })
        groq = groq_client_with_responses([bad] * 10)
        gemini = gemini_client_with_responses([])
        result, usage = asyncio.run(run_validator(groq, gemini, "gm", "gem", self.groq_sem, self.gemini_sem, cluster))
        self.assertEqual(result.verdict, "uncertain")
        self.assertEqual(result.cluster_id, cluster.cluster_id)

    def test_both_providers_refuse_degrades_to_uncertain(self):
        cluster = make_cluster()
        groq = groq_client_with_responses([None])
        gemini = gemini_client_with_responses([None])
        result, usage = asyncio.run(run_validator(groq, gemini, "gm", "gem", self.groq_sem, self.gemini_sem, cluster))
        self.assertEqual(result.verdict, "uncertain")

    def test_false_positive_verdict_allows_empty_comment(self):
        cluster = make_cluster()
        raw = json.dumps({
            "cluster_id": "c1", "verdict": "false_positive", "refutation": "not actually reachable",
            "validator_family": "llama", "validator_confidence": 0.8, "comment_markdown": "",
        })
        groq = groq_client_with_responses([raw])
        gemini = gemini_client_with_responses([])
        result, usage = asyncio.run(run_validator(groq, gemini, "gm", "gem", self.groq_sem, self.gemini_sem, cluster))
        self.assertEqual(result.verdict, "false_positive")


class TestRunAllValidators(unittest.TestCase):
    def setUp(self):
        self.groq_sem = asyncio.Semaphore(1)
        self.gemini_sem = asyncio.Semaphore(1)

    def test_empty_clusters_returns_empty(self):
        groq = groq_client_with_responses([])
        gemini = gemini_client_with_responses([])
        result, usage = asyncio.run(run_all_validators(groq, gemini, "gm", "gem", self.groq_sem, self.gemini_sem, []))
        self.assertEqual(result, [])

    def test_fans_out_over_multiple_clusters(self):
        c1 = make_cluster("c1")
        c2 = make_cluster("c2")
        groq = groq_client_with_responses([confirmed_json("c1"), confirmed_json("c2")])
        gemini = gemini_client_with_responses([])
        results, usage = asyncio.run(run_all_validators(groq, gemini, "gm", "gem", self.groq_sem, self.gemini_sem, [c1, c2]))
        self.assertEqual({r.cluster_id for r in results}, {"c1", "c2"})
        # Both clusters' calls summed into one total.
        self.assertEqual(usage.input_tokens, 20)
        self.assertEqual(usage.output_tokens, 20)


if __name__ == "__main__":
    unittest.main()
