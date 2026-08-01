import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stats import wilson_lower_bound  # noqa: E402


class TestWilsonLowerBound(unittest.TestCase):
    def test_zero_total_returns_zero(self):
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)

    def test_zero_successes_returns_zero(self):
        self.assertEqual(wilson_lower_bound(0, 4), 0.0)

    def test_bounded_between_zero_and_one(self):
        for successes, total in [(1, 1), (4, 4), (2, 4), (1, 100), (99, 100)]:
            lower = wilson_lower_bound(successes, total)
            self.assertGreaterEqual(lower, 0.0)
            self.assertLessEqual(lower, 1.0)

    def test_unanimous_beats_split_at_same_ratio_of_n(self):
        # 4-of-4 unanimous should score higher than 2-of-4 (half agree, half
        # implicitly don't) - the core property the plan's confidence
        # formula depends on.
        self.assertGreater(wilson_lower_bound(4, 4), wilson_lower_bound(2, 4))

    def test_small_unanimous_sample_scores_above_larger_split_sample(self):
        # 2-of-2 unanimous (small n, full agreement) must rank ABOVE 2-of-4
        # split (larger n, half disagree) - this is the exact ranking a
        # naive successes/total * (n/max_n) penalty inverts. See plan.
        self.assertGreater(wilson_lower_bound(2, 2), wilson_lower_bound(2, 4))

    def test_more_data_at_same_ratio_increases_confidence(self):
        # 50% agreement is a weaker signal with 2 samples than with 100.
        self.assertGreater(wilson_lower_bound(50, 100), wilson_lower_bound(1, 2))

    def test_full_agreement_approaches_but_stays_under_one(self):
        lower = wilson_lower_bound(1000, 1000)
        self.assertLess(lower, 1.0)
        self.assertGreater(lower, 0.99)

    def test_rejects_successes_greater_than_total(self):
        with self.assertRaises(ValueError):
            wilson_lower_bound(5, 4)

    def test_rejects_negative_successes(self):
        with self.assertRaises(ValueError):
            wilson_lower_bound(-1, 4)


if __name__ == "__main__":
    unittest.main()
