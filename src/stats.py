"""Shared statistics. wilson_lower_bound backs both A2's published confidence
score (vote_count / passes_surviving as a small-sample proportion) and the
eval harness's precision/recall confidence intervals - one function, two
callers, so the two don't quietly drift into different math.
"""

from __future__ import annotations

import math


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion.

    Penalizes small samples by construction: a 2-of-2 unanimous result and a
    4-of-4 unanimous result don't score the same, because the interval is
    wider with less data. This is why it replaces a naive successes/total
    ratio for A1's vote score - see plan "Published confidence" for the
    2-of-2-vs-2-of-4 ranking-inversion bug a naive extra penalty term causes.
    """
    if total <= 0:
        return 0.0
    if successes < 0 or successes > total:
        raise ValueError(f"successes={successes} out of range for total={total}")

    n = total
    phat = successes / n
    z2 = z * z

    denominator = 1 + z2 / n
    center = phat + z2 / (2 * n)
    adjustment = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)

    lower = (center - adjustment) / denominator
    return max(0.0, min(1.0, lower))
