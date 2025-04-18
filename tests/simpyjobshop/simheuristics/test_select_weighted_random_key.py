import numpy as np
import pytest

from simpyjobshop.simheuristics.dynamic_simheuristic import (
    select_weighted_random_key,
)


def test_select_weighted_random_key_basic():
    scores = {"a": 1, "b": 2, "c": 3}
    np.random.seed(42)
    result = select_weighted_random_key(scores)
    assert result in scores


def test_select_weighted_random_key_deterministic_seed():
    scores = {"x": 1, "y": 1, "z": 1}
    np.random.seed(123)
    result1 = select_weighted_random_key(scores)
    np.random.seed(123)
    result2 = select_weighted_random_key(scores)
    assert result1 == result2  # same seed => same result


def test_select_weighted_random_key_distribution():
    scores = {"yes": 1, "no": 0}
    results = [select_weighted_random_key(scores) for _ in range(100)]
    assert all(r == "yes" for r in results)


def test_all_zero_scores_raises():
    with pytest.raises(ValueError, match="zero"):
        select_weighted_random_key({"a": 0, "b": 0})
