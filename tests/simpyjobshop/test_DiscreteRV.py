import numpy as np
import pytest
from scipy.stats import poisson

from simpyjobshop.DiscreteRV import (
    Constant,
    CustomRV,
    DiscreteRV,
    SeededPoisson,
)


def test_discrete_rv_copy():
    distribution = poisson(mu=5)
    rv = DiscreteRV(distribution, seed=42)
    rv_copy = rv.copy()

    assert isinstance(rv_copy, DiscreteRV)
    assert rv_copy.seed == rv.seed
    assert rv_copy.distribution == rv.distribution
    assert rv_copy.rvs() == rv.rvs()


def test_discrete_rv_rvs():
    distribution = poisson(mu=5)
    rv = DiscreteRV(distribution, seed=42)
    samples = rv.rvs(size=5)

    assert isinstance(samples, np.ndarray)
    assert len(samples) == 5


def test_discrete_rv_mean_var():
    distribution = poisson(mu=5)
    rv = DiscreteRV(distribution, seed=42)

    assert rv.mean() == pytest.approx(5)
    assert rv.var() == pytest.approx(5)


def test_discrete_rv_ppf():
    distribution = poisson(mu=5)
    rv = DiscreteRV(distribution, seed=42)

    quantile = rv.ppf(0.5)
    assert isinstance(quantile, int)
    assert quantile == poisson(mu=5).ppf(0.5)


def test_seeded_poisson_rvs():
    lam = 7
    sp = SeededPoisson(lam=lam, seed=0)

    samples = sp.rvs(size=10)
    assert isinstance(samples, np.ndarray)
    assert len(samples) == 10
    assert np.mean(samples) == pytest.approx(lam, rel=0.1)


def test_seeded_loc_poisson_rvs():
    lam = 7
    loc = 5
    sp = SeededPoisson(lam=lam, loc=loc, seed=0)

    samples = sp.rvs(size=10)
    assert isinstance(samples, np.ndarray)
    assert len(samples) == 10
    assert np.mean(samples) == pytest.approx(lam + loc, rel=0.1)


def test_seeded_poisson_mean_var():
    lam = 7
    sp = SeededPoisson(lam=lam, seed=0)

    assert sp.mean() == pytest.approx(lam)
    assert sp.var() == pytest.approx(lam)


def test_constant_rvs():
    value = 42
    const_rv = Constant(value=value, seed=0)

    samples = const_rv.rvs(size=10)
    assert all(sample == value for sample in samples)
    assert len(samples) == 10


def test_constant_mean_var():
    value = 42
    const_rv = Constant(value=value, seed=0)

    assert const_rv.mean() == pytest.approx(value)
    assert const_rv.var() == 0


def test_constant_ppf():
    value = 42
    const_rv = Constant(value=value, seed=0)

    for p in [0.1, 0.5, 0.9, 1.0]:
        assert const_rv.ppf(p) == value


def test_custom_rv_valid_distribution():
    rv = CustomRV([1, 2, 3], [0.2, 0.3, 0.5])
    sample = rv.rvs(size=10)
    assert all(x in [1, 2, 3] for x in sample)


def test_custom_rv_invalid_length():
    with pytest.raises(ValueError, match="same length"):
        CustomRV([1, 2], [0.5])


def test_custom_rv_negative_probability():
    with pytest.raises(ValueError, match=">= 0"):
        CustomRV([1, 2], [0.6, -0.4])


def test_custom_rv_probabilities_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        CustomRV([1, 2, 3], [0.3, 0.3, 0.3])


def test_custom_rv_sampling_is_deterministic_with_seed():
    rv1 = CustomRV([10, 20], [0.5, 0.5], seed=123)
    rv2 = CustomRV([10, 20], [0.5, 0.5], seed=123)

    samples1 = rv1.rvs(size=10)
    samples2 = rv2.rvs(size=10)

    np.testing.assert_array_equal(samples1, samples2)


def test_custom_rv_sampling_with_prob_1():
    rv = CustomRV([10, 20], [0.0, 1.0], seed=42)

    samples = rv.rvs(size=10)

    assert rv.mean() == 20
    assert rv.var() == 0
    assert rv.ppf(0.5) == 20
    assert rv.ppf(0.9) == 20
    np.testing.assert_array_equal(samples, np.full(10, 20))


def test_custom_rv_sampling_mean():
    rv = CustomRV([10, 20], [0.5, 0.5], seed=42)

    assert rv.mean() == 15
    assert rv.ppf(0.49) == 10
    assert rv.ppf(0.5) == 10
    assert rv.ppf(0.51) == 20
