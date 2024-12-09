import numpy as np
import pytest
from scipy.stats import poisson

from pyjobshop.simheuristic.DiscreteRV import (
    Constant,
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
