# -*- coding: utf-8 -*-
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import betanegbinfit.distributions as bd
from jacobin import MCNB, CompoundDist
from conftest import Ref, logs, logerr, mplog

KINDS = [CompoundDist.NB, CompoundDist.Binomial, CompoundDist.Poisson]
R_GRID = [0.5, 1.5, 3.0, 10.5, 47.9, 250.25, 500.0]
P_GRID = [0.05, 0.25, 0.4, 0.5, 0.75, 0.95]
N = 3001


def args_of(kind, r, p1, p2):
    return (r, p1) if kind == CompoundDist.Poisson else (r, p1, p2)


# The pmf of the paper in the long algebra and in floats against the
# formula with 2F1 evaluated by mpmath.
def test_paper_equation():
    m = MCNB(CompoundDist.Binomial)
    xs = [0, 1, 2, 7, 50, 400, 3000]
    for r in (0.5, 1.0, 2.5, 17.3, 100.0, 499.7):
        for p in (0.05, 0.3, 0.5, 0.8, 0.95):
            truth = np.array([float(Ref.mcnb_paper(x, r, p)) for x in xs])
            assert np.all(logerr(logs(m.long_pmf(xs, r, p)), truth) <= 1e-13 * (1 + np.abs(truth)))
            point = np.asarray(m.logpmf(jnp.array(xs, dtype=float), r, p, 1 - p))
            assert np.all(logerr(point, truth) <= 2e-11 + 1e-14 * np.abs(truth))


# All x from 0 to 3000 over the whole range of r and p. The truth is the
# long pmf, checked against the paper above.
def test_binomial_kind_on_full_grid():
    m = MCNB(CompoundDist.Binomial)
    x = jnp.arange(float(N))
    point = jax.jit(lambda r, p: m.logpmf(x, r, p, 1 - p))
    table = jax.jit(lambda r, p: m.logpmf_recurrent(0, N - 1, N, r, p))
    worst_point = worst_table = 0.0
    for r in R_GRID:
        for p in P_GRID:
            truth = logs(m.long_pmf(np.arange(N), r, p))
            worst_point = max(worst_point, logerr(np.asarray(point(r, p)), truth).max())
            worst_table = max(worst_table, logerr(np.asarray(table(r, p)), truth).max())
    assert worst_point <= 2e-11
    assert worst_table <= 1e-10


# The reference library on the same grid. Its table is right everywhere.
# Its pointwise function loses all digits at some points: for a
# non-integer r with p below 0.39 from some x on, and at single points
# elsewhere. Those points are told apart with the long pmf.
def test_against_betanegbinfit():
    m = MCNB(CompoundDist.Binomial)
    x = jnp.arange(float(N))
    point = jax.jit(lambda r, p: m.logpmf(x, r, p))
    table = jax.jit(lambda r, p: m.logpmf_recurrent(0, N - 1, N, r, p))
    broken = total = 0
    for r in R_GRID:
        for p in P_GRID:
            ref_table = np.asarray(bd.MCNB.logprob_recurrent(jnp.array([N - 1]), jnp.array([r]), p, N))[0]
            assert logerr(np.asarray(table(r, p)), ref_table).max() <= 2e-10
            if p < 0.39:
                continue
            ref = np.asarray(bd.MCNB.logprob(x, r, p))
            mine = np.asarray(point(r, p))
            truth = logs(m.long_pmf(np.arange(N), r, p))
            wrong = ~(logerr(ref, truth) <= 1e-9)
            assert np.all(logerr(mine, ref)[~wrong] <= 1e-9)
            assert np.all(logerr(mine, truth)[wrong] <= 2e-11)
            broken += wrong.sum()
            total += N
    assert broken <= 0.001 * total


def test_betanegbinfit_pointwise_breaks_down():
    m = MCNB(CompoundDist.Binomial)
    r, p, x = 10.5, 0.25, 400
    truth = logs(m.long_pmf(x, r, p))[0]
    assert abs(float(bd.MCNB.logprob(float(x), r, p)) - truth) > 1.0
    assert abs(float(m.logpmf(x, r, p)) - truth) <= 2e-11
    assert abs(float(Ref.mcnb_paper(x, r, p)) - truth) <= 1e-12


# Each kind against the sum over the hidden count, its definition.
@pytest.mark.parametrize('kind', KINDS)
def test_kinds_against_definition(kind):
    m = MCNB(kind)
    xs = [0, 1, 2, 5, 17, 60]
    for r in ((1, 3, 12) if kind == CompoundDist.Binomial else (0.7, 3.0, 12.5)):
        for p1, p2 in ((0.3, 0.6), (0.8, 0.2), (0.5, 0.5)):
            args = args_of(kind, float(r), p1, p2)
            truth = np.array([float(Ref.mcnb_direct(x, r, p1, p2, kind.value)) for x in xs])
            assert np.all(logerr(logs(m.long_pmf(xs, *args)), truth) <= 1e-12)
            assert np.all(logerr(np.asarray(m.logpmf(jnp.array(xs, dtype=float), *args)), truth) <= 1e-11)
            assert np.all(logerr(np.asarray(m.logpmf_recurrent(0, 60, 61, *args))[xs], truth) <= 1e-11)


@pytest.mark.parametrize('kind', KINDS)
def test_cdf_and_sf(kind, rng):
    m = MCNB(kind)
    fun = jax.jit(lambda x, *a: (m.logcdf(x, *a), m.logsf(x, *a), m.cdf(x, *a), m.sf(x, *a)))
    worst = 0.0
    for _ in range(12):
        r = float(np.exp(rng.uniform(np.log(0.3), np.log(40.0 if kind == CompoundDist.Poisson else 300.0))))
        args = args_of(kind, r, float(rng.uniform(0.03, 0.97)), float(rng.uniform(0.03, 0.97)))
        top = int(float(m.mean(*args)) + 12 * float(m.var(*args)) ** 0.5) + 10
        xs = np.unique(np.concatenate([np.arange(4), rng.integers(0, top, 40)]))
        tc, ts = logs(m.long_cdf(xs, *args)), logs(m.long_sf(xs, *args))
        lc, ls, c, s = map(np.asarray, fun(jnp.asarray(xs, dtype=float), *args))
        worst = max(worst, logerr(lc, tc).max(), logerr(ls, ts).max())
        np.testing.assert_allclose(c, np.exp(tc), rtol=0, atol=1e-12)
        np.testing.assert_allclose(s, np.exp(ts), rtol=0, atol=1e-12)
        np.testing.assert_allclose(c + s, 1.0, rtol=1e-12)
    assert worst <= 1e-10


# The sf keeps its relative accuracy far in the tail.
def test_sf_deep_tail():
    m = MCNB(CompoundDist.Binomial)
    for r, p in ((7.5, 0.5), (0.6, 0.9), (120.0, 0.2)):
        xs = np.array([10, 200, 1500, 4000])
        truth = logs(m.long_sf(xs, r, p))
        got = np.asarray(m.logsf(jnp.asarray(xs, dtype=float), r, p))
        assert truth.min() < -400
        assert np.all(logerr(got, truth) <= 1e-10 + 1e-13 * np.abs(truth))


@pytest.mark.parametrize('kind', KINDS)
def test_moments(kind):
    m = MCNB(kind)
    n = 20000
    for r, p1, p2 in ((0.6, 0.3, 0.7), (3.5, 0.8, 0.4), (40.0, 0.5, 0.5)):
        args = args_of(kind, r, p1, p2)
        pmf = np.asarray(m.pmf_recurrent(0, n - 1, n, *args))
        x = np.arange(n)
        mean = (pmf * x).sum()
        np.testing.assert_allclose(pmf.sum(), 1.0, rtol=1e-12)
        np.testing.assert_allclose(m.mean(*args), mean, rtol=1e-11)
        np.testing.assert_allclose(m.var(*args), (pmf * x ** 2).sum() - mean ** 2, rtol=1e-9)
    # The mean of the paper.
    r, p = 7.5, 0.35
    np.testing.assert_allclose(MCNB(CompoundDist.Binomial).mean(r, p), r * p / (1 - p ** r), rtol=1e-13)


def test_second_probability():
    x = jnp.arange(30.0)
    m = MCNB(CompoundDist.Binomial)
    np.testing.assert_array_equal(m.logpmf(x, 3.5, 0.3), m.logpmf(x, 3.5, 0.3, 0.7))
    np.testing.assert_array_equal(m.logpmf(x, r=3.5, p1=0.3), m.logpmf(x, 3.5, 0.3, 0.7))
    # The default complement is taken in the long algebra, it is not the float 0.7.
    assert abs(m.long_sf(40, 3.5, 0.3) / m.long_sf(40, 3.5, 0.3, 0.7) - 1) < 1e-14
    assert not np.allclose(m.logpmf(x, 3.5, 0.3), m.logpmf(x, 3.5, 0.3, 0.5))
    with pytest.raises(TypeError):
        MCNB(CompoundDist.NB).logpmf(x, 3.5, 0.3)
    m = MCNB(CompoundDist.Poisson)
    np.testing.assert_array_equal(m.logpmf(x, 3.5, 0.3), m.logpmf(x, 3.5, 0.3, 0.9))


# A table per slice, the parameter r differs between slices as it does
# in MixALime.
@pytest.mark.parametrize('kind', KINDS)
def test_tables_under_vmap(kind):
    m = MCNB(kind)
    n = 400
    rs = jnp.array([0.5, 2.0, 7.3, 31.0])
    fun = jax.jit(jax.vmap(lambda r, p: m.logpmf_recurrent(0, None, n, *args_of(kind, r, p, 0.35)), in_axes=(0, None)))
    tables = np.asarray(fun(rs, 0.6))
    assert tables.shape == (4, n)
    point = np.asarray(m.logpmf(jnp.arange(float(n)), *args_of(kind, rs[:, None], 0.6, 0.35)))
    assert np.abs(tables - point).max() <= 1e-11
    loss = lambda p: fun(rs, p)[:, 5:200].sum()
    grad = float(jax.grad(loss)(0.6))
    h = 1e-6
    assert abs(grad - float(loss(0.6 + h) - loss(0.6 - h)) / (2 * h)) <= 1e-6 * abs(grad)
