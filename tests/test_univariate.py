# -*- coding: utf-8 -*-
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy import stats
from jacobin import Poisson, Binomial, NB, BetaBinomial, BetaNB, MCNB, CompoundDist, Distribution, reparam
from conftest import Ref, lu, logerr

N = 250


def poisson_case(rng):
    return Poisson(), Ref.poisson, (np.floor(lu(rng, 1, 5001, N)) - 1, lu(rng, 0.01, 4000, N))


def nb_case(rng):
    return NB(), Ref.nb, (np.floor(lu(rng, 1, 5001, N)) - 1, lu(rng, 0.05, 500, N), rng.uniform(0.01, 0.99, N))


def binomial_case(rng):
    n = np.floor(lu(rng, 1, 600, N))
    return Binomial(), Ref.binomial, (np.floor(rng.uniform(0, 1, N) * (n + 1)), n, rng.uniform(0.01, 0.99, N))


def betabinomial_case(rng):
    n = np.floor(lu(rng, 1, 400, N))
    return BetaBinomial(), Ref.betabinomial, (np.floor(rng.uniform(0, 1, N) * (n + 1)), n, lu(rng, 0.1, 5000, N), lu(rng, 0.1, 5000, N))


def betanb_case(rng):
    mu, kappa = rng.uniform(0.02, 0.98, N), lu(rng, 2, 1e4, N)
    return BetaNB(), Ref.betanb, (np.floor(lu(rng, 1, 3001, N)) - 1, lu(rng, 0.05, 500, N), (1 - mu) * kappa, mu * kappa)


CASES = dict(poisson=poisson_case, nb=nb_case, binomial=binomial_case, betabinomial=betabinomial_case, betanb=betanb_case)


def test_x64_is_enabled():
    assert jax.config.jax_enable_x64
    assert NB().logpmf(1, 2.0, 0.5).dtype == jnp.float64


# Random parameters, the ugly ones included: r below one and in the
# hundreds, p next to 0 and 1, concentrations from 2 to 1e4, x in the
# thousands. The error of a logarithm is the relative error of a value.
@pytest.mark.parametrize('name', CASES)
def test_against_mpmath(name, rng):
    dist, ref, args = CASES[name](rng)
    fun = jax.jit(lambda *a: (dist.logpmf(*a), dist.logcdf(*a), dist.logsf(*a), dist.cdf(*a), dist.sf(*a)))
    lp, lc, ls, c, s = map(np.asarray, fun(*args))
    truth = np.array([[float(v) for v in ref(int(a[0]), *map(float, a[1:]))] for a in zip(*args)])
    tp, tc, ts = truth.T
    assert np.all(logerr(lp, tp) <= 2e-11 + 1e-14 * np.abs(tp))
    # The smaller of cdf and sf keeps its relative accuracy.
    lower = tc < ts
    small = np.where(lower, tc, ts)
    assert np.all(logerr(np.where(lower, lc, ls), small) <= 1e-10 + 1e-14 * np.abs(np.where(np.isfinite(small), small, 0.0)))
    np.testing.assert_allclose(c, np.exp(tc), rtol=0, atol=1e-11)
    np.testing.assert_allclose(s, np.exp(ts), rtol=0, atol=1e-11)
    np.testing.assert_allclose(c + s, 1.0, rtol=0, atol=1e-14)


def test_scipy_grid_nb():
    r, p, x = np.meshgrid([0.3, 0.9, 1.0, 2.5, 10.0, 57.3, 200.0, 500.0], [0.01, 0.05, 0.3, 0.5, 0.7, 0.95, 0.99],
                          [0, 1, 2, 5, 10, 30, 100, 300, 1000, 3000], indexing='ij')
    for flag in (True, False):
        d, q = NB(p_success=flag), (1 - p if flag else p)
        np.testing.assert_allclose(d.logpmf(x, r, p), stats.nbinom.logpmf(x, r, q), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(d.cdf(x, r, p), stats.nbinom.cdf(x, r, q), rtol=1e-9, atol=1e-300)
        np.testing.assert_allclose(d.sf(x, r, p), stats.nbinom.sf(x, r, q), rtol=1e-9, atol=1e-300)
        ref = stats.nbinom.logsf(x, r, q)
        ok = np.isfinite(ref) & (ref > -600)
        np.testing.assert_allclose(np.asarray(d.logsf(x, r, p))[ok], ref[ok], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(d.mean(r, p), stats.nbinom.mean(r, q), rtol=1e-13)
        np.testing.assert_allclose(d.var(r, p), stats.nbinom.var(r, q), rtol=1e-13)


def test_scipy_grid_betanb():
    r, mu, kappa, x = np.meshgrid([1, 2, 10, 57, 200, 500], [0.05, 0.3, 0.5, 0.7, 0.95], [2.0, 5.0, 20.0, 100.0, 1e3, 1e4],
                                  [0, 1, 2, 5, 10, 30, 100, 300, 1000, 3000], indexing='ij')
    a, b = (1 - mu) * kappa, mu * kappa
    d, ref = BetaNB(), stats.betanbinom(r, a, b)
    np.testing.assert_allclose(d.logpmf(x, r, a, b), ref.logpmf(x), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(d.cdf(x, r, a, b), ref.cdf(x), rtol=0, atol=2e-9)
    np.testing.assert_allclose(d.sf(x, r, a, b), ref.sf(x), rtol=0, atol=2e-9)
    ok = a > 2.5
    np.testing.assert_allclose(np.asarray(d.mean(r, a, b))[ok], ref.mean()[ok], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(d.var(r, a, b))[ok], ref.var()[ok], rtol=1e-10)
    assert np.all(np.isinf(np.asarray(d.mean(r, a, b))[a <= 1]))


@pytest.mark.parametrize('flag', [True, False])
def test_scipy_grid_binomial(flag):
    n, p, x = np.meshgrid([1, 2, 10, 57, 400, 3000], [0.01, 0.3, 0.5, 0.95, 0.99], [0, 1, 2, 9, 10, 56, 57, 399, 2000, 3000, 3500], indexing='ij')
    d, q = Binomial(p_success=flag), (p if flag else 1 - p)
    ref = stats.binom(n, q)
    np.testing.assert_allclose(d.pmf(x, n, p), ref.pmf(x), rtol=1e-10, atol=1e-300)
    np.testing.assert_allclose(d.cdf(x, n, p), ref.cdf(x), rtol=1e-9, atol=1e-300)
    np.testing.assert_allclose(d.sf(x, n, p), ref.sf(x), rtol=1e-9, atol=1e-300)
    np.testing.assert_allclose(d.mean(n, p), ref.mean(), rtol=1e-13)
    np.testing.assert_allclose(d.var(n, p), ref.var(), rtol=1e-13)


def test_scipy_grid_betabinomial_and_poisson():
    n, a, b, x = np.meshgrid([1, 10, 57, 400], [0.3, 1.0, 4.5, 300.0], [0.4, 1.0, 7.5, 2000.0], [0, 1, 9, 10, 56, 57, 399, 400, 450], indexing='ij')
    d, ref = BetaBinomial(), stats.betabinom(n, a, b)
    np.testing.assert_allclose(d.pmf(x, n, a, b), ref.pmf(x), rtol=1e-9, atol=1e-300)
    np.testing.assert_allclose(d.cdf(x, n, a, b), ref.cdf(x), rtol=0, atol=1e-11)
    np.testing.assert_allclose(d.sf(x, n, a, b), ref.sf(x), rtol=0, atol=1e-11)
    np.testing.assert_allclose(d.mean(n, a, b), ref.mean(), rtol=1e-13)
    np.testing.assert_allclose(d.var(n, a, b), ref.var(), rtol=1e-12)
    rate, x = np.meshgrid([0.01, 0.9, 4.2, 100.0, 3000.0], [0, 1, 5, 99, 100, 101, 3000, 4000], indexing='ij')
    d, ref = Poisson(), stats.poisson(rate)
    np.testing.assert_allclose(d.logpmf(x, rate), ref.logpmf(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d.cdf(x, rate), ref.cdf(x), rtol=1e-10, atol=1e-300)
    np.testing.assert_allclose(d.sf(x, rate), ref.sf(x), rtol=1e-10, atol=1e-300)


def test_outside_of_support():
    x = jnp.array([-3.0, -1.0])
    for dist, args in ((Poisson(), (2.0,)), (NB(), (2.0, 0.4)), (Binomial(), (5, 0.4)), (BetaNB(), (2.0, 3.0, 4.0)),
                       (BetaBinomial(), (5, 3.0, 4.0)), (MCNB(CompoundDist.NB), (2.0, 0.4, 0.3))):
        assert np.all(np.asarray(dist.logpmf(x, *args)) == -np.inf)
        assert np.all(np.asarray(dist.cdf(x, *args)) == 0.0)
        assert np.all(np.asarray(dist.sf(x, *args)) == 1.0)
        assert np.all(np.asarray(dist.logsf(x, *args)) == 0.0)
        assert np.all(np.asarray(dist.logcdf(x, *args)) == -np.inf)
    x = jnp.array([5.0, 6.0, 9.0])
    for dist, args in ((Binomial(), (5, 0.4)), (BetaBinomial(), (5, 3.0, 4.0))):
        np.testing.assert_array_equal(np.isfinite(np.asarray(dist.logpmf(x, *args))), [True, False, False])
        np.testing.assert_array_equal(dist.cdf(x, *args), 1.0)
        np.testing.assert_array_equal(dist.sf(x, *args), 0.0)


def test_parameters_as_keywords():
    x = jnp.arange(6.0)
    for dist, kw in ((Poisson(), dict(rate=2.5)), (NB(), dict(r=2.5, p=0.4)), (Binomial(), dict(n=7, p=0.4)), (BetaNB(), dict(r=2.5, a=3.0, b=4.0)),
                     (BetaBinomial(), dict(n=7, a=3.0, b=4.0)), (MCNB(CompoundDist.NB), dict(r=2.5, p1=0.4, p2=0.3))):
        args = tuple(kw.values())
        for method in ('logpmf', 'pmf', 'cdf', 'sf', 'logcdf', 'logsf'):
            np.testing.assert_array_equal(getattr(dist, method)(x, **kw), getattr(dist, method)(x, *args))
        np.testing.assert_array_equal(dist.logpmf_recurrent(0, 5, 6, **kw), dist.logpmf_recurrent(0, 5, 6, *args))
        assert dist.mean(**kw) == dist.mean(*args) and dist.var(**kw) == dist.var(*args)
        assert dist.long_sf(3, **kw) == dist.long_sf(3, *args)
        np.testing.assert_array_equal(dist.sample(5, seed=1, **kw), dist.sample(5, *args, seed=1))


RECURRENT = [(Poisson(), (17.3,)), (Poisson(), (0.2,)), (NB(), (0.4, 0.95)), (NB(), (250.5, 0.05)), (NB(False), (7.5, 0.3)),
             (Binomial(), (4000, 0.3)), (Binomial(False), (37, 0.9)), (BetaNB(), (7.5, 4000.0, 6000.0)), (BetaNB(), (0.7, 1.2, 0.8)),
             (BetaBinomial(), (4000, 0.7, 150.0)), (BetaBinomial(), (37, 5.0, 2.0))]


@pytest.mark.parametrize('dist,args', RECURRENT)
def test_recurrent_table_matches_pointwise(dist, args):
    n = 3001
    table = np.asarray(jax.jit(lambda *a: dist.logpmf_recurrent(0, n - 1, n, *a))(*args))
    point = np.asarray(dist.logpmf(jnp.arange(float(n)), *args))
    assert np.array_equal(np.isfinite(table), np.isfinite(point))
    ok = np.isfinite(point)
    assert np.all(np.abs(table[ok] - point[ok]) <= 5e-12 + 1e-13 * np.abs(point[ok]))
    assert not np.any(table == 0.0)


# Entries outside of [min_x, max_x] are -inf, both bounds are inclusive
# and may be traced.
def test_recurrent_bounds():
    d, args = NB(), (3.5, 0.6)
    full = np.asarray(d.logpmf(jnp.arange(12.0), *args))
    for lo, hi in ((0, 11), (0, 6), (3, 6), (5, 5), (0, 40), (None, None)):
        table = np.asarray(d.logpmf_recurrent(lo, hi, 12, *args))
        traced = np.asarray(jax.jit(lambda a, b: d.logpmf_recurrent(a, b, 12, *args))(0 if lo is None else lo, 11 if hi is None else hi))
        inds = np.arange(12)
        inside = (inds >= (lo or 0)) & (inds <= (11 if hi is None else hi))
        np.testing.assert_allclose(table[inside], full[inside], rtol=1e-13)
        assert np.all(table[~inside] == -np.inf)
        np.testing.assert_allclose(table, traced, rtol=1e-14)
    np.testing.assert_allclose(d.pmf_recurrent(0, 11, 12, *args), np.exp(full), rtol=1e-13)
    rows = jax.vmap(lambda r, top: d.logpmf_recurrent(0, top, 12, r, 0.6))(jnp.array([1.5, 3.5]), jnp.array([4, 9]))
    assert np.all(np.asarray(rows)[0, 5:] == -np.inf) and np.all(np.isfinite(np.asarray(rows)[1, :10]))
    np.testing.assert_allclose(rows[1, :10], full[:10], rtol=1e-13)


@pytest.mark.parametrize('dist,args', RECURRENT[:5] + [(BetaNB(), (7.5, 40.0, 60.0)), (BetaBinomial(), (37, 5.0, 2.0)), (Binomial(), (300, 0.3))])
def test_moments_against_table(dist, args):
    n = 40000
    pmf = np.asarray(dist.pmf_recurrent(0, n - 1, n, *args))
    x = np.arange(n)
    mean = (pmf * x).sum()
    np.testing.assert_allclose(pmf.sum(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(dist.mean(*args), mean, rtol=1e-9)
    np.testing.assert_allclose(dist.var(*args), (pmf * x ** 2).sum() - mean ** 2, rtol=1e-7)


SAMPLES = [(Poisson(), (4.2,)), (NB(), (3.5, 0.6)), (NB(False), (3.5, 0.6)), (Binomial(), (30, 0.3)), (Binomial(False), (30, 0.3)),
           (BetaNB(), (3.5, 12.0, 5.0)), (BetaBinomial(), (30, 2.0, 5.0)), (MCNB(CompoundDist.NB), (3.5, 0.6, 0.3)),
           (MCNB(CompoundDist.Binomial), (3.5, 0.6, 0.3)), (MCNB(CompoundDist.Poisson), (3.5, 0.6))]


@pytest.mark.parametrize('dist,args', SAMPLES)
def test_sample_moments(dist, args):
    n = 200000
    x = dist.sample(n, *args, seed=5)
    mean, var = float(dist.mean(*args)), float(dist.var(*args))
    assert x.shape == (n,) and x.min() >= 0
    assert abs(x.mean() - mean) < 6 * (var / n) ** 0.5
    assert abs(x.var() - var) < 0.05 * var
    np.testing.assert_array_equal(x[:100], dist.sample(100, *args, seed=5)[:100]) if isinstance(dist, MCNB) else None
    np.testing.assert_array_equal(dist.sample(50, *args, seed=7), dist.sample(50, *args, seed=np.random.default_rng(7)))
    assert dist.sample((3, 4), *args, seed=1).shape == (3, 4)
    freq = np.bincount(x, minlength=8)[:8] / n
    pmf = np.asarray(dist.pmf(jnp.arange(8.0), *args))
    assert np.all(np.abs(freq - pmf) < 6 * (pmf * (1 - pmf) / n) ** 0.5 + 1e-4)


def test_params_are_not_shared():
    assert Distribution.params == dict()
    poisson, nb = MCNB(CompoundDist.Poisson), MCNB(CompoundDist.NB)
    assert poisson.params['p2'] is None
    assert nb.params['p2'] == [(0.0, 1.0)] and MCNB.params['p2'] == [(0.0, 1.0)]


def test_reparam():
    a, b = reparam.beta(0.3, 0.01)
    np.testing.assert_allclose(stats.beta.stats(a, b), (0.3, 0.01), rtol=1e-13)
    for flag in (True, False):
        r, p = reparam.nbinom(4.0, 10.0, p_success=flag)
        np.testing.assert_allclose((NB(flag).mean(r, p), NB(flag).var(r, p)), (4.0, 10.0), rtol=1e-13)
        n, p = reparam.binomial(4.0, 3.0, p_success=flag)
        np.testing.assert_allclose((Binomial(flag).mean(n, p), Binomial(flag).var(n, p)), (4.0, 3.0), rtol=1e-13)


# A draw per element of array parameters, the shape is given by size.
def test_sample_with_array_parameters():
    r = np.array([[0.7, 3.5, 40.0], [2.0, 9.0, 150.0]])
    p = np.array([0.3, 0.5, 0.6])
    a, b = np.array([30.0, 12.0, 400.0]), np.array([12.0, 20.0, 450.0])
    for dist, args in ((NB(), (r, p)), (NB(False), (r, p)), (BetaNB(), (r, a, b)), (Poisson(), (r,))):
        args = np.broadcast_arrays(*args)
        x = dist.sample(r.shape, *args, seed=1)
        assert x.shape == r.shape
        np.testing.assert_array_equal(x, dist.sample(r.shape, *args, seed=np.random.default_rng(1)))
        flat = dist.sample(r.size, *(t.ravel() for t in args), seed=1)
        assert flat.shape == (r.size,)
        draws = dist.sample((40000,) + r.shape, *args, seed=2)
        mean, var = np.asarray(dist.mean(*args)), np.asarray(dist.var(*args))
        assert np.all(np.abs(draws.mean(axis=0) - mean) < 6 * (var / 40000) ** 0.5)
        assert np.all(np.abs(draws.var(axis=0) - var) < 0.15 * var)


# betaln at the edges of its branches against mpmath. Equal arguments
# check the second derivatives at a tie of the two.
def test_betaln():
    import mpmath as mp
    from jacobin.utils import betaln
    edge = [1e-8, 1e-3, 0.5, 1.0, 2.0, 3.0, 9.0, 10 - 1e-9, float(np.nextafter(10, 0)), 10.0, float(np.nextafter(10, 11)), 10 + 1e-9, 20.0, 1e3, 1e6]
    a, b = (t.ravel() for t in np.meshgrid(edge, edge))
    with mp.workdps(60):
        ref = lambda u, v: mp.loggamma(mp.mpf(u)) + mp.loggamma(mp.mpf(v)) - mp.loggamma(mp.mpf(u) + mp.mpf(v))
        truth = np.array([float(ref(u, v)) for u, v in zip(a, b)])
        trig = lambda u: float(mp.polygamma(1, mp.mpf(u)))
        hess = np.array([[[trig(u) - trig(u + v), -trig(u + v)], [-trig(u + v), trig(v) - trig(u + v)]] for u, v in zip(a, b)])
        digamma = np.array([[float(mp.digamma(mp.mpf(u)) - mp.digamma(mp.mpf(u) + mp.mpf(v))) for u, v in zip(a, b)],
                            [float(mp.digamma(mp.mpf(v)) - mp.digamma(mp.mpf(u) + mp.mpf(v))) for u, v in zip(a, b)]])
    got = np.asarray(jax.jit(betaln)(a, b))
    assert np.all(np.abs(got - truth) <= 5e-15 * (1 + np.abs(truth)))
    grad = np.asarray(jax.jit(jax.vmap(jax.grad(betaln, argnums=(0, 1))))(a, b))
    np.testing.assert_allclose(grad, digamma, rtol=1e-12, atol=1e-13)
    got = np.asarray(jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(lambda v: betaln(v[0], v[1])))))(np.stack([a, b], axis=1)))
    assert np.all(np.abs(got - hess) <= 1e-11 * np.abs(hess).max(axis=(1, 2), keepdims=True))


# Equal arguments of betaln inside the pmf: r = x + 1 for NB and a = b,
# that is a mean of one half, for BetaNB.
def test_hessian_at_equal_arguments():
    x = jnp.array([2.0, 7.0])
    for fun, v in ((lambda v: NB().logpmf(x, v[0], v[1]).sum(), (3.0, 0.5)), (lambda v: BetaNB().logpmf(x, v[0], v[1], v[2]).sum(), (3.0, 20.0, 20.0))):
        v = jnp.asarray(v)
        hess = np.asarray(jax.jacfwd(jax.jacfwd(fun))(v))
        grad = jax.jit(jax.jacfwd(fun))
        num = np.stack([(np.asarray(grad(v.at[i].add(1e-5))) - np.asarray(grad(v.at[i].add(-1e-5)))) / 2e-5 for i in range(len(v))], axis=-1)
        np.testing.assert_allclose(hess, num, rtol=1e-6, atol=1e-8)
