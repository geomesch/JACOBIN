# -*- coding: utf-8 -*-
import numpy as np
import jax
import jax.numpy as jnp
from jacobin import Distribution, Poisson, NB, BetaNB, MCNB, CompoundDist, TruncatedDistribution, Mixture

PARAMS = [dict(r=7.5, p=0.6), dict(r=7.5, p=0.4)]


def test_values():
    mix = Mixture(NB(), NB())
    x = jnp.arange(300.0)
    f1, f2 = (np.asarray(NB().pmf(x, **p)) for p in PARAMS)
    want = 0.7 * f1 + 0.3 * f2
    np.testing.assert_allclose(mix.logpmf(x, PARAMS, 0.7), np.log(want), rtol=0, atol=1e-11)
    np.testing.assert_allclose(mix.pmf(x, PARAMS, 0.7), want, rtol=1e-11)
    # The weights of all components, they are normalized by default.
    np.testing.assert_allclose(mix.logpmf(x, PARAMS, jnp.array([0.7, 0.3])), np.log(want), rtol=0, atol=1e-11)
    np.testing.assert_allclose(mix.logpmf(x, PARAMS, jnp.array([7.0, 3.0])), np.log(want), rtol=0, atol=1e-11)
    raw = Mixture(NB(), NB(), normalize_weights=False)
    np.testing.assert_allclose(raw.pmf(x, PARAMS, jnp.array([7.0, 3.0])), 10 * want, rtol=1e-11)
    # Any shape of x.
    grid = x[:12].reshape(3, 4)
    assert mix.logpmf(grid, PARAMS, 0.7).shape == (3, 4)
    np.testing.assert_allclose(mix.logpmf(grid, PARAMS, 0.7), np.log(want[:12]).reshape(3, 4), rtol=0, atol=1e-11)
    np.testing.assert_allclose(mix.logpmf(5.0, PARAMS, 0.7), np.log(want[5]), rtol=1e-12)
    cdf = np.cumsum(want)
    np.testing.assert_allclose(mix.cdf(x, PARAMS, 0.7), cdf, rtol=0, atol=1e-11)
    np.testing.assert_allclose(mix.sf(x, PARAMS, 0.7), 1 - cdf, rtol=0, atol=1e-11)
    np.testing.assert_allclose(np.exp(mix.logcdf(x, PARAMS, 0.7)), cdf, rtol=0, atol=1e-11)
    tail = 0.7 * np.asarray(NB().sf(x, **PARAMS[0])) + 0.3 * np.asarray(NB().sf(x, **PARAMS[1]))
    np.testing.assert_allclose(mix.logsf(x, PARAMS, 0.7), np.log(tail), rtol=0, atol=1e-10)
    table = np.asarray(jax.jit(lambda w: mix.logpmf_recurrent(0, 299, 300, PARAMS, w))(0.7))
    np.testing.assert_allclose(table, np.log(want), rtol=0, atol=1e-10)
    part = np.asarray(mix.logpmf_recurrent(3, 40, 300, PARAMS, 0.7))
    assert np.all(part[:3] == -np.inf) and np.all(part[41:] == -np.inf) and np.all(np.isfinite(part[3:41]))


def test_moments_and_samples():
    mix = Mixture(NB(), Poisson(), BetaNB())
    params = [dict(r=7.5, p=0.6), dict(rate=3.0), dict(r=4.0, a=30.0, b=12.0)]
    w = jnp.array([0.5, 0.2])
    n = 20000
    x = np.arange(n)
    pmf = np.asarray(mix.pmf_recurrent(0, n - 1, n, params, w))
    mean = (pmf * x).sum()
    np.testing.assert_allclose(pmf.sum(), 1.0, rtol=1e-9)
    np.testing.assert_allclose(mix.mean(params, w), mean, rtol=1e-8)
    np.testing.assert_allclose(mix.var(params, w), (pmf * x ** 2).sum() - mean ** 2, rtol=1e-6)
    draws = mix.sample(200000, params, w, seed=4)
    var = float(mix.var(params, w))
    assert draws.shape == (200000,) and abs(draws.mean() - mean) < 6 * (var / 200000) ** 0.5 and abs(draws.var() - var) < 0.08 * var
    np.testing.assert_array_equal(draws[:20], mix.sample(200000, params, w, seed=4)[:20])
    assert mix.sample((2, 5), params, w, seed=np.random.default_rng(4)).shape == (2, 5)


# The two modes of MixALime: a mixture of left truncated distributions.
def test_mixture_of_truncated():
    for dist, params in ((NB(), PARAMS), (MCNB(CompoundDist.Binomial), [dict(r=7.5, p1=0.6), dict(r=7.5, p1=0.4)])):
        t = TruncatedDistribution(dist)
        mix = Mixture(t, t)
        x = jnp.arange(400.0)
        f1, f2 = (np.asarray(t.pmf(x, left=5, **p)) for p in params)
        want = 0.7 * f1 + 0.3 * f2
        lp = np.asarray(mix.logpmf(x, params, 0.7, left=5))
        assert np.all(lp[:5] == -np.inf)
        np.testing.assert_allclose(np.exp(lp), want, rtol=1e-10)
        np.testing.assert_allclose(want.sum(), 1.0, rtol=1e-10)
        np.testing.assert_allclose(mix.mean(params, 0.7, left=5), (want * np.arange(400)).sum(), rtol=1e-9)
        table = np.asarray(mix.logpmf_recurrent(0, 399, 400, params, 0.7, left=5))
        np.testing.assert_allclose(table[5:], lp[5:], rtol=0, atol=1e-10)
        loss = lambda w: mix.logpmf(jnp.array([6.0, 20.0, 31.0]), params, w, left=5).sum()
        grad = float(jax.jit(jax.grad(loss))(0.7))
        np.testing.assert_allclose(grad, float(loss(0.7 + 1e-6) - loss(0.7 - 1e-6)) / 2e-6, rtol=1e-6)
        np.testing.assert_allclose(jax.jacfwd(loss)(0.7), grad, rtol=1e-10)
        draws = mix.sample(50000, params, 0.7, left=5, seed=0)
        assert draws.min() >= 5 and abs(draws.mean() - float(mix.mean(params, 0.7, left=5))) < 0.2


def test_params_are_not_shared():
    one, two = Mixture(NB(), NB()), Mixture(NB(), NB(), NB())
    assert len(one.params['weights']) == 1 and len(two.params['weights']) == 2
    assert Distribution.params == dict() and 'weights' not in NB.params
