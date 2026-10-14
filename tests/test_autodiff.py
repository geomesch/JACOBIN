# -*- coding: utf-8 -*-
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jacobin import Poisson, Binomial, NB, BetaBinomial, BetaNB, MCNB, CompoundDist, TruncatedDistribution

X = jnp.array([0.0, 1.0, 5.0, 6.0, 23.0, 150.0])
# Distribution, differentiable parameters and the ones kept fixed in front.
CASES = [(Poisson(), (), (9.3,)), (NB(), (), (7.5, 0.6)), (NB(), (), (0.4, 0.97)), (NB(False), (), (250.5, 0.9)),
         (Binomial(), (200,), (0.3,)), (BetaBinomial(), (200,), (4.0, 9.0)), (BetaNB(), (), (7.5, 40.0, 60.0)),
         (BetaNB(), (), (0.7, 1.5, 0.8)), (BetaNB(), (), (30.0, 3000.0, 2000.0)), (MCNB(CompoundDist.NB), (), (7.5, 0.6, 0.3)),
         (MCNB(CompoundDist.Binomial), (), (7.5, 0.6)), (MCNB(CompoundDist.Binomial), (), (0.6, 0.2)),
         (MCNB(CompoundDist.Binomial), (), (120.0, 0.45)), (MCNB(CompoundDist.Poisson), (), (7.5, 0.6))]


# Central differences of a scalar function of a vector.
def numeric_grad(fun, v, h=1e-6):
    v = np.asarray(v, dtype=float)
    res = list()
    for i in range(len(v)):
        step = np.zeros(len(v))
        step[i] = h * max(1.0, abs(v[i]))
        res.append((np.asarray(fun(jnp.asarray(v + step))) - np.asarray(fun(jnp.asarray(v - step)))) / (2 * step[i]))
    return np.stack(res, axis=-1)


# Reverse and forward mode, the Hessian by forward mode applied twice,
# all under jit, against central differences. The differences carry
# the rounding of the function divided by the step, noise below.
def check(fun, v, rtol=2e-6, hessian=True):
    v = jnp.asarray(v, dtype=float)
    val = jax.jit(fun)(v)
    assert np.isfinite(float(val))
    fwd = np.asarray(jax.jit(jax.jacfwd(fun))(v))
    rev = np.asarray(jax.jit(jax.grad(fun))(v))
    num = numeric_grad(fun, v)
    scale = np.abs(num).max() + 1e-6
    noise = 3e-10 * (1 + abs(float(val)))
    assert np.all(np.isfinite(fwd)) and np.all(np.isfinite(rev))
    np.testing.assert_allclose(rev, fwd, rtol=1e-9, atol=1e-11 * scale)
    np.testing.assert_allclose(fwd, num, rtol=rtol, atol=rtol * scale + noise)
    if hessian:
        hess = np.asarray(jax.jit(jax.jacfwd(jax.jacfwd(fun)))(v))
        num = numeric_grad(jax.jit(jax.jacfwd(fun)), v)
        scale = np.abs(num).max() + 1e-6
        noise = 3e-10 * (1 + np.abs(fwd).max())
        assert np.all(np.isfinite(hess))
        np.testing.assert_allclose(hess, num, rtol=20 * rtol, atol=20 * rtol * scale + noise)
        np.testing.assert_allclose(hess, hess.T, rtol=1e-8, atol=1e-10 * scale)


@pytest.mark.parametrize('dist,fixed,params', CASES)
def test_logpmf(dist, fixed, params):
    check(lambda v: dist.logpmf(X, *fixed, *v).sum(), params)


@pytest.mark.parametrize('dist,fixed,params', CASES)
def test_logcdf_and_logsf(dist, fixed, params):
    x = jnp.array([1.0, 6.0, 23.0])
    w = jnp.array([1.0, 2.0, 3.0])
    check(lambda v: (dist.logcdf(x, *fixed, *v) * w).sum() + (dist.logsf(x, *fixed, *v) * w[::-1]).sum() * 1.5, params, hessian=False)
    check(lambda v: (dist.cdf(x, *fixed, *v) * w).sum() + (dist.sf(x, *fixed, *v) * w).sum() * 2.5, params, hessian=False)


@pytest.mark.parametrize('dist,fixed,params', CASES)
def test_moments(dist, fixed, params):
    if isinstance(dist, BetaNB) and params[1] < 3:
        return
    check(lambda v: dist.mean(*fixed, *v), params, hessian=False)
    check(lambda v: dist.var(*fixed, *v), params, hessian=False)


# The parameters far from the comfortable region still give finite
# first and second derivatives.
def test_derivatives_are_finite_in_corners():
    x = jnp.array([0.0, 1.0, 40.0, 3000.0])
    for dist, params in ((NB(), (0.05, 0.999)), (NB(), (500.0, 0.01)), (NB(), (1e-3, 0.5)), (BetaNB(), (0.05, 1.0, 1.0)),
                         (BetaNB(), (500.0, 9990.0, 10.0)), (BetaNB(), (3.0, 1.2, 8000.0)), (MCNB(CompoundDist.Binomial), (0.5, 0.05)),
                         (MCNB(CompoundDist.Binomial), (500.0, 0.95)), (MCNB(CompoundDist.Binomial), (0.05, 0.99))):
        fun = lambda v: dist.logpmf(x, *v).sum()
        v = jnp.asarray(params)
        assert np.all(np.isfinite(np.asarray(jax.jit(jax.grad(fun))(v))))
        assert np.all(np.isfinite(np.asarray(jax.jit(jax.jacfwd(jax.jacfwd(fun)))(v))))
        assert np.all(np.isfinite(np.asarray(jax.jit(jax.grad(lambda v: dist.logsf(x, *v).sum()))(v))))


TRUNCATED = [(NB(), (), (7.5, 0.6)), (BetaNB(), (), (7.5, 40.0, 60.0)), (MCNB(CompoundDist.Binomial), (), (7.5, 0.6)),
             (Binomial(), (200,), (0.3,)), (BetaBinomial(), (200,), (4.0, 9.0))]


# What a fit of MixALime differentiates: the log-likelihood of a left
# truncated distribution.
@pytest.mark.parametrize('dist,fixed,params', TRUNCATED)
def test_left_truncated(dist, fixed, params):
    t = TruncatedDistribution(dist)
    x = jnp.array([5.0, 6.0, 23.0, 150.0])
    check(lambda v: t.logpmf(x, *fixed, *v, left=5).sum(), params)
    check(lambda v: t.mean(*fixed, *v, left=5), params, hessian=False)
    check(lambda v: t.logsf(x[:3], *fixed, *v, left=5).sum(), params, hessian=False)
    if isinstance(dist, (NB, MCNB)):
        # The normalizer through the sf of the base distribution.
        far = TruncatedDistribution(dist, max_direct=0)
        check(lambda v: far.logpmf(x, *fixed, *v, left=5).sum(), params, hessian=False)
    if isinstance(dist, NB):
        # A bound on each side.
        check(lambda v: t.logpmf(x[:3], *fixed, *v, left=5, right=90).sum(), params, hessian=False)


# Tables for several slices at once. r is linear in the count of the
# other allele, the weights stand for the counts of the observations.
@pytest.mark.parametrize('dist,extra', [(NB(), (0.6,)), (BetaNB(), (40.0, 60.0)), (MCNB(CompoundDist.Binomial), (0.6,))])
def test_tables_of_slices(dist, extra, rng):
    t = TruncatedDistribution(dist)
    n, left = 150, 5
    slices = jnp.array([5.0, 9.0, 30.0, 61.0])
    weights = jnp.asarray(rng.integers(0, 4, (4, n)) * (np.arange(n) >= left), dtype=float)
    x = jnp.arange(float(n))

    def table_loss(v):
        r = v[0] * slices + v[1]
        tables = jax.vmap(lambda r: t.logpmf_recurrent(0, n - 1, n, r, *v[2:], left=left))(r)
        return jnp.sum(jnp.where(weights > 0, tables, 0.0) * weights)

    def point_loss(v):
        r = (v[0] * slices + v[1])[:, None]
        lp = t.logpmf(x, r, *v[2:], left=left)
        return jnp.sum(jnp.where(weights > 0, lp, 0.0) * weights)

    v = jnp.asarray((1.1, 0.7) + extra)
    np.testing.assert_allclose(jax.jit(table_loss)(v), point_loss(v), rtol=1e-12)
    grad = np.asarray(jax.jit(jax.jacfwd(table_loss))(v))
    np.testing.assert_allclose(grad, jax.jit(jax.grad(table_loss))(v), rtol=1e-9)
    np.testing.assert_allclose(grad, jax.jit(jax.jacfwd(point_loss))(v), rtol=1e-8)
    hess = np.asarray(jax.jit(jax.jacfwd(jax.jacfwd(table_loss)))(v))
    np.testing.assert_allclose(hess, jax.jit(jax.jacfwd(jax.jacfwd(point_loss)))(v), rtol=1e-7, atol=1e-7 * np.abs(hess).max())
    np.testing.assert_allclose(grad, numeric_grad(table_loss, v), rtol=2e-6)


def test_vmap_over_parameters():
    x = jnp.arange(40.0)
    r, p = jnp.array([0.5, 3.0, 40.0]), jnp.array([0.2, 0.5, 0.9])
    for dist in (NB(), MCNB(CompoundDist.Binomial)):
        rows = jax.jit(jax.vmap(lambda r, p: dist.logpmf(x, r, p)))(r, p)
        np.testing.assert_allclose(rows, dist.logpmf(x, r[:, None], p[:, None]), rtol=1e-13)
        rows = jax.jit(jax.vmap(lambda r, p: dist.logsf(x, r, p)))(r, p)
        np.testing.assert_allclose(rows, dist.logsf(x, r[:, None], p[:, None]), rtol=1e-13)
        grads = jax.vmap(jax.grad(lambda r, p: dist.logpmf(x, r, p).sum()))(r, p)
        assert grads.shape == (3,) and np.all(np.isfinite(np.asarray(grads)))


def _count_loops(jaxpr) -> int:
    n = 0
    for eqn in jaxpr.eqns:
        n += eqn.primitive.name == 'while'
        for p in eqn.params.values():
            for sub in (p if isinstance(p, (list, tuple)) else [p]):
                if hasattr(sub, 'eqns'):
                    n += _count_loops(sub)
                elif hasattr(sub, 'jaxpr'):
                    n += _count_loops(sub.jaxpr if hasattr(sub.jaxpr, 'eqns') else sub.jaxpr.jaxpr)
    return n


# A derivative of any order of a series takes the value and the partials
# from a single loop, so the compiled program stays small.
def test_one_loop_at_every_order():
    from jacobin import hyp
    series = hyp.series_logsum(hyp.bnb_upper_ratio, hyp.bnb_upper_bound)
    fun = lambda v: series(10.0, 4.0, v[0], v[1], v[2])
    v = jnp.array([3.0, 20.0, 5.0])
    for f in (fun, jax.jacfwd(fun), jax.grad(fun), jax.jacfwd(jax.jacfwd(fun)), jax.jacfwd(jax.jacfwd(jax.jacfwd(fun)))):
        assert _count_loops(jax.make_jaxpr(f)(v).jaxpr) == 1


# Derivatives of betaln up to the second order over twelve decades of both
# arguments, against mpmath. The differences of digamma and trigamma are
# where the plain formulas lose digits, at b much smaller than a.
def test_betaln_derivatives_against_mpmath():
    import mpmath as mp
    from jacobin.utils import betaln
    rng = np.random.default_rng(0)
    a, b = 10 ** rng.uniform(-6, 6, (2, 200))
    a, b = np.append(a, [40.0, 1e4, 10.0, 2.1e3]), np.append(b, [4e-8, 2.5, 1e-6, 1.1e-4])
    with mp.workdps(50):
        psi = lambda k, u: mp.digamma(u) if k == 0 else mp.polygamma(k, u)
        want = np.array([[float(psi(k, mp.mpf(u)) - psi(k, mp.mpf(u) + mp.mpf(v))) for u, v in zip(a, b)] for k in (0, 1)]
                        + [[float(psi(0, mp.mpf(v)) - psi(0, mp.mpf(u) + mp.mpf(v))) for u, v in zip(a, b)],
                           [float(-psi(1, mp.mpf(u) + mp.mpf(v))) for u, v in zip(a, b)]])
    grad = np.asarray(jax.jit(jax.vmap(jax.grad(betaln, argnums=(0, 1))))(a, b))
    hess = np.asarray(jax.jit(jax.vmap(jax.hessian(betaln, argnums=(0, 1))))(a, b))
    np.testing.assert_allclose(grad[0], want[0], rtol=1e-14)
    np.testing.assert_allclose(grad[1], want[2], rtol=1e-14)
    np.testing.assert_allclose(hess[0][0], want[1], rtol=1e-13)
    np.testing.assert_allclose(hess[0][1], want[3], rtol=1e-13)
