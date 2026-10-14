# -*- coding: utf-8 -*-
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy import stats
from jacobin import CompoundDist, MCNB, NB, Mixture2D, NegativeMultinomial, DirichletNegativeMultinomial, MCNegativeMultinomial

N = 400
CASES = [(NegativeMultinomial(), (3.5, 0.3, 0.25)), (DirichletNegativeMultinomial(), (3.5, 40.0, 6.0, 9.0)),
         (MCNegativeMultinomial(CompoundDist.NB), (3.5, 0.3, 0.25, 0.4)), (MCNegativeMultinomial(CompoundDist.Binomial), (3.5, 0.3, 0.25, 0.4)),
         (MCNegativeMultinomial(CompoundDist.Poisson), (3.5, 0.3, 0.25))]


def grid(n):
    return jnp.meshgrid(jnp.arange(float(n)), jnp.arange(float(n)), indexing='ij')


@pytest.mark.parametrize('dist,args', CASES)
def test_sum_and_tables(dist, args):
    x, y = grid(N)
    point = np.asarray(dist.logpmf(x, y, *args))
    np.testing.assert_allclose(np.exp(point).sum(), 1.0, rtol=1e-9)
    table = np.asarray(jax.jit(lambda *a: dist.logpmf_recurrent(None, None, None, N, N, *a))(*args))
    assert np.abs(table - point).max() <= 1e-10
    # Finite where x <= max_x, y <= max_y and x + y <= max_n, all inclusive.
    part = np.asarray(dist.logpmf_recurrent(50, 30, 60, N, N, *args))
    xi, yi = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    inside = (xi <= 50) & (yi <= 30) & (xi + yi <= 60)
    assert np.array_equal(np.isfinite(part), inside) and np.all(part[~inside] == -np.inf)
    np.testing.assert_allclose(part[inside], point[inside], rtol=0, atol=1e-10)
    # The misspelled names are kept.
    np.testing.assert_array_equal(dist.logpmf_reccurent(50, 30, 60, N, N, *args), part)
    np.testing.assert_allclose(dist.pmf_reccurent(50, 30, 60, 64, 64, *args), np.exp(part[:64, :64]), rtol=1e-12)
    np.testing.assert_allclose(dist.pmf(x[:9, :9], y[:9, :9], *args), np.exp(point[:9, :9]), rtol=1e-13)
    rect = np.asarray(dist.logpmf_recurrent(None, None, None, 7, 12, *args))
    assert rect.shape == (7, 12)
    np.testing.assert_allclose(rect, point[:7, :12], rtol=0, atol=1e-11)


@pytest.mark.parametrize('dist,args', CASES)
def test_mixture(dist, args):
    x, y = grid(N)
    mix = Mixture2D(dist)
    point = np.asarray(dist.logpmf(x, y, *args))
    direct = np.log(0.3 * np.exp(point) + 0.7 * np.exp(point.T))
    got = np.asarray(mix.logpmf(x, y, 0.3, *args))
    ok = np.isfinite(direct)
    np.testing.assert_allclose(got[ok], direct[ok], rtol=0, atol=1e-10)
    table = np.asarray(mix.logpmf_recurrent(None, None, None, N, N, 0.3, *args))
    assert np.abs(table - got).max() <= 1e-10
    np.testing.assert_allclose(np.exp(got).sum(), 1.0, rtol=1e-9)
    np.testing.assert_allclose(mix.logpmf(3.0, 7.0, 0.3, *args), got[3, 7], rtol=1e-13)
    assert 'w' in mix.params and 'w' not in dist.params and 'w' not in Mixture2D.params.keys() - {'w'}


def test_against_closed_forms():
    x, y = grid(60)
    # Negative multinomial: x + y is negative binomial, x given x + y is binomial.
    r, p1, p2 = 3.5, 0.3, 0.25
    got = np.asarray(NegativeMultinomial().logpmf(x, y, r, p1, p2))
    n = np.asarray(x + y)
    want = stats.nbinom.logpmf(n, r, 1 - p1 - p2) + stats.binom.logpmf(np.asarray(x), n, p1 / (p1 + p2))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-11)
    # The same holds for the marginalized compound versions with MCNB.
    for kind in CompoundDist:
        args = (r, p1, p2) if kind == CompoundDist.Poisson else (r, p1, p2, 0.4)
        got = np.asarray(MCNegativeMultinomial(kind).logpmf(x, y, *args))
        total = np.asarray(MCNB(kind).logpmf(n, r, p1 + p2, *args[3:]))
        np.testing.assert_allclose(got, total + stats.binom.logpmf(np.asarray(x), n, p1 / (p1 + p2)), rtol=0, atol=1e-11)
    # Dirichlet negative multinomial against its definition by gamma functions.
    from scipy.special import gammaln
    r, a0, a1, a2 = 3.5, 40.0, 6.0, 9.0
    xs, ys = np.asarray(x), np.asarray(y)
    want = (gammaln(r + n) - gammaln(r) - gammaln(xs + 1) - gammaln(ys + 1) + gammaln(a0 + a1 + a2) - gammaln(a0) - gammaln(a1) - gammaln(a2)
            + gammaln(a0 + r) + gammaln(a1 + xs) + gammaln(a2 + ys) - gammaln(a0 + a1 + a2 + r + n))
    np.testing.assert_allclose(DirichletNegativeMultinomial().logpmf(x, y, r, a0, a1, a2), want, rtol=0, atol=1e-10)


def test_parameters_and_gradients():
    assert MCNegativeMultinomial(CompoundDist.Poisson).params['p'] is None
    assert MCNegativeMultinomial().params['p'] == [(0.0, 1.0)] and MCNegativeMultinomial.params['p'] == [(0.0, 1.0)]
    d = MCNegativeMultinomial(CompoundDist.Binomial)
    table = lambda r: d.logpmf_recurrent(20, 20, 30, 32, 32, r, 0.3, 0.25, 0.4)[3, 7]
    point = lambda r: d.logpmf(3.0, 7.0, r, 0.3, 0.25, 0.4)
    g = float(jax.grad(point)(3.5))
    np.testing.assert_allclose(jax.grad(table)(3.5), g, rtol=1e-9)
    np.testing.assert_allclose(g, float(point(3.5 + 1e-6) - point(3.5 - 1e-6)) / 2e-6, rtol=1e-6)
    rows = jax.jit(jax.vmap(lambda r: NegativeMultinomial().logpmf_recurrent(None, None, None, 16, 16, r, 0.3, 0.25)))(jnp.array([1.5, 3.5]))
    x, y = grid(16)
    np.testing.assert_allclose(rows[1], NegativeMultinomial().logpmf(x, y, 3.5, 0.3, 0.25), rtol=0, atol=1e-11)
