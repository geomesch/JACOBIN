# -*- coding: utf-8 -*-
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import gmpy2
import betanegbinfit.distributions as bd
from jacobin import Poisson, Binomial, NB, BetaBinomial, BetaNB, MCNB, CompoundDist, TruncatedDistribution

CASES = [(Poisson(), (9.3,)), (NB(), (7.5, 0.6)), (NB(False), (0.6, 0.2)), (Binomial(), (60, 0.3)), (BetaBinomial(), (60, 4.0, 9.0)),
         (BetaNB(), (7.5, 40.0, 60.0)), (MCNB(CompoundDist.NB), (7.5, 0.6, 0.3)), (MCNB(CompoundDist.Binomial), (7.5, 0.6)),
         (MCNB(CompoundDist.Poisson), (7.5, 0.6))]
# Small bounds have the mass summed from the pmf, the large ones go
# through cdf and sf of the base distribution.
BOUNDS = [(5, None), (1, None), (None, 30), (None, 80), (5, 30), (5, 5), (3, 90), (0, None), (None, None)]
N = 4000


@pytest.mark.parametrize('dist,args', CASES)
def test_pmf_sums_to_one(dist, args):
    t = TruncatedDistribution(dist)
    x = jnp.arange(float(N))
    base = np.asarray(dist.logpmf(x, *args))
    for left, right in BOUNDS:
        lp = np.asarray(t.logpmf(x, *args, left=left, right=right))
        inside = (np.arange(N) >= (left or 0)) & (np.arange(N) <= (N if right is None else right))
        assert np.all(lp[~inside] == -np.inf)
        assert np.array_equal(np.isfinite(lp), inside & np.isfinite(base))
        np.testing.assert_allclose(np.exp(lp).sum(), 1.0, rtol=1e-9 if isinstance(dist, BetaNB) else 1e-11)
        with np.errstate(invalid='ignore'):
            shift = lp[inside] - base[inside]
        np.testing.assert_allclose(shift[np.isfinite(shift)], shift[np.isfinite(shift)][0], rtol=0, atol=1e-11)
        np.testing.assert_allclose(t.pmf(x, *args, left=left, right=right), np.exp(lp), rtol=1e-14, atol=1e-300)


@pytest.mark.parametrize('dist,args', CASES)
def test_cdf_and_sf(dist, args):
    t = TruncatedDistribution(dist)
    x = jnp.arange(-2.0, 120.0)
    for left, right in ((5, None), (None, 30), (5, 30)):
        kw = dict(left=left, right=right)
        fun = jax.jit(lambda *a: tuple(getattr(t, name)(x, *a, **kw) for name in ('pmf', 'cdf', 'sf', 'logcdf', 'logsf')))
        pmf, c, s, lc, ls = map(np.asarray, fun(*args))
        cdf = np.cumsum(pmf)
        np.testing.assert_allclose(c, cdf, rtol=0, atol=2e-11)
        np.testing.assert_allclose(s, 1 - cdf, rtol=0, atol=2e-11)
        np.testing.assert_allclose(np.exp(lc), cdf, rtol=0, atol=2e-11)
        np.testing.assert_allclose(np.exp(ls), 1 - cdf, rtol=0, atol=2e-11)
        if left:
            assert c[left + 1] == 0.0 and s[left + 1] == 1.0 and ls[left + 1] == 0.0 and lc[left + 1] == -np.inf
        if right is not None:
            assert np.all(c[right + 2:] == 1.0) and np.all(s[right + 2:] == 0.0)


# Windows far in a tail hold a tiny part of the mass. It is found from
# the sf on the right and from the cdf on the left, a complement has no
# digits there. max_direct switches off the sum below a small left bound.
def test_windows_in_the_tails():
    x = jnp.arange(float(N))
    for dist, args, left, right in ((NB(), (7.5, 0.6), 70, None), (NB(), (7.5, 0.6), 70, 90), (NB(), (60.0, 0.8), 3, 40),
                                    (NB(), (60.0, 0.8), None, 40), (MCNB(CompoundDist.Binomial), (7.5, 0.6), 60, None),
                                    (MCNB(CompoundDist.Binomial), (7.5, 0.6), 60, 75), (BetaNB(), (7.5, 400.0, 600.0), 80, None),
                                    (BetaNB(), (7.5, 400.0, 600.0), 80, 100), (Poisson(), (9.3,), 40, 60), (Binomial(), (60, 0.3), 40, None)):
        t = TruncatedDistribution(dist, max_direct=0)
        kw = dict(left=left, right=right)
        lp = np.asarray(t.logpmf(x, *args, **kw))
        mass = float(np.exp(np.asarray(dist.logpmf(x, *args)))[(left or 0):(N if right is None else right + 1)].sum())
        assert mass < 1e-4
        np.testing.assert_allclose(np.exp(lp).sum(), 1.0, rtol=2e-9)
        cdf = np.cumsum(np.exp(lp))[:200]
        np.testing.assert_allclose(t.cdf(x[:200], *args, **kw), cdf, rtol=0, atol=2e-9)
        np.testing.assert_allclose(t.sf(x[:200], *args, **kw), 1 - cdf, rtol=0, atol=2e-9)


# The logarithm of the sf stays accurate where the sf itself underflows
# the precision of a complement.
def test_logsf_in_the_tail():
    for dist, args in ((NB(), (7.5, 0.6)), (MCNB(CompoundDist.Binomial), (7.5, 0.6)), (BetaNB(), (7.5, 400.0, 600.0))):
        t = TruncatedDistribution(dist)
        xs = [5, 60, 400, 1500]
        truth = np.array([float(gmpy2.log(v)) for v in t.long_sf(xs, *args, left=5)])
        got = np.asarray(t.logsf(jnp.asarray(xs, dtype=float), *args, left=5))
        assert np.all(np.abs(got - truth) <= 1e-10 + 1e-13 * np.abs(truth))


# A left bound may be a Python int, then the mass below it is a sum of
# the pmf, or a traced value, then it is a cdf. Both give one answer.
@pytest.mark.parametrize('dist,args', CASES[:3] + CASES[5:])
def test_static_and_traced_bounds_agree(dist, args):
    t = TruncatedDistribution(dist)
    x = jnp.arange(60.0)
    for left, right in ((5, None), (5, 30), (None, 30)):
        static = np.asarray(t.logpmf(x, *args, left=left, right=right))
        lo, hi = (0 if left is None else left), (jnp.inf if right is None else right)
        traced = np.asarray(jax.jit(lambda a, b: t.logpmf(x, *args, left=a, right=b))(lo, hi))
        no_sum = np.asarray(TruncatedDistribution(dist, max_direct=0).logpmf(x, *args, left=left, right=right))
        assert np.array_equal(np.isfinite(static), np.isfinite(traced))
        ok = np.isfinite(static)
        np.testing.assert_allclose(traced[ok], static[ok], rtol=0, atol=1e-10)
        np.testing.assert_allclose(no_sum[ok], static[ok], rtol=0, atol=1e-10)
        np.testing.assert_allclose(jax.jit(lambda a, b: t.sf(x, *args, left=a, right=b))(lo, hi), t.sf(x, *args, left=left, right=right), atol=1e-11)


# The left bound of the reference library is exclusive.
def test_against_betanegbinfit():
    x = jnp.arange(300.0)
    for left in (1, 5, 17):
        for r, p in ((7.5, 0.6), (0.7, 0.3), (120.0, 0.45)):
            ref = np.asarray(bd.LeftTruncatedNB.logprob(x, r, p, left - 1))
            mine = np.asarray(TruncatedDistribution(NB()).logpmf(x, r, p, left=left))
            assert np.array_equal(np.isfinite(ref), np.isfinite(mine))
            # A mass of 1e-9 is left above 17 for r = 0.7. The complement of
            # the sum below the bound keeps 6 digits of it in both libraries.
            atol = 1e-5 if left == 17 and r < 1 else 1e-10
            np.testing.assert_allclose(mine[left:], ref[left:], rtol=0, atol=atol)
            # The means of the reference are accurate to 1e-7, they divide
            # by its sf. Both means lose more digits when little mass is
            # left above the bound, r = 0.7 here.
            rtol = 1e-5
            np.testing.assert_allclose(TruncatedDistribution(NB()).mean(r, p, left=left), bd.LeftTruncatedNB.mean(r, p, left - 1), rtol=rtol)
            ref = np.asarray(bd.LeftTruncatedMCNB.logprob(x, r, p, left - 1))
            t = TruncatedDistribution(MCNB(CompoundDist.Binomial))
            mine = np.asarray(t.logpmf(x, r, p, left=left))
            # The pointwise function of the reference is wrong from some x
            # on for p below 0.39, its table is compared below.
            top = 300 if p > 0.39 else 30
            np.testing.assert_allclose(mine[left:top], ref[left:top], rtol=0, atol=atol)
            # The reference returns inf for its left of zero.
            if left > 1:
                np.testing.assert_allclose(t.mean(r, p, left=left), bd.LeftTruncatedMCNB.mean(r, p, left - 1), rtol=rtol)
            ref = np.asarray(bd.LeftTruncatedMCNB.logprob_recurrent(jnp.array([299]), jnp.array([r]), p, left - 1, 300))[0]
            np.testing.assert_allclose(np.asarray(t.logpmf_recurrent(0, 299, 300, r, p, left=left))[left:], ref[left:], rtol=0, atol=atol)
        # The reference is built on the betaln of jax, accurate to 7e-7.
        # The last set leaves a mass of 5e-8 above 17 and is left out there.
        for mu, kappa, r in ((0.6, 30.0, 7.5), (0.5, 200.0, 20.0), (0.3, 4000.0, 2.5))[:2 if left == 17 else 3]:
            a, b = (1 - mu) * kappa, mu * kappa
            ref = np.asarray(bd.LeftTruncatedBetaNB.logprob(x, mu, kappa, r, left - 1))
            mine = np.asarray(TruncatedDistribution(BetaNB()).logpmf(x, r, a, b, left=left))
            np.testing.assert_allclose(mine[left:], ref[left:], rtol=0, atol=1e-6)
            np.testing.assert_allclose(TruncatedDistribution(BetaNB()).mean(r, a, b, left=left), bd.LeftTruncatedBetaNB.mean(mu, kappa, r, left - 1), rtol=1e-6)


@pytest.mark.parametrize('dist,args', CASES)
def test_moments_by_brute_force(dist, args):
    t = TruncatedDistribution(dist)
    n = 20000
    x = np.arange(n)
    base = np.asarray(dist.pmf_recurrent(0, n - 1, n, *args))
    for left, right in ((5, None), (1, None), (None, 30), (5, 30), (5, 5), (3, 90), (None, None)):
        w = np.where((x >= (left or 0)) & (x <= (n if right is None else right)), base, 0.0)
        mean = (w * x).sum() / w.sum()
        var = (w * x ** 2).sum() / w.sum() - mean ** 2
        np.testing.assert_allclose(t.mean(*args, left=left, right=right), mean, rtol=1e-9)
        np.testing.assert_allclose(t.var(*args, left=left, right=right), var, rtol=1e-7, atol=1e-9)
    means = t.mean(*[jnp.array([v, v]) if isinstance(v, float) else v for v in args], left=5)
    np.testing.assert_allclose(means, float(t.mean(*args, left=5)), rtol=1e-13)
    with pytest.raises(TypeError):
        jax.jit(lambda a: t.mean(*args, left=a))(5)


@pytest.mark.xfail(strict=True, reason='Without a right bound the mean is the mean of the base distribution with the part '
                   'below left subtracted. All digits cancel when nearly all of the mass lies below left.')
def test_mean_with_all_mass_below_left():
    t = TruncatedDistribution(NB())
    r, p, left = 7.5, 0.6, 100
    x = np.arange(4000)
    w = np.where(x >= left, np.asarray(NB().pmf(jnp.asarray(x, dtype=float), r, p)), 0.0)
    np.testing.assert_allclose(t.mean(r, p, left=left), (w * x).sum() / w.sum(), rtol=1e-6)


@pytest.mark.parametrize('dist,args', CASES)
def test_recurrent_table(dist, args):
    t = TruncatedDistribution(dist)
    n = 120
    x = jnp.arange(float(n))
    for left, right in ((5, None), (None, 30), (5, 30), (5, 400), (None, None)):
        kw = dict(left=left, right=right)
        point = np.asarray(t.logpmf(x, *args, **kw))
        table = np.asarray(jax.jit(lambda *a: t.logpmf_recurrent(0, n - 1, n, *a, **kw))(*args))
        assert np.array_equal(np.isfinite(table), np.isfinite(point))
        ok = np.isfinite(point)
        np.testing.assert_allclose(table[ok], point[ok], rtol=0, atol=1e-10)
        assert not np.any(table == 0.0)
        part = np.asarray(t.logpmf_recurrent(7, 50, n, *args, **kw))
        assert np.all(part[:7] == -np.inf) and np.all(part[51:] == -np.inf)
        np.testing.assert_array_equal(part[7:51], np.asarray(t.logpmf_recurrent(None, None, n, *args, **kw))[7:51])
        np.testing.assert_allclose(t.pmf_recurrent(0, n - 1, n, *args, **kw), np.exp(table), rtol=1e-11)
    kept = np.asarray(TruncatedDistribution(dist, clip_bounds=False).logpmf_recurrent(0, n - 1, n, *args, left=5))
    assert np.all(np.isfinite(kept[:5]))


# A table per slice, r and the traced bounds differ between slices.
def test_recurrent_table_under_vmap():
    t = TruncatedDistribution(MCNB(CompoundDist.Binomial))
    n = 200
    rs = jnp.array([3.3, 7.5, 20.0, 55.0])
    tops = jnp.array([199, 150, 40, 199])
    fun = jax.jit(jax.vmap(lambda r, top, p: t.logpmf_recurrent(0, top, n, r, p, left=5), in_axes=(0, 0, None)))
    tables = np.asarray(fun(rs, tops, 0.6))
    assert tables.shape == (4, n)
    for row, r, top in zip(tables, rs, tops):
        point = np.asarray(t.logpmf(jnp.arange(float(n)), float(r), 0.6, left=5))
        np.testing.assert_allclose(row[5:int(top) + 1], point[5:int(top) + 1], rtol=0, atol=1e-10)
        assert np.all(row[:5] == -np.inf) and np.all(row[int(top) + 1:] == -np.inf)


@pytest.mark.parametrize('dist,args', [CASES[1], CASES[3], CASES[5], CASES[7]])
def test_sample(dist, args):
    t = TruncatedDistribution(dist)
    n = 100000
    for left, right in ((5, None), (None, 6), (4, 9)):
        kw = dict(left=left, right=right)
        x = t.sample(n, *args, seed=3, **kw)
        assert x.shape == (n,) and x.min() >= (left or 0) and (right is None or x.max() <= right)
        mean, var = float(t.mean(*args, **kw)), float(t.var(*args, **kw))
        assert abs(x.mean() - mean) < 6 * (var / n) ** 0.5
        assert abs(x.var() - var) < 0.06 * var
        np.testing.assert_array_equal(x[:50], t.sample(n, *args, seed=3, **kw)[:50])
        assert t.sample((2, 3), *args, seed=np.random.default_rng(1), **kw).shape == (2, 3)
    with pytest.raises(ValueError):
        TruncatedDistribution(NB()).sample(10, 2.0, 0.1, left=200)
