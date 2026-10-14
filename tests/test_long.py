# -*- coding: utf-8 -*-
import numpy as np
import jax.numpy as jnp
import mpmath as mp
import gmpy2
import pytest
from jacobin import Poisson, Binomial, NB, BetaBinomial, BetaNB, MCNB, CompoundDist, TruncatedDistribution, Mixture
from conftest import Ref, logs, logerr

# Distribution, parameters, reference and points that reach far into
# the right tail.
DEEP = [(Poisson(), (17.3,), Ref.poisson, [0, 3, 17, 400, 3000]),
        (NB(), (7.5, 0.5), Ref.nb, [0, 3, 17, 1000, 3000]),
        (NB(), (0.3, 0.97), Ref.nb, [0, 40, 5000]),
        (NB(False), (250.5, 0.9), lambda x, r, p: Ref.nb(x, r, 1 - mp.mpf(p)), [0, 30, 900]),
        (Binomial(), (4000, 0.01), Ref.binomial, [0, 40, 700, 3999, 4000]),
        (BetaBinomial(), (300, 0.7, 150.0), Ref.betabinomial, [0, 2, 150, 299, 300]),
        (BetaNB(), (5.0, 3000.0, 2000.0), Ref.betanb, [0, 3, 400, 4000]),
        (BetaNB(), (0.7, 1.2, 0.8), Ref.betanb, [0, 3, 400, 2000])]


@pytest.mark.parametrize('dist,args,ref,xs', DEEP)
def test_against_mpmath(dist, args, ref, xs):
    truth = np.array([[float(v) for v in ref(x, *args)] for x in xs])
    got = [logs(getattr(dist, name)(xs, *args)) for name in ('long_pmf', 'long_cdf', 'long_sf')]
    for g, t in zip(got, truth.T):
        assert np.all(logerr(g, t) <= 1e-13 * (1 + np.abs(t)))


# All digits of a p-value near 1e-300 and far below it.
def test_digits_of_small_p_values():
    r, p = 7.5, 0.5
    with mp.workdps(700):
        truth = [mp.betainc(x + 1, mp.mpf(r), 0, mp.mpf(p), regularized=True) for x in (1000, 5000)]
        sf = NB().long_sf([1000, 5000], r, p)
        assert truth[0] < mp.mpf(10) ** -280 and truth[1] < mp.mpf(10) ** -1400
        for got, t in zip(sf, truth):
            assert abs(mp.mpf(str(got)) / t - 1) < mp.mpf(10) ** -30
    # The tail of the beta negative binomial distribution is a complement,
    # the working precision is raised until digits are left in it.
    d, args = BetaNB(), (5.0, 3000.0, 2000.0)
    sf = d.long_sf(4000, *args)
    assert sf.precision == 1024 and float(gmpy2.log10(sf)) < -400
    assert abs(float(gmpy2.log(sf)) - float(Ref.betanb(4000, *args)[2])) < 1e-10


CASES = [(Poisson(), (40.2,)), (NB(), (7.5, 0.6)), (NB(False), (0.6, 0.1)), (Binomial(), (300, 0.3)), (Binomial(False), (300, 0.3)),
         (BetaBinomial(), (300, 4.0, 9.0)), (BetaNB(), (7.5, 40.0, 60.0)), (MCNB(CompoundDist.NB), (7.5, 0.6, 0.3)),
         (MCNB(CompoundDist.Binomial), (7.5, 0.6)), (MCNB(CompoundDist.Poisson), (7.5, 0.6))]


@pytest.mark.parametrize('dist,args', CASES)
def test_long_matches_float(dist, args):
    xs = np.arange(0, 301, 7)
    x = jnp.asarray(xs, dtype=float)
    assert np.all(logerr(np.asarray(dist.logpmf(x, *args)), logs(dist.long_pmf(xs, *args))) <= 1e-10)
    assert np.all(logerr(np.asarray(dist.logsf(x, *args)), logs(dist.long_sf(xs, *args))) <= 1e-10)
    np.testing.assert_allclose(dist.cdf(x, *args), dist.long_cdf(xs, *args).astype(float), rtol=0, atol=1e-12)
    np.testing.assert_allclose(dist.sf(x, *args), dist.long_sf(xs, *args).astype(float), rtol=0, atol=1e-12)


# The cdf is the running sum of the pmf, and the sf is its complement.
@pytest.mark.parametrize('dist,args', CASES)
def test_cdf_is_cumulative(dist, args):
    xs = np.arange(120)
    pmf, cdf, sf = (getattr(dist, name)(xs, *args) for name in ('long_pmf', 'long_cdf', 'long_sf'))
    total = gmpy2.mpfr(0)
    for f, c, s in zip(pmf, cdf, sf):
        with gmpy2.context(gmpy2.get_context(), precision=1024):
            total += f
            assert abs(c - total) < gmpy2.mpfr(10) ** -290
            # A geometric tail is summed to a relative accuracy of 128 bits.
            assert abs(c + s - 1) <= s * gmpy2.exp2(-120)


def test_points_in_any_order():
    d, args = NB(), (7.5, 0.6)
    base = d.long_cdf(np.arange(50), *args)
    xs = np.array([[40, 3, 3], [0, 49, 7]])
    res = d.long_cdf(xs, *args)
    assert res.shape == (2, 3) and res.dtype == object
    assert all(res[i, j] == base[xs[i, j]] for i in range(2) for j in range(3))
    assert d.long_cdf([40, 3, 3], *args).tolist() == [base[40], base[3], base[3]]
    scalar = d.long_sf(40, *args)
    assert isinstance(scalar, type(gmpy2.mpfr(0))) and scalar == d.long_sf([40], *args)[0]
    assert d.long_pmf(np.int64(40), *args) == d.long_pmf([40.0], *args)[0]
    # Points outside of the support.
    assert [float(v) for v in d.long_pmf([-2, 1], *args)][0] == 0.0
    assert d.long_cdf(-2, *args) == 0 and d.long_sf(-2, *args) == 1
    b = Binomial()
    assert b.long_pmf(11, 10, 0.3) == 0 and b.long_cdf(11, 10, 0.3) == 1 and b.long_sf([10, 11, 500], 10, 0.3).tolist() == [0, 0, 0]
    assert b.long_sf(9, 10, 0.3) == b.long_pmf(10, 10, 0.3)
    assert b.long_sf([3, 500], 10, 0.3)[0] == b.long_sf(3, 10, 0.3) and b.long_cdf([3, 500], 10, 0.3)[0] == b.long_cdf(3, 10, 0.3)
    with pytest.raises(ValueError):
        d.long_cdf(3.5, *args)
    with pytest.raises(ValueError):
        d.long_cdf(3, np.array([7.5, 2.0]), 0.6)
    # A string is read as a decimal number, a float keeps its binary value.
    assert d.long_pmf(3, 7.5, '0.6') != d.long_pmf(3, 7.5, 0.6)
    assert abs(d.long_pmf(3, 7.5, '0.6') / d.long_pmf(3, 7.5, 0.6) - 1) < 1e-14


def test_precision_is_local():
    d, args = NB(), (7.5, 0.6)
    ctx = gmpy2.get_context()
    before = ctx.precision
    ctx.precision = 77
    try:
        default = d.long_sf(300, *args)
        wide = d.long_sf(300, *args, precision=2048)
        narrow = d.long_pmf([300, 2], *args, precision=96)
        mixed = Mixture(d, Poisson()).long_cdf(4, [dict(r=7.5, p=0.6), dict(rate=3.0)], 0.5, precision=200)
        trunc = TruncatedDistribution(d).long_sf(30, *args, left=5, precision=200)
        assert ctx.precision == 77
        assert gmpy2.get_context().precision == 77
    finally:
        ctx.precision = before
    assert default.precision == 1024 and wide.precision == 2048 and narrow[0].precision == 96
    assert mixed.precision == 200 and trunc.precision == 200
    assert abs(gmpy2.mpfr(default, 2048) / wide - 1) < gmpy2.mpfr(10) ** -300
    own = NB()
    own.long_precision = 256
    assert own.long_cdf(3, *args).precision == 256 and NB.long_precision == 1024


class CountingNB(NB):
    calls = 0

    def _long_ratio(self, x, ctx):
        self.calls += 1
        return super()._long_ratio(x, ctx)


# One pass over 0..max(x) however many points there are, plus the terms
# of the tail for the sf.
def test_cost_is_linear():
    d = CountingNB()
    d.long_cdf([3000, 5, 5, 100, 2999], 7.5, 0.6)
    assert d.calls == 3000
    d.calls = 0
    d.long_pmf(np.arange(3001), 7.5, 0.6)
    assert d.calls == 3000
    d.calls = 0
    d.long_sf(np.arange(3001), 7.5, 0.6)
    assert 3000 < d.calls < 3400


# The p-value of MixALime: the sf of a left truncated distribution.
def test_truncated_and_mixture():
    left = 5
    for dist, args in ((NB(), (7.5, 0.6)), (BetaNB(), (7.5, 12.0, 18.0)), (MCNB(CompoundDist.Binomial), (7.5, 0.6)),
                       (Binomial(), (300, 0.03)), (BetaBinomial(), (300, 2.0, 40.0))):
        t = TruncatedDistribution(dist)
        xs = [2, 4, 5, 6, 40, 40, 900]
        pmf, cdf, sf = (getattr(t, name)(xs, *args, left=left) for name in ('long_pmf', 'long_cdf', 'long_sf'))
        norm = dist.long_sf(left - 1, *args)
        with gmpy2.context(gmpy2.get_context(), precision=1024):
            for i, x in enumerate(xs):
                inside = x >= left
                assert abs(sf[i] - (dist.long_sf(x, *args) / norm if inside else 1)) <= sf[i] * gmpy2.exp2(-120)
                assert abs(pmf[i] - (dist.long_pmf(x, *args) / norm if inside else 0)) <= pmf[i] * gmpy2.exp2(-120)
                assert abs(cdf[i] + sf[i] - 1) <= gmpy2.exp2(-120)
            both = t.long_pmf(np.arange(60), *args, left=left, right=30)
            assert abs(sum(both) - 1) <= gmpy2.exp2(-120) and both[4] == 0 and both[31] == 0
        x = jnp.asarray(xs, dtype=float)
        assert np.all(logerr(np.asarray(t.logsf(x, *args, left=left)), logs(sf)) <= 1e-10)
        # Both bounds.
        assert t.long_cdf(30, *args, left=left, right=30) == 1 and t.long_sf(30, *args, left=left, right=30) == 0
        np.testing.assert_allclose(t.cdf(17.0, *args, left=left, right=30), float(t.long_cdf(17, *args, left=left, right=30)), rtol=1e-12)
    mix = Mixture(TruncatedDistribution(NB()), TruncatedDistribution(NB()))
    params = [dict(r=7.5, p=0.6), dict(r=7.5, p=0.4)]
    t = TruncatedDistribution(NB())
    for name in ('long_pmf', 'long_cdf', 'long_sf'):
        got = getattr(mix, name)([6, 20, 300], params, 0.7, left=left)
        one, two = (getattr(t, name)([6, 20, 300], **p, left=left) for p in params)
        with gmpy2.context(gmpy2.get_context(), precision=1024):
            for g, a, b in zip(got, one, two):
                assert abs(g - (gmpy2.mpfr(0.7) * a + (1 - gmpy2.mpfr(0.7)) * b)) <= g * gmpy2.exp2(-120)
    np.testing.assert_allclose(mix.sf(20.0, params, 0.7, left=left), float(mix.long_sf(20, params, 0.7, left=left)), rtol=1e-12)
