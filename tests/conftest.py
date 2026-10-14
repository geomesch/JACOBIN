# -*- coding: utf-8 -*-
import numpy as np
import mpmath as mp
import gmpy2
import pytest
import jax
import jacobin  # noqa: F401, enables 64-bit floats

mp.mp.dps = 60


def lu(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def logs(values):
    return np.array([float(gmpy2.log(v)) if v > 0 else -np.inf for v in np.atleast_1d(values)])


def mplog(v):
    return float(mp.log(v)) if v > 0 else -np.inf


# Absolute difference of two arrays of logarithms, zero where both are
# the same infinity.
def logerr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(invalid='ignore'):
        return np.where(a == b, 0.0, np.abs(a - b))


# Reference log-pmf, log-cdf and log-sf in mpmath. Each function takes
# an integer x and Python floats, they are converted without rounding.
class Ref:
    @staticmethod
    def poisson(x, rate):
        rate = mp.mpf(rate)
        return (x * mp.log(rate) - rate - mp.loggamma(x + 1),
                mp.log(mp.gammainc(x + 1, rate, mp.inf, regularized=True)),
                mp.log(mp.gammainc(x + 1, 0, rate, regularized=True)))

    @staticmethod
    def nb(x, r, p):
        r, p = mp.mpf(r), mp.mpf(p)
        pmf = x * mp.log(p) + r * mp.log(1 - p) + mp.loggamma(x + r) - mp.loggamma(x + 1) - mp.loggamma(r)
        return (pmf, mp.log(mp.betainc(r, x + 1, 0, 1 - p, regularized=True)),
                mp.log(mp.betainc(x + 1, r, 0, p, regularized=True)))

    @staticmethod
    def binomial(x, n, p):
        n, p = int(n), mp.mpf(p)
        pmf = mp.log(mp.binomial(n, x)) + x * mp.log(p) + (n - x) * mp.log(1 - p)
        cdf = sum(mp.binomial(n, k) * p ** k * (1 - p) ** (n - k) for k in range(x + 1))
        sf = sum(mp.binomial(n, k) * p ** k * (1 - p) ** (n - k) for k in range(x + 1, n + 1))
        return pmf, mp.log(cdf) if cdf > 0 else -mp.inf, mp.log(sf) if sf > 0 else -mp.inf

    @staticmethod
    def betabinomial(x, n, a, b):
        n, a, b = int(n), mp.mpf(a), mp.mpf(b)
        f = lambda k: mp.binomial(n, k) * mp.beta(k + a, n - k + b) / mp.beta(a, b)
        cdf = sum(f(k) for k in range(x + 1))
        sf = sum(f(k) for k in range(x + 1, n + 1))
        return mp.log(f(x)), mp.log(cdf), mp.log(sf) if sf > 0 else -mp.inf

    # The sf is a complement, so the working precision grows with x.
    @staticmethod
    def betanb(x, r, a, b):
        with mp.workdps(60 + int(1.5 * x) + 400):
            r, a, b = mp.mpf(r), mp.mpf(a), mp.mpf(b)
            f = mp.beta(a + r, b) / mp.beta(a, b)
            cdf = f
            for k in range(1, x + 1):
                f *= (k + r - 1) / (k + a + b + r - 1) * ((k + b - 1) / k)
                cdf += f
            return mp.log(f), mp.log(cdf), mp.log(1 - cdf)

    # Equation 8 of the MixALime paper.
    @staticmethod
    def mcnb_paper(y, r, p):
        with mp.workdps(120 + y // 4):
            r, p = mp.mpf(r), mp.mpf(p)
            f = r * (1 - p) ** 2 * p ** (r + y - 1) * mp.hyp2f1(1 - r, y + 1, 2, -(1 - p) ** 2 / p) / (1 - p ** r)
            return mp.log(f)

    # Sum over the hidden count k of NB(y | k, p1) times a zero-truncated
    # compound distribution of k, the definition of MCNB.
    @staticmethod
    def mcnb_direct(y, r, p1, p2, kind):
        r, p1, p2 = mp.mpf(r), mp.mpf(p1), mp.mpf(p2)
        nb = lambda k: mp.binomial(y + k - 1, y) * p1 ** y * (1 - p1) ** k
        if kind == 'NB':
            f = lambda k: mp.binomial(k + r - 1, k) * p2 ** k * (1 - p2) ** r / (1 - (1 - p2) ** r)
        elif kind == 'Poisson':
            f = lambda k: r ** k * mp.exp(-r) / mp.factorial(k) / (1 - mp.exp(-r))
        else:
            r = int(r)
            f = lambda k: mp.binomial(r, k) * p2 ** k * (1 - p2) ** (r - k) / (1 - (1 - p2) ** r)
            return mp.log(sum(nb(k) * f(k) for k in range(1, r + 1)))
        return mp.log(mp.nsum(lambda k: nb(int(k)) * f(int(k)), [1, mp.inf]))


@pytest.fixture
def rng():
    return np.random.default_rng(20261014)


# Compiled functions pile up to tens of gigabytes over the whole suite.
@pytest.fixture(autouse=True)
def clear_compiled():
    yield
    jax.clear_caches()
