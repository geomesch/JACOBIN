#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, gammainc, gammaincc, betaln, logsumexp
from scipy.stats import binom as scipy_binom, poisson as scipy_poisson, nbinom as scipy_nb, betanbinom as scipy_betanb,\
    betabinom as scipy_betabinom
import mpmath
import gmpy2
import numpy as np
from abc import ABC
from enum import Enum

from .betainc import logbetainc, betainc
from .hyp import bnb_cdf
from .hyp import hyp_2f1_rec_start, hyp_2f1_rec_terms
from .utils import long_vectorize

class CompoundDist(Enum):
    NB = 'NB'
    Binomial = 'Binomial'

def logbinomial(n, x):
    n = jnp.array(n)
    x = jnp.array(x)
    return gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1) 

def long_logbeta(a, b):
    gammaln = lambda z: gmpy2.lgamma(z)[0]
    return gammaln(a) + gammaln(b) - gammaln(a + b)

class Distribution(ABC):
    
    rec_order = None
    
    def logpmf(self, x: jnp.ndarray, *args, **kwargs):
        raise NotImplementedError
    
    def pmf(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return jnp.exp(self.logpmf(x, *args, **kwargs))
    
    def logcdf(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def cdf(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return jnp.exp(self.logcdf(x, *args, **kwargs))
    
    def sf(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return -jnp.expm1(self.logcdf(x, *args, **kwargs))
    
    def logsf(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return jnp.log1p(-self.cdf(x, *args, **kwargs))
    
    @long_vectorize
    def long_pmf(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        min_x = min(x)
        max_x = max(x)
        res = np.empty_like(x, dtype=object)
        x = set(x)
        rec_order = self.rec_order
        prev = self._long_rec_start(min_x, *args, **kwargs)
        j = 0

        for i in range(rec_order):
            if i + min_x in x:
                res[j] = prev[i]
                j += 1
        for i in range(min_x + rec_order, max_x + 1):
            pmf = self._long_rec_step(i, prev, *args, **kwargs)
            prev[:-1] = prev[1:]
            prev[-1] = pmf
            if i in x:
                res[j] = pmf
                j += 1
        return res * self._long_rec_mult(*args, **kwargs)
    
    @long_vectorize
    def long_cdf(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        res = list()
        max_x = int(max(x))
        x = set(x)
        xs = list(map(gmpy2.mpz, range(max_x + 1)))
        cdf = 0
        pmfs = self.long_pmf(xs, *args, **kwargs)
        prev_cdf = cdf
        for c, pmf in enumerate(pmfs):
            cdf += pmf
            if cdf < 1 or ((1 - prev_cdf) > (cdf  - 1)):
                prev_cdf = cdf if cdf <= 1 else gmpy2.mpz(1)
            if c in x:
                res.append(prev_cdf)
        return np.array(res)

    def mean(self, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def var(self, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def sample(self, size: int, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def pmf_recurrent(self, min_x: int, max_x: int, max_sz: int, *args, **kwargs):
        def loop_body(x, res):
            res = res.at[x].set(self._rec_step(x, res, *args, **kwargs))
            return res
        
        res = jnp.zeros(max_sz, dtype=float)
        rec_order = self.rec_order
        start = self._rec_start(min_x, *args, **kwargs)
        for i in range(rec_order):
            res = res.at[min_x + i].set(start.at[i].get())
        return jax.lax.fori_loop(min_x + rec_order, max_x, loop_body, res) * self._rec_mult(*args, **kwargs)

    def _rec_mult(self, *args, **kwargs) -> float:
        return 1.0
    
    def _rec_step(self, x, prevs: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def _rec_start(self, x: float, *args, **kwargs) -> jnp.ndarray:
        rec = self.rec_order
        return jnp.array([self.pmf(x - rec + i, *args, **kwargs) for i in range(1, rec + 1)])
    
    def _long_rec_mult(self, *args, **kwargs) -> float:
        return 1.0
    
    def _long_rec_step(self, x, prevs: jnp.ndarray, *args, **kwargs) -> gmpy2.mpfr:
        raise NotImplementedError
    
    def _long_rec_start(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

class TruncatedDistribution(Distribution):
    
    def __init__(self, dist: Distribution):
        self.dist = dist
    
    def _lognorm(self, *args, left=None, right=None, **kwargs) -> jnp.array:
        dist = self.dist
        if left is None and right is None:
            denum = 0
        elif right is None:
            denum = dist.logsf(left, *args, **kwargs)
        elif left is None:
            denum = dist.logcdf(right - 1, *args, **kwargs)
        else:
            a = jnp.array([dist.logcdf(right - 1, *args, **kwargs), dist.logcdf(left, *args, **kwargs)])
            b = jnp.array([1, -1])
            denum = logsumexp(a, b=b)
        return denum
    
    def logpmf(self, x, *args, left=None, right=None, **kwargs):
        dist = self.dist
        logpmf = dist.logpmf(x, *args, **kwargs)
        return logpmf - self._lognorm(*args, left=left, right=right, **kwargs)
    
    def logcdf(self, x, *args, left=None, right=None, **kwargs):
        dist = self.dist
        logcdf = dist.logcdf(x, *args, **kwargs)
        denum = self._lognorm(*args, left=left, right=right, **kwargs)
        if left is not None:
            logcdf = jnp.log(jnp.exp(logcdf) - dist.logcdf(left, *args, **kwargs))
        return logcdf - denum
    
    def mean(self, *args, left=None, right=None, eps=1e-3, **kwargs):
        dist = self.dist
        mean = dist.mean(*args, **kwargs)
        
        def for_body(x, carry):
            return carry + x * dist.pmf(x, *args, **kwargs)
        
        def while_body(prev):
            stat, prev_stat, x = prev
            x = x + 1
            new_stat = stat + x * dist.pmf(x, *args, **kwargs)
            return (new_stat, stat, x)
        
        def while_cond(prev):
            stat, prev_stat, x = prev
            return jnp.abs(stat - prev_stat) / prev_stat > eps
        
        sub = 0
        if left is not None:
            sub += jax.lax.fori_loop(1, left, for_body, 0)
        if right is not None:
            print(sub)
            sub += jax.lax.while_loop(while_cond, while_body, (0., 1. + eps, right))
        
        return (mean - sub) / self._lognorm(*args, left=left, right=right, **kwargs)
    
    def var(self, *args, left=None, right=None, **kwargs):
        raise NotImplementedError
    
    def pmf_recurrent(self, min_x: int, max_x: int,  max_sz: int, *args, left=-1, right=float('inf'), **kwargs):
        dist = self.dist
        max_t_x = max_x
        min_t_x = jax.lax.select(left >= 0, 0, min_x)
        max_t_x = jax.lax.select(jnp.isfinite(right), right + 1, max_x)
        res = dist.pmf_recurrent(min_t_x, max_t_x, *args, max_sz=max_sz, **kwargs)
        def for_loop(x, carry):
            carry += res.at[x].get()
            return carry
        conds = [(left >= 0) & jnp.isfinite(right), (left >= 0) & jnp.isinf(right), left < 0]
        conds = sum(i * t for i, t in enumerate(conds))
        a, b = jax.lax.select_n(conds, 
                                *[jnp.array([left + 1, right]),  jnp.array([0, right]), jnp.array([0, left + 1])])
        denum = jax.lax.fori_loop(a, b, for_loop, 0.0)
        denum = jax.lax.select((left >= 0) & jnp.isinf(right), 1 - denum, denum)
        return res / denum
    
    @long_vectorize
    def long_pmf(self, x: np.ndarray, *args, left=-1, right=float('inf'), **kwargs) -> np.ndarray:
        dist = self.dist
        if left < 0 and np.isinf(right):
            return dist.long_pmf(x, *args, **kwargs)
        res = np.zeros(len(x), dtype=object)
        if left < 0:
            tx = list(range(right))
            pmfs = dist.long_pmf(tx, *args, **kwargs)
            denum = sum(pmfs)
        elif np.isinf(right):
            tx = list(range(x.max()))
            pmfs = dist.long_pmf(tx, *args, **kwargs)
            denum = 1 - sum(pmfs[i] for i in range(0, left + 1))
        else:
            tx = list(range(left + 1, right))
            pmfs = dist.long_pmf(tx, *args, **kwargs)
            denum = sum(pmfs)
        shift = tx[0]
        for i, t in enumerate(x):
            t = int(t) - shift
            if (0 <= t < len(pmfs)):
                res[i] = pmfs[t] / denum
        return res
            
class Mixture(Distribution):
    def __init__(self, *distros, normalize_weights=True):
        self.distros = distros
        self.normalize_weights = normalize_weights
    
    def _update_weights(self, weights: jnp.ndarray) -> jnp.ndarray:
        if len(self.distros) == 2 and not jnp.iterable(weights):
            weights = jnp.array([weights], dtype=float)
        else:
            weights = jnp.array(weights, dtype=float)
        if len(weights) == len(self.distros):
            weights = weights / weights.sum() if self.normalize_weights else weights
        elif len(weights) == len(self.distros) - 1:
            weights = jnp.append(weights, 1.0 - weights.sum())
        return weights.reshape(-1, 1)
    
    def _long_update_weights(self, weights: np.ndarray) -> np.ndarray:
        if len(self.distros) == 2 and not np.iterable(weights):
            weights = np.array([weights], dtype=float)
        else:
            weights = np.array(weights, dtype=float)
        if len(weights) == len(self.distros):
            weights = weights / weights.sum() if self.normalize_weights else weights
        elif len(weights) == len(self.distros) - 1:
            weights = np.append(weights, 1.0 - weights.sum())
        return weights.reshape(-1, 1)
            
    def logpmf(self, x: jnp.ndarray, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        logpmfs = list()
        weights = self._update_weights(weights)
        for params, dist in zip(params, self.distros):
            logpmfs.append(dist.logpmf(x, *args, **kwargs, **params))
        logpmfs = jnp.array(logpmfs)
        return logsumexp(logpmfs, axis=0, b=weights)
    
    def pmf_recurrent(self, min_x: int, max_x: int,  max_sz: int, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        logpmfs = list()
        weights = self._update_weights(weights)
        for params, dist in zip(params, self.distros):
            logpmfs.append(dist.pmf_recurrent(min_x, max_x, max_sz, *args, **kwargs, **params))
        logpmfs = jnp.array(logpmfs)
        return jnp.sum(logpmfs * weights, axis=0)
    
    def logcdf(self, x: jnp.ndarray, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        logcdfs = list()
        weights = self._update_weights(weights)
        for params, dist in zip(params, self.distros):
            logcdfs.append(dist.logcdf(x, *args, **kwargs, **params))
        logcdfs = jnp.array(logcdfs)
        return logsumexp(logcdfs, axis=0, b=weights)

    def long_pmf(self, x: np.ndarray, params: list[dict], weights: np.ndarray, *args, **kwargs) -> np.ndarray:
        pmfs = list()
        weights = self._long_update_weights(weights)
        for params, dist in zip(params, self.distros):
            pmfs.append(dist.long_pmf(x, *args, **kwargs, **params))
        pmfs = np.array(pmfs)
        return (pmfs * weights).sum(axis=0)

    def mean(self, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        means = list()
        weights = self._update_weights(weights)
        for params, dist in zip(params, self.distros):
            means.append(dist.mean(*args, **kwargs, **params))
        means = jnp.array(means)
        return jnp.sum(means * weights, axis=0)
    
    def var(self, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        var = list()
        weights = self._update_weights(weights)
        for params, dist in zip(params, self.distros):
            var.append(dist.var(*args, **kwargs, **params))
        var = jnp.array(var)
        return jnp.sum(var * weights, axis=0)

class BinomialFamily(Distribution):
    
    def __init__(self, p_success=True):
        self.p_success = p_success
        
class BetaBinomialFamily(BinomialFamily):
    
    def __init__(self):
        pass

class Poisson(Distribution):
    
    rec_order = 1
    
    def logpmf(self, x: jnp.ndarray, rate: jnp.ndarray) -> jnp.ndarray:
        x, rate = map(jnp.array, (x, rate))
        return x * jnp.log(rate) - rate - gammaln(x + 1)
    
    def cdf(self, x: jnp.ndarray, rate: jnp.ndarray) -> jnp.ndarray:
        x, rate = map(jnp.array, (x, rate))
        return gammaincc(x + 1, rate)
    
    def logcdf(self, x: jnp.ndarray, rate: jnp.ndarray) -> jnp.ndarray:
        x, rate = map(jnp.array, (x, rate))
        return jnp.log(gammaincc(x + 1, rate))

    def sf(self, x: jnp.ndarray, rate: jnp.ndarray) -> jnp.ndarray:
        x, rate = map(jnp.array, (x, rate))
        return gammainc(x + 1, rate)
    
    def logsf(self, x: jnp.ndarray, rate: jnp.ndarray) -> jnp.ndarray:
        x, rate = map(jnp.array, (x, rate))
        return jnp.log(gammainc(x + 1, rate))
    
    def _rec_start(self, x: int, rate) -> jnp.ndarray:
        res = jnp.exp(x * jnp.log(rate) - rate - gammaln(x + 1))
        return jnp.array([res])
    
    def _rec_step(self, x: int, prevs: jnp.array, rate: float) -> float:
        return prevs.at[x-1].get() * rate / x
    
    def _long_rec_start(self, x: int, rate) -> np.ndarray:
        x = gmpy2.mpz(x); rate = gmpy2.mpfr(rate)
        res = gmpy2.exp(x * gmpy2.log(rate) - rate - gmpy2.lngamma(x + 1))
        return np.array([res])
    
    def _long_rec_step(self, x: int, prevs: np.array, rate: float) -> gmpy2.mpfr:
        x = gmpy2.mpz(x); rate = gmpy2.mpfr(rate)
        return prevs[0] * rate / x
            
    def mean(self, rate: jnp.ndarray) -> jnp.ndarray:
        rate = jnp.array(rate)
        return rate
    
    def var(self, rate: jnp.ndarray) -> jnp.ndarray:
        rate = jnp.array(rate)
        return rate
    
    def sample(self, size: int, rate) -> np.ndarray:
        return scipy_poisson.rvs(rate, size=size)

class Binomial(BinomialFamily):
    
    rec_order = 1
    
    def __init__(self, p_success=True, eps=1e-6, max_n=200):
        self.p_success = p_success
        self.eps = eps
        self.max_n = max_n
    
    def logpmf(self, x: jnp.ndarray, n, p) -> jnp.ndarray:
        x, p, n = jnp.array(x), jnp.array(p), jnp.array(n)
        p = x * jnp.log(p) + (n - x) * jnp.log1p(-p) if self.p_success else x * jnp.log1p(-p) + (n - x) * jnp.log(p)
        return p + logbinomial(n, x)
    
    def logcdf(self, x: jnp.ndarray, n, p) -> jnp.ndarray:
        x, p, n = jnp.array(x), jnp.array(p), jnp.array(n)
        eps, max_n = self.eps, self.max_n
        return logbetainc(n - x, x + 1, 1 - p if self.p_success else p, eps=eps, max_n=max_n)
    
    def cdf(self, x: jnp.ndarray, n, p) -> jnp.ndarray:
        x, p, n = jnp.array(x), jnp.array(p), jnp.array(n)
        eps, max_n = self.eps, self.max_n
        return betainc(n - x, x + 1, 1 - p if self.p_success else p, eps=eps, max_n=max_n)
    
    def sf(self, x: jnp.ndarray, n, p) -> jnp.ndarray:
        x, p, n = jnp.array(x), jnp.array(p), jnp.array(n)
        eps, max_n = self.eps, self.max_n
        return betainc(x + 1, n - x, p if self.p_success else 1 - p, eps=eps, max_n=max_n)
    
    def _rec_start(self, x: int, n, p) -> jnp.ndarray:
        return jnp.array([self.pmf(x, n, p)])
    
    def _rec_step(self, x: int, prevs: jnp.array, n, p) -> float:
        q = 1 - p
        if not self.p_success:
            p, q = q, p
        return prevs.at[x-1].get() * (n - x + 1) / x * p / q
    
    def _long_rec_start(self, x: int, n, p) -> np.ndarray:
        x = gmpy2.mpz(x); n = gmpy2.mpfr(n); p = gmpy2.mpfr(p)
        q = 1 - p
        if not self.p_success:
            p, q = q, p
        res = gmpy2.bincoef(n, x) * p ** x * q ** (n - x)
        return np.array([res])
    
    def _long_rec_step(self, x: int, prevs: np.array, n, p) -> gmpy2.mpfr:
        x = gmpy2.mpz(x); n = gmpy2.mpfr(n); p = gmpy2.mpfr(p)
        q = 1 - p
        if not self.p_success:
            p, q = q, p
        return prevs[0] * (n - x + 1) / x * p / q
    
    def sample(self, size: int, p, n) -> np.ndarray:
        return scipy_binom.rvs(n=n, p=p if self.p_success else 1.0 - p, size=size)
    
    def mean(self, p, n) -> jnp.ndarray:
        p, n = jnp.array(p), jnp.array(n)
        return p * n if self.p_success else (1 - p) * n
    
    def var(self, p, n) -> np.ndarray:
        p, n = jnp.array(p), jnp.array(n)
        return p * (1 - p) * n

class NB(BinomialFamily):
    
    rec_order = 1
    
    def __init__(self, p_success=True, eps=1e-6, max_n=200):
        self.p_success = p_success
        self.eps = eps
        self.max_n = max_n
    
    
    def logpmf(self, x: jnp.ndarray, r, p) -> jnp.ndarray:
        x, p, r = jnp.array(x), jnp.array(p), jnp.array(r)
        p = jnp.log(p) * x + jnp.log1p(-p) * r if self.p_success else jnp.log1p(-p) * x + jnp.log(p) * r
        return p + logbinomial(x + r - 1, x)

    def cdf(self, x: jnp.ndarray, r, p):
        x, p, r = jnp.array(x), jnp.array(p), jnp.array(r)
        eps, max_n = self.eps, self.max_n
        return betainc(r, x + 1.0, 1 - p if self.p_success else p, eps=eps, max_n=max_n)
    
    def logcdf(self, x: jnp.ndarray, r, p):
        x, p, r = jnp.array(x), jnp.array(p), jnp.array(r)
        eps, max_n = self.eps, self.max_n
        return logbetainc(r, x + 1.0, 1 - p if self.p_success else p, eps=eps, max_n=max_n)

    def sf(self, x, r, p, r_transform=None):
        x, p, r = jnp.array(x), jnp.array(p), jnp.array(r)
        eps, max_n = self.eps, self.max_n
        return betainc(x + 1.0, r, p if self.p_success else 1 - p, eps=eps, max_n=max_n)

    def _rec_step(self, x: int, prevs: jnp.array, r, p) -> float:
        q = 1 - p
        if not self.p_success:
            p, q = q, p
        return prevs.at[x-1].get() * (x + r - 1) / x * q
    
    def _long_rec_start(self, x: int, r, p) -> np.ndarray:
        q = 1 - p
        if not self.p_success:
            p, q = q, p
        res = gmpy2.bincoef(x + r - 1, x) * p ** x * q ** r
        return np.array([res])
    
    def _long_rec_step(self, x: int, prevs: np.array, r, p) -> gmpy2.mpfr:
        q = 1 - p
        if not self.p_success:
            p, q = q, p
        return prevs[0] * (x + r - 1) / x * q

    def mean(self, r, p) -> jnp.ndarray:
        p, r = jnp.array(p), jnp.array(r)
        if not self.p_success:
            p = 1 - p
        return r * p / (1 - p)

    def var(self, r, p) -> jnp.ndarray:
        p, r = jnp.array(p), jnp.array(r)
        mean = self.mean(r, p)
        if not self.p_success:
            p = 1 - p
        return mean / p

    def sample(self, size: int, r, p) -> np.array:
        return scipy_nb.rvs(size=size, r=r, p=p if not self.p_success else 1 - p)


class BetaNB(BetaBinomialFamily):
    
    rec_order = 1
    
    def __init__(self, eps=1e-6, max_n=200):
        self.eps = eps
        self.max_n = max_n
    
    def logpmf(self, x: jnp.ndarray, r, a, b) -> jnp.ndarray:
        x = jnp.array(x)
        r, a, b = jnp.array(r), jnp.array(a), jnp.array(b)
        return betaln(a + r, b + x) - betaln(a, b) + gammaln(r + x) -\
               gammaln(x + 1.0) - gammaln(r)
    
    def cdf(self, x: jnp.ndarray, r, a, b) -> jnp.ndarray:
        x = jnp.array(x)
        r, a, b = jnp.array(r), jnp.array(a), jnp.array(b)
        eps, max_n = self.eps, self.max_n
        return bnb_cdf(x, r, a, b, eps=eps, max_n=max_n)
    
    def sf(self, x: jnp.ndarray, r, a, b) -> jnp.ndarray:
        x = jnp.array(x)
        r, a, b = jnp.array(r), jnp.array(a), jnp.array(b)
        eps, max_n = self.eps, self.max_n
        return bnb_cdf(x, r, a, b, eps=eps, max_n=max_n, sf=True)
    
    def logsf(self, x: jnp.ndarray, r, a, b) -> jnp.ndarray:
        return jnp.log(self.sf(x, r, a, b))
    
    def logcdf(self, x: jnp.ndarray, r, a, b) -> jnp.ndarray:
        return jnp.log(self.cdf(x, r, a, b))
    
    def _rec_step(self, x, prevs: jnp.ndarray, r, a, b) -> jnp.ndarray:
        return prevs.at[x-1].get() * (x + r - 1) / (x + a + b + r - 1)  * ((x + b - 1) / x)
    
    def _long_rec_start(self, x: int, r, a, b) -> np.ndarray:
        x = gmpy2.mpz(x); r = gmpy2.mpfr(r); a = gmpy2.mpfr(a); b = gmpy2.mpfr(b)
        res = gmpy2.mpfr(str(mpmath.gammaprod(list(map(str, [a + r, b + x, a + b, r + x, x + 1])), 
                                              list(map(str, [a + b + r + x, a, b, r])))))
        return np.array([res])
    
    def _long_rec_step(self, x: int, prevs: np.array, r, a, b) -> gmpy2.mpfr:
        x = gmpy2.mpz(x); r = gmpy2.mpfr(r); a = gmpy2.mpfr(a); b = gmpy2.mpfr(b)
        return prevs[0] * ((x + r - 1) / (x + a + b + r - 1)  * ((x + b - 1) / x))
    
    def sample(self, size: int, r, a, b) -> np.ndarray:
        return scipy_betanb.rvs(size=size, n=r, a=a, b=b)
    
    def mean(self, r, a, b) -> jnp.ndarray:
        return r * b / (a - 1)
    
    def var(self, r, a, b) -> jnp.ndarray:
        return r * b * (r + a - 1) * (b + a - 1) / ((a - 2) * (a - 1) ** 2)


class BetaBinomial(BetaBinomialFamily):
    
    rec_order = 1
    
    def logpmf(self, x: jnp.ndarray, n, a, b) -> jnp.ndarray:
        x, n, a, b = jnp.array(x), jnp.array(n), jnp.array(a), jnp.array(b)
        return betaln(x + a, n - x + b) - betaln(a, b) + logbinomial(n, x)


    def _rec_step(self, x, prevs: jnp.ndarray, n, a, b) -> jnp.ndarray:
        t1 = (n - x + 1) / x
        t2 = (x + a - 1) / (n - x + b)
        return prevs.at[x-1].get() * t1* t2
    
    def _long_rec_start(self, x: int, n, a, b) -> np.ndarray:
        x = gmpy2.mpz(x); n = gmpy2.mpfr(n); a = gmpy2.mpfr(a); b = gmpy2.mpfr(b)
        res = gmpy2.mpfr(str(mpmath.gammaprod(list(map(str, [x + a, n - x + b, a + b, n + 1])), 
                                              list(map(str, [n + a + b, a, b, n - x + 1, x + 1])))))
        return np.array([res])
    
    def _long_rec_step(self, x: int, prevs: np.array, n, a, b) -> gmpy2.mpfr:
        x = gmpy2.mpz(x); n = gmpy2.mpfr(n); a = gmpy2.mpfr(a); b = gmpy2.mpfr(b)
        t1 = (n - x + 1) / x
        t2 = (x + a - 1) / (n - x + b)
        return prevs[0] * t1 * t2

    def sample(self, size: int, n, a, b) -> np.ndarray:
        return scipy_betabinom.rvs(n=n, a=a, b=b, size=size)
    
    def mean(self, n, a, b) -> jnp.ndarray:
        return n * a / (a + b)
    
    def var(self, n, a, b) -> np.ndarray:
        num = n * a * b * (a + b + n)
        denum = (a + b) ** 2 * (a + b + 1)
        return num / denum

class MCNB(BinomialFamily):
    rec_order = 2
    
    def __init__(self, compound_dist=CompoundDist.NB):
        self.dist_name = compound_dist
    
    def _calc_abz(self, x, r, p1, p2):
        if self.dist_name == CompoundDist.NB:
            return 1 + r, 1 + x, p2 * (1 - p1)
        elif self.dist_name == CompoundDist.Binomial:
            return 1 - r, 1 + x, -p2 * (1 - p1) / (1 - p2)
        raise NotImplementedError
    
    def _rec_mult(self, r, p1, p2):
        lt = jnp.log1p(-p2) * r
        num = jnp.log(r) + lt + jnp.log1p(-p1) + jnp.log(p2)
        denum = logsumexp(jnp.array([0.0, lt]), b=jnp.array([1, -1]))
        if self.dist_name == CompoundDist.Binomial:
            num -= jnp.log1p(-p2)
        return jnp.exp(num - denum)

    def _long_rec_mult(self, r, p1, p2):
        mult = (1 - p2) ** r / (1 - (1 - p2) ** r) * (r  * (1 - p1) * p2)
        if self.dist_name == CompoundDist.Binomial:
            mult /= (1 - p2)
        return mult
    
    def _rec_step(self, x, prevs: jnp.ndarray, r, p1, p2) -> jnp.ndarray:
        a, b, z = self._calc_abz(x, r, p1, p2)
        alpha, beta = hyp_2f1_rec_terms(a, b, z)
        return alpha * prevs.at[x-1].get() * p1 + beta * prevs.at[x-2].get() * p1 ** 2
    
    def _rec_start(self, x, r, p1, p2) -> jnp.ndarray:
        a, b, z = self._calc_abz(x, r, p1, p2)
        t1, t2 = hyp_2f1_rec_start(a, b, z)
        return jnp.array([t1, t2 * p1])
    
    def _long_rec_start(self, x, r, p1, p2) -> np.array:
        r = gmpy2.mpfr(r)
        p1 = gmpy2.mpfr(p1); p2 = gmpy2.mpfr(p2)
        a, b, z = self._calc_abz(x, r, p1, p2)
        return np.array(hyp_2f1_rec_start(a, b, z))

    def mean(self, r, p1, p2):
        num = jnp.log(p1)
        denum = 2 * jnp.log1p(-p1) + jnp.log1p(-p2) * (r + 1)
        if self.dist_name == CompoundDist.Binomial:
            num += 2 * jnp.log1p(-p2)
        return jnp.exp(num - denum) * self._rec_mult(r, p1, p2)
    
    def second_moment(self, r, p1, p2):
        if self.dist_name == CompoundDist.NB:
            num = jnp.log(p1) + jnp.log(1 + p1 - p2 + r * p1 * p2)
            denum = jnp.log1p(-p1) * 3 + jnp.log1p(-p2) * (r + 2)
        else:
            num = jnp.log(p1) + jnp.log(1 + p1 + (r - 1) * p1 * p2)
            denum = jnp.log1p(-p1) * 3 + jnp.log1p(-p2) * (r - 1)
        return jnp.exp(num - denum)
    
    def var(self, r, p1, p2):
        x, x2 = self.mean(r, p1, p2), self.second_moment(r, p1, p2)
        return x ** 2 - x2
