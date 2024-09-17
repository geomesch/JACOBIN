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
from .hyp import hyp_long_1f1_rec_start
from .hyp import hyp_2f1_rec_start, hyp_2f1_rec_terms
from .hyp import hyp_1f1_rec_start, hyp_1f1_rec_terms
from .hyp import hyp_log_2f1_rec_start, hyp_log_1f1_rec_start
from .utils import long_vectorize, recurrent_fun, recurrent_fun_long

class CompoundDist(Enum):
    NB = 'NB'
    Binomial = 'Binomial'
    Poisson = 'Poisson'
 
def logbinomial(n, x):
    n = jnp.array(n)
    x = jnp.array(x)
    return gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1) 

def long_logbeta(a, b):
    gammaln = lambda z: gmpy2.lgamma(z)[0]
    return gammaln(a) + gammaln(b) - gammaln(a + b)

def gmpy2_bincoef(n, x):
    return gmpy2.gamma(n + 1) / (gmpy2.gamma(n - x + 1) * gmpy2.fac(x))

class Distribution(ABC):
    
    rec_order = None
    params = dict()
    
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
        min_x = 0 # min(x) TODO
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
        rec_start = self._long_rec_start
        rec_step = self._long_rec_step
        rec_mult = self._long_rec_mult
        return recurrent_fun_long(rec_start, rec_step, rec_mult, self.rec_order, x, *args, **kwargs)

    def mean(self, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def var(self, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def sample(self, size: int, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def logpmf_recurrent(self, min_x: int, max_x: int, max_sz: int, *args, **kwargs):
        return recurrent_fun(self._rec_start, self._rec_step, self.rec_order, 
                             min_x, max_x, max_sz, *args, **kwargs)
    
    def pmf_recurrent(self, min_x: int, max_x: int, max_sz: int, *args, **kwargs):
        return jnp.exp(self.logpmf_recurrent(min_x, max_x, max_sz, *args, **kwargs))

    
    def _rec_step(self, x, prevs: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def _rec_start(self, x: float, *args, **kwargs) -> jnp.ndarray:
        rec = self.rec_order
        return jnp.array([self.logpmf(x - rec + i, *args, **kwargs) for i in range(1, rec + 1)])
    
    def _long_rec_mult(self, *args, **kwargs) -> float:
        return 1.0
    
    def _long_rec_step(self, x, prevs: jnp.ndarray, *args, **kwargs) -> gmpy2.mpfr:
        raise NotImplementedError
    
    def _long_rec_start(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

class TruncatedDistribution(Distribution):
    
    params = None
    
    def __init__(self, dist: Distribution, clip_bounds=True, eps=None):
        self.dist = dist
        self.params = dist.params
        self.clip_bounds = clip_bounds
        self.eps = eps
    
    def _lognorm(self, *args, left=None, right=None, **kwargs) -> jnp.array:
        dist = self.dist
        left = None if left is None or left <= 0 else left
        right = None if right is None or np.isinf(right) else right
        if left is None and right is None:
            denum = 0
        elif right is None:
            denum = dist.logsf(left - 1, *args, **kwargs)
        elif left is None:
            denum = dist.logcdf(right, *args, **kwargs)
        else:
            a = jnp.array([dist.logcdf(right, *args, **kwargs), dist.logcdf(left - 1, *args, **kwargs)])
            b = jnp.array([1, -1])
            denum = logsumexp(a, b=b)
        return denum
    
    def logpmf(self, x, *args, left=0, right=float('inf'), **kwargs):
        dist = self.dist
        logpmf = dist.logpmf(x, *args, **kwargs)
        res = logpmf - self._lognorm(*args, left=left, right=right, **kwargs)
        if self.clip_bounds:
            res = jnp.where((x >= left) & (x <= right), res, -jnp.inf)
        return res
    
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
            sub += jax.lax.while_loop(while_cond, while_body, (0., 1. + eps, right))
        
        return (mean - sub) / self._lognorm(*args, left=left, right=right, **kwargs)
    
    def var(self, *args, left=None, right=None, **kwargs):
        raise NotImplementedError
    
    def logpmf_recurrent(self, min_x: int, max_x: int,  max_sz: int, *args, left=0, right=float('inf'), **kwargs):
        dist = self.dist
        max_t_x = jax.lax.select((right + 1.0 > max_x) & jnp.isfinite(right), right + 1.0, jnp.array(max_x).astype(float)).astype(int)
        res = dist.logpmf_recurrent(0, max_t_x, max_sz, *args, **kwargs)
        def for_loop(x, carry):
            carry += jnp.exp(res.at[x].get())
            return carry
        conds = [(left > 0) & jnp.isfinite(right), (left > 0) & jnp.isinf(right), (left <= 0) & jnp.isfinite(right), (left <= 0) & jnp.isinf(right)]
        conds = sum(i * t for i, t in enumerate(conds))
        left = jnp.array(left).astype(int); right = jnp.array(right).astype(int)
        a, b = jax.lax.select_n(conds, 
                                *[jnp.array([left, right + 1]),  jnp.array([0, left]), jnp.array([0, right + 1]), jnp.array([0, 0])])
        denum = jax.lax.fori_loop(a, b, for_loop, 0.0)
        denum = jnp.clip(denum, self.eps if self.eps is None else 0.0, 1.0)
        denum = jax.lax.select(conds == 3, 1.0, denum)
        denum = jax.lax.select(conds == 1, jnp.log1p(-denum), jnp.log(denum))
        res = res - denum
        if self.clip_bounds:
            inds = jnp.arange(0, max_sz)
            res = jnp.where((inds >= left) & (inds <= right), res, -jnp.inf)
        return res
    
    @long_vectorize
    def long_pmf(self, x: np.ndarray, *args, left=0, right=float('inf'), **kwargs) -> np.ndarray:
        dist = self.dist
        if left < 0 and np.isinf(right):
            return dist.long_pmf(x, *args, **kwargs)
        res = np.zeros(len(x), dtype=object)
        if left <= 0 and not np.isinf(right):
            tx = list(range(right + 1))
            pmfs = dist.long_pmf(tx, *args, **kwargs)
            denum = sum(pmfs)
        elif jnp.isinf(right):
            tx = list(range(x.max() + 1))
            pmfs = dist.long_pmf(tx, *args, **kwargs)
            denum = 1 - sum(pmfs[i] for i in range(0, left))

        else:
            tx = list(range(left, right + 1))
            pmfs = dist.long_pmf(tx, *args, **kwargs)
            denum = sum(pmfs)
        shift = tx[0]
        for i, t in enumerate(x):
            t = int(t) - shift
            if (0 <= t < len(pmfs)) and (left <= t + shift <= right):
                res[i] = pmfs[t] / denum
        return res
            
class Mixture(Distribution):
    def __init__(self, *distros, normalize_weights=True):
        self.distros = distros
        self.normalize_weights = normalize_weights
        self.params['weights'] = [(0.0, 1.0)] * (len(distros) - 1)
    
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
    
    def logpmf_recurrent(self, min_x: int, max_x: int,  max_sz: int, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        logpmfs = list()
        weights = self._update_weights(weights)
        for params, dist in zip(params, self.distros):
            logpmfs.append(dist.logpmf_recurrent(min_x, max_x, max_sz, *args, **kwargs, **params))
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
    params = {'rate': [(0.0, None)]}
    
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
    
    def _rec_step(self, x: int, prevs: jnp.array, rate: float) -> float:
        return prevs.at[x-1].get() + jnp.log(rate) - jnp.log(x)
    
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
    params = {'n': [(0.0, None)], 'p': [(0.0, 1.0)]}
    
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
    
    def _rec_step(self, x: int, prevs: jnp.array, n, p) -> float:
        t = jnp.log(p) - jnp.log1p(-p)
        if not self.p_success:
            t *= -1
        return prevs.at[x-1].get() + jnp.log(n - x + 1) - jnp.log(x) + t
    
    def _long_rec_start(self, x: int, n, p) -> np.ndarray:
        x = gmpy2.mpz(x); n = gmpy2.mpfr(n); p = gmpy2.mpfr(p)
        q = 1 - p
        if not self.p_success:
            p, q = q, p
        res = gmpy2_bincoef(n, x) * p ** x * q ** (n - x)
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
    params = {'r': [(0.0, None)], 'p': [(0.0, 1.0)]}
    
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
        if not self.p_success:
            q = jnp.log(p)
        else:
            q = jnp.log1p(-p)
        return prevs.at[x-1].get() + jnp.log(x + r - 1) - jnp.log(x) + q
    
    def _long_rec_start(self, x: int, r, p) -> np.ndarray:
        q = 1 - p
        if not self.p_success:
            p, q = q, p
        res = gmpy2_bincoef(x + r - 1, x) * p ** x * q ** r
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
    params = {'r': [(0.0, None)], 'a': [(0.0, None)], 'b': [(0.0, None)]}
    
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
        return prevs.at[x-1].get() + jnp.log(x + r - 1) - jnp.log(x + a + b + r - 1)  + jnp.log(x + b - 1) - jnp.log(x)
    
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
    params = {'n': [(0.0, None)], 'a': [(0.0, None)], 'b': [(0.0, None)]}
    
    def logpmf(self, x: jnp.ndarray, n, a, b) -> jnp.ndarray:
        x, n, a, b = jnp.array(x), jnp.array(n), jnp.array(a), jnp.array(b)
        return betaln(x + a, n - x + b) - betaln(a, b) + logbinomial(n, x)


    def _rec_step(self, x, prevs: jnp.ndarray, n, a, b) -> jnp.ndarray:
        t1 = jnp.log(n - x + 1) - jnp.log(x)
        t2 = jnp.log(x + a - 1) - jnp.log(n - x + b)
        return prevs.at[x-1].get() + t1 + t2
    
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
    params = {'r': [(0.0, None)], 'p1': [(0.0, 1.0)], 'p2': [(0.0, 1.0)]}
    
    def __init__(self, compound_dist=CompoundDist.NB):
        self.dist_name = compound_dist
        if compound_dist == CompoundDist.Poisson:
            self.params['p2'] = None
    
    def _calc_abz(self, x, r, p1, p2, long=False, log_z=True):
        lib = jnp if not long else gmpy2
        if self.dist_name == CompoundDist.NB:
            logz = lib.log(p2) + lib.log1p(-p1)
            return 1 + r, 1 + x, logz if log_z else lib.exp(logz)
        elif self.dist_name == CompoundDist.Binomial:
            logz = lib.log1p(-p1) + lib.log(p2) - lib.log1p(-p1 * p2)
            return 1 + r, 1 + x, logz if log_z else lib.exp(logz)
        elif self.dist_name == CompoundDist.Poisson:
            return 1 + x, lib.log(r) + lib.log1p(-p1) if log_z else r * (1 - p1)
        raise NotImplementedError
    
    def _rec_mult(self, r, p1, p2):
        if self.dist_name == CompoundDist.NB:
            t = jnp.log1p(-p2) * r
            num = jnp.log(r) + t + jnp.log1p(-p1) + jnp.log(p2)
            denum = logsumexp(jnp.array([0.0, t]), b=jnp.array([1, -1]))
        elif self.dist_name == CompoundDist.Binomial:
            num = jnp.log1p(-p1) + jnp.log(p2) + jnp.log(r) + jnp.log1p(-p2) * (r - 1)
            denum = logsumexp(jnp.array([0.0, jnp.log1p(-p2) * r]), b=jnp.array([1, -1]))
            zlog = jnp.log1p(-p1) + jnp.log(p2) - jnp.log1p(-p2)
            denum += jnp.logaddexp(0.0, zlog)
        elif self.dist_name == CompoundDist.Poisson:
            num = jnp.log1p(-p1) + jnp.log(r)
            denum = jax.lax.select(jnp.isfinite(jnp.exp(r)), jnp.log(jnp.exp(r) - 1.0), r)
        return num - denum

    def _long_rec_mult(self, r, p1, p2):
        if self.dist_name == CompoundDist.NB:
            t = gmpy2.log1p(-p2) * r
            num = gmpy2.log(r) + t + gmpy2.log1p(-p1) + gmpy2.log(p2)
            denum = gmpy2.log1p(-gmpy2.exp(t))
        elif self.dist_name == CompoundDist.Binomial:
            num = gmpy2.log1p(-p1) + gmpy2.log(p2) + gmpy2.log(r) + gmpy2.log1p(-p2) * (r - 1)
            denum = gmpy2.log1p(-(1-p2) ** r)
            zlog = gmpy2.log1p(-p1) + gmpy2.log(p2) - gmpy2.log1p(-p2)
            denum += gmpy2.log1p(gmpy2.exp(zlog)) 
        elif self.dist_name == CompoundDist.Poisson:
            num = gmpy2.log1p(-p1) + gmpy2.log(r)
            denum = gmpy2.log(gmpy2.exp(r) - 1.0)
            if denum.is_infinite():
                denum = r
        return gmpy2.exp(num - denum)
    
    def _rec_step(self, x, prevs: jnp.ndarray, r, p1, p2) -> jnp.ndarray:
        t = jnp.log(p1)
        t1 = prevs.at[x-1].get() + t
        t2 = prevs.at[x-2].get() + 2 * t
        if self.dist_name == CompoundDist.Binomial:
            zlog = jnp.log1p(-p1) + jnp.log(p2) - jnp.log1p(-p2)
            t = jnp.logaddexp(0.0, zlog)
            t1 -= t
            t2 -= 2 * t
        abz = self._calc_abz(x, r, p1, p2, log_z=False)
        alpha, beta = hyp_1f1_rec_terms(*abz) if self.dist_name == CompoundDist.Poisson else hyp_2f1_rec_terms(*abz)
        return logsumexp(jnp.array([t1, t2]), b=jnp.array([alpha, beta]))
    
    def _long_rec_step(self, x, prevs: np.ndarray, r, p1, p2) -> np.ndarray:
        t1 = prevs[-1] * p1
        t2 = prevs[-2] * p1 ** 2
        if self.dist_name == CompoundDist.Binomial:
            t = (1 + (1 - p1) * p2 / (1 - p2))
            t1 /= t
            t2 /= t ** 2
        abz = self._calc_abz(x, r, p1, p2, long=True, log_z=False)
        alpha, beta = hyp_1f1_rec_terms(*abz) if self.dist_name == CompoundDist.Poisson else hyp_2f1_rec_terms(*abz)
        return alpha * t1 + beta * t2
    
    def _rec_start(self, x, r, p1, p2) -> jnp.ndarray:
        abz = self._calc_abz(x, r, p1, p2)
        t1, t2 = hyp_log_1f1_rec_start(*abz) if self.dist_name == CompoundDist.Poisson else hyp_log_2f1_rec_start(*abz)
        t = jnp.log(p1)
        if self.dist_name == CompoundDist.Binomial:
            zlog = jnp.log1p(-p1) + jnp.log(p2) - jnp.log1p(-p2)
            t -= jnp.logaddexp(0.0, zlog)
        return jnp.array([t1, t2 + t]) + self._rec_mult(r, p1, p2)
    
    def _long_rec_start(self, x, r, p1, p2) -> np.array:
        r = gmpy2.mpfr(r)
        p1 = gmpy2.mpfr(p1); p2 = gmpy2.mpfr(p2)
        abz = self._calc_abz(x, r, p1, p2, long=True, log_z=False)
        t1, t2 = hyp_long_1f1_rec_start(*abz) if self.dist_name == CompoundDist.Poisson else hyp_2f1_rec_start(*abz)
        t = p1
        if self.dist_name == CompoundDist.Binomial:
            t /= 1.0 + (1 - p1) * p2 / (1 - p2)
        return np.array([t1, t2 * t]) #* self._long_rec_mult(r, p1, p2)

    def mean(self, r, p1, p2):
        if self.dist_name == CompoundDist.NB:
            m = jnp.log(p1) + jnp.log(r) - jnp.log1p(-p1) - jnp.log1p(-p2)
            m += jnp.log(p2) - jnp.log1p(-(1-p2) ** r)
        elif self.dist_name == CompoundDist.Binomial:
            m = jnp.log(r) + jnp.log(p1) - jnp.log1p(-p1)
            m += jnp.log(p2) - jnp.log1p(-(1-p2) ** r)
        elif self.dist_name == CompoundDist.Poisson:
            m = jnp.log(p1) + jnp.log(r) + r - jnp.log1p(-p1)
            m -= jax.lax.select(jnp.isfinite(jnp.exp(r)), jnp.log(jnp.exp(r) - 1), r)
        return jnp.exp(m)
    
    def second_moment(self, r, p1, p2):
        if self.dist_name == CompoundDist.NB:
            num = jnp.log(p1) + jnp.log(r) + jnp.log(1 + p1 - p2 + r * p1 * p2)
            denum = jnp.log1p(-p1) * 2 + jnp.log1p(-p2) * 2
            num += jnp.log(p2)
            denum += jnp.log1p(-(1 - p2) ** r)
        elif self.dist_name == CompoundDist.Binomial:
            num = jnp.log(r) + jnp.log(p1) + jnp.log(1 + p1 + (r - 1) * p1 * p2) 
            denum = jnp.log1p(-p1) * 2
            num += jnp.log(p2)
            denum += jnp.log1p(-(1 - p2) ** r)
        elif self.dist_name == CompoundDist.Poisson:
            num = r + jnp.log(p1) + jnp.log(r) + jnp.log(1 + p1 + p1 * r)
            denum = jnp.log1p(-p1) * 2 + jax.lax.select(jnp.isfinite(jnp.exp(r)), jnp.log(jnp.exp(r) - 1), r)
        return jnp.exp(num - denum)
    
    def var(self, r, p1, p2):
        x, x2 = self.mean(r, p1, p2), self.second_moment(r, p1, p2)
        return x2 - x ** 2
