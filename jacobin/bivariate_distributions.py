# -*- coding: utf-8 -*-
'''
The module contains some routines to evaluate bivariate distributions. The emphasis is made on
the reucrrent routines, that heavily exploit the fact that all PMFs here are what I'd call
a "semi-separable" functions, i.e. separable functions in terms of arguments x, y and x+y.
'''
import jax.numpy as jnp
import jax
from jax.scipy.special import gamma, gammaln, beta, betaln, logsumexp
from jax.numpy import logaddexp
from jax.lax import fori_loop
from abc import ABC
from .utils import recurrent_fun, hankel
from . import distributions as dists
from .distributions import Distribution, CompoundDist
from .hyp import hyp_log_2f1_rec_start, hyp_2f1_rec_terms, hyp_log_1f1_rec_start, hyp_1f1_rec_terms


class SSDistribution(ABC):
    
    _rec_order_coverage = 1
    _rec_order_marginal = 1
    params = dict()

    def _coverage_start(self, n, *args, **kwargs):
        raise NotImplementedError
    
    def _coverage_step(self, n, prevs: jnp.ndarray, *args, **kwargs):
        raise NotImplementedError
    
    def _marginal_start(self, swap: bool, z, *args, **kwargs):
        raise NotImplementedError
    
    def _marginal_step(self, swap: bool, z, prevs: jnp.ndarray, *args, **kwargs):
        raise NotImplementedError
    
    def _coverage_part(self, n, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def _marginal_part(self, z, swap: bool, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def _recurrent_marginal_part(self, swap: bool, max_z, max_sz, *args, **kwargs):
        rec_start = lambda *args, **kwargs: self._marginal_start(swap, *args, **kwargs)
        rec_step = lambda *args, **kwargs: self._marginal_step(swap, *args, **kwargs)
        rec_order = self._rec_order_marginal
        part = recurrent_fun(rec_start, rec_step, rec_order, 0, max_z, max_sz, *args, **kwargs)
        return part
    
    def _recurrent_coverage_part(self, max_n, max_sz, *args, **kwargs):
        rec_start = self._coverage_start; rec_step = self._coverage_step;rec_order = self._rec_order_coverage
        cov_part = recurrent_fun(rec_start, rec_step, rec_order, 0, max_n, max_sz, *args, **kwargs)
        return cov_part
    
    def logpmf(self, x: jnp.ndarray, y: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        cov = self._coverage_part(x + y, *args, **kwargs)
        x_part = self._marginal_part(x, False, *args, **kwargs)
        y_part = self._marginal_part(y, True, *args, **kwargs)
        return cov + x_part + y_part
        
    def pmf(self, x: jnp.ndarray, y: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return jnp.exp(self.logpmf(x, y, *args, **kwargs))
    
    def logpmf_reccurent(self, max_x: int, max_y: int, max_n: int, max_sz_x: int, max_sz_y: int, *args, **kwargs):
        cov_part = self._recurrent_coverage_part(max_n, max_sz_x + max_sz_y - 1, *args, **kwargs)
        x_part = self._recurrent_marginal_part(False, max_x, max_sz_x, *args, **kwargs)
        y_part = self._recurrent_marginal_part(True, max_y, max_sz_y, *args, **kwargs)

        # h = hankel(cov_part.at[:max_sz_x].get(), cov_part.at[max_sz_x:].get())
        # return cov_part
        res = x_part.reshape(-1, 1) + y_part #+ h
        def body_x(x, carry):
            def body_y(y, carry):
                return carry.at[x, y].add(cov_part.at[x + y].get())
            return fori_loop(0, max_y, body_y, carry)
        return fori_loop(0, max_x, body_x, res)
    
    def pmf_reccurent(self, max_x: int, max_y: int, max_n: int, max_sz_x: int, max_sz_y: int, *args, **kwargs): 
        return jnp.exp(self.logpmf_reccurent(max_x, max_y, max_n, max_sz_x, max_sz_y, *args, **kwargs))

class Mixture2D(SSDistribution):
    
    dist = None
    params = {'w': [(0.0, 1.0)]}
    
    def __init__(self, dist: SSDistribution):
        self.dist = dist
        self.params.update(self.dist.params)
    
    def logpmf(self, x: jnp.ndarray, y: jnp.ndarray, w: float, *args, **kwargs) -> jnp.ndarray:
        d = self.dist
        cov = d._coverage_part(x + y, *args, **kwargs)
        w1_x_part = d._marginal_part(x, False, *args, **kwargs)
        w1_y_part = d._marginal_part(y, True, *args, **kwargs)
        w2_x_part = d._marginal_part(x, True, *args, **kwargs)
        w2_y_part = d._marginal_part(y, False, *args, **kwargs)
        logpmf1 = cov + w1_x_part + w1_y_part 
        logpmf2 = cov + w2_x_part + w2_y_part
        logpmf1 = jnp.array(logpmf1)
        return logsumexp(jnp.array([logpmf1, logpmf2]), b=jnp.array([w, 1.0 - w]).reshape(-1, *(1) * len(logpmf1.shape)), axis=0)
    
    def logpmf_reccurent(self, max_x: int, max_y: int, max_n: int, max_sz_x: int, max_sz_y: int, w: float, *args, **kwargs):
        d = self.dist
        cov_part = d._recurrent_coverage_part(max_n, max_sz_x + max_sz_y - 1, *args, **kwargs)
        w1_x_part = d._recurrent_marginal_part(False, max_x, max_sz_x, *args, **kwargs)
        w1_y_part = d._recurrent_marginal_part(True, max_y, max_sz_y, *args, **kwargs)
        w2_x_part = d._recurrent_marginal_part(True, max_x, max_sz_x, *args, **kwargs)
        w2_y_part = d._recurrent_marginal_part(False, max_y, max_sz_y, *args, **kwargs)
        
        logpmf1 = w1_x_part.reshape(-1, 1) + w1_y_part
        logpmf2 = w2_x_part.reshape(-1, 1) + w2_y_part
        def body_x(x, carry):
            def body_y(y, carry):
                return carry[0].at[x, y].add(cov_part.at[x + y].get()), carry[1].at[x, y].add(cov_part.at[x + y].get())
            return fori_loop(0, max_y, body_y, carry)
        logpmf1, logpmf2 = fori_loop(0, max_x, body_x, (logpmf1, logpmf2))
        return logsumexp(jnp.array([logpmf1, logpmf2]), b=jnp.array([w, 1.0 - w]).reshape(-1, *([1] * len(logpmf1.shape))), axis=0)

class NegativeMultinomial(SSDistribution):
    params = {'r': [(0.0, None)], 'p1': [(0.0, 1.0)], 'p2': [(0.0, 1.0)]}
    
    def _coverage_part(self, n, r: float, p1: float, p2: float) -> jnp.ndarray:
        return gammaln(n + r) - gammaln(r) + jnp.log1p(-(p1 + p2)) * r
    
    def _marginal_part(self, z, swap: bool, r: float, p1: float, p2: float) -> jnp.ndarray:
        if swap:
            p1, p2 = p2, p1
        return jnp.log(p1) * z - gammaln(z + 1)
    
    def _coverage_start(self, n,  r: float, p1: float, p2: float):
        z0 = jnp.log1p(-(p1 + p2)) * r + gammaln(n + r) - gammaln(r)
        return jnp.array([z0])
    
    def _coverage_step(self, n, prevs: jnp.ndarray, r: float, p1: float, p2: float):
        return prevs.at[n - 1].get() + jnp.log(n + r - 1)
    
    def _marginal_start(self, swap: bool, z, r: float, p1: float, p2: float):
        if swap:
            p1, p2 = p2, p1
        return jnp.array([jnp.log(p1) * z - gammaln(z + 1)])
    
    def _marginal_step(self, swap: bool, z,  prevs: jnp.ndarray, r: float, p1: float, p2: float):
        if swap:
            p1, p2 = p2, p1
        return prevs.at[z - 1].get() + jnp.log(p1) - jnp.log(z)

class DirichletNegativeMultinomial(SSDistribution):
    params = {'r': [(0.0, None)], 'a0': [(0.0, None)], 'a1': [(0.0, None)], 'a2': [(0.0, None)]}

    def _coverage_part(self, n, r: float, a0: float, a1: float, a2: float) -> jnp.ndarray:
        return betaln(n + r, a0 + a1 + a2) - betaln(r, a0)
    
    def _marginal_part(self, z, swap: bool, r: float, a0: float, a1: float, a2: float) -> jnp.ndarray:
        if swap:
            a1, a2 = a2, a1
        return -jnp.log(z + a1) - betaln(z + 1, a1)

    def _coverage_start(self, n,  r: float, a0: float, a1: float, a2: float):
        n0 = betaln(n + r, a0 + a1 + a2) - betaln(r, a0) - gammaln(a0) - gammaln(a1) - gammaln(a2)
        return jnp.array([n0])
    
    def _coverage_step(self, n, prevs: jnp.ndarray, r: float, a0: float, a1: float, a2: float):
        return prevs.at[n - 1].get() + jnp.log(n + r - 1) - jnp.log(n + r - 1 + a0 + a1 + a2)
    
    def _marginal_start(self, swap: bool, z, r: float, a0: float, a1: float, a2: float):
        if swap:
            a1, a2 = a2, a1
        z0 = gammaln(z + a1) - gammaln(z + 1)
        return jnp.array([z0])
    
    def _marginal_step(self, swap: bool, z,  prevs: jnp.ndarray, r: float, a0: float, a1: float, a2: float):
        if swap:
            a1, a2 = a2, a1
        return prevs.at[z - 1].get() + jnp.log(z - 1 + a1) - jnp.log(z)

class MCNegativeMultinomial(SSDistribution):
    _rec_order_coverage = 2
    _rec_order_marginal = 1
    params = {'r': [(0.0, None)], 'p1': [(0.0, 1.0)], 'p2': (0.0, 1.0), 'p': [(0.0, 1.0)]}
    dist_name = None
    
    def __init__(self, compound_dist=CompoundDist.NB):
        self.dist_name = compound_dist
        if compound_dist == CompoundDist.Poisson:
            self.params['p'] = None
    
    def _coverage_start(self, n,  r: float, p1: float, p2: float, p: float):
        if self.dist_name == CompoundDist.NB:
            mult = jnp.log1p(-(p1 + p2)) + jnp.log1p(-p) * r + jnp.log(p) + jnp.log(r)
            mult -= logsumexp(jnp.array([0.0, jnp.log1p(-p) * r]), b=jnp.array([1, -1]))
            zlog = jnp.log(p) + jnp.log1p(-(p1 + p2))
            hyp_term = jnp.array(hyp_log_2f1_rec_start(a=r + 1, b=n + 1, log_z=zlog))
        elif self.dist_name == CompoundDist.Binomial:
            mult = jnp.log(p) + jnp.log1p(-p) * (r - 1) + jnp.log1p(-(p1 + p2)) + jnp.log(r)
            mult -= logsumexp(jnp.array([0.0, jnp.log1p(-p) * r]), b=jnp.array([1, -1]))
            mult -= (logaddexp(0.0, jnp.log(p) + jnp.log1p(-(p1 + p2)) - jnp.log1p(-p))) * (n + 1)
            zlog = jnp.log(p) + jnp.log1p(-(p1 + p2)) - jnp.log1p(-(p1 + p2) * p)
            hyp_term = jnp.array(hyp_log_2f1_rec_start(a=1 + r, b=n + 1, log_z=zlog ))
        elif self.dist_name == CompoundDist.Poisson:
            mult = jnp.log1p(-(p1 + p2)) - r + jnp.log(r)
            zlog = jnp.log1p(-(p1 + p2)) + jnp.log(r)
            hyp_term = jnp.array(hyp_log_1f1_rec_start(a=n + 1, log_z=zlog))
        n0 = hyp_term + mult
        return n0
    
    def _coverage_step(self, n, prevs: jnp.ndarray, r: float, p1: float, p2: float, p: float):
        t1, t2 = prevs.at[n - 1].get(), prevs.at[n - 2].get()
        t1 += jnp.log(n)
        t2 += jnp.log(n) + jnp.log(n - 1)
        if self.dist_name == CompoundDist.NB:
            a, b, z = 1 + r, 1 + n, p * (1 - p1 - p2)
            alpha, beta = hyp_2f1_rec_terms(a=a, b=b, z=z)
        elif self.dist_name == CompoundDist.Binomial:
            a, b = 1 + r, 1 + n
            zlog = jnp.log(p) + jnp.log1p(-(p1 + p2)) - jnp.log1p(-(p1 + p2) * p)
            t = logaddexp(0.0, jnp.log(p) + jnp.log1p(-(p1 + p2)) - jnp.log1p(-p))
            t1 -= t
            t2 -= 2 * t
            alpha, beta = hyp_2f1_rec_terms(a=a, b=b, z=jnp.exp(zlog))
        elif self.dist_name == CompoundDist.Poisson:
            a, z = n + 1, (1 - p1 - p2) * r
            alpha, beta = hyp_1f1_rec_terms(a=a, z=z)
        res = logsumexp(jnp.array([t1, t2]), b=jnp.array([alpha, beta]))  
        return res

    def _marginal_start(self, swap: bool, z, r: float, p1: float, p2: float, p: float):
        return jnp.array([0.0])
    
    def _marginal_step(self, swap: bool, z,  prevs: jnp.ndarray, r: float, p1: float, p2: float, p: float):
        if swap:
            p1, p2 = p2, p1
        return prevs.at[z - 1].get() + jnp.log(p1) - jnp.log(z)