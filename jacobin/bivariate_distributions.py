# -*- coding: utf-8 -*-
'''
The module contains some routines to evaluate bivariate distributions. The emphasis is made on
the reucrrent routines, that heavily exploit the fact that all PMFs here are what I'd call
a "semi-separable" functions, i.e. separable functions in terms of arguments x, y and x+y.
'''
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from abc import ABC
from .utils import recurrent_fun, kahan_add, betaln, log1p
from . import distributions as dists
from .distributions import CompoundDist


class SSDistribution(ABC):

    _rec_order_coverage = 1
    _rec_order_marginal = 1
    params = dict()

    # Values of the coverage part at 0, ..., _rec_order_coverage - 1 and
    # a state of the recurrence.
    def _coverage_start(self, *args, **kwargs) -> tuple:
        start = jnp.stack([self._coverage_part(float(i), *args, **kwargs) for i in range(self._rec_order_coverage)])
        return start, (start[-1], jnp.zeros_like(start[-1]))

    # Coverage part at n minus the one at n - 1 for n > 0.
    def _coverage_ratio(self, n, *args, **kwargs):
        raise NotImplementedError

    def _coverage_step(self, n, state, *args, **kwargs) -> tuple:
        state = kahan_add(*state, self._coverage_ratio(n, *args, **kwargs))
        return state[0], state

    def _marginal_start(self, swap: bool, *args, **kwargs) -> tuple:
        start = jnp.stack([self._marginal_part(float(i), swap, *args, **kwargs) for i in range(self._rec_order_marginal)])
        return start, (start[-1], jnp.zeros_like(start[-1]))

    # Marginal part at z minus the one at z - 1 for z > 0.
    def _marginal_ratio(self, swap: bool, z, *args, **kwargs):
        raise NotImplementedError

    def _marginal_step(self, swap: bool, z, state, *args, **kwargs) -> tuple:
        state = kahan_add(*state, self._marginal_ratio(swap, z, *args, **kwargs))
        return state[0], state

    def _coverage_part(self, n, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    def _marginal_part(self, z, swap: bool, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    # Marginal parts at 0, ..., max_sz - 1, -inf past max_z.
    def _recurrent_marginal_part(self, swap: bool, max_z, max_sz, *args, **kwargs):
        rec_start = lambda *args, **kwargs: self._marginal_start(swap, *args, **kwargs)
        rec_step = lambda *args, **kwargs: self._marginal_step(swap, *args, **kwargs)
        rec_order = self._rec_order_marginal
        part = recurrent_fun(rec_start, rec_step, rec_order, 0, max_z, max_sz, *args, **kwargs)
        return part

    # Coverage parts at 0, ..., max_sz - 1, -inf past max_n.
    def _recurrent_coverage_part(self, max_n, max_sz, *args, **kwargs):
        rec_start = self._coverage_start; rec_step = self._coverage_step;rec_order = self._rec_order_coverage
        cov_part = recurrent_fun(rec_start, rec_step, rec_order, 0, max_n, max_sz, *args, **kwargs)
        return cov_part

    def logpmf(self, x: jnp.ndarray, y: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        x, y = jnp.asarray(x, dtype=float), jnp.asarray(y, dtype=float)
        cov = self._coverage_part(x + y, *args, **kwargs)
        x_part = self._marginal_part(x, False, *args, **kwargs)
        y_part = self._marginal_part(y, True, *args, **kwargs)
        return cov + x_part + y_part

    def pmf(self, x: jnp.ndarray, y: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return jnp.exp(self.logpmf(x, y, *args, **kwargs))

    # Sums of the marginal parts of x and y, one table per order of the
    # two, each of the shape (max_sz_x, max_sz_y).
    def _recurrent_marginals(self, swaps: tuple, max_sz_x: int, max_sz_y: int, *args, **kwargs) -> list:
        res = list()
        for swap in swaps:
            x_part = self._recurrent_marginal_part(swap, None, max_sz_x, *args, **kwargs)
            y_part = self._recurrent_marginal_part(not swap, None, max_sz_y, *args, **kwargs)
            res.append(x_part.reshape(-1, 1) + y_part)
        return res

    # Adds the coverage part to a table of marginal parts and puts -inf
    # where x > max_x, y > max_y or x + y > max_n.
    def _recurrent_finish(self, res, max_x, max_y, max_n, *args, **kwargs):
        max_sz_x, max_sz_y = res.shape
        cov_part = self._recurrent_coverage_part(None, max_sz_x + max_sz_y - 1, *args, **kwargs)
        x, y = jnp.arange(max_sz_x).reshape(-1, 1), jnp.arange(max_sz_y)
        res = res + cov_part.at[x + y].get()
        inside = jnp.ones(res.shape, dtype=bool)
        for bound, t in ((max_x, x), (max_y, y), (max_n, x + y)):
            if bound is not None:
                inside = inside & (t <= bound)
        return jnp.where(inside, res, -jnp.inf)

    # Table of the shape (max_sz_x, max_sz_y) with the log-pmf at (x, y)
    # for x <= max_x, y <= max_y and x + y <= max_n, all inclusive, and
    # -inf elsewhere. A bound may be None or traced, the sizes are static.
    def logpmf_recurrent(self, max_x: int, max_y: int, max_n: int, max_sz_x: int, max_sz_y: int, *args, **kwargs):
        res = self._recurrent_marginals((False,), max_sz_x, max_sz_y, *args, **kwargs)[0]
        return self._recurrent_finish(res, max_x, max_y, max_n, *args, **kwargs)

    def pmf_recurrent(self, max_x: int, max_y: int, max_n: int, max_sz_x: int, max_sz_y: int, *args, **kwargs):
        return jnp.exp(self.logpmf_recurrent(max_x, max_y, max_n, max_sz_x, max_sz_y, *args, **kwargs))

    def logpmf_reccurent(self, *args, **kwargs):
        return self.logpmf_recurrent(*args, **kwargs)

    def pmf_reccurent(self, *args, **kwargs):
        return self.pmf_recurrent(*args, **kwargs)

# w f(x, y) + (1 - w) f(y, x) for the pmf f of dist.
class Mixture2D(SSDistribution):

    dist = None
    params = {'w': [(0.0, 1.0)]}

    def __init__(self, dist: SSDistribution):
        self.dist = dist
        self.params = {**self.params, **self.dist.params}

    def logpmf(self, x: jnp.ndarray, y: jnp.ndarray, w: float, *args, **kwargs) -> jnp.ndarray:
        d = self.dist
        x, y = jnp.asarray(x, dtype=float), jnp.asarray(y, dtype=float)
        cov = d._coverage_part(x + y, *args, **kwargs)
        w1_x_part = d._marginal_part(x, False, *args, **kwargs)
        w1_y_part = d._marginal_part(y, True, *args, **kwargs)
        w2_x_part = d._marginal_part(x, True, *args, **kwargs)
        w2_y_part = d._marginal_part(y, False, *args, **kwargs)
        logpmf1, logpmf2 = jnp.broadcast_arrays(w1_x_part + w1_y_part, w2_x_part + w2_y_part)
        w = jnp.asarray(w, dtype=float)
        return cov + logsumexp(jnp.array([logpmf1, logpmf2]), b=jnp.array([w, 1.0 - w]).reshape(-1, *([1] * logpmf1.ndim)), axis=0)

    def logpmf_recurrent(self, max_x: int, max_y: int, max_n: int, max_sz_x: int, max_sz_y: int, w: float, *args, **kwargs):
        d = self.dist
        logpmf1, logpmf2 = d._recurrent_marginals((False, True), max_sz_x, max_sz_y, *args, **kwargs)
        w = jnp.asarray(w, dtype=float)
        res = logsumexp(jnp.array([logpmf1, logpmf2]), b=jnp.array([w, 1.0 - w]).reshape(-1, 1, 1), axis=0)
        return d._recurrent_finish(res, max_x, max_y, max_n, *args, **kwargs)

class NegativeMultinomial(SSDistribution):
    params = {'r': [(0.0, None)], 'p1': [(0.0, 1.0)], 'p2': [(0.0, 1.0)]}

    def _coverage_part(self, n, r: float, p1: float, p2: float) -> jnp.ndarray:
        return gammaln(n + r) - gammaln(r) + log1p(-(p1 + p2)) * r

    def _marginal_part(self, z, swap: bool, r: float, p1: float, p2: float) -> jnp.ndarray:
        if swap:
            p1, p2 = p2, p1
        return jnp.log(p1) * z - gammaln(z + 1)

    def _coverage_ratio(self, n, r: float, p1: float, p2: float):
        return jnp.log(n + r - 1)

    def _marginal_ratio(self, swap: bool, z, r: float, p1: float, p2: float):
        if swap:
            p1, p2 = p2, p1
        return jnp.log(p1) - jnp.log(z)

class DirichletNegativeMultinomial(SSDistribution):
    params = {'r': [(0.0, None)], 'a0': [(0.0, None)], 'a1': [(0.0, None)], 'a2': [(0.0, None)]}

    def _coverage_part(self, n, r: float, a0: float, a1: float, a2: float) -> jnp.ndarray:
        return betaln(n + r, a0 + a1 + a2) - betaln(r, a0)

    def _marginal_part(self, z, swap: bool, r: float, a0: float, a1: float, a2: float) -> jnp.ndarray:
        if swap:
            a1, a2 = a2, a1
        return -jnp.log(z + a1) - betaln(z + 1, a1)

    def _coverage_ratio(self, n, r: float, a0: float, a1: float, a2: float):
        return jnp.log(n + r - 1) - jnp.log(n + r - 1 + a0 + a1 + a2)

    def _marginal_ratio(self, swap: bool, z, r: float, a0: float, a1: float, a2: float):
        if swap:
            a1, a2 = a2, a1
        return jnp.log(z - 1 + a1) - jnp.log(z)

# Negative multinomial distribution of (x, y) with probabilities p1, p2
# and a number of failures that follows a zero-truncated compound
# distribution with parameters r and p, see MCNB. x + y follows
# MCNB(r, p1 + p2, p), and x given x + y is binomial. p is not used by
# the Poisson compound distribution.
class MCNegativeMultinomial(SSDistribution):
    _rec_order_coverage = 2
    _rec_order_marginal = 1
    params = {'r': [(0.0, None)], 'p1': [(0.0, 1.0)], 'p2': [(0.0, 1.0)], 'p': [(0.0, 1.0)]}
    dist_name = None

    def __init__(self, compound_dist=CompoundDist.NB):
        self.dist_name = compound_dist
        self.params = dict(self.params)
        self._mcnb = dists.MCNB(compound_dist)
        if compound_dist == CompoundDist.Poisson:
            self.params['p'] = None

    # log-pmf of x + y times n!, the powers of p1 + p2 taken out.
    def _coverage_part(self, n, r: float, p1: float, p2: float, p: float=None) -> jnp.ndarray:
        return self._mcnb.logpmf(n, r, p1 + p2, p) + gammaln(n + 1) - n * jnp.log(p1 + p2)

    def _marginal_part(self, z, swap: bool, r: float, p1: float, p2: float, p: float=None) -> jnp.ndarray:
        if swap:
            p1, p2 = p2, p1
        return jnp.log(p1) * z - gammaln(z + 1)

    def _recurrent_coverage_part(self, max_n, max_sz, r: float, p1: float, p2: float, p: float=None):
        n = jnp.arange(max_sz, dtype=float)
        res = self._mcnb.logpmf_recurrent(0, max_n, max_sz, r, p1 + p2, p)
        return res + gammaln(n + 1) - n * jnp.log(p1 + p2)

    def _marginal_ratio(self, swap: bool, z, r: float, p1: float, p2: float, p: float=None):
        if swap:
            p1, p2 = p2, p1
        return jnp.log(p1) - jnp.log(z)
