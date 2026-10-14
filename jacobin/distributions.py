#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from scipy.stats import binom as scipy_binom, poisson as scipy_poisson, nbinom as scipy_nb,\
    betabinom as scipy_betabinom
import gmpy2
import numpy as np
from abc import ABC
from enum import Enum

from .betainc import betainc_parts
from . import hyp
from .utils import recurrent_fun, kahan_add, betaln, log1p, is_static, long_context, long_points, to_mpfr

class CompoundDist(Enum):
    NB = 'NB'
    Binomial = 'Binomial'
    Poisson = 'Poisson'

def logbinomial(n, x):
    n = jnp.array(n)
    x = jnp.array(x)
    return -betaln(x + 1.0, n - x + 1.0) - jnp.log(n + 1.0)

def long_logbeta(a, b):
    gammaln = lambda z: gmpy2.lgamma(z)[0]
    return gammaln(a) + gammaln(b) - gammaln(a + b)

def gmpy2_bincoef(n, x):
    return gmpy2.gamma(n + 1) / (gmpy2.gamma(n - x + 1) * gmpy2.fac(x))

def _float(*args):
    return tuple(jnp.asarray(t, dtype=float) for t in args)

# log(1 - exp(t)), -inf for t >= 0.
def log1mexp(t):
    neg = t < 0
    t = jnp.where(neg, t, -1.0)
    res = jnp.where(t > -0.6931471805599453, jnp.log(-jnp.expm1(t)), log1p(-jnp.exp(t)))
    return jnp.where(neg, res, -jnp.inf)

# log(t), -inf for t <= 0.
def _log_positive(t):
    pos = t > 0
    return jnp.where(pos, jnp.log(jnp.where(pos, t, 1.0)), -jnp.inf)

# A cdf is stored as a pair (lg, swap): it is exp(lg) where swap is
# false and 1 - exp(lg) where swap is true. Returns the cdf, or the sf
# for complement=True, or their logarithms for log=True.
def _from_parts(lg, swap, complement=False, log=False):
    direct = swap if complement else jnp.logical_not(swap)
    if log:
        return jnp.where(direct, lg, log1mexp(lg))
    return jnp.where(direct, jnp.exp(lg), -jnp.expm1(lg))

# Truncation bound: None if it is missing, a Python int if it is known
# before tracing, an array otherwise.
def _bound(v, upper: bool):
    if v is None:
        return None
    if not is_static(v):
        return jnp.asarray(v, dtype=float)
    v = float(v)
    if (upper and v == float('inf')) or (not upper and v <= 0):
        return None
    return math.floor(v) if upper else math.ceil(v)


class Distribution(ABC):

    rec_order = 1
    params = dict()
    # Precision in bits of the long algebra functions.
    long_precision = 1024
    # A tail of a long sum is cut at this relative accuracy in bits.
    long_tail_bits = 128
    long_max_iter = 10 ** 7

    def logpmf(self, x: jnp.ndarray, *args, **kwargs):
        raise NotImplementedError

    def pmf(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return jnp.exp(self.logpmf(x, *args, **kwargs))

    def logcdf(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return _from_parts(*self._cdf_parts(x, *args, **kwargs), log=True)

    def cdf(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return _from_parts(*self._cdf_parts(x, *args, **kwargs))

    # P(X > x).
    def sf(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return _from_parts(*self._cdf_parts(x, *args, **kwargs), complement=True)

    def logsf(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return _from_parts(*self._cdf_parts(x, *args, **kwargs), complement=True, log=True)

    def _cdf_parts(self, x: jnp.ndarray, *args, **kwargs) -> tuple:
        raise NotImplementedError

    # cdf and sf together, from one evaluation of the sum behind them
    # where the distribution has one.
    def _cdf_sf(self, x: jnp.ndarray, *args, **kwargs) -> tuple:
        try:
            lg, swap = self._cdf_parts(x, *args, **kwargs)
        except NotImplementedError:
            return self.cdf(x, *args, **kwargs), self.sf(x, *args, **kwargs)
        return _from_parts(lg, swap), _from_parts(lg, swap, complement=True)

    # Each long function takes a scalar or an array x of integers in any
    # order and scalar parameters. It returns mpfr numbers of the given
    # precision in bits, long_precision by default, as a scalar or as
    # an object array of the shape of x. The cost is max(x) steps.
    def long_pmf(self, x: np.ndarray, *args, precision=None, **kwargs) -> np.ndarray:
        return self._long_public(0, x, precision, args, kwargs)

    def long_cdf(self, x: np.ndarray, *args, precision=None, **kwargs) -> np.ndarray:
        return self._long_public(1, x, precision, args, kwargs)

    # P(X > x). Tails are summed directly where they decay geometrically.
    def long_sf(self, x: np.ndarray, *args, precision=None, **kwargs) -> np.ndarray:
        return self._long_public(2, x, precision, args, kwargs)

    def mean(self, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    def var(self, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    # seed is None, an integer or a numpy Generator.
    def sample(self, size: int, *args, seed=None, **kwargs) -> np.ndarray:
        raise NotImplementedError

    # Table of max_sz log-pmf values at 0, 1, ..., max_sz - 1 computed
    # by a recurrence from zero for scalar parameters. Entries outside
    # of [min_x, max_x], both inclusive, are -inf. max_x=None stands for
    # max_sz - 1. The bounds may be traced, max_sz is static.
    def logpmf_recurrent(self, min_x: int, max_x: int, max_sz: int, *args, **kwargs):
        return recurrent_fun(self._rec_start, self._rec_step, self.rec_order,
                             min_x, max_x, max_sz, *args, **kwargs)

    def pmf_recurrent(self, min_x: int, max_x: int, max_sz: int, *args, **kwargs):
        return jnp.exp(self.logpmf_recurrent(min_x, max_x, max_sz, *args, **kwargs))

    # log pmf(x) - log pmf(x - 1) for x > 0.
    def _log_ratio(self, x, *args, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    # The state is the last value with the compensation of its sum.
    def _rec_step(self, x, state, *args, **kwargs) -> tuple:
        state = kahan_add(*state, self._log_ratio(x, *args, **kwargs))
        return state[0], state

    def _rec_start(self, *args, **kwargs) -> tuple:
        start = jnp.stack([self.logpmf(float(i), *args, **kwargs) for i in range(self.rec_order)])
        return start, (start[-1], jnp.zeros_like(start[-1]))

    # Parameters as mpfr numbers, with whatever else the long recurrence needs.
    def _long_setup(self, *args, **kwargs) -> tuple:
        raise NotImplementedError

    # pmf at 0, ..., rec_order - 1.
    def _long_start(self, ctx: tuple) -> list:
        raise NotImplementedError

    # pmf(x) / pmf(x - 1) for x > 0.
    def _long_ratio(self, x: int, ctx: tuple) -> gmpy2.mpfr:
        raise NotImplementedError

    def _long_step(self, x: int, prevs: list, ctx: tuple) -> gmpy2.mpfr:
        return prevs[-1] * self._long_ratio(x, ctx)

    # The last point of the support, None if there is none.
    def _long_support(self, ctx: tuple):
        return None

    # Limit of pmf(x) / pmf(x - 1) as x grows if it is below 1. None for
    # a heavy tail, its sum is found as a complement then.
    def _long_tail_ratio(self, ctx: tuple):
        return None

    def _long_iter(self, ctx: tuple):
        prevs = list(self._long_start(ctx))
        yield from prevs
        x = self.rec_order
        while True:
            f = self._long_step(x, prevs, ctx)
            prevs = prevs[1:] + [f]
            yield f
            x += 1

    # One pass of the recurrence at the precision of the current context.
    # Returns pmf, cdf and sf at the sorted points pts as three dicts,
    # or None if the precision does not suffice for the complement.
    def _long_run(self, pts: list, need_sf: bool, args, kwargs):
        ctx = self._long_setup(*args, **kwargs)
        zero, one = gmpy2.mpfr(0), gmpy2.mpfr(1)
        support = self._long_support(ctx)
        last = pts[-1] if support is None else min(pts[-1], support)
        inner = [k for k in pts if k <= last]
        want = set(inner)
        pmf, cdf, sf, part = dict(), dict(), dict(), dict()
        total = acc = f = zero
        it = self._long_iter(ctx)
        for k in range(last + 1):
            f = next(it)
            total += f
            acc += f
            if k in want:
                pmf[k], cdf[k], part[k] = f, total, acc
                acc = zero
        for k in pts[len(inner):]:
            pmf[k], cdf[k], sf[k] = zero, one, zero
        if not need_sf:
            return pmf, cdf, sf
        prec = gmpy2.get_context().precision
        rho = self._long_tail_ratio(ctx)
        # The mass past the last requested point of a finite support.
        tail = acc
        if support is not None:
            for k in range(last + 1, support + 1):
                tail += next(it)
        elif rho is None:
            tail = one - total
            if tail <= gmpy2.exp2(self.long_tail_bits + (last + 2).bit_length() - prec):
                return None
        else:
            tol = gmpy2.exp2(-self.long_tail_bits)
            for _ in range(self.long_max_iter):
                prev, f = f, next(it)
                tail += f
                b = max(f / prev, rho)
                if b < 1 and f * b <= tail * tol * (1 - b):
                    break
            else:
                raise ArithmeticError('The sum of the tail did not converge.')
        for k in reversed(inner):
            sf[k] = tail
            tail = tail + part[k]
        return pmf, cdf, sf

    # pmf, cdf and sf at the sorted non-negative points pts as dicts of
    # mpfr numbers of the given precision. The working precision is
    # doubled while a complement loses all of its digits.
    def _long_tables(self, pts: list, precision: int, need_sf: bool, args, kwargs) -> tuple:
        prec = precision
        while True:
            with long_context(prec):
                res = self._long_run(pts, need_sf, args, kwargs)
            if res is not None:
                break
            prec *= 2
            if prec > precision * 1024:
                raise ArithmeticError('The complement of the cdf is below the reachable precision.')
        if prec != precision:
            res = tuple({k: gmpy2.mpfr(v, precision) for k, v in t.items()} for t in res)
        return res

    def _long_public(self, kind: int, x, precision, args, kwargs):
        precision = self.long_precision if precision is None else int(precision)
        xs, shape, scalar = long_points(x)
        pts = sorted(set(t for t in xs if t >= 0))
        table = self._long_tables(pts, precision, kind == 2, args, kwargs)[kind] if pts else dict()
        default = gmpy2.mpfr(1 if kind == 2 else 0, precision)
        res = np.empty(len(xs), dtype=object)
        for i, t in enumerate(xs):
            res[i] = table[t] if t >= 0 else default
        return res[0] if scalar else res.reshape(shape)


# Distribution dist conditioned on left <= X <= right, both bounds are
# inclusive. They are keyword arguments of every method, None stands
# for no bound. A Python int left up to max_direct has the mass below
# it summed directly from the pmf, cdf and sf of dist are used otherwise.
class TruncatedDistribution(Distribution):

    params = None

    def __init__(self, dist: Distribution, clip_bounds=True, eps=None, max_direct=64):
        self.dist = dist
        self.params = dist.params
        self.rec_order = dist.rec_order
        self.clip_bounds = clip_bounds
        self.eps = eps
        self.max_direct = max_direct

    def _is_direct(self, bound) -> bool:
        return isinstance(bound, int) and bound <= self.max_direct

    # pmf of the base distribution at lo, ..., hi - 1 along the first
    # axis. One batched call keeps the graph, and the time to compile
    # its derivatives, independent of the number of points.
    def _direct(self, lo: int, hi: int, args, kwargs) -> jnp.ndarray:
        return jax.vmap(lambda i: self.dist.pmf(i, *args, **kwargs))(jnp.arange(lo, hi, dtype=float))

    # P(X < lo) for the base distribution as a sum of its pmf.
    def _below(self, lo: int, args, kwargs) -> jnp.ndarray:
        return self._direct(0, lo, args, kwargs).sum(axis=0)

    # P(X < lo) and P(X >= lo) for the base distribution.
    def _left_side(self, lo, args, kwargs) -> tuple:
        if lo is None:
            return 0.0, 1.0
        if self._is_direct(lo):
            below = self._below(lo, args, kwargs)
            return below, 1.0 - below
        return self.dist._cdf_sf(lo - 1.0, *args, **kwargs)

    # P(X <= hi) and P(X > hi) for the base distribution, an infinite
    # traced bound included.
    def _right_side(self, hi, args, kwargs) -> tuple:
        dist = self.dist
        if hi is None:
            return 1.0, 0.0
        if isinstance(hi, int):
            return dist._cdf_sf(float(hi), *args, **kwargs)
        fin = jnp.isfinite(hi)
        cdf, sf = dist._cdf_sf(jnp.where(fin, hi, 0.0), *args, **kwargs)
        return jnp.where(fin, cdf, 1.0), jnp.where(fin, sf, 0.0)

    # Mass between two points given by their pairs of cdf and sf. It is
    # a difference of the cdf values or of the sf values, whichever
    # pair is smaller, so that no digits are lost in either tail.
    @staticmethod
    def _mass(left: tuple, right: tuple) -> jnp.ndarray:
        return jnp.where(jnp.asarray(right[0]) <= 0.5, right[0] - left[0], left[1] - right[1])

    def _lognorm(self, lo, hi, args, kwargs) -> jnp.ndarray:
        dist = self.dist
        if lo is None and hi is None:
            return 0.0
        if hi is None:
            if self._is_direct(lo):
                res = log1p(-self._below(lo, args, kwargs))
            else:
                res = dist.logsf(lo - 1.0, *args, **kwargs)
        elif self._is_direct(hi) and (lo is None or isinstance(lo, int)):
            res = jnp.log(self._direct(lo or 0, hi + 1, args, kwargs).sum(axis=0))
        else:
            res = jnp.log(self._mass(self._left_side(lo, args, kwargs), self._right_side(hi, args, kwargs)))
        if self.eps is not None:
            res = jnp.maximum(res, jnp.log(self.eps))
        return res

    def _inside(self, x, lo, hi):
        res = jnp.ones(jnp.shape(x), dtype=bool)
        if lo is not None:
            res = res & (x >= lo)
        if hi is not None:
            res = res & (x <= hi)
        return res

    def logpmf(self, x, *args, left=None, right=None, **kwargs):
        lo, hi = _bound(left, False), _bound(right, True)
        x = jnp.asarray(x, dtype=float)
        res = self.dist.logpmf(x, *args, **kwargs) - self._lognorm(lo, hi, args, kwargs)
        if self.clip_bounds:
            res = jnp.where(self._inside(x, lo, hi), res, -jnp.inf)
        return res

    def cdf(self, x, *args, left=None, right=None, **kwargs):
        lo, hi = _bound(left, False), _bound(right, True)
        x = jnp.asarray(x, dtype=float)
        at_x = self.dist._cdf_sf(x, *args, **kwargs)
        res = self._mass(self._left_side(lo, args, kwargs), at_x)
        res = jnp.clip(res / jnp.exp(self._lognorm(lo, hi, args, kwargs)), 0.0, 1.0)
        if lo is not None:
            res = jnp.where(x < lo, 0.0, res)
        if hi is not None:
            res = jnp.where(x >= hi, 1.0, res)
        return res

    def sf(self, x, *args, left=None, right=None, **kwargs):
        lo, hi = _bound(left, False), _bound(right, True)
        x = jnp.asarray(x, dtype=float)
        at_x = self.dist._cdf_sf(x, *args, **kwargs)
        res = self._mass(at_x, self._right_side(hi, args, kwargs))
        res = jnp.clip(res / jnp.exp(self._lognorm(lo, hi, args, kwargs)), 0.0, 1.0)
        if lo is not None:
            res = jnp.where(x < lo, 1.0, res)
        if hi is not None:
            res = jnp.where(x >= hi, 0.0, res)
        return res

    def logcdf(self, x, *args, left=None, right=None, **kwargs):
        return jnp.log(self.cdf(x, *args, left=left, right=right, **kwargs))

    # Without a right bound the logarithm of the sf of dist is used, so
    # the result is accurate far in the right tail.
    def logsf(self, x, *args, left=None, right=None, **kwargs):
        lo, hi = _bound(left, False), _bound(right, True)
        if hi is not None:
            return jnp.log(self.sf(x, *args, left=left, right=right, **kwargs))
        x = jnp.asarray(x, dtype=float)
        res = self.dist.logsf(x, *args, **kwargs) - self._lognorm(lo, hi, args, kwargs)
        res = jnp.minimum(res, 0.0)
        if lo is not None:
            res = jnp.where(x < lo, 0.0, res)
        return res

    # pmf of the base distribution at 0, ..., n - 1 along the last axis.
    def _table(self, n: int, args, kwargs) -> jnp.ndarray:
        dist = self.dist
        if n <= self.max_direct:
            return jnp.moveaxis(self._direct(0, n, args, kwargs), 0, -1)
        names = list(kwargs)
        def fun(*params):
            kw = dict(zip(names, params[len(args):]))
            return dist.pmf_recurrent(0, n - 1, n, *params[:len(args)], **kw)
        params = args + tuple(kwargs.values())
        try:
            params = _float(*params)
        except (TypeError, ValueError):
            return dist.pmf_recurrent(0, n - 1, n, *args, **kwargs)
        signature = ','.join(['()'] * len(params)) + '->(n)'
        return jnp.vectorize(fun, signature=signature)(*params)

    def _static_bounds(self, left, right) -> tuple:
        lo, hi = _bound(left, False), _bound(right, True)
        for t in (lo, hi):
            if t is not None and not isinstance(t, int):
                raise TypeError('Moments and samples of a truncated distribution need bounds known before tracing.')
        return lo, hi

    # k-th moment about zero. Without a right bound it is the moment
    # full of the base distribution with the part below left taken out.
    def _moment(self, k: int, full, lo, hi, args, kwargs) -> jnp.ndarray:
        if hi is None:
            if lo is None:
                return full
            pmf = self._table(lo, args, kwargs)
            i = jnp.arange(lo, dtype=float)
            return (full - (pmf * i ** k).sum(axis=-1)) / (1.0 - pmf.sum(axis=-1))
        pmf = self._table(hi + 1, args, kwargs)
        i = jnp.arange(hi + 1, dtype=float)
        pmf = jnp.where(i >= (lo or 0), pmf, 0.0)
        return (pmf * i ** k).sum(axis=-1) / pmf.sum(axis=-1)

    def mean(self, *args, left=None, right=None, **kwargs):
        lo, hi = self._static_bounds(left, right)
        full = self.dist.mean(*args, **kwargs) if hi is None else None
        return self._moment(1, full, lo, hi, args, kwargs)

    def var(self, *args, left=None, right=None, **kwargs):
        lo, hi = self._static_bounds(left, right)
        full = None
        if hi is None:
            mean = self.dist.mean(*args, **kwargs)
            full = self.dist.var(*args, **kwargs) + mean ** 2
            mean = self._moment(1, mean, lo, hi, args, kwargs)
        else:
            mean = self._moment(1, None, lo, hi, args, kwargs)
        return self._moment(2, full, lo, hi, args, kwargs) - mean ** 2

    # With a right bound and scalar parameters the cdf is inverted on
    # a table of the pmf. Otherwise samples of the base distribution
    # that fall outside of the bounds are rejected.
    def sample(self, size: int, *args, left=None, right=None, seed=None, **kwargs) -> np.ndarray:
        lo, hi = self._static_bounds(left, right)
        rng = np.random.default_rng(seed)
        n = int(np.prod(size))
        if hi is not None:
            pmf = np.array(self._table(hi + 1, args, kwargs), dtype=float)
            if pmf.ndim == 1:
                pmf[:lo or 0] = 0.0
                cdf = np.cumsum(pmf)
                if not cdf[-1] > 0:
                    raise ValueError('The truncated region holds no mass.')
                return np.searchsorted(cdf / cdf[-1], rng.random(size), side='right')
        prob = float(np.min(np.exp(self._lognorm(lo, hi, args, kwargs))))
        if not prob > 1e-7:
            raise ValueError('The truncated region is too unlikely to sample from by rejection.')
        res = np.empty(0, dtype=int)
        while len(res) < n:
            draw = np.asarray(self.dist.sample(int((n - len(res)) / prob * 1.1) + 16, *args, seed=rng, **kwargs))
            draw = draw.reshape(-1)
            keep = np.ones(len(draw), dtype=bool)
            if lo is not None:
                keep &= draw >= lo
            if hi is not None:
                keep &= draw <= hi
            res = np.concatenate([res, draw[keep].astype(int)])
        return res[:n].reshape(size)

    # The table of the base distribution is computed from zero up to
    # max_sz - 1 and is normalized by sums over that table. With a right bound
    # at or above max_sz the rest of the mass comes from the sf of dist.
    def logpmf_recurrent(self, min_x: int, max_x: int,  max_sz: int, *args, left=None, right=None, **kwargs):
        dist = self.dist
        lo, hi = _bound(left, False), _bound(right, True)
        res = dist.logpmf_recurrent(0, max_sz - 1, max_sz, *args, **kwargs)
        inds = jnp.arange(max_sz)
        inside = self._inside(inds, lo, hi)
        if hi is None:
            if lo is None:
                denum = 0.0
            else:
                below = jnp.where(inside, 0.0, jnp.exp(res)).sum()
                if self.eps is not None:
                    below = jnp.minimum(below, 1.0 - self.eps)
                denum = log1p(-below)
        else:
            mass = jnp.where(inside, jnp.exp(res), 0.0).sum()
            if not (isinstance(hi, int) and hi < max_sz):
                rest = dist.sf(max_sz - 1.0, *args, **kwargs) - self._right_side(hi, args, kwargs)[1]
                mass = mass + jnp.maximum(rest, 0.0)
            if self.eps is not None:
                mass = jnp.maximum(mass, self.eps)
            denum = jnp.log(mass)
        res = res - denum
        if self.clip_bounds:
            res = jnp.where(inside, res, -jnp.inf)
        min_x = 0 if min_x is None else min_x
        max_x = max_sz - 1 if max_x is None else max_x
        return jnp.where((inds >= min_x) & (inds <= max_x), res, -jnp.inf)

    # Sums of the pmf of the base distribution over index ranges. A range
    # is a difference of two cdf values or of two sf values, whichever
    # pair is smaller, so that no digits are lost in either tail.
    def _long_tables(self, pts: list, precision: int, need_sf: bool, args, kwargs) -> tuple:
        kwargs = dict(kwargs)
        lo, hi = self._static_bounds(kwargs.pop('left', None), kwargs.pop('right', None))
        base = set(pts)
        if lo is not None:
            base.add(lo - 1)
        if hi is not None:
            base.add(hi)
        pmf, cdf, sf = self.dist._long_tables(sorted(base), precision, True, args, kwargs)
        res_pmf, res_cdf, res_sf = dict(), dict(), dict()
        with long_context(precision):
            zero, one, half = gmpy2.mpfr(0), gmpy2.mpfr(1), gmpy2.mpfr(0.5)

            # Sum over a < k <= b, None stands for an end of the support.
            def mass(a, b):
                ca, cb = zero if a is None else cdf[a], one if b is None else cdf[b]
                sa, sb = one if a is None else sf[a], zero if b is None else sf[b]
                return cb - ca if cb <= half else sa - sb

            first = None if lo is None else lo - 1
            norm = mass(first, hi)
            for k in pts:
                if lo is not None and k < lo:
                    res_pmf[k], res_cdf[k], res_sf[k] = zero, zero, one
                elif hi is not None and k > hi:
                    res_pmf[k], res_cdf[k], res_sf[k] = zero, one, zero
                else:
                    res_pmf[k] = pmf[k] / norm
                    res_cdf[k] = mass(first, k) / norm
                    res_sf[k] = mass(k, hi) / norm
        return res_pmf, res_cdf, res_sf


# Mixture of distributions. params is a list with a dict of keyword
# arguments per component. weights holds a weight per component or all
# of them but the last one.
class Mixture(Distribution):
    def __init__(self, *distros, normalize_weights=True):
        self.distros = distros
        self.normalize_weights = normalize_weights
        self.params = {'weights': [(0.0, 1.0)] * (len(distros) - 1)}
        self.rec_order = max(d.rec_order for d in distros)

    def _update_weights(self, weights: jnp.ndarray) -> jnp.ndarray:
        weights = jnp.atleast_1d(jnp.asarray(weights, dtype=float))
        if len(weights) == len(self.distros):
            weights = weights / weights.sum() if self.normalize_weights else weights
        elif len(weights) == len(self.distros) - 1:
            weights = jnp.append(weights, 1.0 - weights.sum())
        return weights

    def _long_update_weights(self, weights: np.ndarray) -> list:
        weights = [to_mpfr(w) for w in np.atleast_1d(np.asarray(weights, dtype=object))]
        if len(weights) == len(self.distros):
            sm = sum(weights)
            weights = [w / sm for w in weights] if self.normalize_weights else weights
        elif len(weights) == len(self.distros) - 1:
            weights.append(1 - sum(weights))
        return weights

    # Values of a method for each component along the first axis and
    # the weights shaped to broadcast against them.
    def _components(self, method: str, first: tuple, params: list, weights, args, kwargs) -> tuple:
        vals = [getattr(dist, method)(*first, *args, **kwargs, **p) for p, dist in zip(params, self.distros)]
        vals = jnp.stack(jnp.broadcast_arrays(*vals))
        weights = self._update_weights(weights)
        return vals, weights.reshape(-1, *([1] * (vals.ndim - 1)))

    def logpmf(self, x: jnp.ndarray, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        vals, weights = self._components('logpmf', (x,), params, weights, args, kwargs)
        return logsumexp(vals, axis=0, b=weights)

    def logpmf_recurrent(self, min_x: int, max_x: int,  max_sz: int, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        vals, weights = self._components('logpmf_recurrent', (min_x, max_x, max_sz), params, weights, args, kwargs)
        return logsumexp(vals, axis=0, b=weights)

    def logcdf(self, x: jnp.ndarray, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        vals, weights = self._components('logcdf', (x,), params, weights, args, kwargs)
        return logsumexp(vals, axis=0, b=weights)

    def logsf(self, x: jnp.ndarray, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        vals, weights = self._components('logsf', (x,), params, weights, args, kwargs)
        return logsumexp(vals, axis=0, b=weights)

    def cdf(self, x: jnp.ndarray, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        vals, weights = self._components('cdf', (x,), params, weights, args, kwargs)
        return jnp.sum(vals * weights, axis=0)

    def sf(self, x: jnp.ndarray, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        vals, weights = self._components('sf', (x,), params, weights, args, kwargs)
        return jnp.sum(vals * weights, axis=0)

    def _long_tables(self, pts: list, precision: int, need_sf: bool, args, kwargs) -> tuple:
        params, weights, args = args[0], args[1], args[2:]
        res = None
        with long_context(precision):
            weights = self._long_update_weights(weights)
        for p, w, dist in zip(params, weights, self.distros):
            tables = dist._long_tables(pts, precision, need_sf, args, {**kwargs, **p})
            with long_context(precision):
                tables = [{k: w * v for k, v in t.items()} for t in tables]
                res = tables if res is None else [{k: v + s[k] for k, v in t.items()} for t, s in zip(tables, res)]
        return tuple(res)

    def mean(self, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        vals, weights = self._components('mean', (), params, weights, args, kwargs)
        return jnp.sum(vals * weights, axis=0)

    def var(self, params: list[dict], weights: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        means, weights = self._components('mean', (), params, weights, args, kwargs)
        var = self._components('var', (), params, weights.reshape(-1), args, kwargs)[0]
        mean = jnp.sum(means * weights, axis=0)
        return jnp.sum((var + means ** 2) * weights, axis=0) - mean ** 2

    def sample(self, size: int, params: list[dict], weights: np.ndarray, *args, seed=None, **kwargs) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(np.prod(size))
        weights = np.asarray(self._update_weights(weights), dtype=float)
        counts = rng.multinomial(n, weights / weights.sum())
        res = [np.asarray(dist.sample(int(c), *args, seed=rng, **kwargs, **p)).reshape(-1)
               for c, p, dist in zip(counts, params, self.distros)]
        return rng.permutation(np.concatenate(res)).reshape(size)

class BinomialFamily(Distribution):

    def __init__(self, p_success=True):
        self.p_success = p_success

    # The probability raised to the power of x and its complement.
    def _probs(self, p) -> tuple:
        return (p, 1.0 - p) if self.p_success else (1.0 - p, p)

    def _logprobs(self, p) -> tuple:
        return (jnp.log(p), log1p(-p)) if self.p_success else (log1p(-p), jnp.log(p))

    def _long_probs(self, p) -> tuple:
        p = to_mpfr(p)
        return (p, 1 - p) if self.p_success else (1 - p, p)

class BetaBinomialFamily(BinomialFamily):

    def __init__(self):
        pass

class Poisson(Distribution):

    rec_order = 1
    params = {'rate': [(0.0, None)]}

    def logpmf(self, x: jnp.ndarray, rate: jnp.ndarray) -> jnp.ndarray:
        x, rate = _float(x, rate)
        xs = jnp.maximum(x, 0.0)
        res = xs * jnp.log(rate) - rate - gammaln(xs + 1.0)
        return jnp.where(x >= 0, res, -jnp.inf)

    # The side of x that does not hold the mean is summed directly.
    def _cdf_parts(self, x: jnp.ndarray, rate: jnp.ndarray) -> tuple:
        x, rate = _float(x, rate)
        xs = jnp.maximum(x, 0.0)
        upper = xs + 1.0 >= rate
        lower_sum = hyp.series_logsum(hyp.poisson_lower_ratio, hyp.ratio_bound)
        upper_sum = hyp.series_logsum(hyp.poisson_upper_ratio, hyp.ratio_bound)
        lg = jnp.where(upper,
                       self.logpmf(xs + 1.0, rate) + upper_sum(jnp.where(upper, jnp.inf, 0.0), xs, rate),
                       self.logpmf(xs, rate) + lower_sum(jnp.where(upper, 0.0, xs), xs, rate))
        return jnp.where(x >= 0, lg, -jnp.inf), upper & (x >= 0)

    def _log_ratio(self, x: int, rate: float) -> float:
        return jnp.log(rate) - jnp.log(x)

    def _long_setup(self, rate) -> tuple:
        return to_mpfr(rate),

    def _long_start(self, ctx: tuple) -> list:
        return [gmpy2.exp(-ctx[0])]

    def _long_ratio(self, x: int, ctx: tuple) -> gmpy2.mpfr:
        return ctx[0] / x

    def _long_tail_ratio(self, ctx: tuple):
        return gmpy2.mpfr(0)

    def mean(self, rate: jnp.ndarray) -> jnp.ndarray:
        rate = jnp.array(rate)
        return rate

    def var(self, rate: jnp.ndarray) -> jnp.ndarray:
        rate = jnp.array(rate)
        return rate

    def sample(self, size: int, rate, seed=None) -> np.ndarray:
        return scipy_poisson.rvs(rate, size=size, random_state=np.random.default_rng(seed))

class Binomial(BinomialFamily):

    rec_order = 1
    params = {'n': [(0.0, None)], 'p': [(0.0, 1.0)]}

    def __init__(self, p_success=True, eps=1e-15, max_n=1000):
        self.p_success = p_success
        self.eps = eps
        self.max_n = max_n

    def logpmf(self, x: jnp.ndarray, n, p) -> jnp.ndarray:
        x, n, p = _float(x, n, p)
        inside = (x >= 0) & (x <= n)
        xs = jnp.where(inside, x, 0.0)
        lp, lq = self._logprobs(p)
        res = xs * lp + (n - xs) * lq + logbinomial(n, xs)
        return jnp.where(inside, res, -jnp.inf)

    def _cdf_parts(self, x: jnp.ndarray, n, p) -> tuple:
        x, n, p = _float(x, n, p)
        inside = (x >= 0) & (x < n)
        xs = jnp.where(inside, x, 0.0)
        ns = jnp.where(inside, n, 1.0)
        ps, qs = self._probs(p)
        lg, swap = betainc_parts(ns - xs, xs + 1.0, qs, xc=ps, eps=self.eps, max_n=self.max_n)
        return jnp.where(inside, lg, -jnp.inf), jnp.where(inside, swap, x >= n)

    def _log_ratio(self, x: int, n, p) -> float:
        lp, lq = self._logprobs(p)
        return _log_positive(n - x + 1) - jnp.log(x) + lp - lq

    def _long_setup(self, n, p) -> tuple:
        return (int(n),) + self._long_probs(p)

    def _long_start(self, ctx: tuple) -> list:
        return [ctx[2] ** ctx[0]]

    def _long_ratio(self, x: int, ctx: tuple) -> gmpy2.mpfr:
        n, p, q = ctx
        return p * (n - x + 1) / (q * x)

    def _long_support(self, ctx: tuple):
        return ctx[0]

    def sample(self, size: int, n, p, seed=None) -> np.ndarray:
        return scipy_binom.rvs(n=n, p=self._probs(p)[0], size=size, random_state=np.random.default_rng(seed))

    def mean(self, n, p) -> jnp.ndarray:
        n, p = _float(n, p)
        return self._probs(p)[0] * n

    def var(self, n, p) -> np.ndarray:
        n, p = _float(n, p)
        return p * (1 - p) * n

class NB(BinomialFamily):

    rec_order = 1
    params = {'r': [(0.0, None)], 'p': [(0.0, 1.0)]}

    def __init__(self, p_success=True, eps=1e-15, max_n=1000):
        self.p_success = p_success
        self.eps = eps
        self.max_n = max_n

    def logpmf(self, x: jnp.ndarray, r, p) -> jnp.ndarray:
        x, r, p = _float(x, r, p)
        xs = jnp.maximum(x, 0.0)
        lp, lq = self._logprobs(p)
        res = lp * xs + lq * r - betaln(r, xs + 1.0) - jnp.log(xs + r)
        return jnp.where(x >= 0, res, -jnp.inf)

    def _cdf_parts(self, x: jnp.ndarray, r, p) -> tuple:
        x, r, p = _float(x, r, p)
        xs = jnp.maximum(x, 0.0)
        ps, qs = self._probs(p)
        lg, swap = betainc_parts(r, xs + 1.0, qs, xc=ps, eps=self.eps, max_n=self.max_n)
        return jnp.where(x >= 0, lg, -jnp.inf), swap & (x >= 0)

    def _log_ratio(self, x: int, r, p) -> float:
        return jnp.log(x + r - 1) - jnp.log(x) + self._logprobs(p)[0]

    def _long_setup(self, r, p) -> tuple:
        return (to_mpfr(r),) + self._long_probs(p)

    def _long_start(self, ctx: tuple) -> list:
        return [ctx[2] ** ctx[0]]

    def _long_ratio(self, x: int, ctx: tuple) -> gmpy2.mpfr:
        r, p, _ = ctx
        return p * (x + r - 1) / x

    def _long_tail_ratio(self, ctx: tuple):
        return ctx[1]

    def mean(self, r, p) -> jnp.ndarray:
        r, p = _float(r, p)
        p, q = self._probs(p)
        return r * p / q

    def var(self, r, p) -> jnp.ndarray:
        r, p = _float(r, p)
        p, q = self._probs(p)
        return r * p / q ** 2

    def sample(self, size: int, r, p, seed=None) -> np.array:
        return scipy_nb.rvs(n=r, p=self._probs(p)[1], size=size, random_state=np.random.default_rng(seed))


class BetaNB(BetaBinomialFamily):

    rec_order = 1
    params = {'r': [(0.0, None)], 'a': [(0.0, None)], 'b': [(0.0, None)]}

    # eps and max_n set the stopping rule of the sums behind the cdf.
    # min_decay is the least power of the index at which the terms of
    # a series for the right tail are let to decay.
    def __init__(self, eps=1e-18, max_n=200000, min_decay=12):
        self.eps = eps
        self.max_n = max_n
        self.min_decay = min_decay

    def logpmf(self, x: jnp.ndarray, r, a, b) -> jnp.ndarray:
        x, r, a, b = _float(x, r, a, b)
        xs = jnp.maximum(x, 0.0)
        res = betaln(a + r, b + xs) - betaln(a, b) - betaln(r, xs + 1.0) - jnp.log(xs + r)
        return jnp.where(x >= 0, res, -jnp.inf)

    # Left of the mode the pmf is summed from x down to zero. Right of
    # it the tail over k > x is summed. Its terms decay as k ** -(a + 1),
    # and as j ** -(top + 2) after the transformation by Thomae applied
    # at a point top >= x. The pmf between x and top is summed directly.
    def _cdf_parts(self, x: jnp.ndarray, r, a, b) -> tuple:
        x, r, a, b = _float(x, r, a, b)
        xs = jnp.maximum(x, 0.0)
        series = lambda ratio, bound: hyp.series_logsum(ratio, bound, self.eps, self.max_n)
        upper = (r - 1.0) * (b - 1.0) <= (xs + 2.0) * (a + 1.0)
        plain = upper & (a + 1.0 >= jnp.maximum(xs + 2.0, self.min_decay))
        thomae = upper & ~plain
        top = jnp.maximum(xs, math.ceil(self.min_decay) - 2.0)
        n_plain = jnp.where(plain, jnp.inf, jnp.where(thomae, top - xs - 1.0, 0.0))
        lower_sum = series(hyp.bnb_lower_ratio, hyp.ratio_bound)(jnp.where(upper, 0.0, xs), xs, r, a, b)
        plain_sum = self.logpmf(xs + 1.0, r, a, b) + series(hyp.bnb_upper_ratio, hyp.bnb_upper_bound)(n_plain, xs, r, a, b)
        thomae_sum = series(hyp.bnb_thomae_ratio, hyp.bnb_thomae_bound)(jnp.where(thomae, jnp.inf, 0.0), top, r, a, b)
        thomae_sum = thomae_sum + self.logpmf(top + 1.0, r, a, b) + jnp.log(top + 1.0) - jnp.log(a)
        thomae_sum = jnp.where(top > xs, jnp.logaddexp(plain_sum, thomae_sum), thomae_sum)
        lg = jnp.where(upper, jnp.where(plain, plain_sum, thomae_sum), self.logpmf(xs, r, a, b) + lower_sum)
        return jnp.where(x >= 0, lg, -jnp.inf), upper & (x >= 0)

    def _log_ratio(self, x, r, a, b) -> jnp.ndarray:
        return jnp.log(x + r - 1) - jnp.log(x + a + b + r - 1)  + jnp.log(x + b - 1) - jnp.log(x)

    def _long_setup(self, r, a, b) -> tuple:
        return to_mpfr(r), to_mpfr(a), to_mpfr(b)

    def _long_start(self, ctx: tuple) -> list:
        r, a, b = ctx
        return [gmpy2.exp(long_logbeta(a + r, b) - long_logbeta(a, b))]

    def _long_ratio(self, x: int, ctx: tuple) -> gmpy2.mpfr:
        r, a, b = ctx
        return ((x + r - 1) / (x + a + b + r - 1)) * ((x + b - 1) / x)

    def sample(self, size: int, r, a, b, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return scipy_nb.rvs(n=r, p=rng.beta(a, b, size=size), size=size, random_state=rng)

    # Infinite for a <= 1.
    def mean(self, r, a, b) -> jnp.ndarray:
        r, a, b = _float(r, a, b)
        return jnp.where(a > 1, r * b / jnp.where(a > 1, a - 1, 1.0), jnp.inf)

    # Infinite for a <= 2.
    def var(self, r, a, b) -> jnp.ndarray:
        r, a, b = _float(r, a, b)
        t = jnp.where(a > 2, a, 3.0)
        return jnp.where(a > 2, r * b * (r + t - 1) * (b + t - 1) / ((t - 2) * (t - 1) ** 2), jnp.inf)


class BetaBinomial(BetaBinomialFamily):

    rec_order = 1
    params = {'n': [(0.0, None)], 'a': [(0.0, None)], 'b': [(0.0, None)]}

    def __init__(self, eps=1e-18, max_n=200000):
        self.eps = eps
        self.max_n = max_n

    def logpmf(self, x: jnp.ndarray, n, a, b) -> jnp.ndarray:
        x, n, a, b = _float(x, n, a, b)
        inside = (x >= 0) & (x <= n)
        xs = jnp.where(inside, x, 0.0)
        res = betaln(xs + a, n - xs + b) - betaln(a, b) + logbinomial(n, xs)
        return jnp.where(inside, res, -jnp.inf)

    # The pmf is summed over the side of x that does not hold the mean.
    def _cdf_parts(self, x: jnp.ndarray, n, a, b) -> tuple:
        x, n, a, b = _float(x, n, a, b)
        inside = (x >= 0) & (x < n)
        xs = jnp.where(inside, x, 0.0)
        ns = jnp.where(inside, n, 1.0)
        series = lambda ratio: hyp.series_logsum(ratio, hyp.betabinom_bound, self.eps, self.max_n)
        upper = (xs + 1.0) * (a + b) >= ns * a
        lower_sum = series(hyp.betabinom_lower_ratio)(jnp.where(upper, 0.0, xs), xs, ns, a, b)
        upper_sum = series(hyp.betabinom_upper_ratio)(jnp.where(upper, ns - xs - 1.0, 0.0), xs, ns, a, b)
        lg = jnp.where(upper,
                       self.logpmf(xs + 1.0, ns, a, b) + upper_sum,
                       self.logpmf(xs, ns, a, b) + lower_sum)
        return jnp.where(inside, lg, -jnp.inf), jnp.where(inside, upper, x >= n)

    def _log_ratio(self, x, n, a, b) -> jnp.ndarray:
        t1 = _log_positive(n - x + 1) - jnp.log(x)
        t2 = jnp.log(x + a - 1) - jnp.log(jnp.maximum(n - x, 0) + b)
        return t1 + t2

    def _long_setup(self, n, a, b) -> tuple:
        return int(n), to_mpfr(a), to_mpfr(b)

    def _long_start(self, ctx: tuple) -> list:
        n, a, b = ctx
        return [gmpy2.exp(long_logbeta(a, n + b) - long_logbeta(a, b))]

    def _long_ratio(self, x: int, ctx: tuple) -> gmpy2.mpfr:
        n, a, b = ctx
        return (gmpy2.mpfr(n - x + 1) / x) * ((x + a - 1) / (n - x + b))

    def _long_support(self, ctx: tuple):
        return ctx[0]

    def sample(self, size: int, n, a, b, seed=None) -> np.ndarray:
        return scipy_betabinom.rvs(n=n, a=a, b=b, size=size, random_state=np.random.default_rng(seed))

    def mean(self, n, a, b) -> jnp.ndarray:
        n, a, b = _float(n, a, b)
        return n * a / (a + b)

    def var(self, n, a, b) -> np.ndarray:
        n, a, b = _float(n, a, b)
        num = n * a * b * (a + b + n)
        denum = (a + b) ** 2 * (a + b + 1)
        return num / denum

# Marginalized compound negative binomial distribution: X ~ NB(k, p1)
# with pmf proportional to p1 ** x, where k follows a zero-truncated
# compound distribution: NB(r, p2) with pmf proportional to p2 ** k,
# Binomial(r, p2) or Poisson(r). r is any positive real. The pmf is
# M q ** x H(x), H(x) = 2F1(1 + r, 1 + x; 2; z) or 1F1(1 + x; 2; z) for
# Poisson. p2 is not used by Poisson and is 1 - p1 by default for
# Binomial, that is the MCNB(r, p1) of the MixALime paper.
class MCNB(BinomialFamily):
    rec_order = 2
    params = {'r': [(0.0, None)], 'p1': [(0.0, 1.0)], 'p2': [(0.0, 1.0)]}

    def __init__(self, compound_dist=CompoundDist.NB, eps=1e-18, max_n=200000):
        self.dist_name = compound_dist
        self.eps = eps
        self.max_n = max_n
        self.params = dict(self.params)
        if compound_dist == CompoundDist.Poisson:
            self.params['p2'] = None

    @property
    def _kummer(self) -> bool:
        return self.dist_name == CompoundDist.Poisson

    def _second_prob(self, p1, p2):
        if self.dist_name == CompoundDist.Poisson:
            return p1
        if p2 is None:
            if self.dist_name == CompoundDist.Binomial:
                return 1 - p1
            raise TypeError('p2 is required for the NB compound distribution.')
        return p2

    # Returns log M, log q, z and log(1 - z), the latter is zero for Poisson.
    def _consts(self, r, p1, p2) -> tuple:
        if self.dist_name == CompoundDist.Poisson:
            return log1p(-p1) + jnp.log(r) - hyp.log_expm1(r), jnp.log(p1), r * (1.0 - p1), jnp.zeros_like(r * p1)
        t = log1p(-p2)
        logm = jnp.log(r) + log1p(-p1) + jnp.log(p2) + t * r - jnp.log(-jnp.expm1(t * r))
        if self.dist_name == CompoundDist.NB:
            z = p2 * (1.0 - p1)
            return logm, jnp.log(p1), z, log1p(-z)
        elif self.dist_name == CompoundDist.Binomial:
            l1mz = t - log1p(-p1 * p2)
            z = (1.0 - p1) * p2 / (1.0 - p1 * p2)
            return logm - t + l1mz, jnp.log(p1) + l1mz, z, l1mz
        raise NotImplementedError

    # log H(0) and log H(1).
    def _log_h(self, r, z, l1mz) -> tuple:
        if self._kummer:
            return hyp.log_expm1(z) - jnp.log(z), z
        return hyp.log_expm1(-r * l1mz) - jnp.log(r) - jnp.log(z), -(1.0 + r) * l1mz

    # For x > 0, H(x) is exp(z) 1F1(1 - x; 2; -z) for Poisson and
    # (1 - z) ** -(1 + r) 2F1(1 + r, 1 - x; 2; -z / (1 - z)) otherwise.
    # Both are polynomials of degree x - 1 with positive terms.
    def logpmf(self, x: jnp.ndarray, r, p1, p2=None) -> jnp.ndarray:
        x, r, p1, p2 = _float(x, r, p1, self._second_prob(p1, p2))
        xs = jnp.maximum(x, 0.0)
        logm, logq, z, l1mz = self._consts(r, p1, p2)
        h0, h1 = self._log_h(r, z, l1mz)
        ratio = hyp.hyp_1f1_poly_ratio if self._kummer else hyp.hyp_2f1_poly_ratio
        zeta = z if self._kummer else jnp.exp(jnp.log(z) - l1mz)
        poly = hyp.series_logsum(ratio, hyp.ratio_bound, self.eps, self.max_n)
        logh = jnp.where(xs == 0, h0, h1 + poly(jnp.maximum(xs - 1.0, 0.0), xs, r, zeta))
        return jnp.where(x >= 0, logm + xs * logq + logh, -jnp.inf)

    # Logarithms of the cdf and of the sf along the last axis. The pmf
    # is summed by the recurrence up to x, and past x until the terms
    # are negligible for tail=True.
    def _logsums(self, tail: bool, x, r, p1, p2) -> jnp.ndarray:
        x, r, p1, p2 = _float(x, r, p1, self._second_prob(p1, p2))
        xs = jnp.maximum(x, 0.0)
        logm, logq, z, l1mz = self._consts(r, p1, p2)
        h0, h1 = self._log_h(r, z, l1mz)
        fun = hyp.mc_logsums(self._kummer, tail, self.eps, self.max_n)
        res = fun(xs, logm + h0, logm + logq + h1, r, jnp.exp(logq), z, jnp.exp(l1mz))
        return jnp.where((x >= 0)[..., None], res, jnp.array([-jnp.inf, 0.0]))

    def _cdf_sf(self, x: jnp.ndarray, r, p1, p2=None) -> tuple:
        res = jnp.exp(jnp.minimum(self._logsums(True, x, r, p1, p2), 0.0))
        return res[..., 0], res[..., 1]

    def logcdf(self, x: jnp.ndarray, r, p1, p2=None) -> jnp.ndarray:
        return jnp.minimum(self._logsums(False, x, r, p1, p2)[..., 0], 0.0)

    def cdf(self, x: jnp.ndarray, r, p1, p2=None) -> jnp.ndarray:
        return jnp.exp(self.logcdf(x, r, p1, p2))

    def logsf(self, x: jnp.ndarray, r, p1, p2=None) -> jnp.ndarray:
        return jnp.minimum(self._logsums(True, x, r, p1, p2)[..., 1], 0.0)

    def sf(self, x: jnp.ndarray, r, p1, p2=None) -> jnp.ndarray:
        return jnp.exp(self.logsf(x, r, p1, p2))

    # The state holds log pmf(x - 1) with the compensation of its sum,
    # pmf(x - 1) / pmf(x - 2), q, z and 1 - z. The recurrence runs on
    # ratios of consecutive values of the pmf, they stay of order one
    # for any x.
    def _rec_start(self, r, p1, p2=None) -> tuple:
        r, p1, p2 = _float(r, p1, self._second_prob(p1, p2))
        logm, logq, z, l1mz = self._consts(r, p1, p2)
        h0, h1 = self._log_h(r, z, l1mz)
        start = jnp.stack([logm + h0, logm + logq + h1])
        return start, (start[1], jnp.zeros_like(z), jnp.exp(logq + h1 - h0), jnp.exp(logq), z, jnp.exp(l1mz))

    def _rec_step(self, x, state, r, p1, p2=None) -> tuple:
        lf, comp, rho, q, z, omz = state
        t1, t2 = hyp.mc_rec_terms(self._kummer, x, r, z, omz)
        rho = q * (t1 + t2 * q / rho)
        lf, comp = kahan_add(lf, comp, jnp.log(rho))
        return lf, (lf, comp, rho, q, z, omz)

    # Returns r, q, z, 1 - z and the pmf at 0 and 1.
    def _long_setup(self, r, p1, p2=None) -> tuple:
        r, p1 = to_mpfr(r), to_mpfr(p1)
        p2 = to_mpfr(self._second_prob(p1, p2))
        if self._kummer:
            z, omz, q = r * (1 - p1), gmpy2.mpfr(1), p1
            m = (1 - p1) * r / gmpy2.expm1(r)
            h0, h1 = gmpy2.expm1(z) / z, gmpy2.exp(z)
        else:
            m = r * (1 - p1) * p2 * (1 - p2) ** r / (1 - (1 - p2) ** r)
            if self.dist_name == CompoundDist.NB:
                z, q = p2 * (1 - p1), p1
                omz = 1 - z
            else:
                omz = (1 - p2) / (1 - p1 * p2)
                z = (1 - p1) * p2 / (1 - p1 * p2)
                q = p1 * omz
                m = m * omz / (1 - p2)
            h0, h1 = (omz ** (-r) - 1) / (r * z), omz ** (-1 - r)
        return r, q, z, omz, m * h0, m * q * h1

    def _long_start(self, ctx: tuple) -> list:
        return list(ctx[4:])

    def _long_step(self, x: int, prevs: list, ctx: tuple) -> gmpy2.mpfr:
        r, q, z, omz = ctx[:4]
        t1, t2 = hyp.mc_rec_terms(self._kummer, gmpy2.mpfr(x), r, z, omz)
        return q * (t1 * prevs[1] + t2 * q * prevs[0])

    def _long_tail_ratio(self, ctx: tuple):
        return ctx[1] / ctx[3]

    def mean(self, r, p1, p2=None):
        r, p1, p2 = _float(r, p1, self._second_prob(p1, p2))
        m = jnp.log(p1) + jnp.log(r) - log1p(-p1)
        if self.dist_name == CompoundDist.NB:
            m += jnp.log(p2) - log1p(-p2) - jnp.log(-jnp.expm1(log1p(-p2) * r))
        elif self.dist_name == CompoundDist.Binomial:
            m += jnp.log(p2) - jnp.log(-jnp.expm1(log1p(-p2) * r))
        elif self.dist_name == CompoundDist.Poisson:
            m += r - hyp.log_expm1(r)
        return jnp.exp(m)

    def second_moment(self, r, p1, p2=None):
        r, p1, p2 = _float(r, p1, self._second_prob(p1, p2))
        num = jnp.log(p1) + jnp.log(r)
        denum = log1p(-p1) * 2
        if self.dist_name == CompoundDist.NB:
            num += jnp.log(1 + p1 - p2 + r * p1 * p2) + jnp.log(p2)
            denum += log1p(-p2) * 2 + jnp.log(-jnp.expm1(log1p(-p2) * r))
        elif self.dist_name == CompoundDist.Binomial:
            num += jnp.log(1 + p1 + (r - 1) * p1 * p2) + jnp.log(p2)
            denum += jnp.log(-jnp.expm1(log1p(-p2) * r))
        elif self.dist_name == CompoundDist.Poisson:
            num += r + jnp.log(1 + p1 + p1 * r)
            denum += hyp.log_expm1(r)
        return jnp.exp(num - denum)

    def var(self, r, p1, p2=None):
        x, x2 = self.mean(r, p1, p2), self.second_moment(r, p1, p2)
        return x2 - x ** 2

    # Inversion of the cdf on a table of the pmf. The table is doubled
    # until it holds all of the mass but 1e-13.
    def sample(self, size: int, r, p1, p2=None, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(float(self.mean(r, p1, p2)) + 10 * float(self.var(r, p1, p2)) ** 0.5) + 32
        while True:
            cdf = np.cumsum(np.asarray(self.pmf_recurrent(0, n - 1, n, r, p1, p2)))
            if cdf[-1] > 1 - 1e-13:
                break
            n *= 2
        return np.searchsorted(cdf / cdf[-1], rng.random(size), side='right')
