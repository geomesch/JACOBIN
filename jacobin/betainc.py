# -*- coding: utf-8 -*-
"""
This module concerns mostly effective gradient-friendly computation of betainc
function.
"""
import jax.numpy as jnp
from jax.lax import while_loop
from functools import lru_cache, partial
from .utils import loop_differentiable, betaln, log1p

_TINY = 1e-300


# Logarithm of I_x(p, q) by the continued fraction 8.17.22 of DLMF and
# the modified Lentz method. xc is 1 - x. Fast for x < (p + 1) / (p + q + 2).
def _logbetacf(p, q, x, xc, eps=1e-15, max_n=1000):
    def guard(t):
        return jnp.where(jnp.abs(t) < _TINY, _TINY, t)

    def lentz_iteration(prev):
        h, c, d, n, _ = prev
        n2 = 2.0 * n
        an = n * (q - n) * x / ((p + n2 - 1.0) * (p + n2))
        d = 1.0 / guard(1.0 + an * d)
        c = guard(1.0 + an / c)
        h = h * d * c
        an = -(p + n) * (p + q + n) * x / ((p + n2) * (p + n2 + 1.0))
        d = 1.0 / guard(1.0 + an * d)
        c = guard(1.0 + an / c)
        delta = d * c
        return h * delta, c, d, n + 1.0, delta

    def lentz_cond(prev):
        return (jnp.abs(prev[-1] - 1.0) > eps) & (prev[-2] <= max_n)

    d = 1.0 / guard(1.0 - (p + q) * x / (p + 1.0))
    h = while_loop(lentz_cond, lentz_iteration, (d, 1.0, d, 1.0, 2.0))[0]
    return p * jnp.log(x) + q * jnp.log(xc) - jnp.log(p) - betaln(p, q) + jnp.log(h)


@lru_cache(maxsize=None)
def _kernel(eps, max_n):
    return jnp.vectorize(loop_differentiable(partial(_logbetacf, eps=eps, max_n=max_n)))


# Returns a tuple (lg, swap). The function I_x(p, q) is exp(lg) where
# swap is false and 1 - exp(lg) where swap is true, so exp(lg) is always
# the smaller side. xc is 1 - x, it can be passed to keep precision.
def betainc_parts(p, q, x, xc=None, eps=1e-15, max_n=1000):
    p, q, x = (jnp.asarray(t, dtype=float) for t in (p, q, x))
    xc = 1.0 - x if xc is None else jnp.asarray(xc, dtype=float)
    swap = x > (p + 1.0) / (p + q + 2.0)
    a, b = jnp.where(swap, q, p), jnp.where(swap, p, q)
    t, tc = jnp.where(swap, xc, x), jnp.where(swap, x, xc)
    valid = (t > 0) & (tc > 0)
    lg = _kernel(eps, max_n)(a, b, jnp.where(valid, t, 0.5), jnp.where(valid, tc, 0.5))
    lg = jnp.where(valid, lg, jnp.where(t > 0, 0.0, -jnp.inf))
    return lg, swap


def _log1mexp(t):
    neg = t < 0
    t = jnp.where(neg, t, -1.0)
    res = jnp.where(t > -0.6931471805599453, jnp.log(-jnp.expm1(t)), log1p(-jnp.exp(t)))
    return jnp.where(neg, res, -jnp.inf)


def logbetainc(p, q, x, eps=1e-15, max_n=1000, xc=None):
    lg, swap = betainc_parts(p, q, x, xc, eps=eps, max_n=max_n)
    return jnp.where(swap, _log1mexp(lg), lg)


def betainc(p, q, x, eps=1e-15, max_n=1000, xc=None):
    lg, swap = betainc_parts(p, q, x, xc, eps=eps, max_n=max_n)
    return jnp.where(swap, -jnp.expm1(lg), jnp.exp(lg))
