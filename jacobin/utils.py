# -*- coding: utf-8 -*-
import gmpy2
import jax.numpy as jnp
import numpy as np
import jax
from jax.custom_derivatives import SymbolicZero
from jax.scipy.linalg import toeplitz
from jax.scipy.special import gammaln, polygamma
from collections.abc import Callable


def hankel(a, b):
    a = jnp.flip(a)
    b = jnp.append(jnp.array([0.0]), b)
    return jnp.flip(toeplitz(a, b), axis=0)


# log(1 + x). jnp.log1p has a relative error up to 3e-14 for x between
# -0.5 and -0.01. There the identity log(1 + x) = -log(1 + y) with
# y = -x / (1 + x) > 0 is used, jnp.log1p is accurate for positive y.
def log1p(x):
    x = jnp.asarray(x, dtype=float)
    mid = (x > -0.5) & (x < -0.01)
    xs = jnp.where(mid, x, -0.25)
    return jnp.where(mid, -jnp.log1p(-xs / (1.0 + xs)), jnp.log1p(x))


# gammaln(x) - (x - 0.5) log(x) + x - 0.5 log(2 pi) for x >= 10, the
# first terms of the Stirling series.
def _stirling(x):
    t = 1.0 / x
    t2 = t * t
    res = -691.0 / 360360.0 + t2 / 156.0
    for c in (1.0 / 1188.0, -1.0 / 1680.0, 1.0 / 1260.0, -1.0 / 360.0, 1.0 / 12.0):
        res = c + t2 * res
    return t * res


# Coefficients B_2k / 2k of the asymptotic series of digamma.
_DIGAMMA_SERIES = (1.0 / 12.0, -1.0 / 120.0, 1.0 / 252.0, -1.0 / 240.0, 1.0 / 132.0, -691.0 / 32760.0, 1.0 / 12.0)


# Terms of the asymptotic series of a difference at lo and lo + b. Returns
# 1 / lo, b / lo, w = lo / (lo + b), d = 1 - w ** 2 and the sum of
# coefs[k - 1] lo ** -2k (1 - w ** 2k) over k. The factor 1 - w ** 2k is
# d (1 + w ** 2 + ... + w ** (2k - 2)), so no digits cancel, and no
# transcendental function is called.
def _series_diff(lo, b, coefs):
    u = 1.0 / lo
    t = b * u
    w = 1.0 / (1.0 + t)
    q = w * w
    d = t * (2.0 + t) * q
    u2 = u * u
    uk, qk, part, res = u2, 1.0, 0.0, 0.0
    for c in coefs:
        part = part + qk
        res = res + c * uk * part
        qk, uk = qk * q, uk * u2
    return u, t, w, d, d * res


# digamma(a) - digamma(a + b) for positive a and b, without the
# cancellation of the difference when b is much smaller than a. Below 10
# a is shifted up by 10 with digamma(x) = digamma(x + 10) - sum_{i<10} 1 / (x + i),
# then the asymptotic series is taken for both terms at once.
@jax.custom_jvp
def digamma_diff(a, b):
    a, b = jnp.asarray(a, dtype=float), jnp.asarray(b, dtype=float)
    shift = a < 10.0
    corr = 0.0
    for i in range(10):
        corr = corr + b / ((a + i) * (a + b + i))
    lo = jnp.where(shift, a + 10.0, a)
    u, t, w, d, series = _series_diff(lo, b, _DIGAMMA_SERIES)
    res = -log1p(t) - 0.5 * t * u * w - series
    return res - jnp.where(shift, corr, 0.0)


# The tangent of out: the sum of the partials times the tangents. A
# partial is a function called only for a tangent that is not a symbolic
# zero, so the partial of a constant argument costs nothing.
def _tangent(out, partials, tangents):
    res = None
    for partial, t in zip(partials, tangents):
        if not isinstance(t, SymbolicZero):
            res = partial() * t if res is None else res + partial() * t
    return jnp.zeros_like(out) if res is None else jnp.broadcast_to(res, jnp.shape(out))


def _digamma_diff_jvp(primals, tangents):
    a, b = primals
    out = digamma_diff(a, b)
    return out, _tangent(out, (lambda: trigamma_diff(a, b), lambda: -polygamma(1, a + b)), tangents)


digamma_diff.defjvp(_digamma_diff_jvp, symbolic_zeros=True)


# Coefficients B_2k of the asymptotic series of trigamma.
_TRIGAMMA_SERIES = (1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0, 5.0 / 66.0, -691.0 / 2730.0, 7.0 / 6.0)


# trigamma(a) - trigamma(a + b) for positive a and b, built as digamma_diff.
# A term lo ** -(2k + 1) (1 - w ** (2k + 1)) is split as
# lo ** -(2k + 1) ((1 - w) + w (1 - w ** 2k)), with 1 - w = t w.
@jax.custom_jvp
def trigamma_diff(a, b):
    a, b = jnp.asarray(a, dtype=float), jnp.asarray(b, dtype=float)
    shift = a < 10.0
    corr = 0.0
    for i in range(10):
        corr = corr + b * (2.0 * (a + i) + b) / ((a + i) * (a + b + i)) ** 2
    lo = jnp.where(shift, a + 10.0, a)
    u, t, w, d, series = _series_diff(lo, b, _TRIGAMMA_SERIES)
    plain = 0.0
    uk = u * u
    for c in _TRIGAMMA_SERIES:
        plain = plain + c * uk
        uk = uk * u * u
    res = t * u * w + 0.5 * u * u * d + u * w * (t * plain + series)
    return res + jnp.where(shift, corr, 0.0)


def _trigamma_diff_jvp(primals, tangents):
    a, b = primals
    out = trigamma_diff(a, b)
    partials = (lambda: polygamma(2, a) - polygamma(2, a + b), lambda: -polygamma(2, a + b))
    return out, _tangent(out, partials, tangents)


trigamma_diff.defjvp(_trigamma_diff_jvp, symbolic_zeros=True)


# Logarithm of the beta function with one call of gammaln. The larger
# argument goes through the Stirling series, shifted up by 10 with
# B(lo, hi) = B(lo, hi + 10) prod_{i<10} (lo + hi + i) / (hi + i) if it
# is below 10. A sum of three gammaln values loses the digits that
# cancel, and betaln of jax.scipy.special is accurate to 7e-7 only. The
# derivatives come from digamma_diff: traced through the series, they
# take many times longer to compile.
@jax.custom_jvp
def betaln(a, b):
    return _betaln(a, b)


def _betaln_jvp(primals, tangents):
    a, b = primals
    out = betaln(a, b)
    return out, _tangent(out, (lambda: digamma_diff(a, b), lambda: digamma_diff(b, a)), tangents)


betaln.defjvp(_betaln_jvp, symbolic_zeros=True)


def _betaln(a, b):
    a, b = jnp.asarray(a, dtype=float), jnp.asarray(b, dtype=float)
    first = a < b
    lo, hi = jnp.where(first, a, b), jnp.where(first, b, a)
    both = lo >= 10.0
    shift = hi < 10.0
    corr = 1.0
    for i in range(10):
        corr = corr * ((lo + hi + i) / (hi + i))
    corr = jnp.where(shift, jnp.log(jnp.where(shift, corr, 1.0)), 0.0)
    his = jnp.where(shift, hi + 10.0, hi)
    los = jnp.where(both, lo, 10.0)
    sm = lo + his
    tail = (his - 0.5) * log1p(-lo / sm) + _stirling(his) - _stirling(sm)
    one = gammaln(jnp.where(both, 1.0, lo)) + tail + lo * (1.0 - jnp.log(sm)) + corr
    two = 0.9189385332046727 - 0.5 * jnp.log(his) + (los - 0.5) * jnp.log(los / sm) + 0.5 * log1p(-los / sm)
    two = two + tail + _stirling(los)
    return jnp.where(both, two, one)


# Wraps a function of scalar arguments built on while loops. The result
# is differentiable in forward and reverse mode to any order: partial
# derivatives are taken in forward mode and are wrapped the same way.
# The value comes out of the loop of the partials, so an order of
# derivatives adds one loop to the compiled program.
def loop_differentiable(fun: Callable) -> Callable:
    partials = dict()

    @jax.custom_jvp
    def wrapped(*args):
        return fun(*args)

    def jvp(primals, tangents):
        active = tuple(i for i, t in enumerate(tangents) if not isinstance(t, SymbolicZero))
        if not active:
            out = wrapped(*primals)
            return out, jax.tree_util.tree_map(jnp.zeros_like, out)
        if active not in partials:
            partials[active] = loop_differentiable(_value_and_partials(fun, active))
        out, parts = partials[active](*primals)
        tangent = None
        for k, i in enumerate(active):
            term = jax.tree_util.tree_map(lambda t: t[k] * tangents[i], parts)
            tangent = term if tangent is None else jax.tree_util.tree_map(jnp.add, tangent, term)
        return out, tangent

    wrapped.defjvp(jvp, symbolic_zeros=True)
    return wrapped


# Builds fun(*args) -> (value, partials) for the arguments with indices in
# active. Every leaf of partials has a leading axis over active. The
# forward passes for all of the arguments are batched over one loop.
def _value_and_partials(fun: Callable, active: tuple) -> Callable:
    def res(*args):
        def of_active(*values):
            full = list(args)
            for i, value in zip(active, values):
                full[i] = value
            return fun(*full)

        def push(v):
            tangents = [v[k] * jnp.ones_like(args[i]) for k, i in enumerate(active)]
            return jax.jvp(of_active, [args[i] for i in active], tangents)
        return jax.vmap(push, out_axes=(None, 0))(jnp.eye(len(active)))
    return res


# One step of the compensated summation by Kahan: adds inc to the sum s
# with the compensation c, returns the new sum and compensation. A sum
# of thousands of nearly equal terms keeps all digits this way.
def kahan_add(s, c, inc):
    y = inc - c
    t = s + y
    return t, jnp.where(jnp.isfinite(t), (t - s) - y, 0.0)


# Table of length max_sz filled by a recurrence that starts at zero.
# rec_start returns the first rec_order values and a state, rec_step
# maps (x, state) to the value at x and a new state. Entries with index
# outside of [min_x, max_x], both bounds inclusive, are -inf. The bounds
# may be traced values, max_sz is static.
def recurrent_fun(rec_start: Callable[..., tuple],
                  rec_step: Callable[..., tuple],
                  rec_order: int,
                  min_x: int, max_x: int, max_sz: int, *args, **kwargs) -> jnp.ndarray:
    def loop_body(state, x):
        val, state = rec_step(x, state, *args, **kwargs)
        return state, val

    min_x = 0 if min_x is None else min_x
    max_x = max_sz - 1 if max_x is None else max_x
    n = max_sz
    if is_static(max_x):
        n = max(int(min(max_x, max_sz - 1)) + 1, 0)
    start, state = rec_start(*args, **kwargs)
    res = jnp.asarray(start, dtype=float)[:n]
    if n > rec_order:
        tail = jax.lax.scan(loop_body, state, jnp.arange(rec_order, n))[1]
        res = jnp.concatenate([res, tail])
    res = jnp.concatenate([res, jnp.full(max_sz - len(res), -jnp.inf)])
    inds = jnp.arange(max_sz)
    return jnp.where((inds >= min_x) & (inds <= max_x), res, -jnp.inf)


def is_static(x) -> bool:
    return not isinstance(x, jax.core.Tracer) and np.ndim(x) == 0


# Context manager with the given precision in bits. The context of a
# caller is restored on exit.
def long_context(precision: int):
    try:
        return gmpy2.context(gmpy2.get_context(), precision=int(precision))
    except TypeError:
        return gmpy2.local_context(gmpy2.get_context(), precision=int(precision))


# Converts a scalar to mpfr of the current precision. A float keeps its
# binary value, a string is read as a decimal number.
def to_mpfr(x) -> gmpy2.mpfr:
    if isinstance(x, (str, int, type(gmpy2.mpfr(0)), type(gmpy2.mpz(0)), type(gmpy2.mpq(0)))):
        return gmpy2.mpfr(x)
    x = np.asarray(x)
    if x.size != 1:
        raise ValueError('Long algebra functions accept only scalar parameters.')
    x = x.reshape(-1)[0]
    if isinstance(x, (np.integer, np.bool_)):
        return gmpy2.mpfr(int(x))
    return gmpy2.mpfr(float(x))


# Splits x into a list of Python integers, the shape of x and a flag
# telling whether x is a scalar.
def long_points(x) -> tuple:
    scalar = np.ndim(x) == 0
    points = list()
    for v in np.asarray(x).reshape(-1).tolist():
        t = int(v)
        if t != v:
            raise ValueError('Long algebra functions accept only integer x.')
        points.append(t)
    return points, np.shape(x), scalar
