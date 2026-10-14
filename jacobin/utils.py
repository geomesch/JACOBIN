# -*- coding: utf-8 -*-
import gmpy2
import jax.numpy as jnp
import numpy as np
import jax
from jax.custom_derivatives import SymbolicZero
from jax.scipy.linalg import toeplitz
from jax.scipy.special import gammaln
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


# Logarithm of the beta function with one call of gammaln. The larger
# argument goes through the Stirling series, shifted up by 10 with
# B(lo, hi) = B(lo, hi + 10) prod_{i<10} (lo + hi + i) / (hi + i) if it
# is below 10. A sum of three gammaln values loses the digits that
# cancel, and betaln of jax.scipy.special is accurate to 7e-7 only.
def betaln(a, b):
    a, b = jnp.asarray(a, dtype=float), jnp.asarray(b, dtype=float)
    # minimum and maximum halve the derivatives at a == b, the second
    # derivatives come out wrong there.
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
def loop_differentiable(fun: Callable) -> Callable:
    partials = dict()

    @jax.custom_jvp
    def wrapped(*args):
        return fun(*args)

    def jvp(primals, tangents):
        active = tuple(i for i, t in enumerate(tangents) if not isinstance(t, SymbolicZero))
        out = wrapped(*primals)
        if not active:
            return out, jax.tree_util.tree_map(jnp.zeros_like, out)
        if active not in partials:
            partials[active] = loop_differentiable(jax.jacfwd(fun, argnums=active))
        tangent = None
        for i, part in zip(active, partials[active](*primals)):
            term = jax.tree_util.tree_map(lambda t: t * tangents[i], part)
            tangent = term if tangent is None else jax.tree_util.tree_map(jnp.add, tangent, term)
        return out, tangent

    wrapped.defjvp(jvp, symbolic_zeros=True)
    return wrapped


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
