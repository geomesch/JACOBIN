# -*- coding: utf-8 -*-
"""
This module concerns mostly effective gradient-friendly computation of betainc
function.
"""
import jax.numpy as jnp
from jax.scipy.special import betaln
from jax.lax import cond, while_loop
from functools import partial


def _calc_a(p, q, f, n):
    n2 = 2 * n
    return (2 * (2 + f) * n**2 + n2 * (2 + f) * (-1 + p) +  p * (-2 + p - f * q))/((-2 + n2 + p) * (n2 + p))

def __calc_b(p, q, f, n):
    n2 = 2 * n + p
    return -((f * (-1 + n) * (-1 + n + p) * (n - q)  * (-2 + n + p + q))/((-3 + n2) * (-2 + n2)**2 * (-1 + n2)))

def __calc_b_first(p, q, f, n):
    return (q - 1) / (p + 1)

def _calc_b(p, q, f, n):
    return f * cond(n == 1, __calc_b_first, __calc_b, p, q, f, n)

def __calc_a_add(p, q, f, n):
    return (2 * f * p * (-2 + p + 2 * q)) / ((-4 + 2 * n + p) * (-2 + 2 * n + p) * (2 * n + p))

def _calc_a_add(p, q, f, n):
    return cond(n == 1, _calc_a, __calc_a_add, p, q, f, n)

def _logbetainc(p, q, x, eps=1e-9, max_n=200):
    def lentz_iteration(prev):
        res, params, c, d, i = prev
        an = _calc_a(*params, i)
        bn = _calc_b(*params, i)
        c = an + bn / c
        t = an + bn * d
        c = jnp.where(c <= 0, eps, c)
        t = jnp.where(t <= 0, eps, t)
        d = 1.0 / t
        res += jnp.log(c) - jnp.log(t)
        return res, params, c, d, i + 1
    def lentz_cond(prev):
        return (jnp.abs(1 - prev[-2] * prev[-3]) > eps) & (prev[-1] <= max_n)
    f = x / (1.0 - x)
    b = betaln(p, q)
    res = p * jnp.log(x) + (q - 1.0) * jnp.log1p(-x) - jnp.log(p) - b
    c = 1.0
    d = 0.0
    carry = while_loop(lentz_cond, lentz_iteration, (res, (p, q, f), c, d, 1))
    return carry[0] 

def _betainc(p, q, x, eps=1e-9, max_n=20):
    return jnp.exp(_logbetainc(p, q, x, eps=eps, max_n=max_n))


def _betaincc(p, q, x, eps=1e-9, max_n=20):
    return -jnp.expm1(_logbetainc(q, p, 1.0 - x, eps=eps, max_n=max_n))


def _logbetaincc(p, q, x, eps=1e-9, max_n=20):
    return jnp.log(_betaincc(p, q, x, eps=eps, max_n=max_n))

def logbetainc(p, q, x, eps=1e-12, max_n=1000):
    @jnp.vectorize
    def fun(p, q, x):
        c = q < (1 - x) / x * p
        return cond(c, partial(_logbetainc, eps=eps, max_n=max_n), partial(_logbetaincc, eps=eps, max_n=max_n), p, q, x)
    return fun(p, q, x)


def betainc(p, q, x, eps=1e-12, max_n=1000):
    @jnp.vectorize
    def fun(p, q, x):
        c = q < (1 - x) / x * p 
        return cond(c, partial(_betainc, eps=eps, max_n=max_n), partial(_betaincc, eps=eps, max_n=max_n), p, q, x)
    return fun(p, q, x)
