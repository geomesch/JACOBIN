# -*- coding: utf-8 -*-
"""
This module concerns mostly effective gradient-friendly computation of 3f2
function and CDFs of negative beta binomial and beta-binomial distributions.
"""
import jax.numpy as jnp
from jax.scipy.special import betaln, logsumexp
from jax.lax import cond, switch, while_loop, select
from functools import partial
import gmpy2



def __calc_one(a1, a2, b1, b2, n):
    return 1.0

def __calc_neg_one(a1, a2, b1, b2, n):
    return -1.0

def __calc_r0(a1, a2, b1, b2, n):
    return -((n + b1 - a2 - 1)*(n + b2 - a2 - 1)) / ((2 * n - 1)*(2 * n + a2 - 1))

def __calc_r1(a1, a2, b1, b2, n):
    return -((n + b1 - 1)*(n + b2 - 1)) / (2 * n * (2 * n + a1 - 1))

def __calc_r2(a1, a2, b1, b2, n):
    return -((n + b1 - a1)*(n + b2 - a1)) / ((2 * n + a1)*(2 * n + a2))

def _calc_r(a1, a2, b1, b2, n):
    i = n % 3 + 3 * (n < 2)
    return switch(i, [__calc_r0, __calc_r1, __calc_r2, __calc_one, __calc_neg_one], a1, a2, b1, b2, n // 3)

def __calc_q0(a1, a2, b1, b2, n):
    return ((3 * n + b1 - 1)*(3 * n + b2 - 1) - 2 * n * (2 * n + a2)) / (2 * n * (2 * n + a1 - 1))

def __calc_q1(a1, a2, b1, b2, n):
    return ((3 * n + b1) * (3 * n + b2) - (2 * n + 1) * (2 * n + a1)) / ((2 * n + a1) * (2 * n + a2))

def __calc_q2(a1, a2, b1, b2, n):
    return ((3 * n + b1 + 1) * (3 * n + b2 + 1) - (2 * n + a1 + 1) * (2 * n + a2 + 1)) / ((2 * n + 1) * (2 * n + a2 + 1))

def _calc_q(a1, a2, b1, b2, n):
    i = n % 3 + 3 * (n == 0)
    return switch(i, [__calc_q0, __calc_q1, __calc_q2, __calc_one], a1, a2, b1, b2, n // 3)

def _hyp3f2(a1, a2, b1, b2, eps=1e-9, max_n=1000, tiny=1e-100):
    def lentz_iteration(prev):
        res, params, c, d, i = prev
        rn = _calc_r(*params, i)
        qn = _calc_q(*params, i)
        c = qn + rn / c
        t = qn + rn * d
        c = jnp.where(c == 0, tiny, c)
        t = jnp.where(t == 0, tiny, t)
        d = 1.0 / t
        res *= c * d
        return res, params, c, d, i + 1
    
    def lentz_cond(prev):
        t = prev[-2] * prev[-3]
        return ((jnp.abs(t - 1) > eps))  & (prev[-1] <= max_n)

    params = (a1, a2, b1, b2)
    res = tiny
    c = res
    d = 0.0
    carry = while_loop(lentz_cond, lentz_iteration, (res, params, c, d, 0))
    return carry[0] - tiny


@partial(jnp.vectorize, signature=(4, 5, 6))
def bnb_cdf(x, r, a, b, sf=False, eps=1e-6, max_n=1000):
    k = a + b
    p = a / k
    
    if sf:
        x, r, p = r - 1, x + 1, 1 - p
    
    def compute_c(x, r, p, k):
        return betaln(r + k * p, k * (1 - p) + x + 1) -(betaln( r, x + 1) + betaln(k * p, k * (1 - p)) + jnp.log(x + 1))
    
    def _cdf(x, r, p, k):
        p = 1 - p
        c = jnp.exp(compute_c(x, r, p, k))
        t = _hyp3f2(r + x + 1, -p * k + k + x + 1, r + k + x + 1, x + 2, eps=eps)
        return 1 - t * c
    
    def _cdfc(x, r, p, k):
        x, r = r, x
        x -= 1
        r += 1
        c = jnp.exp(compute_c(x, r, p, k))
        t = _hyp3f2(r + x + 1, -p * k + k + x + 1, r + k + x + 1, x + 2, eps=eps)
        return t * c
    
    def calc_cond(x, p, r):
        return (1 - p) / p * x
    
    c = r < (1-p) / p * x 
    a_fun = _cdf
    b_fun = _cdfc
    return cond(c, a_fun, b_fun, x, r, p, k)

def betabinom_cdf(x, n, a, b, eps=1e-6, max_n=1000):
    @jnp.vectorize
    def fun(x, n, a, b):
        k = a + b
        p = a / k
        
        def compute_c(x, n, p, k):
            a = p * k
            b = (1 - p) * k
            return betaln(n - x + b - 1, 1 + a + x) - (jnp.log(n + 1) + betaln(a, b) + betaln(n -x, x + 2))
        
        def _cdf(x, n, p, k):
            c = jnp.exp(compute_c(x, n, p, k))
            a = p * k
            b = (1 - p) * k
            t = _hyp3f2(x + a + 1, x + 1 - n, x + 2, x + 2 - n - b, eps=eps, max_n=max_n)
            return 1 - t * c
        
        def _cdfc(x, n, p, k):
            x = n - x - 1
            c = jnp.exp(compute_c(x, n, p, k))
            a = (1 - p) * k
            b = p * k
            t = _hyp3f2(x + a + 1, x + 1 - n, x + 2, x + 2 - n - b, eps=eps, max_n=max_n)
            return t * c

        c = x < p * n
        a_fun = _cdf
        b_fun = _cdfc
        return cond(c, a_fun, b_fun, x, n, p, k)
    return fun(x, n, a, b)

def hyp_2f1_rec_start(a, b, z):
    return (1 - z) ** (-a) * (1 - (1-z) ** a - z) / (a - 1 ) / z, (1 - z) ** (-a)

def hyp_log_2f1_rec_start(a, b, log_z):
    z = jnp.exp(log_z)
    t2 = -jnp.log1p(-z) * a
    t1 = t2 + logsumexp(jnp.array([-t2, jnp.log1p(-z)]), b=jnp.array([-1.0, 1.0])) - jnp.log(a - 1) - log_z
    return t1, t2

def hyp_2f1_rec_terms(a, b, z):
    t1 = ( 4 -  2 * b + (b - a - 1) * z) / ((b - 1) * (z - 1))
    t2 = (b - 3) / ((b - 1) * (z - 1))
    return t1, t2

def hyp_1f1_rec_start(a, z):
    return jnp.expm1(z) / z, jnp.exp(z)

def hyp_long_1f1_rec_start(a, z):
    return gmpy2.expm1(z) / z, gmpy2.exp(z)

def hyp_log_1f1_rec_start(a, log_z):
    z = jnp.exp(log_z)
    t = select(jnp.isfinite(jnp.exp(z)), jnp.log(jnp.exp(z) - 1), z)
    return t - log_z, z

def hyp_1f1_rec_terms(a, z):
    t1 = (2 * a - 4 + z) / (a - 1)
    t2 = (3 - a) / (a - 1)
    return t1, t2
    