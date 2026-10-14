# -*- coding: utf-8 -*-
"""
This module concerns mostly effective gradient-friendly computation of sums
of hypergeometric type: sums of probabilities whose consecutive ratio is
known, and 2F1, 1F1 functions of the marginalized compound distributions.
"""
import jax.numpy as jnp
from jax.lax import while_loop
from functools import lru_cache
import gmpy2
from .utils import loop_differentiable, kahan_add, log1p

_BIG = 1e150
_LOG_BIG = 345.38776394910684


# Builds fun(n, *args) that returns the logarithm of t_0 + ... + t_n,
# t_0 = 1, t_{j+1} = t_j * ratio(j, *args). bound(j, rt, *args) bounds
# the absolute values of the ratios from j on, rt is the last ratio.
# The sum stops early when the bound makes the remainder negligible.
def _series(ratio, bound, eps, max_n):
    def fun(n, *args):
        def body(carry):
            s, t, j, shift, _ = carry
            rt = ratio(j, *args)
            t = t * rt
            s = s + t
            j = j + 1.0
            big = jnp.abs(s) > _BIG
            s = jnp.where(big, s / _BIG, s)
            t = jnp.where(big, t / _BIG, t)
            shift = shift + jnp.where(big, _LOG_BIG, 0.0)
            b = bound(j, rt, *args)
            done = jnp.abs(t) * b <= eps * jnp.abs(s) * (1.0 - b)
            return s, t, j, shift, done

        def cond(carry):
            return (carry[2] < n) & (carry[2] < max_n) & ~carry[4]

        s, _, _, shift, _ = while_loop(cond, body, (1.0, 1.0, 0.0, 0.0, False))
        return jnp.log(s) + shift
    return fun


@lru_cache(maxsize=None)
def series_logsum(ratio, bound, eps=1e-18, max_n=200000):
    return jnp.vectorize(loop_differentiable(_series(ratio, bound, eps, max_n)))


def ratio_bound(j, rt, *args):
    return jnp.abs(rt)


# pmf(k - 1) / pmf(k) of the Poisson distribution, k = x - j.
def poisson_lower_ratio(j, x, rate):
    return (x - j) / rate


# pmf(k + 1) / pmf(k) of the Poisson distribution, k = x + 1 + j.
def poisson_upper_ratio(j, x, rate):
    return rate / (x + 2.0 + j)


# pmf(k - 1) / pmf(k) of the beta negative binomial distribution, k = x - j.
def bnb_lower_ratio(j, x, r, a, b):
    k = x - j
    return (k / (k + r - 1.0)) * ((k + a + b + r - 1.0) / (k + b - 1.0))


# pmf(k + 1) / pmf(k) of the beta negative binomial distribution,
# k = x + 1 + j. The terms decay as k ** -(a + 1).
def bnb_upper_ratio(j, x, r, a, b):
    k = x + 1.0 + j
    return ((k + r) / (k + 1.0)) * ((k + b) / (k + a + b + r))


def bnb_upper_bound(j, rt, x, r, a, b):
    k = x + 1.0 + j
    return jnp.maximum(jnp.abs(rt), k / (k + a))


# Ratio of terms of 3F2(1, a + b, a + r; x + 1 + a + b + r, a + 1; 1).
# The series times pmf(x + 1) (x + 1) / a is the sum of the pmf of the
# beta negative binomial distribution over k > x, a transformation by
# Thomae. The terms are positive and decay as j ** -(x + 2).
def bnb_thomae_ratio(j, x, r, a, b):
    return ((a + b + j) / (x + 1.0 + a + b + r + j)) * ((a + r + j) / (a + 1.0 + j))


def bnb_thomae_bound(j, rt, x, r, a, b):
    k = a + 1.0 + j
    return jnp.maximum(jnp.abs(rt), k / (k + x + 1.0))


# pmf(k - 1) / pmf(k) of the beta-binomial distribution, k = x - j.
def betabinom_lower_ratio(j, x, n, a, b):
    k = x - j
    return (k / (n - k + 1.0)) * ((n - k + b) / (k + a - 1.0))


# pmf(k + 1) / pmf(k) of the beta-binomial distribution, k = x + 1 + j.
def betabinom_upper_ratio(j, x, n, a, b):
    k = x + 1.0 + j
    return ((n - k) / (k + 1.0)) * ((k + a) / (n - k - 1.0 + b))


# The pmf has a single mode for a >= 1 and b >= 1 only, an early stop
# is not safe otherwise.
def betabinom_bound(j, rt, x, n, a, b):
    return jnp.where((a >= 1.0) & (b >= 1.0), jnp.abs(rt), 1.0)


# Ratio of terms of 2F1(1 + r, 1 - x; 2; -zeta), a polynomial in zeta of
# degree x - 1 with positive terms.
def hyp_2f1_poly_ratio(j, x, r, zeta):
    return ((1.0 + r + j) / (1.0 + j)) * ((x - 1.0 - j) / (2.0 + j)) * zeta


# Ratio of terms of 1F1(1 - x; 2; -zeta), a polynomial in zeta of
# degree x - 1 with positive terms.
def hyp_1f1_poly_ratio(j, x, r, zeta):
    return ((x - 1.0 - j) / (1.0 + j)) * (zeta / (2.0 + j))


# log(exp(t) - 1) for t > 0.
def log_expm1(t):
    return t + jnp.log(-jnp.expm1(-t))


# Values of 2F1(a, b; 2; z) at b = 1 and b = 2.
def hyp_2f1_rec_start(a, b, z):
    return (1 - z) ** (-a) * (1 - (1-z) ** a - z) / (a - 1 ) / z, (1 - z) ** (-a)

def hyp_log_2f1_rec_start(a, b, log_z):
    z = jnp.exp(log_z)
    t = -log1p(-z)
    return log_expm1(t * (a - 1)) - jnp.log(a - 1) - log_z, t * a

# Coefficients of 2F1(a, b; 2; z) = t1 * 2F1(a, b - 1; 2; z) + t2 * 2F1(a, b - 2; 2; z).
def hyp_2f1_rec_terms(a, b, z):
    t1 = ( 4 -  2 * b + (b - a - 1) * z) / ((b - 1) * (z - 1))
    t2 = (b - 3) / ((b - 1) * (z - 1))
    return t1, t2

# Values of 1F1(a; 2; z) at a = 1 and a = 2.
def hyp_1f1_rec_start(a, z):
    return jnp.expm1(z) / z, jnp.exp(z)

def hyp_long_1f1_rec_start(a, z):
    return gmpy2.expm1(z) / z, gmpy2.exp(z)

def hyp_log_1f1_rec_start(a, log_z):
    z = jnp.exp(log_z)
    return log_expm1(z) - log_z, z

# Coefficients of 1F1(a; 2; z) = t1 * 1F1(a - 1; 2; z) + t2 * 1F1(a - 2; 2; z).
def hyp_1f1_rec_terms(a, z):
    t1 = (2 * a - 4 + z) / (a - 1)
    t2 = (3 - a) / (a - 1)
    return t1, t2


# Coefficients t1, t2 of H(k) = t1 * H(k - 1) + t2 * H(k - 2) where H(k)
# is 2F1(1 + r, 1 + k; 2; z) or 1F1(1 + k; 2; z) for kummer=True. omz is
# 1 - z. Works with jax arrays and with mpfr numbers.
def mc_rec_terms(kummer, k, r, z, omz):
    if kummer:
        return (2 * k - 2 + z) / k, (2 - k) / k
    return (2 * (k - 1) + (r + 1 - k) * z) / (k * omz), (2 - k) / (k * omz)


# Builds fun(x, lf0, lf1, r, q, z, omz) for a pmf f(k) = M q ** k H(k),
# H as in mc_rec_terms. lf0 and lf1 are log f(0) and log f(1). Returns
# the logarithms of the sums of f over k <= x and over k > x, the latter
# is -inf for tail=False.
def _mc_sums(kummer, tail, eps, max_n):
    def fun(x, lf0, lf1, r, q, z, omz):
        rho_inf = q if kummer else q / omz

        def body(carry):
            k, lf, comp, rho, lcdf, lsf, _ = carry
            t1, t2 = mc_rec_terms(kummer, k, r, z, omz)
            rho = q * (t1 + t2 * q / rho)
            lf, comp = kahan_add(lf, comp, jnp.log(rho))
            low = k <= x
            lcdf = jnp.where(low, jnp.logaddexp(lcdf, lf), lcdf)
            lsf = jnp.where(low, lsf, jnp.logaddexp(lsf, lf))
            b = jnp.maximum(rho, rho_inf)
            rest = lf + jnp.log(b) - jnp.log(jnp.abs(1.0 - b))
            done = ~low & (b < 1.0) & (rest <= lsf + jnp.log(eps))
            return k + 1.0, lf, comp, rho, lcdf, lsf, done

        def cond(carry):
            k, done = carry[0], carry[-1]
            return (k <= x) | (tail & ~done & (k <= x + max_n))

        lcdf = jnp.where(x >= 1, jnp.logaddexp(lf0, lf1), lf0)
        lsf = jnp.where(x >= 1, -jnp.inf, lf1)
        carry = while_loop(cond, body, (2.0, lf1, 0.0, jnp.exp(lf1 - lf0), lcdf, lsf, False))
        return jnp.stack([carry[4], carry[5] if tail else -jnp.inf])
    return fun


@lru_cache(maxsize=None)
def mc_logsums(kummer, tail, eps=1e-18, max_n=200000):
    fun = loop_differentiable(_mc_sums(kummer, tail, eps, max_n))
    return jnp.vectorize(fun, signature='(),(),(),(),(),(),()->(2)')
