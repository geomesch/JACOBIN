# -*- coding: utf-8 -*-
"""
Some helper functions for reparametrizing distributions in terms of  mean and variance.
"""

def binomial(mean, var, p_success=True):
    p =  1 - var / mean
    n = mean / p
    return n, p if p_success else 1 - p

def nbinom(mean, var, p_success=True):
    p = mean / var
    r = mean * p / (1 - p)
    return r, 1 - p if p_success else p

# Shape parameters of a beta distribution with the given mean and
# variance, var < mean * (1 - mean).
def beta(mean, var):
    t = mean * (1 - mean) / var - 1
    a = mean * t
    b = (1 - mean) * t
    return a, b
