# -*- coding: utf-8 -*-
import gmpy2
import jax.numpy as jnp
import numpy as np
import jax
from jax.scipy.linalg import toeplitz
from collections.abc import Callable
from itertools import chain


def hankel(a, b):
    a = jnp.flip(a)
    b = jnp.append(jnp.array([0.0]), b)
    return jnp.flip(toeplitz(a, b), axis=0)


def recurrent_fun(rec_start: Callable[[int, ...], jnp.ndarray],
                  rec_step: Callable[[int, jnp.ndarray, ...], float],
                  rec_order: int,
                  min_x: int, max_x: int, max_sz: int, *args, **kwargs) -> jnp.ndarray:
    def loop_body(x, res):
        res = res.at[x].set(rec_step(x, res, *args, **kwargs))
        return res
    
    res = jnp.zeros(max_sz, dtype=float)
    start = rec_start(min_x, *args, **kwargs)
    for i in range(rec_order):
        res = res.at[min_x + i].set(start.at[i].get())
    return jax.lax.fori_loop(min_x + rec_order, max_x, loop_body, res) 

def recurrent_fun_long(rec_start: Callable[[int, ...], np.ndarray],
                  rec_step: Callable[[int, np.ndarray, ...], float],
                  rec_mult: Callable[[...], float], 
                  rec_order: int,
                  x: np.ndarray, *args, **kwargs) -> np.ndarray:
    min_x = min(x)
    max_x = max(x)
    res = np.empty_like(x, dtype=object)
    x = set(x)
    prev = rec_start(min_x, *args, **kwargs)
    j = 0

    for i in range(rec_order):
        if i + min_x in x:
            res[j] = prev[i]
            j += 1
    for i in range(min_x + rec_order, max_x + 1):
        val = rec_step(i, prev, *args, **kwargs)
        prev[:-1] = prev[1:]
        prev[-1] = val
        if i in x:
            res[j] = val
            j += 1
    return res * rec_mult(*args, **kwargs)


def long_vectorize(fun):
    def wrapper(self, x, *args, **kwargs):
        iterable = np.iterable(x)
        if not iterable:
            x = [x]
        assert is_sorted(x), 'Long algebra function accepts only sorted lists of unique values.'
        x = np.array([gmpy2.mpfr(x) if x != round(x) else gmpy2.mpz(x) for x in x])
        non_scalar_shape = None
        for param in chain(args, kwargs.values()):
            try:
                size = len(param)
                if non_scalar_shape is None:
                    non_scalar_shape = size
                elif size != non_scalar_shape:
                    raise IndexError('All arrays must have same size.')
            except TypeError:
                continue
        if non_scalar_shape is None:
            res = fun(self, x, *args, **kwargs)
            return res if iterable else res[0]
        res = list()
        for i in range(non_scalar_shape):
            res.append(fun(self, x, *(r[i] if np.iterable(r) else r for r in args),
                               **{k: v[i] if np.iterable(v) else v for k, v in kwargs.items()}))
        return np.array(res).T
    return wrapper

def is_sorted(x):
    for i in range(1, len(x)):
        if x[i - 1] >= x[i]:
            return False
    return True