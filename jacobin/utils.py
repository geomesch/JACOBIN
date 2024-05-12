# -*- coding: utf-8 -*-
import gmpy2
import numpy as np
from itertools import chain


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