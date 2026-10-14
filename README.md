# jacobin
JACOBIN: JAX Compound Binomials, head-counting distributions

Count distributions in JAX: `Poisson`, `Binomial`, `NB`, `BetaBinomial`, `BetaNB`, and `MCNB`, the marginalized compound negative binomial of MixALime. Each has `logpmf`, `pmf`, `cdf`, `sf`, `logcdf`, `logsf`, `mean`, `var`, `sample`, a table of the log-pmf by a recurrence (`logpmf_recurrent`), and `long_pmf`, `long_cdf`, `long_sf` in multiple precision (gmpy2) for p-values far below 1e-16. `TruncatedDistribution` and `Mixture` wrap any of them. The float functions work under `jit`, `vmap`, `grad` and `jacfwd`.

Importing the package switches JAX to 64-bit floats (`jax_enable_x64`). The accuracy of every routine relies on it.

## Installation

```
pip install .
```

The tests need `pytest`, `mpmath` and `betanegbinfit`, run them with `python -m pytest tests`.

## Conventions

- Positional order: `Poisson(x, rate)`, `Binomial(x, n, p)`, `NB(x, r, p)`, `BetaBinomial(x, n, a, b)`, `BetaNB(x, r, a, b)`, `MCNB(x, r, p1, p2)`. Parameters can be passed as keywords.
- `NB()` has pmf proportional to `p ** x`, `NB(p_success=False)` to `(1 - p) ** x`. `BetaNB` is `NB` with the probability of the power of `r` drawn from Beta(a, b).
- `MCNB(CompoundDist.Binomial).logpmf(x, r, p)` is the MCNB(r, p) of the MixALime paper, `p2` is `1 - p1` by default.
- `sf(x)` is P(X > x).
- Bounds of `TruncatedDistribution` are the keyword arguments `left` and `right` of each method, both inclusive.
- `logpmf_recurrent(min_x, max_x, max_sz, *params)` returns `max_sz` values for x = 0, ..., `max_sz - 1`. Entries outside of [`min_x`, `max_x`], both inclusive, are `-inf`.

## Example

```python
import numpy as np
import jax
import jax.numpy as jnp
from jacobin import NB, MCNB, CompoundDist, TruncatedDistribution

x = jnp.array([5.0, 9.0, 40.0])
dist = TruncatedDistribution(MCNB(CompoundDist.Binomial))

loglik = lambda v: dist.logpmf(x, v[0], v[1], left=5).sum()
v = jnp.array([7.5, 0.6])
grad = jax.jit(jax.grad(loglik))(v)
hess = jax.jit(jax.jacfwd(jax.jacfwd(loglik)))(v)

# A table per value of r: 3 rows, counts 0..99 in columns.
tables = jax.vmap(lambda r: dist.logpmf_recurrent(0, None, 100, r, 0.6, left=5))(jnp.array([3.3, 7.5, 20.0]))

mean = dist.mean(7.5, 0.6, left=5)

# P(X > x | X >= 5) as mpfr numbers of 1024 bits: 9.18e-190, 0.788, 2.75e-06.
pvals = dist.long_sf(np.array([900, 5, 40]), 7.5, 0.6, left=5)
```
