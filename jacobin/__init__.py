__version__ = '1.3.0'
from jax import config
# The package works in 64-bit floats, the accuracy of every routine relies on it.
config.update("jax_enable_x64", True)

from . import distributions
from . import bivariate_distributions
from . import reparam
from .distributions import Distribution, TruncatedDistribution, Mixture, CompoundDist, \
    Poisson, Binomial, NB, BetaBinomial, BetaNB, MCNB
from .bivariate_distributions import SSDistribution, Mixture2D, NegativeMultinomial, \
    DirichletNegativeMultinomial, MCNegativeMultinomial

__all__ = ['distributions', 'bivariate_distributions', 'reparam',
           'Distribution', 'TruncatedDistribution', 'Mixture', 'CompoundDist',
           'Poisson', 'Binomial', 'NB', 'BetaBinomial', 'BetaNB', 'MCNB',
           'SSDistribution', 'Mixture2D', 'NegativeMultinomial',
           'DirichletNegativeMultinomial', 'MCNegativeMultinomial']
