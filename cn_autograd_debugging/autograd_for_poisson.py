# https://github.com/HIPS/autograd/blob/master/autograd/scipy/stats/poisson.py
import autograd
from autograd.scipy.stats.poisson import logpmf
params2= {'mu':2.0,'loc':5.0}
print(logpmf(5.0-params2['loc'],params2['mu']))
g2 = autograd.grad(logpmf)
g2(5.0-params2['loc'],params2['mu'])