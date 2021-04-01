# https://github.com/HIPS/autograd/blob/master/autograd/scipy/stats/poisson.py
import autograd
from autograd.scipy.stats.poisson import logpmf
def logpmf_at_x(params,x):
    return logpmf(x-params['loc'],params['mu'])
params2= {'mu':2.0,'loc':5.0}
print(logpmf_at_x(params2,5.0))
g2 = autograd.grad(logpmf_at_x)
g2(params2,5.0)