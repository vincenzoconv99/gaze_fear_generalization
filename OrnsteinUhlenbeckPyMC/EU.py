import theano.tensor as tt
import pymc3 as pm
from pymc3.distributions import distribution, multivariate, continuous

class Mv_EulerMaruyama(distribution.Continuous):
    """
    Stochastic differential equation discretized with the Euler-Maruyama method.
    Parameters
    ----------
    dt : float
        time step of discretization
    sde_fn : callable
        function returning the drift and diffusion coefficients of SDE
    sde_pars : tuple
        parameters of the SDE, passed as *args to sde_fn
    """
    def __init__(self, dt, sde_fn, sde_pars, *args, **kwds):
        super(Mv_EulerMaruyama, self).__init__(*args, **kwds)
        self.dt = dt = tt.as_tensor_variable(dt)
        self.sde_fn = sde_fn
        self.sde_pars = sde_pars

    def logp(self, x):
        xt = x[:-1,:]
        f, g = self.sde_fn(x[:-1,:], *self.sde_pars)
        mu = xt + self.dt * f
        cov = self.dt * g
        #cov = tt.sqrt(self.dt) * g
        #sd = extract_diag(cov)
        #print(sd.tag.test_value.shape)
        #print(mu.tag.test_value.shape)
        #print(cov.tag.test_value.shape)
        #print(x[1:,:].tag.test_value.shape)
        #input('pause')
        #res = pm.MvNormal.dist(mu=mu, cov=cov).logp(x[:,1:])
        res = pm.MvNormal.dist(mu=mu, cov=cov).logp(x[1:,:])
        #res = pm.Normal.dist(mu=mu, sd=sd).logp(x[1:,:])
        return tt.sum(res)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        dt = dist.dt
        name = r'\text{%s}' % name
        return r'${} \sim \text{EulerMaruyama}(\mathit{{dt}}={})$'.format(name,get_variable_name(dt))