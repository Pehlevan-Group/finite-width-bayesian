import jax.numpy as jnp
from jax import random
from jax.experimental.optimizers import optimizer, make_schedule, Optimizer

## Uses Stochastic Gradient Langeving Dynamics, Welling, Teh, 2011
## https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf
@optimizer
def sgld(step_size, beta, batch_factor):
    step_size = make_schedule(step_size)
    def init(x0):
        key = random.PRNGKey(1)
        return x0, key
    
    def update(i, g, state):
        x, key = state
        _,key = random.split(key)
        x -= step_size(i) * (batch_factor*g + x/beta) + jnp.sqrt(2*step_size(i)/beta) * random.normal(key, shape=g.shape)
        return x, key
    
    def get_params(state):
        x,_ = state
        return x
    
    return Optimizer(init, update, get_params)

## Uses Preconditioned Stochastic Gradient Langeving Dynamics, Li, Chen, Carlson, Carin 2015
## https://arxiv.org/abs/1512.07666
@optimizer
def psgld(step_size, beta, batch_factor, alpha=0.99, eps=1e-4): ## RMSprop implementation with noise
    step_size = make_schedule(step_size)
    def init(x0):
        v = jnp.zeros_like(x0)
        key = random.PRNGKey(1)
        return x0, v, key
    
    def update(i, g, state):
        
        x, v, key = state
        _,key = random.split(key)
        
        dt = step_size(i)
        ghat = g*batch_factor + x/beta
        
        v = alpha*v + (1. - jnp.asarray(alpha, ghat.dtype))*jnp.square(ghat)
        vhat = v / (1 - jnp.asarray(alpha, v.dtype) ** (i + 1))
        G = 1./(jnp.sqrt(vhat) + eps)
        
        x = x - dt*G*ghat + jnp.sqrt(2*dt/beta*G)*random.normal(key, shape=g.shape) 
        
        return x, v, key
    
    def get_params(state):
        x, _, _ = state
        return x
    
    return Optimizer(init, update, get_params)


## Adam optimizer with noise injection based on https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/optimizers.html#adam
@optimizer
def adam(step_size, beta, batch_factor, b1=0.9, b2=0.999, eps=1e-8):
    
    step_size = make_schedule(step_size)
    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        key = random.PRNGKey(1)
        return x0, m0, v0, key
    def update(i, g, state):
        x, m, v, key = state
        _,key = random.split(key)
        
        dt = step_size(i)/(1 - jnp.asarray(b1, m.dtype) ** (i + 1))
        ghat = g*batch_factor + x/beta
        
        # First moment estimate.
        m = (1 - b1) * ghat + b1 * m 
        
        # Second moment estimate.
        v = (1 - b2) * jnp.square(ghat) + b2 * v
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        G = 1./(jnp.sqrt(vhat) + eps)

        x = x - dt*m*G + jnp.sqrt(2*dt/beta*G)*random.normal(key, shape=g.shape)

        return x, m, v, key
  
    def get_params(state):
        x, _, _, _ = state
        return x
    return init, update, get_params

def optimizer_sgld(opt_mode, step_size, beta, batch_factor, **kwargs):
    
    if opt_mode == 'sgld':
        return sgld(step_size, beta, batch_factor)
    elif opt_mode == 'psgld':
        return psgld(step_size, beta, batch_factor, alpha = 0.99, eps=1e-4)
    elif opt_mode == 'adam':
        return adam(step_size, beta, batch_factor, b1=0.9, b2=0.999, eps=1e-8)
    else:
        raise NotImplementedError('%s is not implemented'%opt_mode)
