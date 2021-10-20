import jax.numpy as jnp
from jax import random
from jax.experimental.optimizers import optimizer, make_schedule, Optimizer


@optimizer
def sgld(step_size, beta, batch_factor):
    step_size = make_schedule(step_size)
    def init(x0):
        key0 = random.PRNGKey(1)
        return x0, key0
    
    def update(i, g, state):
        x, key = state
        _, key = random.split(key)
        x -= step_size(i) * (batch_factor*g + x/beta) + jnp.sqrt(2*step_size(i)/beta) * random.normal(key, shape=g.shape)
        return x, key
    
    def get_params(state):
        x, _ = state
        return x
    
    return Optimizer(init, update, get_params)