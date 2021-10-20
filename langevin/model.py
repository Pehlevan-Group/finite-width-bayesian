import numpy as np
from neural_tangents import stax
import neural_tangents as nt

from jax.api import jit, grad, vmap
from jax.config import config
config.update("jax_enable_x64", True)

from functools import partial


def fnn(hidden_widths, nonlin, args):
    layers = []
    layers_ker =[]
    
    for i, width in enumerate(hidden_widths):
        layers += [stax.Dense(width, parameterization = 'ntk')]
        if nonlin != 'relu':
            layers_ker += [i]
        else:
            layers += [stax.Relu()]
            layers_ker += [2*i+1]
        
    layers += [stax.Dense(10, parameterization = 'ntk')]
    if nonlin != 'relu': 
        layers_ker += [i+1]
    else:
        layers_ker += [2*i+2]
        
    return layers, layers_ker

def cnn(hidden_widths, nonlin, args):
    layers = []
    layers_ker =[]
    
    window = (3, 3)
    stride = (1,1)
    if args != None:
        window = args
    
    for i, ch in enumerate(hidden_widths):
            layers += [stax.Conv(ch, window, stride, 'CIRCULAR', b_std=0, parameterization = 'ntk')]
            if nonlin != 'relu': 
                layers_ker += [i]
            else:
                layers += [stax.Relu()]
                layers_ker += [2*i+1]
    layers += [stax.Flatten()]
    if nonlin != 'relu': 
        layers_ker += [i+1]
    else:
        layers += [stax.Relu()]
        layers_ker += [2*i+3]
    layers += [stax.Dense(10, parameterization = 'ntk')]
    if nonlin != 'relu': 
        layers_ker += [i+2]
    else:
        layers_ker += [2*i+4]
        
    return layers, layers_ker


def cnn1d(hidden_widths, nonlin, args):
    layers = []
    layers_ker =[]
    
    window = (3,)
    stride = (1,)
    if args != None:
        window = args
    
    for i, ch in enumerate(hidden_widths):
            layers += [stax.Conv(ch, window, stride, 'CIRCULAR', b_std=0, parameterization = 'ntk')]
            if nonlin != 'relu': 
                layers_ker += [i]
            else:
                layers += [stax.Relu()]
                layers_ker += [2*i+1]
    layers += [stax.Flatten()]
    if nonlin != 'relu': 
        layers_ker += [i+1]
    else:
        layers += [stax.Relu()]
        layers_ker += [2*i+3]
    layers += [stax.Dense(10, parameterization = 'ntk')]
    if nonlin != 'relu': 
        layers_ker += [i+2]
    else:
        layers_ker += [2*i+4]
        
    return layers, layers_ker

def model(hidden_widths, nonlin = 'linear', model_type='fnn', args = None):
    
    if model_type == 'fnn':
        layers, layers_ker = fnn(hidden_widths, nonlin, args)
    elif model_type == 'cnn':
        layers, layers_ker = cnn(hidden_widths, nonlin, args)
    elif model_type == 'cnn1d':
        layers, layers_ker = cnn1d(hidden_widths, nonlin, args)
    else:
        raise NotImplementedError("%s is not a valid architecture!"%model_type)

    return layers, layers_ker

def network_fns(layers, x_train):
    ## Create the model functions for each layer
    layer_fns = []
    kernel_fns = []
    emp_kernel_fns = []
    for i, layer in enumerate(layers):
        init_fn, apply_fn, kernel_fn = stax.serial(*(layers[:i+1]))
        layer_fns += [jit(apply_fn)]
        kernel_fns += [jit(kernel_fn)]
        emp_kernel_fns += [jit(partial(nt.empirical_nngp_fn(layer_fns[i]), x_train, None))]
    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn)
    
    return init_fn, apply_fn, kernel_fn, layer_fns, kernel_fns, emp_kernel_fns