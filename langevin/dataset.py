import numpy as np
import jax.numpy as jnp
from tensorflow.keras.datasets import mnist, cifar10
from skimage.transform import resize


def dataset(N_tr, dataset_name = 'mnist', model_type = 'fnn', resized = None):
    
    if dataset_name == 'mnist':
        width = 28
        n0 = width**2
        nd = 10
        ch = 1
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == 'cifar10':
        width = 32
        n0 = width**2
        nd = 10
        ch = 3
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape(-1)
            
    sort_idx = np.argsort(y_train[:N_tr])
    x_train = x_train[:N_tr][sort_idx]
    y_train = y_train[:N_tr][sort_idx]
    y_train = np.eye(nd)[y_train]
    
    if resized != None:
        width = resized
        n0 = width**2
        x_train = resize(x_train, (N_tr,width,width,ch))
        
    x_train = x_train.reshape(N_tr, -1)
    x_train = x_train.T - jnp.mean(x_train, axis = 1)
    x_train = (x_train / jnp.linalg.norm(x_train, axis = 0)).T
    
    if model_type == 'cnn':
        x_train = x_train.reshape(N_tr,width,width,ch)
        
    if model_type == 'cnn1d':
        x_train = x_train.reshape(N_tr,width**2,ch)

    return jnp.array(x_train), jnp.array(y_train.reshape(N_tr,nd))