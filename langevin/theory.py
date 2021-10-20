import numpy as np
import jax.numpy as jnp
import neural_tangents as nt
import langevin.utils.convert_nt as convert_nt 

from jax.config import config
config.update("jax_enable_x64", True)


def theory_fnn(x_train, y_train, beta, kernel_fns, hidden_widths):
    
    N_tr = x_train.shape[0]
    n0 = x_train.shape[1]
    nd = y_train.shape[1]
    
    Gxx = x_train @ x_train.T / n0
    Gyy = y_train @ y_train.T / nd
    
    K_nngp =  [jnp.zeros(shape=(N_tr,N_tr)) for _ in range(len(kernel_fns))]
    for i in range(len(kernel_fns)):
        K_nngp[i] = kernel_fns[i](x_train,).nngp

    I = jnp.eye(N_tr)
    gamma = I/beta + Gxx
    gamma_inv = jnp.linalg.inv(gamma)
    prefactor = jnp.cumsum(nd/jnp.array(hidden_widths))

    self_energy = (Gyy - Gxx - I/beta) 

    K_theory = []

    for i in range(len(prefactor)):
        K_theory += [(Gxx @ (I + prefactor[i]*gamma_inv @ self_energy @ gamma_inv @ Gxx))]
#     for i in range(len(prefactor)):
#         K_theory += [(K_nngp[i] @ (I + prefactor[i]*gamma_inv @ self_energy @ gamma_inv @ K_nngp[i]))]
        
    return K_nngp, K_theory, Gxx, Gyy

def theory_fnn_new(x_train, y_train, beta, kernel_fns, hidden_widths):
    
    N_tr = x_train.shape[0]
    n0 = x_train.shape[1]
    nd = y_train.shape[1]
    
    Gxx = x_train @ x_train.T / n0
    Gyy = y_train @ y_train.T / nd
    
    K_nngp =  [jnp.zeros(shape=(N_tr,N_tr)) for _ in range(len(kernel_fns))]
    for i in range(len(kernel_fns)):
        K_nngp[i] = kernel_fns[i](x_train,).nngp

    I = jnp.eye(N_tr)
    gamma = I/beta + Gxx
    gamma_inv = jnp.linalg.inv(gamma)
    Phi = gamma_inv@(Gyy - K_nngp[-1] - I/beta)@gamma_inv
    
    prefactor = jnp.cumsum(nd/jnp.array(hidden_widths))
        
    K_theory = []
    for i in range(len(prefactor)):
        K_theory += [(K_nngp[i] + prefactor[i]*correction_layer(K_nngp[i], Phi))]
        
#     K_theory = []
#     for i in range(len(prefactor)):
#         print((abs(K_nngp[i])-abs(Gxx)).std())
#         K_theory += [(K_nngp[i] @ (I + prefactor[i]*Phi @ K_nngp[i]))]
#     K_theory = []    
#     for i in range(len(prefactor)):
#         K_theory += [(Gxx @ (I + prefactor[i]*Phi @ Gxx))]

    return K_nngp, K_theory, Gxx, Gyy


def theory_cnn(x_train, y_train, beta, kernel_fns, hidden_widths):
    
    N_tr = x_train.shape[0]
    n0 = x_train.shape[1]*x_train.shape[2]
    nd = y_train.shape[1]
    
    Gxx = jnp.moveaxis(jnp.tensordot(x_train, x_train, (3, 3)), (3), (1)) ## Tensordot in channel axis
    Gyy = y_train @ y_train.T / nd
    
    K_nngp =  []
    for i in range(len(kernel_fns)):
        print(convert_nt(kernel_fns[i](x_train,).nngp).shape)
        K_nngp += [convert_nt(kernel_fns[i](x_train,).nngp, i)]
        
    KPsi = jnp.trace(Gxx.reshape(N_tr,N_tr,D,D), axis1=2, axis2 =3)/n0
#     KPsi_2 = x_train.reshape(N_tr,-1)@x_train.reshape(N_tr,-1).T/D
#     print((KPsi-KPsi_2).std())
    
    I = jnp.eye(N_tr)
    gamma = KPsi + I/beta
    gamma_inv = jnp.linalg.inv(gamma)
    Phi = gamma_inv @ (Gyy - KPsi - I/beta) @ gamma_inv

    prefactor = jnp.cumsum(nd/jnp.array(hidden_widths))
    
    K_theory = []
    for i in range(len(prefactor)):
        K_theory += [K_nngp[i] + prefactor[i]*correction_layer(K_nngp[i], Phi)]

    return K_nngp, K_theory, Gxx, Gyy


def theory_linear(x_train, y_train, beta, kernel_fns, hidden_widths):
    
    N_tr = x_train.shape[0]
    if len(x_train.shape) == 4:
        n0 = x_train.shape[1]*x_train.shape[2]
    elif len(x_train.shape) == 3:
        n0 = x_train.shape[1]
    else:
        n0 = x_train.shape[-1]
        
    x_train_flat = x_train.reshape(N_tr,-1)
    nd = y_train.shape[1]
    
    Gxx = x_train_flat @ x_train_flat.T / n0
    Gyy = y_train @ y_train.T / nd
    
    K_nngp =  []
    for i in range(len(kernel_fns)):
        K_nngp += [convert_nt(kernel_fns[i](x_train,).nngp, i)]
    
    I = jnp.eye(N_tr)
    gamma = Gxx + I/beta
    gamma_inv = jnp.linalg.inv(gamma)
    Phi = gamma_inv @ (Gyy - Gxx - I/beta) @ gamma_inv

    prefactor = jnp.cumsum(nd/jnp.array(hidden_widths))
    
    K_theory = []
    for i in range(len(prefactor)):
        K_theory += [K_nngp[i] + prefactor[i]*correction_layer(K_nngp[i], Phi)]

    return K_nngp, K_theory, Gxx, Gyy


def correction_layer(Kl, Phi):
    if len(Kl.shape) == 2: ## For FNNs
        return Kl @ Phi @ Kl
    
    elif len(Kl.shape) == 4: ## For 1D CNNs
        N_tr = Kl.shape[0]
        D = Kl.shape[-1]

        correction = 0
        for i in range(N_tr):
            for j in range(N_tr):
                correction += Phi[i,j]*jnp.tensordot(Kl[:,i],Kl[j,:], axes=((-1),(1)))/D

        return np.moveaxis(correction, 2,1)
    
    elif len(Kl.shape) == 6: ## For 2D CNNs
        N_tr = Kl.shape[0]
        w = Kl.shape[-1]
        D = w**2
    
        Kl = Kl.reshape(N_tr, N_tr, D, D)
        correction = 0
        for i in range(N_tr):
            for j in range(N_tr):
                correction += Phi[i,j]*jnp.tensordot(Kl[:,i],Kl[j,:], axes=((-1),(1)))/D

        return np.moveaxis(correction, 2,1).reshape(N_tr,N_tr,w,w,w,w)
    
    
    else:
        raise NotImplementedError('wtf')
        
        
# def convert_nt(tensor, conv_layer = None):
    
#     if len(tensor.shape) != 6:
#         return tensor
#     elif conv_layer != None:
#         if (conv_layer + 1) % 2 == 1:
#             tensor = np.moveaxis(tensor, (2,3), (4,5))
    
#     return np.moveaxis(tensor, (3,4),(4,3))
