import jax.numpy as jnp
import numpy as np

#%%
kQ12 = lambda x,theta: theta['k_alpha']*jnp.exp(-.5*jnp.square(jnp.tensordot(x,theta['inv_k_sigma'],(x.ndim-1,1))).sum(x.ndim-1))

# evaluate k on two pairs of landmark configurations
kQ12_q = lambda q1,q2,theta: kQ12(q1.reshape((-1,theta['d']))[:,np.newaxis,:]-q2.reshape((-1,theta['d']))[np.newaxis,:,:],theta)

# evaluate k on one landmark configurations against itself with each landmark pair resulting in a dxd matric
# i,jth entry of result is kQ12(x_i,x_j)*eye(d)
Q12 = lambda q,theta: jnp.einsum('ij,kl->ikjl',kQ12_q(q,q,theta),jnp.eye(2)).reshape((theta['n']*theta['d'],theta['n']*theta['d']))


def Q(q,theta):
    Q12qtheta = Q12(q,theta)
    return jnp.einsum('ij,kj->ik',Q12qtheta,Q12qtheta)

# fast evaluation of Q at endpoint
#Qv = lambda theta: Q(theta['v'],theta) if theta['Qv'] is None else theta['Qv']
#invQv = lambda theta: jnp.linalg.inv(Q(theta['v'],theta)) if theta['invQv'] is None else theta['invQv']

#Qv = lambda theta: Q(theta['v'],theta) 
#invQv = lambda theta: jnp.linalg.inv(Q(theta['v'],theta)) 
# %%
