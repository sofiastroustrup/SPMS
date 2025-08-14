import jax
import jax.numpy as jnp
import numpy as np 
from functools import partial

def dWs(d, key, _dts):
    return jnp.sqrt(_dts)[:,None]*jax.random.normal(key,(_dts.shape[0],d)) 

# time increments
def dts(T=1.,n_steps=100):
    return jnp.array([T/n_steps]*n_steps)

def dtsdWsT(node, key, dWs):
    children = node.children
    keys = jax.random.split(key, num=len(children))
    _dts = dts(T=node.T, n_steps = node.n_steps)
    return (_dts,dWs(key, _dts)),[dtsdWsT(child,key, dWs) for child, key in zip(children, keys)]

# Ito integral
def ito(carry, y, b, sigma, theta):
    t, X = carry
    dt, dW = y
    X_cum = X + b(t,X, theta)*dt+jnp.dot(sigma(t,X, theta),dW)
    t_cum = t + dt
    return((t_cum, X_cum), (t, X)) 

def ito_integrator(x, b, sigma, _dts, _dWs, theta):
    _ito = partial(ito, b=b, sigma=sigma, theta=theta)
    (T, X), (ts, Xs)=jax.lax.scan(_ito,(0., x),(_dts, _dWs))
    Xs = jnp.vstack((Xs, X))
    return(Xs)

# Stratonovich-Ito conversion
# see e.g. https://math.stackexchange.com/questions/2296945/conversion-between-solution-to-stratonovich-sde-and-it%C3%B4-sde
def Stratonovich_to_Ito(b,sigma):
    correction = lambda t,x,theta: .5*jnp.einsum('ikj,jk->i',jax.jacfwd(sigma,1)(t,x,theta),sigma(t,x,theta))
    b_ito = lambda t,x,theta: b(t,x,theta)+correction(t,x,theta)
    return b_ito,sigma,correction







