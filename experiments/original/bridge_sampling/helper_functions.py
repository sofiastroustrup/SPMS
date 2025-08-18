# script with all functions that are used in several notebooks. 
import matplotlib.pyplot as plt 
import jax 
import jax.numpy as jnp
from .setup_SDEs import dWs
import numpy as np
#import matplotlib

def mirrored_gaussian(key, mu, sd, a=0, b=1):
    key, subkey = jax.random.split(key, 2)
    x = mu + sd*jax.random.normal(subkey)
    while x<a or x>b:
        if x<a:
            x = 2*a-x
        elif x>b:
            x = 2*b-x
    return(x)

def crank_nicholson_step(key, dtsdWsT, lambd):
    '''lambda denotes the degree of correlation. If lambda = 0 the samples are 
    independent. If lambda = 1, the samples are 100% correlated.''' 
    (_dts,_dWs),dWs_children = dtsdWsT
    d = _dWs.shape[1]
    key, *keys = jax.random.split(key, num=len(dWs_children)+1)
    W = dWs(d, key, _dts)
    Zcirc = _dWs*lambd + jnp.sqrt((1-lambd**2))*W
    return((_dts, Zcirc), [crank_nicholson_step(key, child, lambd) for child, key in zip (dWs_children, keys)])

def get_flat_values_sim(tree):
    '''function for getting flat values from simulated tree, level order traversal '''
    _,_,vals, children = tree
    values_flat = [vals[-1]]
    queue = [children[0], children[1]]
    while len(queue)>0: 
        _,_,vals, children = queue.pop(0)
        values_flat.append(vals[-1])
        if len(children)>0:
            queue.append(children[0])
            queue.append(children[1])
    return(np.array(values_flat))

def get_flat_values(tree):  
    '''function for getting flat values from proposed tree, levelorder traversal'''
    _,_,vals,_, children = tree
    values_flat = [vals[-1]]
    queue = [children[0], children[1]]
    while len(queue)>0: 
        _,_,vals, _, children = queue.pop(0)
        values_flat.append(vals[-1])
        if len(children)>0:
            queue.append(children[0])
            queue.append(children[1])
    return(np.array(values_flat))



#def prior_pars(kalpha_loc, kalpha_scale, gtheta_loc, gtheta_scale, key):
#    key, *subkeys = jax.random.split(key,3)
#    kalpha = jax.random.uniform(subkeys[0],minval = kalpha_loc  , maxval= kalpha_loc+kalpha_scale)
#    gtheta = jax.random.uniform(subkeys[1], minval = gtheta_loc, maxval = gtheta_loc+gtheta_scale)
#    while gtheta-20*kalpha < 0.2:
#        key, *subkeys = jax.random.split(key,3)
#        kalpha = jax.random.uniform(subkeys[0],minval = kalpha_loc  , maxval= kalpha_loc+kalpha_scale)
#        gtheta = jax.random.uniform(subkeys[1], minval = gtheta_loc, maxval = gtheta_loc+gtheta_scale)
#    return(kalpha, gtheta)
    


#def folded_gaussian(key, mu, sd):
#    key, subkey = jax.random.split(key, 2)
#    x = mu + sd*jax.random.normal(subkey) 
#    return(abs(x))

#def folded_gaussian_logpdf(x, mu, sd):
#    densx = jax.scipy.stats.norm.logpdf(x-mu, loc=0, scale=sd)
#    densxabs = jax.scipy.stats.norm.logpdf(x+mu, loc=0, scale=sd)
#    print(densx)
#    print(densxabs)
#    return(np.log(np.exp(densx)+np.exp(densxabs)))

