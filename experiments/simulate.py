import os
import numpy as np
from ete3 import Tree
import jax
import jax.numpy as jnp
import argparse
import sys 
import subprocess

from bridge_sampling.setup_SDEs import Stratonovich_to_Ito, dtsdWsT, dWs
from bridge_sampling.noise_kernel import Q12
from bridge_sampling.helper_functions import get_flat_values_sim
from bridge_sampling.simulate import simulate_tree


def simulate_shapes(ds, dt, sigma, alpha, root, tree, rb=0, d=2, outputfolder='', sti=1):
    '''
    Function to simulate shapes
    
    Parameters
    ----------
    ds : int
        Random seed for the simulation.
    dt : float
        Time step size for the simulation.
    sigma : float
        Sigma parameter for the simulation.
    alpha : float
        Alpha parameter for the simulation.
    root : np.ndarray
        Root data as a numpy array.
    tree : ete3.Tree
        ETE3 tree object for simulation.
    rb : float
        Random branch length for the simulation.
    sti : 0, 1 
        Whether to use Stratonovich-Ito correction (1) or not (0).
    d : int, optional
        Dimension of the data, by default 2.
    outputfolder : str, optional
        Path to the output folder, by default ''.

    Returns
    -------
    ete3.Tree
        Simulated tree with data.   
    '''
    n = root.shape[0] // d

    theta_true = {
    'k_alpha': alpha, # kernel amplitud
    'inv_k_sigma': 1./(sigma)*jnp.eye(d), # kernel width, gets squared in kQ12
    'd':d,
    'n':n}
    
    # define stochastic process terms
    if sti ==1:
        b,sigma,_ = Stratonovich_to_Ito(lambda t,x,theta: jnp.zeros(n*d),
                               lambda t,x,theta: Q12(x,theta))
    else:
        b = lambda t,x,theta: jnp.zeros(n*d)
        sigma = lambda t,x,theta: Q12(x,theta)
        
    # simulate data 
    key = jax.random.PRNGKey(ds)
    key, subkey = jax.random.split(key)
    tree.dist = rb
    for node in tree.traverse("levelorder"): 
        node.add_feature('T', round(node.dist,1)) # this is a choice for simulation, could be different
        node.add_feature('n_steps', round(node.T/dt))
        node.add_feature('message', None)
        node.dist = node.T
    if rb==0:
        key, *subkeys = jax.random.split(key, len(tree.children)+1)
        _dts = jnp.array([0]); _dWs = jnp.array([0]); Xscirc = root.reshape(1,-1) # set variables for root 
        children = [tree.children[0],tree.children[1]]
        dWs_children = [dtsdWsT(tree.children[i],subkeys[i], lambda ckey, _dts: dWs(n*d,ckey, _dts)) for i in range(len(tree.children))]
        stree = [_dts, _dWs, Xscirc, [simulate_tree(Xscirc[-1], b, sigma, theta_true, dtsdWs_child) for dtsdWs_child in dWs_children]]
    else:
        stree = simulate_tree(root, b, sigma, theta_true, dtsdWsT(tree,subkey, lambda ckey, _dts: dWs(n*d,ckey, _dts)))

    return(stree)