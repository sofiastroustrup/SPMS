# script with all functions that are used in several notebooks. 
import matplotlib.pyplot as plt 
import jax 
import jax.numpy as jnp
import numpy as np
import Bio.Phylo as Phylo


from .setup_SDEs import dWs


def get_tree_covariance(treepath):
    tree = Phylo.read(treepath, "newick")
    terminals = tree.get_terminals()
    n = len(terminals)
    depths = tree.depths()
    tree_cov = np.zeros((n, n))
    for i, t1 in enumerate(terminals):
        for j, t2 in enumerate(terminals):
            mrca = tree.common_ancestor(t1, t2)
            tree_cov[i, j] = depths[mrca]
    return tree_cov


class Uniform:
    """
    Uniform prior distribution with bounded support.
    Provides both sampling and log PDF calculation.
    """
    
    def __init__(self, minval=0, maxval=10):
        """
        Initialize the uniform prior distribution.

        Parameters:
        -----------
        a : float
            Lower bound
        b : float
            Upper bound
        """
        self.minval = minval
        self.maxval = maxval
        self.scale = maxval - minval

    def sample(self, key):
        """
        Sample from a uniform distribution.
        
        Parameters:
        -----------
        key : jax.random.PRNGKey
            JAX random key
            
        Returns:
        --------
        float
            Sampled value from uniform distribution
        """
        return jax.random.uniform(key, minval=self.minval, maxval=self.maxval)

    def logpdf(self, x):
        """
        Calculate log PDF of the uniform prior distribution.
        
        Parameters:
        -----------
        x : float or array
            Point(s) at which to evaluate the log PDF
            
        Returns:
        --------
        float or array
            Log probability density (constant for all values in range)
        """
        # Log PDF is log(1/(b-a)) for a <= x <= b, -inf otherwise
        log_density = -jnp.log(self.scale)
        
        # Check if x is within bounds
        within_bounds = (x >= self.minval) & (x <= self.maxval)

        # Return log density if within bounds, -inf otherwise
        return jnp.where(within_bounds, log_density, -jnp.inf)
    

class MirroredGaussian:
    """
    Mirrored Gaussian proposal distribution with bounded support.
    Provides both sampling and log PDF calculation.
    """

    def __init__(self, tau, minval=0, maxval=10):
        """
        Initialize the proposal distribution.
        As the mirrored Gaussian is symmetric we do not need to compute the log PDF

        Parameters:
        -----------
        tau : float
            Standard deviation of the Gaussian
        a : float
            Lower bound
        b : float
            Upper bound
        """
        self.tau = tau
        self.minval = minval
        self.maxval = maxval

    def sample(self, key, mu):
        """
        Sample from a mirrored Gaussian centered at mu.
        
        Parameters:
        -----------
        key : jax.random.PRNGKey
            JAX random key
        mu : float
            Current parameter value
            
        Returns:
        --------
        float
            New proposed value, mirrored if outside [a, b]
        """
        key, subkey = jax.random.split(key, 2)
        x = mu + self.tau*jax.random.normal(subkey)
        while x<self.minval or x>self.maxval:
            if x<self.minval:
                x = 2*self.minval-x
            elif x>self.maxval:
                x = 2*self.maxval-x
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
    return(jnp.array(values_flat))







#def mirrored_gaussian(key, mu, sd, a=0, b=1):
#    key, subkey = jax.random.split(key, 2)
#    x = mu + sd*jax.random.normal(subkey)
#    while x<a or x>b:
#        if x<a:
#            x = 2*a-x
#        elif x>b:
#            x = 2*b-x
#    return(x)

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

