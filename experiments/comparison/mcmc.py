import os
import sys
import numpy as np
from ete3 import Tree
import Bio.Phylo as Phylo
import jax
import jax.numpy as jnp
from tqdm import tqdm 
import pandas as pd
import wandb
#import argparse
import scipy
import pickle 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.backends.backend_pdf as backend_pdf


from bridge_sampling.BFFG import backward_filter, forward_guide, forward_guide_edge, get_logpsi
from bridge_sampling.setup_SDEs import Stratonovich_to_Ito, dtsdWsT, dWs
from bridge_sampling.noise_kernel import Q12
from bridge_sampling.helper_functions import *
import subprocess
import time





def expand_mcmc_samples(trees, tree_counter):
    """
    Expand MCMC samples according to tree counter values.
    
    Parameters:
    -----------
    trees : ndarray
        Array of tree states from MCMC (shape: [n_samples, ...])
    tree_counter : list
        Number of times each tree state should be repeated
        
    Returns:
    --------
    ndarray
        Expanded array of tree states
    """
    expanded = []
    assert len(tree_counter) <= trees.shape[0], "More counters than trees"
    
    for i, count in enumerate(tree_counter):
        if i < trees.shape[0]:  # Safety check
            for _ in range(count):
                expanded.append(trees[i])
    
    return np.array(expanded)


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


get_logpsi_jit = jax.jit(get_logpsi)

def compute_logrhotilde(tree_bf, xs):
    return -tree_bf.message['c'] - 0.5*xs.T@tree_bf.message['H'][0]@xs + tree_bf.message['F'][0].T@xs
# -data_tree_bf.message['c']-0.5*xs.T@data_tree_bf.message['H'][0]@xs+data_tree_bf.message['F'][0].T@xs


def metropolis_hastings(
    N,                   # Number of MCMC steps
    dt,                  # stepsize
    lambd,               # Lambda parameter
    obs_var,             # Observation variance
    rb,                  # Root branch length
    xs,                  # Shape at super root 
    drift_term,       # Drift term for the SDE
    diffusion_term,      # Diffusion term for the SDE
    prior_sigma,         # Prior on gtheta
    prior_alpha,         # Prior on kalpha
    proposal_alpha,      # Proposal variance for kalpha
    proposal_sigma,      # Proposal variance for gtheta
    tree,                # ETE3 tree object
    leaves,               # Leaf values for the tree
    #burnin_percent=0.2, # Percentage of burn-in
    n=20,               # Number of landmarks 
    d=2,                # Dimensionality of landmarks
    #sti=1,               # Whether to use Stratonovich to Ito conversion
    outputpath='mcmc',     # Path to save output
    seed_mcmc=42,        # MCMC seed
    use_wandb=False,     # Whether to log to wandb
    wandb_project=None,  # Wandb project name
    wandb_config=None,   # Additional wandb config
    save_interval=100    # How often to save results
):
    """
    Run MCMC for SPMS model with optional wandb logging
    
    Prior on parameters: uniform distribution
    Proposal distribution: mirrored Gaussian on interval (0,10)
    
    Parameters:
    -----------
        
    Returns:
    --------
    dict
        Dictionary with MCMC results
    """
    
    # Initialize output path
    #if outputpath is None:
    outputpath = f"{outputpath}" #os.path.join(f"mcmc_seed={seed_mcmc}")
    os.makedirs(outputpath, exist_ok=True)
    
    # Initialize wandb if requested
    if use_wandb:
        if wandb_config is None:
            wandb_config = {}
            
        # Add parameters to config
        run_config = {
            "mcmc_steps": N,
            "dt": dt,
            "lambda": lambd,
            "obs_var": obs_var,
            "root_branch": rb,
            "seed": int(seed_mcmc),
            **wandb_config
        }
        
        # Initialize wandb
        wandb.init(
            project=wandb_project,
            #entity=wandb_entity,
            config=run_config
        )    

    # prep tree inference tree
    tree.dist = rb # set ls i.e. super root branch length
    for node in tree.traverse("levelorder"): 
        if not abs(round(node.dist/dt) - node.dist/dt) < 1e-10:
            print(f"Node distance must be divisible by dt, got {node.dist}")
            sys.exit(1)
        node.add_feature('T', node.dist) 
        node.add_feature('n_steps', round(node.T/dt)) 
        node.add_feature('message', None)
        node.add_feature('theta', False)

    i=0
    for leaf in tree: 
        leaf.name = i
        leaf.add_feature('v', leaves[i])
        leaf.add_feature('obs_var', obs_var)
        i+=1
    
    
    # Initialize MCMC chain
    key = jax.random.PRNGKey(seed_mcmc)
    key, *subkeys = jax.random.split(key,3)
    alpha_cur = prior_alpha.sample(subkeys[0])
    sigma_cur = prior_sigma.sample(subkeys[1])

    theta_cur = {
    'k_alpha': alpha_cur, # kernel amplitude
    'inv_k_sigma': 1./(sigma_cur)*jnp.eye(d),
    'd':d,
    'n':n, 
    }
    
    # backwards filter 
    data_tree_bf = backward_filter(tree, theta_cur, diffusion_term)
    # get Wiener process and steps on entire tree
    key, subkey = jax.random.split(key,2) 
    _dtsdWsT = dtsdWsT(tree, subkey, lambda ckey,_dts: dWs(n*d,ckey,_dts))

    # Initiate tree
    fge = jax.jit(lambda *x: forward_guide_edge(*x, drift_term, diffusion_term, theta_cur))
    #fge = lambda *x: forward_guide_edge(*x, drift_term, diffusion_term, theta_cur) #create_forward_guide_edge_jit(drift_term, diffusion_term, theta_cur)
    initialized_tree = forward_guide(xs, data_tree_bf,_dtsdWsT, fge) 
    logpsicur = get_logpsi_jit(initialized_tree)
    logrhotildecur = compute_logrhotilde(data_tree_bf, xs) #-data_tree_bf.message['c']-0.5*xs.T@data_tree_bf.message['H'][0]@xs+data_tree_bf.message['F'][0].T@xs

    # results
    guided_tree = get_flat_values(initialized_tree) 
    #print(guided_tree)
    #trees = np.expand_dims(guided_tree, axis=0)
    tree_counter = [1]

    # organize logging
    max_trees = 1 + N * 3  # Initial + worst case acceptances
    trees = np.zeros((max_trees, *guided_tree.shape))
    trees[0] = guided_tree
    tree_idx = 1  # index for next
    alphas = np.zeros(N+1); alphas[0] = alpha_cur
    sigmas = np.zeros(N+1); sigmas[0] = sigma_cur
    log_posterior = np.zeros(N)
    acceptpath = np.zeros(N+1)
    acceptsigmas = np.zeros(N)
    acceptalphas = np.zeros(N)
    acceptpathall = []
    for j in tqdm(range(N)):
        ##################
        ## Propose path ##
        ##################
        key, subkey = jax.random.split(key, 2)
        # take a step
        _dtsdWsTcirc = crank_nicholson_step(subkey, _dtsdWsT, lambd)
        guidedcirc = forward_guide(xs, data_tree_bf,_dtsdWsTcirc, fge)
        logpsicirc = get_logpsi(guidedcirc)
        
        # calculate acceptance probability
        log_r = logpsicirc - logpsicur
        A = min(1, np.exp(log_r))
        #print(f'path acceptance probability {A}')      
        key, subkey = jax.random.split(key, 2)
        if jax.random.uniform(subkey)<A:
            # update driving noise 
            _dtsdWsT = _dtsdWsTcirc
            # update probabilities
            logpsicur = logpsicirc
            # update statistics
            acceptpath[j] = 1
            acceptpathall.append(1)
            # save new paths             
            guided_tree = get_flat_values(guidedcirc) 
            #trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
            trees[tree_idx] = guided_tree
            tree_idx += 1
            tree_counter.append(1)
        else: 
            acceptpathall.append(0)
            tree_counter[-1]+=1   
        
        # log 
        #log_posterior[j] = logpsicur 
        #inner = dict([(str(i),guided_tree[2][i]) for i in range(2)])
        #tolog = dict([('root-'+str(l),guided_tree[0][l]) for l in range(2)])
        #tolog.update(inner)
        #tolog.update({'log_posterior': logpsicur})
        
        #######################
        ##   propose sigma  ##
        #######################
        # propose parameter, proposal is mirrored gaussian with sd
        key, subkey = jax.random.split(key, 2)
        sigmacirc = proposal_sigma.sample(subkey, sigma_cur)#mirrored_gaussian(subkey, gtheta_cur, tau_s, 0, 10) # symmetric proposal
        thetacirc = theta_cur.copy()
        thetacirc['inv_k_sigma']= 1./(sigmacirc)*jnp.eye(d) # update kernel width
        
        # do backwards filter using new parameter
        tree_bf_circ = backward_filter(tree, thetacirc, diffusion_term)
        # get paths for new parameter same wiener process 
        fgecirc = jax.jit(lambda *x: forward_guide_edge(*x, drift_term, diffusion_term, thetacirc))
        guidedcirc = forward_guide(xs, tree_bf_circ,_dtsdWsT, fgecirc)  
        logpsicirc = get_logpsi(guidedcirc)
        logrhotildecirc = compute_logrhotilde(data_tree_bf, xs) #-tree_bf_circ.message['c']-0.5*xs.T@tree_bf_circ.message['H'][0]@xs+tree_bf_circ.message['F'][0].T@xs
        
        # get acceptance probability
        logprob_cur = logpsicur + logrhotildecur + prior_sigma.logpdf(sigma_cur)
        logprob_circ = logpsicirc + logrhotildecirc + prior_sigma.logpdf(sigmacirc)
        log_r = logprob_circ - logprob_cur #logpsicirc - logpsicur + logrhotildecirc - logrhotildecur + scipy.stats.uniform.logpdf(sigmacirc, loc=gtheta_loc, scale=gtheta_scale) - scipy.stats.uniform.logpdf(gtheta_cur, loc=gtheta_loc, scale=gtheta_scale) 
        A = min(1, np.exp(log_r))
        #print(f'gtheta acceptance probability {A}')

        key, subkey = jax.random.split(key, 2)
        if jax.random.uniform(subkey)<A: 
            # update variables
            sigma_cur = sigmacirc
            data_tree_bf = tree_bf_circ
            theta_cur = thetacirc
            fge = fgecirc

            # update probabilities
            logrhotildecur = logrhotildecirc 
            logpsicur = logpsicirc

            # update statistics 
            acceptsigmas[j] = 1

            # save new paths
            guided_tree = get_flat_values(guidedcirc) #used to be get_flat_values_root_branch
            #trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
            trees[tree_idx] = guided_tree
            tree_idx += 1
            tree_counter.append(1)

        else: 
            tree_counter[-1]+=1  

        # store values 
        acceptpathall.append(0) # store in order to have path updates and innernode match
        sigmas[j+1] = sigma_cur
        #inner = dict([(str(i),guided_tree[2][i]) for i in range(2)])
        #tolog = dict([('root-'+str(l),guided_tree[0][l]) for l in range(2)])
        #tolog.update(inner)
        #tolog.update({"sigma": sigma_cur})
        #wandb.log(tolog) 


        #######################
        ##   propose kalpha  ##
        #######################
        # propose parameter, proposal is mirrored gaussian with sd
        key, subkey = jax.random.split(key, 2)
        alphacirc = proposal_alpha.sample(subkey, alpha_cur) #mirrored_gaussian(subkey, kalpha_cur, proposal_sd_kalpha, 0, 10) 
        thetacirc = theta_cur.copy()
        thetacirc['k_alpha']= alphacirc # propose rate 

        # do backwards filter using new parameter
        tree_bf_circ = backward_filter(tree, thetacirc, diffusion_term)
        
        # get paths for new parameter same wiener process 
        fgecirc = jax.jit(lambda *x: forward_guide_edge(*x, drift_term, diffusion_term, thetacirc))
        guidedcirc = forward_guide(xs, tree_bf_circ,_dtsdWsT, fgecirc)
        logpsicirc = get_logpsi(guidedcirc)
        logrhotildecirc = -tree_bf_circ.message['c']-0.5*xs.T@tree_bf_circ.message['H'][0]@xs+tree_bf_circ.message['F'][0].T@xs

        # get acceptance probability
        logprob_cur = logpsicur + logrhotildecur + prior_alpha.logpdf(alpha_cur)        
        logprob_circ = logpsicirc + logrhotildecirc + prior_alpha.logpdf(alphacirc)
        #log_r = logpsicirc - logpsicur + logrhotildecirc - logrhotildecur + scipy.stats.uniform.logpdf(kalphacirc, loc=kalpha_loc, scale=kalpha_scale) - scipy.stats.uniform.logpdf(kalpha_cur, loc=kalpha_loc, scale=kalpha_scale)  --- IGNORE ---
        log_r = logprob_circ - logprob_cur #logpsicirc - logpsicur + logrhotildecirc - logrhotildecur + scipy.stats.uniform.logpdf(kalphacirc, loc=kalpha_loc, scale=kalpha_scale) - scipy.stats.uniform.logpdf(kalpha_cur, loc=kalpha_loc, scale=kalpha_scale) 
        A = min(1, np.exp(log_r))
        #print(f'kalpha acceptance probability {A}')

        key, subkey = jax.random.split(key, 2)
        if jax.random.uniform(subkey)<A: 
            # update variables 
            alpha_cur = alphacirc
            theta_cur = thetacirc
            data_tree_bf = tree_bf_circ
            fge = fgecirc
            
            # update probabilities
            logrhotildecur = logrhotildecirc
            logpsicur = logpsicirc 

            # update statistics
            acceptalphas[j] = 1

            # save new paths
            guided_tree = get_flat_values(guidedcirc)
            #trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
            trees[tree_idx] = guided_tree
            tree_idx += 1
            tree_counter.append(1)
        else: 
            tree_counter[-1]+=1
        
        # store values 
        log_posterior[j] = logpsicur + logrhotildecur + prior_alpha.logpdf(alpha_cur) + prior_sigma.logpdf(sigma_cur)
        acceptpathall.append(0) # store in order to have path updates and innernode match
        alphas[j+1] = alpha_cur
        inner = dict([(str(i),guided_tree[2][i]) for i in range(2)])
        tolog = dict([('root-'+str(l),guided_tree[0][l]) for l in range(2)])
        tolog.update(inner)
        tolog.update({"alpha": alpha_cur, "sigma": sigma_cur, "log_posterior": log_posterior})

        # Conditional logging, every 10 iterations
        if use_wandb and j % 10 == 0:
            wandb.log(tolog)
            #wandb.log({
            #    "alpha": alpha_cur,
            #    "sigma": sigma_cur,
            #    "log_posterior": log_posterior,
            #    "acceptance_rate_path": jnp.mean(acceptpath[:j+1])
            #})

        # Write to disk periodically
        if j % save_interval == 0 or j == N-1:
            pass
            #np.savetxt(f"{outputpath}/tree_counter.txt", tree_counter, fmt='%s')
            #with open(f"{outputpath}/trees.pkl", 'wb') as f:  # 'wb' for write binary mode
            #    pickle.dump(trees, f)
            #np.savetxt(f"{outputpath}/flat_tree.txt", trees, fmt='%s')
            #np.savetxt(f"{outputpath}/kalpha_{j}.txt", kalpha_samples[:j+1])
            #np.savetxt(f"{outputpath}/gtheta_{j}.txt", gtheta_samples[:j+1])
    
    # Finalize wandb run
    if use_wandb:
        wandb.finish()
    
    # Return results dictionary
    results = {
        "acceptpath": acceptpath,
        "log_posterior": log_posterior,
        "trees": trees,
        "tree_counter": tree_counter,
        "alphas": alphas,
        "sigmas": sigmas,
        "acceptpathall": acceptpathall,
        "acceptsigma": acceptsigmas,  # Uncomment if gtheta acceptance is used
        "acceptalpha": acceptalphas,  # Uncomment if kalpha acceptance is used
        "settings": {
            "N": N,
            "dt": dt,
            "lambd": lambd,
            "obs_var": obs_var,
            "rb": rb,
            "seed_mcmc": seed_mcmc,
            "prior_sigma_min": prior_sigma.minval,
            "prior_sigma_max": prior_sigma.maxval,
            "prior_alpha_min": prior_alpha.minval,
            "prior_alpha_max": prior_alpha.maxval,
            "proposal_sigma_tau": proposal_sigma.tau,
            "proposal_alpha_tau": proposal_alpha.tau,
            "tree_string": tree.write(format=1),  # Newick format
            "outputpath": outputpath,
            "use_wandb": use_wandb,
            "wandb_project": wandb_project
        }
    }
    
    return results



def run_mcmc_in_screens(num_chains, script_path="run_mcmc.py", screen_prefix="mcmc_chain", 
                         seed_start=42, seed_param="--seed_mcmc", script_args=None):
    """
    Run multiple MCMC chains in separate screen sessions with customizable arguments.
    
    Args:
        num_chains: Number of chains to run
        script_path: Path to the MCMC script
        screen_prefix: Prefix for screen session names
        seed_start: Starting seed value
        seed_param: Parameter name for seed in script
        script_args: Dictionary of additional arguments to pass to the script
                    (e.g., {"--dt": 0.05, "--N": 5000})
    
    Returns:
        List of screen session names
    """
    screen_names = []
    
    # Convert script_args dictionary to command-line string
    args_str = ""
    if script_args:
        for key, value in script_args.items():
            args_str += f" {key} {value}"
    
    for i in range(num_chains):
        seed = seed_start + i
        screen_name = f"{screen_prefix}_{i+1}"
        screen_names.append(screen_name)
        
        # Create the screen session and run the script with all arguments
        cmd = (
            f"screen -dmS {screen_name} bash -c '"
            f"python {script_path} {seed_param} {seed}{args_str}; "
            "echo \"Chain complete, press Ctrl+C to exit.\"; "
            "sleep infinity'"
        )
        
        print(f"Starting chain {i+1} with seed {seed} in screen '{screen_name}'")
        subprocess.run(cmd, shell=True)
        time.sleep(1)  # Small delay to avoid resource contention
    
    print(f"\n{num_chains} MCMC chains started in separate screen sessions.")
    print("To attach to a screen session: screen -r <screen_name>")
    print("To detach from a screen session: Ctrl+A, then D")
    print(f"Screen sessions: {', '.join(screen_names)}")
    
    return screen_names