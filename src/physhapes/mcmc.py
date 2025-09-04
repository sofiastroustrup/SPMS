import os
import sys
import numpy as np
from ete3 import Tree
import jax
import jax.numpy as jnp
from tqdm import tqdm 
import pandas as pd
import wandb
import pickle 
import glob

from .BFFG import backward_filter, forward_guide, forward_guide_edge, get_logpsi
from .setup_SDEs import dtsdWsT, dWs
from .helper_functions import *



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
    n=20,               # Number of landmarks 
    d=2,                # Dimensionality of landmarks
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
    initialized_tree = forward_guide(xs, data_tree_bf,_dtsdWsT, fge) 
    logpsicur = get_logpsi_jit(initialized_tree)
    #print(f'Initial logpsicur {logpsicur}')
    logrhotildecur = compute_logrhotilde(data_tree_bf, xs) #-data_tree_bf.message['c']-0.5*xs.T@data_tree_bf.message['H'][0]@xs+data_tree_bf.message['F'][0].T@xs
    #print(f'Initial logrhotildecur {logrhotildecur}')
    # results
    guided_tree = get_flat_values(initialized_tree) 

    # organize logging
    max_trees = 1 + N * 3  # Initial + worst case acceptances
    trees = np.zeros((max_trees, *guided_tree.shape))
    trees[0] = guided_tree
    tree_idx = 1  # index for next
    alphas = np.zeros(N+1); alphas[0] = alpha_cur
    sigmas = np.zeros(N+1); sigmas[0] = sigma_cur
    log_posterior = np.zeros(N)
    acceptpath = np.zeros(N+1); acceptpath[0] = 1 # initial path always accepted
    acceptsigmas = np.zeros(N+1); acceptsigmas[0] = 1 # initial sigma always accepted
    acceptalphas = np.zeros(N+1); acceptalphas[0] = 1 # initial alpha always accepted
    #acceptpathall = []
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
            acceptpath[j+1] = 1
            # save new paths             
            guided_tree = get_flat_values(guidedcirc) 
            #trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
            #tree_idx += 1
            #tree_counter.append(1)
        #else: 
        #    acceptpathall.append(0)
            #tree_counter[-1]+=1   
        # log 
        trees[tree_idx] = guided_tree; tree_idx += 1
        #print(f"logpsicur after path update {logpsicur}")
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
        logrhotildecirc = compute_logrhotilde(tree_bf_circ, xs) #-tree_bf_circ.message['c']-0.5*xs.T@tree_bf_circ.message['H'][0]@xs+tree_bf_circ.message['F'][0].T@xs
        
        # get acceptance probability added if/else to handle out of bounds proposals
        if sigmacirc < prior_sigma.minval or sigmacirc > prior_sigma.maxval:
            A = 0  # Explicitly reject proposals outside prior range
        else:
            # Regular acceptance calculation
            logprob_cur = logpsicur + logrhotildecur + prior_sigma.logpdf(sigma_cur)
            logprob_circ = logpsicirc + logrhotildecirc + prior_sigma.logpdf(sigmacirc)
            log_r = logprob_circ - logprob_cur 
            A = min(1, np.exp(log_r))
        
        #logprob_cur = logpsicur + logrhotildecur + prior_sigma.logpdf(sigma_cur)
        #logprob_circ = logpsicirc + logrhotildecirc + prior_sigma.logpdf(sigmacirc)
        #log_r = logprob_circ - logprob_cur #logpsicirc - logpsicur + logrhotildecirc - logrhotildecur + scipy.stats.uniform.logpdf(sigmacirc, loc=gtheta_loc, scale=gtheta_scale) - scipy.stats.uniform.logpdf(gtheta_cur, loc=gtheta_loc, scale=gtheta_scale) 
        #A = min(1, np.exp(log_r))
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
            acceptsigmas[j+1] = 1

            # save new paths
            guided_tree = get_flat_values(guidedcirc) #used to be get_flat_values_root_branch
            #trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
            #trees[tree_idx] = guided_tree
           # tree_idx += 1
           # tree_counter.append(1)
        #else: 
            #tree_counter[-1]+=1  
        #    pass
        # store values 
        trees[tree_idx] = guided_tree; tree_idx += 1
        sigmas[j+1] = sigma_cur
        #print(f"logpsicur after sigma update {logpsicur}")
        #print(f"logrhotildecur after sigma update {logrhotildecur}")
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
        logrhotildecirc = compute_logrhotilde(tree_bf_circ, xs) #-tree_bf_circ.message['c']-0.5*xs.T@tree_bf_circ.message['H'][0]@xs+tree_bf_circ.message['F'][0].T@xs

        # get acceptance probability added if/else to handle out of bounds proposals
        if alphacirc < prior_alpha.minval or alphacirc > prior_alpha.maxval:
            A = 0  # Explicitly reject proposals outside prior range
        else:
            # Regular acceptance calculation
            logprob_cur = logpsicur + logrhotildecur + prior_alpha.logpdf(alpha_cur)
            logprob_circ = logpsicirc + logrhotildecirc + prior_alpha.logpdf(alphacirc)
            log_r = logprob_circ - logprob_cur 
            A = min(1, np.exp(log_r))
        #logprob_cur = logpsicur + logrhotildecur + prior_alpha.logpdf(alpha_cur)        
        #logprob_circ = logpsicirc + logrhotildecirc + prior_alpha.logpdf(alphacirc)
        #log_r = logpsicirc - logpsicur + logrhotildecirc - logrhotildecur + scipy.stats.uniform.logpdf(kalphacirc, loc=kalpha_loc, scale=kalpha_scale) - scipy.stats.uniform.logpdf(kalpha_cur, loc=kalpha_loc, scale=kalpha_scale)  --- IGNORE ---
        #log_r = logprob_circ - logprob_cur #logpsicirc - logpsicur + logrhotildecirc - logrhotildecur + scipy.stats.uniform.logpdf(kalphacirc, loc=kalpha_loc, scale=kalpha_scale) - scipy.stats.uniform.logpdf(kalpha_cur, loc=kalpha_loc, scale=kalpha_scale) 
        #A = min(1, np.exp(log_r))
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
            acceptalphas[j+1] = 1

            # save new paths
            guided_tree = get_flat_values(guidedcirc)
            #trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
            #trees[tree_idx] = guided_tree
            #tree_idx += 1
            #tree_counter.append(1)
        #else: 
            #tree_counter[-1]+=1
        #print(f"logpsicur after alpha update {logpsicur}")
        #print(f"logrhotildecur after alpha update {logrhotildecur}")
        # store values 
        alphas[j+1] = alpha_cur
        trees[tree_idx] = guided_tree; tree_idx += 1
        log_posterior[j] = logpsicur + logrhotildecur + prior_alpha.logpdf(alpha_cur) + prior_sigma.logpdf(sigma_cur)        
        inner = dict([(str(i),guided_tree[2][i]) for i in range(2)])
        tolog = dict([('root-'+str(l),guided_tree[0][l]) for l in range(2)])
        tolog.update(inner)
        tolog.update({"alpha": alpha_cur, "sigma": sigma_cur, "log_posterior": log_posterior[j]})

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
        "log_posterior": log_posterior,
        "trees": trees,
        "alpha": alphas,
        "sigma": sigmas,
        "acceptsigma": acceptsigmas,  # Uncomment if gtheta acceptance is used
        "acceptalpha": acceptalphas,  # Uncomment if kalpha acceptance is used
        "acceptpath": acceptpath,
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


def load_mcmc_results(filepath_pattern):
    """
    Load MCMC results from pickle files matching the given pattern.
    
    Args:
        filepath_pattern: Pattern to match pickle files (e.g., "results/chain_*.pkl")
        
    Returns:
        List of loaded results
    """
    results = []
    for filepath in sorted(glob.glob(filepath_pattern)):
        print(f"Loading {filepath}")
        with open(filepath, 'rb') as f:
            results.append(pickle.load(f))
    return results

