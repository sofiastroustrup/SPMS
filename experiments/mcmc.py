import os
import numpy as np
from ete3 import Tree
import jax
import jax.numpy as jnp
from tqdm import tqdm 
import wandb
import argparse
import scipy

from bridge_sampling.BFFG import backward_filter, forward_guide, forward_guide_edge, get_logpsi
from bridge_sampling.setup_SDEs import Stratonovich_to_Ito, dtsdWsT, dWs
from bridge_sampling.noise_kernel import Q12
from bridge_sampling.helper_functions import *


def metropolis_hastings(
    N,                   # Number of MCMC steps
    dt,                  # stepsize
    lambd,               # Lambda parameter
    ov,                  # Observation variance
    rb,                  # Root branch length
    xs,                  # Shape at super root 
    proposal_,               # Proposal variance for kalpha
    tau_s,               # Proposal variance for gtheta
    tree,                # ETE3 tree object
    datapath,            # Path to data
    n=20,               # Number of landmarks 
    d=2,                # Dimensionality of landmarks
    sti=1,               # Whether to use Stratonovich to Ito conversion
    outputpath=None,     # Path to save output
    seed_mcmc=42,        # MCMC seed
    use_wandb=False,     # Whether to log to wandb
    wandb_project=None,  # Wandb project name
    wandb_entity=None,   # Wandb entity name
    wandb_config=None,   # Additional wandb config
    save_interval=100    # How often to save results/thinning
):
    """
    Run MCMC for SPMS model with optional wandb logging
    
    Prior on parameters: uniform distribution
    Proposal distribution: mirrored Gaussian on interval (0,10)
    
    Parameters:
    -----------
    ms : int
        Number of MCMC steps
    dt : float
        Time step for simulation
    lambd : float
        Lambda parameter value
    ov : float
        Observation variance
    rb : float
        Root branch length
    tau_a : float
        Proposal variance for kalpha parameter
    tau_s : float
        Proposal variance for sigma parameter
    datapath : str
        Path to data directory
    outputpath : str, optional
        Path to save output, if None will use datapath/mcmc_results
    seed_mcmc : int, optional
        Random seed for MCMC
    use_wandb : bool, optional
        Whether to log to wandb
    wandb_project : str, optional
        Wandb project name
    wandb_entity : str, optional
        Wandb entity name
    wandb_config : dict, optional
        Additional wandb config
    save_interval : int, optional
        How often to save results
        
    Returns:
    --------
    dict
        Dictionary with MCMC results
    """
    
    # Initialize output path
    if outputpath is None:
        outputpath = os.path.join(datapath, f"mcmc_seed={seed_mcmc}")
    os.makedirs(outputpath, exist_ok=True)
    
    # Initialize wandb if requested
    if use_wandb:
        if wandb_config is None:
            wandb_config = {}
            
        # Add parameters to config
        run_config = {
            "mcmc_steps": ms,
            "dt": dt,
            "lambda": lambd,
            "obs_var": ov,
            "root_branch": rb,
            "tau_alpha": tau_a,
            "tau_sigma": tau_s,
            "seed": seed_mcmc,
            **wandb_config
        }
        
        # Initialize wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config=run_config
        )
    
    # define drift and diffusion for process of interest 
    if sti ==1:
        b,sigma,_ = Stratonovich_to_Ito(lambda t,x,theta: jnp.zeros(n*d),
                                lambda t,x,theta: Q12(x,theta))
    else:
        b = lambda t,x,theta: jnp.zeros(n*d)
        sigma = lambda t,x,theta: Q12(x,theta)
    
    # Load data
    leaves = np.genfromtxt(datapath + '/leaves.csv', delimiter=',')

    # prep tree inference tree
    tree.dist = rb # set ls i.e. super root branch length
    for node in tree.traverse("levelorder"): 
        if not abs(round(node.dist/dt) - node.dist/dt) < 1e-10:
            print(f"Node distance must be divisible by dt, got {node.dist}")
            sys.exit(1)
        node.add_feature('n_steps', round(node.T/dt)) # obs add some kind of control here 
        node.add_feature('T', node.dist) 
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
    kalpha_cur = jax.random.uniform(subkeys[0], (1,), minval=kalpha_loc, maxval=kalpha_loc+kalpha_scale)[0]
    gtheta_cur = jax.random.uniform(subkeys[1], (1,), minval=gtheta_loc, maxval=gtheta_loc+gtheta_scale)[0]    

    theta_cur = {
    'k_alpha': kalpha_cur, # kernel amplitude
    'inv_k_sigma': 1./(gtheta_cur)*jnp.eye(d),
    'd':d,
    'n':n, 
    }
    
    # backwards filter 
    data_tree_bf = backward_filter(tree, theta_cur, sigma)
    # get Wiener process and steps on entire tree
    key, subkey = jax.random.split(key,2) 
    _dtsdWsT = dtsdWsT(tree, subkey, lambda ckey,_dts: dWs(n*d,ckey,_dts))

    # Initiate tree
    fge = jax.jit(lambda *x: forward_guide_edge(*x, b, sigma, theta_cur))
    initialized_tree = forward_guide(super_root, data_tree_bf,_dtsdWsT, fge) 
    logpsicur = get_logpsi(initialized_tree)
    logrhotildecur = -data_tree_bf.message['c']-0.5*xs.T@data_tree_bf.message['H'][0]@xs+data_tree_bf.message['F'][0].T@xs

    # results 
    guided_tree = get_flat_values(initialized_tree) 
    trees = np.expand_dims(guided_tree, axis=0)
    tree_counter = [1]

    kalphas = [kalpha_cur]
    gthetas = [gtheta_cur]

    acceptpath = np.zeros(N+1)
    acceptgtheta = np.zeros(N+1)
    acceptkalpha = np.zeros(N+1)
    acceptpathall = []
    # Example of conditional wandb logging
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
        print(f'path acceptance probability {A}')      
        key, subkey = jax.random.split(key, 2)
        if jax.random.uniform(subkey)<A:
            # update driving noise 
            _dtsdWsT = _dtsdWsTcirc
            # update probabilities
            logpsicur = logpsicirc
            # update statistics
            acceptpath[j+1] = 1
            acceptpathall.append(1)
            # save new paths 
            guided_tree = get_flat_values(guidedcirc) 
            trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
            tree_counter.append(1)
        else: 
            acceptpathall.append(0)
            tree_counter[-1]+=1   
                     
        #######################
        ##   propose sigma  ##
        #######################
        # propose parameter, proposal is mirrored gaussian with sd
        key, subkey = jax.random.split(key, 2)
        sigmacirc = mirrored_gaussian(subkey, gtheta_cur, tau_s, 0, 10) # symmetric proposal
        thetacirc = theta_cur.copy()
        thetacirc['inv_k_sigma']= 1./(sigmacirc)*jnp.eye(d) # update kernel width

        # do backwards filter using new parameter
        tree_bf_circ = backward_filter(tree, thetacirc, sigma)
        # get paths for new parameter same wiener process 
        fgecirc = jax.jit(lambda *x: forward_guide_edge(*x, b, sigma, thetacirc))
        guidedcirc = forward_guide(xs, tree_bf_circ,_dtsdWsT, fgecirc)  
        logpsicirc = get_logpsi(guidedcirc)
        logrhotildecirc = -tree_bf_circ.message['c']-0.5*xs.T@tree_bf_circ.message['H'][0]@xs+tree_bf_circ.message['F'][0].T@xs
        
        # get acceptance probability
        log_r = logpsicirc - logpsicur + logrhotildecirc - logrhotildecur + scipy.stats.uniform.logpdf(sigmacirc, loc=gtheta_loc, scale=gtheta_scale) - scipy.stats.uniform.logpdf(gtheta_cur, loc=gtheta_loc, scale=gtheta_scale) 
        A = min(1, np.exp(log_r))
        print(f'gtheta acceptance probability {A}')

        key, subkey = jax.random.split(key, 2)
        if jax.random.uniform(subkey)<A: 
            # update variables
            gtheta_cur = gthetacirc
            data_tree_bf = tree_bf_circ
            theta_cur = thetacirc
            fge = fgecirc

            # update probabilities
            logrhotildecur = logrhotildecirc 
            logpsicur = logpsicirc

            # update statistics 
            acceptgtheta[j+1] = 1

            # save new paths
            guided_tree = get_flat_values(guidedcirc) #used to be get_flat_values_root_branch
            trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
            tree_counter.append(1)

        else: 
            tree_counter[-1]+=1  

        # store values 
        acceptpathall.append(0) # store in order to have path updates and innernode match
        gthetas.append(gtheta_cur)
        inner = dict([(str(i),guided_tree[2][i]) for i in range(2)])
        tolog = dict([('root-'+str(l),guided_tree[0][l]) for l in range(2)])
        tolog.update(inner)
        tolog.update({"gtheta": gtheta_cur})
        wandb.log(tolog) 


        #######################
        ##   propose kalpha  ##
        #######################
        # propose parameter, proposal is mirrored gaussian with sd
        key, subkey = jax.random.split(key, 2)
        kalphacirc = mirrored_gaussian(subkey, kalpha_cur, proposal_sd_kalpha, 0, 10) 
        thetacirc = theta_cur.copy()
        thetacirc['k_alpha']= kalphacirc # propose rate 

        # do backwards filter using new parameter
        tree_bf_circ = backward_filter(bphylogeny, thetacirc, sigma)
        
        # get paths for new parameter same wiener process 
        fgecirc = jax.jit(lambda *x: forward_guide_edge(*x, b, sigma, thetacirc))
        guidedcirc = forward_guide(super_root, tree_bf_circ,_dtsdWsT, fgecirc)
        logpsicirc = get_logpsi(guidedcirc)
        logrhotildecirc = -tree_bf_circ.message['c']-0.5*super_root.T@tree_bf_circ.message['H'][0]@super_root+tree_bf_circ.message['F'][0].T@super_root
        
        # get acceptance probability
        log_r = logpsicirc - logpsicur + logrhotildecirc - logrhotildecur + scipy.stats.uniform.logpdf(kalphacirc, loc=kalpha_loc, scale=kalpha_scale) - scipy.stats.uniform.logpdf(kalpha_cur, loc=kalpha_loc, scale=kalpha_scale) 
        A = min(1, np.exp(log_r))
        print(f'kalpha acceptance probability {A}')

        key, subkey = jax.random.split(key, 2)
        if jax.random.uniform(subkey)<A: 
            # update variables 
            kalpha_cur = kalphacirc
            theta_cur = thetacirc
            data_tree_bf = tree_bf_circ
            fge = fgecirc
            
            # update probabilities
            logrhotildecur = logrhotildecirc
            logpsicur = logpsicirc 

            # update statistics
            acceptkalpha[j+1] = 1

            # save new paths
            guided_tree = get_flat_values(guidedcirc)
            trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
            tree_counter.append(1)
        else: 
            tree_counter[-1]+=1

        # Conditional logging
        if use_wandb and j % 10 == 0:
            wandb.log({
                "kalpha": kalpha_cur,
                "gtheta": gtheta_cur,
                "log_posterior": log_posterior,
                "acceptance_rate": acceptance_rate
            })
        
        # Save results periodically
        if j % save_interval == 0:
            np.savetxt(f"{outputpath}/kalpha_{j}.txt", kalpha_samples[:j+1])
            np.savetxt(f"{outputpath}/gtheta_{j}.txt", gtheta_samples[:j+1])
    
    # Finalize wandb run
    if use_wandb:
        wandb.finish()
    
    # Return results dictionary
    results = {
        "kalpha_samples": kalpha_samples,
        "gtheta_samples": gtheta_samples,
        "acceptance_rate": acceptance_rate,
        # Other results
    }
    
    return results


# Example of how to call the function with and without wandb
if __name__ == "__main__":
    pass