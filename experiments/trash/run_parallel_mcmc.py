import os
import sys
import numpy as np
from ete3 import Tree
import jax
import jax.numpy as jnp
import pandas as pd
import copy
import pickle
from multiprocessing import Pool
from functools import partial

# Import your modules
from bridge_sampling.BFFG import backward_filter, forward_guide, forward_guide_edge, get_logpsi
from bridge_sampling.setup_SDEs import dtsdWsT, dWs
from bridge_sampling.noise_kernel import Q12
from bridge_sampling.helper_functions import *
from mcmc import *

# Define fully picklable functions (outside of any local scope)
def zero_drift(t, x, theta):
    """Drift function that returns zeros."""
    return jnp.zeros(x.shape[0])

def q12_diffusion(t, x, theta):
    """Diffusion function based on Q12."""
    return Q12(x, theta)

def run_mcmc_chain(params):
    """
    Run a single MCMC chain with given parameters
    
    Parameters:
    -----------
    params : tuple
        Tuple of (seed, output_path, **kwargs)
    
    Returns:
    --------
    tuple
        (seed, results)
    """
    seed, output_path, iterations, all_params = params
    
    # Create chain-specific output path
    chain_output = f"{output_path}/chain_{seed}"
    os.makedirs(chain_output, exist_ok=True)
    
    print(f"Starting chain with seed {seed}")
    
    # Create a deep copy of the tree
    tree_copy = copy.deepcopy(all_params['tree'])
    
    # Run MCMC
    results = metropolis_hastings(
        N=iterations,
        dt=all_params['dt'],
        lambd=all_params['lambd'],
        obs_var=all_params['obs_var'],
        rb=all_params['rb'],
        xs=all_params['xs'],
        drift_term=zero_drift, 
        diffusion_term=q12_diffusion,
        prior_sigma=all_params['prior_sigma'],
        prior_alpha=all_params['prior_alpha'],
        proposal_sigma=all_params['proposal_sigma'],
        proposal_alpha=all_params['proposal_alpha'],
        tree=tree_copy,
        leaves=all_params['leaves'],
        n=all_params['n'],
        d=all_params['d'],
        outputpath=chain_output,
        seed_mcmc=seed,
        use_wandb=False
    )
    
    # Save results to file
    with open(f"{chain_output}/results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return seed, results

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run parallel MCMC chains")
    parser.add_argument('--n_chains', type=int, default=4, help="Number of chains to run")
    parser.add_argument('--n_iterations', type=int, default=50, help="Number of iterations per chain")
    parser.add_argument('--output', type=str, default="mcmc_parallel", help="Output directory")
    parser.add_argument('--tree_path', type=str, required=True, help="Path to tree file")
    parser.add_argument('--leaves_path', type=str, required=True, help="Path to leaves CSV file")
    parser.add_argument('--n_cores', type=int, default=None, help="Number of CPU cores to use")
    args = parser.parse_args()
    
    # Set up parameters
    n = 20
    d = 2
    
    # Load data
    leaves = jnp.array(pd.read_csv(args.leaves_path, delimiter=',', header=0, index_col=0))
    tree = Tree(args.tree_path)
    
    # Calculate super root
    vcv = get_tree_covariance(args.tree_path)
    w = np.linalg.solve(vcv, np.ones(leaves.shape[0])) 
    w_norm = w / np.sum(w)
    super_root = w_norm.T @ leaves
    
    # Define distributions
    proposal_sigma = MirroredGaussian(tau=0.001, minval=0, maxval=10)
    proposal_alpha = MirroredGaussian(tau=0.001, minval=0, maxval=10)
    prior_sigma = Uniform(minval=0.7, maxval=1.3)
    prior_alpha = Uniform(minval=0.0005, maxval=0.03)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Prepare common parameters dictionary
    all_params = {
        'dt': 0.1,
        'lambd': 0.95,
        'obs_var': 0.001,
        'rb': 1.0,
        'xs': super_root,
        'prior_sigma': prior_sigma,
        'prior_alpha': prior_alpha,
        'proposal_sigma': proposal_sigma,
        'proposal_alpha': proposal_alpha,
        'tree': tree,
        'leaves': leaves,
        'n': n,
        'd': d
    }
    
    # Generate seeds for chains
    np.random.seed(42)  # For reproducibility
    seeds = np.random.randint(0, 1_000_000, size=args.n_chains)
    
    # Create arguments for each chain
    chain_args = [(seed, args.output, args.n_iterations, all_params) for seed in seeds]
    
    # Determine number of cores
    if args.n_cores is None:
        args.n_cores = min(args.n_chains, os.cpu_count())
    
    # Run chains in parallel with proper initialization and cleanup
    print(f"Running {args.n_chains} chains on {args.n_cores} cores")
    
    # Use 'spawn' method for better compatibility
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.n_cores) as pool:
        all_results = pool.map(run_mcmc_chain, chain_args)
    
    # Save combined results
    with open(f"{args.output}/combined_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"All chains complete. Results saved to {args.output}/")

if __name__ == "__main__":
    import multiprocessing as mp
    # Make sure we're using spawn method
    mp.set_start_method('spawn', force=True)
    main()