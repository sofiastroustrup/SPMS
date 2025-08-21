import jax
import jax.numpy as jnp
from tqdm import tqdm 
import pandas as pd
import wandb
import argparse
import scipy
import pickle 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.backends.backend_pdf as backend_pdf
import os

from bridge_sampling.BFFG import backward_filter, forward_guide, forward_guide_edge, get_logpsi
from bridge_sampling.setup_SDEs import Stratonovich_to_Ito, dtsdWsT, dWs
from bridge_sampling.noise_kernel import Q12
from bridge_sampling.helper_functions import *

from mcmc import *

# Set up argument parser
parser = argparse.ArgumentParser(description='Run MCMC simulation with customizable parameters')

# Path arguments
parser.add_argument('--outputpath', type=str, default="seed=10_sigma=0.5_alpha=0.05_dt=0.05/mcmc",
                    help='Output directory path')
parser.add_argument('--phylopath', type=str, default="../data/chazot_subtree_rounded.nw",
                    help='Path to phylogenetic tree file')
parser.add_argument('--datapath', type=str, default="seed=10_sigma=0.5_alpha=0.05_dt=0.05",
                    help='Path to input data directory')

# Prior distribution parameters
parser.add_argument('--prior_sigma_min', type=float, default=0.7, 
                    help='Minimum value for sigma prior')
parser.add_argument('--prior_sigma_max', type=float, default=1.3, 
                    help='Maximum value for sigma prior')
parser.add_argument('--prior_alpha_min', type=float, default=0.0005, 
                    help='Minimum value for alpha prior')
parser.add_argument('--prior_alpha_max', type=float, default=0.03, 
                    help='Maximum value for alpha prior')

# Proposal distribution parameters
parser.add_argument('--proposal_sigma_tau', type=float, default=0.001, 
                    help='Tau for sigma proposal')
parser.add_argument('--proposal_alpha_tau', type=float, default=0.001, 
                    help='Tau for alpha proposal')

# Model parameters
parser.add_argument('--rb', type=float, default=1.0, 
                    help='Rb parameter')
parser.add_argument('--obs_var', type=float, default=0.001, 
                    help='Observation variance')
parser.add_argument('--lambd', type=float, default=0.95, 
                    help='Lambda parameter')
parser.add_argument('--dt', type=float, default=0.1, 
                    help='Time step')
parser.add_argument('--n', type=int, default=20, 
                    help='Number of landmarks')
parser.add_argument('--d', type=int, default=2, 
                    help='Dimension of landmarks')
parser.add_argument('--N', type=int, default=3000, 
                    help='Number of MCMC iterations')

# Other options
parser.add_argument('--seed_mcmc', type=int, default=None, 
                    help='Random seed for MCMC (default: random)')
parser.add_argument('--use_wandb', type=bool,
                    help='Use Weights & Biases for logging')
parser.add_argument('--wandb_project', type=str, default="SPMS_MCMC",
                    help='Weights & Biases project name')
parser.add_argument('--stratonovich', type=int, default=1,
                    help='Use Stratonovich (1) or Ito (0) formulation')

# Parse arguments
args = parser.parse_args()

# Use parsed arguments
n = args.n
d = args.d
outputpath = args.outputpath

# Make sure the output directory exists
os.makedirs(outputpath, exist_ok=True)

# Load tree and leaves
leaves_path = args.datapath#os.path.join(args.datapath, 'leaves.csv')
leaves = jnp.array(pd.read_csv(leaves_path, delimiter=',', header=None, index_col=None))
tree = Tree(args.phylopath)
print(leaves.shape)

# Choose super root
vcv = get_tree_covariance(args.phylopath)
print(vcv)

# Step 1: Solve the linear system vcv @ x = ones
w = np.linalg.solve(vcv, np.ones(leaves.shape[0])) 
# Step 2: Normalize the weights
w_norm = w / np.sum(w)
# Step 3: Compute weighted average of leaves
super_root = w_norm.T @ leaves

# Define prior and proposal distributions  
proposal_sigma = MirroredGaussian(tau=args.proposal_sigma_tau, minval=0, maxval=10)
proposal_alpha = MirroredGaussian(tau=args.proposal_alpha_tau, minval=0, maxval=10)
prior_sigma = Uniform(minval=args.prior_sigma_min, maxval=args.prior_sigma_max)
prior_alpha = Uniform(minval=args.prior_alpha_min, maxval=args.prior_alpha_max)

# Define stochastic process
if args.stratonovich == 1:
    b, sigma, _ = Stratonovich_to_Ito(
        lambda t, x, theta: jnp.zeros(n*d),
        lambda t, x, theta: Q12(x, theta)
    )
else:
    b = lambda t, x, theta: jnp.zeros(n*d)
    sigma = lambda t, x, theta: Q12(x, theta)
    
# Set random seed
if args.seed_mcmc is None:
    seed = np.random.randint(0, 1000000)
else:
    seed = args.seed_mcmc

# Run MCMC
results = metropolis_hastings(
    N=args.N,
    dt=args.dt,
    lambd=args.lambd,
    obs_var=args.obs_var,
    rb=args.rb,
    xs=super_root,
    drift_term=b, 
    diffusion_term=sigma,
    prior_sigma=prior_sigma,
    prior_alpha=prior_alpha,
    proposal_sigma=proposal_sigma,
    proposal_alpha=proposal_alpha,
    tree=tree,
    leaves=leaves,
    n=n, 
    d=d,
    outputpath=outputpath,
    seed_mcmc=seed,
    use_wandb=args.use_wandb,
    wandb_project=args.wandb_project
)



# Save results
results_path = os.path.join(outputpath, f"results_chain={seed}.pkl")
with open(results_path, "wb") as f:
    pickle.dump(results, f)

print(f"Results saved to {results_path}")



'''import jax
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

from mcmc import *

# change such that I read from stdin 
outputpath = "seed=10_sigma=0.5_alpha=0.05_dt=0.05/mcmc" # should be the same as used in the simulation
# phylopath 
# datapath 
# params prior 
# params proposal
# rb, obs_var, lambda, dt

n = 20
d = 2

# load tree 
leaves = jnp.array(pd.read_csv('seed=10_sigma=0.5_alpha=0.05_dt=0.05/leaves.csv', delimiter=',', header=0, index_col=0))#np.genfromtxt('comparison/seed=10_sigma=0.5_alpha=0.05_dt=0.1/leaves.csv', delimiter=','))
tree = Tree("../data/chazot_subtree_rounded.nw")
print(leaves.shape)

# choose super root
vcv = get_tree_covariance("../data/chazot_subtree_rounded.nw")
print(vcv)
#vcv = np.genfromtxt(datapath + '/phylogeny_vcv.csv', delimiter=' ')
#super_root = 1/(np.ones(leaves.shape[0]).T@np.linalg.inv(vcv)@np.ones(leaves.shape[0]))*np.ones(leaves.shape[0]).T@np.linalg.inv(vcv)@leaves
# Step 1: Solve the linear system vcv @ x = ones
w = np.linalg.solve(vcv, np.ones(leaves.shape[0])) 
# Step 2: Normalize the weights
w_norm = w / np.sum(w)
# Step 3: Compute weighted average of leaves
super_root = w_norm.T @ leaves

# define prior and proposal distributions  
proposal_sigma = MirroredGaussian(tau=0.001, minval=0, maxval=10)
proposal_alpha = MirroredGaussian(tau=0.001, minval=0, maxval=10)
prior_sigma = Uniform(minval=0.7, maxval=1.3)
prior_alpha = Uniform(minval=0.0005, maxval=0.03)

# define stochastic process 
# define drift and diffusion for process of interest 
sti=1
if sti ==1:
    b,sigma,_ = Stratonovich_to_Ito(lambda t,x,theta: jnp.zeros(n*d),
                                lambda t,x,theta: Q12(x,theta))
else:
    b = lambda t,x,theta: jnp.zeros(n*d)
    sigma = lambda t,x,theta: Q12(x,theta)
    
seed = np.random.randint(0, 1000000)


# run mcmc

results = metropolis_hastings(
    N=3000,
    dt=0.1, # should pick the same as used in the simulation
    lambd=0.95,
    obs_var=0.001,
    rb=1.0,
    xs=super_root,
    drift_term=b, 
    diffusion_term=sigma,
    prior_sigma=prior_sigma,
    prior_alpha=prior_alpha,
    proposal_sigma=proposal_sigma,
    proposal_alpha=proposal_alpha,
    tree=tree,
    leaves = leaves,
    n=n, 
    d=d,
    outputpath=outputpath,
    seed_mcmc = seed,
    use_wandb=False,
    wandb_project="SPMS_MCMC")


# save results
with open(f"{outputpath}/results_chain={seed}.pkl", "wb") as f:
    pickle.dump(results, f)'''