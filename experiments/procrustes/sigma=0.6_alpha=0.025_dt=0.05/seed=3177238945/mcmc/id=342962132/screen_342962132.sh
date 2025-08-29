#!/bin/bash
# Set resource limits to control CPU usage
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_DISABLE_JIT=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Make output directory
mkdir -p sigma=0.6_alpha=0.025_dt=0.05/seed=3177238945/mcmc/id=342962132
for i in $(seq 1 3); do
  screen -md -S 342962132_chain=$i python run_mcmc.py --outputpath sigma=0.6_alpha=0.025_dt=0.05/seed=3177238945/mcmc/id=342962132 --phylopath ../data/chazot_subtree_rounded.nw --datapath sigma=0.6_alpha=0.025_dt=0.05/seed=3177238945/leaves.csv --prior_sigma_min 0.0 --prior_sigma_max 1.0 --prior_alpha_min 0.0 --prior_alpha_max 0.05 --proposal_sigma_tau 0.1 --proposal_alpha_tau 0.005 --rb 1 --obs_var 0.0001 --lambd 0.95 --dt 0.05 --n 20 --d 2 --N 3000 --super_root phylomean --use_wandb True --wandb_project "SPMS_MCMC"
done
