#!/bin/bash
# Make output directory
mkdir -p sigma=0.6_alpha=0.025_dt=0.05/seed=4074397383/mcmc_procrustes/id=211875092
for i in $(seq 1 3); do
  screen -md -S 211875092_chain=$i python run_mcmc.py --outputpath sigma=0.6_alpha=0.025_dt=0.05/seed=4074397383/mcmc_procrustes/id=211875092 --phylopath ../data/chazot_subtree_rounded.nw --datapath sigma=0.6_alpha=0.025_dt=0.05/seed=4074397383/procrustes_aligned.csv --prior_sigma_min 0.0 --prior_sigma_max 1.0 --prior_alpha_min 0.0 --prior_alpha_max 0.05 --proposal_sigma_tau 0.1 --proposal_alpha_tau 0.005 --rb 1 --obs_var 0.0001 --lambd 0.95 --dt 0.05 --n 20 --d 2 --N 3000 --super_root phylomean --use_wandb True --wandb_project "SPMS_MCMC"
done
