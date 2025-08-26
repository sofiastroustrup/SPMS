#!/bin/bash

for i in $(seq 1 3); do
  screen -md -S test python run_mcmc.py --outputpath test/runs --phylopath ../data/chazot_subtree_rounded.nw --datapath exp_1_sigma=0.5_alpha=0.025_dt=0.05/seed=3047476154/procrustes_aligned.csv --prior_sigma_min 0.0 --prior_sigma_max 2.5 --prior_alpha_min 0.0 --prior_alpha_max 0.03 --proposal_sigma_tau 0.2 --proposal_alpha_tau 0.005 --rb 2 --obs_var 0.001 --lambd 0.7 --dt 0.05 --n 20 --d 2 --N 3000 --use_wandb True --wandb_project "SPMS MCMC"
done
