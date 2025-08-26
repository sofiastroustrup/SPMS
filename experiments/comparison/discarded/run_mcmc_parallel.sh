#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export JAX_PLATFORM_NAME=cpu

for dataset in exp_1_sigma=0.5_alpha=0.025_dt=0.05/seed=*; do
  if [ -f "$dataset/procrustes_aligned.csv" ]; then
    SEED=$(python -c "import numpy as np; print(np.random.randint(0, 1000000000))")
    OUTPUT="$dataset/mcmc_seed=${SEED}_N=3000"
    mkdir -p "$OUTPUT"
    
    # Run 3 independent chains with no input from seq
    parallel --memsuspend 5G -j2 -N0 \
      'CHAIN_SEED=$(python -c "import numpy as np; print(np.random.randint(0, 1000000000))") && \
       echo Running chain with seed $CHAIN_SEED && \
       python run_mcmc.py \
         --seed_mcmc $CHAIN_SEED \
         --outputpath '"$OUTPUT"' \
         --phylopath ../data/chazot_subtree_rounded.nw \
         --datapath '"$dataset"'/procrustes_aligned.csv \
         --dt 0.05 --lambd 0.7 \
         --obs_var 0.001 --rb 2 --N 3000 \
         --prior_sigma_min 0.0 --prior_sigma_max 2.5 \
         --prior_alpha_min 0.0 --prior_alpha_max 0.03 \
         --proposal_sigma_tau 0.2 --proposal_alpha_tau 0.005 \
         --use_wandb True' \
    ::: $(seq 1 3)
  fi
done