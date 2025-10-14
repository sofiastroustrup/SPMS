#!/bin/bash

# Set environment variables to control JAX
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export JAX_PLATFORM_NAME=cpu
# Add these important thread control variables
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_DISABLE_JIT=0
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""
# Limit threads per process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Make output directory
mkdir -p exp_1_sigma=0.5_alpha=0.025_dt=0.05/seed=3047476154/mcmc/parallel_11144201

# Run chains in parallel with explicit chain numbers
parallel --memsuspend 5G -j2 --delay 5 --env XLA_FLAGS,JAX_DISABLE_JIT,TF_CPP_MIN_LOG_LEVEL,OMP_NUM_THREADS,XLA_PYTHON_CLIENT_MEM_FRACTION,JAX_PLATFORM_NAME '
  CHAIN_NUM={}
  CHAIN_SEED=$(python -c "import numpy as np; print(np.random.randint(0, 1000000000))")
  echo "Running chain $CHAIN_NUM with seed $CHAIN_SEED"
  python run_mcmc.py \
    --seed_mcmc "$CHAIN_SEED" \
    --outputpath "exp_1_sigma=0.5_alpha=0.025_dt=0.05/seed=3047476154/mcmc/parallel_11144201" \
    --phylopath "../data/chazot_subtree_rounded.nw" \
    --datapath "exp_1_sigma=0.5_alpha=0.025_dt=0.05/seed=3047476154/procrustes_aligned.csv" \
    --dt 0.05 \
    --lambd 0.7 \
    --obs_var 0.001 \
    --rb 2 \
    --n 20 \
    --d 2 \
    --N 30 \
    --prior_sigma_min 0.0 \
    --prior_sigma_max 2.5 \
    --prior_alpha_min 0.0 \
    --prior_alpha_max 0.03 \
    --proposal_sigma_tau 0.2 \
    --proposal_alpha_tau 0.005 \
    --use_wandb True \
    --wandb_project "SPMS_MCMC"
' ::: $(seq 1 3)
