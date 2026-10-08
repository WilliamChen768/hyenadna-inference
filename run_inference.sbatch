#!/bin/bash
#SBATCH -J hyenadna-infer
#SBATCH -A coc                 # CHANGE if your Slurm account differs
#SBATCH -p coc-gpu             # CHANGE if your GPU partition differs
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --time=00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"
ENV_PREFIX="${ENV_PREFIX:-$SCRATCH_DIR/.conda/envs/hyenadna_env}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/.hf_cache}"

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

mkdir -p outputs

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Env:     $ENV_PREFIX"
nvidia-smi
pip freeze > "logs/pip_freeze_${SLURM_JOB_ID}.txt"

python infer.py

echo "Finished: $(date)"