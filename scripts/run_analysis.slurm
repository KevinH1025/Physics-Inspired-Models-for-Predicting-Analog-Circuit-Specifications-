#!/bin/bash
#SBATCH --job-name=ss_analysis
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

CHECKPOINT=$1
DATASET=$2

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH
export PYTHONPATH=/dss/dsshome1/03/go49jit2/thesis:$PYTHONPATH

echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET"
echo "Start: $(date)"

srun python scripts/analyze_ss_errors.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET"

echo "Done: $(date)"
