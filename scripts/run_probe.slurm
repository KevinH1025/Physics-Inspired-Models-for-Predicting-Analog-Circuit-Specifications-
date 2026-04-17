#!/bin/bash
#SBATCH --job-name=probe_emb
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/probe_%j.out
#SBATCH --error=slurm_logs/probe_%j.err

source ~/.bashrc
conda activate thesis 2>/dev/null || true

CHECKPOINT=$1
DATASET=$2

echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET"
echo "Start: $(date)"

python3 scripts/probe_embeddings.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET"

echo "Done: $(date)"
