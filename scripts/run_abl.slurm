#!/bin/bash
#SBATCH --job-name=gnn_abl
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

CONFIG=$1
NAME=$2
shift 2
EXTRA_ARGS="$@"

if [ -z "$CONFIG" ] || [ -z "$NAME" ]; then
    echo "Usage: sbatch scripts/run_abl.slurm <config> <name> [extra args...]"
    exit 1
fi

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG | Name: $NAME | Extra: $EXTRA_ARGS"
echo "Start: $(date)"

srun python scripts/train_v3.py --config "$CONFIG" --name "$NAME" $EXTRA_ARGS

echo "End: $(date)"
