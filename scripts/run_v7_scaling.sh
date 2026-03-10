#!/bin/bash
# Train baseline on 10k and 20k v7 datasets
set -e

echo "=========================================="
echo "  Train baseline on 10k"
echo "=========================================="
python scripts/train_v3.py --config configs/gnn/3stage_v7_10k_baseline.yaml

echo "=========================================="
echo "  Train baseline on 20k"
echo "=========================================="
python scripts/train_v3.py --config configs/gnn/3stage_v7_20k_baseline.yaml

echo "=========================================="
echo "  Results comparison:"
echo "=========================================="
echo "--- 5k ---"
tail -12 datasets/opamp_3stage_fan_smc_v7/training.log
echo ""
echo "--- 10k ---"
tail -12 datasets/opamp_3stage_fan_smc_v7_10k/training.log
echo ""
echo "--- 20k ---"
tail -12 datasets/opamp_3stage_fan_smc_v7_20k/training.log
