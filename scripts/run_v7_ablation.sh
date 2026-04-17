#!/bin/bash
# V7 5k Fixed — Ablation Study
# Runs sequentially: baseline → +KCL → +SS → +AC → +gm_physics → +triode_physics
#
# Baseline (abl0) already trained: 9.54mV

set -e

DATASET="datasets/opamp_3stage_fan_smc_v7_5k_fixed"

echo "=============================================="
echo "  V7 5k Fixed — Ablation Study"
echo "=============================================="

# Abl 1: + KCL 2.0 (DONE: 10.03mV)

# Abl 2: + KCL + SS
echo ""
echo ">>> Ablation 2: + SS head"
python scripts/train_v3.py \
    --config configs/gnn/3stage_v7_abl2_kcl_ss.yaml \
    --name 3stage_v7_abl2_kcl_ss

# Abl 3: + KCL + SS + AC
echo ""
echo ">>> Ablation 3: + AC head"
python scripts/train_v3.py \
    --config configs/gnn/3stage_v7_abl3_kcl_ss_ac.yaml \
    --name 3stage_v7_abl3_kcl_ss_ac

# Abl 4: + KCL + SS + AC + gm sat physics
echo ""
echo ">>> Ablation 4: + gm saturation physics"
python scripts/train_v3.py \
    --config configs/gnn/3stage_v7_abl4_kcl_ss_ac_gmphy.yaml \
    --name 3stage_v7_abl4_kcl_ss_ac_gmphy

# Abl 5: + KCL + SS + AC + gm sat physics + triode physics
echo ""
echo ">>> Ablation 5: + triode physics"
python scripts/train_v3.py \
    --config configs/gnn/3stage_v7_abl5_kcl_ss_ac_gmphy_triphy.yaml \
    --name 3stage_v7_abl5_kcl_ss_ac_gmphy_triphy

echo ""
echo "=============================================="
echo "  ABLATION COMPLETE — Summary"
echo "=============================================="
echo ""
echo "Results in: ${DATASET}/experiments/"
echo ""
for exp in 3stage_v7_5k_fixed_baseline 3stage_v7_abl1_kcl 3stage_v7_abl2_kcl_ss 3stage_v7_abl3_kcl_ss_ac 3stage_v7_abl4_kcl_ss_ac_gmphy 3stage_v7_abl5_kcl_ss_ac_gmphy_triphy; do
    log="${DATASET}/experiments/${exp}/training.log"
    if [ -f "$log" ]; then
        mae=$(grep "Best Val MAE" "$log" | tail -1)
        echo "  ${exp}: ${mae}"
    fi
done
