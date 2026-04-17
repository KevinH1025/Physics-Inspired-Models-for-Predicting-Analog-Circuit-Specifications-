#!/bin/bash
# Ablation Study: How does adding physics improve AC prediction?
#
# Baseline (already done): V + AC (VN readout) — see ac_readout_test/vn.log
#
# Level 1: V + AC + SS         (add gm/gds heads)
# Level 2: V + AC + SS + Curr  (add current head)
# Level 3: V + AC + SS + Curr + KCL  (add KCL loss, weight=2.0)

set -e

DATASET_DIR="datasets/opamp_5k_ss_v1"
LOG_DIR="${DATASET_DIR}/ablation"
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "  ABLATION: Progressive Physics → AC Performance"
echo "=========================================================="
echo ""
echo "  Baseline: V + AC (VN) — already done"
echo "  Level 1:  V + AC + SS"
echo "  Level 2:  V + AC + SS + Current"
echo "  Level 3:  V + AC + SS + Current + KCL (2.0)"
echo ""
echo "  Output: ${LOG_DIR}/"
echo "=========================================================="
echo ""

# --- Experiment 1: V + AC + SS ---
echo "[1/3] Running V + AC + SS (add gm/gds heads)..."
echo ""
python scripts/train_v3.py --config configs/gnn/ablation_1_v_ac_ss.yaml 2>&1 | tee "${LOG_DIR}/1_v_ac_ss.log"

cp "${DATASET_DIR}/best_model.pt" "${LOG_DIR}/best_model_1_v_ac_ss.pt" 2>/dev/null || true
cp "${DATASET_DIR}/training_curve.png" "${LOG_DIR}/training_curve_1_v_ac_ss.png" 2>/dev/null || true
cp "${DATASET_DIR}/training.log" "${LOG_DIR}/training_1_v_ac_ss.log" 2>/dev/null || true
echo ""
echo "=== Level 1 done ==="
echo ""

# --- Experiment 2: V + AC + SS + Current ---
echo "[2/3] Running V + AC + SS + Current..."
echo ""
python scripts/train_v3.py --config configs/gnn/ablation_2_v_ac_ss_current.yaml 2>&1 | tee "${LOG_DIR}/2_v_ac_ss_current.log"

cp "${DATASET_DIR}/best_model.pt" "${LOG_DIR}/best_model_2_v_ac_ss_current.pt" 2>/dev/null || true
cp "${DATASET_DIR}/training_curve.png" "${LOG_DIR}/training_curve_2_v_ac_ss_current.png" 2>/dev/null || true
cp "${DATASET_DIR}/training.log" "${LOG_DIR}/training_2_v_ac_ss_current.log" 2>/dev/null || true
echo ""
echo "=== Level 2 done ==="
echo ""

# --- Experiment 3: V + AC + SS + Current + KCL ---
echo "[3/3] Running V + AC + SS + Current + KCL (2.0)..."
echo ""
python scripts/train_v3.py --config configs/gnn/ablation_3_v_ac_ss_current_kcl.yaml 2>&1 | tee "${LOG_DIR}/3_v_ac_ss_current_kcl.log"

cp "${DATASET_DIR}/best_model.pt" "${LOG_DIR}/best_model_3_v_ac_ss_current_kcl.pt" 2>/dev/null || true
cp "${DATASET_DIR}/training_curve.png" "${LOG_DIR}/training_curve_3_v_ac_ss_current_kcl.png" 2>/dev/null || true
cp "${DATASET_DIR}/training.log" "${LOG_DIR}/training_3_v_ac_ss_current_kcl.log" 2>/dev/null || true
echo ""
echo "=== Level 3 done ==="
echo ""

# --- Summary ---
echo "=========================================================="
echo "  ABLATION SUMMARY — AC Performance"
echo "=========================================================="

# Baseline from previous VN readout test
echo ""
echo "--- Baseline: V + AC (VN) ---"
if [ -f "${DATASET_DIR}/ac_readout_test/vn.log" ]; then
    grep -E "(AC Evaluation|UGBW|PM |AM |Voltage.*MAE:|@80=.*@10=)" "${DATASET_DIR}/ac_readout_test/vn.log" | tail -10
fi

for i in 1 2 3; do
    case $i in
        1) name="V + AC + SS" ; logfile="${LOG_DIR}/1_v_ac_ss.log" ;;
        2) name="V + AC + SS + Current" ; logfile="${LOG_DIR}/2_v_ac_ss_current.log" ;;
        3) name="V + AC + SS + Current + KCL" ; logfile="${LOG_DIR}/3_v_ac_ss_current_kcl.log" ;;
    esac

    echo ""
    echo "--- Level ${i}: ${name} ---"
    grep -E "(AC Evaluation|UGBW|PM |AM |Voltage.*MAE:|@80=.*@10=|SS Evaluation|gm |gds )" "$logfile" | tail -15
done

echo ""
echo "=========================================================="
echo "Full logs: ${LOG_DIR}/"
echo "=========================================================="
