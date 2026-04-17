#!/usr/bin/env python3
"""Run all physics loss ablation experiments sequentially."""

import os
import subprocess
import sys
import time

EXP_DIR = "datasets/opamp_3stage_fan_smc_v7_2k_nofil/experiments"

experiments = [
    ("abl3_kcl5_ss_ac_gmsat_v3", "configs/gnn/3stage_v7_2k_nofil_abl3_kcl5_ss_ac_gmsat.yaml"),
    ("abl4_kcl5_ss_ac_gmsat_trieq1_v3", "configs/gnn/3stage_v7_2k_nofil_abl4_kcl5_ss_ac_gmsat_trieq1.yaml"),
    ("abl5_kcl5_ss_ac_gmsat_trieq12_v3", "configs/gnn/3stage_v7_2k_nofil_abl5_kcl5_ss_ac_gmsat_trieq12.yaml"),
    ("abl6_kcl5_ss_ac_gmsat_trieq12_cutoff_v3", "configs/gnn/3stage_v7_2k_nofil_abl6_kcl5_ss_ac_gmsat_trieq12_cutoff.yaml"),
]

for i, (name, config) in enumerate(experiments):
    log_file = os.path.join(EXP_DIR, name, "training.log")
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(experiments)}] Starting: {name}")
    print(f"  Config: {config}")
    print(f"  Log:    {log_file}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "scripts/train_v3.py", "--config", config, "--fast", "--name", name],
        cwd="/home/kevin/projects/thesis",
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n[{name}] {status} in {elapsed/60:.1f} min")

print(f"\n{'='*60}")
print("All ablations complete.")
print(f"Logs in: {EXP_DIR}/<name>/training.log")
print(f"{'='*60}")
