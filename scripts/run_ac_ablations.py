#!/usr/bin/env python3
"""Run AC loss ablation experiments sequentially."""

import os
import subprocess
import sys
import time

EXP_DIR = "datasets/opamp_3stage_fan_smc_v7_2k_nofil/experiments"

experiments = [
    ("ac_abl1_lowac", "configs/gnn/3stage_v7_2k_nofil_ac_abl1_lowac.yaml"),
    ("ac_abl2_poolac", "configs/gnn/3stage_v7_2k_nofil_ac_abl2_poolac.yaml"),
    ("ac_abl3_detachac", "configs/gnn/3stage_v7_2k_nofil_ac_abl3_detachac.yaml"),
    ("ac_abl4_ss_gmsat", "configs/gnn/3stage_v7_2k_nofil_ac_abl4_ss_gmsat.yaml"),
]

for i, (name, config) in enumerate(experiments):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(experiments)}] Starting: {name}")
    print(f"  Config: {config}")
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
print("All AC ablations complete.")
print(f"Logs in: {EXP_DIR}/<name>/training.log")
print(f"{'='*60}")
