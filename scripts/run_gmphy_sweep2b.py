#!/usr/bin/env python3
"""Re-run gmphy_w005_v2 (3-stage) + 2stage_gmphy."""

import subprocess
import sys
import time

experiments = [
    ("gmphy_w005_v2", "configs/gnn/3stage_v7_2k_nofil_gmphy_w005_v2.yaml"),
    ("2stage_gmphy", "configs/gnn/2stage_5k_ss_gmphy.yaml"),
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
print("All experiments complete.")
print(f"{'='*60}")
