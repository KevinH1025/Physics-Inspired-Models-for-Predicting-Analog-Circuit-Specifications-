#!/usr/bin/env python3
"""Run multiple training configs sequentially with one command.

Usage:
    python scripts/run_ablation.py \
        configs/gnn/3stage_v7_2k_nofil_region.yaml:3stage_v7_2k_nofil_region \
        configs/gnn/3stage_v7_2k_nofil_kcl2.yaml:3stage_v7_2k_nofil_kcl2 \
        configs/gnn/3stage_v7_2k_nofil_kcl5.yaml:3stage_v7_2k_nofil_kcl5

    Format: config_path:experiment_name
    If :name is omitted, uses the config filename (without .yaml) as name.
"""

import subprocess
import sys
import time


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_ablation.py config1.yaml:name1 config2.yaml:name2 ...")
        sys.exit(1)

    runs = []
    for spec in sys.argv[1:]:
        if ':' in spec:
            config_path, name = spec.rsplit(':', 1)
        else:
            config_path = spec
            name = spec.replace('.yaml', '').split('/')[-1]
        runs.append((config_path, name))

    print(f"{'=' * 60}")
    print(f"  ABLATION: {len(runs)} runs")
    print(f"{'=' * 60}")
    for i, (cfg, name) in enumerate(runs):
        print(f"  {i+1}. {name} <- {cfg}")
    print()

    results = []
    total_start = time.time()

    for i, (config_path, name) in enumerate(runs):
        print(f"\n{'=' * 60}")
        print(f"  RUN {i+1}/{len(runs)}: {name}")
        print(f"{'=' * 60}\n")

        start = time.time()
        cmd = [sys.executable, 'scripts/train_v3.py', '--config', config_path, '--name', name]
        ret = subprocess.run(cmd)
        elapsed = time.time() - start

        status = 'OK' if ret.returncode == 0 else 'FAILED'
        results.append((name, status, elapsed))
        print(f"\n  {name}: {status} ({elapsed:.0f}s / {elapsed/60:.1f}min)")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  ABLATION COMPLETE ({total_elapsed:.0f}s / {total_elapsed/60:.1f}min)")
    print(f"{'=' * 60}")
    for name, status, elapsed in results:
        print(f"  {status:6s} | {elapsed/60:5.1f}min | {name}")


if __name__ == '__main__':
    main()
