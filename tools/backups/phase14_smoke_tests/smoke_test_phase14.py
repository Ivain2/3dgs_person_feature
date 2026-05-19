#!/usr/bin/env python3
"""
Phase 14: Unified Smoke Runner for Reset Density Ablation

Runs a short geometry training (3k-5k iterations) with controlled configuration
for comparing different reset_density settings.

Usage:
  python tools/phase14_smoke_run.py \
    --output_dir outputs/phase14_clean_geometry/reset_density_ablation/smoke_old_config \
    --n_iterations 5000 \
    --reset_density_frequency 3000 \
    --reset_density_new_max 0.01 \
    --seed 42
"""

import os
import sys
import json
import time
import subprocess
import signal
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 14 Smoke Run")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_iterations", type=int, default=5000)
    parser.add_argument("--val_frequency", type=int, default=1000)
    parser.add_argument("--reset_density_frequency", type=int, default=3000)
    parser.add_argument("--reset_density_new_max", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/apps/wildtrack_full_3dgut.yaml")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Phase 14 Smoke Run")
    print("=" * 80)
    print(f"  Output: {args.output_dir}")
    print(f"  Iterations: {args.n_iterations}")
    print(f"  Reset density frequency: {args.reset_density_frequency}")
    print(f"  Reset density new_max_density: {args.reset_density_new_max}")
    print(f"  Seed: {args.seed}")
    print()

    # Build Hydra command with overrides
    # train.py uses config_path="configs", so config_name needs "apps/" prefix
    # and should NOT include .yaml extension
    config_stem = Path(args.config).stem
    config_dir = Path(args.config).parent.name
    config_name = f"{config_dir}/{config_stem}"  # e.g., "apps/wildtrack_full_3dgut"
    cmd = [
        sys.executable, "train.py",
        f"--config-name={config_name}",
        f"path=/data02/zhangrunxiang/data/Wildtrack",
        f"n_iterations={args.n_iterations}",
        f"val_frequency={args.val_frequency}",
        f"out_dir={args.output_dir}",
        "loss.use_reid=false",
        "initialization.num_gaussians=50000",
        "initialization.xyz_min=-700.0",
        "initialization.xyz_max=2100.0",
        "dataset.downsample_factor=4",
        "dataset.test_split_interval=5",
        "model.background.color=black",
        "render=3dgut",
        # Reset density overrides
        f"strategy.reset_density.frequency={args.reset_density_frequency}",
        f"strategy.reset_density.new_max_density={args.reset_density_new_max}",
        # Other strategy settings (keep defaults)
        "strategy.densify.frequency=300",
        "strategy.densify.start_iteration=500",
        "strategy.densify.end_iteration=15000",
        "strategy.densify.clone_grad_threshold=0.0002",
        "strategy.densify.split_grad_threshold=0.0002",
        "strategy.prune.frequency=100",
        "strategy.prune.start_iteration=500",
        "strategy.prune.end_iteration=15000",
        "strategy.prune.density_threshold=0.005",
        # Learning rates
        "optimizer.params.positions.lr=0.00016",
        "optimizer.params.density.lr=0.05",
        "optimizer.params.features_albedo.lr=0.0025",
        "optimizer.params.rotation.lr=0.001",
        "optimizer.params.scale.lr=0.005",
        # Loss
        "loss.lambda_l1=0.8",
        "loss.lambda_ssim=0.2",
        # Progressive training (keep default)
        "model.progressive_training.init_n_features=0",
        "model.progressive_training.max_n_features=3",
        "model.progressive_training.increase_frequency=1000",
    ]

    print("Command:")
    print(" ".join(cmd))
    print()

    # Save command
    with open(output_dir / "smoke_command.txt", "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.write(f"reset_density_frequency: {args.reset_density_frequency}\n")
        f.write(f"reset_density_new_max_density: {args.reset_density_new_max}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"n_iterations: {args.n_iterations}\n")

    # Run training with timeout (30 minutes max)
    start_time = time.time()
    timeout_sec = 1800
    
    log_file = open(output_dir / "train.log", "w")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        # Stream output to log file and stdout
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            
            # Check timeout
            if time.time() - start_time > timeout_sec:
                process.kill()
                print(f"\nTIMEOUT after {timeout_sec}s")
                break
        
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode != 0:
            print(f"\nTraining exited with code {process.returncode}")
        else:
            print(f"\nTraining completed successfully in {elapsed:.1f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR: Training failed: {e}")
        log_file.write(f"ERROR: {e}\n")
    finally:
        log_file.close()
        generate_smoke_report(output_dir, elapsed)


def generate_smoke_report(output_dir: Path, elapsed: float):
    report_path = output_dir / "smoke_report.md"
    with open(report_path, "w") as f:
        f.write("# Phase 14 Smoke Report\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Training time: {elapsed:.1f}s\n")
        f.write(f"- See `smoke_command.txt` for full configuration\n\n")
        
        f.write("## Status\n\n")
        f.write("See `train.log` for full training output.\n")


if __name__ == "__main__":
    main()
