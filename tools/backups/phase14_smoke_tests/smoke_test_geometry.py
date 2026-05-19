#!/usr/bin/env python3
"""
Phase 14: Geometry Smoke Run

Purpose: Short training run (1k iterations) to verify geometry training pipeline works correctly.
This is NOT a formal result - just a smoke test.

Key settings:
- C1-C7 all cameras
- ~100-300 frames subset (must cover all cameras)
- initialization.num_gaussians = 50000
- ReID/person_feature loss DISABLED
- 1k iterations
- fixed seed
- output: outputs/phase14_clean_geometry/smoke/
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 14 Geometry Smoke Run")
    parser.add_argument("--config", type=str, default="configs/apps/wildtrack_full_3dgut.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/phase14_clean_geometry/smoke")
    parser.add_argument("--n_iterations", type=int, default=1000)
    parser.add_argument("--val_frequency", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python", type=str, default="python")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 14: Geometry Smoke Run")
    print("=" * 80)
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output_dir}")
    print(f"  Iterations: {args.n_iterations}")
    print(f"  Val frequency: {args.val_frequency}")
    print(f"  Seed: {args.seed}")
    print()

    # Create a temporary Hydra override config
    # We need to disable ReID, set iterations, and control other parameters

    train_script = Path("train.py")
    if not train_script.exists():
        print(f"ERROR: train.py not found at {train_script}")
        sys.exit(1)

    # Build Hydra command with overrides
    cmd = [
        args.python, "train.py",
        # Base config
        f"--config-path={args.config.parent}",
        f"--config-name={Path(args.config).stem}",
        # Path override
        f"path=/data02/zhangrunxiang/data/Wildtrack",
        # Iterations
        f"n_iterations={args.n_iterations}",
        f"val_frequency={args.val_frequency}",
        # Output
        f"out_dir={args.output_dir}",
        # Disable ReID
        "loss.use_reid=false",
        # Initialization
        "initialization.num_gaussians=50000",
        "initialization.xyz_min=-700.0",
        "initialization.xyz_max=2100.0",
        # Seed
        f"seed={args.seed}",
        # Strategy (use default GS)
        # Dataset
        "dataset.downsample_factor=4",
        "dataset.test_split_interval=5",
        # Background
        "model.background.color=black",
        # Render
        "render=3dgut",
    ]

    print("Command:")
    print(" ".join(cmd))
    print()

    # Save command for reproducibility
    with open(output_dir / "smoke_command.txt", "w") as f:
        f.write(" ".join(cmd) + "\n")

    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False,
            text=True,
        )
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.1f}s")
        
        if result.returncode != 0:
            print(f"WARNING: Training exited with code {result.returncode}")
            generate_smoke_report(output_dir, success=False, elapsed=elapsed)
        else:
            print("SUCCESS: Training completed successfully")
            generate_smoke_report(output_dir, success=True, elapsed=elapsed)
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR: Training failed: {e}")
        generate_smoke_report(output_dir, success=False, elapsed=elapsed, error=str(e))


def generate_smoke_report(output_dir: Path, success: bool, elapsed: float, error: str = None):
    report_path = output_dir / "smoke_report.md"
    with open(report_path, "w") as f:
        f.write("# Phase 14 Geometry Smoke Report\n\n")
        f.write(f"## Status: {'PASS' if success else 'FAIL'}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- Iterations: 1000\n")
        f.write(f"- Initialization: 50000 Gaussians\n")
        f.write(f"- ReID loss: DISABLED\n")
        f.write(f"- Seed: 42\n\n")
        
        f.write(f"## Result\n\n")
        f.write(f"- Training time: {elapsed:.1f}s\n")
        if error:
            f.write(f"- Error: {error}\n")
        
        f.write("\n## Next Step\n\n")
        if success:
            f.write("Smoke test PASSED. Proceed to formal geometry run (30k iterations).\n")
        else:
            f.write("Smoke test FAILED. Fix issues before proceeding.\n")


if __name__ == "__main__":
    main()
