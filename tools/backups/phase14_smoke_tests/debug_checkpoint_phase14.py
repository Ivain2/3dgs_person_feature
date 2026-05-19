#!/usr/bin/env python3
"""
Phase 14: Debug checkpoint analysis to understand why training collapsed.
"""

import torch
import numpy as np
from pathlib import Path

def analyze_checkpoint(ckpt_path: str):
    """Analyze a checkpoint to understand density distribution."""
    print(f"\n{'='*60}")
    print(f"Analyzing checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    # Load checkpoint
    if ckpt_path.endswith('.pt'):
        ckpt = torch.load(ckpt_path, map_location='cpu')
    else:
        # Try to find ckpt in directory
        ckpt_files = list(Path(ckpt_path).glob("*.pt"))
        if ckpt_files:
            ckpt = torch.load(str(ckpt_files[0]), map_location='cpu')
        else:
            print(f"No checkpoint found in {ckpt_path}")
            return
    
    # Analyze density
    density = ckpt.get('density')
    if density is not None:
        if hasattr(density, 'data'):
            density_data = density.data
        else:
            density_data = density
        
        print(f"\nDensity Analysis:")
        print(f"  Shape: {density_data.shape}")
        print(f"  Device: {density_data.device}")
        print(f"  Mean: {density_data.mean().item():.6f}")
        print(f"  Std: {density_data.std().item():.6f}")
        print(f"  Min: {density_data.min().item():.6f}")
        print(f"  Max: {density_data.max().item():.6f}")
        print(f"  Median: {density_data.median().item():.6f}")
        
        # Count density distribution
        density_np = density_data.cpu().numpy()
        hist, bin_edges = np.histogram(density_np, bins=20)
        print(f"\n  Density Distribution:")
        for i, (count, edge) in enumerate(zip(hist, bin_edges)):
            print(f"    [{edge:.4f}, {bin_edges[i+1]:.4f}): {count} ({count/len(density_np)*100:.1f}%)")
        
        # Check how many are below prune threshold
        prune_threshold = 0.005
        below_prune = (density_np < prune_threshold).sum()
        print(f"\n  Below prune threshold ({prune_threshold}): {below_prune} ({below_prune/len(density_np)*100:.1f}%)")
        
        # Check how many are in various ranges
        ranges = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        print(f"\n  Density Ranges:")
        prev = 0
        for r in ranges:
            count = ((density_np >= prev) & (density_np < r)).sum()
            print(f"    [{prev:.3f}, {r:.3f}): {count} ({count/len(density_np)*100:.1f}%)")
            prev = r
        count = (density_np >= ranges[-1]).sum()
        print(f"    >={ranges[-1]:.3f}: {count} ({count/len(density_np)*100:.1f}%)")
    
    # Analyze positions
    positions = ckpt.get('positions')
    if positions is not None:
        if hasattr(positions, 'data'):
            pos_data = positions.data
        else:
            pos_data = positions
        print(f"\nPositions Analysis:")
        print(f"  Shape: {pos_data.shape}")
        print(f"  X: min={pos_data[:, 0].min().item():.2f}, max={pos_data[:, 0].max().item():.2f}, mean={pos_data[:, 0].mean().item():.2f}")
        print(f"  Y: min={pos_data[:, 1].min().item():.2f}, max={pos_data[:, 1].max().item():.2f}, mean={pos_data[:, 1].mean().item():.2f}")
        print(f"  Z: min={pos_data[:, 2].min().item():.2f}, max={pos_data[:, 2].max().item():.2f}, mean={pos_data[:, 2].mean().item():.2f}")
    
    # Analyze scale
    scale = ckpt.get('scale')
    if scale is not None:
        if hasattr(scale, 'data'):
            scale_data = scale.data
        else:
            scale_data = scale
        print(f"\nScale Analysis (pre-activation):")
        print(f"  Shape: {scale_data.shape}")
        print(f"  Mean: {scale_data.mean().item():.6f}")
        print(f"  Std: {scale_data.std().item():.6f}")
        print(f"  Min: {scale_data.min().item():.6f}")
        print(f"  Max: {scale_data.max().item():.6f}")
        
        # Apply exp activation
        scale_activated = torch.exp(scale_data)
        print(f"\nScale Analysis (post-exp activation):")
        print(f"  Mean: {scale_activated.mean().item():.6f}")
        print(f"  Std: {scale_activated.std().item():.6f}")
        print(f"  Min: {scale_activated.min().item():.6f}")
        print(f"  Max: {scale_activated.max().item():.6f}")
    
    # Analyze features_albedo
    features = ckpt.get('features_albedo')
    if features is not None:
        if hasattr(features, 'data'):
            feat_data = features.data
        else:
            feat_data = features
        print(f"\nFeatures Albedo Analysis:")
        print(f"  Shape: {feat_data.shape}")
        print(f"  Mean: {feat_data.mean().item():.6f}")
        print(f"  Std: {feat_data.std().item():.6f}")
    
    # Other metadata
    print(f"\nCheckpoint Keys: {list(ckpt.keys())}")
    if 'n_active_features' in ckpt:
        print(f"  n_active_features: {ckpt['n_active_features']}")
    if 'max_n_features' in ckpt:
        print(f"  max_n_features: {ckpt['max_n_features']}")
    if 'scene_extent' in ckpt:
        print(f"  scene_extent: {ckpt['scene_extent']}")
    if 'global_step' in ckpt:
        print(f"  global_step: {ckpt['global_step']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="Path to checkpoint file or directory")
    args = parser.parse_args()
    
    analyze_checkpoint(args.ckpt_path)
    
    print(f"\n{'='*60}")
    print("Analysis Complete")
    print(f"{'='*60}")
