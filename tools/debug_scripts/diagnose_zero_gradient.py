#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Diagnose why person_feature.grad is zero.

Key question: Is this Case A (need warmup) or Case B (feature rendering bug)?

Usage:
    python tools/diagnose_zero_gradient.py \
        --config configs/apps/wildtrack_full_3dgut.yaml
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.roi_pooling import roi_pool


def print_tensor_stats(tensor, name):
    """Print comprehensive tensor statistics."""
    if tensor is None:
        print(f"{name}: None")
        return
    
    print(f"{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  requires_grad: {tensor.requires_grad}")
    print(f"  is_leaf: {tensor.is_leaf}")
    print(f"  grad_fn: {tensor.grad_fn}")
    
    with torch.no_grad():
        print(f"  mean: {tensor.mean().item():.6f}")
        print(f"  std: {tensor.std().item():.6f}")
        print(f"  min: {tensor.min().item():.6f}")
        print(f"  max: {tensor.max().item():.6f}")
        print(f"  abs().mean(): {tensor.abs().mean().item():.6f}")
        
        nonzero = (tensor.abs() > 1e-6).sum().item()
        total = tensor.numel()
        print(f"  nonzero ratio: {nonzero}/{total} ({100*nonzero/total:.4f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print("="*70)
    print("Diagnose Zero Gradient - Phase 4")
    print("="*70)

    # Load config
    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)
    
    conf.model.person_feature_dim = 512
    conf.loss.use_reid = True
    
    # Initialize
    print("\nInitializing trainer...")
    trainer = Trainer3DGRUT(conf)
    
    print(f"Gaussians: {trainer.model.num_gaussians}")
    print(f"person_feature_dim: {trainer.model.get_person_feature().shape[1]}")
    
    # ====================
    # Step 1: Check person_feature parameter status
    # ====================
    print("\n" + "="*70)
    print("Step 1: Check person_feature parameter status")
    print("="*70)
    
    person_feature = trainer.model.get_person_feature()
    print(f"person_feature.shape: {person_feature.shape}")
    print(f"person_feature.requires_grad: {person_feature.requires_grad}")
    print(f"person_feature.is_leaf: {person_feature.is_leaf}")
    
    # Check if in optimizer
    in_optimizer = False
    for group in trainer.model.optimizer.param_groups:
        if 'person_feature' in group.get('name', ''):
            in_optimizer = True
            print(f"In optimizer: YES (group '{group['name']}', lr={group['lr']})")
            # Check object identity
            for param in group['params']:
                if param is person_feature:
                    print(f"  → Object identity: SAME (correct)")
                    break
            else:
                print(f"  → Object identity: DIFFERENT (BUG!)")
            break
    
    if not in_optimizer:
        print("❌ person_feature NOT in optimizer!")
        return False
    else:
        print("✅ person_feature correctly registered in optimizer")
    
    # ====================
    # Step 2: Get batch and forward pass
    # ====================
    print("\n" + "="*70)
    print("Step 2: Forward pass")
    print("="*70)
    
    batch_data = next(iter(trainer.train_dataloader))
    gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)
    
    print(f"Batch instances: {len(gpu_batch.instances)}")
    valid_count = sum(1 for inst in gpu_batch.instances if inst.get('valid', False))
    print(f"Valid instances: {valid_count}")
    
    trainer.model.train()
    trainer.model.zero_grad()
    
    render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
    
    pred_rgb = render_out.get('pred_rgb')
    person_feature_map = render_out.get('person_feature_map')
    
    # Check for additional outputs
    alpha = render_out.get('pred_alpha')
    accumulated_opacity = render_out.get('accumulated_opacity')
    visibility = render_out.get('visibility')
    
    # ====================
    # Step 3: Print statistics
    # ====================
    print("\n" + "="*70)
    print("Step 3: Render output statistics")
    print("="*70)
    
    print_tensor_stats(pred_rgb, "pred_rgb")
    print_tensor_stats(person_feature_map, "person_feature_map")
    
    if alpha is not None:
        print_tensor_stats(alpha, "pred_alpha")
    
    if accumulated_opacity is not None:
        print_tensor_stats(accumulated_opacity, "accumulated_opacity")
    
    if visibility is not None:
        print_tensor_stats(visibility, "visibility")
    
    # ====================
    # Step 4: Case A/B judgment
    # ====================
    print("\n" + "="*70)
    print("Step 4: Case A/B Judgment")
    print("="*70)
    
    with torch.no_grad():
        rgb_nonzero_ratio = (pred_rgb.abs() > 1e-6).sum().item() / pred_rgb.numel()
        feature_nonzero_ratio = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel()
    
    print(f"pred_rgb nonzero ratio: {rgb_nonzero_ratio*100:.4f}%")
    print(f"person_feature_map nonzero ratio: {feature_nonzero_ratio*100:.4f}%")
    
    if rgb_nonzero_ratio < 0.01:  # < 0.01% nonzero
        print("\n⚠️  JUDGMENT: Case A - Random initialization has no rendering contribution")
        print("   Both pred_rgb and person_feature_map are near zero.")
        print("   This is NORMAL for random initialization.")
        print("   Solution: Warmup training with RGB loss first.")
        case = 'A'
    else:
        print("\n⚠️  JUDGMENT: Case B - Feature rendering branch may have bug")
        print("   pred_rgb has content but person_feature_map is zero.")
        print("   Need to check _FeatureRenderWrapper and renderer.")
        case = 'B'
    
    # ====================
    # Step 5: Minimal gradient test
    # ====================
    print("\n" + "="*70)
    print("Step 5: Minimal gradient test")
    print("="*70)
    
    trainer.model.zero_grad()
    
    # Test 1: Gradient from person_feature_map.sum()
    loss_map = person_feature_map.abs().mean()
    loss_map.backward()
    
    print(f"\nBackward from person_feature_map.abs().mean():")
    print(f"  person_feature.grad is None: {person_feature.grad is None}")
    
    if person_feature.grad is not None:
        grad_mean = person_feature.grad.abs().mean().item()
        grad_max = person_feature.grad.abs().max().item()
        nonzero_grad = (person_feature.grad.abs() > 1e-8).sum().item()
        total = person_feature.grad.numel()
        
        print(f"  person_feature.grad.abs().mean(): {grad_mean:.6f}")
        print(f"  person_feature.grad.abs().max(): {grad_max:.6f}")
        print(f"  Nonzero gradient elements: {nonzero_grad}/{total} ({100*nonzero_grad/total:.4f}%)")
        
        if grad_mean > 1e-8:
            print(f"  ✅ Gradient flows from person_feature_map to person_feature!")
        else:
            print(f"  ⚠️  Gradient is zero despite connected computation graph")
    
    # ====================
    # Step 6: Check _FeatureRenderWrapper if Case B
    # ====================
    if case == 'B':
        print("\n" + "="*70)
        print("Step 6: Inspect _FeatureRenderWrapper (Case B)")
        print("="*70)
        
        # Check get_features() output
        wrapper = type('obj', (object,), {'_model': trainer.model})
        from threedgrut.model.model import _FeatureRenderWrapper
        
        person_feature = trainer.model.get_person_feature()
        chunk = person_feature[:, :3]
        
        test_wrapper = _FeatureRenderWrapper(trainer.model, chunk)
        features = test_wrapper.get_features()
        
        print(f"_FeatureRenderWrapper.get_features():")
        print(f"  output shape: {features.shape}")
        print(f"  output requires_grad: {features.requires_grad}")
        print(f"  output dtype: {features.dtype}")
        
        with torch.no_grad():
            print(f"  output mean: {features.mean().item():.6f}")
            print(f"  output[:, :3].mean(): {features[:, :3].mean().item():.6f}")
            print(f"  output[:, 3:].mean(): {features[:, 3:].mean().item():.6f}")
    
    # ====================
    # Step 7: Recommendation
    # ====================
    print("\n" + "="*70)
    print("Step 7: Recommendation")
    print("="*70)
    
    if case == 'A':
        print("Recommended action:")
        print("  1. Run warmup training (100-500 iterations) with RGB loss")
        print("  2. Then re-test person_feature.grad")
        print("  3. Expected: person_feature_map becomes non-zero")
        print("  4. Then L_reid gradient should flow correctly")
        print("\nExample command:")
        print(f"  python tools/quick_phase4_test.py --config {args.config}")
    else:  # Case B
        print("Recommended action:")
        print("  1. Check if renderer uses wrapper.get_features()")
        print("  2. Verify feature channel packing/unpacking")
        print("  3. Check if feature rendering is accidentally disabled")
        print("  4. Inspect rasterizer feature handling code")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
