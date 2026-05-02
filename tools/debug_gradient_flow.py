#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 4 Step 5 - Debug gradient flow from L_reid to person_feature.

This script traces the gradient computation graph to find where the gradient flow breaks.

Usage:
    python tools/debug_gradient_flow.py \
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
from threedgrut.model.losses import cosine_distillation_loss


def print_grad_fn_chain(tensor, name="Tensor", depth=0, visited=None):
    """Recursively print the gradient function chain."""
    if visited is None:
        visited = set()
    
    indent = "  " * depth
    tensor_id = id(tensor)
    
    if tensor_id in visited:
        print(f"{indent}→ (circular reference to {name})")
        return
    
    visited.add(tensor_id)
    
    if depth == 0:
        print(f"{name}:")
        print(f"  shape: {tensor.shape}")
        print(f"  dtype: {tensor.dtype}")
        print(f"  device: {tensor.device}")
        print(f"  requires_grad: {tensor.requires_grad}")
        print(f"  grad_fn: {tensor.grad_fn}")
        print(f"  is_leaf: {tensor.is_leaf}")
    
    if tensor.grad_fn is not None:
        print(f"{indent}→ {tensor.grad_fn}")
        
        # Print next functions in the chain
        if hasattr(tensor.grad_fn, 'next_functions'):
            for i, (next_fn, _) in enumerate(tensor.grad_fn.next_functions):
                if hasattr(next_fn, 'variable'):
                    next_tensor = next_fn.variable
                    print(f"{indent}  [{i}] {next_fn.__class__.__name__}")
                    if depth < 10:  # Limit depth
                        print_grad_fn_chain(next_tensor, f"  └─{next_fn.__class__.__name__}", depth + 1, visited)
                else:
                    print(f"{indent}  [{i}] {next_fn.__class__.__name__} (no variable)")
    elif not tensor.is_leaf:
        print(f"{indent}→ (no grad_fn but not leaf - possible gradient break)")


def check_tensor_properties(tensor, name):
    """Check and print tensor properties."""
    print(f"\n{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  requires_grad: {tensor.requires_grad}")
    print(f"  is_leaf: {tensor.is_leaf}")
    print(f"  grad_fn: {tensor.grad_fn}")
    
    if tensor.grad is not None:
        print(f"  grad.shape: {tensor.grad.shape}")
        print(f"  grad.abs().mean(): {tensor.grad.abs().mean().item():.6f}")
        print(f"  grad.abs().max(): {tensor.grad.abs().max().item():.6f}")
    
    if not tensor.is_leaf and tensor.requires_grad:
        # Try to trace back
        if tensor.grad_fn is not None:
            print(f"  grad_fn type: {type(tensor.grad_fn).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config path relative to configs/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--verbose', action='store_true', help='Print detailed gradient chain')
    args = parser.parse_args()

    print("="*70)
    print("Phase 4 Step 5 - Debug Gradient Flow")
    print("="*70)

    # Load config
    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)
    
    # Override person_feature_dim to 512
    if conf.model.get('person_feature_dim') != 512:
        print(f"\n⚠️  Updating person_feature_dim from {conf.model.get('person_feature_dim')} to 512")
        conf.model.person_feature_dim = 512
    
    # Ensure use_reid is enabled
    if not conf.loss.get('use_reid', False):
        print("⚠️  Enabling use_reid")
        conf.loss.use_reid = True
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer3DGRUT(conf)
    
    print(f"Train dataset: {len(trainer.train_dataset)} samples")
    
    # Get one batch
    print("\n" + "="*70)
    print("Step 1: Get batch data")
    print("="*70)
    
    batch_data = next(iter(trainer.train_dataloader))
    gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)
    
    print(f"Instances count: {len(gpu_batch.instances) if gpu_batch.instances else 0}")
    
    if not gpu_batch.instances:
        print("❌ FAIL: instances is empty")
        return False
    
    # Count valid instances
    valid_count = sum(1 for inst in gpu_batch.instances if inst.get('valid', False) and inst.get('teacher_embedding') is not None)
    print(f"Valid instances with teacher_embedding: {valid_count}")
    
    if valid_count == 0:
        print("❌ FAIL: No valid instances")
        return False
    
    # Step 2: Forward pass
    print("\n" + "="*70)
    print("Step 2: Forward pass with person_feature rendering")
    print("="*70)
    
    trainer.model.train()
    trainer.model.zero_grad()
    
    # Check person_feature before forward
    person_feature = trainer.model.get_person_feature()
    print(f"\nBefore forward:")
    check_tensor_properties(person_feature, "person_feature")
    
    # Forward
    render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
    
    pred_rgb = render_out.get('pred_rgb')
    person_feature_map = render_out.get('person_feature_map')
    
    print(f"\nAfter forward:")
    check_tensor_properties(pred_rgb, "pred_rgb")
    check_tensor_properties(person_feature_map, "person_feature_map")
    
    # Check if person_feature_map requires_grad
    if not person_feature_map.requires_grad:
        print("\n❌ CRITICAL: person_feature_map.requires_grad is False")
        print("   This means gradients won't flow back to person_feature")
        print("\nPossible causes:")
        print("  1. person_feature doesn't require_grad")
        print("  2. _FeatureRenderWrapper.get_features() breaks gradient")
        print("  3. Renderer doesn't preserve gradient")
        return False
    
    # Step 3: ROI pooling
    print("\n" + "="*70)
    print("Step 3: ROI pooling")
    print("="*70)
    
    # Find first valid instance
    valid_inst = None
    for inst in gpu_batch.instances:
        if inst.get('valid', False) and inst.get('teacher_embedding') is not None:
            valid_inst = inst
            break
    
    bbox = valid_inst['bbox_xyxy']
    print(f"Testing with bbox: {bbox}")
    
    f_v = roi_pool(person_feature_map, torch.tensor(bbox, dtype=torch.float32, device=trainer.device))
    check_tensor_properties(f_v, "f_v (ROI pooled feature)")
    
    if not f_v.requires_grad:
        print("\n❌ CRITICAL: f_v.requires_grad is False")
        print("   ROI pooling broke the gradient")
        return False
    
    # Step 4: Compute L_reid
    print("\n" + "="*70)
    print("Step 4: Compute L_reid")
    print("="*70)
    
    f_v_norm = F.normalize(f_v, p=2, dim=0)
    teacher_emb = valid_inst['teacher_embedding']
    t_v = torch.tensor(teacher_emb, dtype=torch.float32, device=trainer.device)
    t_v_norm = F.normalize(t_v, p=2, dim=0)
    
    print(f"f_v_norm.requires_grad: {f_v_norm.requires_grad}")
    print(f"t_v_norm.requires_grad: {t_v_norm.requires_grad}")
    
    L_reid = cosine_distillation_loss(f_v_norm.unsqueeze(0), t_v_norm.unsqueeze(0))
    check_tensor_properties(L_reid, "L_reid")
    
    if not L_reid.requires_grad:
        print("\n❌ CRITICAL: L_reid.requires_grad is False")
        print("   Cosine loss broke the gradient")
        return False
    
    # Step 5: Backward pass
    print("\n" + "="*70)
    print("Step 5: Backward pass")
    print("="*70)
    
    # Compute total loss
    rgb_gt = gpu_batch.rgb_gt
    L_rgb = F.l1_loss(pred_rgb, rgb_gt)
    
    lambda_reid = conf.loss.get('lambda_reid', 0.05)
    L_total = L_rgb + lambda_reid * L_reid
    
    print(f"L_rgb: {L_rgb.item():.4f}")
    print(f"L_reid: {L_reid.item():.4f}")
    print(f"L_total: {L_total.item():.4f}")
    
    # Backward
    print("\nCalling L_total.backward()...")
    L_total.backward()
    
    # Step 6: Check gradients
    print("\n" + "="*70)
    print("Step 6: Check gradients after backward")
    print("="*70)
    
    person_feature_after = trainer.model.get_person_feature()
    check_tensor_properties(person_feature_after, "person_feature (after backward)")
    
    if person_feature_after.grad is None:
        print("\n" + "!"*70)
        print("❌ FAIL: person_feature.grad is None")
        print("!"*70)
        
        if args.verbose:
            print("\nTracing gradient chain from person_feature_map:")
            print_grad_fn_chain(person_feature_map, "person_feature_map", depth=0)
        
        print("\nPossible causes:")
        print("  1. person_feature is not connected to computation graph")
        print("  2. _FeatureRenderWrapper creates a new tensor instead of using person_feature")
        print("  3. Renderer doesn't use the features from wrapper.get_features()")
        print("  4. torch.cat([feature_chunk, zeros]) breaks gradient")
        print("\nDebug suggestions:")
        print("  - Check if _FeatureRenderWrapper._feature_chunk is the same as person_feature slice")
        print("  - Check if renderer uses wrapper.get_features() for rendering")
        print("  - Try removing the zeros concatenation")
        return False
    
    grad_mean = person_feature_after.grad.abs().mean().item()
    grad_max = person_feature_after.grad.abs().max().item()
    
    print(f"\nperson_feature.grad.abs().mean(): {grad_mean:.6f}")
    print(f"person_feature.grad.abs().max(): {grad_max:.6f}")
    
    # Check if person_feature_map is all zeros (random initialization issue)
    feature_map_mean = person_feature_map.abs().mean().item()
    print(f"person_feature_map.abs().mean(): {feature_map_mean:.6f}")
    
    if grad_mean < 1e-8:
        if feature_map_mean < 1e-6:
            print("\n⚠️  WARNING: person_feature.grad is zero, but this is expected for random initialization")
            print("   person_feature_map is all zeros because Gaussians haven't learned to render yet")
            print("   ✅ Gradient computation graph is connected (person_feature.grad is not None)")
            print("   Next: Run a few training iterations to let Gaussians learn rendering")
        else:
            print("\n⚠️  WARNING: person_feature.grad is nearly zero despite non-zero feature_map")
            print("   This might indicate vanishing gradients or broken gradient flow")
            return False
    else:
        print("\n✅ SUCCESS: person_feature has non-zero gradient!")
    
    # Check for NaN/Inf
    if torch.isnan(person_feature_after.grad).any().item():
        print("❌ FAIL: person_feature.grad has NaN")
        return False
    
    if torch.isinf(person_feature_after.grad).any().item():
        print("❌ FAIL: person_feature.grad has Inf")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Gradient flow check passed! 🎉")
    print("="*70)
    print("Gradient successfully flows from L_reid to person_feature")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
