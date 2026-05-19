#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 4 Sanity Check - Minimal version.

Usage:
    python tools/sanity_check_minimal.py \
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config path relative to configs/')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print("="*70)
    print("Phase 4 Sanity Check - Minimal")
    print("="*70)

    # Load config via hydra (like train.py)
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
    
    print(f"Config loaded: person_feature_dim={conf.model.person_feature_dim}, use_reid={conf.loss.use_reid}")
    
    # Initialize trainer (this loads dataset + model)
    print("\nInitializing trainer...")
    trainer = Trainer3DGRUT(conf)
    
    print(f"Train dataset: {len(trainer.train_dataset)} samples")
    print(f"Model device: {trainer.device}")
    
    # Get one batch
    print("\n" + "="*70)
    print("Step 1: Checking dataloader + teacher cache")
    print("="*70)
    
    batch_data = next(iter(trainer.train_dataloader))
    
    # Convert to GPU batch
    gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)
    
    print(f"Batch image shape: {gpu_batch.rgb_gt.shape if gpu_batch.rgb_gt is not None else 'N/A'}")
    print(f"Batch rays: {gpu_batch.rays_ori.shape}")
    print(f"Instances count: {len(gpu_batch.instances) if gpu_batch.instances else 0}")
    
    if not gpu_batch.instances:
        print("❌ FAIL: instances is empty")
        return False
    
    # Check instances
    valid_count = 0
    for i, inst in enumerate(gpu_batch.instances):
        if inst.get('valid', False):
            valid_count += 1
            if i == 0:
                print(f"\nFirst valid instance:")
                print(f"  train_id: {inst.get('train_id')}")
                print(f"  raw_id: {inst.get('raw_id')}")
                print(f"  bbox_xyxy: {inst.get('bbox_xyxy')}")
                
                teacher_emb = inst.get('teacher_embedding')
                if teacher_emb is not None:
                    print(f"  teacher_embedding.shape: {teacher_emb.shape}")
                    print(f"  teacher_embedding.norm: {np.linalg.norm(teacher_emb):.6f}")
                else:
                    print(f"  ❌ teacher_embedding is None")
    
    print(f"\nValid instances: {valid_count}/{len(gpu_batch.instances)}")
    
    if valid_count == 0:
        print("❌ FAIL: No valid instances with teacher_embedding")
        return False
    
    print("\n✅ PASS: dataloader check")
    
    # Step 2: Check renderer output
    print("\n" + "="*70)
    print("Step 2: Checking renderer output")
    print("="*70)
    
    trainer.model.train()
    render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
    
    print(f"Render output keys: {list(render_out.keys())}")
    
    pred_rgb = render_out.get('pred_rgb')
    person_feature_map = render_out.get('person_feature_map')
    
    if pred_rgb is None:
        print("❌ FAIL: pred_rgb is None")
        return False
    
    print(f"pred_rgb shape: {pred_rgb.shape}")
    
    if person_feature_map is None:
        print("❌ FAIL: person_feature_map is None")
        return False
    
    D = conf.model.person_feature_dim
    H, W = gpu_batch.rays_ori.shape[1], gpu_batch.rays_ori.shape[2]
    
    print(f"person_feature_map shape: {person_feature_map.shape}")
    print(f"Expected shape: ({D}, {H}, {W})")
    
    if person_feature_map.shape != (D, H, W):
        print(f"❌ FAIL: Shape mismatch")
        return False
    
    if torch.isnan(person_feature_map).any():
        print("❌ FAIL: person_feature_map has NaN")
        return False
    
    if torch.isinf(person_feature_map).any():
        print("❌ FAIL: person_feature_map has Inf")
        return False
    
    print("\n✅ PASS: renderer check")
    
    # Step 3: Check ROI pooling
    print("\n" + "="*70)
    print("Step 3: Checking ROI pooling")
    print("="*70)
    
    # Find first valid instance
    valid_inst = None
    for inst in gpu_batch.instances:
        if inst.get('valid', False) and inst.get('teacher_embedding') is not None:
            valid_inst = inst
            break
    
    if valid_inst is None:
        print("❌ FAIL: No valid instance for ROI pooling")
        return False
    
    bbox = valid_inst['bbox_xyxy']
    print(f"Testing ROI pooling with bbox: {bbox}")
    
    f_v = roi_pool(person_feature_map, torch.tensor(bbox, dtype=torch.float32, device=trainer.device))
    
    print(f"f_v shape: {f_v.shape}")
    print(f"f_v norm: {f_v.norm().item():.6f}")
    
    if f_v.shape != (D,):
        print(f"❌ FAIL: f_v shape {f_v.shape} != ({D},)")
        return False
    
    if torch.isnan(f_v).any():
        print("❌ FAIL: f_v has NaN")
        return False
    
    if f_v.abs().sum() < 1e-6:
        print("⚠️  WARNING: f_v is nearly zero")
    
    print("\n✅ PASS: ROI pooling check")
    
    # Step 4: Check L_reid calculation
    print("\n" + "="*70)
    print("Step 4: Checking L_reid calculation")
    print("="*70)
    
    loss_reid_list = []
    num_valid = 0
    
    for inst in gpu_batch.instances:
        if not inst.get('valid', False):
            continue
        
        teacher_emb = inst.get('teacher_embedding')
        if teacher_emb is None:
            continue
        
        bbox = inst['bbox_xyxy']
        f_v = roi_pool(person_feature_map, torch.tensor(bbox, dtype=torch.float32, device=trainer.device))
        
        f_v_norm = F.normalize(f_v, p=2, dim=0)
        t_v = torch.tensor(teacher_emb, dtype=torch.float32, device=trainer.device)
        t_v_norm = F.normalize(t_v, p=2, dim=0)
        
        loss_i = cosine_distillation_loss(f_v_norm.unsqueeze(0), t_v_norm.unsqueeze(0))
        loss_reid_list.append(loss_i)
        num_valid += 1
        
        if num_valid <= 3:
            print(f"  Instance train_id={inst.get('train_id')}: loss={loss_i.item():.4f}, cos_sim={1-loss_i.item():.4f}")
    
    if len(loss_reid_list) == 0:
        print("⚠️  WARNING: No valid instances for L_reid")
        L_reid = torch.zeros(1, device=trainer.device)
    else:
        L_reid = torch.stack(loss_reid_list).mean()
    
    print(f"\nL_reid: {L_reid.item():.4f}")
    print(f"Number of valid instances: {num_valid}")
    
    if L_reid.item() <= 0:
        print("❌ FAIL: L_reid should be > 0")
        return False
    
    if torch.isnan(L_reid).item():
        print("❌ FAIL: L_reid has NaN")
        return False
    
    print("\n✅ PASS: L_reid calculation check")
    
    # Step 5: Check backward and gradients
    print("\n" + "="*70)
    print("Step 5: Checking backward and gradients")
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
    L_total.backward()
    
    # Check gradients
    person_feature = trainer.model.get_person_feature()
    
    print(f"\nperson_feature.shape: {person_feature.shape}")
    print(f"person_feature.grad is None: {person_feature.grad is None}")
    
    if person_feature.grad is None:
        print("❌ FAIL: person_feature.grad is None")
        return False
    
    grad_mean = person_feature.grad.abs().mean().item()
    grad_max = person_feature.grad.abs().max().item()
    
    print(f"person_feature.grad.abs().mean(): {grad_mean:.6f}")
    print(f"person_feature.grad.abs().max(): {grad_max:.6f}")
    
    if grad_mean < 1e-8:
        print("❌ FAIL: person_feature.grad is all zeros")
        return False
    
    if torch.isnan(person_feature.grad).any().item():
        print("❌ FAIL: person_feature.grad has NaN")
        return False
    
    if torch.isinf(person_feature.grad).any().item():
        print("❌ FAIL: person_feature.grad has Inf")
        return False
    
    print("\n✅ PASS: backward and gradient check")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: All checks passed! 🎉")
    print("="*70)
    print("Training pipeline is ready.")
    print("\nNext steps:")
    print("  1. Run small-scale training: python train.py --config configs/apps/wildtrack_full_3dgut.yaml")
    print("  2. Monitor L_reid and cos(f_v, t_v) in tensorboard")
    print("  3. Verify L_reid decreases and cosine similarity increases")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
