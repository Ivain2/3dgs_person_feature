#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 4 Sanity Check Script.

Verifies the complete training pipeline:
  annotations_remapped → teacher cache → dataloader → renderer
  → person_feature_map → ROI pooling → cosine loss → backward
  → person_feature.grad is not None

Usage:
    python tools/sanity_check_phase4.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --batch_size 1 \
        --check_dataloader \
        --check_renderer \
        --check_roi \
        --check_loss \
        --check_backward \
        --device cuda:0
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from threedgrut.datasets import make
from threedgrut.datasets.protocols import Batch
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import roi_pool
from threedgrut.model.losses import cosine_distillation_loss


def check_dataloader(conf, batch_size=1, device="cuda:0"):
    """Step 1: Check dataloader + teacher cache."""
    print("\n" + "="*60)
    print("Step 1: Checking dataloader + teacher cache")
    print("="*60)

    train_dataset, val_dataset = make(name=conf.dataset.type, config=conf, ray_jitter=None)
    
    # Get one batch
    batch = next(iter(train_dataset))
    
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape if 'image' in batch else 'N/A'}")
    print(f"Camera: {batch.get('camera', 'N/A')}")
    print(f"Frame ID: {batch.get('frame_id', 'N/A')}")
    print(f"Camera ID: {batch.get('camera_id', 'N/A')}")
    
    # Check instances
    instances = batch.get('instances')
    if instances is None:
        print("\n❌ FAIL: instances is None")
        return False
    
    print(f"\nInstances count: {len(instances)}")
    
    if len(instances) == 0:
        print("⚠️  WARNING: instances is empty")
        return False
    
    # Check first instance
    inst = instances[0]
    print(f"\nFirst instance keys: {inst.keys()}")
    print(f"  bbox_xyxy: {inst.get('bbox_xyxy')}")
    print(f"  train_id: {inst.get('train_id')}")
    print(f"  raw_id: {inst.get('raw_id')}")
    print(f"  valid: {inst.get('valid')}")
    
    teacher_emb = inst.get('teacher_embedding')
    if teacher_emb is None:
        print("\n❌ FAIL: teacher_embedding is None")
        return False
    
    print(f"\nteacher_embedding shape: {teacher_emb.shape}")
    print(f"teacher_embedding dtype: {teacher_emb.dtype}")
    print(f"teacher_embedding norm: {np.linalg.norm(teacher_emb):.6f}")
    
    # Assertions
    assert teacher_emb.shape == (512,), f"Expected shape (512,), got {teacher_emb.shape}"
    assert 0.99 < np.linalg.norm(teacher_emb) < 1.01, f"Expected norm ≈ 1.0, got {np.linalg.norm(teacher_emb)}"
    assert inst.get('train_id') is not None, "train_id is missing"
    assert inst.get('raw_id') is not None, "raw_id is missing (needed for debug)"
    
    # Check all instances
    valid_count = sum(1 for inst in instances if inst.get('valid', False))
    print(f"\nValid instances (with teacher_embedding): {valid_count}/{len(instances)}")
    
    for i, inst in enumerate(instances):
        if inst.get('valid', False):
            emb = inst.get('teacher_embedding')
            if emb is not None:
                norm = np.linalg.norm(emb)
                if not (0.99 < norm < 1.01):
                    print(f"❌ FAIL: Instance {i} has embedding norm {norm:.4f}")
                    return False
    
    print("\n✅ PASS: dataloader + teacher cache check")
    return True


def check_renderer(conf, device="cuda:0"):
    """Step 2: Check renderer output."""
    print("\n" + "="*60)
    print("Step 2: Checking renderer output")
    print("="*60)
    
    train_dataset, _ = make(name=conf.dataset.type, config=conf, ray_jitter=None)
    batch_data = next(iter(train_dataset))
    
    # Convert to Batch
    gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(batch_data)
    
    # Create model
    model = MixtureOfGaussians(conf).to(device)
    
    # Forward with render_person_feature=True
    render_out = model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
    
    print(f"\nRender output keys: {render_out.keys()}")
    
    # Check pred_rgb
    pred_rgb = render_out.get('pred_rgb')
    if pred_rgb is None:
        print("❌ FAIL: pred_rgb is None")
        return False
    
    print(f"pred_rgb shape: {pred_rgb.shape}")
    
    # Check person_feature_map
    person_feature_map = render_out.get('person_feature_map')
    if person_feature_map is None:
        print("❌ FAIL: person_feature_map is None")
        return False
    
    print(f"person_feature_map shape: {person_feature_map.shape}")
    print(f"person_feature_map dtype: {person_feature_map.dtype}")
    print(f"person_feature_map has NaN: {torch.isnan(person_feature_map).any().item()}")
    print(f"person_feature_map has Inf: {torch.isinf(person_feature_map).any().item()}")
    
    # Get expected dimensions
    D_expected = conf.model.get('person_feature_dim', 64)
    H, W = gpu_batch.rays_ori.shape[1], gpu_batch.rays_ori.shape[2]
    
    print(f"\nExpected: D={D_expected}, H={H}, W={W}")
    
    # Assertions
    assert person_feature_map.shape == (D_expected, H, W), \
        f"Expected shape ({D_expected}, {H}, {W}), got {person_feature_map.shape}"
    
    assert not torch.isnan(person_feature_map).any(), "person_feature_map has NaN"
    assert not torch.isinf(person_feature_map).any(), "person_feature_map has Inf"
    
    print("\n✅ PASS: renderer output check")
    return True


def check_roi_pooling(conf, device="cuda:0"):
    """Step 3: Check ROI pooling."""
    print("\n" + "="*60)
    print("Step 3: Checking ROI pooling")
    print("="*60)
    
    train_dataset, _ = make(conf, 1)
    batch_data = next(iter(train_dataset))
    gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(batch_data)
    
    model = MixtureOfGaussians(conf).to(device)
    render_out = model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
    
    person_feature_map = render_out.get('person_feature_map')
    if person_feature_map is None:
        print("❌ FAIL: person_feature_map is None")
        return False
    
    instances = gpu_batch.instances
    if not instances:
        print("❌ FAIL: instances is empty")
        return False
    
    # Find a valid instance
    valid_inst = None
    for inst in instances:
        if inst.get('valid', False) and inst.get('teacher_embedding') is not None:
            valid_inst = inst
            break
    
    if valid_inst is None:
        print("⚠️  WARNING: No valid instance found, skipping ROI pooling check")
        return True
    
    bbox = valid_inst['bbox_xyxy']
    print(f"\nTesting ROI pooling with bbox: {bbox}")
    
    # ROI pooling
    f_v = roi_pool(person_feature_map, torch.tensor(bbox, dtype=torch.float32, device=device))
    
    print(f"f_v shape: {f_v.shape}")
    print(f"f_v dtype: {f_v.dtype}")
    print(f"f_v norm: {f_v.norm().item():.6f}")
    print(f"f_v has NaN: {torch.isnan(f_v).any().item()}")
    print(f"f_v has Inf: {torch.isinf(f_v).any().item()}")
    
    D_expected = conf.model.get('person_feature_dim', 64)
    
    # Assertions
    assert f_v.shape == (D_expected,), f"Expected shape ({D_expected},), got {f_v.shape}"
    assert not torch.isnan(f_v).any(), "f_v has NaN"
    assert not torch.isinf(f_v).any(), "f_v has Inf"
    
    # Check if ROI is non-empty (feature should be non-zero)
    if f_v.abs().sum() < 1e-6:
        print("⚠️  WARNING: f_v is nearly zero, ROI might be empty")
    
    print("\n✅ PASS: ROI pooling check")
    return True


def check_loss(conf, device="cuda:0"):
    """Step 4: Check L_reid calculation."""
    print("\n" + "="*60)
    print("Step 4: Checking L_reid calculation")
    print("="*60)
    
    train_dataset, _ = make(conf, 1)
    batch_data = next(iter(train_dataset))
    gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(batch_data)
    
    model = MixtureOfGaussians(conf).to(device)
    render_out = model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
    
    person_feature_map = render_out.get('person_feature_map')
    instances = gpu_batch.instances
    
    if not instances:
        print("❌ FAIL: instances is empty")
        return False
    
    # Collect valid instances
    loss_list = []
    num_valid = 0
    
    for inst in instances:
        if not inst.get('valid', False):
            continue
        
        teacher_emb = inst.get('teacher_embedding')
        if teacher_emb is None:
            continue
        
        bbox = inst['bbox_xyxy']
        f_v = roi_pool(person_feature_map, torch.tensor(bbox, dtype=torch.float32, device=device))
        
        # Normalize
        f_v_norm = F.normalize(f_v, p=2, dim=0)
        t_v = torch.tensor(teacher_emb, dtype=torch.float32, device=device)
        t_v_norm = F.normalize(t_v, p=2, dim=0)
        
        # Cosine distance
        loss_i = cosine_distillation_loss(f_v_norm.unsqueeze(0), t_v_norm.unsqueeze(0))
        loss_list.append(loss_i)
        num_valid += 1
        
        print(f"  Instance train_id={inst.get('train_id')}: loss={loss_i.item():.4f}, "
              f"cos_sim={(1-loss_i.item()):.4f}")
    
    if len(loss_list) == 0:
        print("⚠️  WARNING: No valid instances for L_reid, L_reid=0")
        return True
    
    L_reid = torch.stack(loss_list).mean()
    
    print(f"\nL_reid: {L_reid.item():.4f}")
    print(f"L_reid has NaN: {torch.isnan(L_reid).any() if hasattr(L_reid, 'any') else torch.isnan(L_reid).item()}")
    print(f"L_reid has Inf: {torch.isinf(L_reid).any() if hasattr(L_reid, 'any') else torch.isinf(L_reid).item()}")
    print(f"Number of valid instances: {num_valid}")
    
    # Assertions
    assert L_reid.item() > 0, "L_reid should be > 0"
    assert not torch.isnan(L_reid).item(), "L_reid has NaN"
    assert not torch.isinf(L_reid).item(), "L_reid has Inf"
    
    print("\n✅ PASS: L_reid calculation check")
    return True


def check_backward(conf, device="cuda:0"):
    """Step 5: Check backward and gradients."""
    print("\n" + "="*60)
    print("Step 5: Checking backward and gradients")
    print("="*60)
    
    train_dataset, _ = make(conf, 1)
    batch_data = next(iter(train_dataset))
    gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(batch_data)
    
    model = MixtureOfGaussians(conf).to(device)
    
    # Forward
    render_out = model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
    
    pred_rgb = render_out.get('pred_rgb')
    person_feature_map = render_out.get('person_feature_map')
    rgb_gt = gpu_batch.rgb_gt
    
    # L_rgb
    L_rgb = F.l1_loss(pred_rgb, rgb_gt)
    
    # L_reid
    instances = gpu_batch.instances
    loss_reid_list = []
    
    if person_feature_map is not None and instances:
        for inst in instances:
            if not inst.get('valid', False):
                continue
            teacher_emb = inst.get('teacher_embedding')
            if teacher_emb is None:
                continue
            
            bbox = inst['bbox_xyxy']
            f_v = roi_pool(person_feature_map, torch.tensor(bbox, dtype=torch.float32, device=device))
            f_v_norm = F.normalize(f_v, p=2, dim=0)
            t_v = torch.tensor(teacher_emb, dtype=torch.float32, device=device)
            t_v_norm = F.normalize(t_v, p=2, dim=0)
            
            loss_i = cosine_distillation_loss(f_v_norm.unsqueeze(0), t_v_norm.unsqueeze(0))
            loss_reid_list.append(loss_i)
    
    if loss_reid_list:
        L_reid = torch.stack(loss_reid_list).mean()
    else:
        L_reid = torch.zeros(1, device=device)
    
    # Total loss
    lambda_reid = conf.loss.get('lambda_reid', 0.05)
    L_total = L_rgb + lambda_reid * L_reid
    
    print(f"L_rgb: {L_rgb.item():.4f}")
    print(f"L_reid: {L_reid.item():.4f}")
    print(f"L_total: {L_total.item():.4f}")
    
    # Backward
    L_total.backward()
    
    # Check gradients
    person_feature = model.get_person_feature()
    
    print(f"\nperson_feature.shape: {person_feature.shape}")
    print(f"person_feature.grad is None: {person_feature.grad is None}")
    
    if person_feature.grad is not None:
        grad_mean = person_feature.grad.abs().mean().item()
        grad_max = person_feature.grad.abs().max().item()
        print(f"person_feature.grad.abs().mean(): {grad_mean:.6f}")
        print(f"person_feature.grad.abs().max(): {grad_max:.6f}")
        print(f"person_feature.grad has NaN: {torch.isnan(person_feature.grad).any().item()}")
        print(f"person_feature.grad has Inf: {torch.isinf(person_feature.grad).any().item()}")
    
    # Assertions
    assert person_feature.grad is not None, "person_feature.grad is None"
    assert person_feature.grad.abs().mean().item() > 0, "person_feature.grad is all zeros"
    assert not torch.isnan(person_feature.grad).any().item(), "person_feature.grad has NaN"
    assert not torch.isinf(person_feature.grad).any().item(), "person_feature.grad has Inf"
    
    print("\n✅ PASS: backward and gradient check")
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Sanity Check")
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--check_dataloader', action='store_true', help='Run dataloader check')
    parser.add_argument('--check_renderer', action='store_true', help='Run renderer check')
    parser.add_argument('--check_roi', action='store_true', help='Run ROI pooling check')
    parser.add_argument('--check_loss', action='store_true', help='Run loss calculation check')
    parser.add_argument('--check_backward', action='store_true', help='Run backward check')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--all', action='store_true', help='Run all checks')
    args = parser.parse_args()
    
    # Load config
    from yacs.config import CfgNode as CN
    conf = CN()
    conf.set_new_allowed(True)
    conf.merge_from_file(args.config)
    
    # Override person_feature_dim to 512 for direct alignment
    if conf.model.get('person_feature_dim', 64) != 512:
        print(f"\n⚠️  WARNING: person_feature_dim={conf.model.get('person_feature_dim')} != 512")
        print("   For MVP simplicity, recommend setting person_feature_dim=512 to align with teacher")
        print("   Continuing with current config...\n")
    
    results = {}
    
    if args.all or args.check_dataloader:
        results['dataloader'] = check_dataloader(conf, args.batch_size, args.device)
    
    if args.all or args.check_renderer:
        results['renderer'] = check_renderer(conf, args.device)
    
    if args.all or args.check_roi:
        results['roi'] = check_roi_pooling(conf, args.device)
    
    if args.all or args.check_loss:
        results['loss'] = check_loss(conf, args.device)
    
    if args.all or args.check_backward:
        results['backward'] = check_backward(conf, args.device)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_pass = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    if all_pass:
        print("\n🎉 All checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
