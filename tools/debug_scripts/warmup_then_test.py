#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 4 Warmup Test - Train RGB first, then verify ReID gradient.

Usage:
    python tools/warmup_then_test.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --warmup_iters 200 \
        --test_iters 50
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


def print_stats(tensor, name):
    """Print tensor statistics."""
    with torch.no_grad():
        mean = tensor.mean().item()
        std = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        nonzero = (tensor.abs() > 1e-6).sum().item()
        total = tensor.numel()
        print(f"{name}: mean={mean:.6f}, std={std:.6f}, " +
              f"min={min_val:.6f}, max={max_val:.6f}, " +
              f"nonzero={nonzero}/{total} ({100*nonzero/total:.4f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--test_iters', type=int, default=50)
    parser.add_argument('--lambda_reid', type=float, default=0.0, 
                        help='Use 0.0 for pure RGB warmup, >0 for ReID training')
    args = parser.parse_args()

    print("="*70)
    print("Phase 4 Warmup Test")
    print("="*70)

    # Load config
    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)
    
    conf.model.person_feature_dim = 512
    conf.loss.use_reid = args.lambda_reid > 0
    conf.loss.lambda_reid = args.lambda_reid
    
    # Initialize
    print(f"\nInitializing trainer (warmup_iters={args.warmup_iters}, " +
          f"test_iters={args.test_iters}, lambda_reid={args.lambda_reid})...")
    trainer = Trainer3DGRUT(conf)
    
    print(f"Gaussians: {trainer.model.num_gaussians}")
    print(f"person_feature_dim: {trainer.model.get_person_feature().shape[1]}")
    
    # Get data iterator
    train_iter = iter(trainer.train_dataloader)
    
    # ====================
    # Warmup Phase
    # ====================
    print("\n" + "="*70)
    print(f"Warmup Phase: {args.warmup_iters} iterations")
    print("="*70)
    
    trainer.model.train()
    
    losses_rgb = []
    losses_reid = []
    pred_rgb_stats = []
    person_feature_map_stats = []
    
    for step in range(args.warmup_iters + args.test_iters):
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(trainer.train_dataloader)
            batch_data = next(train_iter)
        
        gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)
        
        # Skip if no valid instances
        valid_count = sum(1 for inst in gpu_batch.instances 
                         if inst.get('valid', False) and inst.get('teacher_embedding') is not None)
        if valid_count == 0:
            continue
        
        trainer.model.zero_grad()
        
        # Forward
        render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
        pred_rgb = render_out['pred_rgb']
        person_feature_map = render_out['person_feature_map']
        
        # Compute L_rgb
        rgb_gt = gpu_batch.rgb_gt
        L_rgb = F.l1_loss(pred_rgb, rgb_gt)
        
        # Compute L_reid (only if lambda_reid > 0)
        L_reid = torch.zeros(1, device=trainer.device)
        if args.lambda_reid > 0:
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
                L_reid = L_reid + (1 - torch.dot(f_v_norm, t_v_norm))
            
            if valid_count > 0:
                L_reid = L_reid / valid_count
        
        # Total loss
        L_total = L_rgb + args.lambda_reid * L_reid
        
        # Backward
        L_total.backward()
        
        # Optimizer step
        trainer.model.optimizer.step()
        
        # Log
        if step < 10 or step % 20 == 0 or step == args.warmup_iters - 1:
            with torch.no_grad():
                rgb_nonzero = (pred_rgb.abs() > 1e-6).sum().item() / pred_rgb.numel() * 100
                feature_nonzero = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
            print(f"Step {step:4d}: L_rgb={L_rgb.item():.4f}, " +
                  f"L_reid={L_reid.item():.4f}, " +
                  f"rgb_nonzero={rgb_nonzero:.4f}%, " +
                  f"feature_nonzero={feature_nonzero:.4f}%")
        
        # Collect stats
        if step >= args.warmup_iters:
            losses_rgb.append(L_rgb.item())
            losses_reid.append(L_reid.item())
            pred_rgb_stats.append({
                'mean': pred_rgb.mean().item(),
                'std': pred_rgb.std().item(),
                'nonzero': (pred_rgb.abs() > 1e-6).sum().item() / pred_rgb.numel() * 100
            })
            person_feature_map_stats.append({
                'mean': person_feature_map.mean().item(),
                'std': person_feature_map.std().item(),
                'nonzero': (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
            })
    
    # ====================
    # Test Phase Analysis
    # ====================
    print("\n" + "="*70)
    print("Test Phase Analysis")
    print("="*70)
    
    if len(pred_rgb_stats) > 0:
        avg_rgb_nonzero = np.mean([s['nonzero'] for s in pred_rgb_stats])
        avg_feature_nonzero = np.mean([s['nonzero'] for s in person_feature_map_stats])
        
        print(f"Average during test phase:")
        print(f"  L_rgb: {np.mean(losses_rgb):.4f} ± {np.std(losses_rgb):.4f}")
        print(f"  L_reid: {np.mean(losses_reid):.4f} ± {np.std(losses_reid):.4f}")
        print(f"  pred_rgb nonzero: {avg_rgb_nonzero:.4f}%")
        print(f"  person_feature_map nonzero: {avg_feature_nonzero:.4f}%")
        
        # Check improvement
        if len(pred_rgb_stats) > 10:
            early_rgb = np.mean([s['nonzero'] for s in pred_rgb_stats[:10]])
            late_rgb = np.mean([s['nonzero'] for s in pred_rgb_stats[-10:]])
            early_feature = np.mean([s['nonzero'] for s in person_feature_map_stats[:10]])
            late_feature = np.mean([s['nonzero'] for s in person_feature_map_stats[-10:]])
            
            print(f"\nImprovement:")
            print(f"  pred_rgb nonzero: {early_rgb:.4f}% → {late_rgb:.4f}% " +
                  f"({'✅ increased' if late_rgb > early_rgb else '⚠️  decreased'})")
            print(f"  person_feature_map nonzero: {early_feature:.4f}% → {late_feature:.4f}% " +
                  f"({'✅ increased' if late_feature > early_feature else '⚠️  decreased'})")
    
    # ====================
    # Final Gradient Check
    # ====================
    print("\n" + "="*70)
    print("Final Gradient Check")
    print("="*70)
    
    # Get a fresh batch
    batch_data = next(iter(trainer.train_dataloader))
    gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)
    
    if len(gpu_batch.instances) > 0:
        trainer.model.zero_grad()
        
        render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
        pred_rgb = render_out['pred_rgb']
        person_feature_map = render_out['person_feature_map']
        
        print_stats(pred_rgb, "pred_rgb")
        print_stats(person_feature_map, "person_feature_map")
        
        # Test gradient from person_feature_map
        loss_map = person_feature_map.abs().mean()
        loss_map.backward()
        
        person_feature = trainer.model.get_person_feature()
        
        print(f"\nGradient from person_feature_map.abs().mean():")
        print(f"  person_feature.grad is None: {person_feature.grad is None}")
        
        if person_feature.grad is not None:
            grad_mean = person_feature.grad.abs().mean().item()
            grad_max = person_feature.grad.abs().max().item()
            nonzero_grad = (person_feature.grad.abs() > 1e-8).sum().item()
            total = person_feature.grad.numel()
            
            print(f"  person_feature.grad.abs().mean(): {grad_mean:.6f}")
            print(f"  person_feature.grad.abs().max(): {grad_max:.6f}")
            print(f"  Nonzero gradient: {nonzero_grad}/{total} ({100*nonzero_grad/total:.4f}%)")
            
            if grad_mean > 1e-8:
                print(f"  ✅ SUCCESS: Non-zero gradient flows to person_feature!")
            else:
                print(f"  ⚠️  Gradient still zero - may need more training")
        
        # Test gradient from L_reid
        if len(gpu_batch.instances) > 0:
            trainer.model.zero_grad()
            
            valid_inst = None
            for inst in gpu_batch.instances:
                if inst.get('valid', False) and inst.get('teacher_embedding') is not None:
                    valid_inst = inst
                    break
            
            if valid_inst is not None:
                bbox = valid_inst['bbox_xyxy']
                f_v = roi_pool(person_feature_map, torch.tensor(bbox, dtype=torch.float32, device=trainer.device))
                t_v = torch.tensor(valid_inst['teacher_embedding'], dtype=torch.float32, device=trainer.device)
                f_v_norm = F.normalize(f_v, p=2, dim=0)
                t_v_norm = F.normalize(t_v, p=2, dim=0)
                
                cos_sim = torch.dot(f_v_norm, t_v_norm).item()
                print(f"\nL_reid test:")
                print(f"  cos(f_v, t_v): {cos_sim:.4f}")
                
                L_reid = 1 - cos_sim
                L_reid.backward()
                
                print(f"  After L_reid.backward():")
                print(f"    person_feature.grad is None: {person_feature.grad is None}")
                if person_feature.grad is not None:
                    grad_mean = person_feature.grad.abs().mean().item()
                    print(f"    person_feature.grad.abs().mean(): {grad_mean:.6f}")
    
    # ====================
    # Summary
    # ====================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if avg_rgb_nonzero > 0.1 and avg_feature_nonzero > 0.1:
        print("✅ Warmup successful: Both RGB and feature maps are non-zero")
        print("✅ Gradient should now flow correctly")
    elif avg_rgb_nonzero > 0.1:
        print("⚠️  RGB is non-zero but feature map still weak")
        print("   May need more training or higher lambda_reid")
    else:
        print("⚠️  Both RGB and feature maps are still weak")
        print("   Need more warmup training")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
