#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 5 Experiment A: Small-scale overfit test for ReID feature distillation.

Strategy:
  Stage 1: RGB warmup with DIVERSE batches (let Gaussians learn rendering)
  Stage 2: ReID overfit on a FIXED small batch (overfit teacher embeddings)

Usage:
    python tools/phase5_overfit_reid.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --warmup_iters 200 --overfit_iters 500 --lambda_reid 0.05
"""

import argparse
import os
import sys
import json

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.roi_pooling import roi_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--overfit_iters', type=int, default=500)
    parser.add_argument('--lambda_reid', type=float, default=0.05)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5 Experiment A: Small-scale Overfit Test")
    print("=" * 70)
    print(f"  warmup_iters  = {args.warmup_iters}")
    print(f"  overfit_iters = {args.overfit_iters}")
    print(f"  lambda_reid   = {args.lambda_reid}")

    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)

    conf.model.person_feature_dim = 512
    conf.loss.use_reid = True
    conf.loss.lambda_reid = args.lambda_reid

    print("\nInitializing trainer...")
    trainer = Trainer3DGRUT(conf)
    print(f"Gaussians: {trainer.model.num_gaussians}")

    # ===== Stage 1: RGB Warmup with DIVERSE batches =====
    print("\n" + "=" * 70)
    print(f"Stage 1: RGB Warmup with diverse batches ({args.warmup_iters} iters)")
    print("=" * 70)

    trainer.model.train()
    warmup_log = {'L_rgb': [], 'rgb_nz': [], 'feat_nz': [], 'feat_abs': []}
    train_iter = iter(trainer.train_dataloader)

    for step in range(args.warmup_iters):
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(trainer.train_dataloader)
            batch_data = next(train_iter)

        gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)

        trainer.model.zero_grad()
        render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
        pred_rgb = render_out['pred_rgb']
        person_feature_map = render_out['person_feature_map']

        L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)
        L_rgb.backward()
        trainer.model.optimizer.step()

        with torch.no_grad():
            rgb_nz = (pred_rgb.abs() > 1e-6).sum().item() / pred_rgb.numel() * 100
            feat_nz = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
            feat_abs = person_feature_map.abs().mean().item()

        warmup_log['L_rgb'].append(L_rgb.item())
        warmup_log['rgb_nz'].append(rgb_nz)
        warmup_log['feat_nz'].append(feat_nz)
        warmup_log['feat_abs'].append(feat_abs)

        if step % 50 == 0 or step == args.warmup_iters - 1:
            print(f"  Step {step:4d}: L_rgb={L_rgb.item():.4f}, "
                  f"rgb_nz={rgb_nz:.2f}%, feat_nz={feat_nz:.2f}%, feat_abs={feat_abs:.6f}")

    # Check if warmup produced non-zero rendering
    warmup_feat_nz_avg = np.mean(warmup_log['feat_nz'][-20:])
    print(f"\n  Warmup avg feat_nz (last 20): {warmup_feat_nz_avg:.2f}%")

    # Find a good fixed batch for overfitting
    print("\nSearching for a batch with valid instances and non-zero rendering...")
    fixed_batch = None
    best_score = -1

    for i, batch_data in enumerate(trainer.train_dataloader):
        gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)
        valid_count = sum(
            1 for inst in gpu_batch.instances
            if inst.get('valid', False) and inst.get('teacher_embedding') is not None
        )
        if valid_count == 0:
            continue

        # Quick render check
        with torch.no_grad():
            trainer.model.eval()
            render_out = trainer.model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
            feat_nz = (render_out['person_feature_map'].abs() > 1e-6).sum().item() / render_out['person_feature_map'].numel() * 100
            trainer.model.train()

        score = valid_count * 10 + feat_nz
        if score > best_score:
            best_score = score
            fixed_batch = batch_data

        if i >= 30:
            break

    if fixed_batch is None:
        print("❌ No suitable batch found!")
        return False

    gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(fixed_batch)
    valid_count = sum(1 for inst in gpu_batch.instances if inst.get('valid', False) and inst.get('teacher_embedding') is not None)
    print(f"Fixed batch: {valid_count} valid instances")

    # ===== Stage 2: ReID Overfit on fixed batch =====
    print("\n" + "=" * 70)
    print(f"Stage 2: ReID Overfit on fixed batch ({args.overfit_iters} iters)")
    print("=" * 70)

    overfit_log = {
        'L_rgb': [], 'L_reid': [], 'L_total': [], 'cos_sim': [],
        'valid_inst': [], 'grad_mean': [], 'grad_max': [], 'grad_nz': [],
        'rgb_nz': [], 'feat_nz': [], 'feat_abs': [], 'nan_inf': [],
    }

    any_nan_inf = False

    for step in range(args.overfit_iters):
        # Re-create gpu_batch each step to avoid stale references
        gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(fixed_batch)

        trainer.model.zero_grad()
        render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
        pred_rgb = render_out['pred_rgb']
        person_feature_map = render_out['person_feature_map']

        step_nan = torch.isnan(pred_rgb).any() or torch.isnan(person_feature_map).any()

        L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)

        L_reid = torch.zeros(1, device=trainer.device)
        cos_sims_step = []
        step_valid = 0

        for inst in gpu_batch.instances:
            if not inst.get('valid', False):
                continue
            teacher_emb = inst.get('teacher_embedding')
            if teacher_emb is None:
                continue

            bbox = inst['bbox_xyxy']
            f_v = roi_pool(person_feature_map, torch.tensor(bbox, dtype=torch.float32, device=trainer.device))
            f_v_norm = F.normalize(f_v, p=2, dim=0, eps=1e-6)
            t_v = torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
            t_v_norm = F.normalize(t_v, p=2, dim=0, eps=1e-6)

            cos_sim = torch.dot(f_v_norm, t_v_norm)
            cos_sims_step.append(cos_sim.item())
            L_reid = L_reid + (1 - cos_sim)
            step_valid += 1

        if step_valid > 0:
            L_reid = L_reid / step_valid

        L_total = L_rgb + args.lambda_reid * L_reid

        if not step_nan:
            L_total.backward()
            trainer.model.optimizer.step()

        pf = trainer.model.get_person_feature()
        grad_mean = 0.0
        grad_max = 0.0
        grad_nz = 0.0
        if pf.grad is not None:
            grad_mean = pf.grad.abs().mean().item()
            grad_max = pf.grad.abs().max().item()
            grad_nz = (pf.grad.abs() > 1e-8).sum().item() / pf.grad.numel() * 100

        with torch.no_grad():
            rgb_nz = (pred_rgb.abs() > 1e-6).sum().item() / pred_rgb.numel() * 100
            feat_nz = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
            feat_abs = person_feature_map.abs().mean().item()

        avg_cos = np.mean(cos_sims_step) if cos_sims_step else 0

        overfit_log['L_rgb'].append(L_rgb.item())
        overfit_log['L_reid'].append(L_reid.item())
        overfit_log['L_total'].append(L_total.item())
        overfit_log['cos_sim'].append(avg_cos)
        overfit_log['valid_inst'].append(step_valid)
        overfit_log['grad_mean'].append(grad_mean)
        overfit_log['grad_max'].append(grad_max)
        overfit_log['grad_nz'].append(grad_nz)
        overfit_log['rgb_nz'].append(rgb_nz)
        overfit_log['feat_nz'].append(feat_nz)
        overfit_log['feat_abs'].append(feat_abs)
        overfit_log['nan_inf'].append(bool(step_nan))

        should_print = (step < 5 or step % 50 == 0 or step == args.overfit_iters - 1)
        if should_print:
            print(f"  Step {step:4d}: L_rgb={L_rgb.item():.4f}, L_reid={L_reid.item():.4f}, "
                  f"cos={avg_cos:.4f}, grad_mean={grad_mean:.6f}, grad_nz={grad_nz:.4f}%, "
                  f"feat_nz={feat_nz:.2f}%")

    # ===== Analysis =====
    print("\n" + "=" * 70)
    print("EXPERIMENT A: OVERFIT TEST RESULTS")
    print("=" * 70)

    n = len(overfit_log['L_reid'])
    window = min(20, max(5, n // 4))

    early_L_reid = np.mean(overfit_log['L_reid'][:window])
    late_L_reid = np.mean(overfit_log['L_reid'][-window:])
    early_cos = np.mean(overfit_log['cos_sim'][:window])
    late_cos = np.mean(overfit_log['cos_sim'][-window:])
    early_grad = np.mean(overfit_log['grad_mean'][:window])
    late_grad = np.mean(overfit_log['grad_mean'][-window:])

    reid_decrease = (early_L_reid - late_L_reid) / max(early_L_reid, 1e-8) * 100
    cos_increase = late_cos - early_cos

    print(f"\n  Config: warmup={args.warmup_iters}, overfit={args.overfit_iters}, lambda_reid={args.lambda_reid}")
    print(f"  Valid instances per step: {np.mean(overfit_log['valid_inst']):.1f}")
    print(f"\n  L_reid:   first{window}={early_L_reid:.4f}  last{window}={late_L_reid:.4f}  decrease={reid_decrease:.2f}%")
    print(f"  cos_sim:  first{window}={early_cos:.4f}  last{window}={late_cos:.4f}  increase={cos_increase:.4f}")
    print(f"  grad_mean: first{window}={early_grad:.6f}  last{window}={late_grad:.6f}")
    print(f"  grad_nz:  mean={np.mean(overfit_log['grad_nz']):.4f}%  max={np.max(overfit_log['grad_nz']):.4f}%")
    print(f"  feat_nz:  mean={np.mean(overfit_log['feat_nz']):.2f}%")
    print(f"  NaN/Inf:  {sum(overfit_log['nan_inf'])}/{n} steps")

    print("\n  OVERFIT CHECKLIST:")
    checks = {}
    checks['no_nan_inf'] = sum(overfit_log['nan_inf']) == 0 and not any_nan_inf
    checks['feat_nonzero'] = np.mean(overfit_log['feat_nz']) > 0.1
    checks['grad_nonzero'] = np.mean(overfit_log['grad_nz']) > 0
    checks['L_reid_decreasing'] = late_L_reid < early_L_reid
    checks['cos_sim_increasing'] = late_cos > early_cos

    for k, v in checks.items():
        print(f"    {'✅' if v else '❌'} {k}: {v}")

    all_pass = all(checks.values())
    print(f"\n  {'🎉 OVERFIT TEST PASSED' if all_pass else '⚠️  OVERFIT TEST FAILED'}")

    # Save log
    log_path = os.path.join(REPO_ROOT, "tools", "phase5_overfit_log.json")
    save_log = {
        'warmup': {k: [float(x) for x in v] for k, v in warmup_log.items()},
        'overfit': {k: [float(x) for x in v] for k, v in overfit_log.items()},
        'config': {
            'warmup_iters': args.warmup_iters,
            'overfit_iters': args.overfit_iters,
            'lambda_reid': args.lambda_reid,
        },
        'results': {
            'L_reid_first': float(early_L_reid),
            'L_reid_last': float(late_L_reid),
            'L_reid_decrease_pct': float(reid_decrease),
            'cos_first': float(early_cos),
            'cos_last': float(late_cos),
            'cos_increase': float(cos_increase),
        },
    }
    with open(log_path, 'w') as f:
        json.dump(save_log, f, indent=2)
    print(f"\n  Log saved: {log_path}")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
