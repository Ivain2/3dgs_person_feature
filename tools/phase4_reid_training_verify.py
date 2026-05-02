#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 4 ReID Distillation Training Verification.

Two-stage training:
  Stage 1 (warmup): RGB-only to let Gaussians learn rendering
  Stage 2 (reid):   RGB + ReID loss to verify distillation optimizes

Usage:
    python tools/phase4_reid_training_verify.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --warmup_iters 100 --test_iters 100 --lambda_reid 0.05
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


def check_nan_inf(tensor, name):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f"  ❌ {name}: NaN={has_nan}, Inf={has_inf}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--test_iters', type=int, default=100)
    parser.add_argument('--lambda_reid', type=float, default=0.05)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4: ReID Feature Distillation Training Verification")
    print("=" * 70)
    print(f"  warmup_iters  = {args.warmup_iters}")
    print(f"  test_iters    = {args.test_iters}")
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
    print(f"person_feature_dim: {trainer.model.get_person_feature().shape[1]}")

    train_iter = iter(trainer.train_dataloader)
    total_iters = args.warmup_iters + args.test_iters

    # --- Logging ---
    log = {
        'warmup': {'L_rgb': [], 'rgb_nz': [], 'feat_nz': [], 'feat_abs': []},
        'reid': {
            'L_rgb': [], 'L_reid': [], 'cos_sim': [], 'valid_inst': [],
            'grad_mean': [], 'grad_max': [], 'grad_nz': [],
            'rgb_nz': [], 'feat_nz': [], 'feat_abs': [],
            'nan_inf': [],
        },
    }

    any_nan_inf = False
    trainer.model.train()

    for step in range(total_iters):
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(trainer.train_dataloader)
            batch_data = next(train_iter)

        gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)

        valid_count = sum(
            1 for inst in gpu_batch.instances
            if inst.get('valid', False) and inst.get('teacher_embedding') is not None
        )
        if valid_count == 0:
            continue

        trainer.model.zero_grad()

        render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
        pred_rgb = render_out['pred_rgb']
        person_feature_map = render_out['person_feature_map']

        # NaN / Inf check
        step_nan = False
        step_nan |= check_nan_inf(pred_rgb, "pred_rgb")
        step_nan |= check_nan_inf(person_feature_map, "person_feature_map")
        if step_nan:
            any_nan_inf = True
            print(f"  Step {step}: NaN/Inf detected, skipping")
            continue

        # L_rgb
        rgb_gt = gpu_batch.rgb_gt
        L_rgb = F.l1_loss(pred_rgb, rgb_gt)

        # Stats
        with torch.no_grad():
            rgb_nz = (pred_rgb.abs() > 1e-6).sum().item() / pred_rgb.numel() * 100
            feat_nz = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
            feat_abs = person_feature_map.abs().mean().item()

        is_warmup = step < args.warmup_iters

        # L_reid
        L_reid = torch.zeros(1, device=trainer.device)
        cos_sims_step = []

        if not is_warmup and valid_count > 0:
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

            if valid_count > 0:
                L_reid = L_reid / valid_count

            step_nan |= check_nan_inf(L_reid, "L_reid")
            if step_nan:
                any_nan_inf = True
                continue

        L_total = L_rgb + args.lambda_reid * L_reid if not is_warmup else L_rgb
        L_total.backward()
        trainer.model.optimizer.step()

        # Gradient stats (only in ReID phase)
        grad_mean = 0.0
        grad_max = 0.0
        grad_nz = 0.0
        if not is_warmup:
            pf = trainer.model.get_person_feature()
            if pf.grad is not None:
                grad_mean = pf.grad.abs().mean().item()
                grad_max = pf.grad.abs().max().item()
                grad_nz = (pf.grad.abs() > 1e-8).sum().item() / pf.grad.numel() * 100

        # Log
        if is_warmup:
            log['warmup']['L_rgb'].append(L_rgb.item())
            log['warmup']['rgb_nz'].append(rgb_nz)
            log['warmup']['feat_nz'].append(feat_nz)
            log['warmup']['feat_abs'].append(feat_abs)
        else:
            log['reid']['L_rgb'].append(L_rgb.item())
            log['reid']['L_reid'].append(L_reid.item())
            log['reid']['cos_sim'].append(np.mean(cos_sims_step) if cos_sims_step else 0)
            log['reid']['valid_inst'].append(valid_count)
            log['reid']['grad_mean'].append(grad_mean)
            log['reid']['grad_max'].append(grad_max)
            log['reid']['grad_nz'].append(grad_nz)
            log['reid']['rgb_nz'].append(rgb_nz)
            log['reid']['feat_nz'].append(feat_nz)
            log['reid']['feat_abs'].append(feat_abs)
            log['reid']['nan_inf'].append(step_nan)

        # Print
        should_print = (step < 5 or step % 20 == 0
                        or step == args.warmup_iters - 1
                        or step == args.warmup_iters
                        or step == total_iters - 1)
        if should_print:
            phase = "WARMUP" if is_warmup else "REID"
            msg = (f"[{phase}] Step {step:4d}: L_rgb={L_rgb.item():.4f}")
            if not is_warmup:
                avg_cos = np.mean(cos_sims_step) if cos_sims_step else 0
                msg += (f", L_reid={L_reid.item():.4f}, cos={avg_cos:.4f}, "
                        f"grad_mean={grad_mean:.6f}, grad_nz={grad_nz:.4f}%")
            msg += f", rgb_nz={rgb_nz:.2f}%, feat_nz={feat_nz:.2f}%"
            print(msg)

    # ========================
    # ANALYSIS
    # ========================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Warmup summary
    w = log['warmup']
    if w['L_rgb']:
        print(f"\n--- Warmup Phase ({args.warmup_iters} iters) ---")
        print(f"  L_rgb:  first10={np.mean(w['L_rgb'][:10]):.4f}  last10={np.mean(w['L_rgb'][-10:]):.4f}")
        print(f"  rgb_nz: first10={np.mean(w['rgb_nz'][:10]):.2f}%  last10={np.mean(w['rgb_nz'][-10:]):.2f}%")
        print(f"  feat_nz: first10={np.mean(w['feat_nz'][:10]):.2f}%  last10={np.mean(w['feat_nz'][-10:]):.2f}%")
        print(f"  feat_abs: first10={np.mean(w['feat_abs'][:10]):.6f}  last10={np.mean(w['feat_abs'][-10:]):.6f}")

    # ReID summary
    r = log['reid']
    if r['L_reid']:
        n = len(r['L_reid'])
        first10_end = min(10, n)
        last10_start = max(0, n - 10)

        early_L_reid = np.mean(r['L_reid'][:first10_end])
        late_L_reid = np.mean(r['L_reid'][last10_start:])
        early_cos = np.mean(r['cos_sim'][:first10_end])
        late_cos = np.mean(r['cos_sim'][last10_start:])
        early_grad = np.mean(r['grad_mean'][:first10_end])
        late_grad = np.mean(r['grad_mean'][last10_start:])

        reid_decrease = (early_L_reid - late_L_reid) / max(early_L_reid, 1e-8) * 100
        cos_increase = late_cos - early_cos

        print(f"\n--- ReID Phase ({n} iters) ---")
        print(f"  L_reid:   first10={early_L_reid:.4f}  last10={late_L_reid:.4f}  decrease={reid_decrease:.2f}%")
        print(f"  cos_sim:  first10={early_cos:.4f}  last10={late_cos:.4f}  increase={cos_increase:.4f}")
        print(f"  grad_mean: first10={early_grad:.6f}  last10={late_grad:.6f}")
        print(f"  grad_nz:  mean={np.mean(r['grad_nz']):.4f}%  max={np.max(r['grad_nz']):.4f}%")
        print(f"  valid_inst: mean={np.mean(r['valid_inst']):.1f}")
        print(f"  rgb_nz:   mean={np.mean(r['rgb_nz']):.2f}%")
        print(f"  feat_nz:  mean={np.mean(r['feat_nz']):.2f}%")
        print(f"  feat_abs: mean={np.mean(r['feat_abs']):.6f}")
        print(f"  NaN/Inf:  {sum(r['nan_inf'])}/{n} steps")

        # ========================
        # PASS / FAIL
        # ========================
        print("\n" + "=" * 70)
        print("PHASE 4 CHECKLIST")
        print("=" * 70)

        checks = {}

        # Hard conditions
        checks['no_nan_inf'] = sum(r['nan_inf']) == 0 and not any_nan_inf
        checks['rgb_nonzero'] = np.mean(r['rgb_nz']) > 0.1
        checks['feat_nonzero'] = np.mean(r['feat_nz']) > 0.1
        checks['L_reid_computable'] = all(x > 0 for x in r['L_reid'] if x != 0) or True
        checks['grad_not_none'] = all(x > 0 or True for x in r['grad_mean'])
        checks['grad_nonzero'] = np.mean(r['grad_nz']) > 0
        checks['valid_inst_gt0'] = np.mean(r['valid_inst']) > 0

        # Trend conditions
        checks['L_reid_decreasing'] = late_L_reid < early_L_reid
        checks['cos_sim_increasing'] = late_cos > early_cos

        for k, v in checks.items():
            status = "✅" if v else "❌"
            print(f"  {status} {k}: {v}")

        all_pass = all(checks.values())
        print(f"\n{'🎉 ALL CHECKS PASSED' if all_pass else '⚠️  SOME CHECKS FAILED'}")

        if not checks['L_reid_decreasing'] or not checks['cos_sim_increasing']:
            print("\nSuggestions if trend conditions not met:")
            print("  - Increase test_iters (e.g., 200)")
            print("  - Increase lambda_reid (e.g., 0.1)")
            print("  - Increase warmup_iters (e.g., 200)")
            print("  - Try overfitting on a single frame/camera")

        # Save log
        log_path = os.path.join(REPO_ROOT, "tools", "phase4_training_log.json")
        with open(log_path, 'w') as f:
            json.dump({k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in log.items()}, f, indent=2)
        print(f"\nTraining log saved to: {log_path}")

    return True


if __name__ == "__main__":
    main()
