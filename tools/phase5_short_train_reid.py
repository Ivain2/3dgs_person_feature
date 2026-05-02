#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 5 Experiment B: Short training stability test for ReID distillation.

Use normal dataloader, train 1000-5000 iters, verify ReID + RGB coexist stably.

Usage:
    python tools/phase5_short_train_reid.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --warmup_iters 300 --train_iters 2000 --lambda_reid 0.05
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
    parser.add_argument('--warmup_iters', type=int, default=300)
    parser.add_argument('--train_iters', type=int, default=2000)
    parser.add_argument('--lambda_reid', type=float, default=0.05)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5 Experiment B: Short Training Stability Test")
    print("=" * 70)
    print(f"  warmup_iters  = {args.warmup_iters}")
    print(f"  train_iters   = {args.train_iters}")
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

    train_iter = iter(trainer.train_dataloader)
    total_iters = args.warmup_iters + args.train_iters

    warmup_log = {'L_rgb': [], 'rgb_nz': [], 'feat_nz': [], 'feat_abs': []}
    train_log = {
        'L_rgb': [], 'L_reid': [], 'L_total': [], 'cos_sim': [],
        'valid_inst': [], 'grad_mean': [], 'grad_max': [], 'grad_nz': [],
        'rgb_nz': [], 'feat_nz': [], 'feat_abs': [], 'nan_inf': [],
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

        step_nan = torch.isnan(pred_rgb).any() or torch.isnan(person_feature_map).any()

        L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)

        with torch.no_grad():
            rgb_nz = (pred_rgb.abs() > 1e-6).sum().item() / pred_rgb.numel() * 100
            feat_nz = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
            feat_abs = person_feature_map.abs().mean().item()

        is_warmup = step < args.warmup_iters

        # L_reid (only in training phase)
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

        L_total = L_rgb + args.lambda_reid * L_reid if not is_warmup else L_rgb

        if not step_nan:
            L_total.backward()
            trainer.model.optimizer.step()
        else:
            any_nan_inf = True

        # Gradient stats
        pf = trainer.model.get_person_feature()
        grad_mean = 0.0
        grad_max = 0.0
        grad_nz = 0.0
        if pf.grad is not None:
            grad_mean = pf.grad.abs().mean().item()
            grad_max = pf.grad.abs().max().item()
            grad_nz = (pf.grad.abs() > 1e-8).sum().item() / pf.grad.numel() * 100

        avg_cos = np.mean(cos_sims_step) if cos_sims_step else 0

        # Log
        if is_warmup:
            warmup_log['L_rgb'].append(L_rgb.item())
            warmup_log['rgb_nz'].append(rgb_nz)
            warmup_log['feat_nz'].append(feat_nz)
            warmup_log['feat_abs'].append(feat_abs)
        else:
            train_log['L_rgb'].append(L_rgb.item())
            train_log['L_reid'].append(L_reid.item())
            train_log['L_total'].append(L_total.item())
            train_log['cos_sim'].append(avg_cos)
            train_log['valid_inst'].append(valid_count)
            train_log['grad_mean'].append(grad_mean)
            train_log['grad_max'].append(grad_max)
            train_log['grad_nz'].append(grad_nz)
            train_log['rgb_nz'].append(rgb_nz)
            train_log['feat_nz'].append(feat_nz)
            train_log['feat_abs'].append(feat_abs)
            train_log['nan_inf'].append(bool(step_nan))

        # Print
        should_print = (step < 3 or step % 100 == 0
                        or step == args.warmup_iters - 1
                        or step == args.warmup_iters
                        or step == total_iters - 1)
        if should_print:
            phase = "WARMUP" if is_warmup else "REID"
            msg = f"[{phase}] Step {step:4d}: L_rgb={L_rgb.item():.4f}"
            if not is_warmup:
                msg += f", L_reid={L_reid.item():.4f}, cos={avg_cos:.4f}, grad_nz={grad_nz:.4f}%"
            msg += f", rgb_nz={rgb_nz:.2f}%, feat_nz={feat_nz:.2f}%"
            print(msg)

    # ===== Analysis =====
    print("\n" + "=" * 70)
    print("EXPERIMENT B: SHORT TRAINING STABILITY RESULTS")
    print("=" * 70)

    # Warmup summary
    if warmup_log['L_rgb']:
        print(f"\n--- Warmup Phase ({args.warmup_iters} iters) ---")
        print(f"  L_rgb:  first10={np.mean(warmup_log['L_rgb'][:10]):.4f}  last10={np.mean(warmup_log['L_rgb'][-10:]):.4f}")
        print(f"  rgb_nz: last10={np.mean(warmup_log['rgb_nz'][-10:]):.2f}%")
        print(f"  feat_nz: last10={np.mean(warmup_log['feat_nz'][-10:]):.2f}%")

    # Training summary
    r = train_log
    if r['L_reid']:
        n = len(r['L_reid'])
        window = min(20, max(5, n // 10))

        early_L_rgb = np.mean(r['L_rgb'][:window])
        late_L_rgb = np.mean(r['L_rgb'][-window:])
        early_L_reid = np.mean(r['L_reid'][:window])
        late_L_reid = np.mean(r['L_reid'][-window:])
        early_cos = np.mean(r['cos_sim'][:window])
        late_cos = np.mean(r['cos_sim'][-window:])
        early_grad = np.mean(r['grad_mean'][:window])
        late_grad = np.mean(r['grad_mean'][-window:])

        reid_decrease = (early_L_reid - late_L_reid) / max(early_L_reid, 1e-8) * 100
        cos_increase = late_cos - early_cos

        print(f"\n--- ReID Training Phase ({n} iters) ---")
        print(f"  L_rgb:    first{window}={early_L_rgb:.4f}  last{window}={late_L_rgb:.4f}")
        print(f"  L_reid:   first{window}={early_L_reid:.4f}  last{window}={late_L_reid:.4f}  decrease={reid_decrease:.2f}%")
        print(f"  cos_sim:  first{window}={early_cos:.4f}  last{window}={late_cos:.4f}  increase={cos_increase:.4f}")
        print(f"  grad_mean: first{window}={early_grad:.6f}  last{window}={late_grad:.6f}")
        print(f"  grad_nz:  mean={np.mean(r['grad_nz']):.4f}%  max={np.max(r['grad_nz']):.4f}%")
        print(f"  feat_nz:  mean={np.mean(r['feat_nz']):.2f}%")
        print(f"  valid_inst: mean={np.mean(r['valid_inst']):.1f}")
        print(f"  NaN/Inf:  {sum(r['nan_inf'])}/{n} steps")

        # Checklist
        print("\n  STABILITY CHECKLIST:")
        checks = {}
        checks['no_nan_inf'] = sum(r['nan_inf']) == 0 and not any_nan_inf
        checks['L_rgb_stable'] = late_L_rgb < early_L_rgb * 1.5  # not crashed
        checks['feat_nonzero'] = np.mean(r['feat_nz']) > 0.1
        checks['grad_nonzero'] = np.mean(r['grad_nz']) > 0
        checks['valid_inst_gt0'] = np.mean(r['valid_inst']) > 0
        checks['L_reid_decreasing'] = late_L_reid < early_L_reid
        checks['cos_sim_increasing'] = late_cos > early_cos

        for k, v in checks.items():
            print(f"    {'✅' if v else '❌'} {k}: {v}")

        all_pass = all(checks.values())
        print(f"\n  {'🎉 SHORT TRAINING TEST PASSED' if all_pass else '⚠️  SHORT TRAINING TEST FAILED'}")

        # Save log
        log_path = os.path.join(REPO_ROOT, "tools", "phase5_short_train_log.json")
        save_log = {
            'warmup': {k: [float(x) for x in v] for k, v in warmup_log.items()},
            'train': {k: [float(x) for x in v] for k, v in r.items()},
            'config': {
                'warmup_iters': args.warmup_iters,
                'train_iters': args.train_iters,
                'lambda_reid': args.lambda_reid,
            },
            'results': {
                'L_reid_first': float(early_L_reid),
                'L_reid_last': float(late_L_reid),
                'L_reid_decrease_pct': float(reid_decrease),
                'cos_first': float(early_cos),
                'cos_last': float(late_cos),
                'cos_increase': float(cos_increase),
                'L_rgb_first': float(early_L_rgb),
                'L_rgb_last': float(late_L_rgb),
            },
        }
        with open(log_path, 'w') as f:
            json.dump(save_log, f, indent=2)
        print(f"\n  Log saved: {log_path}")

        return all_pass

    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
