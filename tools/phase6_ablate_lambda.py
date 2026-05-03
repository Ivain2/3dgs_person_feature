#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 6C: Lambda_reid ablation study.

Compare lambda_reid = 0.01, 0.05, 0.1 with short training.

Usage:
    python tools/phase6_ablate_lambda.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --warmup_iters 500 --train_iters 3000 \
        --lambdas 0.01 0.05 0.1
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


def run_single_lambda(conf, lambda_reid, warmup_iters, train_iters):
    """Run a single training with given lambda_reid."""
    conf.loss.lambda_reid = lambda_reid
    conf.loss.use_reid = True
    conf.model.person_feature_dim = 512

    trainer = Trainer3DGRUT(conf)
    train_iter = iter(trainer.train_dataloader)
    total_iters = warmup_iters + train_iters

    trainer.model.train()
    log = {
        'L_rgb': [], 'L_reid': [], 'cos_sim': [],
        'grad_nz': [], 'feat_nz': [], 'nan_inf': [],
    }
    any_nan = False

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
        if step_nan:
            any_nan = True

        L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)

        is_warmup = step < warmup_iters
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

        L_total = L_rgb + lambda_reid * L_reid if not is_warmup else L_rgb

        if not step_nan:
            L_total.backward()
            trainer.model.optimizer.step()

        with torch.no_grad():
            feat_nz = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100

        pf = trainer.model.get_person_feature()
        grad_nz = 0.0
        if pf.grad is not None:
            grad_nz = (pf.grad.abs() > 1e-8).sum().item() / pf.grad.numel() * 100

        avg_cos = np.mean(cos_sims_step) if cos_sims_step else 0

        if not is_warmup:
            log['L_rgb'].append(L_rgb.item())
            log['L_reid'].append(L_reid.item())
            log['cos_sim'].append(avg_cos)
            log['grad_nz'].append(grad_nz)
            log['feat_nz'].append(feat_nz)
            log['nan_inf'].append(bool(step_nan))

    # Compute summary
    n = len(log['L_reid'])
    if n == 0:
        return None

    window = min(20, max(5, n // 4))
    result = {
        'lambda_reid': lambda_reid,
        'L_rgb_first': float(np.mean(log['L_rgb'][:window])),
        'L_rgb_last': float(np.mean(log['L_rgb'][-window:])),
        'L_reid_first': float(np.mean(log['L_reid'][:window])),
        'L_reid_last': float(np.mean(log['L_reid'][-window:])),
        'cos_first': float(np.mean(log['cos_sim'][:window])),
        'cos_last': float(np.mean(log['cos_sim'][-window:])),
        'grad_nz_mean': float(np.mean(log['grad_nz'])),
        'feat_nz_mean': float(np.mean(log['feat_nz'])),
        'nan_count': int(sum(log['nan_inf'])),
        'stable': not any_nan and np.mean(log['L_rgb'][-window:]) < 1.0,
    }
    result['L_reid_decrease_pct'] = (result['L_reid_first'] - result['L_reid_last']) / max(result['L_reid_first'], 1e-8) * 100
    result['cos_increase'] = result['cos_last'] - result['cos_first']

    return result, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--train_iters', type=int, default=3000)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.01, 0.05, 0.1])
    parser.add_argument('--output', type=str, default='tools/phase6_ablate_log.json')
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 6C: Lambda ReID Ablation")
    print("=" * 70)
    print(f"  warmup_iters = {args.warmup_iters}")
    print(f"  train_iters  = {args.train_iters}")
    print(f"  lambdas      = {args.lambdas}")

    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)

    all_results = []
    all_logs = {}

    for lambda_val in args.lambdas:
        print(f"\n{'='*70}")
        print(f"Training with lambda_reid = {lambda_val}")
        print(f"{'='*70}")

        result, log = run_single_lambda(conf, lambda_val, args.warmup_iters, args.train_iters)

        if result is None:
            print(f"  ❌ No valid training data for lambda={lambda_val}")
            continue

        all_results.append(result)
        all_logs[str(lambda_val)] = {k: [float(x) for x in v] for k, v in log.items()}

        print(f"\n  Results for lambda={lambda_val}:")
        print(f"    L_rgb:  {result['L_rgb_first']:.4f} -> {result['L_rgb_last']:.4f}")
        print(f"    L_reid: {result['L_reid_first']:.4f} -> {result['L_reid_last']:.4f} (decrease={result['L_reid_decrease_pct']:.2f}%)")
        print(f"    cos:    {result['cos_first']:.4f} -> {result['cos_last']:.4f} (increase={result['cos_increase']:.4f})")
        print(f"    grad_nz: {result['grad_nz_mean']:.4f}%")
        print(f"    feat_nz: {result['feat_nz_mean']:.2f}%")
        print(f"    NaN:     {result['nan_count']}")
        print(f"    stable:  {'✅' if result['stable'] else '❌'}")

    # ===== Summary Table =====
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'lambda':>8} | {'L_rgb':>8} | {'L_reid':>8} | {'cos_tch':>8} | {'cos_inc':>8} | {'gap':>8} | {'stable':>7}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['lambda_reid']:8.2f} | {r['L_rgb_last']:8.4f} | {r['L_reid_last']:8.4f} | "
              f"{r['cos_last']:8.4f} | {r['cos_increase']:8.4f} | "
              f"{r['L_reid_decrease_pct']:7.2f}% | {'✅' if r['stable'] else '❌':>7}")

    # Save log
    log_path = os.path.join(REPO_ROOT, args.output)
    with open(log_path, 'w') as f:
        json.dump({'results': all_results, 'logs': all_logs}, f, indent=2)
    print(f"\nLog saved: {log_path}")

    return True


if __name__ == "__main__":
    main()
