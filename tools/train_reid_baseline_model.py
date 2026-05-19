#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 6A: Medium-length ReID distillation training with checkpoint saving.

Usage:
    python tools/phase6_train_reid.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --warmup_iters 1000 --train_iters 5000 \
        --lambda_reid 0.05 --save_every 1000 \
        --ckpt_dir runs/phase6_reid_main
"""

import argparse
import os
import sys
import json
import time

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.roi_pooling import roi_pool


def save_checkpoint(model, optimizer, step, path):
    state = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--train_iters', type=int, default=5000)
    parser.add_argument('--lambda_reid', type=float, default=0.05)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_path', type=str, default='tools/phase6_train_log.json')
    parser.add_argument('--ckpt_dir', type=str, default='runs/phase6_reid_main')
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 6A: Medium-length ReID Training")
    print("=" * 70)
    print(f"  warmup_iters  = {args.warmup_iters}")
    print(f"  train_iters   = {args.train_iters}")
    print(f"  lambda_reid   = {args.lambda_reid}")
    print(f"  save_every    = {args.save_every}")
    print(f"  ckpt_dir      = {args.ckpt_dir}")

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

    warmup_log = {'L_rgb': [], 'rgb_nz': [], 'feat_nz': []}
    train_log = {
        'L_rgb': [], 'L_reid': [], 'L_total': [], 'cos_sim': [],
        'valid_inst': [], 'grad_mean': [], 'grad_max': [], 'grad_nz': [],
        'rgb_nz': [], 'feat_nz': [], 'feat_abs': [], 'nan_inf': [],
        'iter_time': [],
    }

    any_nan_inf = False
    trainer.model.train()
    t_start = time.time()

    for step in range(total_iters):
        t_iter_start = time.time()

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

        pf = trainer.model.get_person_feature()
        grad_mean = grad_max = grad_nz = 0.0
        if pf.grad is not None:
            grad_mean = pf.grad.abs().mean().item()
            grad_max = pf.grad.abs().max().item()
            grad_nz = (pf.grad.abs() > 1e-8).sum().item() / pf.grad.numel() * 100

        avg_cos = np.mean(cos_sims_step) if cos_sims_step else 0
        iter_time = time.time() - t_iter_start

        if is_warmup:
            warmup_log['L_rgb'].append(L_rgb.item())
            warmup_log['rgb_nz'].append(rgb_nz)
            warmup_log['feat_nz'].append(feat_nz)
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
            train_log['iter_time'].append(iter_time)

        # Print
        should_print = (step < 3 or step % 200 == 0
                        or step == args.warmup_iters - 1
                        or step == args.warmup_iters
                        or step == total_iters - 1)
        if should_print:
            phase = "WARMUP" if is_warmup else "REID"
            msg = f"[{phase}] Step {step:5d}: L_rgb={L_rgb.item():.4f}"
            if not is_warmup:
                msg += f", L_reid={L_reid.item():.4f}, cos={avg_cos:.4f}, grad_nz={grad_nz:.4f}%"
            msg += f", feat_nz={feat_nz:.2f}%, t={iter_time:.2f}s"
            print(msg)

        # Checkpoint
        reid_step = step - args.warmup_iters
        if not is_warmup and reid_step > 0 and reid_step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"step_{step}.pth")
            save_checkpoint(trainer.model, trainer.model.optimizer, step, ckpt_path)

    # Save final checkpoint
    final_path = os.path.join(args.ckpt_dir, "latest.pth")
    save_checkpoint(trainer.model, trainer.model.optimizer, total_iters - 1, final_path)

    # Also save as step number
    final_step_path = os.path.join(args.ckpt_dir, f"step_{total_iters - 1}.pth")
    save_checkpoint(trainer.model, trainer.model.optimizer, total_iters - 1, final_step_path)

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # ===== Analysis =====
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)

    if warmup_log['L_rgb']:
        print(f"\n--- Warmup ({args.warmup_iters} iters) ---")
        print(f"  L_rgb: first10={np.mean(warmup_log['L_rgb'][:10]):.4f}  last10={np.mean(warmup_log['L_rgb'][-10:]):.4f}")
        print(f"  feat_nz: last10={np.mean(warmup_log['feat_nz'][-10:]):.2f}%")

    r = train_log
    if r['L_reid']:
        n = len(r['L_reid'])
        window = min(50, max(10, n // 10))

        early_L_rgb = np.mean(r['L_rgb'][:window])
        late_L_rgb = np.mean(r['L_rgb'][-window:])
        early_L_reid = np.mean(r['L_reid'][:window])
        late_L_reid = np.mean(r['L_reid'][-window:])
        early_cos = np.mean(r['cos_sim'][:window])
        late_cos = np.mean(r['cos_sim'][-window:])

        reid_decrease = (early_L_reid - late_L_reid) / max(early_L_reid, 1e-8) * 100
        cos_increase = late_cos - early_cos

        print(f"\n--- ReID Training ({n} iters) ---")
        print(f"  L_rgb:    first{window}={early_L_rgb:.4f}  last{window}={late_L_rgb:.4f}")
        print(f"  L_reid:   first{window}={early_L_reid:.4f}  last{window}={late_L_reid:.4f}  decrease={reid_decrease:.2f}%")
        print(f"  cos_sim:  first{window}={early_cos:.4f}  last{window}={late_cos:.4f}  increase={cos_increase:.4f}")
        print(f"  grad_nz:  mean={np.mean(r['grad_nz']):.4f}%  max={np.max(r['grad_nz']):.4f}%")
        print(f"  feat_nz:  mean={np.mean(r['feat_nz']):.2f}%")
        print(f"  iter_time: mean={np.mean(r['iter_time']):.2f}s")
        print(f"  NaN/Inf:  {sum(r['nan_inf'])}/{n}")

    # Save log
    log_path = os.path.join(REPO_ROOT, args.log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    save_data = {
        'warmup': {k: [float(x) for x in v] for k, v in warmup_log.items()},
        'train': {k: [float(x) for x in v] for k, v in r.items()},
        'config': {
            'warmup_iters': args.warmup_iters,
            'train_iters': args.train_iters,
            'lambda_reid': args.lambda_reid,
            'total_time_s': total_time,
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
        json.dump(save_data, f, indent=2)
    print(f"\nLog saved: {log_path}")

    return True


if __name__ == "__main__":
    main()
