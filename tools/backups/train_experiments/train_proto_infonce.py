#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 11A: Prototype InfoNCE ReID training.

L_reid = 1.0 * L_proto + 0.1 * L_teacher
L_proto = CE((f @ prototypes.T) / tau, person_id)
L_teacher = 1 - cos(f, teacher_feature)

Usage:
    PYTHONPATH=. python tools/train_phase11A_proto_infonce.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --experiment_name phase11A_proto_infonce_opacity_lam005_lr1e4 \
        --pooling opacity --lambda_reid 0.05 --person_feature_lr 1e-4 \
        --loss_type prototype_infonce --tau 0.07 --teacher_loss_weight 0.1 \
        --warmup_iters 1000 --train_iters 20000 --save_every 2000
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
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--train_iters', type=int, default=10000)
    parser.add_argument('--lambda_reid', type=float, default=0.05)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--ckpt_dir', type=str, default='runs/phase11A')
    parser.add_argument('--log_path', type=str, default='tools/phase11A_log.json')
    parser.add_argument('--pooling', type=str, default='opacity',
                        choices=['mean', 'opacity', 'topk_opacity'])
    parser.add_argument('--topk_ratio', type=float, default=0.3)
    parser.add_argument('--min_alpha_sum', type=float, default=0.01)
    parser.add_argument('--prototype_path', type=str,
                        default='/data02/zhangrunxiang/data/Wildtrack/reid_teacher_prototypes.pt')
    parser.add_argument('--detach_reid_geometry', action='store_true')
    parser.add_argument('--detach_opacity_weight', action='store_true', default=True)
    parser.add_argument('--no_detach_opacity_weight', action='store_false', dest='detach_opacity_weight')
    parser.add_argument('--person_feature_lr', type=float, default=1e-4)
    parser.add_argument('--loss_type', type=str, default='prototype_infonce',
                        choices=['prototype_infonce', 'prototype_infonce_warm'])
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--teacher_loss_weight', type=float, default=0.1)
    parser.add_argument('--experiment_name', type=str, default='phase11A')
    parser.add_argument('--cosine_warmup_iters', type=int, default=500)
    parser.add_argument('--proto_loss_weight_start', type=float, default=0.0)
    parser.add_argument('--proto_loss_weight_end', type=float, default=0.5)
    args = parser.parse_args()

    if args.experiment_name:
        args.ckpt_dir = f'runs/{args.experiment_name}'
        args.log_path = f'tools/{args.experiment_name}_log.json'

    print("=" * 70)
    print("Phase 11A: Prototype InfoNCE ReID Training")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:30s} = {v}")

    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)

    conf.model.person_feature_dim = 512
    conf.model.person_feature_lr = args.person_feature_lr
    conf.loss.use_reid = True
    conf.loss.lambda_reid = args.lambda_reid

    print("\nInitializing trainer...")
    trainer = Trainer3DGRUT(conf)
    print(f"Gaussians: {trainer.model.num_gaussians}")

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded from step {ckpt.get('step', 'unknown')}")

    if args.detach_reid_geometry:
        trainer.model.positions.requires_grad_(False)
        trainer.model.rotation.requires_grad_(False)
        trainer.model.scale.requires_grad_(False)
        trainer.model.density.requires_grad_(False)
        trainer.model.features_albedo.requires_grad_(False)
        trainer.model.features_specular.requires_grad_(False)
        trainer.model._person_feature.requires_grad_(True)
        print("[Detach Geometry] ReID loss only updates person_feature")

    proto_data = torch.load(args.prototype_path, map_location=trainer.device, weights_only=False)
    prototypes = proto_data['prototypes'].to(trainer.device)
    valid_mask = proto_data['valid_mask'].to(trainer.device)
    print(f"\nLoaded prototypes: {prototypes.shape}, valid IDs: {valid_mask.sum().item()}")

    train_iter = iter(trainer.train_dataloader)
    total_iters = args.warmup_iters + args.train_iters

    warmup_log = {'L_rgb': [], 'feat_nz': []}
    train_log = {
        'L_rgb': [], 'L_proto': [], 'L_teacher': [], 'L_reid_total': [],
        'cos_fv_tv': [], 'cos_fv_Pi': [],
        'valid_inst': [], 'skipped_low_alpha': [],
        'alpha_sum_mean': [], 'alpha_mean': [], 'alpha_max': [],
        'raw_feature_norm_mean': [], 'weighted_feature_norm_mean': [],
        'grad_mean': [], 'grad_max': [], 'grad_nz': [],
        'feat_nz': [], 'feat_abs': [],
        'nan_inf': [], 'iter_time': [],
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
        person_opacity_map = render_out.get('person_opacity_map')

        step_nan = torch.isnan(pred_rgb).any() or torch.isnan(person_feature_map).any()

        L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)

        with torch.no_grad():
            feat_nz = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
            feat_abs = person_feature_map.abs().mean().item()

        is_warmup = step < args.warmup_iters

        L_proto = torch.zeros(1, device=trainer.device)
        L_teacher = torch.zeros(1, device=trainer.device)
        L_reid = torch.zeros(1, device=trainer.device)
        cos_fv_tv_list = []
        cos_fv_Pi_list = []
        skipped_low_alpha = 0
        alpha_sum_list = []
        alpha_mean_list = []
        alpha_max_list = []
        raw_norm_list = []
        weighted_norm_list = []
        valid_inst_count = 0

        if not is_warmup and valid_count > 0:
            proto_infonce_logits = []
            proto_infonce_labels = []
            teacher_cos_list = []

            reid_step = step - args.warmup_iters

            for inst in gpu_batch.instances:
                if not inst.get('valid', False):
                    continue
                teacher_emb = inst.get('teacher_embedding')
                if teacher_emb is None:
                    continue

                bbox = inst['bbox_xyxy']
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)

                f_v, pool_stats = roi_pool(
                    person_feature_map, bbox_t,
                    opacity_map=person_opacity_map,
                    pooling=args.pooling,
                    topk_ratio=args.topk_ratio,
                    min_alpha_sum=args.min_alpha_sum,
                    detach_opacity_weight=args.detach_opacity_weight,
                )

                if f_v is None:
                    skipped_low_alpha += 1
                    continue

                valid_inst_count += 1

                if not pool_stats.get('skipped', False):
                    alpha_sum_list.append(pool_stats.get('alpha_sum', 0))
                    alpha_mean_list.append(pool_stats.get('alpha_mean', 0))
                    alpha_max_list.append(pool_stats.get('alpha_max', 0))
                    raw_norm_list.append(pool_stats.get('raw_feature_norm_mean', 0))
                    weighted_norm_list.append(pool_stats.get('weighted_feature_norm_mean', 0))

                t_v = torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                t_v_norm = F.normalize(t_v, p=2, dim=0, eps=1e-6)

                cos_view = torch.dot(f_v, t_v_norm)
                cos_fv_tv_list.append(cos_view.item())

                train_id = inst.get('train_id', -1)

                if args.loss_type == 'prototype_infonce' and prototypes is not None:
                    if train_id >= 0 and train_id < valid_mask.shape[0] and valid_mask[train_id]:
                        P_i = prototypes[train_id].squeeze()
                        cos_proto = torch.dot(f_v, P_i)
                        cos_fv_Pi_list.append(cos_proto.item())
                        proto_infonce_logits.append(f_v)
                        proto_infonce_labels.append(train_id)
                        teacher_cos_list.append(cos_view)
                    else:
                        teacher_cos_list.append(cos_view)
                elif args.loss_type == 'prototype_infonce_warm':
                    if reid_step < args.cosine_warmup_iters:
                        teacher_cos_list.append(cos_view)
                    else:
                        P_i = prototypes[train_id].squeeze()
                        cos_proto = torch.dot(f_v, P_i)
                        cos_fv_Pi_list.append(cos_proto.item())
                        proto_infonce_logits.append(f_v)
                        proto_infonce_labels.append(train_id)
                        teacher_cos_list.append(cos_view)

            if args.loss_type == 'prototype_infonce' and proto_infonce_logits:
                f_stack = torch.stack(proto_infonce_logits)  # [N, D]
                logits = f_stack @ prototypes.T / args.tau   # [N, n_protos]
                labels = torch.tensor(proto_infonce_labels, device=trainer.device)
                L_proto = F.cross_entropy(logits, labels)
                if teacher_cos_list:
                    L_teacher = 1 - torch.tensor(teacher_cos_list, device=trainer.device).mean()
                L_reid = 1.0 * L_proto + args.teacher_loss_weight * L_teacher
            elif args.loss_type == 'prototype_infonce_warm':
                if reid_step < args.cosine_warmup_iters:
                    L_teacher = 1 - torch.tensor(teacher_cos_list, device=trainer.device).mean()
                    L_proto = torch.zeros(1, device=trainer.device)
                    L_reid = L_teacher
                else:
                    warm_progress = min(1.0, (reid_step - args.cosine_warmup_iters) / args.train_iters)
                    proto_weight = args.proto_loss_weight_start + warm_progress * (args.proto_loss_weight_end - args.proto_loss_weight_start)
                    
                    if proto_infonce_logits:
                        f_stack = torch.stack(proto_infonce_logits)
                        logits = f_stack @ prototypes.T / args.tau
                        labels = torch.tensor(proto_infonce_labels, device=trainer.device)
                        L_proto = F.cross_entropy(logits, labels)
                    if teacher_cos_list:
                        L_teacher = 1 - torch.tensor(teacher_cos_list, device=trainer.device).mean()
                    L_reid = proto_weight * L_proto + args.teacher_loss_weight * L_teacher

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

        avg_cos_tv = np.mean(cos_fv_tv_list) if cos_fv_tv_list else 0
        avg_cos_Pi = np.mean(cos_fv_Pi_list) if cos_fv_Pi_list else 0
        iter_time = time.time() - t_iter_start

        if is_warmup:
            warmup_log['L_rgb'].append(L_rgb.item())
            warmup_log['feat_nz'].append(feat_nz)
        else:
            train_log['L_rgb'].append(L_rgb.item())
            train_log['L_proto'].append(L_proto.item())
            train_log['L_teacher'].append(L_teacher.item())
            train_log['L_reid_total'].append(L_reid.item())
            train_log['cos_fv_tv'].append(avg_cos_tv)
            train_log['cos_fv_Pi'].append(avg_cos_Pi)
            train_log['valid_inst'].append(valid_inst_count)
            train_log['skipped_low_alpha'].append(skipped_low_alpha)
            train_log['alpha_sum_mean'].append(np.mean(alpha_sum_list) if alpha_sum_list else 0)
            train_log['alpha_mean'].append(np.mean(alpha_mean_list) if alpha_mean_list else 0)
            train_log['alpha_max'].append(np.mean(alpha_max_list) if alpha_max_list else 0)
            train_log['raw_feature_norm_mean'].append(np.mean(raw_norm_list) if raw_norm_list else 0)
            train_log['weighted_feature_norm_mean'].append(np.mean(weighted_norm_list) if weighted_norm_list else 0)
            train_log['grad_mean'].append(grad_mean)
            train_log['grad_max'].append(grad_max)
            train_log['grad_nz'].append(grad_nz)
            train_log['feat_nz'].append(feat_nz)
            train_log['feat_abs'].append(feat_abs)
            train_log['nan_inf'].append(bool(step_nan))
            train_log['iter_time'].append(iter_time)

        should_print = (step < 3 or step % 200 == 0
                        or step == args.warmup_iters - 1
                        or step == args.warmup_iters
                        or step == total_iters - 1)
        if should_print:
            phase = "WARMUP" if is_warmup else "REID-11A"
            msg = f"[{phase}] Step {step:5d}: L_rgb={L_rgb.item():.4f}"
            if not is_warmup:
                msg += (f", L_proto={L_proto.item():.4f}, L_teacher={L_teacher.item():.4f}, "
                        f"L_reid={L_reid.item():.4f}, cos_tv={avg_cos_tv:.4f}, "
                        f"cos_Pi={avg_cos_Pi:.4f}, valid={valid_inst_count}, "
                        f"skip_alpha={skipped_low_alpha}, "
                        f"alpha_sum={np.mean(alpha_sum_list) if alpha_sum_list else 0:.2f}, "
                        f"raw_norm={np.mean(raw_norm_list) if raw_norm_list else 0:.4f}, "
                        f"grad_nz={grad_nz:.4f}%")
            msg += f", feat_nz={feat_nz:.2f}%, t={iter_time:.2f}s"
            print(msg)

        reid_step = step - args.warmup_iters
        if not is_warmup and reid_step > 0 and reid_step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"step_{step}.pth")
            save_checkpoint(trainer.model, trainer.model.optimizer, step, ckpt_path)

    final_path = os.path.join(args.ckpt_dir, "latest.pth")
    save_checkpoint(trainer.model, trainer.model.optimizer, total_iters - 1, final_path)
    final_step_path = os.path.join(args.ckpt_dir, f"step_{total_iters - 1}.pth")
    save_checkpoint(trainer.model, trainer.model.optimizer, total_iters - 1, final_step_path)

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")

    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)

    if warmup_log['L_rgb']:
        print(f"\n--- Warmup ({args.warmup_iters} iters) ---")
        print(f"  L_rgb: first10={np.mean(warmup_log['L_rgb'][:10]):.4f}  last10={np.mean(warmup_log['L_rgb'][-10:]):.4f}")
        print(f"  feat_nz: last10={np.mean(warmup_log['feat_nz'][-10:]):.2f}%")

    r = train_log
    if r['L_reid_total']:
        n = len(r['L_reid_total'])
        window = min(50, max(10, n // 10))

        def w(k): return np.mean(r[k][:window]), np.mean(r[k][-window:])

        early_L_rgb, late_L_rgb = w('L_rgb')
        early_L_proto, late_L_proto = w('L_proto')
        early_L_teacher, late_L_teacher = w('L_teacher')
        early_L_reid, late_L_reid = w('L_reid_total')
        early_cos_tv, late_cos_tv = w('cos_fv_tv')
        early_cos_Pi, late_cos_Pi = w('cos_fv_Pi')

        print(f"\n--- ReID Training ({n} iters, pooling={args.pooling}) ---")
        print(f"  L_rgb:         first{window}={early_L_rgb:.4f}  last{window}={late_L_rgb:.4f}")
        print(f"  L_proto:       first{window}={early_L_proto:.4f}  last{window}={late_L_proto:.4f}")
        print(f"  L_teacher:     first{window}={early_L_teacher:.4f}  last{window}={late_L_teacher:.4f}")
        print(f"  L_reid:        first{window}={early_L_reid:.4f}  last{window}={late_L_reid:.4f}")
        print(f"  cos(f_v,t_v):  first{window}={early_cos_tv:.4f}  last{window}={late_cos_tv:.4f}")
        print(f"  cos(f_v,P_i):  first{window}={early_cos_Pi:.4f}  last{window}={late_cos_Pi:.4f}")
        print(f"  grad_nz:       mean={np.mean(r['grad_nz']):.4f}%")
        print(f"  NaN/Inf:       {sum(r['nan_inf'])}/{n}")

    log_path = os.path.join(REPO_ROOT, args.log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    save_data = {
        'warmup': {k: [float(x) for x in v] for k, v in warmup_log.items()},
        'train': {k: [float(x) for x in v] for k, v in r.items()},
        'config': {k: v for k, v in vars(args).items()},
        'config_total_time_s': total_time,
    }
    with open(log_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nLog saved: {log_path}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
