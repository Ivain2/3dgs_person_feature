#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 9: ReID training with improved methods.

Variant A: Freeze geometry, train only person_feature
Variant B: Add contrastive loss for better feature discrimination

Usage (9A - Freeze Geometry):
    PYTHONPATH=/data02/zhangrunxiang/3dgrut:$PYTHONPATH python tools/phase9_train.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --checkpoint runs/phase6_reid_main/latest.pth \
        --freeze_geometry --train_iters 10000 --lambda_reid 0.05 \
        --save_every 2000 --ckpt_dir runs/phase9A --log_path tools/phase9A_log.json

Usage (9B - Contrastive Loss):
    PYTHONPATH=/data02/zhangrunxiang/3dgrut:$PYTHONPATH python tools/phase9_train.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --checkpoint runs/phase6_reid_main/latest.pth \
        --use_contrastive_loss --train_iters 10000 --lambda_reid 0.05 \
        --save_every 2000 --ckpt_dir runs/phase9B --log_path tools/phase9B_log.json
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


def freeze_geometry(trainer):
    """Freeze all geometry-related parameters, keep person_feature trainable."""
    model = trainer.model
    model.positions.requires_grad_(False)
    model.rotation.requires_grad_(False)
    model.scale.requires_grad_(False)
    model.density.requires_grad_(False)
    model.features_albedo.requires_grad_(False)
    model.features_specular.requires_grad_(False)
    model._person_feature.requires_grad_(True)
    print("[Freeze Geometry] Frozen: positions, rotation, scale, density, features_albedo, features_specular")
    print("[Freeze Geometry] Trainable: person_feature only")


def setup_optimizer_after_freeze(model):
    """Rebuild optimizer with only trainable parameters."""
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            lr = model.conf.model.get("person_feature_lr", 1e-4) if "person_feature" in name else 1e-4
            params.append({'params': [param], 'name': name, 'lr': lr})
    optimizer = torch.optim.Adam(params, lr=0)
    return optimizer


def compute_contrastive_loss(f_v_list, t_v_list, temperature=0.07):
    """
    InfoNCE contrastive loss.

    For each student feature f_v in a batch, push it toward its teacher t_v
    and away from other teachers in the same batch.

    L = -log( exp(sim(f_i, t_i)/tau) / sum_j exp(sim(f_i, t_j)/tau) )
    """
    if len(f_v_list) < 2:
        return torch.zeros(1, device=f_v_list[0].device)

    f_batch = torch.stack(f_v_list)   # [N, D]
    t_batch = torch.stack(t_v_list)   # [M, D]

    sim_matrix = f_batch @ t_batch.T / temperature   # [N, M]
    labels = torch.arange(sim_matrix.shape[0], device=f_batch.device)
    labels = labels[:min(sim_matrix.shape[0], sim_matrix.shape[1])]

    if sim_matrix.shape[0] != sim_matrix.shape[1]:
        sim_matrix = sim_matrix[:min(sim_matrix.shape[0], sim_matrix.shape[1]),
                                 :min(sim_matrix.shape[0], sim_matrix.shape[1])]

    loss = F.cross_entropy(sim_matrix, labels[:sim_matrix.shape[0]])
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--train_iters', type=int, default=5000)
    parser.add_argument('--lambda_reid', type=float, default=0.05)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--ckpt_dir', type=str, default='runs/phase9')
    parser.add_argument('--log_path', type=str, default='tools/phase9_log.json')
    parser.add_argument('--freeze_geometry', action='store_true',
                        help='Freeze all geometry parameters, only train person_feature')
    parser.add_argument('--use_contrastive_loss', action='store_true',
                        help='Use InfoNCE contrastive loss instead of cosine distillation')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--alpha_proto', type=float, default=0.3,
                        help='Weight for prototype loss (used with contrastive loss)')
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 9: ReID Training with Improved Methods")
    print("=" * 70)
    print(f"  config              = {args.config}")
    print(f"  checkpoint          = {args.checkpoint}")
    print(f"  warmup_iters        = {args.warmup_iters}")
    print(f"  train_iters         = {args.train_iters}")
    print(f"  lambda_reid         = {args.lambda_reid}")
    print(f"  freeze_geometry     = {args.freeze_geometry}")
    print(f"  use_contrastive_loss = {args.use_contrastive_loss}")
    print(f"  temperature         = {args.temperature}")
    print(f"  alpha_proto         = {args.alpha_proto}")
    print(f"  save_every          = {args.save_every}")
    print(f"  ckpt_dir            = {args.ckpt_dir}")

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

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded from step {ckpt.get('step', 'unknown')}")

    if args.freeze_geometry:
        freeze_geometry(trainer)
        trainer.model.optimizer = setup_optimizer_after_freeze(trainer.model)
        print(f"  Optimizer rebuilt with {sum(1 for p in trainer.model.parameters() if p.requires_grad)} parameter groups")

    train_iter = iter(trainer.train_dataloader)
    total_iters = args.warmup_iters + args.train_iters

    warmup_log = {'L_rgb': [], 'rgb_nz': [], 'feat_nz': []}
    train_log = {
        'L_rgb': [], 'L_view': [], 'L_proto': [], 'L_contrastive': [], 'L_reid_total': [],
        'cos_fv_tv': [], 'cos_fv_Pi': [],
        'valid_inst': [], 'valid_proto_inst': [],
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

        step_nan = torch.isnan(pred_rgb).any() or torch.isnan(person_feature_map).any()

        L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)

        with torch.no_grad():
            rgb_nz = (pred_rgb.abs() > 1e-6).sum().item() / pred_rgb.numel() * 100
            feat_nz = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
            feat_abs = person_feature_map.abs().mean().item()

        # In freeze_geometry mode, skip warmup entirely since RGB loss has no trainable params
        is_warmup = step < args.warmup_iters and not args.freeze_geometry

        L_view = torch.zeros(1, device=trainer.device)
        L_proto = torch.zeros(1, device=trainer.device)
        L_contrastive = torch.zeros(1, device=trainer.device)
        L_reid = torch.zeros(1, device=trainer.device)
        cos_fv_tv_list = []
        cos_fv_Pi_list = []
        valid_proto_count = 0
        f_v_list = []
        t_v_list = []

        if not is_warmup and valid_count > 0:
            view_losses = []
            proto_losses = []

            for inst in gpu_batch.instances:
                if not inst.get('valid', False):
                    continue
                teacher_emb = inst.get('teacher_embedding')
                if teacher_emb is None:
                    continue

                bbox = inst['bbox_xyxy']
                f_v = roi_pool(person_feature_map,
                               torch.tensor(bbox, dtype=torch.float32, device=trainer.device))
                f_v_norm = F.normalize(f_v, p=2, dim=0, eps=1e-6)

                t_v = torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                t_v_norm = F.normalize(t_v, p=2, dim=0, eps=1e-6)

                f_v_list.append(f_v_norm)
                t_v_list.append(t_v_norm)

                cos_view = torch.dot(f_v_norm, t_v_norm)
                cos_fv_tv_list.append(cos_view.item())
                l_view = 1 - cos_view
                view_losses.append(l_view)

                train_id = inst.get('train_id', -1)
                if train_id >= 0 and hasattr(trainer.model, 'prototypes_loaded') and inst.get('prototype_index', -1) >= 0:
                    P_i = trainer.model.prototypes[inst['prototype_index']].squeeze()
                    cos_proto = torch.dot(f_v_norm, P_i)
                    cos_fv_Pi_list.append(cos_proto.item())
                    l_proto = 1 - cos_proto
                    proto_losses.append(l_proto)
                    valid_proto_count += 1

            if args.use_contrastive_loss:
                if len(f_v_list) >= 2:
                    L_contrastive = compute_contrastive_loss(f_v_list, t_v_list, temperature=args.temperature)
                    L_view = L_contrastive
                else:
                    if view_losses:
                        L_view = torch.stack(view_losses).mean()
            else:
                if view_losses:
                    L_view = torch.stack(view_losses).mean()
                if proto_losses:
                    L_proto = torch.stack(proto_losses).mean()
                L_view = L_view + args.alpha_proto * L_proto if proto_losses else L_view

            if args.use_contrastive_loss:
                L_reid = L_contrastive
            else:
                L_reid = L_view

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
            warmup_log['rgb_nz'].append(rgb_nz)
            warmup_log['feat_nz'].append(feat_nz)
        else:
            train_log['L_rgb'].append(L_rgb.item())
            train_log['L_view'].append(L_view.item())
            train_log['L_proto'].append(L_proto.item())
            train_log['L_contrastive'].append(L_contrastive.item())
            train_log['L_reid_total'].append(L_reid.item())
            train_log['cos_fv_tv'].append(avg_cos_tv)
            train_log['cos_fv_Pi'].append(avg_cos_Pi)
            train_log['valid_inst'].append(valid_count)
            train_log['valid_proto_inst'].append(valid_proto_count)
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
            phase = "WARMUP" if is_warmup else "REID"
            if args.freeze_geometry:
                phase = "FREEZE" if is_warmup else "FREEZE_REID"
            if args.use_contrastive_loss:
                phase = "CONTRAST" if not is_warmup else "WARMUP"
            msg = f"[{phase}] Step {step:5d}: L_rgb={L_rgb.item():.4f}"
            if not is_warmup:
                if args.use_contrastive_loss:
                    msg += f", L_contrast={L_contrastive.item():.4f}"
                else:
                    msg += (f", L_view={L_view.item():.4f}, L_proto={L_proto.item():.4f}, ")
                msg += (f"L_reid={L_reid.item():.4f}, cos_tv={avg_cos_tv:.4f}, ")
                msg += (f"cos_Pi={avg_cos_Pi:.4f}, proto_inst={valid_proto_count}, ")
                msg += f"grad_nz={grad_nz:.4f}%"
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
    if r['L_view']:
        n = len(r['L_view'])
        window = min(50, max(10, n // 10))

        early_L_rgb = np.mean(r['L_rgb'][:window])
        late_L_rgb = np.mean(r['L_rgb'][-window:])
        early_L_view = np.mean(r['L_view'][:window])
        late_L_view = np.mean(r['L_view'][-window:])
        early_L_reid = np.mean(r['L_reid_total'][:window])
        late_L_reid = np.mean(r['L_reid_total'][-window:])
        early_cos_tv = np.mean(r['cos_fv_tv'][:window])
        late_cos_tv = np.mean(r['cos_fv_tv'][-window:])
        early_cos_Pi = np.mean(r['cos_fv_Pi'][:window])
        late_cos_Pi = np.mean(r['cos_fv_Pi'][-window:])

        view_decrease = (early_L_view - late_L_view) / max(early_L_view, 1e-8) * 100

        print(f"\n--- ReID Training ({n} iters) ---")
        print(f"  L_rgb:     first{window}={early_L_rgb:.4f}  last{window}={late_L_rgb:.4f}")
        print(f"  L_view:    first{window}={early_L_view:.4f}  last{window}={late_L_view:.4f}  decrease={view_decrease:.2f}%")
        print(f"  L_reid:    first{window}={early_L_reid:.4f}  last{window}={late_L_reid:.4f}")
        print(f"  cos(f_v,t_v): first{window}={early_cos_tv:.4f}  last{window}={late_cos_tv:.4f}  increase={late_cos_tv - early_cos_tv:.4f}")
        print(f"  cos(f_v,P_i): first{window}={early_cos_Pi:.4f}  last{window}={late_cos_Pi:.4f}  increase={late_cos_Pi - early_cos_Pi:.4f}")
        print(f"  grad_nz:   mean={np.mean(r['grad_nz']):.4f}%  max={np.max(r['grad_nz']):.4f}%")
        print(f"  feat_nz:   mean={np.mean(r['feat_nz']):.2f}%")
        print(f"  iter_time: mean={np.mean(r['iter_time']):.2f}s")
        print(f"  NaN/Inf:   {sum(r['nan_inf'])}/{n}")
    else:
        early_L_view = late_L_view = early_cos_tv = late_cos_tv = early_cos_Pi = late_cos_Pi = 0
        early_L_rgb = late_L_rgb = 0

    log_path = os.path.join(REPO_ROOT, args.log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    save_data = {
        'warmup': {k: [float(x) for x in v] for k, v in warmup_log.items()},
        'train': {k: [float(x) for x in v] for k, v in r.items()},
        'config': {
            'warmup_iters': args.warmup_iters,
            'train_iters': args.train_iters,
            'lambda_reid': args.lambda_reid,
            'freeze_geometry': args.freeze_geometry,
            'use_contrastive_loss': args.use_contrastive_loss,
            'total_time_s': total_time,
        },
        'results': {
            'L_view_first': float(early_L_view),
            'L_view_last': float(late_L_view),
            'cos_fv_tv_first': float(early_cos_tv),
            'cos_fv_tv_last': float(late_cos_tv),
            'cos_fv_Pi_first': float(early_cos_Pi),
            'cos_fv_Pi_last': float(late_cos_Pi),
        },
    }
    with open(log_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nLog saved: {log_path}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
