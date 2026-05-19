#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 11B: Multi-view supervised contrastive ReID training.

Random timestamp + P persons + K views per person sampling.
L_reid = 1.0 * L_proto + 0.5 * L_mv + 0.1 * L_teacher

Usage:
    PYTHONPATH=. python tools/train_phase11B_mv_supcon.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --experiment_name phase11B_mv_supcon_P4K2_opacity_lam005_lr1e4 \
        --pooling opacity --lambda_reid 0.05 --person_feature_lr 1e-4 \
        --loss_type prototype_infonce_mv_supcon --tau 0.07 --mv_tau 0.07 \
        --mv_num_person 4 --mv_num_views 2 --mv_loss_weight 0.5 --teacher_loss_weight 0.1 \
        --warmup_iters 1000 --train_iters 20000 --checkpoint_every 2000
"""

import argparse
import os
import sys
import json
import time
import random
from collections import defaultdict

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


class MultiViewSampler:
    """Random timestamp + P persons + K views per person sampler."""

    def __init__(self, dataset, num_person, num_views, max_retries=5):
        self.dataset = dataset
        self.num_person = num_person
        self.num_views = num_views
        self.max_retries = max_retries

        self.camera_ids = dataset.camera_ids
        self.annotations = dataset.annotations

        self.timestamp_to_persons = defaultdict(set)
        self.person_to_frames = defaultdict(list)
        self.all_timestamps = set()

        for frame_id, annots in self.annotations.items():
            if not isinstance(annots, list):
                continue
            person_ids = [p.get('train_id') or p.get('new_id') for p in annots
                          if p.get('train_id') is not None or p.get('new_id') is not None]
            if person_ids:
                self.timestamp_to_persons[frame_id] = set(person_ids)
                self.all_timestamps.add(frame_id)
                for pid in person_ids:
                    for cam_id in self.camera_ids:
                        self.person_to_frames[pid].append((cam_id, frame_id))

        self.all_timestamps = sorted(self.all_timestamps)
        self.person_ids = sorted(self.person_to_frames.keys())

        self.person_to_cam_frames = defaultdict(lambda: defaultdict(list))
        for pid in self.person_to_frames:
            for cam_id, frame_id in self.person_to_frames[pid]:
                self.person_to_cam_frames[pid][cam_id].append(frame_id)

    def sample_batch(self):
        for _ in range(self.max_retries):
            timestamp = random.choice(self.all_timestamps)
            available_persons = list(self.timestamp_to_persons.get(timestamp, set()))
            if len(available_persons) < self.num_person:
                continue
            selected_persons = random.sample(available_persons, self.num_person)

            mv_samples = []
            for pid in selected_persons:
                valid_views = []
                cam_frames = self.person_to_cam_frames.get(pid, {})
                for cam_id, frames in cam_frames.items():
                    if timestamp in frames:
                        valid_views.append(cam_id)

                if len(valid_views) < self.num_views:
                    valid_views = []
                    for cam_id in self.camera_ids:
                        if cam_id in cam_frames:
                            valid_views.append(cam_id)
                    if len(valid_views) < self.num_views:
                        continue

                chosen_views = random.sample(valid_views, self.num_views)
                mv_samples.append({
                    'person_id': pid,
                    'views': [(cam_id, timestamp) for cam_id in chosen_views],
                })

            if len(mv_samples) >= 2:
                return mv_samples

        timestamp = random.choice(self.all_timestamps)
        available_persons = list(self.timestamp_to_persons.get(timestamp, set()))
        if len(available_persons) < 2:
            return None
        selected_persons = random.sample(available_persons, min(self.num_person, len(available_persons)))

        mv_samples = []
        for pid in selected_persons:
            cam_frames = self.person_to_cam_frames.get(pid, {})
            valid_views = list(cam_frames.keys())
            if len(valid_views) < 2:
                continue
            chosen_views = random.sample(valid_views, min(self.num_views, len(valid_views)))
            chosen_ts = []
            for cam_id in chosen_views:
                ts_list = cam_frames[cam_id]
                chosen_ts.append((cam_id, random.choice(ts_list)))
            mv_samples.append({
                'person_id': pid,
                'views': chosen_ts,
            })

        return mv_samples if len(mv_samples) >= 2 else None


def compute_supcon_loss(features, labels, temperature=0.07):
    """
    Supervised contrastive loss.
    features: [N, D] - L2 normalized feature vectors
    labels: [N] - person IDs
    """
    device = features.device
    N = features.shape[0]

    sim_matrix = features @ features.T / temperature  # [N, N]

    labels_expanded = labels.unsqueeze(1)  # [N, 1]
    positive_mask = (labels_expanded == labels_expanded.T).float()  # [N, N]
    positive_mask.fill_diagonal_(0)

    num_positives = positive_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
    exp_sim = torch.exp(sim_matrix)
    exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)

    pos_exp = exp_sim * positive_mask
    log_prob = pos_exp.sum(dim=1, keepdim=True) / num_positives
    log_prob = log_prob - torch.log(exp_sim_sum)
    log_prob = log_prob.mean()

    return -log_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--train_iters', type=int, default=10000)
    parser.add_argument('--lambda_reid', type=float, default=0.05)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--ckpt_dir', type=str, default='runs/phase11B')
    parser.add_argument('--log_path', type=str, default='tools/phase11B_log.json')
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
    parser.add_argument('--loss_type', type=str, default='prototype_infonce_mv_supcon')
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--teacher_loss_weight', type=float, default=0.1)
    parser.add_argument('--experiment_name', type=str, default='phase11B')
    parser.add_argument('--mv_num_person', type=int, default=4)
    parser.add_argument('--mv_num_views', type=int, default=2)
    parser.add_argument('--mv_loss_weight', type=float, default=0.5)
    parser.add_argument('--mv_tau', type=float, default=0.07)
    parser.add_argument('--mv_max_retries', type=int, default=5)
    args = parser.parse_args()

    if args.experiment_name:
        args.ckpt_dir = f'runs/{args.experiment_name}'
        args.log_path = f'tools/{args.experiment_name}_log.json'

    print("=" * 70)
    print("Phase 11B: Multi-view Supervised Contrastive ReID Training")
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

    prototypes = None
    valid_mask = None
    print(f"\nLoading prototypes: {args.prototype_path}")
    proto_data = torch.load(args.prototype_path, map_location=trainer.device, weights_only=False)
    prototypes = proto_data['prototypes'].to(trainer.device)
    valid_mask = proto_data['valid_mask'].to(trainer.device)
    print(f"  prototypes shape: {prototypes.shape}")
    print(f"  valid IDs: {proto_data.get('num_valid_ids', valid_mask.sum().item())}")

    mv_sampler = MultiViewSampler(
        trainer.train_dataset,
        num_person=args.mv_num_person,
        num_views=args.mv_num_views,
        max_retries=args.mv_max_retries,
    )
    print(f"\nMultiViewSampler: {len(mv_sampler.all_timestamps)} timestamps, "
          f"{len(mv_sampler.person_ids)} persons")

    total_iters = args.warmup_iters + args.train_iters

    warmup_log = {'L_rgb': [], 'feat_nz': []}
    train_log = {
        'L_rgb': [], 'L_proto': [], 'L_mv': [], 'L_teacher': [], 'L_reid_total': [],
        'cos_fv_Pi': [], 'cos_teacher': [],
        'cross_view_same_cos': [], 'cross_view_diff_cos': [], 'cross_view_gap': [],
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

    train_iter = iter(trainer.train_dataloader)
    def get_next_batch():
        nonlocal train_iter
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(trainer.train_dataloader)
            batch_data = next(train_iter)
        return trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)

    for step in range(total_iters):
        t_iter_start = time.time()

        is_warmup = step < args.warmup_iters
        if is_warmup:
            gpu_batch = get_next_batch()
            trainer.model.zero_grad()
            render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
            pred_rgb = render_out['pred_rgb']
            person_feature_map = render_out['person_feature_map']

            step_nan = torch.isnan(pred_rgb).any() or torch.isnan(person_feature_map).any()
            L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)
            L_total = L_rgb

            if not step_nan:
                L_total.backward()
                trainer.model.optimizer.step()
            else:
                any_nan_inf = True

            with torch.no_grad():
                feat_nz = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
                feat_abs = person_feature_map.abs().mean().item()

            warmup_log['L_rgb'].append(L_rgb.item())
            warmup_log['feat_nz'].append(feat_nz)

            if step < 3 or step % 200 == 0:
                print(f"[WARMUP] Step {step:5d}: L_rgb={L_rgb.item():.4f}, feat_nz={feat_nz:.2f}%")
            continue

        mv_samples = mv_sampler.sample_batch()
        if mv_samples is None or len(mv_samples) < 2:
            gpu_batch = get_next_batch()
            trainer.model.zero_grad()
            render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
            pred_rgb = render_out['pred_rgb']
            L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)
            L_total = L_rgb
            if not torch.isnan(L_rgb):
                L_total.backward()
                trainer.model.optimizer.step()

            with torch.no_grad():
                feat_nz = (pred_rgb.abs() > 1e-6).sum().item() / pred_rgb.numel() * 100
            train_log['L_rgb'].append(L_rgb.item())
            train_log['L_proto'].append(0)
            train_log['L_mv'].append(0)
            train_log['L_teacher'].append(0)
            train_log['L_reid_total'].append(0)
            train_log['cos_fv_Pi'].append(0)
            train_log['cos_teacher'].append(0)
            train_log['cross_view_same_cos'].append(0)
            train_log['cross_view_diff_cos'].append(0)
            train_log['cross_view_gap'].append(0)
            train_log['valid_inst'].append(0)
            train_log['skipped_low_alpha'].append(0)
            train_log['alpha_sum_mean'].append(0)
            train_log['alpha_mean'].append(0)
            train_log['alpha_max'].append(0)
            train_log['raw_feature_norm_mean'].append(0)
            train_log['weighted_feature_norm_mean'].append(0)
            pf = trainer.model.get_person_feature()
            train_log['grad_mean'].append(pf.grad.abs().mean().item() if pf.grad is not None else 0)
            train_log['grad_max'].append(pf.grad.abs().max().item() if pf.grad is not None else 0)
            train_log['grad_nz'].append((pf.grad.abs() > 1e-8).sum().item() / pf.grad.numel() * 100 if pf.grad is not None else 0)
            train_log['feat_nz'].append(feat_nz)
            train_log['feat_abs'].append(pred_rgb.abs().mean().item())
            train_log['nan_inf'].append(False)
            train_log['iter_time'].append(time.time() - t_iter_start)
            continue

        trainer.model.zero_grad()

        all_features = []
        all_person_ids = []
        all_teacher_cos = []
        all_proto_cos = []
        alpha_sum_list = []
        valid_inst_count = 0
        skipped_low_alpha = 0

        L_rgb_total = torch.zeros(1, device=trainer.device)
        num_views_rendered = 0

        for person_sample in mv_samples:
            pid = person_sample['person_id']
            for cam_id, timestamp in person_sample['views']:
                frame_idx = trainer.train_dataset.camera_ids.index(cam_id) if cam_id in trainer.train_dataset.camera_ids else 0
                gpu_batch = get_next_batch()
                render_out = trainer.model(gpu_batch, train=True, frame_id=frame_idx, render_person_feature=True)
                pred_rgb = render_out['pred_rgb']
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')

                L_rgb_total = L_rgb_total + F.l1_loss(pred_rgb, gpu_batch.rgb_gt)
                num_views_rendered += 1

                inst = None
                for i in gpu_batch.instances:
                    if i.get('train_id') == pid and i.get('valid', False):
                        inst = i
                        break

                if inst is None:
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
                all_features.append(f_v)
                all_person_ids.append(pid)

                if not pool_stats.get('skipped', False):
                    alpha_sum_list.append(pool_stats.get('alpha_sum', 0))

                teacher_emb = inst.get('teacher_embedding')
                if teacher_emb is not None:
                    t_v = torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                    t_v_norm = F.normalize(t_v, p=2, dim=0, eps=1e-6)
                    cos_teacher = torch.dot(f_v, t_v_norm)
                    all_teacher_cos.append(cos_teacher.item())

                if valid_mask[pid]:
                    P_i = prototypes[pid].squeeze()
                    cos_proto = torch.dot(f_v, P_i)
                    all_proto_cos.append(cos_proto.item())

        if num_views_rendered > 0:
            L_rgb = L_rgb_total / num_views_rendered
        else:
            L_rgb = L_rgb_total

        if len(all_features) < 2:
            L_proto = torch.zeros(1, device=trainer.device)
            L_mv = torch.zeros(1, device=trainer.device)
            L_teacher = torch.zeros(1, device=trainer.device)
            L_reid = torch.zeros(1, device=trainer.device)
        else:
            f_stack = torch.stack(all_features)
            p_ids = torch.tensor(all_person_ids, device=trainer.device)

            logits = f_stack @ prototypes.T / args.tau
            L_proto = F.cross_entropy(logits, p_ids)

            L_mv = compute_supcon_loss(f_stack, p_ids, temperature=args.mv_tau)

            if all_teacher_cos:
                L_teacher = 1 - torch.tensor(all_teacher_cos, device=trainer.device).mean()
            else:
                L_teacher = torch.zeros(1, device=trainer.device)

            L_reid = 1.0 * L_proto + args.mv_loss_weight * L_mv + args.teacher_loss_weight * L_teacher

        L_total = L_rgb + args.lambda_reid * L_reid

        step_nan = torch.isnan(L_total)
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

        avg_proto_cos = np.mean(all_proto_cos) if all_proto_cos else 0
        avg_teacher_cos = np.mean(all_teacher_cos) if all_teacher_cos else 0

        same_person_pairs = []
        diff_person_pairs = []
        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                cos_ij = torch.dot(all_features[i], all_features[j]).item()
                if all_person_ids[i] == all_person_ids[j]:
                    same_person_pairs.append(cos_ij)
                else:
                    diff_person_pairs.append(cos_ij)

        cross_same = np.mean(same_person_pairs) if same_person_pairs else 0
        cross_diff = np.mean(diff_person_pairs) if diff_person_pairs else 0
        cross_gap = cross_same - cross_diff

        iter_time = time.time() - t_iter_start

        train_log['L_rgb'].append(L_rgb.item())
        train_log['L_proto'].append(L_proto.item())
        train_log['L_mv'].append(L_mv.item())
        train_log['L_teacher'].append(L_teacher.item())
        train_log['L_reid_total'].append(L_reid.item())
        train_log['cos_fv_Pi'].append(avg_proto_cos)
        train_log['cos_teacher'].append(avg_teacher_cos)
        train_log['cross_view_same_cos'].append(cross_same)
        train_log['cross_view_diff_cos'].append(cross_diff)
        train_log['cross_view_gap'].append(cross_gap)
        train_log['valid_inst'].append(valid_inst_count)
        train_log['skipped_low_alpha'].append(skipped_low_alpha)
        train_log['alpha_sum_mean'].append(np.mean(alpha_sum_list) if alpha_sum_list else 0)
        train_log['alpha_mean'].append(0)
        train_log['alpha_max'].append(0)
        train_log['raw_feature_norm_mean'].append(0)
        train_log['weighted_feature_norm_mean'].append(0)
        train_log['grad_mean'].append(grad_mean)
        train_log['grad_max'].append(grad_max)
        train_log['grad_nz'].append(grad_nz)
        train_log['feat_nz'].append(0)
        train_log['feat_abs'].append(0)
        train_log['nan_inf'].append(bool(step_nan))
        train_log['iter_time'].append(iter_time)

        should_print = (step < 3 or step % 200 == 0
                        or step == args.warmup_iters - 1
                        or step == args.warmup_iters
                        or step == total_iters - 1)
        if should_print:
            msg = (f"[MV-REID] Step {step:5d}: L_rgb={L_rgb.item():.4f}, "
                   f"L_proto={L_proto.item():.4f}, L_mv={L_mv.item():.4f}, "
                   f"L_teacher={L_teacher.item():.4f}, L_reid={L_reid.item():.4f}, "
                   f"cos_proto={avg_proto_cos:.4f}, cos_teacher={avg_teacher_cos:.4f}, "
                   f"cross_same={cross_same:.4f}, cross_diff={cross_diff:.4f}, "
                   f"cross_gap={cross_gap:.6f}, valid={valid_inst_count}, "
                   f"skip_alpha={skipped_low_alpha}, "
                   f"grad_nz={grad_nz:.4f}%, t={iter_time:.2f}s")
            print(msg)

        reid_step = step - args.warmup_iters
        if reid_step > 0 and reid_step % args.save_every == 0:
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

    r = train_log
    if r['L_proto']:
        n = len(r['L_proto'])
        window = min(50, max(10, n // 10))

        def w(k): return np.mean(r[k][:window]), np.mean(r[k][-window:])

        early_L_rgb, late_L_rgb = w('L_rgb')
        early_L_proto, late_L_proto = w('L_proto')
        early_L_mv, late_L_mv = w('L_mv')
        early_L_teacher, late_L_teacher = w('L_teacher')
        early_cos_proto, late_cos_proto = w('cos_fv_Pi')
        early_same_cos, late_same_cos = w('cross_view_same_cos')
        early_diff_cos, late_diff_cos = w('cross_view_diff_cos')
        early_gap, late_gap = w('cross_view_gap')

        print(f"\n--- ReID Training ({n} iters, pooling={args.pooling}) ---")
        print(f"  L_rgb:         first{window}={early_L_rgb:.4f}  last{window}={late_L_rgb:.4f}")
        print(f"  L_proto:       first{window}={early_L_proto:.4f}  last{window}={late_L_proto:.4f}")
        print(f"  L_mv:          first{window}={early_L_mv:.4f}  last{window}={late_L_mv:.4f}")
        print(f"  L_teacher:     first{window}={early_L_teacher:.4f}  last{window}={late_L_teacher:.4f}")
        print(f"  cos_proto:     first{window}={early_cos_proto:.4f}  last{window}={late_cos_proto:.4f}")
        print(f"  cross_same:    first{window}={early_same_cos:.4f}  last{window}={late_same_cos:.4f}")
        print(f"  cross_diff:    first{window}={early_diff_cos:.4f}  last{window}={late_diff_cos:.4f}")
        print(f"  cross_gap:     first{window}={early_gap:.6f}  last{window}={late_gap:.6f}")
        print(f"  valid_inst:    mean={np.mean(r['valid_inst']):.1f}")
        print(f"  iter_time:     mean={np.mean(r['iter_time']):.2f}s")
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
