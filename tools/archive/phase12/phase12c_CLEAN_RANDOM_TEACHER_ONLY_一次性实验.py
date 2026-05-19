#!/usr/bin/env python3
"""
Phase 12C: Gaussian-Set Clean Random Teacher-Only Training + Fixed Eval

Verify that Gaussian-Set can learn generalization under random sampling.
Uses the same direct Gaussian xyz projection method validated in Phase 12B.

Goal: Answer whether fixed eval cosine steadily rises and same/diff gap doesn't degrade.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rich import pretty, traceback

pretty.install()
traceback.install()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from hydra import initialize_config_dir, compose


def normalize_feat(x, eps=1e-6):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def gaussian_set_pooling(model, gpu_batch, bbox, cam_id, frame_id, args, device):
    """
    Gaussian-Set pooling via direct Gaussian xyz projection.

    Same method as Phase 12A/12B: project Gaussian centers to image plane,
    filter by bbox, and pool person_feature weighted by opacity.

    Returns: G (pooled feature [D,]), debug_info
    """
    x1, y1, x2, y2 = bbox

    try:
        xyz = model.positions  # [N, 3]
        opacity = model.get_density().squeeze(-1)  # [N]
        person_feature = model.get_person_feature()  # [N, D]

        N = xyz.shape[0]
        if N == 0:
            return None, {
                'num_gaussians_in_bbox': 0,
                'weight_sum': 0.0,
                'failure_reason': 'no_gaussians',
            }

        intrinsics = gpu_batch.intrinsics
        if intrinsics is None or len(intrinsics) < 4:
            return None, {
                'num_gaussians_in_bbox': 0,
                'weight_sum': 0.0,
                'failure_reason': 'no_intrinsics',
            }

        fx, fy, cx, cy = intrinsics

        T_to_world = gpu_batch.T_to_world[0]
        R_world_to_cam = T_to_world[:3, :3].t()
        t_world_to_cam = -R_world_to_cam @ T_to_world[:3, 3]

        xyz_cam = (R_world_to_cam @ xyz.t()).t() + t_world_to_cam
        depth = xyz_cam[:, 2]
        valid_depth = depth > 0

        x_img = fx * xyz_cam[:, 0] / xyz_cam[:, 2] + cx
        y_img = fy * xyz_cam[:, 1] / xyz_cam[:, 2] + cy

        h_img = gpu_batch.rays_dir.shape[1]
        w_img = gpu_batch.rays_dir.shape[2]

        x_finite = torch.isfinite(x_img)
        y_finite = torch.isfinite(y_img)
        x_in_bounds = (x_img >= 0) & (x_img < w_img)
        y_in_bounds = (y_img >= 0) & (y_img < h_img)
        opacity_positive = opacity > 0

        valid = valid_depth & x_finite & y_finite & x_in_bounds & y_in_bounds & opacity_positive
        inside_bbox = (x_img >= x1) & (x_img < x2) & (y_img >= y1) & (y_img < y2)
        inside = valid & inside_bbox

        if inside.sum() == 0:
            return None, {
                'num_gaussians_in_bbox': 0,
                'weight_sum': 0.0,
                'failure_reason': 'no_gaussians_in_bbox',
            }

        weights = opacity[inside]
        z = person_feature[inside]

        weight_sum = weights.sum()
        denom = weight_sum.clamp(min=args.denom_eps)
        weighted_sum = (weights[:, None] * z).sum(dim=0)
        G = weighted_sum / denom
        G = normalize_feat(G)

        num_inside = int(inside.sum().item())
        depth_inside = depth[inside]

        debug_info = {
            'num_gaussians_in_bbox': num_inside,
            'weight_sum': float(weight_sum.item()),
            'weight_min': float(weights.min().item()),
            'weight_mean': float(weights.mean().item()),
            'weight_max': float(weights.max().item()),
            'depth_min': float(depth_inside.min().item()),
            'depth_mean': float(depth_inside.mean().item()),
            'depth_max': float(depth_inside.max().item()),
            'bbox': list(bbox),
            'cam_id': cam_id,
            'frame_id': frame_id,
        }

        return G, debug_info

    except Exception as e:
        return None, {
            'num_gaussians_in_bbox': 0,
            'weight_sum': 0.0,
            'failure_reason': f'{str(e)[:60]}',
        }


class BatchBuilder:
    """Build gpu_batch from cam_id and frame_idx using dataset indices."""
    def __init__(self, dataset):
        self.dataset = dataset
        self._cam_frame_to_index = {}
        for idx in range(len(dataset)):
            cam_id, frame_idx = dataset.indices[idx]
            self._cam_frame_to_index[(cam_id, int(frame_idx))] = idx

    def get_batch(self, cam_id, frame_idx):
        key = (cam_id, int(frame_idx))
        idx = self._cam_frame_to_index.get(key)
        if idx is None:
            return None
        raw_batch = self.dataset[idx]
        return self.dataset.get_gpu_batch_with_intrinsics(raw_batch)


def build_candidate_pool(args, trainer, batch_builder, allowed_cameras):
    """Build candidate pool of P*K samples for training."""
    print(f"\nBuilding candidate pool (target: {args.pre_collect_pool_size})...")

    dataset = trainer.train_dataset
    candidates = []

    # Iterate over all dataset entries
    for idx in range(len(dataset)):
        cam_id, frame_idx = dataset.indices[idx]
        if allowed_cameras and cam_id not in allowed_cameras:
            continue

        anns = dataset.annotations.get(int(frame_idx), [])
        for ann in anns:
            ann_cam_id = ann.get('camera_id')
            if ann_cam_id is None:
                continue
            ann_cam_str = f"C{ann_cam_id + 1}"
            if ann_cam_str != cam_id:
                continue

            pid = ann.get('new_id')
            if pid is None:
                continue

            # Get teacher embedding from cache
            teacher_emb = None
            if dataset.teacher_cache is not None:
                bbox_dict = ann.get('bbox', {})
                if isinstance(bbox_dict, dict) and len(bbox_dict) >= 4:
                    # Use string camera_id for cache key
                    cam_id_str = f"C{ann_cam_id + 1}"
                    x1_c = int(bbox_dict.get('xmin', 0))
                    y1_c = int(bbox_dict.get('ymin', 0))
                    x2_c = int(bbox_dict.get('xmax', 0))
                    y2_c = int(bbox_dict.get('ymax', 0))
                    cache_key = (int(frame_idx), cam_id_str, int(pid),
                                 x1_c, y1_c, x2_c, y2_c)
                    cache_entry = dataset.teacher_cache.get(cache_key)
                    if cache_entry is not None:
                        teacher_emb = cache_entry.get('embedding')
                        # Flatten if needed
                        if hasattr(teacher_emb, 'squeeze'):
                            teacher_emb = teacher_emb.squeeze()

            if teacher_emb is None:
                continue

            bbox_dict = ann.get('bbox', {})
            if not isinstance(bbox_dict, dict):
                continue

            x1 = int(bbox_dict.get('xmin', 0))
            y1 = int(bbox_dict.get('ymin', 0))
            x2 = int(bbox_dict.get('xmax', 0))
            y2 = int(bbox_dict.get('ymax', 0))
            if x2 <= x1 or y2 <= y1:
                continue

            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area < args.min_bbox_area:
                continue

            # Build gpu_batch and test projection
            gpu_batch = batch_builder.get_batch(cam_id, int(frame_idx))
            if gpu_batch is None:
                continue

            G, debug_info = gaussian_set_pooling(
                trainer.model, gpu_batch, [x1, y1, x2, y2], cam_id, int(frame_idx), args, trainer.device
            )

            if G is None or debug_info['num_gaussians_in_bbox'] == 0:
                continue
            if debug_info['weight_sum'] <= 0:
                continue

            candidates.append({
                'person_id': int(pid),
                'cam_id': cam_id,
                'frame_idx': int(frame_idx),
                'dataset_index': idx,
                'bbox': [x1, y1, x2, y2],
                'bbox_area': int(bbox_area),
                'teacher_emb': teacher_emb.cpu().numpy().tolist() if hasattr(teacher_emb, 'cpu') else teacher_emb.tolist(),
                'num_gaussians_in_bbox': debug_info['num_gaussians_in_bbox'],
                'weight_sum': float(debug_info['weight_sum']),
            })

            if len(candidates) >= args.pre_collect_pool_size:
                break

        if len(candidates) >= args.pre_collect_pool_size:
            break

    print(f"Candidate pool: {len(candidates)} samples")
    if len(candidates) > 0:
        unique_persons = set(c['person_id'] for c in candidates)
        unique_cams = set(c['cam_id'] for c in candidates)
        print(f"  {len(unique_persons)} unique persons, {len(unique_cams)} unique cameras")
        print(f"  num_gaussians: min={min(c['num_gaussians_in_bbox'] for c in candidates)}, "
              f"mean={np.mean([c['num_gaussians_in_bbox'] for c in candidates]):.1f}, "
              f"max={max(c['num_gaussians_in_bbox'] for c in candidates)}")

    return candidates


def select_fixed_eval_samples(candidate_pool, fixed_eval_size=16):
    """Select fixed eval samples covering multiple persons with multiple views each.
    
    Goal: ensure we have same-person multi-view pairs AND different-person pairs
    so that same_cos, diff_cos, and cross_view_gap can be computed.
    
    Strategy:
    1. Find persons that have >= 2 views (different cameras).
    2. Select up to min(P, available) persons with >= 2 views each.
    3. Fill remaining slots with other persons (1 view each) to reach fixed_eval_size.
    """
    print(f"\nSelecting {fixed_eval_size} fixed eval samples...")

    if len(candidate_pool) == 0:
        print("ERROR: candidate_pool is empty")
        return []

    # Group by person
    person_to_samples = defaultdict(list)
    for c in candidate_pool:
        person_to_samples[c['person_id']].append(c)

    # Find persons with >= 2 views (different cameras)
    multi_view_persons = {
        pid: samples
        for pid, samples in person_to_samples.items()
        if len(set(s['cam_id'] for s in samples)) >= 2
    }
    
    # Find persons with only 1 view
    single_view_persons = {
        pid: samples
        for pid, samples in person_to_samples.items()
        if pid not in multi_view_persons
    }

    selected = []
    used_sample_keys = set()
    num_multi_view = min(len(multi_view_persons), fixed_eval_size // 2)

    # Priority 1: select multi-view persons (up to fixed_eval_size // 2 persons, 2 views each)
    sorted_mv = sorted(multi_view_persons.keys(),
                       key=lambda p: len(multi_view_persons[p]), reverse=True)
    
    for pid in sorted_mv:
        if num_multi_view <= 0:
            break
        samples = multi_view_persons[pid]
        # Pick at most 2 views from different cameras
        cam_to_sample = {}
        for s in samples:
            if s['cam_id'] not in cam_to_sample:
                cam_to_sample[s['cam_id']] = s
                sample_key = (s['person_id'], s['cam_id'], s['frame_idx'])
                used_sample_keys.add(sample_key)
                selected.append(s)
                if len(cam_to_sample) >= 2:
                    break
        if len(cam_to_sample) >= 2:
            num_multi_view -= 1
        
        if len(selected) >= fixed_eval_size:
            break

    # Priority 2: fill remaining with other persons (1 view each)
    remaining_slots = fixed_eval_size - len(selected)
    if remaining_slots > 0 and single_view_persons:
        sorted_sv = sorted(single_view_persons.keys(),
                           key=lambda p: len(single_view_persons[p]), reverse=True)
        for pid in sorted_sv:
            if remaining_slots <= 0:
                break
            samples = single_view_persons[pid]
            for s in samples:
                sample_key = (s['person_id'], s['cam_id'], s['frame_idx'])
                if sample_key not in used_sample_keys:
                    selected.append(s)
                    used_sample_keys.add(sample_key)
                    remaining_slots -= 1
                    break

    # If still not enough, fill from any remaining samples
    if len(selected) < fixed_eval_size:
        remaining = [c for c in candidate_pool
                     if (c['person_id'], c['cam_id'], c['frame_idx']) not in used_sample_keys]
        extra = random.sample(remaining, min(fixed_eval_size - len(selected), len(remaining)))
        selected.extend(extra)

    print(f"Selected {len(selected)} fixed eval samples")
    if len(selected) > 0:
        unique_persons = set(c['person_id'] for c in selected)
        unique_cams = set(c['cam_id'] for c in selected)
        multi_view_count = sum(1 for pid in unique_persons
                              if len(set(c['cam_id'] for c in selected if c['person_id'] == pid)) >= 2)
        print(f"  {len(unique_persons)} unique persons, {len(unique_cams)} unique cameras")
        print(f"  {multi_view_count} persons with >= 2 views")

    return selected


def compute_same_diff_cos(gaussianset_features, samples):
    """Compute same_cos, diff_cos, gap from Gaussian-Set features."""
    # Filter out None features
    valid_features = [(i, f) for i, f in enumerate(gaussianset_features) if f is not None]
    if len(valid_features) < 2:
        return None, None, None, 0, 0

    same_cos_list = []
    diff_cos_list = []
    positive_pair_count = 0
    negative_pair_count = 0

    person_to_indices = {}
    for idx, (feat_idx, feat) in enumerate(valid_features):
        s = samples[feat_idx]
        pid = s['person_id']
        if pid not in person_to_indices:
            person_to_indices[pid] = []
        person_to_indices[pid].append(idx)

    for pid, indices in person_to_indices.items():
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_i, idx_j = indices[i], indices[j]
                    feat_i = valid_features[idx_i][1]
                    feat_j = valid_features[idx_j][1]
                    sc = F.cosine_similarity(
                        feat_i.unsqueeze(0),
                        feat_j.unsqueeze(0)
                    ).item()
                    same_cos_list.append(sc)
                    positive_pair_count += 1

    all_valid_indices = list(range(len(valid_features)))
    neg_checked = 0
    max_neg_samples = min(100, len(all_valid_indices) * (len(all_valid_indices) - 1) // 2)
    for _ in range(max_neg_samples):
        i, j = random.sample(all_valid_indices, 2)
        pid_i = samples[valid_features[i][0]]['person_id']
        pid_j = samples[valid_features[j][0]]['person_id']
        if pid_i != pid_j:
            feat_i = valid_features[i][1]
            feat_j = valid_features[j][1]
            dc = F.cosine_similarity(
                feat_i.unsqueeze(0),
                feat_j.unsqueeze(0)
            ).item()
            diff_cos_list.append(dc)
            negative_pair_count += 1
            neg_checked += 1
            if neg_checked >= 50:
                break

    same_cos = float(np.mean(same_cos_list)) if same_cos_list else None
    diff_cos = float(np.mean(diff_cos_list)) if diff_cos_list else None
    gap = (same_cos - diff_cos) if (same_cos is not None and diff_cos is not None) else None

    return same_cos, diff_cos, gap, positive_pair_count, negative_pair_count


def run_fixed_eval(args, model, batch_builder, fixed_eval_samples, device):
    """Evaluate on fixed eval samples (no gradient)."""
    losses = []
    cos_values = []
    gaussianset_features = []
    num_gaussians_list = []
    weight_sum_list = []
    invalid_count = 0
    per_camera_cos = defaultdict(list)
    per_camera_valid = defaultdict(int)

    for sample in fixed_eval_samples:
        gpu_batch = batch_builder.get_batch(sample['cam_id'], sample['frame_idx'])
        if gpu_batch is None:
            invalid_count += 1
            continue

        G, debug_info = gaussian_set_pooling(
            model, gpu_batch, sample['bbox'], sample['cam_id'], sample['frame_idx'], args, device
        )

        if G is None:
            invalid_count += 1
            continue

        T = torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=device)
        T = normalize_feat(T)

        cos_sim = F.cosine_similarity(G.unsqueeze(0), T.unsqueeze(0)).item()
        loss_i = 1.0 - cos_sim

        gaussianset_features.append(G)
        losses.append(loss_i)
        cos_values.append(cos_sim)
        num_gaussians_list.append(debug_info['num_gaussians_in_bbox'])
        weight_sum_list.append(debug_info['weight_sum'])

        per_camera_cos[sample['cam_id']].append(cos_sim)
        per_camera_valid[sample['cam_id']] += 1

    if len(losses) == 0:
        return {
            'fixed_loss_teacher': 1.0,
            'fixed_cos_mean': 0.0,
            'fixed_cos_min': 0.0,
            'fixed_cos_max': 0.0,
            'fixed_valid_roi_count': 0,
            'fixed_invalid_roi_count': invalid_count,
            'fixed_num_gaussians_mean': 0,
            'fixed_weight_sum_mean': 0.0,
            'fixed_same_cos': None,
            'fixed_diff_cos': None,
            'fixed_cross_view_gap': None,
            'fixed_positive_pair_count': 0,
            'fixed_negative_pair_count': 0,
            'per_camera_fixed_cos': {},
            'per_camera_fixed_valid_count': dict(per_camera_valid),
        }

    loss = float(np.mean(losses))
    cos_mean = float(np.mean(cos_values))
    cos_min = float(np.min(cos_values))
    cos_max = float(np.max(cos_values))

    same_cos, diff_cos, gap, pos_count, neg_count = compute_same_diff_cos(
        gaussianset_features, fixed_eval_samples
    )

    return {
        'fixed_loss_teacher': loss,
        'fixed_cos_mean': cos_mean,
        'fixed_cos_min': cos_min,
        'fixed_cos_max': cos_max,
        'fixed_valid_roi_count': len(losses),
        'fixed_invalid_roi_count': invalid_count,
        'fixed_num_gaussians_mean': float(np.mean(num_gaussians_list)),
        'fixed_weight_sum_mean': float(np.mean(weight_sum_list)),
        'fixed_same_cos': same_cos,
        'fixed_diff_cos': diff_cos,
        'fixed_cross_view_gap': gap,
        'fixed_positive_pair_count': pos_count,
        'fixed_negative_pair_count': neg_count,
        'per_camera_fixed_cos': {k: float(np.mean(v)) for k, v in per_camera_cos.items()},
        'per_camera_fixed_valid_count': dict(per_camera_valid),
    }


def run_phase12c(args, trainer, batch_builder, allowed_cameras):
    """Phase 12C: Gaussian-Set Clean Random Training + Fixed Eval."""
    print(f"\n{'='*70}")
    print(f"PHASE 12C: GAUSSIAN-SET CLEAN RANDOM FIXED EVAL")
    print(f"{'='*70}")
    print(f"P={args.P}, K={args.K}, batch_size={args.P * args.K}")
    print(f"Allowed cameras: {allowed_cameras}")
    print(f"Steps: {args.num_steps}, LR: {args.person_feature_lr}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Build or load candidate pool
    candidate_pool_path = getattr(args, 'candidate_pool_path', None)
    if candidate_pool_path and os.path.exists(candidate_pool_path):
        print(f"Loading candidate pool from {candidate_pool_path}")
        with open(candidate_pool_path, 'r') as f:
            candidate_pool = json.load(f)
    else:
        candidate_pool = build_candidate_pool(args, trainer, batch_builder, allowed_cameras)
        with open(os.path.join(args.output_dir, 'candidate_pool.json'), 'w') as f:
            json.dump(candidate_pool, f, indent=2, default=str)

    if len(candidate_pool) == 0:
        print("ERROR: No candidates in pool")
        return False

    # Select fixed eval samples
    fixed_eval_size = getattr(args, 'fixed_eval_size', 16)
    fixed_eval_samples = select_fixed_eval_samples(candidate_pool, fixed_eval_size)
    with open(os.path.join(args.output_dir, 'fixed_eval_samples.json'), 'w') as f:
        json.dump(fixed_eval_samples, f, indent=2, default=str)

    # Setup optimizer
    pf = trainer.model.get_person_feature()
    pf_before = pf.clone().detach()

    optimizer = torch.optim.Adam(
        [trainer.model._person_feature],
        lr=args.person_feature_lr,
    )

    # Training state
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    metrics_log = []
    best_fixed_cos = -1.0
    best_step = 0
    total_nan = 0
    total_inf = 0
    total_skipped = 0

    print(f"\n{'='*70}")
    print(f"TRAINING: pool_size={len(candidate_pool)}, fixed_eval={len(fixed_eval_samples)}")
    print(f"{'='*70}")

    # Initial fixed eval
    with torch.no_grad():
        eval_result = run_fixed_eval(args, trainer.model, batch_builder, fixed_eval_samples, trainer.device)
    print(f"[EVAL-0] fixed_cos={eval_result['fixed_cos_mean']:.4f} "
          f"fixed_loss={eval_result['fixed_loss_teacher']:.4f} "
          f"valid={eval_result['fixed_valid_roi_count']}/{len(fixed_eval_samples)}")
    if eval_result['fixed_same_cos'] is not None:
        print(f"  same_cos={eval_result['fixed_same_cos']:.4f}, "
              f"diff_cos={eval_result['fixed_diff_cos']:.4f}, "
              f"gap={eval_result['fixed_cross_view_gap']:.4f}")

    # Save initial checkpoint
    torch.save({
        'model_state_dict': {
            '_person_feature': trainer.model._person_feature.cpu().clone(),
        },
    }, os.path.join(args.output_dir, 'checkpoint_initial.pt'))

    for step in range(args.num_steps):
        step_start = time.time()

        # Sample P*K from candidate pool
        batch_size = args.P * args.K
        if len(candidate_pool) < batch_size:
            total_skipped += 1
            continue

        batch_samples = random.sample(candidate_pool, batch_size)

        # Group by (cam_id, frame_idx) to avoid redundant batch building
        batch_by_cam_frame = defaultdict(list)
        for s in batch_samples:
            key = (s['cam_id'], s['frame_idx'])
            batch_by_cam_frame[key].append(s)

        optimizer.zero_grad()

        losses = []
        cos_values = []
        gaussianset_features = []
        valid_samples = []
        num_gaussians_list = []
        weight_sum_list = []
        invalid_count = 0

        for (cam_id, frame_idx), samples in batch_by_cam_frame.items():
            gpu_batch = batch_builder.get_batch(cam_id, frame_idx)
            if gpu_batch is None:
                invalid_count += len(samples)
                continue

            for sample in samples:
                G, debug_info = gaussian_set_pooling(
                    trainer.model, gpu_batch, sample['bbox'], sample['cam_id'], sample['frame_idx'],
                    args, trainer.device
                )

                if G is None:
                    invalid_count += 1
                    continue

                T = torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=trainer.device)
                T = normalize_feat(T)

                cos_sim = F.cosine_similarity(G.unsqueeze(0), T.unsqueeze(0))
                loss_i = 1.0 - cos_sim

                gaussianset_features.append(G)
                valid_samples.append(sample)
                losses.append(loss_i)
                cos_values.append(cos_sim.item())
                num_gaussians_list.append(debug_info['num_gaussians_in_bbox'])
                weight_sum_list.append(debug_info['weight_sum'])

        valid_count = len(losses)
        if valid_count == 0:
            total_skipped += 1
            continue

        loss = torch.stack(losses).mean()
        loss.backward()

        grad_norm_before_clip = None
        grad_norm_after_clip = None
        if trainer.model._person_feature.grad is not None:
            grad_norm_before_clip = trainer.model._person_feature.grad.norm().item()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [trainer.model._person_feature], args.grad_clip_norm
                )
            grad_norm_after_clip = trainer.model._person_feature.grad.norm().item()

        optimizer.step()

        pf_after = trainer.model.get_person_feature()
        param_delta_tensor = pf_after - pf_before
        param_delta_norm = param_delta_tensor.norm().item()
        param_delta_max = param_delta_tensor.abs().max().item()
        pf_before = pf_after.clone().detach()

        cos_mean = float(np.mean(cos_values)) if cos_values else 0
        cos_min = float(np.min(cos_values)) if cos_values else 0
        cos_max = float(np.max(cos_values)) if cos_values else 0

        nan_count = int(torch.isnan(trainer.model._person_feature).sum().item())
        inf_count = int(torch.isinf(trainer.model._person_feature).sum().item())
        total_nan += nan_count
        total_inf += inf_count

        step_time = time.time() - step_start

        # Logging
        if step % args.log_interval == 0:
            log_line = (f"[PHASE12C] Step {step:5d}: "
                        f"train_loss={loss.item():.4f} "
                        f"train_cos={cos_mean:.4f} "
                        f"grad={grad_norm_before_clip:.4e}->{grad_norm_after_clip:.4e} "
                        f"delta={param_delta_norm:.6e} "
                        f"valid={valid_count}/{batch_size} "
                        f"invalid={invalid_count} "
                        f"skipped={total_skipped} "
                        f"t={step_time:.2f}s")
            print(log_line)

        # Fixed eval
        fixed_eval_result = {}
        if step % args.eval_interval == 0 or step == args.num_steps - 1:
            with torch.no_grad():
                fixed_eval_result = run_fixed_eval(
                    args, trainer.model, batch_builder, fixed_eval_samples, trainer.device
                )
            fc = fixed_eval_result['fixed_cos_mean']
            fl = fixed_eval_result['fixed_loss_teacher']
            print(f"  [EVAL] fixed_cos={fc:.4f} fixed_loss={fl:.4f} "
                  f"valid={fixed_eval_result['fixed_valid_roi_count']}/{len(fixed_eval_samples)}")
            if fixed_eval_result['fixed_same_cos'] is not None:
                print(f"    same_cos={fixed_eval_result['fixed_same_cos']:.4f}, "
                      f"diff_cos={fixed_eval_result['fixed_diff_cos']:.4f}, "
                      f"gap={fixed_eval_result['fixed_cross_view_gap']:.4f}")

            if fc > best_fixed_cos:
                best_fixed_cos = fc
                best_step = step
                torch.save({
                    'model_state_dict': {
                        '_person_feature': trainer.model._person_feature.cpu().clone(),
                    },
                    'step': step,
                    'fixed_cos': fc,
                }, os.path.join(args.output_dir, 'checkpoint_best_fixed_cos.pt'))

        # Save metrics
        step_record = {
            'step': step,
            'train_loss_teacher': float(loss.item()),
            'train_cos_mean': cos_mean,
            'train_cos_min': cos_min,
            'train_cos_max': cos_max,
            'valid_roi_count': valid_count,
            'invalid_roi_count': invalid_count,
            'skipped_step_count': total_skipped,
            'num_gaussians_min': int(np.min(num_gaussians_list)) if num_gaussians_list else 0,
            'num_gaussians_mean': float(np.mean(num_gaussians_list)) if num_gaussians_list else 0,
            'num_gaussians_max': int(np.max(num_gaussians_list)) if num_gaussians_list else 0,
            'weight_sum_min': float(np.min(weight_sum_list)) if weight_sum_list else 0,
            'weight_sum_mean': float(np.mean(weight_sum_list)) if weight_sum_list else 0,
            'weight_sum_max': float(np.max(weight_sum_list)) if weight_sum_list else 0,
            'grad_norm_before_clip': float(grad_norm_before_clip) if grad_norm_before_clip is not None else 0,
            'grad_norm_after_clip': float(grad_norm_after_clip) if grad_norm_after_clip is not None else 0,
            'param_delta_norm': float(param_delta_norm),
            'param_delta_max': float(param_delta_max),
            'nan_count': nan_count,
            'inf_count': inf_count,
        }
        step_record.update(fixed_eval_result)
        metrics_log.append(step_record)

    # Write metrics
    with open(metrics_path, 'w') as f:
        for r in metrics_log:
            f.write(json.dumps(r, default=str) + "\n")

    # Save latest checkpoint
    torch.save({
        'model_state_dict': {
            '_person_feature': trainer.model._person_feature.cpu().clone(),
        },
        'step': args.num_steps - 1,
    }, os.path.join(args.output_dir, 'checkpoint_latest.pt'))

    # Compute summary
    fixed_cos_values = [m['fixed_cos_mean'] for m in metrics_log if 'fixed_cos_mean' in m]
    fixed_loss_values = [m['fixed_loss_teacher'] for m in metrics_log if 'fixed_loss_teacher' in m]
    fixed_gap_values = [m['fixed_cross_view_gap'] for m in metrics_log if 'fixed_cross_view_gap' in m]

    first_fixed_cos = fixed_cos_values[0] if fixed_cos_values else 0
    best_fixed_cos_final = max(fixed_cos_values) if fixed_cos_values else 0
    last_fixed_cos = fixed_cos_values[-1] if fixed_cos_values else 0
    fixed_cos_delta = last_fixed_cos - first_fixed_cos

    first_fixed_loss = fixed_loss_values[0] if fixed_loss_values else 1.0
    last_fixed_loss = fixed_loss_values[-1] if fixed_loss_values else 1.0
    fixed_loss_delta = last_fixed_loss - first_fixed_loss

    first_fixed_gap = fixed_gap_values[0] if fixed_gap_values else None
    last_fixed_gap = fixed_gap_values[-1] if fixed_gap_values else None
    fixed_gap_delta = None
    if first_fixed_gap is not None and last_fixed_gap is not None:
        fixed_gap_delta = last_fixed_gap - first_fixed_gap

    param_deltas = [m['param_delta_norm'] for m in metrics_log if m['param_delta_norm'] > 0]

    # Per-camera summary
    per_camera_summary = {}
    cam_keys = set()
    for m in metrics_log:
        for k in m.get('per_camera_fixed_valid_count', {}):
            cam_keys.add(k)
    for cam_key in sorted(cam_keys):
        cos_list = [m['per_camera_fixed_cos'].get(cam_key) for m in metrics_log
                    if cam_key in m.get('per_camera_fixed_cos', {})]
        valid_list = [m['per_camera_fixed_valid_count'].get(cam_key, 0) for m in metrics_log
                      if cam_key in m.get('per_camera_fixed_valid_count', {})]
        if cos_list:
            per_camera_summary[cam_key] = {
                'first_cos': cos_list[0],
                'last_cos': cos_list[-1],
                'best_cos': max(cos_list),
                'mean_valid': float(np.mean(valid_list)),
            }

    # Verdict
    mean_param_delta = float(np.mean(param_deltas)) if param_deltas else 0
    avg_num_gaussians = np.mean([m['num_gaussians_mean'] for m in metrics_log]) if metrics_log else 0

    if total_nan > 0 or total_inf > 0:
        verdict = "phase12c_numerical_failure"
    elif mean_param_delta < 1e-8:
        verdict = "phase12c_optimizer_failure"
    elif avg_num_gaussians < 1 and total_skipped > args.num_steps * 0.5:
        verdict = "phase12c_projection_failure"
    elif fixed_cos_delta > 0.10 and fixed_loss_delta < 0 and (last_fixed_gap is None or last_fixed_gap > 0):
        verdict = "phase12c_strong_success"
    elif fixed_cos_delta > 0.05 and fixed_loss_delta < 0 and (fixed_gap_delta is None or fixed_gap_delta > -0.05):
        verdict = "phase12c_teacher_alignment_success"
    elif first_fixed_cos > 0 and last_fixed_cos <= first_fixed_cos + 0.02 and mean_param_delta > 1e-6:
        verdict = "phase12c_overfit_no_generalization"
    else:
        verdict = "phase12c_learning_failure"

    summary = {
        'first_fixed_cos': float(first_fixed_cos),
        'best_fixed_cos': float(best_fixed_cos_final),
        'last_fixed_cos': float(last_fixed_cos),
        'fixed_cos_delta': float(fixed_cos_delta),
        'first_fixed_loss': float(first_fixed_loss),
        'last_fixed_loss': float(last_fixed_loss),
        'fixed_loss_delta': float(fixed_loss_delta),
        'first_fixed_gap': float(first_fixed_gap) if first_fixed_gap is not None else None,
        'last_fixed_gap': float(last_fixed_gap) if last_fixed_gap is not None else None,
        'fixed_gap_delta': float(fixed_gap_delta) if fixed_gap_delta is not None else None,
        'best_step': best_step,
        'max_grad_norm': max(m['grad_norm_before_clip'] for m in metrics_log) if metrics_log else 0,
        'mean_param_delta_norm': mean_param_delta,
        'skipped_step_count': total_skipped,
        'nan_total': total_nan,
        'inf_total': total_inf,
        'per_camera_summary': per_camera_summary,
        'verdict': verdict,
    }

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"PHASE 12C RESULT")
    print(f"{'='*70}")
    print(f"First fixed cos: {first_fixed_cos:.4f}")
    print(f"Best fixed cos:  {best_fixed_cos_final:.4f} (step {best_step})")
    print(f"Last fixed cos:  {last_fixed_cos:.4f}")
    print(f"Cos delta:       {fixed_cos_delta:+.4f}")
    print(f"Loss delta:      {fixed_loss_delta:+.4f}")
    if last_fixed_gap is not None:
        print(f"Last gap:        {last_fixed_gap:.4f}")
        print(f"Gap delta:       {fixed_gap_delta:+.4f}")
    print(f"Mean param delta: {mean_param_delta:.6e}")
    print(f"Skipped steps:   {total_skipped}/{args.num_steps}")
    print(f"NaN/Inf:         {total_nan}/{total_inf}")
    print(f"VERDICT:         {verdict}")

    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/apps/wildtrack_full_3dgut.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs/phase12c_gaussianset_clean_random_fixed_eval_lr1e3')
    parser.add_argument('--person_feature_lr', type=float, default=1e-3)
    parser.add_argument('--num_steps', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--allowed_cameras', type=str, default='C1,C4,C6,C7')
    parser.add_argument('--pre_collect_pool_size', type=int, default=2000)
    parser.add_argument('--fixed_eval_size', type=int, default=16)
    parser.add_argument('--P', type=int, default=4)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--min_bbox_area', type=int, default=100)
    parser.add_argument('--grad_clip_norm', type=float, default=10.0)
    parser.add_argument('--denom_eps', type=float, default=1e-8)
    parser.add_argument('--candidate_pool_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    allowed_cameras = [c.strip() for c in args.allowed_cameras.split(',')] if args.allowed_cameras else None

    config_dir = os.path.join(REPO_ROOT, 'configs')
    config_file = 'apps/' + os.path.basename(args.config).replace('.yaml', '')

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_file)

    cfg.model.person_feature_dim = 512

    from threedgrut.trainer import Trainer3DGRUT

    trainer = Trainer3DGRUT(cfg)

    batch_builder = BatchBuilder(trainer.train_dataset)

    verdict = run_phase12c(args, trainer, batch_builder, allowed_cameras)
    print(f"\nPhase 12C verdict: {verdict}")
    sys.exit(0)


if __name__ == '__main__':
    main()
