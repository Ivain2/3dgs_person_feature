#!/usr/bin/env python3
"""
Phase 12F: Gaussian-Set + EMA Prototype + Stop-Gradient Multi-View InfoNCE

Goal: Stabilize cross_view_gap via EMA prototype tracking while maintaining
teacher alignment and improving identity discrimination.

Loss: L_total = L_teacher + lambda_mv * L_mv + lambda_proto * L_proto
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from rich import pretty, traceback

pretty.install()
traceback.install()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from hydra import initialize_config_dir, compose


def normalize_feat(x, eps=1e-6):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def gaussian_set_pooling(model, gpu_batch, bbox, cam_id, frame_id, args, device):
    """Direct Gaussian xyz projection pooling (Phase 12A/12B method)."""
    x1, y1, x2, y2 = bbox
    try:
        xyz = model.positions
        opacity = model.get_density().squeeze(-1)
        person_feature = model.get_person_feature()

        N = xyz.shape[0]
        if N == 0:
            return None, {'num_gaussians_in_bbox': 0, 'weight_sum': 0.0, 'failure_reason': 'no_gaussians'}

        intrinsics = gpu_batch.intrinsics
        if intrinsics is None or len(intrinsics) < 4:
            return None, {'num_gaussians_in_bbox': 0, 'weight_sum': 0.0, 'failure_reason': 'no_intrinsics'}

        fx, fy, cx, cy = intrinsics
        T_to_world = gpu_batch.T_to_world[0]
        R_world_to_cam = T_to_world[:3, :3].t()
        t_world_to_cam = -R_world_to_cam @ T_to_world[:3, 3]

        xyz_cam = (R_world_to_cam @ xyz.t()).t() + t_world_to_cam
        depth = xyz_cam[:, 2]
        valid_depth = depth > 0

        x_img = fx * xyz_cam[:, 0] / xyz_cam[:, 2] + cx
        y_img = fy * xyz_cam[:, 1] / xyz_cam[:, 2] + cy
        h_img, w_img = gpu_batch.rays_dir.shape[1], gpu_batch.rays_dir.shape[2]

        x_finite = torch.isfinite(x_img)
        y_finite = torch.isfinite(y_img)
        x_in_bounds = (x_img >= 0) & (x_img < w_img)
        y_in_bounds = (y_img >= 0) & (y_img < h_img)
        opacity_positive = opacity > 0

        valid = valid_depth & x_finite & y_finite & x_in_bounds & y_in_bounds & opacity_positive
        inside_bbox = (x_img >= x1) & (x_img < x2) & (y_img >= y1) & (y_img < y2)
        inside = valid & inside_bbox

        if inside.sum() == 0:
            return None, {'num_gaussians_in_bbox': 0, 'weight_sum': 0.0, 'failure_reason': 'no_gaussians_in_bbox'}

        weights = opacity[inside]
        z = person_feature[inside]
        weight_sum = weights.sum()
        denom = weight_sum.clamp(min=args.denom_eps)
        weighted_sum = (weights[:, None] * z).sum(dim=0)
        G = weighted_sum / denom
        G = normalize_feat(G)

        num_inside = int(inside.sum().item())
        depth_inside = depth[inside]
        return G, {
            'num_gaussians_in_bbox': num_inside, 'weight_sum': float(weight_sum.item()),
            'weight_min': float(weights.min().item()), 'weight_mean': float(weights.mean().item()),
            'weight_max': float(weights.max().item()), 'depth_min': float(depth_inside.min().item()),
            'depth_mean': float(depth_inside.mean().item()), 'depth_max': float(depth_inside.max().item()),
            'bbox': list(bbox), 'cam_id': cam_id, 'frame_id': frame_id,
        }
    except Exception as e:
        return None, {'num_gaussians_in_bbox': 0, 'weight_sum': 0.0, 'failure_reason': f'{str(e)[:60]}'}


class BatchBuilder:
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
    print(f"\nBuilding candidate pool (target: {args.pre_collect_pool_size})...")
    dataset = trainer.train_dataset
    candidates = []
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
            teacher_emb = None
            if dataset.teacher_cache is not None:
                bbox_dict = ann.get('bbox', {})
                if isinstance(bbox_dict, dict) and len(bbox_dict) >= 4:
                    cam_id_str = f"C{ann_cam_id + 1}"
                    x1_c = int(bbox_dict.get('xmin', 0))
                    y1_c = int(bbox_dict.get('ymin', 0))
                    x2_c = int(bbox_dict.get('xmax', 0))
                    y2_c = int(bbox_dict.get('ymax', 0))
                    cache_key = (int(frame_idx), cam_id_str, int(pid), x1_c, y1_c, x2_c, y2_c)
                    cache_entry = dataset.teacher_cache.get(cache_key)
                    if cache_entry is not None:
                        teacher_emb = cache_entry.get('embedding')
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
            gpu_batch = batch_builder.get_batch(cam_id, int(frame_idx))
            if gpu_batch is None:
                continue
            G, debug_info = gaussian_set_pooling(
                trainer.model, gpu_batch, [x1, y1, x2, y2], cam_id, int(frame_idx), args, trainer.device
            )
            if G is None or debug_info['num_gaussians_in_bbox'] == 0 or debug_info['weight_sum'] <= 0:
                continue
            candidates.append({
                'person_id': int(pid), 'cam_id': cam_id, 'frame_idx': int(frame_idx),
                'dataset_index': idx, 'bbox': [x1, y1, x2, y2], 'bbox_area': int(bbox_area),
                'teacher_emb': teacher_emb.cpu().numpy().tolist() if hasattr(teacher_emb, 'cpu') else teacher_emb.tolist(),
                'num_gaussians_in_bbox': debug_info['num_gaussians_in_bbox'],
                'weight_sum': float(debug_info['weight_sum']),
            })
            if len(candidates) >= args.pre_collect_pool_size:
                break
        if len(candidates) >= args.pre_collect_pool_size:
            break
    print(f"Candidate pool: {len(candidates)} samples")
    if candidates:
        print(f"  {len(set(c['person_id'] for c in candidates))} unique persons, "
              f"{len(set(c['cam_id'] for c in candidates))} unique cameras")
    return candidates


def select_fixed_eval_samples(candidate_pool, fixed_eval_size=16):
    """Select fixed eval samples with multi-view persons for gap computation."""
    print(f"\nSelecting {fixed_eval_size} fixed eval samples...")
    if not candidate_pool:
        return []
    person_to_samples = defaultdict(list)
    for c in candidate_pool:
        person_to_samples[c['person_id']].append(c)
    multi_view_persons = {pid: samples for pid, samples in person_to_samples.items()
                          if len(set(s['cam_id'] for s in samples)) >= 2}
    single_view_persons = {pid: samples for pid, samples in person_to_samples.items()
                           if pid not in multi_view_persons}
    selected, used_keys = [], set()
    num_multi_view = min(len(multi_view_persons), fixed_eval_size // 2)
    sorted_mv = sorted(multi_view_persons.keys(), key=lambda p: len(multi_view_persons[p]), reverse=True)
    for pid in sorted_mv:
        if num_multi_view <= 0:
            break
        cam_to_sample = {}
        for s in multi_view_persons[pid]:
            if s['cam_id'] not in cam_to_sample:
                cam_to_sample[s['cam_id']] = s
                used_keys.add((s['person_id'], s['cam_id'], s['frame_idx']))
                selected.append(s)
                if len(cam_to_sample) >= 2:
                    break
        if len(cam_to_sample) >= 2:
            num_multi_view -= 1
        if len(selected) >= fixed_eval_size:
            break
    remaining_slots = fixed_eval_size - len(selected)
    if remaining_slots > 0 and single_view_persons:
        for pid in sorted(single_view_persons.keys(), key=lambda p: len(single_view_persons[p]), reverse=True):
            if remaining_slots <= 0:
                break
            for s in single_view_persons[pid]:
                sk = (s['person_id'], s['cam_id'], s['frame_idx'])
                if sk not in used_keys:
                    selected.append(s)
                    used_keys.add(sk)
                    remaining_slots -= 1
                    break
    if len(selected) < fixed_eval_size:
        remaining = [c for c in candidate_pool if (c['person_id'], c['cam_id'], c['frame_idx']) not in used_keys]
        selected.extend(random.sample(remaining, min(fixed_eval_size - len(selected), len(remaining))))
    mv_count = sum(1 for pid in set(c['person_id'] for c in selected)
                   if len(set(c['cam_id'] for c in selected if c['person_id'] == pid)) >= 2)
    print(f"Selected {len(selected)} fixed eval samples, {len(set(c['person_id'] for c in selected))} persons, "
          f"{mv_count} with >=2 views")
    return selected


def compute_stopgrad_infonce(features, person_ids, tau=0.2):
    """
    Stop-gradient InfoNCE.
    features: [B, D], person_ids: list of int, tau: float.
    Returns: L_mv, valid_anchor_count, same_cos, diff_cos, gap, pos_count, neg_count.
    """
    B = len(features)
    if B < 2:
        return None, 0, None, None, None, 0, 0

    target_bank = torch.stack([f.detach() for f in features])  # [B, D]

    person_to_indices = defaultdict(list)
    for i, pid in enumerate(person_ids):
        person_to_indices[pid].append(i)

    losses = []
    same_cos_list, diff_cos_list = [], []
    pos_count, neg_count, valid_anchor = 0, 0, 0

    for i in range(B):
        anchor = features[i]
        pid = person_ids[i]

        positives = [j for j in person_to_indices[pid] if j != i]
        negatives = [j for j in range(B) if person_ids[j] != pid]

        if not positives or not negatives:
            continue

        logits = torch.matmul(anchor, target_bank.T) / tau  # [B]
        pos_logits = torch.stack([logits[j] for j in positives])
        neg_logits = torch.stack([logits[j] for j in negatives])

        all_logits = torch.cat([pos_logits, neg_logits])
        loss_i = -torch.logsumexp(pos_logits, dim=0) + torch.logsumexp(all_logits, dim=0)

        losses.append(loss_i)
        valid_anchor += 1
        pos_count += len(positives)
        neg_count += len(negatives)

        for j in positives:
            sc = F.cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0)).item()
            same_cos_list.append(sc)
        for j in negatives[:min(len(negatives), 10)]:
            dc = F.cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0)).item()
            diff_cos_list.append(dc)

    if not losses:
        return None, 0, None, None, None, 0, 0

    L_mv = torch.stack(losses).mean()
    same_cos = float(np.mean(same_cos_list)) if same_cos_list else None
    diff_cos = float(np.mean(diff_cos_list)) if diff_cos_list else None
    gap = (same_cos - diff_cos) if (same_cos is not None and diff_cos is not None) else None
    return L_mv, valid_anchor, same_cos, diff_cos, gap, pos_count, neg_count


def compute_proto_loss(features, person_ids, prototypes, tau_proto=0.1):
    """
    Compute prototype alignment loss.
    For each feature, align to its person's EMA prototype vs all prototypes.
    """
    B = len(features)
    unique_pids = list(set(person_ids))
    if len(unique_pids) < 2 or B < 2:
        return None, 0

    proto_tensor = torch.stack([prototypes[pid] for pid in unique_pids])  # [P, D]
    pid_to_idx = {pid: idx for idx, pid in enumerate(unique_pids)}

    losses = []
    for i in range(B):
        pid = person_ids[i]
        p_idx = pid_to_idx[pid]
        logits = torch.matmul(features[i], proto_tensor.T) / tau_proto  # [P]
        loss_i = -logits[p_idx] + torch.logsumexp(logits, dim=0)
        losses.append(loss_i)

    if not losses:
        return None, 0

    L_proto = torch.stack(losses).mean()
    return L_proto, len(losses)


def update_ema_prototypes(ema_prototypes, person_features, person_ids, alpha=0.9):
    """Update EMA prototypes with new features."""
    person_to_features = defaultdict(list)
    for feat, pid in zip(person_features, person_ids):
        person_to_features[pid].append(feat)

    for pid, feats in person_to_features.items():
        avg_feat = torch.stack(feats).mean(dim=0)
        if pid not in ema_prototypes:
            ema_prototypes[pid] = avg_feat.detach().clone()
        else:
            ema_prototypes[pid] = alpha * ema_prototypes[pid] + (1 - alpha) * normalize_feat(avg_feat.unsqueeze(0)).squeeze(0)
            ema_prototypes[pid] = normalize_feat(ema_prototypes[pid].unsqueeze(0)).squeeze(0)


def run_fixed_eval(args, model, batch_builder, fixed_eval_samples, device, ema_prototypes=None):
    """Evaluate fixed eval samples: teacher loss + InfoNCE + proto alignment."""
    losses_t, cos_values, gaussianset_features, valid_samples = [], [], [], []
    num_gaussians_list, weight_sum_list, invalid_count = [], [], 0
    per_camera_cos = defaultdict(list)

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
        gaussianset_features.append(G)
        valid_samples.append(sample)
        losses_t.append(1.0 - cos_sim)
        cos_values.append(cos_sim)
        num_gaussians_list.append(debug_info['num_gaussians_in_bbox'])
        weight_sum_list.append(debug_info['weight_sum'])
        per_camera_cos[sample['cam_id']].append(cos_sim)

    if not losses_t:
        return {'fixed_loss_teacher': 1.0, 'fixed_cos_mean': 0.0, 'fixed_valid_roi_count': 0,
                'fixed_same_cos': None, 'fixed_diff_cos': None, 'fixed_cross_view_gap': None,
                'fixed_L_mv': None, 'fixed_L_proto': None}

    loss_t = float(np.mean(losses_t))
    cos_mean, cos_min, cos_max = float(np.mean(cos_values)), float(np.min(cos_values)), float(np.max(cos_values))

    person_to_indices = defaultdict(list)
    for idx, s in enumerate(valid_samples):
        person_to_indices[s['person_id']].append(idx)
    same_cos_list, diff_cos_list = [], []
    for pid, indices in person_to_indices.items():
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sc = F.cosine_similarity(
                        gaussianset_features[indices[i]].unsqueeze(0),
                        gaussianset_features[indices[j]].unsqueeze(0)
                    ).item()
                    same_cos_list.append(sc)
    all_indices = list(range(len(gaussianset_features)))
    for _ in range(min(50, len(all_indices) * (len(all_indices) - 1) // 2)):
        i, j = random.sample(all_indices, 2)
        if valid_samples[i]['person_id'] != valid_samples[j]['person_id']:
            dc = F.cosine_similarity(
                gaussianset_features[i].unsqueeze(0), gaussianset_features[j].unsqueeze(0)
            ).item()
            diff_cos_list.append(dc)

    same_cos = float(np.mean(same_cos_list)) if same_cos_list else None
    diff_cos = float(np.mean(diff_cos_list)) if diff_cos_list else None
    gap = (same_cos - diff_cos) if (same_cos is not None and diff_cos is not None) else None

    L_mv, _, mv_same, mv_diff, mv_gap, pos_c, neg_c = compute_stopgrad_infonce(
        gaussianset_features, [s['person_id'] for s in valid_samples], args.tau_mv
    )

    L_proto = None
    if ema_prototypes and len(ema_prototypes) >= 2:
        pids = [s['person_id'] for s in valid_samples]
        filtered_feats, filtered_pids = [], []
        for feat, pid in zip(gaussianset_features, pids):
            if pid in ema_prototypes:
                filtered_feats.append(feat)
                filtered_pids.append(pid)
        if len(set(filtered_pids)) >= 2 and len(filtered_feats) >= 2:
            L_proto, _ = compute_proto_loss(filtered_feats, filtered_pids, ema_prototypes, tau_proto=getattr(args, 'tau_proto', 0.1))

    return {
        'fixed_loss_teacher': loss_t, 'fixed_cos_mean': cos_mean,
        'fixed_cos_min': cos_min, 'fixed_cos_max': cos_max,
        'fixed_valid_roi_count': len(losses_t), 'fixed_invalid_roi_count': invalid_count,
        'fixed_num_gaussians_mean': float(np.mean(num_gaussians_list)),
        'fixed_same_cos': same_cos, 'fixed_diff_cos': diff_cos,
        'fixed_cross_view_gap': gap, 'fixed_L_mv': float(L_mv.item()) if L_mv is not None else None,
        'fixed_mv_same_cos': mv_same, 'fixed_mv_diff_cos': mv_diff, 'fixed_mv_gap': mv_gap,
        'fixed_positive_pair_count': pos_c, 'fixed_negative_pair_count': neg_c,
        'fixed_L_proto': float(L_proto.item()) if L_proto is not None else None,
        'per_camera_fixed_cos': {k: float(np.mean(v)) for k, v in per_camera_cos.items()},
    }


def run_phase12f(args, trainer, batch_builder, allowed_cameras):
    print(f"\n{'='*70}")
    print(f"PHASE 12F: GAUSSIAN-SET + EMA PROTOTYPE + STOP-GRADIENT INFO-NCE")
    print(f"{'='*70}")
    print(f"lambda_mv={args.lambda_mv}, lambda_proto={args.lambda_proto}")
    print(f"tau_mv={args.tau_mv}, tau_proto={args.tau_proto}")
    print(f"P={args.P}, K={args.K}, ema_alpha={args.ema_alpha}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    pool_path = getattr(args, 'candidate_pool_path', None)
    if pool_path and os.path.exists(pool_path):
        with open(pool_path) as f:
            candidate_pool = json.load(f)
    else:
        candidate_pool = build_candidate_pool(args, trainer, batch_builder, allowed_cameras)
        with open(os.path.join(args.output_dir, 'candidate_pool.json'), 'w') as f:
            json.dump(candidate_pool, f, indent=2, default=str)

    if not candidate_pool:
        print("ERROR: No candidates"); return False

    fixed_eval_samples = select_fixed_eval_samples(candidate_pool, getattr(args, 'fixed_eval_size', 16))
    with open(os.path.join(args.output_dir, 'fixed_eval_samples.json'), 'w') as f:
        json.dump(fixed_eval_samples, f, indent=2, default=str)

    pf_before = trainer.model.get_person_feature().clone().detach()
    optimizer = torch.optim.Adam([trainer.model._person_feature], lr=args.person_feature_lr)

    ema_prototypes = {}
    metrics_log, best_fixed_cos, best_step = [], -1.0, 0
    total_nan, total_inf, total_skipped = 0, 0, 0

    print(f"\nTRAINING: pool_size={len(candidate_pool)}, fixed_eval={len(fixed_eval_samples)}")

    with torch.no_grad():
        eval_result = run_fixed_eval(args, trainer.model, batch_builder, fixed_eval_samples, trainer.device, ema_prototypes)
    print(f"[EVAL-0] fixed_cos={eval_result['fixed_cos_mean']:.4f} "
          f"loss={eval_result['fixed_loss_teacher']:.4f} "
          f"gap={eval_result.get('fixed_cross_view_gap') or 0:+.4f}")

    torch.save({'model_state_dict': {'_person_feature': trainer.model._person_feature.cpu().clone()}},
               os.path.join(args.output_dir, 'checkpoint_initial.pt'))

    batch_size = args.P * args.K
    for step in range(args.num_steps):
        step_start = time.time()
        if len(candidate_pool) < batch_size:
            total_skipped += 1; continue
        batch_samples = random.sample(candidate_pool, batch_size)

        batch_by_cam_frame = defaultdict(list)
        for s in batch_samples:
            batch_by_cam_frame[(s['cam_id'], s['frame_idx'])].append(s)

        optimizer.zero_grad()
        losses_t, gaussianset_features, valid_samples = [], [], []
        num_gaussians_list, weight_sum_list, invalid_count = [], [], 0

        for (cam_id, frame_idx), samples in batch_by_cam_frame.items():
            gpu_batch = batch_builder.get_batch(cam_id, frame_idx)
            if gpu_batch is None:
                invalid_count += len(samples); continue
            for sample in samples:
                G, debug_info = gaussian_set_pooling(
                    trainer.model, gpu_batch, sample['bbox'], sample['cam_id'], sample['frame_idx'],
                    args, trainer.device
                )
                if G is None:
                    invalid_count += 1; continue
                T = torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=trainer.device)
                T = normalize_feat(T)
                cos_sim = F.cosine_similarity(G.unsqueeze(0), T.unsqueeze(0))
                gaussianset_features.append(G)
                valid_samples.append(sample)
                losses_t.append(1.0 - cos_sim)
                num_gaussians_list.append(debug_info['num_gaussians_in_bbox'])
                weight_sum_list.append(debug_info['weight_sum'])

        valid_count = len(losses_t)
        if valid_count == 0:
            total_skipped += 1; continue

        L_teacher = torch.stack(losses_t).mean()

        person_ids = [s['person_id'] for s in valid_samples]

        L_mv, valid_anchor, same_cos, diff_cos, gap, pos_c, neg_c = compute_stopgrad_infonce(
            gaussianset_features, person_ids, args.tau_mv
        )

        L_proto = None
        proto_valid = 0
        if ema_prototypes and len(ema_prototypes) >= 2:
            filtered_feats, filtered_pids = [], []
            for feat, pid in zip(gaussianset_features, person_ids):
                if pid in ema_prototypes:
                    filtered_feats.append(feat)
                    filtered_pids.append(pid)
            if len(set(filtered_pids)) >= 2 and len(filtered_feats) >= 2:
                L_proto, proto_valid = compute_proto_loss(filtered_feats, filtered_pids, ema_prototypes, tau_proto=args.tau_proto)

        L_total = L_teacher
        if L_mv is not None:
            L_total = L_total + args.lambda_mv * L_mv
        if L_proto is not None:
            L_total = L_total + args.lambda_proto * L_proto

        L_total.backward()

        grad_norm_before = None; grad_norm_after = None
        if trainer.model._person_feature.grad is not None:
            grad_norm_before = trainer.model._person_feature.grad.norm().item()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([trainer.model._person_feature], args.grad_clip_norm)
            grad_norm_after = trainer.model._person_feature.grad.norm().item()

        optimizer.step()
        pf_after = trainer.model.get_person_feature()
        param_delta_norm = (pf_after - pf_before).norm().item()
        pf_before = pf_after.clone().detach()

        with torch.no_grad():
            update_ema_prototypes(ema_prototypes, gaussianset_features, person_ids, alpha=args.ema_alpha)

        cos_values = [1.0 - l.item() for l in losses_t]
        cos_mean = float(np.mean(cos_values)) if cos_values else 0

        nan_count = int(torch.isnan(trainer.model._person_feature).sum().item())
        inf_count = int(torch.isinf(trainer.model._person_feature).sum().item())
        total_nan += nan_count; total_inf += inf_count

        step_time = time.time() - step_start

        if step % args.log_interval == 0:
            log_line = f"[PHASE12F] Step {step:5d}: train_loss={L_total.item():.4f} " \
                       f"teacher={L_teacher.item():.4f} " \
                       f"mv={L_mv.item() if L_mv is not None else 0:.4f} " \
                       f"proto={L_proto.item() if L_proto is not None else 0:.4f} " \
                       f"train_cos={cos_mean:.4f} " \
                       f"delta={param_delta_norm:.6e} valid={valid_count}/{batch_size} t={step_time:.2f}s"
            if gap is not None:
                log_line += f" same={same_cos:.4f} diff={diff_cos:.4f} gap={gap:+.4f}"
            print(log_line)

        fixed_eval_result = {}
        if step % args.eval_interval == 0 or step == args.num_steps - 1:
            with torch.no_grad():
                fixed_eval_result = run_fixed_eval(args, trainer.model, batch_builder, fixed_eval_samples, trainer.device, ema_prototypes)
            fc = fixed_eval_result['fixed_cos_mean']
            fl = fixed_eval_result['fixed_loss_teacher']
            fg = fixed_eval_result.get('fixed_cross_view_gap')
            gap_str = f"gap={fg:+.4f}" if fg is not None else "gap=N/A"
            print(f"  [EVAL] fixed_cos={fc:.4f} loss={fl:.4f} {gap_str} "
                  f"valid={fixed_eval_result['fixed_valid_roi_count']}/{len(fixed_eval_samples)}")
            if fc > best_fixed_cos:
                best_fixed_cos = fc; best_step = step
                torch.save({
                    'model_state_dict': {'_person_feature': trainer.model._person_feature.cpu().clone()},
                    'step': step, 'fixed_cos': fc,
                }, os.path.join(args.output_dir, 'checkpoint_best_fixed_cos.pt'))

        step_record = {
            'step': step, 'loss_total': float(L_total.item()),
            'loss_teacher': float(L_teacher.item()),
            'loss_mv': float(L_mv.item()) if L_mv is not None else None,
            'loss_proto': float(L_proto.item()) if L_proto is not None else None,
            'train_cos_mean': cos_mean, 'valid_roi_count': valid_count,
            'invalid_roi_count': invalid_count, 'skipped_step_count': total_skipped,
            'train_same_cos': same_cos, 'train_diff_cos': diff_cos,
            'train_cross_view_gap': gap, 'valid_anchor_count': valid_anchor,
            'positive_pair_count': pos_c, 'negative_pair_count': neg_c,
            'grad_norm_before_clip': float(grad_norm_before) if grad_norm_before else 0,
            'grad_norm_after_clip': float(grad_norm_after) if grad_norm_after else 0,
            'param_delta_norm': float(param_delta_norm),
            'nan_count': nan_count, 'inf_count': inf_count,
            'ema_prototype_count': len(ema_prototypes),
        }
        step_record.update(fixed_eval_result)
        metrics_log.append(step_record)

    with open(os.path.join(args.output_dir, 'metrics.jsonl'), 'w') as f:
        for r in metrics_log:
            f.write(json.dumps(r, default=str) + "\n")

    torch.save({'model_state_dict': {'_person_feature': trainer.model._person_feature.cpu().clone()},
                'step': args.num_steps - 1, 'ema_prototypes': {k: v.cpu() for k, v in ema_prototypes.items()}},
               os.path.join(args.output_dir, 'checkpoint_latest.pt'))

    fixed_cos_values = [m['fixed_cos_mean'] for m in metrics_log if 'fixed_cos_mean' in m]
    fixed_gap_values = [m['fixed_cross_view_gap'] for m in metrics_log if m.get('fixed_cross_view_gap') is not None]
    fixed_diff_values = [m['fixed_diff_cos'] for m in metrics_log if m.get('fixed_diff_cos') is not None]

    first_cos = fixed_cos_values[0] if fixed_cos_values else 0
    best_cos = max(fixed_cos_values) if fixed_cos_values else 0
    last_cos = fixed_cos_values[-1] if fixed_cos_values else 0
    first_gap = fixed_gap_values[0] if fixed_gap_values else None
    last_gap = fixed_gap_values[-1] if fixed_gap_values else None
    best_gap = max(fixed_gap_values) if fixed_gap_values else None
    first_diff = fixed_diff_values[0] if fixed_diff_values else None
    last_diff = fixed_diff_values[-1] if fixed_diff_values else None
    diff_cos_delta = (last_diff - first_diff) if (last_diff is not None and first_diff is not None) else None
    gap_delta = (last_gap - first_gap) if (last_gap is not None and first_gap is not None) else None

    param_deltas = [m['param_delta_norm'] for m in metrics_log if m['param_delta_norm'] > 0]
    mean_param_delta = float(np.mean(param_deltas)) if param_deltas else 0

    if total_nan > 0 or total_inf > 0 or mean_param_delta < 1e-8:
        verdict = "phase12f_failure"
    elif last_cos >= 0.95 and last_gap is not None and last_gap > 0.05 and (diff_cos_delta is None or diff_cos_delta < 0):
        verdict = "phase12f_mv_success"
    elif last_cos >= 0.95 and last_gap is not None and last_gap > (first_gap or 0):
        verdict = "phase12f_mv_partial_success"
    elif last_gap is not None and last_gap > (first_gap or 0) and last_cos < 0.90:
        verdict = "phase12f_mv_too_strong"
    elif last_cos >= 0.90 and (last_gap is None or last_gap <= (first_gap or 0)):
        verdict = "phase12f_mv_no_gap_gain"
    else:
        verdict = "phase12f_failure"

    summary = {
        'lambda_mv': args.lambda_mv, 'lambda_proto': args.lambda_proto,
        'tau_mv': args.tau_mv, 'tau_proto': args.tau_proto, 'ema_alpha': args.ema_alpha,
        'first_fixed_cos': float(first_cos), 'best_fixed_cos': float(best_cos),
        'last_fixed_cos': float(last_cos), 'fixed_cos_delta': float(last_cos - first_cos),
        'first_fixed_gap': float(first_gap) if first_gap is not None else None,
        'best_fixed_gap': float(best_gap) if best_gap is not None else None,
        'last_fixed_gap': float(last_gap) if last_gap is not None else None,
        'fixed_gap_delta': float(gap_delta) if gap_delta is not None else None,
        'first_fixed_diff_cos': float(first_diff) if first_diff is not None else None,
        'last_fixed_diff_cos': float(last_diff) if last_diff is not None else None,
        'diff_cos_delta': float(diff_cos_delta) if diff_cos_delta is not None else None,
        'best_step': best_step,
        'max_grad_norm': max(m['grad_norm_before_clip'] for m in metrics_log) if metrics_log else 0,
        'mean_param_delta_norm': mean_param_delta,
        'nan_total': total_nan, 'inf_total': total_inf,
        'final_ema_prototype_count': len(ema_prototypes),
        'verdict': verdict,
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"PHASE 12F RESULT (lambda_mv={args.lambda_mv}, lambda_proto={args.lambda_proto})")
    print(f"{'='*70}")
    print(f"First/B/Last fixed_cos: {first_cos:.4f} / {best_cos:.4f} / {last_cos:.4f}")
    print(f"First/B/Last gap:       {first_gap} / {best_gap} / {last_gap}")
    print(f"Diff_cos delta:         {diff_cos_delta}")
    print(f"Mean param delta:       {mean_param_delta:.6e}")
    print(f"NaN/Inf:                {total_nan}/{total_inf}")
    print(f"EMA prototypes:         {len(ema_prototypes)}")
    print(f"VERDICT:                {verdict}")
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs/phase12f_gaussianset_ema_proto_mv_lam01_proto005')
    parser.add_argument('--person_feature_lr', type=float, default=1e-3)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--allowed_cameras', type=str, default='C1,C4,C6,C7')
    parser.add_argument('--pre_collect_pool_size', type=int, default=2000)
    parser.add_argument('--fixed_eval_size', type=int, default=16)
    parser.add_argument('--P', type=int, default=4)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--min_bbox_area', type=int, default=100)
    parser.add_argument('--lambda_teacher', type=float, default=1.0)
    parser.add_argument('--lambda_mv', type=float, default=0.1)
    parser.add_argument('--lambda_proto', type=float, default=0.05)
    parser.add_argument('--tau_mv', type=float, default=0.2)
    parser.add_argument('--tau_proto', type=float, default=0.1)
    parser.add_argument('--ema_alpha', type=float, default=0.9)
    parser.add_argument('--grad_clip_norm', type=float, default=10.0)
    parser.add_argument('--denom_eps', type=float, default=1e-8)
    parser.add_argument('--candidate_pool_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    allowed_cameras = [c.strip() for c in args.allowed_cameras.split(',')] if args.allowed_cameras else None

    config_dir = os.path.join(REPO_ROOT, 'configs')
    config_file = 'apps/wildtrack_full_3dgut'

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_file)
    cfg.model.person_feature_dim = 512

    from threedgrut.trainer import Trainer3DGRUT
    trainer = Trainer3DGRUT(cfg)
    batch_builder = BatchBuilder(trainer.train_dataset)
    run_phase12f(args, trainer, batch_builder, allowed_cameras)
    sys.exit(0)


if __name__ == '__main__':
    main()
