#!/usr/bin/env python3
"""
Phase 12: Gaussian-Set Final ReID Evaluation

Compares multiple methods using the same fixed_eval_samples:
1. 2D ReID Teacher
2. ROI baseline 11B-v4 (if checkpoint available)
3. Gaussian-Set 12C teacher-only
4. Gaussian-Set 12E MV InfoNCE
5. Gaussian-Set 12F EMA Proto

Outputs:
- eval_summary.json
- retrieval_metrics.json
- similarity_matrix.npy
- per_camera_metrics.json
- per_identity_metrics.json
- final_report.md
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from hydra import initialize_config_dir, compose


def normalize_feat(x, eps=1e-6):
    """L2 normalize features."""
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def compute_retrieval_metrics(sim_matrix, person_ids, cam_ids, exclude_self=True):
    """
    Compute Top-1, Top-5, mAP for retrieval.
    sim_matrix: [N, N]
    person_ids: list of int
    cam_ids: list of str
    exclude_self: exclude same frame from retrieval
    """
    N = len(person_ids)
    top1_correct = 0
    top5_correct = 0
    aps = []

    for i in range(N):
        query_pid = person_ids[i]
        query_cam = cam_ids[i]

        sims = sim_matrix[i].copy()

        if exclude_self:
            for j in range(N):
                if person_ids[j] == query_pid and cam_ids[j] == query_cam:
                    sims[j] = -np.inf

        ranked_indices = np.argsort(sims)[::-1]

        gt_indices = [j for j in range(N) if person_ids[j] == query_pid and j != i]
        if not gt_indices:
            continue

        top1_idx = ranked_indices[0]
        if top1_idx in gt_indices:
            top1_correct += 1

        top5_indices = ranked_indices[:5]
        if any(idx in gt_indices for idx in top5_indices):
            top5_correct += 1

        ap = 0.0
        num_gt = len(gt_indices)
        tp_count = 0
        for rank, idx in enumerate(ranked_indices):
            if idx in gt_indices:
                tp_count += 1
                ap += tp_count / (rank + 1)
        ap /= num_gt
        aps.append(ap)

    total_queries = len([i for i in range(N) if any(person_ids[j] == person_ids[i] for j in range(N) if j != i)])
    top1 = top1_correct / total_queries if total_queries > 0 else 0
    top5 = top5_correct / total_queries if total_queries > 0 else 0
    mAP = np.mean(aps) if aps else 0

    return top1, top5, mAP


def compute_cosine_metrics(features, person_ids, cam_ids, teacher_features=None):
    """
    Compute same_cos, diff_cos, gap, cos_to_teacher, per_camera metrics.
    features: [N, D] or list of [D]
    person_ids: list of int
    cam_ids: list of str
    teacher_features: [N, D] optional
    """
    if isinstance(features, list):
        features = torch.stack(features)

    device = features.device

    N = len(features)
    same_cos_list = []
    diff_cos_list = []
    cos_to_teacher_list = []
    per_camera_data = defaultdict(lambda: {'same': [], 'diff': []})

    person_to_indices = defaultdict(list)
    for i, pid in enumerate(person_ids):
        person_to_indices[pid].append(i)

    if teacher_features is not None:
        teacher_features = teacher_features.to(device)

    for i in range(N):
        for j in range(i + 1, N):
            cos_ij = F.cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0)).item()

            if person_ids[i] == person_ids[j]:
                same_cos_list.append(cos_ij)
                per_camera_data[cam_ids[i]]['same'].append(cos_ij)
                per_camera_data[cam_ids[j]]['same'].append(cos_ij)
            else:
                diff_cos_list.append(cos_ij)
                per_camera_data[cam_ids[i]]['diff'].append(cos_ij)
                per_camera_data[cam_ids[j]]['diff'].append(cos_ij)

        if teacher_features is not None:
            cos_t = F.cosine_similarity(features[i].unsqueeze(0), teacher_features[i].unsqueeze(0)).item()
            cos_to_teacher_list.append(cos_t)

    same_cos = np.mean(same_cos_list) if same_cos_list else 0
    diff_cos = np.mean(diff_cos_list) if diff_cos_list else 0
    gap = same_cos - diff_cos
    cos_to_teacher = np.mean(cos_to_teacher_list) if cos_to_teacher_list else None

    per_camera_metrics = {}
    for cam_id, data in per_camera_data.items():
        cam_same = np.mean(data['same']) if data['same'] else None
        cam_diff = np.mean(data['diff']) if data['diff'] else None
        cam_gap = (cam_same - cam_diff) if (cam_same is not None and cam_diff is not None) else None
        per_camera_metrics[cam_id] = {
            'same_cos': cam_same,
            'diff_cos': cam_diff,
            'gap': cam_gap,
            'num_same_pairs': len(data['same']),
            'num_diff_pairs': len(data['diff']),
        }

    return {
        'same_cos': same_cos,
        'diff_cos': diff_cos,
        'gap': gap,
        'cos_to_teacher': cos_to_teacher,
        'per_camera': per_camera_metrics,
    }


def compute_per_identity_metrics(features, person_ids):
    """Compute per-identity retrieval accuracy."""
    if isinstance(features, list):
        features = torch.stack(features)

    N = len(features)
    person_to_indices = defaultdict(list)
    for i, pid in enumerate(person_ids):
        person_to_indices[pid].append(i)

    per_identity = {}
    for pid, indices in person_to_indices.items():
        if len(indices) < 2:
            continue

        identity_features = features[indices]
        sim_matrix = identity_features @ identity_features.T

        correct = 0
        total = 0
        for i in range(len(indices)):
            for j in range(len(indices)):
                if i != j:
                    query_sim = sim_matrix[i]
                    ranked = np.argsort(query_sim.detach().cpu().numpy())[::-1]
                    if ranked[0] == j:
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0
        per_identity[str(pid)] = {
            'accuracy': accuracy,
            'num_views': len(indices),
            'correct': correct,
            'total': total,
        }

    return per_identity


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
        denom = weight_sum.clamp(min=getattr(args, 'denom_eps', 1e-8))
        weighted_sum = (weights[:, None] * z).sum(dim=0)
        G = weighted_sum / denom
        G = normalize_feat(G)

        num_inside = int(inside.sum().item())
        depth_inside = depth[inside]
        return G, {
            'num_gaussians_in_bbox': num_inside, 'weight_sum': float(weight_sum.item()),
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


def load_checkpoint_person_feature(checkpoint_path, device):
    """Load person_feature from checkpoint."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None

    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt and '_person_feature' in ckpt['model_state_dict']:
        pf = ckpt['model_state_dict']['_person_feature']
        if hasattr(pf, 'to'):
            return pf.to(device)
    return None


def evaluate_method_12c(fixed_eval_samples, batch_builder, model, device, args):
    """Evaluate Phase 12C checkpoint."""
    print("  Evaluating Phase 12C (teacher-only)...")
    features, teacher_features, person_ids, cam_ids = [], [], [], []

    for sample in fixed_eval_samples:
        gpu_batch = batch_builder.get_batch(sample['cam_id'], sample['frame_idx'])
        if gpu_batch is None:
            continue

        G, _ = gaussian_set_pooling(
            model, gpu_batch, sample['bbox'], sample['cam_id'], sample['frame_idx'], args, device
        )
        if G is None:
            continue

        features.append(G)
        teacher_features.append(torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=device))
        person_ids.append(sample['person_id'])
        cam_ids.append(sample['cam_id'])

    if not features:
        return None

    return {
        'features': features,
        'teacher_features': teacher_features,
        'person_ids': person_ids,
        'cam_ids': cam_ids,
    }


def evaluate_method_12e(fixed_eval_samples, batch_builder, model, device, args):
    """Evaluate Phase 12E checkpoint."""
    print("  Evaluating Phase 12E (MV InfoNCE)...")
    features, teacher_features, person_ids, cam_ids = [], [], [], []

    for sample in fixed_eval_samples:
        gpu_batch = batch_builder.get_batch(sample['cam_id'], sample['frame_idx'])
        if gpu_batch is None:
            continue

        G, _ = gaussian_set_pooling(
            model, gpu_batch, sample['bbox'], sample['cam_id'], sample['frame_idx'], args, device
        )
        if G is None:
            continue

        features.append(G)
        teacher_features.append(torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=device))
        person_ids.append(sample['person_id'])
        cam_ids.append(sample['cam_id'])

    if not features:
        return None

    return {
        'features': features,
        'teacher_features': teacher_features,
        'person_ids': person_ids,
        'cam_ids': cam_ids,
    }


def evaluate_method_12f(fixed_eval_samples, batch_builder, model, device, args):
    """Evaluate Phase 12F checkpoint."""
    print("  Evaluating Phase 12F (EMA Proto + MV InfoNCE)...")
    features, teacher_features, person_ids, cam_ids = [], [], [], []

    for sample in fixed_eval_samples:
        gpu_batch = batch_builder.get_batch(sample['cam_id'], sample['frame_idx'])
        if gpu_batch is None:
            continue

        G, _ = gaussian_set_pooling(
            model, gpu_batch, sample['bbox'], sample['cam_id'], sample['frame_idx'], args, device
        )
        if G is None:
            continue

        features.append(G)
        teacher_features.append(torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=device))
        person_ids.append(sample['person_id'])
        cam_ids.append(sample['cam_id'])

    if not features:
        return None

    return {
        'features': features,
        'teacher_features': teacher_features,
        'person_ids': person_ids,
        'cam_ids': cam_ids,
    }


def evaluate_method_teacher(fixed_eval_samples):
    """Evaluate 2D ReID Teacher baseline."""
    print("  Evaluating 2D ReID Teacher...")
    features, person_ids, cam_ids = [], [], []

    for sample in fixed_eval_samples:
        teacher_emb = torch.tensor(sample['teacher_emb'], dtype=torch.float32)
        teacher_emb = normalize_feat(teacher_emb)

        features.append(teacher_emb)
        person_ids.append(sample['person_id'])
        cam_ids.append(sample['cam_id'])

    if not features:
        return None

    features_tensor = torch.stack(features)
    sim_matrix = features_tensor @ features_tensor.T

    return {
        'features': features,
        'teacher_features': features,
        'person_ids': person_ids,
        'cam_ids': cam_ids,
        'sim_matrix': sim_matrix.cpu().numpy(),
    }


def run_eval(args):
    """Run complete ReID evaluation."""
    print("=" * 70)
    print("Phase 12: Gaussian-Set Final ReID Evaluation")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load fixed eval samples
    with open(args.fixed_eval_samples) as f:
        fixed_eval_samples = json.load(f)

    # Filter by allowed cameras
    allowed_cameras = [c.strip() for c in args.allowed_cameras.split(',')] if args.allowed_cameras else None
    if allowed_cameras:
        fixed_eval_samples = [s for s in fixed_eval_samples if s['cam_id'] in allowed_cameras]

    print(f"Loaded {len(fixed_eval_samples)} fixed eval samples")
    print(f"Allowed cameras: {allowed_cameras}")

    unique_ids = set(s['person_id'] for s in fixed_eval_samples)
    print(f"Unique person IDs: {len(unique_ids)}")

    # ========== Method 1: 2D ReID Teacher ==========
    print("\n[1/5] Evaluating 2D ReID Teacher...")
    teacher_result = evaluate_method_teacher(fixed_eval_samples)
    if teacher_result is None:
        print("  ERROR: No teacher features found")
        sys.exit(1)

    teacher_features_tensor = torch.stack(teacher_result['teacher_features'])

    # ========== Initialize model for student methods ==========
    config_dir = os.path.join(REPO_ROOT, 'configs')
    config_file = 'apps/wildtrack_full_3dgut'

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_file)
    cfg.model.person_feature_dim = 512

    from threedgrut.trainer import Trainer3DGRUT
    trainer = Trainer3DGRUT(cfg)
    model = trainer.model
    device = trainer.device
    batch_builder = BatchBuilder(trainer.train_dataset)

    # ========== Method 2: ROI baseline 11B-v4 ==========
    roi_result = None
    if args.roi_checkpoint and os.path.exists(args.roi_checkpoint):
        print("\n[2/5] Evaluating ROI baseline 11B-v4...")
        roi_pf = load_checkpoint_person_feature(args.roi_checkpoint, device)
        if roi_pf is not None:
            model._person_feature.data.copy_(roi_pf)
            roi_result = evaluate_method_12c(fixed_eval_samples, batch_builder, model, device, args)
            print(f"  Loaded ROI checkpoint: {args.roi_checkpoint}")
        else:
            print("  WARNING: ROI checkpoint not usable, skipping")
    else:
        print("\n[2/5] Skipping ROI baseline (no checkpoint)")

    # ========== Method 3: Gaussian-Set 12C teacher-only ==========
    result_12c = None
    if args.checkpoint_12c and os.path.exists(args.checkpoint_12c):
        print("\n[3/5] Evaluating Phase 12C (teacher-only)...")
        pf_12c = load_checkpoint_person_feature(args.checkpoint_12c, device)
        if pf_12c is not None:
            model._person_feature.data.copy_(pf_12c)
            result_12c = evaluate_method_12c(fixed_eval_samples, batch_builder, model, device, args)
            print(f"  Loaded 12C checkpoint: {args.checkpoint_12c}")
        else:
            print("  WARNING: 12C checkpoint not usable, skipping")
    else:
        print("\n[3/5] Skipping Phase 12C (no checkpoint)")

    # ========== Method 4: Gaussian-Set 12E MV InfoNCE ==========
    result_12e = None
    if args.checkpoint_12e and os.path.exists(args.checkpoint_12e):
        print("\n[4/5] Evaluating Phase 12E (MV InfoNCE)...")
        pf_12e = load_checkpoint_person_feature(args.checkpoint_12e, device)
        if pf_12e is not None:
            model._person_feature.data.copy_(pf_12e)
            result_12e = evaluate_method_12e(fixed_eval_samples, batch_builder, model, device, args)
            print(f"  Loaded 12E checkpoint: {args.checkpoint_12e}")
        else:
            print("  WARNING: 12E checkpoint not usable, skipping")
    else:
        print("\n[4/5] Skipping Phase 12E (no checkpoint)")

    # ========== Method 5: Gaussian-Set 12F EMA Proto ==========
    result_12f = None
    if args.checkpoint_12f and os.path.exists(args.checkpoint_12f):
        print("\n[5/5] Evaluating Phase 12F (EMA Proto + MV InfoNCE)...")
        pf_12f = load_checkpoint_person_feature(args.checkpoint_12f, device)
        if pf_12f is not None:
            model._person_feature.data.copy_(pf_12f)
            result_12f = evaluate_method_12f(fixed_eval_samples, batch_builder, model, device, args)
            print(f"  Loaded 12F checkpoint: {args.checkpoint_12f}")
        else:
            print("  WARNING: 12F checkpoint not usable, skipping")
    else:
        print("\n[5/5] Skipping Phase 12F (no checkpoint)")

    # ========== Compute metrics for all methods ==========
    print("\n" + "=" * 70)
    print("Computing ReID metrics...")
    print("=" * 70)

    all_results = {
        '2D_Teacher': teacher_result,
        'ROI_11B_v4': roi_result,
        '12C_Teacher_Only': result_12c,
        '12E_MV_InfoNCE': result_12e,
        '12F_EMA_Proto': result_12f,
    }

    eval_summaries = []
    retrieval_metrics = {}
    all_sim_matrices = {}
    per_camera_all = {}
    per_identity_all = {}

    # Teacher baseline
    teacher_sim = teacher_result['sim_matrix']
    all_sim_matrices['2D_Teacher'] = teacher_sim

    teacher_top1, teacher_top5, teacher_mAP = compute_retrieval_metrics(
        teacher_sim, teacher_result['person_ids'], teacher_result['cam_ids']
    )

    teacher_cos_metrics = compute_cosine_metrics(
        teacher_result['features'], teacher_result['person_ids'],
        teacher_result['cam_ids']
    )

    teacher_identity = compute_per_identity_metrics(
        torch.stack(teacher_result['features']), teacher_result['person_ids']
    )

    per_identity_all['2D_Teacher'] = teacher_identity
    per_camera_all['2D_Teacher'] = teacher_cos_metrics['per_camera']

    teacher_gap = teacher_cos_metrics['gap']

    retrieval_metrics['2D_Teacher'] = {
        'top1': teacher_top1,
        'top5': teacher_top5,
        'mAP': teacher_mAP,
        'same_cos': teacher_cos_metrics['same_cos'],
        'diff_cos': teacher_cos_metrics['diff_cos'],
        'gap': teacher_gap,
        'cos_to_teacher': 1.0,
    }

    eval_summaries.append({
        'method_name': '2D_Teacher',
        'num_samples': len(teacher_result['person_ids']),
        'num_ids': len(set(teacher_result['person_ids'])),
        'top1': teacher_top1,
        'top5': teacher_top5,
        'mAP': teacher_mAP,
        'same_cos': teacher_cos_metrics['same_cos'],
        'diff_cos': teacher_cos_metrics['diff_cos'],
        'gap': teacher_gap,
        'cos_to_teacher': 1.0,
        'teacher_gap': teacher_gap,
        'student_minus_teacher_gap': 0.0,
        'student_minus_teacher_mAP': 0.0,
        'verdict': 'baseline',
    })

    # Student methods
    for method_name, result in all_results.items():
        if method_name == '2D_Teacher' or result is None:
            continue

        features_tensor = torch.stack(result['features'])
        sim_matrix = features_tensor @ features_tensor.T
        all_sim_matrices[method_name] = sim_matrix.detach().cpu().numpy()

        top1, top5, mAP = compute_retrieval_metrics(
            sim_matrix.detach().cpu().numpy(), result['person_ids'], result['cam_ids']
        )

        cos_metrics = compute_cosine_metrics(
            result['features'], result['person_ids'],
            result['cam_ids'], teacher_features_tensor
        )

        identity_metrics = compute_per_identity_metrics(
            features_tensor, result['person_ids']
        )

        per_identity_all[method_name] = identity_metrics
        per_camera_all[method_name] = cos_metrics['per_camera']

        student_gap = cos_metrics['gap']
        student_minus_teacher_gap = student_gap - teacher_gap
        student_minus_teacher_mAP = mAP - teacher_mAP

        # Verdict
        if mAP > teacher_mAP and student_gap > teacher_gap:
            verdict = 'student_outperforms_teacher'
        elif abs(mAP - teacher_mAP) < 0.05 and student_gap > teacher_gap:
            verdict = 'student_improves_identity_separation'
        elif cos_metrics['cos_to_teacher'] is not None and cos_metrics['cos_to_teacher'] > 0.9 and \
             mAP < teacher_mAP * 0.8 and student_gap < teacher_gap:
            verdict = 'student_only_mimics_teacher'
        elif mAP < teacher_mAP * 0.7 or student_gap < teacher_gap * 0.5:
            verdict = 'student_below_teacher'
        else:
            verdict = 'student_comparable_to_teacher'

        retrieval_metrics[method_name] = {
            'top1': top1,
            'top5': top5,
            'mAP': mAP,
            'same_cos': cos_metrics['same_cos'],
            'diff_cos': cos_metrics['diff_cos'],
            'gap': student_gap,
            'cos_to_teacher': cos_metrics['cos_to_teacher'],
        }

        eval_summaries.append({
            'method_name': method_name,
            'num_samples': len(result['person_ids']),
            'num_ids': len(set(result['person_ids'])),
            'top1': top1,
            'top5': top5,
            'mAP': mAP,
            'same_cos': cos_metrics['same_cos'],
            'diff_cos': cos_metrics['diff_cos'],
            'gap': student_gap,
            'cos_to_teacher': cos_metrics['cos_to_teacher'],
            'teacher_gap': teacher_gap,
            'student_minus_teacher_gap': student_minus_teacher_gap,
            'student_minus_teacher_mAP': student_minus_teacher_mAP,
            'verdict': verdict,
        })

    # ========== Save outputs ==========
    print("\n" + "=" * 70)
    print("Saving evaluation results...")
    print("=" * 70)

    # 1. eval_summary.json
    with open(os.path.join(args.output_dir, 'eval_summary.json'), 'w') as f:
        json.dump(eval_summaries, f, indent=2, default=str)
    print(f"  Saved: {args.output_dir}/eval_summary.json")

    # 2. retrieval_metrics.json
    with open(os.path.join(args.output_dir, 'retrieval_metrics.json'), 'w') as f:
        json.dump(retrieval_metrics, f, indent=2, default=str)
    print(f"  Saved: {args.output_dir}/retrieval_metrics.json")

    # 3. similarity_matrix.npy
    np.save(os.path.join(args.output_dir, 'similarity_matrices.npy'), all_sim_matrices)
    print(f"  Saved: {args.output_dir}/similarity_matrices.npy")

    # 4. per_camera_metrics.json
    with open(os.path.join(args.output_dir, 'per_camera_metrics.json'), 'w') as f:
        json.dump(per_camera_all, f, indent=2, default=str)
    print(f"  Saved: {args.output_dir}/per_camera_metrics.json")

    # 5. per_identity_metrics.json
    with open(os.path.join(args.output_dir, 'per_identity_metrics.json'), 'w') as f:
        json.dump(per_identity_all, f, indent=2, default=str)
    print(f"  Saved: {args.output_dir}/per_identity_metrics.json")

    # 6. final_report.md
    generate_final_report(args.output_dir, eval_summaries, retrieval_metrics)
    print(f"  Saved: {args.output_dir}/final_report.md")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


def generate_final_report(output_dir, eval_summaries, retrieval_metrics):
    """Generate final_report.md answering all required questions."""

    # Find teacher and student metrics
    teacher = next((s for s in eval_summaries if s['method_name'] == '2D_Teacher'), None)
    roi = next((s for s in eval_summaries if s['method_name'] == 'ROI_11B_v4'), None)
    c12c = next((s for s in eval_summaries if s['method_name'] == '12C_Teacher_Only'), None)
    c12e = next((s for s in eval_summaries if s['method_name'] == '12E_MV_InfoNCE'), None)
    c12f = next((s for s in eval_summaries if s['method_name'] == '12F_EMA_Proto'), None)

    # Find best method for each metric
    methods_with_data = [m for m in [teacher, roi, c12c, c12e, c12f] if m is not None]

    best_rank1 = max(methods_with_data, key=lambda x: x['top1'])
    best_mAP = max(methods_with_data, key=lambda x: x['mAP'])
    best_gap = max(methods_with_data, key=lambda x: x['gap'])

    # Compare Gaussian-Set vs ROI
    gs_methods = [m for m in [c12c, c12e, c12f] if m is not None]
    gs_vs_roi = "无法比较（ROI 不可用）"
    if roi is not None and gs_methods:
        gs_best_mAP = max(gs_methods, key=lambda x: x['mAP'])
        gs_best_gap = max(gs_methods, key=lambda x: x['gap'])
        gs_vs_roi = (
            f"Gaussian-Set 最佳 mAP ({gs_best_mAP['method_name']}: {gs_best_mAP['mAP']:.4f}) "
            f"{'>' if gs_best_mAP['mAP'] > roi['mAP'] else '<'} ROI ({roi['mAP']:.4f})\n"
            f"Gaussian-Set 最佳 gap ({gs_best_gap['method_name']}: {gs_best_gap['gap']:.4f}) "
            f"{'>' if gs_best_gap['gap'] > roi['gap'] else '<'} ROI ({roi['gap']:.4f})"
        )

    # Compare 12E/12F vs 12C
    identity_improvement = "无法比较（数据不足）"
    if c12c is not None:
        improvements = []
        if c12e is not None:
            improvements.append(
                f"12E gap {c12e['gap']:.4f} {'>' if c12e['gap'] > c12c['gap'] else '<'} 12C gap {c12c['gap']:.4f}"
            )
        if c12f is not None:
            improvements.append(
                f"12F gap {c12f['gap']:.4f} {'>' if c12f['gap'] > c12c['gap'] else '<'} 12C gap {c12c['gap']:.4f}"
            )
        identity_improvement = "; ".join(improvements) if improvements else "无法比较"

    # Compare 12F vs Teacher
    c12f_vs_teacher = "无法比较（12F 不可用）"
    if c12f is not None and teacher is not None:
        mAP_diff = c12f['mAP'] - teacher['mAP']
        gap_diff = c12f['gap'] - teacher['gap']
        cos_to_t = c12f['cos_to_teacher']
        c12f_vs_teacher = (
            f"12F vs 2D Teacher:\n"
            f"- mAP 差异: {mAP_diff:+.4f} ({c12f['mAP']:.4f} vs {teacher['mAP']:.4f})\n"
            f"- gap 差异: {gap_diff:+.4f} ({c12f['gap']:.4f} vs {teacher['gap']:.4f})\n"
            f"- cos_to_teacher: {cos_to_t:.4f}\n"
            f"- 结论: {'接近或超过' if mAP_diff > -0.05 and gap_diff > -0.02 else '仍有差距'}"
        )

    # Recommend 12G?
    need_12g = "无法判断（数据不足）"
    if c12f is not None and teacher is not None:
        mAP_ratio = c12f['mAP'] / teacher['mAP'] if teacher['mAP'] > 0 else 0
        if mAP_ratio < 0.8:
            need_12g = (
                f"12F mAP 仅为 Teacher 的 {mAP_ratio:.1%}，身份区分能力仍有显著提升空间。\n"
                f"建议继续训练 12G Teacher-Regularized SupCon，以进一步提升 mAP 和 gap。"
            )
        elif mAP_ratio < 0.95:
            need_12g = (
                f"12F mAP 达到 Teacher 的 {mAP_ratio:.1%}，差距较小但仍可优化。\n"
                f"可以尝试 12G Teacher-Regularized SupCon 作为可选提升方向。"
            )
        else:
            need_12g = (
                f"12F mAP 已达到 Teacher 的 {mAP_ratio:.1%}，接近 2D Teacher 水平。\n"
                f"12G 可能带来的提升有限，可视需求决定是否进行。"
            )

    report = f"""# Phase 12 Gaussian-Set Final ReID Evaluation Report

## 1. 最佳 Rank-1 / mAP / gap 方法

| 指标 | 最佳方法 | 值 |
|------|----------|-----|
| Rank-1 | {best_rank1['method_name']} | {best_rank1['top1']:.4f} |
| mAP | {best_mAP['method_name']} | {best_mAP['mAP']:.4f} |
| gap | {best_gap['method_name']} | {best_gap['gap']:.4f} |

## 2. 完整对比表

| 方法 | Rank-1 | Top-5 | mAP | same_cos | diff_cos | gap | cos_to_teacher | verdict |
|------|--------|-------|-----|----------|----------|-----|----------------|---------|
"""

    for s in eval_summaries:
        cos_t_val = f"{s['cos_to_teacher']:.4f}" if isinstance(s['cos_to_teacher'], (int, float)) else 'N/A'
        report += (
            f"| {s['method_name']} | {s['top1']:.4f} | {s['top5']:.4f} | {s['mAP']:.4f} | "
            f"{s['same_cos']:.4f} | {s['diff_cos']:.4f} | {s['gap']:.4f} | "
            f"{cos_t_val} | "
            f"{s['verdict']} |\n"
        )

    report += f"""
## 3. Gaussian-Set vs ROI Baseline

{gs_vs_roi}

## 4. 12E/12F vs 12C 身份区分能力

{identity_improvement}

## 5. 12F vs 2D Teacher

{c12f_vs_teacher}

## 6. 是否有必要训练 12G Teacher-Regularized SupCon

{need_12g}

## 7. 总结

基于本次评估结果：

"""

    # Add summary conclusions
    if c12f is not None:
        if c12f['mAP'] > teacher['mAP'] * 0.9:
            report += "- **Gaussian-Set 已达到 2D Teacher 90% 以上的 mAP 水平**，证明 3DGS 方法具有可行性。\n"
        else:
            report += "- **Gaussian-Set 与 2D Teacher 仍有差距**，但已展现出身份区分能力。\n"

    if c12f is not None and c12e is not None and c12c is not None:
        if c12f['gap'] > c12e['gap'] > c12c['gap']:
            report += "- **12F > 12E > 12C 的 gap 趋势**表明：MV InfoNCE 和 EMA Proto 有效提升了跨视角身份一致性。\n"
        else:
            report += "- **gap 提升趋势不完全符合预期**，需要进一步分析原因。\n"

    report += f"""
- **最佳方法：{best_mAP['method_name']}**（mAP={best_mAP['mAP']:.4f}）
- **最大 gap：{best_gap['method_name']}**（gap={best_gap['gap']:.4f}）
- **推荐方向：继续优化 Gaussian-Set pooling 或引入更强身份约束**

---

*报告生成时间：2026-05-13*
*所有结论基于 eval_summary.json 和 retrieval_metrics.json*
"""

    with open(os.path.join(output_dir, 'final_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Phase 12 Gaussian-Set Final ReID Evaluation')

    parser.add_argument('--fixed_eval_samples', type=str, required=True,
                        help='Path to fixed_eval_samples.json')
    parser.add_argument('--teacher_features', type=str, default=None,
                        help='Path to teacher features (optional, features in fixed_eval_samples)')
    parser.add_argument('--checkpoint_12c', type=str, default=None,
                        help='Path to Phase 12C checkpoint')
    parser.add_argument('--checkpoint_12e', type=str, default=None,
                        help='Path to Phase 12E best checkpoint')
    parser.add_argument('--checkpoint_12f', type=str, default=None,
                        help='Path to Phase 12F best checkpoint')
    parser.add_argument('--roi_checkpoint', type=str, default=None,
                        help='Path to ROI 11B-v4 checkpoint (optional)')
    parser.add_argument('--allowed_cameras', type=str, default='C1,C4,C6,C7',
                        help='Comma-separated list of cameras to evaluate')
    parser.add_argument('--output_dir', type=str, default='outputs/phase12_final_reid_eval',
                        help='Output directory for evaluation results')
    parser.add_argument('--denom_eps', type=float, default=1e-8,
                        help='Epsilon for Gaussian pooling denominator')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_eval(args)


if __name__ == '__main__':
    main()
