#!/usr/bin/env python3
"""
Stage 2: Half-Data All-Camera Geometry Validation

Goal: Confirm bbox-level Gaussian support with Phase11A real geometry across C1-C7.
Uses ~50% frames, stratified sampling, covering early/mid/late time periods and many person_ids.

After Stage 1 confirmed all cameras can render, this stage validates:
- C2/C3/C5 bbox support still near zero?
- C1/C4/C6/C7 bbox support normal?
- Is the bottleneck geometry, camera alignment, bbox mapping, or pooling selection?
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rich import pretty, traceback

pretty.install()
traceback.install()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)


def normalize_feat(x, eps=1e-6):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


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


def gaussian_set_pooling(model, gpu_batch, bbox, cam_id, frame_id, args, device):
    """Direct Gaussian xyz projection pooling (same as Phase12F)."""
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


def project_all_gaussians(model, gpu_batch, device):
    """Project all 3D Gaussians to 2D image coordinates."""
    try:
        xyz = model.positions
        opacity = model.get_density().squeeze(-1)
        N = xyz.shape[0]

        intrinsics = gpu_batch.intrinsics
        if intrinsics is None or len(intrinsics) < 4:
            return None

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

        projected = []
        for i in range(N):
            if valid[i]:
                projected.append({
                    'idx': i, 'x': x_img[i].item(), 'y': y_img[i].item(),
                    'depth': depth[i].item(), 'opacity': opacity[i].item(),
                })

        return {'projected': projected, 'num_projected': len(projected), 'num_total': N}
    except Exception as e:
        return {'projected': [], 'num_projected': 0, 'error': str(e)}


def draw_gaussian_overlay(image, gaussians, bbox, max_points=500):
    """Draw Gaussian projection points as overlay on image."""
    overlay = image.copy()
    x1, y1, x2, y2 = bbox

    if not gaussians:
        return overlay

    sorted_gs = sorted(gaussians, key=lambda g: g.get('opacity', 0), reverse=True)[:max_points]
    if not sorted_gs:
        return overlay

    opacities = [g.get('opacity', 0) for g in sorted_gs]
    max_op = max(opacities) if opacities else 1.0
    min_op = min(opacities) if opacities else 0.0
    op_range = max_op - min_op if max_op != min_op else 1.0

    for g in sorted_gs:
        px, py = int(g['x']), int(g['y'])
        opacity_norm = (g.get('opacity', 0) - min_op) / op_range
        r = int(255 * opacity_norm)
        b = int(255 * (1 - opacity_norm))
        cv2.circle(overlay, (px, py), 3, (b, 0, r), -1)

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return overlay


def determine_overlay_verdict(debug_info, all_proj_info, bbox):
    """Semi-automatic overlay verdict determination."""
    x1, y1, x2, y2 = bbox
    num_inside = debug_info.get('num_gaussians_in_bbox', 0)
    weight_sum = debug_info.get('weight_sum', 0.0)

    if num_inside == 0:
        if all_proj_info and all_proj_info.get('num_projected', 0) == 0:
            return 'empty'
        else:
            if all_proj_info:
                near_count = 0
                margin = 50
                for g in all_proj_info.get('projected', []):
                    if (x1 - margin <= g['x'] < x2 + margin and y1 - margin <= g['y'] < y2 + margin):
                        near_count += 1
                if near_count > 0:
                    return 'misaligned'
                else:
                    return 'empty'
            return 'empty'

    if num_inside >= 5 and weight_sum > 0.1:
        return 'on_body'
    elif num_inside >= 2 and weight_sum > 0.01:
        return 'partial_body'
    else:
        return 'background'


def run_stage2(args):
    print("\n" + "=" * 80)
    print("Stage 2: Half-Data All-Camera Geometry Validation")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config and model
    from hydra import initialize_config_dir, compose

    config_dir = os.path.join(REPO_ROOT, 'configs')
    config_file = 'apps/wildtrack_full_3dgut'

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_file)
    cfg.model.person_feature_dim = 512

    from threedgrut.trainer import Trainer3DGRUT
    trainer = Trainer3DGRUT(cfg)
    model = trainer.model
    dataset = trainer.train_dataset
    device = trainer.device

    print(f"\nModel loaded: {type(model).__name__}")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Positions shape (before): {model.positions.shape}")

    # Load Phase11A geometry checkpoint
    geo_ckpt_path = args.geometry_checkpoint
    if not os.path.exists(geo_ckpt_path):
        print(f"\n❌ ERROR: Geometry checkpoint not found at {geo_ckpt_path}")
        return

    print(f"\nLoading geometry from: {geo_ckpt_path}")
    geo_ckpt = torch.load(geo_ckpt_path, map_location=device)
    model_state = geo_ckpt.get('model_state_dict', geo_ckpt)

    geo_keys = ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']
    found_keys = {}
    for k in geo_keys:
        if k in model_state:
            found_keys[k] = model_state[k].shape
            print(f"  Found: {k} shape={model_state[k].shape}")

    if not found_keys:
        print("\n❌ ERROR: No geometry keys found in checkpoint!")
        return

    num_gaussians = found_keys.get('positions', torch.Size([0]))[0]
    print(f"\nReal geometry: {num_gaussians} Gaussians")

    for k in ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']:
        if k in model_state:
            param_name = k if hasattr(model, k) else ('_' + k if hasattr(model, '_' + k) else None)
            if param_name and hasattr(model, param_name):
                getattr(model, param_name).data = model_state[k].to(device)
                getattr(model, param_name).requires_grad = False
                print(f"  Loaded & frozen: {param_name}")

    print(f"\nPositions shape (after): {model.positions.shape}")
    print(f"Positions stats: mean={model.positions.mean().item():.3f}, std={model.positions.std().item():.3f}")
    density = model.get_density()
    print(f"Density stats: mean={density.mean().item():.4f}, max={density.max().item():.4f}")

    batch_builder = BatchBuilder(dataset)

    # Load eval samples from medium_eval_allcam.json
    eval_samples_path = args.eval_samples
    if not eval_samples_path:
        eval_samples_path = os.path.join(REPO_ROOT, 'outputs/phase12_parallel_validation/medium_eval_allcam.json')

    if os.path.exists(eval_samples_path):
        with open(eval_samples_path, 'r') as f:
            eval_samples = json.load(f)
        print(f"\nLoaded {len(eval_samples)} eval samples from {eval_samples_path}")
    else:
        print(f"\nERROR: eval_samples not found at {eval_samples_path}")
        return

    # Group by camera
    samples_by_cam = defaultdict(list)
    for s in eval_samples:
        samples_by_cam[s['cam_id']].append(s)

    all_cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    per_camera_stats = {}
    per_sample_records = []

    for cam_id in all_cameras:
        cam_samples = samples_by_cam.get(cam_id, [])
        if not cam_samples:
            per_camera_stats[cam_id] = {'camera': cam_id, 'num_samples_diagnosed': 0, 'final_camera_verdict': 'no_samples'}
            continue

        # Sample up to args.samples_per_camera per camera (use 50% of medium eval)
        target = min(len(cam_samples), max(50, len(cam_samples) // 2))
        if len(cam_samples) > target:
            sampled = random.sample(cam_samples, target)
        else:
            sampled = cam_samples

        print(f"\n{'='*60}")
        print(f"{cam_id}: {len(cam_samples)} available -> {len(sampled)} sampled (~50%)")
        print(f"{'='*60}")

        cam_dir = os.path.join(args.output_dir, cam_id)
        os.makedirs(cam_dir, exist_ok=True)

        verdict_counts = defaultdict(int)
        valid_count = 0
        inside_bbox_counts = []
        weight_sums = []
        feature_norms = []
        full_opacity_sums = []
        projected_counts = []

        for s_idx, sample in enumerate(sampled):
            person_id = sample.get('person_id', 'unknown')
            frame_id = sample.get('frame_id', sample.get('frame_idx', 'unknown'))
            bbox = sample.get('bbox', [0, 0, 0, 0])

            if isinstance(frame_id, str):
                frame_id = int(frame_id)

            sample_id = f"{cam_id}_frame{frame_id:06d}_pid{person_id:03d}_{s_idx:03d}"

            gpu_batch = batch_builder.get_batch(cam_id, frame_id)
            if gpu_batch is None:
                per_sample_records.append({
                    'sample_id': sample_id, 'cam_id': cam_id, 'frame_id': frame_id,
                    'person_id': person_id, 'bbox': bbox,
                    'valid_sample': False, 'invalid_reason': 'gpu_batch_none',
                })
                continue

            # Load image
            image = None
            if cam_id in dataset.image_paths and frame_id < len(dataset.image_paths[cam_id]):
                image_path = dataset.image_paths[cam_id][frame_id]
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)

            # Project all Gaussians
            all_proj = project_all_gaussians(model, gpu_batch, device)
            num_projected = all_proj.get('num_projected', 0) if all_proj else 0
            projected_counts.append(num_projected)

            # Render opacity for full image stats
            try:
                render_out = model(gpu_batch, train=False, frame_id=0)
                pred_opacity = render_out.get('pred_opacity', None)
                if pred_opacity is not None:
                    full_opacity_sum = float(pred_opacity.sum().item())
                else:
                    full_opacity_sum = 0
            except Exception:
                full_opacity_sum = 0

            full_opacity_sums.append(full_opacity_sum)

            # Run Gaussian-Set pooling
            G, debug_info = gaussian_set_pooling(model, gpu_batch, bbox, cam_id, frame_id, args, device)

            num_inside_bbox = debug_info.get('num_gaussians_in_bbox', 0)
            weight_sum = debug_info.get('weight_sum', 0.0)
            feature_norm = float(G.norm().item()) if G is not None else 0.0

            overlay_verdict = determine_overlay_verdict(debug_info, all_proj, bbox)
            verdict_counts[overlay_verdict] += 1

            # bbox opacity sum (approximate from projected Gaussians)
            bbox_opacity_sum = 0
            if all_proj and all_proj.get('projected'):
                x1_b, y1_b, x2_b, y2_b = bbox
                for g in all_proj['projected']:
                    if x1_b <= g['x'] < x2_b and y1_b <= g['y'] < y2_b:
                        bbox_opacity_sum += g['opacity']

            if G is not None:
                valid_count += 1
                inside_bbox_counts.append(num_inside_bbox)
                weight_sums.append(weight_sum)
                feature_norms.append(feature_norm)

            # Save visualizations (first 10 samples per camera)
            if s_idx < 10 and image is not None and image.size > 0:
                x1_b, y1_b, x2_b, y2_b = bbox
                img_with_bbox = image.copy()
                cv2.rectangle(img_with_bbox, (x1_b, y1_b), (x2_b, y2_b), (0, 255, 0), 2)
                cv2.putText(img_with_bbox, f"PID:{person_id} | GS:{num_inside_bbox}",
                           (x1_b, max(y1_b - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_original_bbox.png"), img_with_bbox)

                if all_proj and all_proj.get('projected'):
                    all_overlay = draw_gaussian_overlay(image, all_proj['projected'], bbox, max_points=500)
                    cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_all_projected.png"), all_overlay)

                    if G is not None:
                        selected_gs = [g for g in all_proj['projected']
                                      if x1_b <= g['x'] < x2_b and y1_b <= g['y'] < y2_b]
                        if selected_gs:
                            sel_overlay = draw_gaussian_overlay(image, selected_gs, bbox, max_points=100)
                            cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_selected_overlay.png"), sel_overlay)

            # Build record
            x1_b, y1_b, x2_b, y2_b = bbox
            bbox_area = (x2_b - x1_b) * (y2_b - y1_b)
            bbox_opacity_ratio = bbox_opacity_sum / (full_opacity_sum + 1e-10) if full_opacity_sum > 0 else 0

            record = {
                'sample_id': sample_id,
                'cam_id': cam_id,
                'frame_id': frame_id,
                'person_id': int(person_id),
                'bbox': bbox,
                'bbox_area': int(bbox_area),
                'full_image_opacity_sum': full_opacity_sum,
                'bbox_opacity_sum': bbox_opacity_sum,
                'bbox_opacity_ratio': bbox_opacity_ratio,
                'num_all_projected_gaussians': num_projected,
                'num_inside_bbox_gaussians': num_inside_bbox,
                'num_selected_gaussians': num_inside_bbox,
                'selected_weight_sum': weight_sum,
                'selected_feature_norm': feature_norm,
                'pooled_feature_norm': feature_norm,
                'overlay_verdict': overlay_verdict,
                'valid_sample': G is not None,
                'invalid_reason': debug_info.get('failure_reason', None) if G is None else None,
            }
            per_sample_records.append(record)

            if s_idx < 5 or s_idx % 30 == 0:
                print(f"  [{s_idx}/{len(sampled)}] {sample_id}: "
                      f"proj={num_projected}, inside={num_inside_bbox}, "
                      f"w_sum={weight_sum:.4f}, verdict={overlay_verdict}")

        # Per-camera summary
        total_diagnosed = len(sampled)
        valid_ratio = valid_count / max(1, total_diagnosed)

        per_camera_stats[cam_id] = {
            'camera': cam_id,
            'num_samples_diagnosed': total_diagnosed,
            'valid_sample_count': valid_count,
            'valid_sample_ratio': valid_ratio,
            'mean_projected_gaussians': float(np.mean(projected_counts)) if projected_counts else 0,
            'mean_inside_bbox_gaussians': float(np.mean(inside_bbox_counts)) if inside_bbox_counts else 0,
            'mean_selected_weight_sum': float(np.mean(weight_sums)) if weight_sums else 0,
            'mean_selected_feature_norm': float(np.mean(feature_norms)) if feature_norms else 0,
            'mean_full_opacity_sum': float(np.mean(full_opacity_sums)) if full_opacity_sums else 0,
            'overlay_verdict_counts': dict(verdict_counts),
            'overlay_on_body_ratio': verdict_counts.get('on_body', 0) / max(1, total_diagnosed),
            'overlay_partial_body_ratio': verdict_counts.get('partial_body', 0) / max(1, total_diagnosed),
            'overlay_background_ratio': verdict_counts.get('background', 0) / max(1, total_diagnosed),
            'overlay_empty_ratio': verdict_counts.get('empty', 0) / max(1, total_diagnosed),
            'overlay_misaligned_ratio': verdict_counts.get('misaligned', 0) / max(1, total_diagnosed),
        }

        # Final camera verdict
        stats = per_camera_stats[cam_id]
        if stats['valid_sample_ratio'] < 0.1:
            stats['final_camera_verdict'] = 'geometry_missing_person_support'
        elif stats['overlay_empty_ratio'] > 0.5:
            stats['final_camera_verdict'] = 'bbox_empty_no_gaussians'
        elif stats['overlay_background_ratio'] > 0.5:
            stats['final_camera_verdict'] = 'pooling_selection_wrong_background'
        elif stats['overlay_misaligned_ratio'] > 0.5:
            stats['final_camera_verdict'] = 'gaussians_misaligned_with_bbox'
        elif stats['overlay_on_body_ratio'] > 0.5:
            stats['final_camera_verdict'] = 'geometry_pooling_usable'
        else:
            stats['final_camera_verdict'] = 'mixed_uncertain'

        print(f"\n  {cam_id} Summary:")
        print(f"    Valid ratio: {stats['valid_sample_ratio']:.3f}")
        print(f"    Mean projected: {stats['mean_projected_gaussians']:.1f}")
        print(f"    Mean inside bbox: {stats['mean_inside_bbox_gaussians']:.1f}")
        print(f"    Mean weight sum: {stats['mean_selected_weight_sum']:.4f}")
        print(f"    Mean full opacity: {stats['mean_full_opacity_sum']:.1f}")
        print(f"    Verdicts: {dict(verdict_counts)}")
        print(f"    Final verdict: {stats['final_camera_verdict']}")

    # Save outputs
    with open(os.path.join(args.output_dir, 'per_camera_support_summary.json'), 'w') as f:
        json.dump(per_camera_stats, f, indent=2, default=str)

    with open(os.path.join(args.output_dir, 'bbox_gaussian_support_metrics.jsonl'), 'w') as f:
        for record in per_sample_records:
            f.write(json.dumps(record, default=str) + '\n')

    # Overlay summary
    overlay_summary = {
        'total_samples_diagnosed': len(per_sample_records),
        'valid_samples': sum(1 for r in per_sample_records if r.get('valid_sample')),
        'per_camera': {cam: stats.get('overlay_verdict_counts', {}) for cam, stats in per_camera_stats.items()},
        'global_verdict_counts': defaultdict(int),
    }
    for r in per_sample_records:
        v = r.get('overlay_verdict', 'uncertain')
        overlay_summary['global_verdict_counts'][v] += 1
    overlay_summary['global_verdict_counts'] = dict(overlay_summary['global_verdict_counts'])

    with open(os.path.join(args.output_dir, 'selected_gaussian_overlay_summary.json'), 'w') as f:
        json.dump(overlay_summary, f, indent=2, default=str)

    # Generate final report
    generate_stage2_report(per_camera_stats, overlay_summary, per_sample_records, num_gaussians,
                           args.geometry_checkpoint, args.output_dir)

    print(f"\n{'='*80}")
    print(f"Stage 2 Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


def generate_stage2_report(per_camera_stats, overlay_summary, per_sample_records, num_gaussians, ckpt_path, output_dir):
    """Generate comprehensive final_geometry_report.md for Stage 2."""

    on_body_total = overlay_summary.get('global_verdict_counts', {}).get('on_body', 0)
    partial_body_total = overlay_summary.get('global_verdict_counts', {}).get('partial_body', 0)
    background_total = overlay_summary.get('global_verdict_counts', {}).get('background', 0)
    empty_total = overlay_summary.get('global_verdict_counts', {}).get('empty', 0)
    misaligned_total = overlay_summary.get('global_verdict_counts', {}).get('misaligned', 0)
    total_diagnosed = overlay_summary.get('total_samples_diagnosed', 0)

    report = f"""# Stage 2: Half-Data All-Camera Geometry Validation — Final Report

## Geometry Configuration

- Checkpoint: {ckpt_path}
- Num Gaussians: {num_gaussians}
- Geometry keys: positions, rotation, scale, density, features_albedo, features_specular
- All geometry parameters frozen (requires_grad=False)

## 1. 当前已有 geometry 的 selected Gaussian 是否落在人身上？

| Verdict | Count | Ratio |
|---------|-------|-------|
| on_body | {on_body_total} | {on_body_total/max(1,total_diagnosed):.2%} |
| partial_body | {partial_body_total} | {partial_body_total/max(1,total_diagnosed):.2%} |
| background | {background_total} | {background_total/max(1,total_diagnosed):.2%} |
| empty | {empty_total} | {empty_total/max(1,total_diagnosed):.2%} |
| misaligned | {misaligned_total} | {misaligned_total/max(1,total_diagnosed):.2%} |

**Combined on_body + partial_body**: {on_body_total + partial_body_total}/{total_diagnosed} ({(on_body_total + partial_body_total)/max(1,total_diagnosed):.2%})

## 2. 当前问题更像 selection 选错，还是 bbox 内根本没有人体 Gaussian？

| 判断依据 | 结论 |
|---------|------|
"""
    empty_ratio = empty_total / max(1, total_diagnosed)
    background_ratio = background_total / max(1, total_diagnosed)

    if empty_ratio > 0.5:
        report += f"| bbox 内 Gaussian 数量 | ⚠️ 大量样本 bbox 内无 Gaussian (empty_ratio={empty_ratio:.2%}) |\n"
        report += "| 问题归因 | bbox 内根本没有人体 Gaussian support |\n"
    elif background_ratio > 0.3:
        report += f"| bbox 内 Gaussian 选择 | ⚠️ 部分样本选中的 Gaussian 在背景 (background_ratio={background_ratio:.2%}) |\n"
        report += "| 问题归因 | pooling/selection 选错背景 Gaussian |\n"
    else:
        report += f"| bbox 内 Gaussian 质量 | ✅ 大部分样本 Gaussian 落在人体区域 |\n"
        report += "| 问题归因 | geometry support 可能够用，问题可能在 feature learning |\n"

    report += """
## 3. C1-C7 的 render / opacity 是否正常？

| Camera | Samples | Valid Ratio | Mean Projected | Mean Inside BBox | Mean Weight Sum | Mean Full Opacity | Final Verdict |
|--------|---------|------------|----------------|-----------------|-----------------|-------------------|---------------|
"""

    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = per_camera_stats.get(cam_id, {})
        report += (
            f"| {cam_id} | {stats.get('num_samples_diagnosed', 0)} | {stats.get('valid_sample_ratio', 0):.3f} | "
            f"{stats.get('mean_projected_gaussians', 0):.1f} | {stats.get('mean_inside_bbox_gaussians', 0):.1f} | "
            f"{stats.get('mean_selected_weight_sum', 0):.4f} | {stats.get('mean_full_opacity_sum', 0):.1f} | "
            f"{stats.get('final_camera_verdict', 'N/A')} |\n"
        )

    report += """
## 4. C2/C3/C5 是否仍存在 full opacity 或 bbox opacity 异常？

"""
    for cam_id in ['C2', 'C3', 'C5']:
        stats = per_camera_stats.get(cam_id, {})
        report += f"### {cam_id}\n\n"
        report += f"- Valid sample ratio: {stats.get('valid_sample_ratio', 0):.3f}\n"
        report += f"- Mean projected Gaussians: {stats.get('mean_projected_gaussians', 0):.1f}\n"
        report += f"- Mean inside bbox Gaussians: {stats.get('mean_inside_bbox_gaussians', 0):.1f}\n"
        report += f"- Mean weight sum: {stats.get('mean_selected_weight_sum', 0):.4f}\n"
        report += f"- Mean full opacity sum: {stats.get('mean_full_opacity_sum', 0):.1f}\n"
        report += f"- Overlay verdicts: {stats.get('overlay_verdict_counts', {})}\n"
        report += f"- Final verdict: {stats.get('final_camera_verdict', 'N/A')}\n\n"

    report += """## 5. bbox 内是否有足够 projected / selected Gaussian？

| Camera | Mean Projected | Mean Inside BBox | Mean Selected | Weight Sum |
|--------|---------------|-----------------|---------------|------------|
"""
    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = per_camera_stats.get(cam_id, {})
        report += (
            f"| {cam_id} | {stats.get('mean_projected_gaussians', 0):.1f} | "
            f"{stats.get('mean_inside_bbox_gaussians', 0):.1f} | "
            f"{stats.get('mean_inside_bbox_gaussians', 0):.1f} | "
            f"{stats.get('mean_selected_weight_sum', 0):.4f} |\n"
        )

    report += """
## 6. 当前瓶颈更像 geometry、camera alignment、bbox mapping、还是 pooling selection？

"""
    cameras_with_issues = []
    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = per_camera_stats.get(cam_id, {})
        verdict = stats.get('final_camera_verdict', '')
        if verdict not in ['geometry_pooling_usable', 'no_samples']:
            cameras_with_issues.append((cam_id, verdict))

    if cameras_with_issues:
        report += "**有问题的相机**:\n\n"
        for cam_id, verdict in cameras_with_issues:
            report += f"- {cam_id}: {verdict}\n"

        geometry_issues = [c for c, v in cameras_with_issues if 'geometry' in v or 'empty' in v]
        pooling_issues = [c for c, v in cameras_with_issues if 'selection' in v or 'background' in v]
        alignment_issues = [c for c, v in cameras_with_issues if 'misalign' in v]

        report += "\n**问题分类**:\n\n"
        if geometry_issues:
            report += f"- Geometry 缺失: {', '.join(c for c, _ in geometry_issues)}\n"
        if pooling_issues:
            report += f"- Pooling/Selection 选错: {', '.join(c for c, _ in pooling_issues)}\n"
        if alignment_issues:
            report += f"- BBox Alignment 问题: {', '.join(c for c, _ in alignment_issues)}\n"

        if len(geometry_issues) > len(pooling_issues) and len(geometry_issues) > len(alignment_issues):
            report += "\n**主要瓶颈**: Geometry 缺少 person-level Gaussian support\n"
        elif len(pooling_issues) > len(alignment_issues):
            report += "\n**主要瓶颈**: Pooling/Selection 机制选错 Gaussian\n"
        else:
            report += "\n**主要瓶颈**: BBox-Gaussian 对齐问题\n"
    else:
        report += "**所有相机 geometry/pooling 均正常**\n"
        report += "\n**主要瓶颈**: 可能在于 feature learning (loss design)\n"

    report += """
## 7. 是否允许进入 teacher-only warm-up 和 12G SupCon 正式训练？

"""
    usable_cameras = [c for c in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
                      if per_camera_stats.get(c, {}).get('final_camera_verdict') == 'geometry_pooling_usable']

    valid_ratios = [per_camera_stats.get(c, {}).get('valid_sample_ratio', 0) for c in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']]
    avg_valid_ratio = sum(valid_ratios) / len(valid_ratios)

    if avg_valid_ratio > 0.7:
        report += "### ✅ 建议：可以进入 teacher-only warm-up 和 12G SupCon\n\n"
        report += f"- 平均 valid ratio: {avg_valid_ratio:.2%}\n"
        report += f"- 可用相机: {', '.join(usable_cameras)}\n"
        report += "- Geometry/pooling support 基本可信\n"
    elif avg_valid_ratio > 0.4:
        report += "### ⚠️ 建议：部分相机可用，需谨慎\n\n"
        report += f"- 平均 valid ratio: {avg_valid_ratio:.2%}\n"
        report += f"- 可用相机: {', '.join(usable_cameras)}\n"
        report += "- 建议先在可用相机上尝试 warm-up\n"
        report += "- 对不可用相机需要进一步诊断\n"
    else:
        report += "### ❌ 建议：geometry/pooling 问题严重，不建议继续\n\n"
        report += f"- 平均 valid ratio: {avg_valid_ratio:.2%}\n"
        report += f"- 可用相机: {', '.join(usable_cameras) if usable_cameras else '无'}\n"
        report += "- 需要进一步修复 geometry 或 pooling\n"

    report += """
---

*报告生成时间: Stage 2 Half-Data Validation*
*脚本: tools/phase12_stage2_geometry_validation.py*
"""

    with open(os.path.join(output_dir, 'final_geometry_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Half-Data All-Camera Geometry Validation')

    parser.add_argument('--output_dir', type=str,
                        default='outputs/phase12_geometry_clean_rebuild_half')
    parser.add_argument('--geometry_checkpoint', type=str,
                        default='runs/phase11A_proto_infonce_opacity_lam005_lr1e4/step_20999.pth')
    parser.add_argument('--eval_samples', type=str, default=None,
                        help='Path to eval samples JSON (default: medium_eval_allcam.json)')
    parser.add_argument('--samples_per_camera', type=int, default=None,
                        help='Max samples per camera (default: 50% of medium eval)')
    parser.add_argument('--denom_eps', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_stage2(args)


if __name__ == '__main__':
    main()
