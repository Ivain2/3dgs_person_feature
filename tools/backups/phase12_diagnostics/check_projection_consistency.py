#!/usr/bin/env python3
"""
Phase12 Projection Consistency Check — Stage A

Goal: Verify whether Gaussian-Set pooling's manual projected Gaussians align with
renderer's pred_opacity / pred_rgb in the same coordinate system.

Key questions:
1. Are projected points falling on high-opacity rendered regions?
2. Is there overall offset, scaling error, or flip?
3. Are projected coordinates in render space (e.g., 480x272)?
4. Is bbox correctly scaled to render resolution?
5. Are there camera-specific projection failures?
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


def manual_projection(model, gpu_batch, device):
    """Manual Gaussian projection to 2D (same as current pooling code)."""
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
                    'idx': i,
                    'x': float(x_img[i].item()),
                    'y': float(y_img[i].item()),
                    'depth': float(depth[i].item()),
                    'opacity': float(opacity[i].item()),
                    'x_cam': float(xyz_cam[i, 0].item()),
                    'y_cam': float(xyz_cam[i, 1].item()),
                    'z_cam': float(xyz_cam[i, 2].item()),
                })

        valid_count = int(valid.sum().item())
        return {
            'projected': projected,
            'num_valid': valid_count,
            'num_total': N,
            'render_h': h_img,
            'render_w': w_img,
            'x_min': float(x_img[valid].min().item()) if valid_count > 0 else 0,
            'x_max': float(x_img[valid].max().item()) if valid_count > 0 else 0,
            'y_min': float(y_img[valid].min().item()) if valid_count > 0 else 0,
            'y_max': float(y_img[valid].max().item()) if valid_count > 0 else 0,
            'x_mean': float(x_img[valid].mean().item()) if valid_count > 0 else 0,
            'y_mean': float(y_img[valid].mean().item()) if valid_count > 0 else 0,
        }
    except Exception as e:
        return {'projected': [], 'num_valid': 0, 'error': str(e)}


def check_bbox_projection_alignment(manual_proj, render_out, bbox_original, image_original, gpu_batch):
    """Check if bbox coordinates align between original image and render space."""
    # Get render dimensions
    if 'pred_opacity' in render_out:
        render_h, render_w = render_out['pred_opacity'].shape[1], render_out['pred_opacity'].shape[2]
    else:
        render_h, render_w = gpu_batch.rays_dir.shape[1], gpu_batch.rays_dir.shape[2]

    # Get original image dimensions
    orig_h, orig_w = image_original.shape[:2]

    # Compute scale factors
    scale_x = render_w / orig_w
    scale_y = render_h / orig_h

    # Scale bbox to render space
    x1_orig, y1_orig, x2_orig, y2_orig = bbox_original
    x1_render = x1_orig * scale_x
    y1_render = y1_orig * scale_y
    x2_render = x2_orig * scale_x
    y2_render = y2_orig * scale_y

    return {
        'original_image_size': (orig_w, orig_h),
        'render_size': (render_w, render_h),
        'bbox_original': list(bbox_original),
        'bbox_scaled': [x1_render, y1_render, x2_render, y2_render],
        'scale_x': scale_x,
        'scale_y': scale_y,
    }


def compute_opacity_bbox_stats(render_out, bbox_scaled):
    """Compute opacity statistics within the scaled bbox."""
    if 'pred_opacity' not in render_out:
        return {}

    opacity = render_out['pred_opacity']
    if opacity.ndim == 4 and opacity.shape[0] == 1:
        opacity = opacity[0]  # [H, W, 1]
    if opacity.ndim == 3 and opacity.shape[-1] == 1:
        opacity = opacity[:, :, 0]  # [H, W]

    x1, y1, x2, y2 = bbox_scaled
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(opacity.shape[1], int(x2))
    y2 = min(opacity.shape[0], int(y2))

    if x2 <= x1 or y2 <= y1:
        return {'bbox_clipped': True, 'opacity_bbox_sum': 0}

    opacity_np = opacity.cpu().numpy()
    bbox_opacity = opacity_np[y1:y2, x1:x2]

    return {
        'bbox_clipped': False,
        'opacity_bbox_sum': float(np.sum(bbox_opacity)),
        'opacity_bbox_mean': float(np.mean(bbox_opacity)),
        'opacity_bbox_max': float(np.max(bbox_opacity)),
        'opacity_nonzero_ratio': float(np.mean(bbox_opacity > 0.01)),
        'opacity_full_sum': float(np.sum(opacity_np)),
        'opacity_full_mean': float(np.mean(opacity_np)),
    }


def check_projection_matches_opacity(manual_proj, render_out, bbox_scaled):
    """Check if projected Gaussians fall on high-opacity regions."""
    if 'pred_opacity' not in render_out or not manual_proj or not manual_proj.get('projected'):
        return {'projection_matches_opacity': 'uncertain', 'suspected_issue': 'no_data'}

    opacity = render_out['pred_opacity']
    if opacity.ndim == 4 and opacity.shape[0] == 1:
        opacity = opacity[0]
    if opacity.ndim == 3 and opacity.shape[-1] == 1:
        opacity = opacity[:, :, 0]

    opacity_np = opacity.cpu().numpy()
    h, w = opacity_np.shape

    projected = manual_proj['projected']
    if not projected:
        return {'projection_matches_opacity': 'no', 'suspected_issue': 'no_projected_points'}

    # Sample opacity at projected points
    opacities_at_points = []
    for g in projected[:500]:  # Check first 500
        px, py = int(g['x']), int(g['y'])
        if 0 <= px < w and 0 <= py < h:
            opacities_at_points.append(float(opacity_np[py, px]))

    if not opacities_at_points:
        return {'projection_matches_opacity': 'uncertain', 'suspected_issue': 'points_out_of_bounds'}

    mean_opacity_at_points = np.mean(opacities_at_points)
    mean_opacity_global = np.mean(opacity_np)

    # Check if projected points fall on higher-opacity regions than average
    if mean_opacity_at_points > mean_opacity_global * 1.5:
        matches = 'yes'
        issue = 'none'
    elif mean_opacity_at_points > mean_opacity_global * 0.5:
        matches = 'uncertain'
        issue = 'partial_overlap'
    else:
        matches = 'no'
        issue = 'projected_points_on_low_opacity_regions'

    return {
        'projection_matches_opacity': matches,
        'suspected_issue': issue,
        'mean_opacity_at_projected_points': float(mean_opacity_at_points),
        'mean_opacity_global': float(mean_opacity_global),
        'opacity_ratio': float(mean_opacity_at_points / max(mean_opacity_global, 1e-10)),
    }


def draw_projection_overlay(image, manual_proj, render_out, bbox_scaled, sample_id, output_dir):
    """Create visualization overlays."""
    if image is None or image.size == 0:
        return

    orig_h, orig_w = image.shape[:2]

    # 1. Original image + bbox
    x1, y1, x2, y2 = [int(c) for c in bbox_scaled]
    overlay_orig = image.copy()
    cv2.rectangle(overlay_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_01_original_bbox.png"), overlay_orig)

    # 2. Rendered RGB
    if 'pred_rgb' in render_out:
        rgb = render_out['pred_rgb']
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]
        rgb_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if rgb_np.ndim == 3 and rgb_np.shape[-1] != 3:
            rgb_np = np.repeat(rgb_np, 3, axis=-1)
        cv2.imwrite(os.path.join(output_dir, f"{sample_id}_02_rendered_rgb.png"), rgb_np)

    # 3. Rendered opacity overlay
    if 'pred_opacity' in render_out:
        opacity = render_out['pred_opacity']
        if opacity.ndim == 4 and opacity.shape[0] == 1:
            opacity = opacity[0]
        if opacity.ndim == 3 and opacity.shape[-1] == 1:
            opacity = opacity[:, :, 0]
        opacity_np = (opacity.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        opacity_colormap = cv2.applyColorMap(opacity_np, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_dir, f"{sample_id}_03_opacity_heatmap.png"), opacity_colormap)

        # Opacity overlay on original
        if orig_h != opacity_np.shape[0] or orig_w != opacity_np.shape[1]:
            opacity_resized = cv2.resize(opacity_colormap, (orig_w, orig_h))
        else:
            opacity_resized = opacity_colormap
        blended = cv2.addWeighted(image, 0.5, opacity_resized, 0.5, 0)
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"{sample_id}_04_opacity_overlay.png"), blended)

    # 4. Manual all-projected Gaussian overlay
    if manual_proj and manual_proj.get('projected'):
        projected_img = image.copy()
        projected = manual_proj['projected'][:1000]
        opacities = [g['opacity'] for g in projected]
        max_op = max(opacities) if opacities else 1.0
        min_op = min(opacities) if opacities else 0.0
        op_range = max_op - min_op if max_op != min_op else 1.0

        for g in projected:
            px, py = int(g['x']), int(g['y'])
            if 0 <= px < orig_w and 0 <= py < orig_h:
                norm_op = (g['opacity'] - min_op) / op_range
                r = int(255 * norm_op)
                b = int(255 * (1 - norm_op))
                cv2.circle(projected_img, (px, py), 2, (b, 0, r), -1)

        cv2.rectangle(projected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(projected_img, f"Proj: {manual_proj['num_valid']}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"{sample_id}_05_manual_projected.png"), projected_img)

    # 5. Combined overlay: opacity + projected points + bbox
    if 'pred_opacity' in render_out and manual_proj and manual_proj.get('projected'):
        combined = blended.copy()
        projected = manual_proj['projected'][:500]
        for g in projected:
            px, py = int(g['x']), int(g['y'])
            if 0 <= px < orig_w and 0 <= py < orig_h:
                cv2.circle(combined, (px, py), 2, (255, 255, 0), -1)  # Yellow points

        cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"{sample_id}_06_combined_overlay.png"), combined)


def run_stage_a(args):
    print("\n" + "=" * 80)
    print("Stage A: Projection Consistency Check")
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

    print(f"\nPositions shape (after): {model.positions.shape}")
    print(f"Positions stats: mean={model.positions.mean().item():.3f}, std={model.positions.std().item():.3f}")

    batch_builder = BatchBuilder(dataset)

    # Load eval samples
    eval_samples_path = args.eval_samples
    if not eval_samples_path:
        eval_samples_path = os.path.join(REPO_ROOT, 'outputs/phase12_parallel_validation/medium_eval_allcam.json')

    if os.path.exists(eval_samples_path):
        with open(eval_samples_path, 'r') as f:
            eval_samples = json.load(f)
        print(f"\nLoaded {len(eval_samples)} eval samples")
    else:
        print(f"\nERROR: eval_samples not found at {eval_samples_path}")
        return

    samples_by_cam = defaultdict(list)
    for s in eval_samples:
        samples_by_cam[s['cam_id']].append(s)

    all_cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    per_camera_stats = {}
    per_sample_records = []

    for cam_id in all_cameras:
        cam_samples = samples_by_cam.get(cam_id, [])
        if not cam_samples:
            per_camera_stats[cam_id] = {'camera': cam_id, 'num_samples': 0, 'final_camera_verdict': 'no_samples'}
            continue

        target = min(len(cam_samples), args.samples_per_camera)
        sampled = random.sample(cam_samples, target) if len(cam_samples) > target else cam_samples

        print(f"\n{'='*60}")
        print(f"{cam_id}: {len(cam_samples)} available -> {len(sampled)} sampled")
        print(f"{'='*60}")

        cam_dir = os.path.join(args.output_dir, cam_id)
        os.makedirs(cam_dir, exist_ok=True)

        match_counts = {'yes': 0, 'no': 0, 'uncertain': 0}
        issue_counts = defaultdict(int)
        projection_counts = []
        inside_bbox_counts = []
        inside_bbox_scaled_counts = []
        opacity_bbox_sums = []

        for s_idx, sample in enumerate(sampled):
            person_id = sample.get('person_id', 'unknown')
            frame_id = sample.get('frame_id', sample.get('frame_idx', 'unknown'))
            bbox_original = sample.get('bbox', [0, 0, 0, 0])

            if isinstance(frame_id, str):
                frame_id = int(frame_id)

            sample_id = f"{cam_id}_frame{frame_id:06d}_pid{person_id:03d}_{s_idx:03d}"

            gpu_batch = batch_builder.get_batch(cam_id, frame_id)
            if gpu_batch is None:
                continue

            # Load original image
            image = None
            if cam_id in dataset.image_paths and frame_id < len(dataset.image_paths[cam_id]):
                image_path = dataset.image_paths[cam_id][frame_id]
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)

            # Render
            try:
                render_out = model(gpu_batch, train=False, frame_id=0)
            except Exception as e:
                per_sample_records.append({
                    'sample_id': sample_id, 'cam_id': cam_id, 'frame_id': frame_id,
                    'person_id': person_id, 'render_error': str(e)[:100],
                })
                continue

            # Manual projection
            manual_proj = manual_projection(model, gpu_batch, device)
            num_projected = manual_proj.get('num_valid', 0) if manual_proj else 0
            projection_counts.append(num_projected)

            # Bbox scale check
            bbox_info = check_bbox_projection_alignment(manual_proj, render_out, bbox_original, image if image is not None else np.zeros((1080, 1920, 3)), gpu_batch)
            bbox_scaled = bbox_info.get('bbox_scaled', bbox_original)

            # Compute opacity bbox stats
            opacity_stats = compute_opacity_bbox_stats(render_out, bbox_scaled)
            opacity_bbox_sums.append(opacity_stats.get('opacity_bbox_sum', 0))

            # Count projected Gaussians inside bbox (original coordinates - wrong!)
            x1o, y1o, x2o, y2o = bbox_original
            inside_orig = 0
            inside_scaled = 0
            x1s, y1s, x2s, y2s = bbox_scaled
            for g in (manual_proj or {}).get('projected', []):
                gx, gy = g['x'], g['y']
                if x1o <= gx < x2o and y1o <= gy < y2o:
                    inside_orig += 1
                if x1s <= gx < x2s and y1s <= gy < y2s:
                    inside_scaled += 1
            inside_bbox_counts.append(inside_orig)
            inside_bbox_scaled_counts.append(inside_scaled)

            # Check projection-opacity alignment
            alignment = check_projection_matches_opacity(manual_proj, render_out, bbox_scaled)
            match_result = alignment.get('projection_matches_opacity', 'uncertain')
            match_counts[match_result] = match_counts.get(match_result, 0) + 1
            issue = alignment.get('suspected_issue', 'unknown')
            issue_counts[issue] += 1

            # Determine suspected issue
            suspected = alignment.get('suspected_issue', 'unknown')
            if inside_orig == 0 and inside_scaled > 0:
                suspected = 'bbox_not_scaled_to_render_space'
            elif inside_orig == 0 and inside_scaled == 0:
                if alignment.get('opacity_ratio', 1) < 0.5:
                    suspected = 'projection_misaligned_with_opacity'
                else:
                    suspected = 'geometry_lacks_person_support_in_bbox'

            # Record
            record = {
                'sample_id': sample_id,
                'cam_id': cam_id,
                'frame_id': frame_id,
                'person_id': int(person_id),
                'original_image_size': bbox_info.get('original_image_size', [0, 0]),
                'render_size': bbox_info.get('render_size', [0, 0]),
                'bbox_original': list(bbox_original),
                'bbox_scaled': bbox_info.get('bbox_scaled', []),
                'scale_x': bbox_info.get('scale_x', 1),
                'scale_y': bbox_info.get('scale_y', 1),
                'num_projected_gaussians': num_projected,
                'num_valid_depth_gaussians': num_projected,
                'num_inside_bbox_original': inside_orig,
                'num_inside_bbox_scaled': inside_scaled,
                'projected_x_min': manual_proj.get('x_min', 0) if manual_proj else 0,
                'projected_x_max': manual_proj.get('x_max', 0) if manual_proj else 0,
                'projected_y_min': manual_proj.get('y_min', 0) if manual_proj else 0,
                'projected_y_max': manual_proj.get('y_max', 0) if manual_proj else 0,
                'projected_x_mean': manual_proj.get('x_mean', 0) if manual_proj else 0,
                'projected_y_mean': manual_proj.get('y_mean', 0) if manual_proj else 0,
                'opacity_nonzero_ratio': opacity_stats.get('opacity_nonzero_ratio', 0),
                'opacity_bbox_sum': opacity_stats.get('opacity_bbox_sum', 0),
                'projection_matches_opacity': match_result,
                'suspected_issue': suspected,
                'mean_opacity_at_projected_points': alignment.get('mean_opacity_at_projected_points', 0),
                'mean_opacity_global': alignment.get('mean_opacity_global', 0),
                'opacity_ratio': alignment.get('opacity_ratio', 0),
            }
            per_sample_records.append(record)

            # Draw visualizations (first 5 samples per camera)
            if s_idx < 5 and image is not None:
                draw_projection_overlay(image, manual_proj, render_out, bbox_scaled, sample_id, cam_dir)

            if s_idx < 3:
                print(f"  [{s_idx}] proj={num_projected}, inside_orig={inside_orig}, inside_scaled={inside_scaled}, "
                      f"match={match_result}, issue={suspected}, opacity_ratio={alignment.get('opacity_ratio', 0):.3f}")

        # Per-camera summary
        total = len(sampled)
        inside_orig_mean = np.mean(inside_bbox_counts) if inside_bbox_counts else 0
        inside_scaled_mean = np.mean(inside_bbox_scaled_counts) if inside_bbox_scaled_counts else 0
        opacity_bbox_mean = np.mean(opacity_bbox_sums) if opacity_bbox_sums else 0

        per_camera_stats[cam_id] = {
            'camera': cam_id,
            'num_samples': total,
            'mean_projected_gaussians': float(np.mean(projection_counts)) if projection_counts else 0,
            'mean_inside_bbox_original': float(inside_orig_mean),
            'mean_inside_bbox_scaled': float(inside_scaled_mean),
            'mean_opacity_bbox_sum': float(opacity_bbox_mean),
            'projection_match_counts': dict(match_counts),
            'projection_match_yes_ratio': match_counts.get('yes', 0) / max(1, total),
            'projection_match_no_ratio': match_counts.get('no', 0) / max(1, total),
            'issue_counts': dict(issue_counts),
            'final_camera_verdict': 'projection_aligned' if match_counts.get('yes', 0) > total * 0.5 else 'projection_misaligned',
        }

        print(f"\n  {cam_id} Summary:")
        print(f"    Mean projected: {np.mean(projection_counts) if projection_counts else 0:.1f}")
        print(f"    Mean inside bbox (orig coords): {inside_orig_mean:.1f}")
        print(f"    Mean inside bbox (scaled coords): {inside_scaled_mean:.1f}")
        print(f"    Projection matches opacity: {match_counts}")
        print(f"    Issues: {dict(issue_counts)}")

    # Save outputs
    with open(os.path.join(args.output_dir, 'projection_consistency_summary.json'), 'w') as f:
        json.dump(per_camera_stats, f, indent=2, default=str)

    with open(os.path.join(args.output_dir, 'projection_consistency_per_sample.jsonl'), 'w') as f:
        for record in per_sample_records:
            f.write(json.dumps(record, default=str) + '\n')

    # Generate report
    generate_stage_a_report(per_camera_stats, per_sample_records, args.output_dir)

    print(f"\n{'='*80}")
    print(f"Stage A Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


def generate_stage_a_report(per_camera_stats, per_sample_records, output_dir):
    """Generate Stage A projection consistency report."""

    total = len(per_sample_records)
    match_yes = sum(1 for r in per_sample_records if r.get('projection_matches_opacity') == 'yes')
    match_no = sum(1 for r in per_sample_records if r.get('projection_matches_opacity') == 'no')
    match_uncertain = sum(1 for r in per_sample_records if r.get('projection_matches_opacity') == 'uncertain')

    issue_counts = defaultdict(int)
    for r in per_sample_records:
        issue_counts[r.get('suspected_issue', 'unknown')] += 1

    # Check scale issues
    scale_issues = sum(1 for r in per_sample_records if r.get('num_inside_bbox_original', 0) == 0 and r.get('num_inside_bbox_scaled', 0) > 0)
    both_zero = sum(1 for r in per_sample_records if r.get('num_inside_bbox_original', 0) == 0 and r.get('num_inside_bbox_scaled', 0) == 0)

    report = f"""# Stage A: Projection Consistency Check Report

## 1. manual projection 是否和 renderer opacity 对齐？

| Alignment | Count | Ratio |
|-----------|-------|-------|
| yes | {match_yes} | {match_yes/max(1,total):.2%} |
| no | {match_no} | {match_no/max(1,total):.2%} |
| uncertain | {match_uncertain} | {match_uncertain/max(1,total):.2%} |

"""
    if match_yes > total * 0.5:
        report += "**结论**: ✅ 大部分投影点与 opacity 高响应区域对齐，projection 坐标系正确。\n"
    else:
        report += "**结论**: ❌ 投影点与 opacity 高响应区域不对齐，可能存在 projection / coordinate bug。\n"

    report += """
## 2. bbox 是否正确 scale？

"""
    if scale_issues > total * 0.3:
        report += f"⚠️ **发现**: {scale_issues}/{total} 样本的 bbox 未正确 scale 到 render space。\n"
        report += "原始 bbox 坐标用于 render resolution 导致 inside_bbox 计数为 0。\n"
        report += "**修复建议**: pooling 中应使用 bbox_scaled 而不是 bbox_original。\n\n"
    else:
        report += "✅ bbox scaling 不是主要问题。\n\n"

    # Per-camera table
    report += """## 3. Per-Camera Analysis

| Camera | Samples | Mean Projected | Inside BBox (Orig) | Inside BBox (Scaled) | Match Yes | Match No | Final Verdict |
|--------|---------|---------------|-------------------|---------------------|-----------|----------|---------------|
"""
    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = per_camera_stats.get(cam_id, {})
        report += (
            f"| {cam_id} | {stats.get('num_samples', 0)} | {stats.get('mean_projected_gaussians', 0):.1f} | "
            f"{stats.get('mean_inside_bbox_original', 0):.1f} | {stats.get('mean_inside_bbox_scaled', 0):.1f} | "
            f"{stats.get('projection_match_yes_ratio', 0):.2%} | {stats.get('projection_match_no_ratio', 0):.2%} | "
            f"{stats.get('final_camera_verdict', 'N/A')} |\n"
        )

    report += f"""
## 4. Stage2 bbox-empty 是否由 projection / coordinate bug 导致？

| Issue | Count |
|-------|-------|
| bbox_not_scaled_to_render_space | {issue_counts.get('bbox_not_scaled_to_render_space', 0)} |
| projection_misaligned_with_opacity | {issue_counts.get('projection_misaligned_with_opacity', 0)} |
| geometry_lacks_person_support_in_bbox | {issue_counts.get('geometry_lacks_person_support_in_bbox', 0)} |
| no_projected_points | {issue_counts.get('no_projected_points', 0)} |
| points_out_of_bounds | {issue_counts.get('points_out_of_bounds', 0)} |

"""
    if scale_issues > total * 0.3:
        report += f"**主要问题**: bbox scale issue 导致 {scale_issues}/{total} 样本 bbox-empty。\n"
        report += "这是一个可以修复的 bug，修复后 bbox support 应该显著改善。\n"
    elif issue_counts.get('projection_misaligned_with_opacity', 0) > total * 0.3:
        report += "**主要问题**: projection 与 renderer opacity 不对齐。\n"
        report += "需要检查投影公式是否与 renderer 一致。\n"
    else:
        report += "**主要问题**: geometry 确实缺少 person-level support。\n"
        report += "即使 projection 和 bbox 都正确，bbox 内也没有足够 Gaussian。\n"

    report += """
## 5. 修复后 C1-C7 bbox support 是否改善？

如果 bbox scale 是主要问题：
- 修复后 inside_bbox 应该从 0 变为 bbox_scaled 计数
- 预期改善幅度取决于 render vs original size ratio

## 6. 当前瓶颈

| 可能瓶颈 | 证据 | 严重度 |
|---------|------|--------|
"""
    if scale_issues > total * 0.3:
        report += f"| bbox scale bug | {scale_issues}/{total} 样本 bbox 未 scale | 🔴 高 (可修复) |\n"
    if issue_counts.get('projection_misaligned_with_opacity', 0) > total * 0.3:
        report += f"| projection 不对齐 | {issue_counts.get('projection_misaligned_with_opacity', 0)}/{total} 样本 | 🔴 高 |\n"
    if issue_counts.get('geometry_lacks_person_support_in_bbox', 0) > total * 0.3:
        report += f"| geometry 缺少 support | {issue_counts.get('geometry_lacks_person_support_in_bbox', 0)}/{total} 样本 | 🟡 中 |\n"

    report += """
## 7. 下一步建议

"""
    if scale_issues > total * 0.3:
        report += "1. 修复 pooling 中的 bbox scaling bug\n"
        report += "2. 使用 bbox_scaled 而不是 bbox_original\n"
        report += "3. clamp bbox 到 render image bounds\n"
        report += "4. 重新运行 Stage D overlay diagnostic\n"
    elif issue_counts.get('projection_misaligned_with_opacity', 0) > total * 0.3:
        report += "1. 使用 renderer-derived visibility pooling (Stage C)\n"
        report += "2. 不要维护手写 projection，复用 renderer 中间量\n"
        report += "3. 重新运行 Stage D overlay diagnostic\n"
    else:
        report += "1. projection / bbox 都不是主要问题\n"
        report += "2. geometry 确实缺少 person-level support\n"
        report += "3. 需要考虑 human-aware geometry retraining\n"

    with open(os.path.join(output_dir, 'stage_a_projection_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Stage A: Projection Consistency Check')

    parser.add_argument('--output_dir', type=str, default='outputs/phase12_projection_consistency_check')
    parser.add_argument('--geometry_checkpoint', type=str,
                        default='runs/phase11A_proto_infonce_opacity_lam005_lr1e4/step_20999.pth')
    parser.add_argument('--eval_samples', type=str, default=None)
    parser.add_argument('--samples_per_camera', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_stage_a(args)


if __name__ == '__main__':
    main()
