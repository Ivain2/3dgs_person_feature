#!/usr/bin/env python3
"""
Phase12 Stage B: BBox Gaussian Spatial Distribution Diagnostic

Goal: Since Stage A showed opacity is ~1.0 everywhere (non-discriminative),
we need a different approach to understand whether projected Gaussians
actually have person-level spatial support inside bboxes.

This script:
1. For each sample, compute the spatial relationship between projected Gaussians and bbox
2. Check if any Gaussians are near/inside the bbox (with margin)
3. Compute Gaussian density per bbox area
4. Compare against random geometry baseline
5. Generate detailed spatial distribution visualizations
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


def get_projected_gaussians(model, gpu_batch, device):
    """Get all projected Gaussian coordinates in render space."""
    xyz = model.positions
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

    valid = valid_depth & torch.isfinite(x_img) & torch.isfinite(y_img)

    return {
        'x': x_img.cpu().numpy(),
        'y': y_img.cpu().numpy(),
        'depth': depth.cpu().numpy(),
        'valid': valid.cpu().numpy(),
        'num_total': N,
        'num_valid': int(valid.sum().item()),
    }


def compute_gaussian_bbox_relationship(proj, bbox_scaled, render_w, render_h, margin_factor=0.5):
    """Compute detailed spatial relationship between Gaussians and bbox."""
    valid_mask = proj['valid']
    x_valid = proj['x'][valid_mask]
    y_valid = proj['y'][valid_mask]
    depth_valid = proj['depth'][valid_mask]
    num_valid = len(x_valid)

    x1, y1, x2, y2 = bbox_scaled
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(render_w, int(x2))
    y2 = min(render_h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return {
            'bbox_valid': False,
            'bbox_area': 0,
            'num_inside': 0,
            'num_inside_margin': 0,
            'gaussian_density': 0,
            'min_dist_to_center': 999999,
            'mean_dist_to_center': 999999,
            'median_dist_to_center': 999999,
            'depth_inside_mean': 0,
            'depth_inside_std': 0,
            'depth_all_mean': 0,
            'depth_all_std': 0,
            'fraction_within_1x_diag': 0,
            'fraction_within_2x_diag': 0,
            'fraction_within_3x_diag': 0,
            'num_valid_gaussians': num_valid,
        }

    bbox_area = (x2 - x1) * (y2 - y1)
    bbox_cx = (x1 + x2) / 2
    bbox_cy = (y1 + y2) / 2
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    valid_mask = proj['valid']
    x_valid = proj['x'][valid_mask]
    y_valid = proj['y'][valid_mask]
    depth_valid = proj['depth'][valid_mask]

    # Inside bbox
    inside = (x_valid >= x1) & (x_valid < x2) & (y_valid >= y1) & (y_valid < y2)
    num_inside = int(inside.sum())

    # Inside bbox + margin
    margin_x = bbox_w * margin_factor
    margin_y = bbox_h * margin_factor
    x1_m = max(0, x1 - margin_x)
    y1_m = max(0, y1 - margin_y)
    x2_m = min(render_w, x2 + margin_x)
    y2_m = min(render_h, y2 + margin_y)
    inside_margin = (x_valid >= x1_m) & (x_valid < x2_m) & (y_valid >= y1_m) & (y_valid < y2_m)
    num_inside_margin = int(inside_margin.sum())

    # Distance from bbox center
    dist_to_center = np.sqrt((x_valid - bbox_cx)**2 + (y_valid - bbox_cy)**2)
    min_dist = float(dist_to_center.min()) if len(dist_to_center) > 0 else 999999
    mean_dist = float(dist_to_center.mean()) if len(dist_to_center) > 0 else 999999
    median_dist = float(np.median(dist_to_center)) if len(dist_to_center) > 0 else 999999

    # Gaussian density per bbox area
    density = num_inside / max(bbox_area, 1)

    # Depth stats for Gaussians inside bbox
    if num_inside > 0:
        depth_inside = depth_valid[inside]
        depth_mean = float(depth_inside.mean())
        depth_std = float(depth_inside.std())
    else:
        depth_mean = 0
        depth_std = 0

    # Overall depth stats
    depth_mean_all = float(depth_valid.mean()) if len(depth_valid) > 0 else 0
    depth_std_all = float(depth_valid.std()) if len(depth_valid) > 0 else 0

    # Distance distribution: fraction within 1x, 2x, 3x bbox diagonal
    bbox_diag = np.sqrt(bbox_w**2 + bbox_h**2)
    within_1x = float((dist_to_center < bbox_diag).sum()) / max(len(dist_to_center), 1)
    within_2x = float((dist_to_center < 2 * bbox_diag).sum()) / max(len(dist_to_center), 1)
    within_3x = float((dist_to_center < 3 * bbox_diag).sum()) / max(len(dist_to_center), 1)

    return {
        'bbox_valid': True,
        'bbox_area': int(bbox_area),
        'bbox_cx': float(bbox_cx),
        'bbox_cy': float(bbox_cy),
        'bbox_w': int(bbox_w),
        'bbox_h': int(bbox_h),
        'num_inside': num_inside,
        'num_inside_margin': num_inside_margin,
        'gaussian_density': float(density),
        'min_dist_to_center': float(min_dist),
        'mean_dist_to_center': float(mean_dist),
        'median_dist_to_center': float(median_dist),
        'depth_inside_mean': float(depth_mean),
        'depth_inside_std': float(depth_std),
        'depth_all_mean': float(depth_mean_all),
        'depth_all_std': float(depth_std_all),
        'fraction_within_1x_diag': float(within_1x),
        'fraction_within_2x_diag': float(within_2x),
        'fraction_within_3x_diag': float(within_3x),
        'num_valid_gaussians': len(x_valid),
    }


def draw_spatial_overlay(image, proj, bbox_scaled, bbox_orig, sample_id, output_dir, render_w, render_h):
    """Create detailed spatial distribution visualization."""
    orig_h, orig_w = image.shape[:2]

    # 1. Original image with original bbox
    img1 = image.copy()
    ox1, oy1, ox2, oy2 = [int(c) for c in bbox_orig]
    cv2.rectangle(img1, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_01_orig_bbox.png"), img1)

    # 2. All valid Gaussians projected onto original image (need to scale up)
    scale_x = orig_w / render_w
    scale_y = orig_h / render_h
    img2 = image.copy()
    valid_mask = proj['valid']
    x_valid = proj['x'][valid_mask]
    y_valid = proj['y'][valid_mask]
    depth_valid = proj['depth'][valid_mask]

    # Color by depth
    if len(depth_valid) > 0:
        d_min, d_max = depth_valid.min(), depth_valid.max()
        d_range = d_max - d_min if d_max != d_min else 1.0
        for i in range(len(x_valid)):
            px = int(x_valid[i] * scale_x)
            py = int(y_valid[i] * scale_y)
            if 0 <= px < orig_w and 0 <= py < orig_h:
                norm_d = (depth_valid[i] - d_min) / d_range
                r = int(255 * norm_d)
                b = int(255 * (1 - norm_d))
                cv2.circle(img2, (px, py), 1, (b, 0, r), -1)

    # Draw original bbox
    cv2.rectangle(img2, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
    cv2.putText(img2, f"Total: {len(x_valid)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_02_all_proj_depth.png"), img2)

    # 3. Scaled bbox on original image
    img3 = image.copy()
    sx1, sy1, sx2, sy2 = [int(c) for c in bbox_scaled]
    cv2.rectangle(img3, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
    cv2.rectangle(img3, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
    cv2.putText(img3, "Green=Orig, Red=Scaled", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_03_bbox_compare.png"), img3)

    # 4. Close-up of bbox area (original bbox, scaled to fit view)
    if ox2 > ox1 and oy2 > oy1:
        crop = image[max(0,oy1-20):min(orig_h,oy2+20), max(0,ox1-20):min(orig_w,ox2+20)]
        if crop.size > 0:
            cv2.imwrite(os.path.join(output_dir, f"{sample_id}_04_bbox_crop.png"), crop)

    # 5. Gaussians inside bbox + margin on render-sized blank image
    x1, y1, x2, y2 = [int(c) for c in bbox_scaled]
    margin_x = int((x2 - x1) * 0.5)
    margin_y = int((y2 - y1) * 0.5)
    x1_m = max(0, x1 - margin_x)
    y1_m = max(0, y1 - margin_y)
    x2_m = min(render_w, x2 + margin_x)
    y2_m = min(render_h, y2 + margin_y)

    img5 = np.zeros((render_h, render_w, 3), dtype=np.uint8)
    inside_margin_mask = ((x_valid >= x1_m) & (x_valid < x2_m) & 
                          (y_valid >= y1_m) & (y_valid < y2_m))
    
    # Draw all in margin as yellow
    for i in range(len(x_valid)):
        if inside_margin_mask[i]:
            px, py = int(x_valid[i]), int(y_valid[i])
            cv2.circle(img5, (px, py), 2, (0, 255, 255), -1)
    
    # Draw inside as green
    inside_mask = ((x_valid >= x1) & (x_valid < x2) & (y_valid >= y1) & (y_valid < y2))
    for i in range(len(x_valid)):
        if inside_mask[i]:
            px, py = int(x_valid[i]), int(y_valid[i])
            cv2.circle(img5, (px, py), 3, (0, 255, 0), -1)

    cv2.rectangle(img5, (x1_m, y1_m), (x2_m, y2_m), (0, 165, 255), 1)
    cv2.rectangle(img5, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(img5, f"Inside: {int(inside_mask.sum())}, Margin: {int(inside_margin_mask.sum())}",
               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_05_bbox_gaussians.png"), img5)


def run_stage_b(args):
    print("\n" + "=" * 80)
    print("Stage B: BBox Gaussian Spatial Distribution Diagnostic")
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

    # Load checkpoint
    geo_ckpt_path = args.geometry_checkpoint
    if not os.path.exists(geo_ckpt_path):
        print(f"\nERROR: Geometry checkpoint not found at {geo_ckpt_path}")
        return

    print(f"\nLoading geometry from: {geo_ckpt_path}")
    geo_ckpt = torch.load(geo_ckpt_path, map_location=device)
    model_state = geo_ckpt.get('model_state_dict', geo_ckpt)

    for k in ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']:
        if k in model_state:
            param_name = k if hasattr(model, k) else ('_' + k if hasattr(model, '_' + k) else None)
            if param_name and hasattr(model, param_name):
                getattr(model, param_name).data = model_state[k].to(device)
                getattr(model, param_name).requires_grad = False

    print(f"Positions: {model.positions.shape}, mean={model.positions.mean().item():.3f}, std={model.positions.std().item():.3f}")

    batch_builder = BatchBuilder(dataset)

    # Load eval samples
    eval_samples_path = args.eval_samples
    if not eval_samples_path:
        eval_samples_path = os.path.join(REPO_ROOT, 'outputs/phase12_parallel_validation/medium_eval_allcam.json')

    with open(eval_samples_path, 'r') as f:
        eval_samples = json.load(f)
    print(f"\nLoaded {len(eval_samples)} eval samples")

    samples_by_cam = defaultdict(list)
    for s in eval_samples:
        samples_by_cam[s['cam_id']].append(s)

    all_cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    per_camera_stats = {}
    per_sample_records = []

    for cam_id in all_cameras:
        cam_samples = samples_by_cam.get(cam_id, [])
        if not cam_samples:
            per_camera_stats[cam_id] = {'camera': cam_id, 'num_samples': 0}
            continue

        target = min(len(cam_samples), args.samples_per_camera)
        sampled = random.sample(cam_samples, target) if len(cam_samples) > target else cam_samples

        print(f"\n{'='*60}")
        print(f"{cam_id}: {len(cam_samples)} available -> {len(sampled)} sampled")
        print(f"{'='*60}")

        cam_dir = os.path.join(args.output_dir, cam_id)
        os.makedirs(cam_dir, exist_ok=True)

        all_rels = []
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

            proj = get_projected_gaussians(model, gpu_batch, device)
            if proj is None:
                continue

            render_w, render_h = int(gpu_batch.rays_dir.shape[2]), int(gpu_batch.rays_dir.shape[1])

            # Scale bbox to render space
            image = None
            if cam_id in dataset.image_paths and frame_id < len(dataset.image_paths[cam_id]):
                image_path = dataset.image_paths[cam_id][frame_id]
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)

            if image is not None:
                orig_h, orig_w = image.shape[:2]
                scale_x = render_w / orig_w
                scale_y = render_h / orig_h
            else:
                orig_w, orig_h = 1920, 1080
                scale_x = render_w / orig_w
                scale_y = render_h / orig_h

            x1o, y1o, x2o, y2o = bbox_original
            bbox_scaled = [x1o * scale_x, y1o * scale_y, x2o * scale_x, y2o * scale_y]

            rel = compute_gaussian_bbox_relationship(proj, bbox_scaled, render_w, render_h, margin_factor=0.5)

            record = {
                'sample_id': sample_id,
                'cam_id': cam_id,
                'frame_id': frame_id,
                'person_id': int(person_id),
                'render_size': [render_w, render_h],
                'bbox_original': list(bbox_original),
                'bbox_scaled': bbox_scaled,
                'num_valid_gaussians': proj['num_valid'],
                **rel
            }
            per_sample_records.append(record)
            all_rels.append(rel)

            # Draw visualizations for first 5 samples
            if s_idx < 5 and image is not None:
                draw_spatial_overlay(image, proj, bbox_scaled, bbox_original, sample_id, cam_dir, render_w, render_h)

            if s_idx < 3:
                print(f"  [{s_idx}] valid={proj['num_valid']}, inside={rel['num_inside']}, "
                      f"inside_margin={rel['num_inside_margin']}, density={rel['gaussian_density']:.4f}, "
                      f"min_dist={rel['min_dist_to_center']:.1f}, "
                      f"within_1x={rel['fraction_within_1x_diag']:.1%}")

        # Per-camera summary
        valid_rels = [r for r in all_rels if r['bbox_valid']]
        if valid_rels:
            per_camera_stats[cam_id] = {
                'camera': cam_id,
                'num_samples': len(sampled),
                'mean_num_valid_gaussians': float(np.mean([r['num_valid_gaussians'] for r in all_rels])),
                'mean_inside': float(np.mean([r['num_inside'] for r in valid_rels])),
                'mean_inside_margin': float(np.mean([r['num_inside_margin'] for r in valid_rels])),
                'mean_density': float(np.mean([r['gaussian_density'] for r in valid_rels])),
                'mean_min_dist': float(np.mean([r['min_dist_to_center'] for r in valid_rels])),
                'mean_median_dist': float(np.mean([r['median_dist_to_center'] for r in valid_rels])),
                'mean_within_1x': float(np.mean([r['fraction_within_1x_diag'] for r in valid_rels])),
                'mean_within_2x': float(np.mean([r['fraction_within_2x_diag'] for r in valid_rels])),
                'mean_within_3x': float(np.mean([r['fraction_within_3x_diag'] for r in valid_rels])),
                'mean_bbox_area': float(np.mean([r['bbox_area'] for r in valid_rels])),
                'median_inside': float(np.median([r['num_inside'] for r in valid_rels])),
                'max_inside': int(np.max([r['num_inside'] for r in valid_rels])),
                'samples_with_any_inside': int(sum(1 for r in valid_rels if r['num_inside'] > 0)),
                'samples_with_any_margin': int(sum(1 for r in valid_rels if r['num_inside_margin'] > 0)),
            }
        else:
            per_camera_stats[cam_id] = {
                'camera': cam_id,
                'num_samples': len(sampled),
                'bbox_valid_count': 0,
            }

        print(f"\n  {cam_id} Summary:")
        stats = per_camera_stats[cam_id]
        for k, v in stats.items():
            if k != 'camera':
                print(f"    {k}: {v}")

    # Save outputs
    with open(os.path.join(args.output_dir, 'spatial_distribution_summary.json'), 'w') as f:
        json.dump(per_camera_stats, f, indent=2, default=str)

    with open(os.path.join(args.output_dir, 'spatial_distribution_per_sample.jsonl'), 'w') as f:
        for record in per_sample_records:
            f.write(json.dumps(record, default=str) + '\n')

    # Generate report
    generate_stage_b_report(per_camera_stats, per_sample_records, args.output_dir)

    print(f"\n{'='*80}")
    print(f"Stage B Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


def generate_stage_b_report(per_camera_stats, per_sample_records, output_dir):
    total = len(per_sample_records)
    any_inside = sum(1 for r in per_sample_records if r.get('num_inside', 0) > 0)
    any_margin = sum(1 for r in per_sample_records if r.get('num_inside_margin', 0) > 0)

    report = f"""# Stage B: BBox Gaussian Spatial Distribution Report

## Key Findings

| Metric | Value |
|--------|-------|
| Total samples | {total} |
| Samples with any Gaussian inside bbox | {any_inside} ({any_inside/max(1,total):.1%}) |
| Samples with any Gaussian inside bbox+margin | {any_margin} ({any_margin/max(1,total):.1%}) |

## Per-Camera Spatial Distribution

| Camera | Valid Gaussians | Mean Inside | Mean Inside Margin | Mean Density | Mean Min Dist | Median Inside | Within 1x Diag | Within 2x Diag |
|--------|----------------|-------------|-------------------|--------------|---------------|---------------|----------------|----------------|
"""
    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = per_camera_stats.get(cam_id, {})
        report += (
            f"| {cam_id} | {stats.get('mean_num_valid_gaussians', 0):.0f} | "
            f"{stats.get('mean_inside', 0):.1f} | {stats.get('mean_inside_margin', 0):.1f} | "
            f"{stats.get('mean_density', 0):.4f} | {stats.get('mean_min_dist', 0):.1f} | "
            f"{stats.get('median_inside', 0):.0f} | "
            f"{stats.get('mean_within_1x', 0):.1%} | {stats.get('mean_within_2x', 0):.1%} |\n"
        )

    report += f"""
## Interpretation

### 1. Are projected Gaussians falling inside bbox?
"""
    if any_inside > total * 0.3:
        report += f"✅ YES: {any_inside}/{total} samples have Gaussians inside bbox.\n"
    elif any_margin > total * 0.3:
        report += f"⚠️ PARTIAL: Only {any_margin}/{total} have Gaussians in bbox+margin, {any_inside}/{total} inside bbox.\n"
    else:
        report += f"❌ NO: Only {any_inside}/{total} inside bbox, {any_margin}/{total} inside margin.\n"

    report += f"""
### 2. Is geometry providing person-level support?

The key question: are the few Gaussians inside the bbox actually on the person,
or are they just randomly scattered background Gaussians?

Evidence from spatial distribution:
- Mean Gaussians inside bbox per camera: see table above
- If mean_inside < 5: geometry is NOT providing person-level support
- If mean_inside > 20: geometry has moderate person-level support
- If mean_inside > 50: geometry has strong person-level support

### 3. Is projection / bbox scale the bottleneck?

Stage A showed:
- Projection coordinates are correct (render space)
- Bbox scaling is a minor issue (120/280 samples affected)
- Even with correct bbox_scaled, inside_bbox counts are near zero

This confirms the bottleneck is NOT projection or bbox scale.

### 4. Next Steps Recommendation

"""
    mean_all_inside = np.mean([r.get('num_inside', 0) for r in per_sample_records])
    if mean_all_inside < 5:
        report += "**Conclusion**: Geometry lacks person-level support.\n\n"
        report += "Recommended actions:\n"
        report += "1. Human-aware geometry retraining (bbox-guided densification)\n"
        report += "2. Person-aware opacity support loss\n"
        report += "3. Increase Gaussian density in person regions\n"
        report += "4. Consider hybrid approach: 2D features + 3D geometry\n"
    else:
        report += f"**Conclusion**: Mean {mean_all_inside:.1f} Gaussians inside bbox — moderate support.\n\n"
        report += "Recommended actions:\n"
        report += "1. Improve pooling selection (top-alpha, center-aware)\n"
        report += "2. Consider teacher-only warm-up\n"

    with open(os.path.join(output_dir, 'stage_b_spatial_distribution_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Stage B: BBox Gaussian Spatial Distribution')

    parser.add_argument('--output_dir', type=str, default='outputs/phase12_spatial_distribution_check')
    parser.add_argument('--geometry_checkpoint', type=str,
                        default='runs/phase11A_proto_infonce_opacity_lam005_lr1e4/step_20999.pth')
    parser.add_argument('--eval_samples', type=str, default=None)
    parser.add_argument('--samples_per_camera', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_stage_b(args)


if __name__ == '__main__':
    main()
