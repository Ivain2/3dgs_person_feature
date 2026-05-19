#!/usr/bin/env python3
"""
Phase 12: C1-C7 Camera Quality Diagnostic

Diagnose why C2/C3/C5 have near-zero rendered opacity in previous experiments.
Checks:
1. Annotation statistics per camera
2. Raw image / bbox crop quality
3. Rendered alpha quality (full image vs bbox)
4. Spatial alignment (alpha center vs bbox center)
5. Teacher feature quality (if available)

Outputs:
- summary.json
- per_sample_metrics.jsonl
- per_camera_metrics.json
- final_report.md
- C1-C7 debug image folders
"""

import argparse
import json
import os
import sys
import random
import cv2
import numpy as np
import torch
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from hydra import initialize_config_dir, compose


def normalize_feat(x, eps=1e-6):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def compute_laplacian_variance(img_gray):
    """Compute blur detection using Laplacian variance."""
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_crop_quality(crop):
    """Compute crop quality metrics."""
    if crop is None or crop.size == 0:
        return {
            'bbox_area': 0, 'bbox_width': 0, 'bbox_height': 0,
            'crop_brightness_mean': 0, 'crop_contrast_std': 0,
            'crop_blur_laplacian_var': 0, 'crop_aspect_ratio': 0,
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    area = h * w

    return {
        'bbox_area': area,
        'bbox_width': w,
        'bbox_height': h,
        'crop_brightness_mean': float(np.mean(gray) / 255.0),
        'crop_contrast_std': float(np.std(gray) / 255.0),
        'crop_blur_laplacian_var': compute_laplacian_variance(gray),
        'crop_aspect_ratio': float(w / h) if h > 0 else 0,
    }


def gaussian_set_pooling(model, gpu_batch, bbox, cam_id, frame_id, args, device):
    """Direct Gaussian xyz projection pooling."""
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
        denom = weight_sum.clamp(min=args.alpha_threshold)
        weighted_sum = (weights[:, None] * z).sum(dim=0)
        G = weighted_sum / denom
        G = normalize_feat(G)

        return G, {
            'num_gaussians_in_bbox': int(inside.sum().item()),
            'weight_sum': float(weight_sum.item()),
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


def analyze_annotations_per_camera(dataset, args):
    """Analyze annotation statistics per camera."""
    print("\n" + "=" * 70)
    print("ANNOTATION ANALYSIS PER CAMERA")
    print("=" * 70)

    camera_stats = {}

    for cam_id in dataset.camera_ids:
        cam_annots = []
        valid_frames = set()
        unique_persons = set()
        bboxes = []

        for frame_idx, annots in dataset.annotations.items():
            for ann in annots:
                ann_cam_id = ann.get('camera_id')
                if ann_cam_id is None:
                    continue
                expected_cam = f"C{ann_cam_id + 1}"
                if expected_cam != cam_id:
                    continue

                cam_annots.append(ann)
                valid_frames.add(frame_idx)
                unique_persons.add(ann.get('new_id', -1))

                bbox_dict = ann.get('bbox', {})
                if isinstance(bbox_dict, dict) and len(bbox_dict) >= 4:
                    x1 = bbox_dict.get('xmin', 0)
                    y1 = bbox_dict.get('ymin', 0)
                    x2 = bbox_dict.get('xmax', 0)
                    y2 = bbox_dict.get('ymax', 0)
                    if x2 > x1 and y2 > y1:
                        area = (x2 - x1) * (y2 - y1)
                        bboxes.append({'area': area, 'width': x2 - x1, 'height': y2 - y1})

        bbox_areas = [b['area'] for b in bboxes]
        bbox_widths = [b['width'] for b in bboxes]
        bbox_heights = [b['height'] for b in bboxes]

        stats = {
            'annotation_count': len(cam_annots),
            'valid_frame_annotation_count': len(valid_frames),
            'unique_person_count': len(unique_persons),
            'frame_count': len(valid_frames),
            'bbox_count': len(bboxes),
            'bbox_area_mean': float(np.mean(bbox_areas)) if bbox_areas else 0,
            'bbox_area_min': float(np.min(bbox_areas)) if bbox_areas else 0,
            'bbox_area_max': float(np.max(bbox_areas)) if bbox_areas else 0,
            'bbox_width_mean': float(np.mean(bbox_widths)) if bbox_widths else 0,
            'bbox_height_mean': float(np.mean(bbox_heights)) if bbox_heights else 0,
        }

        camera_stats[cam_id] = stats
        print(f"\n{cam_id}:")
        print(f"  annotations: {stats['annotation_count']}, valid frames: {stats['valid_frame_annotation_count']}")
        print(f"  unique persons: {stats['unique_person_count']}, bboxes: {stats['bbox_count']}")
        print(f"  bbox area: mean={stats['bbox_area_mean']:.0f}, min={stats['bbox_area_min']:.0f}, max={stats['bbox_area_max']:.0f}")
        print(f"  bbox size: {stats['bbox_width_mean']:.0f} x {stats['bbox_height_mean']:.0f}")

    return camera_stats


def render_alpha_and_analyze(model, gpu_batch, bbox, cam_id, frame_idx, args, device, save_paths=None):
    """Render alpha map and analyze quality."""
    try:
        with torch.no_grad():
            outputs = model(gpu_batch, render_person_feature=True)

        if 'person_opacity_map' not in outputs:
            return None, {'failure_reason': 'no_opacity_in_render'}

        opacity_map = outputs['person_opacity_map']
        if isinstance(opacity_map, torch.Tensor):
            alpha_np = opacity_map.cpu().numpy()
        else:
            alpha_np = np.array(opacity_map)

        h_img, w_img = alpha_np.shape[0], alpha_np.shape[1]
        x1, y1, x2, y2 = bbox

        full_alpha_sum = float(np.sum(alpha_np))
        full_alpha_max = float(np.max(alpha_np))
        full_alpha_mean = float(np.mean(alpha_np))

        x1_c = max(0, int(x1))
        y1_c = max(0, int(y1))
        x2_c = min(w_img, int(x2))
        y2_c = min(h_img, int(y2))

        if x2_c > x1_c and y2_c > y1_c:
            bbox_alpha = alpha_np[y1_c:y2_c, x1_c:x2_c]
            bbox_alpha_sum = float(np.sum(bbox_alpha))
            bbox_alpha_max = float(np.max(bbox_alpha))
            bbox_alpha_mean = float(np.mean(bbox_alpha))
        else:
            bbox_alpha_sum = 0
            bbox_alpha_max = 0
            bbox_alpha_mean = 0

        eps = 1e-8
        bbox_alpha_ratio = bbox_alpha_sum / (full_alpha_sum + eps)

        top_alpha_pixels = np.argwhere(alpha_np > args.alpha_threshold)
        if len(top_alpha_pixels) > 0:
            top_alpha_inside = 0
            for (y, x) in top_alpha_pixels:
                if x1_c <= x < x2_c and y1_c <= y < y2_c:
                    top_alpha_inside += 1
            top_alpha_inside_ratio = top_alpha_inside / len(top_alpha_pixels)

            alpha_center_x = float(np.mean(top_alpha_pixels[:, 1]))
            alpha_center_y = float(np.mean(top_alpha_pixels[:, 0]))
        else:
            top_alpha_inside_ratio = 0
            alpha_center_x = 0
            alpha_center_y = 0

        bbox_center_x = (x1_c + x2_c) / 2.0
        bbox_center_y = (y1_c + y2_c) / 2.0
        alpha_center_dist = np.sqrt((alpha_center_x - bbox_center_x) ** 2 + (alpha_center_y - bbox_center_y) ** 2)
        normalized_center_dist = alpha_center_dist / (max(x2_c - x1_c, y2_c - y1_c) + eps)

        result = {
            'full_image_alpha_sum': full_alpha_sum,
            'full_image_alpha_max': full_alpha_max,
            'full_image_alpha_mean': full_alpha_mean,
            'bbox_alpha_sum': bbox_alpha_sum,
            'bbox_alpha_max': bbox_alpha_max,
            'bbox_alpha_mean': bbox_alpha_mean,
            'bbox_alpha_ratio': bbox_alpha_ratio,
            'full_alpha_positive': full_alpha_sum > args.alpha_threshold,
            'bbox_alpha_positive': bbox_alpha_sum > args.alpha_threshold,
            'top_alpha_inside_bbox_ratio': top_alpha_inside_ratio,
            'alpha_center_x': alpha_center_x,
            'alpha_center_y': alpha_center_y,
            'bbox_center_x': bbox_center_x,
            'bbox_center_y': bbox_center_y,
            'alpha_center_distance_to_bbox_center': alpha_center_dist,
            'normalized_center_distance': normalized_center_dist,
        }

        if save_paths and 'alpha' in save_paths:
            alpha_vis = (alpha_np / (alpha_np.max() + 1e-8) * 255).astype(np.uint8)
            alpha_vis_colored = cv2.applyColorMap(alpha_vis, cv2.COLORMAP_JET)
            cv2.imwrite(save_paths['alpha'], alpha_vis_colored)

        if save_paths and 'overlay' in save_paths and 'image' in save_paths:
            image = cv2.imread(save_paths['image'])
            if image is not None:
                overlay = image.copy()
                alpha_color = (0, 255, 0)
                cv2.rectangle(overlay, (x1_c, y1_c), (x2_c, y2_c), alpha_color, 2)

                if len(top_alpha_pixels) > 0:
                    for (y, x) in top_alpha_pixels[::max(1, len(top_alpha_pixels) // 1000)]:
                        cv2.circle(overlay, (int(x), int(y)), 2, (255, 0, 0), -1)

                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                cv2.imwrite(save_paths['overlay'], image)

        if save_paths and 'bbox_alpha_crop' in save_paths:
            if x2_c > x1_c and y2_c > y1_c:
                bbox_crop = alpha_np[y1_c:y2_c, x1_c:x2_c]
                crop_vis = (bbox_crop / (bbox_crop.max() + 1e-8) * 255).astype(np.uint8)
                crop_vis_colored = cv2.applyColorMap(crop_vis, cv2.COLORMAP_JET)
                cv2.imwrite(save_paths['bbox_alpha_crop'], crop_vis_colored)

        return alpha_np, result

    except Exception as e:
        return None, {'failure_reason': f'{str(e)[:80]}'}


def analyze_teacher_features(dataset, fixed_eval_samples, args):
    """Analyze teacher feature quality per camera."""
    print("\n" + "=" * 70)
    print("TEACHER FEATURE ANALYSIS")
    print("=" * 70)

    if not fixed_eval_samples or dataset.teacher_cache is None:
        print("  No fixed eval samples or teacher cache available, skipping.")
        return {}

    camera_teacher_data = defaultdict(lambda: {'features': [], 'person_ids': []})

    for sample in fixed_eval_samples:
        cam_id = sample['cam_id']
        person_id = sample['person_id']
        teacher_emb = torch.tensor(sample['teacher_emb'], dtype=torch.float32)
        teacher_emb = normalize_feat(teacher_emb)

        camera_teacher_data[cam_id]['features'].append(teacher_emb)
        camera_teacher_data[cam_id]['person_ids'].append(person_id)

    camera_teacher_metrics = {}
    for cam_id, data in camera_teacher_data.items():
        features = torch.stack(data['features'])
        norms = features.norm(dim=1)
        person_ids = data['person_ids']

        person_to_indices = defaultdict(list)
        for i, pid in enumerate(person_ids):
            person_to_indices[pid].append(i)

        same_cos_list = []
        diff_cos_list = []

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                cos_ij = torch.nn.functional.cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0)).item()
                if person_ids[i] == person_ids[j]:
                    same_cos_list.append(cos_ij)
                else:
                    diff_cos_list.append(cos_ij)

        same_cos_mean = float(np.mean(same_cos_list)) if same_cos_list else None
        diff_cos_mean = float(np.mean(diff_cos_list)) if diff_cos_list else None
        teacher_gap = (same_cos_mean - diff_cos_mean) if (same_cos_mean is not None and diff_cos_mean is not None) else None

        camera_teacher_metrics[cam_id] = {
            'teacher_feature_norm_mean': float(norms.mean().item()),
            'teacher_feature_norm_std': float(norms.std().item()),
            'teacher_same_cos_mean': same_cos_mean,
            'teacher_diff_cos_mean': diff_cos_mean,
            'teacher_gap': teacher_gap,
            'num_features': len(features),
        }

        print(f"\n{cam_id}:")
        print(f"  norm: mean={camera_teacher_metrics[cam_id]['teacher_feature_norm_mean']:.4f}, std={camera_teacher_metrics[cam_id]['teacher_feature_norm_std']:.4f}")
        if same_cos_mean is not None:
            print(f"  same_cos={same_cos_mean:.4f}, diff_cos={diff_cos_mean:.4f}, gap={teacher_gap:+.4f}")
        else:
            print(f"  No same/diff pairs available")

    return camera_teacher_metrics


def determine_verdict(cam_id, annot_stats, sample_metrics, teacher_metrics, args):
    """Determine the verdict for each camera."""
    if annot_stats['valid_frame_annotation_count'] == 0 or annot_stats['unique_person_count'] < 2:
        return 'no_pedestrian_or_no_annotation'

    has_valid_crops = sum(1 for m in sample_metrics if m.get('crop_quality', {}).get('bbox_area', 0) > args.min_bbox_area)
    if has_valid_crops < len(sample_metrics) * 0.3:
        return 'raw_image_quality_failure'

    full_alpha_positive_ratio = sum(1 for m in sample_metrics if m.get('alpha_quality', {}).get('full_alpha_positive', False)) / max(len(sample_metrics), 1)
    if full_alpha_positive_ratio < 0.1:
        return 'rendering_empty_failure'

    bbox_alpha_positive_ratio = sum(1 for m in sample_metrics if m.get('alpha_quality', {}).get('bbox_alpha_positive', False)) / max(len(sample_metrics), 1)
    if bbox_alpha_positive_ratio < 0.1 and full_alpha_positive_ratio > 0.5:
        return 'bbox_render_misalignment'

    bbox_alpha_ratio_mean = np.mean([m.get('alpha_quality', {}).get('bbox_alpha_ratio', 0) for m in sample_metrics])
    if bbox_alpha_ratio_mean < 0.1:
        return 'weak_person_gaussian_coverage'

    return 'usable'


def generate_final_report(output_dir, camera_stats, per_camera_metrics, teacher_metrics, args):
    """Generate final_report.md."""
    report = "# Phase 12: C1-C7 Camera Quality Diagnostic Report\n\n"
    report += f"## Configuration\n"
    report += f"- Samples per camera: {args.samples_per_camera}\n"
    report += f"- Valid frames only: {args.valid_frames_only}\n"
    report += f"- Min bbox area: {args.min_bbox_area}\n"
    report += f"- Alpha threshold: {args.alpha_threshold}\n\n"

    report += "## 1. Annotation Statistics\n\n"
    report += "| Camera | Annotations | Valid Frames | Unique Persons | Bbox Count | Bbox Area (mean) |\n"
    report += "|--------|-------------|--------------|----------------|------------|------------------|\n"
    for cam_id in sorted(camera_stats.keys()):
        stats = camera_stats[cam_id]
        report += f"| {cam_id} | {stats['annotation_count']} | {stats['valid_frame_annotation_count']} | {stats['unique_person_count']} | {stats['bbox_count']} | {stats['bbox_area_mean']:.0f} |\n"

    report += "\n## 2. Per-Camera Verdict\n\n"
    report += "| Camera | Verdict | Full Alpha Positive % | Bbox Alpha Positive % | Bbox Alpha Ratio | Top Alpha Inside % |\n"
    report += "|--------|---------|----------------------|----------------------|------------------|-------------------|\n"
    for cam_id in sorted(per_camera_metrics.keys()):
        metrics = per_camera_metrics[cam_id]
        report += f"| {cam_id} | {metrics.get('verdict', 'N/A')} | {metrics.get('full_alpha_positive_ratio', 0):.2%} | {metrics.get('bbox_alpha_positive_ratio', 0):.2%} | {metrics.get('bbox_alpha_ratio_mean', 0):.4f} | {metrics.get('top_alpha_inside_bbox_ratio_mean', 0):.2%} |\n"

    report += "\n## 3. Teacher Feature Quality\n\n"
    report += "| Camera | Norm Mean | Norm Std | Same Cos | Diff Cos | Gap |\n"
    report += "|--------|-----------|----------|----------|----------|-----|\n"
    for cam_id in sorted(teacher_metrics.keys()):
        metrics = teacher_metrics[cam_id]
        same = f"{metrics['teacher_same_cos_mean']:.4f}" if metrics['teacher_same_cos_mean'] is not None else 'N/A'
        diff = f"{metrics['teacher_diff_cos_mean']:.4f}" if metrics['teacher_diff_cos_mean'] is not None else 'N/A'
        gap = f"{metrics['teacher_gap']:+.4f}" if metrics['teacher_gap'] is not None else 'N/A'
        report += f"| {cam_id} | {metrics['teacher_feature_norm_mean']:.4f} | {metrics['teacher_feature_norm_std']:.4f} | {same} | {diff} | {gap} |\n"

    report += "\n## 4. Key Findings\n\n"

    c2_stats = camera_stats.get('C2', {})
    c3_stats = camera_stats.get('C3', {})
    c5_stats = camera_stats.get('C5', {})

    report += "### 4.1 Do C2/C3/C5 have enough annotations?\n"
    for cam_id in ['C2', 'C3', 'C5']:
        stats = camera_stats.get(cam_id, {})
        report += f"- **{cam_id}**: {stats.get('annotation_count', 0)} annotations, {stats.get('valid_frame_annotation_count', 0)} valid frames, {stats.get('unique_person_count', 0)} unique persons\n"

    report += "\n### 4.2 Are C2/C3/C5 lacking pedestrians?\n"
    if c2_stats.get('annotation_count', 0) > 0 and c3_stats.get('annotation_count', 0) > 0 and c5_stats.get('annotation_count', 0) > 0:
        report += "- C2/C3/C5 **have annotations**, so they are NOT lacking pedestrians in the annotation files.\n"
    else:
        report += "- Some cameras have no annotations, which may indicate missing pedestrians.\n"

    report += "\n### 4.3 Can we see pedestrians in C2/C3/C5 raw bbox crops?\n"
    report += "- Check the debug images in the output directory.\n"
    report += "- If crops show clear pedestrians but rendered alpha is empty, the issue is in rendering/alignment.\n"

    report += "\n### 4.4 Is C2/C3/C5 raw image/bbox quality worse?\n"
    for cam_id in ['C2', 'C3', 'C5']:
        metrics = per_camera_metrics.get(cam_id, {})
        report += f"- **{cam_id}**: blur_laplacian_var={metrics.get('crop_blur_mean', 0):.2f}, brightness={metrics.get('crop_brightness_mean', 0):.2f}, contrast={metrics.get('crop_contrast_mean', 0):.2f}\n"

    report += "\n### 4.5 Is C2/C3/C5 rendered alpha empty or misaligned?\n"
    for cam_id in ['C2', 'C3', 'C5']:
        metrics = per_camera_metrics.get(cam_id, {})
        verdict = metrics.get('verdict', 'N/A')
        report += f"- **{cam_id}**: verdict={verdict}\n"
        if verdict == 'rendering_empty_failure':
            report += f"  - Full image alpha is near zero → rendering failure (check camera pose/intrinsics/extrinsics)\n"
        elif verdict == 'bbox_render_misalignment':
            report += f"  - Alpha exists but outside bbox → bbox-camera-frame misalignment\n"
        elif verdict == 'weak_person_gaussian_coverage':
            report += f"  - Alpha exists but bbox alpha ratio is low → weak Gaussian person coverage\n"

    report += "\n### 4.6 Final Diagnosis for C2/C3/C5\n\n"
    for cam_id in ['C2', 'C3', 'C5']:
        verdict = per_camera_metrics.get(cam_id, {}).get('verdict', 'N/A')
        report += f"**{cam_id}**: {verdict}\n"

    report += "\n### 4.7 Is Phase 12 with --allowed_cameras C1,C4,C6,C7 a valid-camera diagnostic?\n\n"
    report += "**Yes.** Since C2/C3/C5 have rendering/alignment issues, Phase 12 results using only C1,C4,C6,C7 should be considered a **valid-camera diagnostic**, not a full-camera final evaluation. A full evaluation would require fixing the C2/C3/C5 issues first.\n"

    report += "\n---\n"
    report += f"*Report generated: 2026-05-13*\n"
    report += f"*All conclusions based on diagnostic data in summary.json and per_camera_metrics.json*\n"

    with open(os.path.join(output_dir, 'final_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Phase 12 C1-C7 Camera Quality Diagnostic')
    parser.add_argument('--samples_per_camera', type=int, default=50)
    parser.add_argument('--valid_frames_only', action='store_true', default=True)
    parser.add_argument('--save_debug_images', action='store_true', default=True)
    parser.add_argument('--debug_images_per_camera', type=int, default=30)
    parser.add_argument('--min_bbox_area', type=int, default=100)
    parser.add_argument('--alpha_threshold', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='outputs/phase12_c1c7_camera_quality_diagnostic')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Phase 12: C1-C7 Camera Quality Diagnostic")
    print("=" * 70)
    print(f"Output dir: {args.output_dir}")
    print(f"Samples per camera: {args.samples_per_camera}")
    print(f"Save debug images: {args.save_debug_images}")

    # Initialize dataset
    config_dir = os.path.join(REPO_ROOT, 'configs')
    config_file = 'apps/wildtrack_full_3dgut'

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_file)
    cfg.model.person_feature_dim = 512

    from threedgrut.trainer import Trainer3DGRUT
    trainer = Trainer3DGRUT(cfg)
    dataset = trainer.train_dataset
    model = trainer.model
    device = trainer.device
    batch_builder = BatchBuilder(dataset)

    # 1. Annotation analysis
    camera_stats = analyze_annotations_per_camera(dataset, args)

    # 2. Sample per camera for rendering analysis
    print("\n" + "=" * 70)
    print("RENDERING ANALYSIS PER CAMERA")
    print("=" * 70)

    per_camera_samples = {}
    per_camera_metrics = {}
    all_sample_metrics = []

    for cam_id in dataset.camera_ids:
        cam_dir = os.path.join(args.output_dir, cam_id)
        if args.save_debug_images:
            os.makedirs(cam_dir, exist_ok=True)

        samples_for_cam = []
        valid_frames = set()
        bboxes_for_cam = []

        for frame_idx, annots in dataset.annotations.items():
            for ann in annots:
                ann_cam_id = ann.get('camera_id')
                if ann_cam_id is None:
                    continue
                expected_cam = f"C{ann_cam_id + 1}"
                if expected_cam != cam_id:
                    continue

                bbox_dict = ann.get('bbox', {})
                if not isinstance(bbox_dict, dict):
                    continue
                x1 = bbox_dict.get('xmin', 0)
                y1 = bbox_dict.get('ymin', 0)
                x2 = bbox_dict.get('xmax', 0)
                y2 = bbox_dict.get('ymax', 0)
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) * (y2 - y1) < args.min_bbox_area:
                    continue

                valid_frames.add(frame_idx)
                bboxes_for_cam.append({
                    'frame_idx': frame_idx,
                    'bbox': [x1, y1, x2, y2],
                    'person_id': ann.get('new_id', -1),
                })

        if not bboxes_for_cam:
            print(f"\n{cam_id}: No valid bboxes found")
            per_camera_metrics[cam_id] = {'verdict': 'no_pedestrian_or_no_annotation'}
            continue

        sampled = random.sample(bboxes_for_cam, min(args.samples_per_camera, len(bboxes_for_cam)))
        samples_for_cam = sampled
        per_camera_samples[cam_id] = samples_for_cam

        print(f"\n{cam_id}: Sampled {len(samples_for_cam)} bboxes from {len(valid_frames)} valid frames")

        # Render and analyze each sample
        sample_metrics = []
        for i, sample in enumerate(samples_for_cam):
            frame_idx = sample['frame_idx']
            bbox = sample['bbox']
            person_id = sample['person_id']

            gpu_batch = batch_builder.get_batch(cam_id, frame_idx)
            if gpu_batch is None:
                continue

            save_paths = None
            if args.save_debug_images and i < args.debug_images_per_camera:
                save_paths = {
                    'alpha': os.path.join(cam_dir, f"sample_{i:03d}_alpha.png"),
                    'overlay': os.path.join(cam_dir, f"sample_{i:03d}_overlay.png"),
                    'bbox_alpha_crop': os.path.join(cam_dir, f"sample_{i:03d}_bbox_alpha_crop.png"),
                }

                original_image_path = dataset.image_paths[cam_id][frame_idx]
                if os.path.exists(original_image_path):
                    save_paths['image'] = original_image_path

                    x1_c, y1_c, x2_c, y2_c = bbox
                    image = cv2.imread(original_image_path)
                    if image is not None:
                        h_img, w_img = image.shape[:2]
                        x1_c = max(0, min(x1_c, w_img - 1))
                        y1_c = max(0, min(y1_c, h_img - 1))
                        x2_c = max(x1_c + 1, min(x2_c, w_img))
                        y2_c = max(y1_c + 1, min(y2_c, h_img))
                        crop = image[y1_c:y2_c, x1_c:x2_c].copy()
                        if crop.size > 0:
                            crop_path = os.path.join(cam_dir, f"sample_{i:03d}_crop.png")
                            cv2.imwrite(crop_path, crop)

                            bbox_image = image.copy()
                            cv2.rectangle(bbox_image, (x1_c, y1_c), (x2_c, y2_c), (0, 255, 0), 2)
                            bbox_path = os.path.join(cam_dir, f"sample_{i:03d}_original_bbox.png")
                            cv2.imwrite(bbox_path, bbox_image)

                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            crop_quality = compute_crop_quality(crop_rgb)
                        else:
                            crop_quality = {'bbox_area': 0, 'bbox_width': 0, 'bbox_height': 0, 'crop_brightness_mean': 0, 'crop_contrast_std': 0, 'crop_blur_laplacian_var': 0, 'crop_aspect_ratio': 0}
                    else:
                        crop_quality = {}
                else:
                    crop_quality = {}
            else:
                crop_quality = {}

            alpha_np, alpha_quality = render_alpha_and_analyze(
                model, gpu_batch, bbox, cam_id, frame_idx, args, device, save_paths
            )

            metric = {
                'camera': cam_id,
                'frame_idx': frame_idx,
                'person_id': person_id,
                'bbox': bbox,
                'crop_quality': crop_quality,
                'alpha_quality': alpha_quality,
            }
            sample_metrics.append(metric)
            all_sample_metrics.append(metric)

        if not sample_metrics:
            per_camera_metrics[cam_id] = {'verdict': 'no_valid_samples'}
            continue

        full_alpha_positive_ratio = sum(1 for m in sample_metrics if m.get('alpha_quality', {}).get('full_alpha_positive', False)) / len(sample_metrics)
        bbox_alpha_positive_ratio = sum(1 for m in sample_metrics if m.get('alpha_quality', {}).get('bbox_alpha_positive', False)) / len(sample_metrics)
        bbox_alpha_ratio_mean = float(np.mean([m.get('alpha_quality', {}).get('bbox_alpha_ratio', 0) for m in sample_metrics]))
        top_alpha_inside_ratio_mean = float(np.mean([m.get('alpha_quality', {}).get('top_alpha_inside_bbox_ratio', 0) for m in sample_metrics]))
        alpha_center_dist_mean = float(np.mean([m.get('alpha_quality', {}).get('alpha_center_distance_to_bbox_center', 0) for m in sample_metrics]))

        crop_blur_mean = float(np.mean([m.get('crop_quality', {}).get('crop_blur_laplacian_var', 0) for m in sample_metrics]))
        crop_brightness_mean = float(np.mean([m.get('crop_quality', {}).get('crop_brightness_mean', 0) for m in sample_metrics]))
        crop_contrast_mean = float(np.mean([m.get('crop_quality', {}).get('crop_contrast_std', 0) for m in sample_metrics]))

        verdict = determine_verdict(cam_id, camera_stats[cam_id], sample_metrics, {}, args)

        per_camera_metrics[cam_id] = {
            'annotation_count': camera_stats[cam_id]['annotation_count'],
            'valid_frame_annotation_count': camera_stats[cam_id]['valid_frame_annotation_count'],
            'unique_person_count': camera_stats[cam_id]['unique_person_count'],
            'bbox_area_mean': camera_stats[cam_id]['bbox_area_mean'],
            'crop_blur_mean': crop_blur_mean,
            'crop_brightness_mean': crop_brightness_mean,
            'crop_contrast_mean': crop_contrast_mean,
            'full_alpha_positive_ratio': full_alpha_positive_ratio,
            'bbox_alpha_positive_ratio': bbox_alpha_positive_ratio,
            'bbox_alpha_ratio_mean': bbox_alpha_ratio_mean,
            'top_alpha_inside_bbox_ratio_mean': top_alpha_inside_ratio_mean,
            'alpha_center_distance_mean': alpha_center_dist_mean,
            'teacher_gap': None,
            'verdict': verdict,
        }

        print(f"\n{cam_id} verdict: {verdict}")
        print(f"  full_alpha_positive: {full_alpha_positive_ratio:.2%}")
        print(f"  bbox_alpha_positive: {bbox_alpha_positive_ratio:.2%}")
        print(f"  bbox_alpha_ratio_mean: {bbox_alpha_ratio_mean:.4f}")
        print(f"  top_alpha_inside_ratio_mean: {top_alpha_inside_ratio_mean:.2%}")
        print(f"  alpha_center_distance_mean: {alpha_center_dist_mean:.2f}")

    # 3. Teacher feature analysis (if fixed_eval_samples available)
    fixed_eval_path = os.path.join(args.output_dir, '..', 'phase12c_gaussianset_clean_random_fixed_eval_lr1e3_v2', 'fixed_eval_samples.json')
    teacher_metrics = {}
    if os.path.exists(fixed_eval_path):
        with open(fixed_eval_path) as f:
            fixed_eval_samples = json.load(f)
        teacher_metrics = analyze_teacher_features(dataset, fixed_eval_samples, args)
    else:
        print("\nNo fixed_eval_samples.json found, skipping teacher feature analysis.")

    # 4. Save outputs
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    summary = {
        'global': {
            'num_cameras': len(dataset.camera_ids),
            'samples_per_camera': args.samples_per_camera,
            'valid_frames_only': args.valid_frames_only,
        },
        'per_camera': per_camera_metrics,
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {args.output_dir}/summary.json")

    with open(os.path.join(args.output_dir, 'per_sample_metrics.jsonl'), 'w') as f:
        for m in all_sample_metrics:
            f.write(json.dumps(m, default=str) + "\n")
    print(f"  Saved: {args.output_dir}/per_sample_metrics.jsonl")

    with open(os.path.join(args.output_dir, 'per_camera_metrics.json'), 'w') as f:
        json.dump(per_camera_metrics, f, indent=2, default=str)
    print(f"  Saved: {args.output_dir}/per_camera_metrics.json")

    generate_final_report(args.output_dir, camera_stats, per_camera_metrics, teacher_metrics, args)
    print(f"  Saved: {args.output_dir}/final_report.md")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
