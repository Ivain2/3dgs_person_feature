#!/usr/bin/env python3
"""
Phase12 Geometry / Pooling Watershed Diagnostic — Stage 0

Goal: Visualize selected Gaussians using existing geometry/checkpoint to determine
whether the failure is from:
A. bbox has person Gaussians but pooling/selection picks wrong ones
B. bbox has no sufficient person Gaussian support
C. selected Gaussians are on-body but ReID loss/feature learning still fails

Uses existing Phase12 checkpoint and the same Gaussian-Set pooling path.
Covers C1-C7, uses medium_eval_allcam.json.
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
                    'idx': i,
                    'x': x_img[i].item(),
                    'y': y_img[i].item(),
                    'depth': depth[i].item(),
                    'opacity': opacity[i].item(),
                    'x_world': xyz[i, 0].item(),
                    'y_world': xyz[i, 1].item(),
                    'z_world': xyz[i, 2].item(),
                })

        return {
            'projected': projected,
            'num_projected': len(projected),
            'num_total': N,
            'image_h': h_img,
            'image_w': w_img,
        }
    except Exception as e:
        return {'projected': [], 'num_projected': 0, 'error': str(e)}


def determine_overlay_verdict(debug_info, all_proj_info, bbox, img):
    """Semi-automatic overlay verdict determination."""
    x1, y1, x2, y2 = bbox
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2
    bbox_area = (x2 - x1) * (y2 - y1)

    num_inside = debug_info.get('num_gaussians_in_bbox', 0)
    weight_sum = debug_info.get('weight_sum', 0.0)

    if num_inside == 0:
        # Check if any Gaussians are projected at all
        if all_proj_info and all_proj_info.get('num_projected', 0) == 0:
            return 'empty'
        else:
            # Some Gaussians projected but none in bbox
            # Check if any projected Gaussians are near bbox
            if all_proj_info:
                near_count = 0
                margin = 50
                for g in all_proj_info.get('projected', []):
                    if (x1 - margin <= g['x'] < x2 + margin and 
                        y1 - margin <= g['y'] < y2 + margin):
                        near_count += 1
                if near_count > 0:
                    return 'misaligned'
                else:
                    return 'empty'
            return 'empty'

    # Some Gaussians in bbox - check weight distribution
    weight_mean = debug_info.get('weight_mean', 0.0)
    depth_mean = debug_info.get('depth_mean', 0.0)
    
    # Heuristic: if many Gaussians in bbox with reasonable opacity and depth
    if num_inside >= 5 and weight_sum > 0.1:
        return 'on_body'
    elif num_inside >= 2 and weight_sum > 0.01:
        return 'partial_body'
    else:
        return 'background'


def draw_gaussian_overlay(image, gaussians, bbox, mode='all'):
    """Draw Gaussian projection points as overlay on image."""
    overlay = image.copy()
    x1, y1, x2, y2 = bbox

    if not gaussians:
        return overlay

    # Sort by opacity for coloring
    sorted_gs = sorted(gaussians, key=lambda g: g.get('opacity', 0), reverse=True)
    
    # Take top-k for visualization
    top_k = min(len(sorted_gs), 500 if mode == 'all' else 100)
    visible_gs = sorted_gs[:top_k]

    if not visible_gs:
        return overlay

    opacities = [g.get('opacity', 0) for g in visible_gs]
    max_op = max(opacities) if opacities else 1.0
    min_op = min(opacities) if opacities else 0.0
    op_range = max_op - min_op if max_op != min_op else 1.0

    for g in visible_gs:
        px, py = int(g['x']), int(g['y'])
        opacity_norm = (g.get('opacity', 0) - min_op) / op_range
        
        # Color by opacity: blue (low) -> red (high)
        r = int(255 * opacity_norm)
        b = int(255 * (1 - opacity_norm))
        color = (b, 0, r)  # BGR
        
        # Draw point
        cv2.circle(overlay, (px, py), 3, color, -1)

    # Draw bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return overlay


def draw_bbox_crop_overlay(image, gaussians, bbox, mode='crop'):
    """Draw Gaussian overlay on bbox crop only."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    x1_c = max(0, min(x1, w - 1))
    y1_c = max(0, min(y1, h - 1))
    x2_c = max(0, min(x2, w))
    y2_c = max(0, min(y2, h))
    
    if x2_c <= x1_c or y2_c <= y1_c:
        return image[y1_c:y2_c, x1_c:x2_c].copy()
    
    crop = image[y1_c:y2_c, x1_c:x2_c].copy()
    
    # Gaussians relative to crop
    for g in gaussians:
        px, py = int(g['x']) - x1_c, int(g['y']) - y1_c
        if 0 <= px < crop.shape[1] and 0 <= py < crop.shape[0]:
            opacity_norm = g.get('opacity', 0)
            if opacity_norm > 0.1:
                r = int(255 * opacity_norm)
                b = int(255 * (1 - opacity_norm))
                cv2.circle(crop, (px, py), 2, (b, 0, r), -1)
    
    return crop


def run_diagnostic(args):
    print("\n" + "="*80)
    print("Phase12 Geometry / Pooling Watershed Diagnostic — Stage 0")
    print("="*80)

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
    print(f"Positions shape: {model.positions.shape}")
    print(f"Density shape: {model.get_density().shape}")
    print(f"Person feature shape: {model.get_person_feature().shape}")

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
        
        loaded_keys = list(state.keys())
        print(f"Checkpoint keys: {loaded_keys[:10]}...")
        
        # Load _person_feature if available
        if '_person_feature' in state:
            model._person_feature.data = state['_person_feature'].to(device)
            print("Loaded _person_feature from checkpoint")
        
        # Try to load geometry keys if present
        geo_keys = ['positions', '_positions', 'means3d', '_means3d', 'scaling', 'rotation', 'opacity']
        for k in geo_keys:
            if k in state:
                target_key = k.lstrip('_')
                if hasattr(model, target_key):
                    getattr(model, target_key).data = state[k].to(device)
                    print(f"Loaded {target_key} from checkpoint")

    batch_builder = BatchBuilder(dataset)

    # Load eval samples
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
    sample_idx = 0

    for cam_id in all_cameras:
        cam_samples = samples_by_cam.get(cam_id, [])
        if not cam_samples:
            print(f"\n{cam_id}: No samples")
            per_camera_stats[cam_id] = {
                'camera': cam_id,
                'num_samples_diagnosed': 0,
                'valid_sample_count': 0,
                'valid_sample_ratio': 0.0,
                'mean_inside_bbox_gaussians': 0.0,
                'mean_selected_gaussians': 0.0,
                'mean_selected_weight_sum': 0.0,
                'mean_selected_feature_norm': 0.0,
                'overlay_verdict_counts': {},
                'mean_center_distance_to_bbox': 0.0,
                'final_camera_verdict': 'no_samples',
            }
            continue

        # Sample up to args.samples_per_camera per camera
        if len(cam_samples) > args.samples_per_camera:
            sampled = random.sample(cam_samples, args.samples_per_camera)
        else:
            sampled = cam_samples

        print(f"\n{'='*60}")
        print(f"{cam_id}: {len(cam_samples)} available -> {len(sampled)} sampled")
        print(f"{'='*60}")

        cam_dir = os.path.join(args.output_dir, cam_id)
        os.makedirs(cam_dir, exist_ok=True)

        verdict_counts = defaultdict(int)
        valid_count = 0
        inside_bbox_counts = []
        selected_counts = []
        weight_sums = []
        feature_norms = []
        center_distances = []

        for s_idx, sample in enumerate(sampled):
            person_id = sample.get('person_id', 'unknown')
            frame_id = sample.get('frame_id', sample.get('frame_idx', 'unknown'))
            bbox = sample.get('bbox', [0, 0, 0, 0])
            
            if isinstance(frame_id, str):
                frame_id = int(frame_id)
            
            sample_id = f"{cam_id}_frame{frame_id:06d}_pid{person_id:03d}_{s_idx:03d}"

            # Get GPU batch
            gpu_batch = batch_builder.get_batch(cam_id, frame_id)
            if gpu_batch is None:
                record = {
                    'sample_id': sample_id, 'cam_id': cam_id, 'frame_id': frame_id,
                    'person_id': person_id, 'bbox': bbox,
                    'valid_sample': False, 'invalid_reason': 'gpu_batch_none',
                }
                per_sample_records.append(record)
                continue

            # Load original image
            image_path = None
            if hasattr(gpu_batch, 'image_path') and gpu_batch.image_path:
                image_path = gpu_batch.image_path
            elif cam_id in dataset.image_paths and frame_id < len(dataset.image_paths[cam_id]):
                image_path = dataset.image_paths[cam_id][frame_id]
            
            if image_path and os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is None:
                    record = {
                        'sample_id': sample_id, 'cam_id': cam_id, 'frame_id': frame_id,
                        'person_id': person_id, 'bbox': bbox,
                        'valid_sample': False, 'invalid_reason': 'image_read_failed',
                    }
                    per_sample_records.append(record)
                    continue
            else:
                # Create a blank image for visualization
                h_img, w_img = gpu_batch.rays_dir.shape[1], gpu_batch.rays_dir.shape[2]
                image = np.zeros((h_img, w_img, 3), dtype=np.uint8)
                record_warning = f"image_not_found, using blank {w_img}x{h_img}"

            # Project all Gaussians
            all_proj = project_all_gaussians(model, gpu_batch, device)
            num_all_projected = all_proj.get('num_projected', 0) if all_proj else 0

            # Run Gaussian-Set pooling
            G, debug_info = gaussian_set_pooling(
                model, gpu_batch, bbox, cam_id, frame_id, args, device
            )

            num_inside_bbox = debug_info.get('num_gaussians_in_bbox', 0)
            weight_sum = debug_info.get('weight_sum', 0.0)
            feature_norm = float(G.norm().item()) if G is not None else 0.0

            # Determine overlay verdict
            overlay_verdict = determine_overlay_verdict(debug_info, all_proj, bbox, image)
            verdict_counts[overlay_verdict] += 1

            if G is not None:
                valid_count += 1
                inside_bbox_counts.append(num_inside_bbox)
                selected_counts.append(num_inside_bbox)
                weight_sums.append(weight_sum)
                feature_norms.append(feature_norm)

                # Calculate center distance
                x1_b, y1_b, x2_b, y2_b = bbox
                bbox_cx = (x1_b + x2_b) / 2
                bbox_cy = (y1_b + y2_b) / 2
                
                # Selected Gaussian center (weighted by opacity)
                if all_proj and all_proj.get('projected'):
                    # Get projected Gaussians inside bbox
                    inside_gs = []
                    for g in all_proj['projected']:
                        gx, gy = g['x'], g['y']
                        if x1_b <= gx < x2_b and y1_b <= gy < y2_b:
                            inside_gs.append(g)
                    
                    if inside_gs:
                        total_w = sum(g['opacity'] for g in inside_gs)
                        if total_w > 0:
                            sel_cx = sum(g['x'] * g['opacity'] for g in inside_gs) / total_w
                            sel_cy = sum(g['y'] * g['opacity'] for g in inside_gs) / total_w
                        else:
                            sel_cx = np.mean([g['x'] for g in inside_gs])
                            sel_cy = np.mean([g['y'] for g in inside_gs])
                    else:
                        sel_cx, sel_cy = bbox_cx, bbox_cy
                else:
                    sel_cx, sel_cy = bbox_cx, bbox_cy
                
                dist = ((sel_cx - bbox_cx)**2 + (sel_cy - bbox_cy)**2)**0.5
                center_distances.append(dist)

            # Save visualizations
            if image is not None and image.size > 0:
                x1_b, y1_b, x2_b, y2_b = bbox
                
                # 1. Original image + bbox
                img_with_bbox = image.copy()
                cv2.rectangle(img_with_bbox, (x1_b, y1_b), (x2_b, y2_b), (0, 255, 0), 2)
                cv2.putText(img_with_bbox, f"PID:{person_id} | GS:{num_inside_bbox}", 
                           (x1_b, max(y1_b - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_original_bbox.png"), img_with_bbox)

                # 2. All projected Gaussian overlay
                if all_proj and all_proj.get('projected'):
                    all_overlay = draw_gaussian_overlay(image, all_proj['projected'], bbox, mode='all')
                    cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_all_projected.png"), all_overlay)

                # 3. Selected Gaussian overlay (Gaussians inside bbox)
                if G is not None and all_proj and all_proj.get('projected'):
                    selected_gs = []
                    for g in all_proj['projected']:
                        gx, gy = g['x'], g['y']
                        if x1_b <= gx < x2_b and y1_b <= gy < y2_b:
                            selected_gs.append(g)
                    
                    if selected_gs:
                        sel_overlay = draw_gaussian_overlay(image, selected_gs, bbox, mode='selected')
                        cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_selected_overlay.png"), sel_overlay)

                        # 4. Selected Gaussian colored by weight/opacity
                        sel_weighted = draw_gaussian_overlay(image, selected_gs, bbox, mode='weighted')
                        cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_selected_by_weight.png"), sel_weighted)

                        # 5. Bbox crop overlay
                        crop_overlay = draw_bbox_crop_overlay(image, selected_gs, bbox)
                        cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_bbox_crop_overlay.png"), crop_overlay)

                        # 6. Top-k selected Gaussian overlay
                        top_gs = sorted(selected_gs, key=lambda g: g.get('opacity', 0), reverse=True)[:20]
                        top_overlay = draw_gaussian_overlay(image, top_gs, bbox, mode='top_k')
                        cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_top20_overlay.png"), top_overlay)

            # Build record
            record = {
                'sample_id': sample_id,
                'cam_id': cam_id,
                'frame_id': frame_id,
                'person_id': int(person_id),
                'bbox': bbox,
                'num_all_projected_gaussians': num_all_projected,
                'num_inside_bbox_gaussians': num_inside_bbox,
                'num_selected_gaussians': num_inside_bbox,
                'selected_weight_sum': weight_sum,
                'selected_weight_mean': debug_info.get('weight_mean', 0.0),
                'selected_feature_norm': feature_norm,
                'pooled_feature_norm': feature_norm,
                'overlay_verdict': overlay_verdict,
                'valid_sample': G is not None,
                'invalid_reason': debug_info.get('failure_reason', None) if G is None else None,
            }
            
            if center_distances:
                record['center_distance_to_bbox'] = center_distances[-1]
            
            per_sample_records.append(record)

            if s_idx < 5 or s_idx % 20 == 0:
                print(f"  [{s_idx}/{len(sampled)}] {sample_id}: "
                      f"proj={num_all_projected}, inside={num_inside_bbox}, "
                      f"w_sum={weight_sum:.4f}, verdict={overlay_verdict}")

        # Per-camera summary
        total_diagnosed = len(sampled)
        valid_ratio = valid_count / max(1, total_diagnosed)
        
        per_camera_stats[cam_id] = {
            'camera': cam_id,
            'num_samples_diagnosed': total_diagnosed,
            'valid_sample_count': valid_count,
            'valid_sample_ratio': valid_ratio,
            'mean_inside_bbox_gaussians': float(np.mean(inside_bbox_counts)) if inside_bbox_counts else 0.0,
            'mean_selected_gaussians': float(np.mean(selected_counts)) if selected_counts else 0.0,
            'mean_selected_weight_sum': float(np.mean(weight_sums)) if weight_sums else 0.0,
            'mean_selected_feature_norm': float(np.mean(feature_norms)) if feature_norms else 0.0,
            'overlay_verdict_counts': dict(verdict_counts),
            'overlay_on_body_ratio': verdict_counts.get('on_body', 0) / max(1, total_diagnosed),
            'overlay_partial_body_ratio': verdict_counts.get('partial_body', 0) / max(1, total_diagnosed),
            'overlay_background_ratio': verdict_counts.get('background', 0) / max(1, total_diagnosed),
            'overlay_empty_ratio': verdict_counts.get('empty', 0) / max(1, total_diagnosed),
            'overlay_misaligned_ratio': verdict_counts.get('misaligned', 0) / max(1, total_diagnosed),
            'mean_center_distance_to_bbox': float(np.mean(center_distances)) if center_distances else 0.0,
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
        print(f"    Mean inside bbox: {stats['mean_inside_bbox_gaussians']:.1f}")
        print(f"    Mean weight sum: {stats['mean_selected_weight_sum']:.4f}")
        print(f"    Verdicts: {dict(verdict_counts)}")
        print(f"    Final verdict: {stats['final_camera_verdict']}")

    # Save outputs
    with open(os.path.join(args.output_dir, 'per_camera_support_summary.json'), 'w') as f:
        json.dump(per_camera_stats, f, indent=2, default=str)

    with open(os.path.join(args.output_dir, 'bbox_gaussian_support_metrics.jsonl'), 'w') as f:
        for record in per_sample_records:
            f.write(json.dumps(record, default=str) + '\n')

    # Generate overlay summary
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
    generate_final_report(per_camera_stats, overlay_summary, per_sample_records, args.output_dir)

    print(f"\n{'='*80}")
    print(f"Stage 0 Diagnostic complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


def generate_final_report(per_camera_stats, overlay_summary, per_sample_records, output_dir):
    """Generate final_geometry_report.md."""
    
    report = """# Phase12 Geometry / Pooling Watershed Diagnostic — Stage 0 Final Report

## 1. 当前已有 geometry 的 selected Gaussian 是否落在人身上？

"""
    
    on_body_total = overlay_summary.get('global_verdict_counts', {}).get('on_body', 0)
    partial_body_total = overlay_summary.get('global_verdict_counts', {}).get('partial_body', 0)
    background_total = overlay_summary.get('global_verdict_counts', {}).get('background', 0)
    empty_total = overlay_summary.get('global_verdict_counts', {}).get('empty', 0)
    misaligned_total = overlay_summary.get('global_verdict_counts', {}).get('misaligned', 0)
    total_diagnosed = overlay_summary.get('total_samples_diagnosed', 0)
    
    report += f"| Verdict | Count | Ratio |\n"
    report += f"|---------|-------|-------|\n"
    report += f"| on_body | {on_body_total} | {on_body_total/max(1,total_diagnosed):.2%} |\n"
    report += f"| partial_body | {partial_body_total} | {partial_body_total/max(1,total_diagnosed):.2%} |\n"
    report += f"| background | {background_total} | {background_total/max(1,total_diagnosed):.2%} |\n"
    report += f"| empty | {empty_total} | {empty_total/max(1,total_diagnosed):.2%} |\n"
    report += f"| misaligned | {misaligned_total} | {misaligned_total/max(1,total_diagnosed):.2%} |\n"
    
    on_body_plus_partial = on_body_total + partial_body_total
    report += f"\n**Combined on_body + partial_body**: {on_body_plus_partial}/{total_diagnosed} ({on_body_plus_partial/max(1,total_diagnosed):.2%})\n"

    report += """
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

| Camera | Valid Ratio | Mean Inside Gaussians | Mean Weight Sum | Final Verdict |
|--------|------------|---------------------|-----------------|---------------|
"""
    
    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = per_camera_stats.get(cam_id, {})
        report += (
            f"| {cam_id} | {stats.get('valid_sample_ratio', 0):.3f} | "
            f"{stats.get('mean_inside_bbox_gaussians', 0):.1f} | "
            f"{stats.get('mean_selected_weight_sum', 0):.4f} | "
            f"{stats.get('final_camera_verdict', 'N/A')} |\n"
        )

    report += """
## 4. C2/C3/C5 是否仍存在 full opacity 或 bbox opacity 异常？

"""
    
    for cam_id in ['C2', 'C3', 'C5']:
        stats = per_camera_stats.get(cam_id, {})
        report += f"### {cam_id}\n\n"
        report += f"- Valid sample ratio: {stats.get('valid_sample_ratio', 0):.3f}\n"
        report += f"- Mean inside bbox Gaussians: {stats.get('mean_inside_bbox_gaussians', 0):.1f}\n"
        report += f"- Mean weight sum: {stats.get('mean_selected_weight_sum', 0):.4f}\n"
        report += f"- Overlay verdicts: {stats.get('overlay_verdict_counts', {})}\n"
        report += f"- Final verdict: {stats.get('final_camera_verdict', 'N/A')}\n\n"

    report += """## 5. bbox 内是否有足够 projected / selected Gaussian？

| Camera | Mean Projected | Mean Inside BBox | Mean Selected | Weight Sum |
|--------|---------------|-----------------|---------------|------------|
"""
    
    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = per_camera_stats.get(cam_id, {})
        report += (
            f"| {cam_id} | {stats.get('mean_inside_bbox_gaussians', 0):.1f} | "
            f"{stats.get('mean_inside_bbox_gaussians', 0):.1f} | "
            f"{stats.get('mean_selected_gaussians', 0):.1f} | "
            f"{stats.get('mean_selected_weight_sum', 0):.4f} |\n"
        )

    report += """
## 6. 当前瓶颈更像 geometry、camera alignment、bbox mapping、还是 pooling selection？

"""
    
    # Determine primary bottleneck
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
        
        # Categorize issues
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
        
        # Primary bottleneck
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
    
    # Decision logic
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
        report += "- 需要先进入 Stage 1/2/3 修复 geometry\n"
        report += "- 或者改进 pooling/selection 机制\n"

    report += """
---

*报告生成时间: Stage 0 Diagnostic*
*脚本: tools/phase12_geometry_pooling_watershed_diagnostic.py*
"""

    with open(os.path.join(output_dir, 'final_geometry_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Phase12 Geometry/Pooling Watershed Diagnostic — Stage 0')
    
    parser.add_argument('--output_dir', type=str,
                        default='outputs/phase12_existing_geometry_overlay_diagnostic')
    parser.add_argument('--eval_samples', type=str, default=None,
                        help='Path to eval samples JSON (default: medium_eval_allcam.json)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to Phase12 checkpoint')
    parser.add_argument('--samples_per_camera', type=int, default=50,
                        help='Max samples to diagnose per camera')
    parser.add_argument('--min_bbox_area', type=int, default=100)
    parser.add_argument('--denom_eps', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    run_diagnostic(args)


if __name__ == '__main__':
    main()
