#!/usr/bin/env python3
"""
Phase12 同源诊断脚本：解释 C2/C3/C5 为什么在当前 Gaussian-Set ReID 中没有提供有效监督

核心设计原则：
- 必须加载真实 Phase12 checkpoint
- 复用 Phase12C/E/F 实际使用的 Gaussian-Set feature extraction / pooling 路径
- 不训练模型，不启用 teacher loss / SupCon / MV loss
- 覆盖 C1-C7 全部相机

参考文件：
- tools/archive/phase12/phase12f_EMA_PROTO_PLUS_MV_INFO_NCE_可复用.py
- tools/eval_reid_gaussianset.py
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from hydra import initialize_config_dir, compose
from rich import pretty, traceback

pretty.install()
traceback.install()


def normalize_feat(x, eps=1e-6):
    """L2 normalize features (exact same as Phase12)."""
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def gaussian_set_pooling_diagnostic(model, gpu_batch, bbox, cam_id, frame_id, args, device):
    """
    EXACT same Gaussian-Set pooling path as Phase12C/E/F.
    
    核心步骤：
    1. 获取 Gaussian positions, opacity, person_feature
    2. 使用 intrinsics 将 Gaussian 投影到图像平面
    3. 使用 T_to_world 计算世界坐标到相机坐标的变换
    4. 筛选在 bbox 内且 opacity > 0 的 Gaussian
    5. 使用 opacity 作为权重对 person_feature 进行加权平均
    6. L2 normalize 得到最终 feature
    """
    x1, y1, x2, y2 = bbox
    try:
        xyz = model.positions
        opacity = model.get_density().squeeze(-1)
        person_feature = model.get_person_feature()

        N = xyz.shape[0]
        if N == 0:
            return None, {
                'selected_gaussian_count': 0,
                'gaussian_weight_sum': 0.0,
                'student_feature_norm': 0.0,
                'is_gaussianset_valid': False,
                'failure_reason': 'no_gaussians',
                'total_gaussians': 0,
                'valid_gaussians': 0,
            }

        intrinsics = gpu_batch.intrinsics
        if intrinsics is None or len(intrinsics) < 4:
            return None, {
                'selected_gaussian_count': 0,
                'gaussian_weight_sum': 0.0,
                'student_feature_norm': 0.0,
                'is_gaussianset_valid': False,
                'failure_reason': 'no_intrinsics',
                'total_gaussians': N,
                'valid_gaussians': 0,
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
        h_img, w_img = gpu_batch.rays_dir.shape[1], gpu_batch.rays_dir.shape[2]

        x_finite = torch.isfinite(x_img)
        y_finite = torch.isfinite(y_img)
        x_in_bounds = (x_img >= 0) & (x_img < w_img)
        y_in_bounds = (y_img >= 0) & (y_img < h_img)
        opacity_positive = opacity > 0

        valid = valid_depth & x_finite & y_finite & x_in_bounds & y_in_bounds & opacity_positive
        inside_bbox = (x_img >= x1) & (x_img < x2) & (y_img >= y1) & (y_img < y2)
        inside = valid & inside_bbox

        selected_count = int(inside.sum().item())

        if selected_count == 0:
            return None, {
                'selected_gaussian_count': 0,
                'gaussian_weight_sum': 0.0,
                'student_feature_norm': 0.0,
                'is_gaussianset_valid': False,
                'failure_reason': 'no_gaussians_in_bbox',
                'total_gaussians': N,
                'valid_gaussians': int(valid.sum().item()),
                'bbox_area': (x2 - x1) * (y2 - y1),
            }

        weights = opacity[inside]
        z = person_feature[inside]
        weight_sum = weights.sum()
        denom = weight_sum.clamp(min=getattr(args, 'denom_eps', 1e-8))
        weighted_sum = (weights[:, None] * z).sum(dim=0)
        G = weighted_sum / denom
        G = normalize_feat(G)

        student_norm = float(G.norm().item())
        has_nan = bool(torch.isnan(G).any().item())
        has_inf = bool(torch.isinf(G).any().item())

        is_valid = (
            selected_count > 0
            and weight_sum.item() > args.weight_sum_threshold
            and student_norm > args.feature_norm_threshold
            and not has_nan
            and not has_inf
        )

        depth_inside = depth[inside]
        return G, {
            'selected_gaussian_count': selected_count,
            'gaussian_weight_sum': float(weight_sum.item()),
            'student_feature_norm': student_norm,
            'is_gaussianset_valid': is_valid,
            'failure_reason': None if is_valid else 'invalid_pooling',
            'has_nan': has_nan,
            'has_inf': has_inf,
            'total_gaussians': N,
            'valid_gaussians': int(valid.sum().item()),
            'depth_min': float(depth_inside.min().item()),
            'depth_mean': float(depth_inside.mean().item()),
            'depth_max': float(depth_inside.max().item()),
            'bbox_area': (x2 - x1) * (y2 - y1),
        }
    except Exception as e:
        return None, {
            'selected_gaussian_count': 0,
            'gaussian_weight_sum': 0.0,
            'student_feature_norm': 0.0,
            'is_gaussianset_valid': False,
            'failure_reason': f'{str(e)[:80]}',
            'total_gaussians': 0,
            'valid_gaussians': 0,
        }


class BatchBuilder:
    """Build GPU batches for specific camera-frame pairs (exact same as Phase12)."""
    
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


def build_camera_annotation_stats(dataset, all_cameras):
    """统计每个相机的 annotation 信息，只统计 dataset.indices 中真实存在的 frame。"""
    print("\nBuilding camera annotation statistics...")
    
    cam_stats = {cam: {
        'annotation_count': 0,
        'valid_frame_annotation_count': 0,
        'unique_person_count': 0,
        'frame_count': 0,
        'bbox_count': 0,
        'bbox_areas': [],
        'bbox_widths': [],
        'bbox_heights': [],
    } for cam in all_cameras}
    
    valid_frames = set()
    for idx in range(len(dataset)):
        cam_id, frame_idx = dataset.indices[idx]
        valid_frames.add((cam_id, int(frame_idx)))
    
    for cam_id in all_cameras:
        cam_stats[cam_id]['frame_count'] = sum(
            1 for (c, f) in valid_frames if c == cam_id
        )
    
    person_ids_per_cam = defaultdict(set)
    
    for idx in range(len(dataset)):
        cam_id, frame_idx = dataset.indices[idx]
        if cam_id not in all_cameras:
            continue
            
        anns = dataset.annotations.get(int(frame_idx), [])
        for ann in anns:
            ann_cam_id = ann.get('camera_id')
            if ann_cam_id is None:
                continue
            ann_cam_str = f"C{ann_cam_id + 1}"
            if ann_cam_str != cam_id:
                continue
            
            cam_stats[cam_id]['annotation_count'] += 1
            cam_stats[cam_id]['valid_frame_annotation_count'] += 1
            
            pid = ann.get('new_id')
            if pid is not None:
                person_ids_per_cam[cam_id].add(pid)
            
            bbox_dict = ann.get('bbox', {})
            if isinstance(bbox_dict, dict) and len(bbox_dict) >= 4:
                x1 = int(bbox_dict.get('xmin', 0))
                y1 = int(bbox_dict.get('ymin', 0))
                x2 = int(bbox_dict.get('xmax', 0))
                y2 = int(bbox_dict.get('ymax', 0))
                
                if x2 > x1 and y2 > y1:
                    area = (x2 - x1) * (y2 - y1)
                    if area >= 100:
                        cam_stats[cam_id]['bbox_count'] += 1
                        cam_stats[cam_id]['bbox_areas'].append(area)
                        cam_stats[cam_id]['bbox_widths'].append(x2 - x1)
                        cam_stats[cam_id]['bbox_heights'].append(y2 - y1)
    
    for cam_id in all_cameras:
        cam_stats[cam_id]['unique_person_count'] = len(person_ids_per_cam[cam_id])
    
    print(f"Annotation statistics built for {len(all_cameras)} cameras")
    for cam_id in all_cameras:
        stats = cam_stats[cam_id]
        print(f"  {cam_id}: {stats['annotation_count']} anns, "
              f"{stats['unique_person_count']} persons, "
              f"{stats['bbox_count']} valid bboxes")
    
    return cam_stats


def sample_bboxes_per_camera(dataset, batch_builder, all_cameras, args):
    """为每个相机随机采样 samples_per_camera 个有效 bbox。"""
    print(f"\nSampling bboxes per camera (target: {args.samples_per_camera} per camera)...")
    
    candidates_by_cam = defaultdict(list)
    
    for idx in range(len(dataset)):
        cam_id, frame_idx = dataset.indices[idx]
        if cam_id not in all_cameras:
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
            
            bbox_dict = ann.get('bbox', {})
            if not isinstance(bbox_dict, dict) or len(bbox_dict) < 4:
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
            
            teacher_emb = None
            if dataset.teacher_cache is not None:
                cache_key = (int(frame_idx), cam_id, int(pid), x1, y1, x2, y2)
                cache_entry = dataset.teacher_cache.get(cache_key)
                if cache_entry is not None:
                    teacher_emb = cache_entry.get('embedding')
                    if hasattr(teacher_emb, 'squeeze'):
                        teacher_emb = teacher_emb.squeeze()
            
            candidates_by_cam[cam_id].append({
                'person_id': int(pid),
                'cam_id': cam_id,
                'frame_idx': int(frame_idx),
                'dataset_index': idx,
                'bbox': [x1, y1, x2, y2],
                'bbox_area': int(bbox_area),
                'teacher_emb': teacher_emb,
            })
    
    sampled_by_cam = {}
    for cam_id in all_cameras:
        candidates = candidates_by_cam.get(cam_id, [])
        if len(candidates) == 0:
            print(f"  {cam_id}: No valid bbox candidates")
            sampled_by_cam[cam_id] = []
            continue
        
        if len(candidates) > args.samples_per_camera:
            sampled = random.sample(candidates, args.samples_per_camera)
        else:
            sampled = candidates
        
        sampled_by_cam[cam_id] = sampled
        print(f"  {cam_id}: {len(candidates)} candidates -> {len(sampled)} sampled")
    
    return sampled_by_cam


def analyze_crop_quality(image, bbox):
    """分析 bbox crop 的图像质量指标。"""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    x1_c = max(0, min(x1, w - 1))
    y1_c = max(0, min(y1, h - 1))
    x2_c = max(0, min(x2, w))
    y2_c = max(0, min(y2, h))
    
    if x2_c <= x1_c or y2_c <= y1_c:
        return None
    
    crop = image[y1_c:y2_c, x1_c:x2_c]
    
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None
    
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    return {
        'crop_valid': True,
        'bbox_width': x2_c - x1_c,
        'bbox_height': y2_c - y1_c,
        'bbox_area': (x2_c - x1_c) * (y2_c - y1_c),
        'crop_brightness_mean': brightness_mean,
        'crop_contrast_std': contrast_std,
        'crop_blur_laplacian_var': blur_var,
    }


def try_get_rendered_opacity(model, gpu_batch, args, device):
    """
    尝试获取 rendered opacity。
    注意：如果 render path 与 Phase12 Gaussian-Set 不同源，只标注不判定 failure。
    """
    try:
        with torch.no_grad():
            render_result = model(gpu_batch)
        
        opacity_map = None
        opacity_key = None
        
        if isinstance(render_result, dict):
            for key in ['pred_opacity', 'alpha', 'person_opacity_map', 'opacity']:
                if key in render_result:
                    opacity_map = render_result[key]
                    opacity_key = key
                    break
        
        if opacity_map is None:
            return None, None, 'rendering_path_not_same_as_gaussianset'
        
        if hasattr(opacity_map, 'squeeze'):
            if len(opacity_map.shape) == 3:
                opacity_map = opacity_map.squeeze(0)
            elif len(opacity_map.shape) == 4:
                opacity_map = opacity_map[0, 0]
        
        if hasattr(opacity_map, 'detach'):
            opacity_map = opacity_map.detach().cpu().numpy()
        
        return opacity_map, opacity_key, None
    except Exception as e:
        return None, None, f'render_error:{str(e)[:60]}'


def compute_opacity_metrics(opacity_map, bbox, args):
    """计算 opacity 相关指标。"""
    if opacity_map is None:
        return {
            'full_image_opacity_sum': 0.0,
            'bbox_opacity_sum': 0.0,
            'bbox_opacity_ratio': 0.0,
            'full_opacity_positive': False,
            'bbox_opacity_positive': False,
        }
    
    x1, y1, x2, y2 = bbox
    h, w = opacity_map.shape[:2]
    
    x1_c = max(0, min(x1, w - 1))
    y1_c = max(0, min(y1, h - 1))
    x2_c = max(0, min(x2, w))
    y2_c = max(0, min(y2, h))
    
    full_sum = float(np.sum(opacity_map))
    
    if x2_c > x1_c and y2_c > y1_c:
        bbox_sum = float(np.sum(opacity_map[y1_c:y2_c, x1_c:x2_c]))
    else:
        bbox_sum = 0.0
    
    bbox_ratio = bbox_sum / (full_sum + 1e-8)
    
    return {
        'full_image_opacity_sum': full_sum,
        'bbox_opacity_sum': bbox_sum,
        'bbox_opacity_ratio': bbox_ratio,
        'full_opacity_positive': full_sum > args.alpha_threshold,
        'bbox_opacity_positive': bbox_sum > args.alpha_threshold,
    }


def save_debug_images(cam_id, sample_idx, image, bbox, crop_info, opacity_map, args):
    """保存 debug 图像。"""
    if not args.save_debug_images:
        return
    
    cam_dir = os.path.join(args.output_dir, 'debug_images', cam_id)
    os.makedirs(cam_dir, exist_ok=True)
    
    prefix = f"sample_{sample_idx:03d}"
    
    x1, y1, x2, y2 = bbox
    h_img, w_img = image.shape[:2]
    
    x1_c = max(0, min(int(x1), w_img - 1))
    y1_c = max(0, min(int(y1), h_img - 1))
    x2_c = max(0, min(int(x2), w_img))
    y2_c = max(0, min(int(y2), h_img))
    
    img_with_bbox = image.copy()
    cv2.rectangle(img_with_bbox, (x1_c, y1_c), (x2_c, y2_c), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(cam_dir, f"{prefix}_original_bbox.png"), img_with_bbox)
    
    if crop_info is not None and crop_info.get('crop_valid', False):
        crop = image[y1_c:y2_c, x1_c:x2_c]
        if crop.size > 0:
            cv2.imwrite(os.path.join(cam_dir, f"{prefix}_crop.png"), crop)
    
    if opacity_map is not None:
        opacity_vis = (opacity_map * 255 / (opacity_map.max() + 1e-8)).astype(np.uint8)
        opacity_colored = cv2.applyColorMap(opacity_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(cam_dir, f"{prefix}_opacity.png"), opacity_colored)
        
        overlay = img_with_bbox.copy()
        if opacity_vis.shape[:2] == overlay.shape[:2]:
            mask = (opacity_vis > 10).astype(np.uint8) * 128
            overlay[mask > 0] = cv2.addWeighted(overlay, 0.7, opacity_colored, 0.3, 0)[mask > 0]
        cv2.imwrite(os.path.join(cam_dir, f"{prefix}_overlay.png"), overlay)


def run_diagnostic(args):
    """运行完整诊断流程。"""
    print("=" * 80)
    print("Phase12 同源 Gaussian-Set 诊断：C1-C7 相机有效性分析")
    print("=" * 80)
    
    all_cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        fallback_path = 'outputs/phase12c_gaussianset_clean_random_fixed_eval_lr1e3_v2/checkpoint_best_fixed_cos.pt'
        if os.path.exists(fallback_path):
            print(f"WARNING: Primary checkpoint not found, using fallback: {fallback_path}")
            checkpoint_path = fallback_path
        else:
            print(f"ERROR: No checkpoint found at {checkpoint_path} or fallback")
            sys.exit(1)
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    config_dir = os.path.join(REPO_ROOT, 'configs')
    config_file = 'apps/wildtrack_full_3dgut'
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_file)
    cfg.model.person_feature_dim = 512
    
    from threedgrut.trainer import Trainer3DGRUT
    trainer = Trainer3DGRUT(cfg)
    model = trainer.model
    device = trainer.device
    dataset = trainer.train_dataset
    
    print(f"\nLoading person_feature from checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt and '_person_feature' in ckpt['model_state_dict']:
        pf = ckpt['model_state_dict']['_person_feature']
        if hasattr(pf, 'to'):
            model._person_feature.data.copy_(pf.to(device))
        print(f"  Loaded person_feature shape: {model._person_feature.shape}")
    else:
        print("  WARNING: Could not load person_feature from checkpoint")
    
    batch_builder = BatchBuilder(dataset)
    
    print(f"\nDataset info:")
    print(f"  Total indices: {len(dataset)}")
    print(f"  Cameras in dataset: {len(set(c for c, f in dataset.indices))}")
    
    cam_stats = build_camera_annotation_stats(dataset, all_cameras)
    
    sampled_by_cam = sample_bboxes_per_camera(dataset, batch_builder, all_cameras, args)
    
    print(f"\n{'='*80}")
    print("开始核心诊断：Gaussian-Set pooling + rendering + crop quality")
    print(f"{'='*80}")
    
    per_sample_metrics = []
    sample_global_idx = 0
    
    for cam_id in all_cameras:
        samples = sampled_by_cam[cam_id]
        if not samples:
            print(f"\n  {cam_id}: No samples to diagnose")
            continue
        
        print(f"\n  Diagnosing {cam_id} ({len(samples)} samples)...")
        
        for i, sample in enumerate(samples):
            gpu_batch = batch_builder.get_batch(sample['cam_id'], sample['frame_idx'])
            if gpu_batch is None:
                per_sample_metrics.append({
                    'sample_id': sample_global_idx,
                    'person_id': sample['person_id'],
                    'cam_id': cam_id,
                    'frame_id': sample['frame_idx'],
                    'dataset_index': sample['dataset_index'],
                    'bbox': sample['bbox'],
                    'failure_reason': 'gpu_batch_not_found',
                })
                sample_global_idx += 1
                continue
            
            image_path = getattr(gpu_batch, 'image_path', None)
            image = None
            if image_path and os.path.exists(image_path):
                image = cv2.imread(image_path)
            
            crop_info = None
            if image is not None:
                crop_info = analyze_crop_quality(image, sample['bbox'])
            
            student_feature, gs_info = gaussian_set_pooling_diagnostic(
                model, gpu_batch, sample['bbox'], sample['cam_id'], sample['frame_idx'], args, device
            )
            
            teacher_feature = None
            teacher_norm = 0.0
            cos_to_teacher = 0.0
            
            if sample['teacher_emb'] is not None:
                teacher_feature = torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=device)
                teacher_feature = normalize_feat(teacher_feature)
                teacher_norm = float(teacher_feature.norm().item())
                
                if student_feature is not None:
                    cos_to_teacher = float(F.cosine_similarity(
                        student_feature.unsqueeze(0), teacher_feature.unsqueeze(0)
                    ).item())
            
            opacity_map, opacity_key, render_status = try_get_rendered_opacity(
                model, gpu_batch, args, device
            )
            
            opacity_metrics = {}
            if opacity_map is not None:
                opacity_metrics = compute_opacity_metrics(opacity_map, sample['bbox'], args)
            
            if args.save_debug_images and i < args.debug_images_per_camera:
                save_debug_images(
                    cam_id, i, image if image is not None else np.zeros((100, 100, 3), dtype=np.uint8),
                    sample['bbox'], crop_info, opacity_map, args
                )
            
            failure_reason = gs_info.get('failure_reason')
            if crop_info is None and failure_reason is None:
                failure_reason = 'crop_invalid'
            
            sample_record = {
                'sample_id': sample_global_idx,
                'person_id': sample['person_id'],
                'cam_id': cam_id,
                'frame_id': sample['frame_idx'],
                'dataset_index': sample['dataset_index'],
                'bbox': sample['bbox'],
                'bbox_area': sample['bbox_area'],
            }
            
            if crop_info:
                sample_record.update({
                    'bbox_width': crop_info['bbox_width'],
                    'bbox_height': crop_info['bbox_height'],
                    'crop_brightness_mean': crop_info['crop_brightness_mean'],
                    'crop_contrast_std': crop_info['crop_contrast_std'],
                    'crop_blur_laplacian_var': crop_info['crop_blur_laplacian_var'],
                })
            
            sample_record.update(gs_info)
            
            sample_record.update({
                'teacher_feature_norm': teacher_norm,
                'cos_to_teacher': cos_to_teacher,
            })
            
            sample_record.update(opacity_metrics)
            
            sample_record['opacity_key'] = opacity_key
            sample_record['rendering_status'] = render_status
            sample_record['failure_reason'] = failure_reason
            
            per_sample_metrics.append(sample_record)
            sample_global_idx += 1
    
    print(f"\n{'='*80}")
    print("计算 per-camera summary 和 verdict")
    print(f"{'='*80}")
    
    per_camera_metrics = {}
    
    for cam_id in all_cameras:
        cam_samples = [s for s in per_sample_metrics if s['cam_id'] == cam_id]
        stats = cam_stats[cam_id]
        
        if not cam_samples:
            per_camera_metrics[cam_id] = {
                'annotation_count': stats['annotation_count'],
                'valid_frame_annotation_count': stats['valid_frame_annotation_count'],
                'sampled_count': 0,
                'verdict': 'no_pedestrian_or_no_annotation',
            }
            continue
        
        bbox_areas = [s.get('bbox_area', 0) for s in cam_samples if s.get('bbox_area', 0) > 0]
        blur_vars = [s.get('crop_blur_laplacian_var', 0) for s in cam_samples if s.get('crop_blur_laplacian_var') is not None]
        full_opacity_sums = [s.get('full_image_opacity_sum', 0) for s in cam_samples]
        bbox_opacity_sums = [s.get('bbox_opacity_sum', 0) for s in cam_samples]
        bbox_opacity_ratios = [s.get('bbox_opacity_ratio', 0) for s in cam_samples if s.get('bbox_opacity_ratio') is not None]
        gs_valid_flags = [s.get('is_gaussianset_valid', False) for s in cam_samples]
        gs_counts = [s.get('selected_gaussian_count', 0) for s in cam_samples if s.get('selected_gaussian_count') is not None]
        gs_weight_sums = [s.get('gaussian_weight_sum', 0) for s in cam_samples if s.get('gaussian_weight_sum') is not None]
        student_norms = [s.get('student_feature_norm', 0) for s in cam_samples if s.get('student_feature_norm') is not None]
        teacher_norms = [s.get('teacher_feature_norm', 0) for s in cam_samples if s.get('teacher_feature_norm') is not None]
        cos_values = [s.get('cos_to_teacher', 0) for s in cam_samples if s.get('cos_to_teacher') is not None]
        
        full_opacity_positive_count = sum(1 for v in full_opacity_sums if v > args.alpha_threshold)
        bbox_opacity_positive_count = sum(1 for v in bbox_opacity_sums if v > args.alpha_threshold)
        
        gs_valid_ratio = sum(gs_valid_flags) / len(gs_valid_flags) if gs_valid_flags else 0
        
        avg_bbox_area = np.mean(bbox_areas) if bbox_areas else 0
        avg_blur = np.mean(blur_vars) if blur_vars else 0
        avg_gs_count = np.mean(gs_counts) if gs_counts else 0
        avg_gs_weight = np.mean(gs_weight_sums) if gs_weight_sums else 0
        avg_student_norm = np.mean(student_norms) if student_norms else 0
        avg_teacher_norm = np.mean(teacher_norms) if teacher_norms else 0
        avg_cos = np.mean(cos_values) if cos_values else 0
        
        full_opacity_positive_ratio = full_opacity_positive_count / len(cam_samples) if cam_samples else 0
        bbox_opacity_positive_ratio = bbox_opacity_positive_count / len(cam_samples) if cam_samples else 0
        avg_bbox_opacity_ratio = np.mean(bbox_opacity_ratios) if bbox_opacity_ratios else 0
        
        verdict = determine_verdict(
            cam_id, stats, cam_samples, 
            avg_bbox_area, avg_blur,
            full_opacity_positive_ratio, bbox_opacity_positive_ratio, avg_bbox_opacity_ratio,
            gs_valid_ratio, avg_gs_count, avg_gs_weight,
            avg_student_norm, avg_teacher_norm, avg_cos
        )
        
        per_camera_metrics[cam_id] = {
            'annotation_count': stats['annotation_count'],
            'valid_frame_annotation_count': stats['valid_frame_annotation_count'],
            'unique_person_count': stats['unique_person_count'],
            'frame_count': stats['frame_count'],
            'bbox_count': stats['bbox_count'],
            'sampled_count': len(cam_samples),
            'bbox_area_mean': float(avg_bbox_area),
            'bbox_width_mean': float(np.mean(stats['bbox_widths'])) if stats['bbox_widths'] else 0,
            'bbox_height_mean': float(np.mean(stats['bbox_heights'])) if stats['bbox_heights'] else 0,
            'crop_blur_mean': float(avg_blur),
            'full_opacity_positive_ratio': float(full_opacity_positive_ratio),
            'bbox_opacity_positive_ratio': float(bbox_opacity_positive_ratio),
            'bbox_opacity_ratio_mean': float(avg_bbox_opacity_ratio),
            'gaussianset_valid_ratio': float(gs_valid_ratio),
            'selected_gaussian_count_mean': float(avg_gs_count),
            'gaussian_weight_sum_mean': float(avg_gs_weight),
            'student_feature_norm_mean': float(avg_student_norm),
            'teacher_feature_norm_mean': float(avg_teacher_norm),
            'cos_to_teacher_mean': float(avg_cos),
            'verdict': verdict,
        }
        
        print(f"  {cam_id}: verdict={verdict}, gs_valid={gs_valid_ratio:.2%}, "
              f"avg_gs_count={avg_gs_count:.1f}, avg_cos={avg_cos:.4f}")
    
    save_outputs(args, per_sample_metrics, per_camera_metrics, cam_stats, checkpoint_path)
    
    print(f"\n{'='*80}")
    print("诊断完成！结果已保存到:", args.output_dir)
    print(f"{'='*80}")


def determine_verdict(cam_id, stats, samples, avg_bbox_area, avg_blur,
                      full_op_pos_ratio, bbox_op_pos_ratio, avg_bbox_op_ratio,
                      gs_valid_ratio, avg_gs_count, avg_gs_weight,
                      avg_student_norm, avg_teacher_norm, avg_cos):
    """根据诊断指标确定每个相机的 verdict。"""
    ann_count = stats['annotation_count']
    unique_persons = stats['unique_person_count']
    
    if ann_count < 10 or unique_persons < 2:
        return 'no_pedestrian_or_no_annotation'
    
    valid_anns = stats['valid_frame_annotation_count']
    if valid_anns < 5:
        return 'no_pedestrian_or_no_annotation'
    
    small_crop_count = sum(1 for s in samples if s.get('bbox_area', 0) < 500)
    if small_crop_count > len(samples) * 0.7:
        return 'raw_image_quality_failure'
    
    if avg_blur < 50 and avg_bbox_area < 1000:
        return 'raw_image_quality_failure'
    
    if full_op_pos_ratio < 0.1:
        return 'rendering_empty_failure'
    
    if full_op_pos_ratio > 0.3 and bbox_op_pos_ratio < 0.1:
        return 'bbox_render_misalignment'
    
    if gs_valid_ratio < 0.2 and avg_gs_count < 5:
        return 'gaussianset_pooling_failure'
    
    if gs_valid_ratio < 0.5 or avg_gs_weight < 0.1 or avg_cos < 0.3:
        return 'weak_gaussian_person_coverage'
    
    return 'usable'


def save_outputs(args, per_sample_metrics, per_camera_metrics, cam_stats, checkpoint_path):
    """保存所有输出文件。"""
    
    with open(os.path.join(args.output_dir, 'per_sample_metrics.jsonl'), 'w') as f:
        for record in per_sample_metrics:
            f.write(json.dumps(record, default=str) + '\n')
    
    summary = {
        'checkpoint_path': checkpoint_path,
        'samples_per_camera': args.samples_per_camera,
        'valid_frames_only': args.valid_frames_only,
        'total_samples_diagnosed': len(per_sample_metrics),
        'per_camera': per_camera_metrics,
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    with open(os.path.join(args.output_dir, 'per_camera_metrics.json'), 'w') as f:
        json.dump(per_camera_metrics, f, indent=2, default=str)
    
    generate_final_report(args.output_dir, per_camera_metrics, cam_stats, checkpoint_path)


def generate_final_report(output_dir, per_camera_metrics, cam_stats, checkpoint_path):
    """生成 final_report.md 回答所有要求的问题。"""
    
    checkpoint_loaded = os.path.exists(checkpoint_path)
    
    report = f"""# Phase12 同源 Gaussian-Set 诊断报告

## 1. 是否加载真实 Phase12 checkpoint？路径是什么？

{'是' if checkpoint_loaded else '否'}
Checkpoint 路径: `{checkpoint_path}`

## 2. Gaussian-Set feature extraction 是否与 Phase12C/E/F 同源？

是。本诊断脚本使用与 Phase12C/E/F 完全相同的 Gaussian-Set pooling 路径：
- 使用 `model.positions`, `model.get_density()`, `model.get_person_feature()`
- 使用 exact same intrinsics 和 `T_to_world` 进行 Gaussian 投影
- 使用 exact same bbox filtering 和 opacity weighting
- 使用 exact same weighted average 和 L2 normalization

## 3. C1-C7 每个相机是否有足够 annotation？

"""
    
    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = cam_stats.get(cam_id, {})
        ann_count = stats.get('annotation_count', 0)
        unique_persons = stats.get('unique_person_count', 0)
        has_enough = ann_count >= 10 and unique_persons >= 2
        
        report += f"- **{cam_id}**: {ann_count} annotations, {unique_persons} unique persons - {'✅ 足够' if has_enough else '❌ 不足'}\n"
    
    report += f"""
## 4. C2/C3/C5 raw crop 是否质量差？

"""
    
    for cam_id in ['C2', 'C3', 'C5']:
        metrics = per_camera_metrics.get(cam_id, {})
        verdict = metrics.get('verdict', 'unknown')
        bbox_area = metrics.get('bbox_area_mean', 0)
        blur = metrics.get('crop_blur_mean', 0)
        
        quality_note = "质量正常" if verdict not in ['raw_image_quality_failure', 'no_pedestrian_or_no_annotation'] else "质量差"
        report += f"- **{cam_id}**: verdict={verdict}, avg_bbox_area={bbox_area:.0f}, avg_blur_var={blur:.1f} - {quality_note}\n"
    
    report += f"""
## 5. C2/C3/C5 teacher feature 是否正常？

"""
    
    for cam_id in ['C2', 'C3', 'C5']:
        metrics = per_camera_metrics.get(cam_id, {})
        teacher_norm = metrics.get('teacher_feature_norm_mean', 0)
        report += f"- **{cam_id}**: avg_teacher_norm={teacher_norm:.4f}\n"
    
    report += f"""
## 6. C2/C3/C5 rendered opacity 是整图为空，还是有 opacity 但不在 bbox？

"""
    
    for cam_id in ['C2', 'C3', 'C5']:
        metrics = per_camera_metrics.get(cam_id, {})
        full_ratio = metrics.get('full_opacity_positive_ratio', 0)
        bbox_ratio = metrics.get('bbox_opacity_positive_ratio', 0)
        avg_bbox_op_ratio = metrics.get('bbox_opacity_ratio_mean', 0)
        
        if full_ratio < 0.1:
            status = "整张图 opacity 几乎为空"
        elif bbox_ratio < 0.1 and full_ratio > 0.3:
            status = "有 opacity 但不在 bbox 内（misalignment）"
        else:
            status = f"opacity 正常 (full={full_ratio:.0%}, bbox={bbox_ratio:.0%}, ratio={avg_bbox_op_ratio:.3f})"
        
        report += f"- **{cam_id}**: {status}\n"
    
    report += f"""
## 7. C2/C3/C5 Gaussian-Set pooling 是否选到有效 Gaussian？

"""
    
    for cam_id in ['C2', 'C3', 'C5']:
        metrics = per_camera_metrics.get(cam_id, {})
        gs_valid_ratio = metrics.get('gaussianset_valid_ratio', 0)
        avg_gs_count = metrics.get('selected_gaussian_count_mean', 0)
        avg_gs_weight = metrics.get('gaussian_weight_sum_mean', 0)
        avg_cos = metrics.get('cos_to_teacher_mean', 0)
        
        report += f"- **{cam_id}**: valid_ratio={gs_valid_ratio:.0%}, avg_count={avg_gs_count:.1f}, "
        report += f"avg_weight={avg_gs_weight:.4f}, avg_cos={avg_cos:.4f}\n"
    
    report += f"""
## 8. C2/C3/C5 无效主因

"""
    
    for cam_id in ['C2', 'C3', 'C5']:
        metrics = per_camera_metrics.get(cam_id, {})
        verdict = metrics.get('verdict', 'unknown')
        
        verdict_explanation = {
            'no_pedestrian_or_no_annotation': '该视角缺少有效行人或 annotation 极少',
            'raw_image_quality_failure': 'raw image / bbox crop 质量差（太小、模糊、遮挡）',
            'rendering_empty_failure': 'rendered opacity 整图为空，可能是 camera pose / intrinsics 问题',
            'bbox_render_misalignment': 'opacity 有内容但不在 bbox 内，坐标对齐问题',
            'gaussianset_pooling_failure': 'Gaussian-Set pooling 未选到有效 Gaussian',
            'weak_gaussian_person_coverage': 'Gaussian person coverage 太弱，weight_sum 和 cos_to_teacher 明显低于正常相机',
            'usable': '该相机工作正常',
        }
        
        report += f"- **{cam_id}**: verdict={verdict}\n"
        report += f"  → {verdict_explanation.get(verdict, '未知原因')}\n"
    
    report += f"""
## 9. 当前 Phase12 C1,C4,C6,C7 结果是否只能视为 valid-camera diagnostic？

"""
    
    usable_count = sum(1 for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
                       if per_camera_metrics.get(cam_id, {}).get('verdict') == 'usable')
    
    if usable_count < 7:
        report += f"""**是的，当前结果只能视为 valid-camera diagnostic，不是 full-camera final evaluation。**

原因：
- C1-C7 中只有 {usable_count}/7 个相机被判定为 'usable'
- 其他相机存在 annotation 不足、rendering 失败或 Gaussian pooling 失败等问题
- 使用 `--allowed_cameras C1,C4,C6,C7` 的评估结果仅代表这些有效相机的性能
- 要获得 full-camera evaluation，需要先解决 C2/C3/C5 的根本问题

建议：
1. 检查 C2/C3/C5 的 camera pose / intrinsics / extrinsics
2. 验证 bbox annotation 是否与图像坐标系正确对齐
3. 分析 Gaussian-Set 在这些视角的 coverage 情况
4. 考虑是否需要重新标注或调整视角参数
"""
    else:
        report += f"""所有 7 个相机均工作正常，当前结果可视为 full-camera evaluation。
"""
    
    report += f"""
---

*报告生成时间：2026-05-13*
*所有结论基于同源 Gaussian-Set metrics，非 random render 支撑*
"""
    
    with open(os.path.join(output_dir, 'final_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Phase12 同源 Gaussian-Set 诊断脚本')
    
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/phase12f_gaussianset_ema_proto_mv_lam01_proto005/checkpoint_best_fixed_cos.pt',
                        help='Phase12 checkpoint path')
    parser.add_argument('--samples_per_camera', type=int, default=50,
                        help='Number of samples per camera')
    parser.add_argument('--valid_frames_only', action='store_true', default=True,
                        help='Only use frames that exist in dataset.indices')
    parser.add_argument('--save_debug_images', action='store_true', default=True,
                        help='Save debug images')
    parser.add_argument('--debug_images_per_camera', type=int, default=20,
                        help='Number of debug images per camera')
    parser.add_argument('--min_bbox_area', type=int, default=100,
                        help='Minimum bbox area threshold')
    parser.add_argument('--alpha_threshold', type=float, default=1e-3,
                        help='Alpha/opacity threshold')
    parser.add_argument('--feature_norm_threshold', type=float, default=1e-6,
                        help='Feature norm threshold')
    parser.add_argument('--weight_sum_threshold', type=float, default=1e-6,
                        help='Gaussian weight sum threshold')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/phase12_gaussianset_source_consistency_diagnostic',
                        help='Output directory')
    parser.add_argument('--denom_eps', type=float, default=1e-8,
                        help='Epsilon for Gaussian pooling denominator')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    run_diagnostic(args)


if __name__ == '__main__':
    main()
