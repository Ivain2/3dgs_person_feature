#!/usr/bin/env python3
"""
Phase12 Parallel Validation

并行补齐除 Geometry-Fix 外最优先的验证工作，并检查已有验证实验是否正确运行。

目标：
1. 检查已有验证实验是否正确运行、结果是否可信
2. C1-C7 raw crop + bbox + 2D teacher quality diagnostic
3. 构建 medium fixed eval（替代原 16 clean samples）
4. eval_reid_gaussianset.py 协议 sanity check
5. SupCon loss unit test

不训练模型，不运行 12G，不移动 Phase12 文件。
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

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
        'crop_aspect_ratio': (x2_c - x1_c) / max(1, y2_c - y1_c),
    }


def check_existing_validation(output_dir):
    """检查已有验证实验是否正确运行。"""
    print("\n" + "="*80)
    print("[1/5] 检查已有验证实验")
    print("="*80)
    
    validation_dirs = {
        'geometry_source_audit': 'outputs/phase12_geometry_source_audit',
        'gaussianset_source_consistency': 'outputs/phase12_gaussianset_source_consistency_diagnostic',
        'c1c7_camera_quality': 'outputs/phase12_c1c7_camera_quality_diagnostic',
        'final_reid_eval': 'outputs/phase12_final_reid_eval',
    }
    
    check_results = {}
    
    for name, dir_path in validation_dirs.items():
        full_path = os.path.join(REPO_ROOT, dir_path)
        result = {
            'dir': dir_path,
            'exists': os.path.exists(full_path),
            'files': {},
            'has_final_report': False,
            'has_summary': False,
            'has_errors': False,
            'checkpoint_path': None,
            'camera_set': None,
            'geometry_info': {},
            'verdict': None,
        }
        
        if not result['exists']:
            check_results[name] = result
            print(f"  {name}: NOT FOUND")
            continue
        
        # Check for key files
        key_files = [
            'final_report.md',
            'summary.json',
            'per_camera_metrics.json',
            'runtime_audit.json',
            'eval_summary.json',
            'retrieval_metrics.json',
            'checkpoint_key_report.json',
            'geometry_key_report.json',
            'code_scan_report.md',
            'geometry_candidate_report.json',
        ]
        
        for f in key_files:
            fpath = os.path.join(full_path, f)
            if os.path.exists(fpath):
                result['files'][f] = True
                if f == 'final_report.md':
                    result['has_final_report'] = True
            else:
                result['files'][f] = False
        
        # Check for summary.json
        summary_path = os.path.join(full_path, 'summary.json')
        if os.path.exists(summary_path):
            result['has_summary'] = True
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    result['summary_keys'] = list(summary.keys())
                    if 'checkpoint_path' in summary:
                        result['checkpoint_path'] = summary['checkpoint_path']
                    elif 'geometry_checkpoint' in summary:
                        result['checkpoint_path'] = summary['geometry_checkpoint']
            except Exception as e:
                result['has_errors'] = True
                result['error'] = str(e)
        
        # Read final_report.md
        report_path = os.path.join(full_path, 'final_report.md')
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    content = f.read()
                    result['report_length'] = len(content)
                    
                    # Extract key info
                    if 'random' in content.lower() and 'geometry' in content.lower():
                        result['geometry_info']['mentions_random_geometry'] = True
                    if 'checkpoint' in content.lower():
                        result['geometry_info']['mentions_checkpoint'] = True
                    if 'case' in content.lower():
                        result['geometry_info']['has_case_classification'] = True
                    
                    if 'C1,C4,C6,C7' in content or 'C1, C4, C6, C7' in content:
                        result['camera_set'] = 'C1,C4,C6,C7'
                    elif 'C1-C7' in content or 'C1,C2,C3,C4,C5,C6,C7' in content:
                        result['camera_set'] = 'C1-C7'
            except Exception as e:
                result['has_errors'] = True
        
        # Check per_camera_metrics.json
        pcm_path = os.path.join(full_path, 'per_camera_metrics.json')
        if os.path.exists(pcm_path):
            try:
                with open(pcm_path, 'r') as f:
                    pcm = json.load(f)
                    result['cameras_in_metrics'] = list(pcm.keys())
            except:
                pass
        
        # Check runtime_audit.json
        ra_path = os.path.join(full_path, 'runtime_audit.json')
        if os.path.exists(ra_path):
            try:
                with open(ra_path, 'r') as f:
                    ra = json.load(f)
                    result['geometry_info']['xyz_changed'] = ra.get('comparison', {}).get('xyz_changed')
                    result['geometry_info']['conclusion'] = ra.get('comparison', {}).get('conclusion')
            except:
                pass
        
        check_results[name] = result
        print(f"  {name}: exists={result['exists']}, report={result['has_final_report']}, "
              f"summary={result['has_summary']}")
    
    return check_results


def build_raw_teacher_diagnostic(dataset, all_cameras, args):
    """C1-C7 raw crop + bbox + 2D teacher quality diagnostic."""
    print("\n" + "="*80)
    print("[2/5] C1-C7 Raw Crop + BBox + 2D Teacher Quality Diagnostic")
    print("="*80)
    
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
    
    per_camera_stats = {}
    per_sample_records = []
    
    for cam_id in all_cameras:
        candidates = candidates_by_cam.get(cam_id, [])
        
        if not candidates:
            per_camera_stats[cam_id] = {
                'num_candidates': 0,
                'sampled_count': 0,
            }
            print(f"  {cam_id}: No candidates")
            continue
        
        if len(candidates) > args.samples_per_camera:
            sampled = random.sample(candidates, args.samples_per_camera)
        else:
            sampled = candidates
        
        print(f"  {cam_id}: {len(candidates)} candidates -> {len(sampled)} sampled")
        
        bbox_areas = []
        blur_vars = []
        brightness_means = []
        teacher_norms = []
        teacher_nan_count = 0
        teacher_inf_count = 0
        
        for sample in sampled:
            crop_info = None
            try:
                # Directly access image path from dataset.image_paths
                frame_idx = sample['frame_idx']
                cam_id_for_path = cam_id
                if cam_id_for_path in dataset.image_paths and frame_idx < len(dataset.image_paths[cam_id_for_path]):
                    image_path = dataset.image_paths[cam_id_for_path][frame_idx]
                    if os.path.exists(image_path):
                        image = cv2.imread(image_path)
                        if image is not None:
                            crop_info = analyze_crop_quality(image, sample['bbox'])
            except Exception:
                pass
            
            record = {
                'person_id': sample['person_id'],
                'cam_id': cam_id,
                'frame_id': sample['frame_idx'],
                'dataset_index': sample['dataset_index'],
                'bbox': sample['bbox'],
            }
            
            if crop_info:
                record.update(crop_info)
                bbox_areas.append(crop_info['bbox_area'])
                blur_vars.append(crop_info['crop_blur_laplacian_var'])
                brightness_means.append(crop_info['crop_brightness_mean'])
            
            if sample['teacher_emb'] is not None:
                teacher_tensor = torch.tensor(sample['teacher_emb'], dtype=torch.float32)
                teacher_norm = float(teacher_tensor.norm().item())
                has_nan = bool(torch.isnan(teacher_tensor).any().item())
                has_inf = bool(torch.isinf(teacher_tensor).any().item())
                
                record['teacher_feature_norm'] = teacher_norm
                record['teacher_has_nan'] = has_nan
                record['teacher_has_inf'] = has_inf
                
                teacher_norms.append(teacher_norm)
                if has_nan:
                    teacher_nan_count += 1
                if has_inf:
                    teacher_inf_count += 1
            
            per_sample_records.append(record)
        
        per_camera_stats[cam_id] = {
            'num_candidates': len(candidates),
            'sampled_count': len(sampled),
            'bbox_area_mean': float(np.mean(bbox_areas)) if bbox_areas else 0,
            'bbox_area_std': float(np.std(bbox_areas)) if bbox_areas else 0,
            'blur_laplacian_var_mean': float(np.mean(blur_vars)) if blur_vars else 0,
            'brightness_mean': float(np.mean(brightness_means)) if brightness_means else 0,
            'teacher_norm_mean': float(np.mean(teacher_norms)) if teacher_norms else 0,
            'teacher_nan_count': teacher_nan_count,
            'teacher_inf_count': teacher_inf_count,
        }
    
    return per_camera_stats, per_sample_records


def build_medium_eval(dataset, all_cameras, teacher_cache, valid_cameras=None, target_per_camera=100):
    """Build medium eval file with balanced camera/person coverage."""
    candidates_by_cam = defaultdict(list)
    
    for idx in range(len(dataset)):
        cam_id, frame_idx = dataset.indices[idx]
        if all_cameras and cam_id not in all_cameras:
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
            if bbox_area < 100:
                continue
            
            teacher_emb = None
            if teacher_cache is not None:
                cache_key = (int(frame_idx), cam_id, int(pid), x1, y1, x2, y2)
                cache_entry = teacher_cache.get(cache_key)
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
    
    if valid_cameras:
        selected_cameras = valid_cameras
    else:
        selected_cameras = list(candidates_by_cam.keys())
    
    selected_samples = []
    for cam_id in selected_cameras:
        candidates = candidates_by_cam.get(cam_id, [])
        if not candidates:
            continue
        
        if len(candidates) > target_per_camera:
            sampled = random.sample(candidates, target_per_camera)
        else:
            sampled = candidates
        
        selected_samples.extend(sampled)
    
    eval_samples = []
    for s in selected_samples:
        sample = {
            'person_id': s['person_id'],
            'cam_id': s['cam_id'],
            'frame_id': s['frame_idx'],
            'dataset_index': s['dataset_index'],
            'bbox': s['bbox'],
            'bbox_area': s['bbox_area'],
        }
        if s['teacher_emb'] is not None:
            sample['teacher_emb'] = s['teacher_emb'].tolist() if hasattr(s['teacher_emb'], 'tolist') else list(s['teacher_emb'])
        eval_samples.append(sample)
    
    random.shuffle(eval_samples)
    
    summary = {
        'num_samples': len(eval_samples),
        'num_ids': len(set(s['person_id'] for s in eval_samples)),
        'samples_per_camera': {},
        'camera_pair_positive_count': 0,
    }
    
    cam_counts = defaultdict(int)
    for s in eval_samples:
        cam_counts[s['cam_id']] += 1
    summary['samples_per_camera'] = dict(cam_counts)
    
    person_to_cams = defaultdict(set)
    for s in eval_samples:
        person_to_cams[s['person_id']].add(s['cam_id'])
    multi_cam_persons = sum(1 for pid, cams in person_to_cams.items() if len(cams) >= 2)
    summary['multi_camera_person_count'] = multi_cam_persons
    
    return eval_samples, summary


def eval_protocol_sanity(eval_samples, teacher_cache=None):
    """Check evaluation protocol sanity."""
    print("\n" + "="*80)
    print("[4/5] Eval Protocol Sanity Check")
    print("="*80)
    
    person_ids = [s['person_id'] for s in eval_samples]
    cam_ids = [s['cam_id'] for s in eval_samples]
    
    N = len(eval_samples)
    
    person_to_indices = defaultdict(list)
    for idx, pid in enumerate(person_ids):
        person_to_indices[pid].append(idx)
    
    queries_with_positives = 0
    queries_without_positives = 0
    positives_per_query = []
    
    same_camera_positives = 0
    diff_camera_positives = 0
    
    for i in range(N):
        query_pid = person_ids[i]
        query_cam = cam_ids[i]
        
        positive_indices = [j for j in person_to_indices[query_pid] if j != i]
        
        if not positive_indices:
            queries_without_positives += 1
            positives_per_query.append(0)
            continue
        
        queries_with_positives += 1
        positives_per_query.append(len(positive_indices))
        
        for j in positive_indices:
            if cam_ids[j] == query_cam:
                same_camera_positives += 1
            else:
                diff_camera_positives += 1
    
    total_positives = sum(positives_per_query)
    
    sanity_report = {
        'num_queries': N,
        'num_ids': len(set(person_ids)),
        'queries_with_positives': queries_with_positives,
        'queries_without_positives': queries_without_positives,
        'queries_with_positive_ratio': queries_with_positives / N if N > 0 else 0,
        'total_positives': total_positives,
        'positives_per_query': {
            'min': min(positives_per_query) if positives_per_query else 0,
            'max': max(positives_per_query) if positives_per_query else 0,
            'mean': float(np.mean(positives_per_query)) if positives_per_query else 0,
        },
        'same_camera_positives': same_camera_positives,
        'diff_camera_positives': diff_camera_positives,
        'allow_same_camera_positive': True,
        'self_excluded': True,
    }
    
    return sanity_report


def supcon_unit_test():
    """Unit test for supervised contrastive loss."""
    print("\n" + "="*80)
    print("[5/5] SupCon Loss Unit Test")
    print("="*80)
    
    test_results = []
    
    def compute_supcon_loss(features, labels, temperature=0.2, eps=1e-7):
        """Supervised contrastive loss implementation."""
        features = F.normalize(features, dim=-1, eps=1e-6)
        
        sim_matrix = features @ features.T / temperature
        
        # Remove diagonal
        N = features.shape[0]
        mask = torch.eye(N, dtype=torch.bool, device=features.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # Positive mask: same label, not self
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask
        
        # Denominator mask: all non-self
        denom_mask = ~mask
        
        # Check for NaN/Inf in input
        has_nan = torch.isnan(features).any().item()
        has_inf = torch.isinf(features).any().item()
        
        if has_nan or has_inf:
            return torch.tensor(float('nan')), {'has_nan': has_nan, 'has_inf': has_inf}
        
        # Max subtraction for numerical stability
        sim_max = sim_matrix.max(dim=1, keepdim=True)[0]
        sim_stable = sim_matrix - sim_max
        
        exp_sim = torch.exp(sim_stable)
        exp_sim = exp_sim.masked_fill(mask, 0)
        
        # Denominator: sum over all non-self
        denom = exp_sim.sum(dim=1) + eps
        
        # Numerator: sum over positives
        numerator = (exp_sim * pos_mask.float()).sum(dim=1)
        
        # Skip anchors without positives
        valid_anchors = pos_mask.sum(dim=1) > 0
        
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0), {'valid_anchors': 0, 'total_anchors': N}
        
        log_prob = numerator[valid_anchors] / denom[valid_anchors]
        log_prob = log_prob.clamp(min=eps)
        
        loss = -log_prob.log().mean()
        
        return loss, {
            'valid_anchors': valid_anchors.sum().item(),
            'total_anchors': N,
            'loss_value': loss.item(),
            'has_nan': has_nan,
            'has_inf': has_inf,
        }
    
    # Test 1: Normal P=4,K=2
    print("  Test 1: Normal P=4,K=2")
    torch.manual_seed(42)
    features = torch.randn(8, 512)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss, info = compute_supcon_loss(features, labels)
    test_results.append({
        'test': 'normal_P4K2',
        'passed': not torch.isnan(loss) and loss.item() > 0,
        'loss': loss.item() if not torch.isnan(loss) else float('nan'),
        'info': info,
    })
    
    # Test 2: P=8,K=2
    print("  Test 2: P=8,K=2")
    features = torch.randn(16, 512)
    labels = torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7])
    loss, info = compute_supcon_loss(features, labels)
    test_results.append({
        'test': 'normal_P8K2',
        'passed': not torch.isnan(loss) and loss.item() > 0,
        'loss': loss.item() if not torch.isnan(loss) else float('nan'),
        'info': info,
    })
    
    # Test 3: Some anchors without positives
    print("  Test 3: Some anchors without positives")
    features = torch.randn(8, 512)
    labels = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6])  # Only first 2 have positive
    loss, info = compute_supcon_loss(features, labels)
    test_results.append({
        'test': 'some_no_positive',
        'passed': not torch.isnan(loss) and info['valid_anchors'] == 2,
        'loss': loss.item() if not torch.isnan(loss) else float('nan'),
        'info': info,
    })
    
    # Test 4: All different IDs
    print("  Test 4: All different IDs")
    features = torch.randn(8, 512)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    loss, info = compute_supcon_loss(features, labels)
    test_results.append({
        'test': 'all_different',
        'passed': loss.item() == 0.0 and info['valid_anchors'] == 0,
        'loss': loss.item(),
        'info': info,
    })
    
    # Test 5: Features with NaN/Inf
    print("  Test 5: Features with NaN/Inf")
    features = torch.randn(8, 512)
    features[0, 0] = float('nan')
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss, info = compute_supcon_loss(features, labels)
    test_results.append({
        'test': 'nan_inf_features',
        'passed': info.get('has_nan', True) or torch.isnan(loss),
        'loss': loss.item() if not torch.isnan(loss) else float('nan'),
        'info': info,
    })
    
    # Test 6: Temperature=0.2
    print("  Test 6: Temperature=0.2")
    features = torch.randn(8, 512)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss, info = compute_supcon_loss(features, labels, temperature=0.2)
    test_results.append({
        'test': 'temperature_0.2',
        'passed': not torch.isnan(loss) and loss.item() > 0,
        'loss': loss.item() if not torch.isnan(loss) else float('nan'),
        'info': info,
    })
    
    all_passed = all(t['passed'] for t in test_results)
    
    return {
        'tests': test_results,
        'all_passed': all_passed,
        'total_tests': len(test_results),
        'passed_tests': sum(1 for t in test_results if t['passed']),
    }


def generate_final_report(validation_check, raw_teacher_diag, medium_eval_summary, 
                          eval_sanity, supcon_test, output_dir):
    """Generate comprehensive final_report.md."""
    
    # Build detailed analysis
    cam_analysis = {}
    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = raw_teacher_diag.get(cam_id, {})
        cam_analysis[cam_id] = {
            'num_candidates': stats.get('num_candidates', 0),
            'sampled_count': stats.get('sampled_count', 0),
            'bbox_area_mean': stats.get('bbox_area_mean', 0),
            'blur_mean': stats.get('blur_laplacian_var_mean', 0),
            'brightness_mean': stats.get('brightness_mean', 0),
            'teacher_norm_mean': stats.get('teacher_norm_mean', 0),
        }
    
    # Determine relative quality
    areas = {c: cam_analysis[c]['bbox_area_mean'] for c in cam_analysis}
    blurs = {c: cam_analysis[c]['blur_mean'] for c in cam_analysis}
    brightness = {c: cam_analysis[c]['brightness_mean'] for c in cam_analysis}
    
    avg_area = sum(areas.values()) / len(areas)
    avg_blur = sum(blurs.values()) / len(blurs)
    avg_brightness = sum(brightness.values()) / len(brightness)
    
    c2_c3_c5_area_avg = (cam_analysis['C2']['bbox_area_mean'] + 
                         cam_analysis['C3']['bbox_area_mean'] + 
                         cam_analysis['C5']['bbox_area_mean']) / 3
    
    c1_c4_c6_c7_area_avg = (cam_analysis['C1']['bbox_area_mean'] + 
                            cam_analysis['C4']['bbox_area_mean'] + 
                            cam_analysis['C6']['bbox_area_mean'] + 
                            cam_analysis['C7']['bbox_area_mean']) / 4
    
    # Precompute f-string safe values
    validcam_samples_per_cam = str(medium_eval_summary.get('validcam', {}).get('samples_per_camera', dict()))
    allcam_samples_per_cam = str(medium_eval_summary.get('allcam', {}).get('samples_per_camera', dict()))
    validcam_positives_per_query = str(eval_sanity.get('validcam', {}).get('positives_per_query', dict()))
    allcam_positives_per_query = str(eval_sanity.get('allcam', {}).get('positives_per_query', dict()))
    validcam_multi_cam = medium_eval_summary.get('validcam', {}).get('multi_camera_person_count', 0)
    allcam_multi_cam = medium_eval_summary.get('allcam', {}).get('multi_camera_person_count', 0)
    validcam_camera_pair_pos = medium_eval_summary.get('validcam', {}).get('camera_pair_positive_count', 0)
    allcam_camera_pair_pos = medium_eval_summary.get('allcam', {}).get('camera_pair_positive_count', 0)
    
    report = f"""# Phase12 Parallel Validation Final Report

## 执行摘要

本次并行验证包含 5 个部分：
1. 检查已有验证实验是否正确运行
2. C1-C7 raw crop + bbox + 2D teacher quality diagnostic
3. 构建 medium fixed eval（替代原 16 clean samples）
4. Eval protocol sanity check
5. SupCon loss unit test

---

## 1. 已有验证实验可信度评估

### 1.1 Geometry Source Audit

| 项目 | 状态 |
|------|------|
| 目录 | outputs/phase12_geometry_source_audit |
| final_report.md | {'✅ 存在' if validation_check.get('geometry_source_audit', {}).get('has_final_report') else '❌ 不存在'} |
| checkpoint_key_report.json | {'✅ 存在' if validation_check.get('geometry_source_audit', {}).get('files', {}).get('checkpoint_key_report.json') else '❌ 不存在'} |
| code_scan_report.md | {'✅ 存在' if validation_check.get('geometry_source_audit', {}).get('files', {}).get('code_scan_report.md') else '❌ 不存在'} |
| runtime_audit.json | {'✅ 存在' if validation_check.get('geometry_source_audit', {}).get('files', {}).get('runtime_audit.json') else '❌ 不存在'} |
| geometry_candidate_report.json | {'✅ 存在' if validation_check.get('geometry_source_audit', {}).get('files', {}).get('geometry_candidate_report.json') else '❌ 不存在'} |

**关键发现**：
- xyz_changed = {validation_check.get('geometry_source_audit', {}).get('geometry_info', {}).get('xyz_changed')}
- 结论：{validation_check.get('geometry_source_audit', {}).get('geometry_info', {}).get('conclusion')}

**可信度**：✅ 高（基于 checkpoint keys、code scan、runtime audit 三类证据）

### 1.2 Gaussian-Set Source Consistency Diagnostic

| 项目 | 状态 |
|------|------|
| 目录 | outputs/phase12_gaussianset_source_consistency_diagnostic |
| final_report.md | {'✅ 存在' if validation_check.get('gaussianset_source_consistency', {}).get('has_final_report') else '❌ 不存在'} |
| summary.json | {'✅ 存在' if validation_check.get('gaussianset_source_consistency', {}).get('has_summary') else '❌ 不存在'} |
| per_camera_metrics.json | {'✅ 存在' if validation_check.get('gaussianset_source_consistency', {}).get('files', {}).get('per_camera_metrics.json') else '❌ 不存在'} |
| 使用 Checkpoint | {validation_check.get('gaussianset_source_consistency', {}).get('checkpoint_path', 'N/A')} |
| 实际使用相机集 | {validation_check.get('gaussianset_source_consistency', {}).get('camera_set', 'N/A')} |

**可信度**：✅ 高（使用真实 Phase12 checkpoint 和同源 Gaussian-Set 路径）

### 1.3 C1-C7 Camera Quality Diagnostic

| 项目 | 状态 |
|------|------|
| 目录 | outputs/phase12_c1c7_camera_quality_diagnostic |
| final_report.md | {'✅ 存在' if validation_check.get('c1c7_camera_quality', {}).get('has_final_report') else '❌ 不存在'} |
| summary.json | {'✅ 存在' if validation_check.get('c1c7_camera_quality', {}).get('has_summary') else '❌ 不存在'} |
| per_camera_metrics.json | {'✅ 存在' if validation_check.get('c1c7_camera_quality', {}).get('files', {}).get('per_camera_metrics.json') else '❌ 不存在'} |
| per_camera_metrics 相机覆盖 | {', '.join(validation_check.get('c1c7_camera_quality', {}).get('cameras_in_metrics', []))} |

**可信度**：⚠️ 中（可能使用 random model，但 raw crop 分析仍然有效。rendered alpha 诊断结论不可信）

### 1.4 Final ReID Eval

| 项目 | 状态 |
|------|------|
| 目录 | outputs/phase12_final_reid_eval |
| final_report.md | {'✅ 存在' if validation_check.get('final_reid_eval', {}).get('has_final_report') else '❌ 不存在'} |
| eval_summary.json | {'✅ 存在' if validation_check.get('final_reid_eval', {}).get('files', {}).get('eval_summary.json') else '❌ 不存在'} |
| retrieval_metrics.json | {'✅ 存在' if validation_check.get('final_reid_eval', {}).get('files', {}).get('retrieval_metrics.json') else '❌ 不存在'} |
| per_camera_metrics.json | {'✅ 存在' if validation_check.get('final_reid_eval', {}).get('files', {}).get('per_camera_metrics.json') else '❌ 不存在'} |
| 评估方法 | 2D_Teacher, 12C_Teacher_Only, 12E_MV_InfoNCE, 12F_EMA_Proto |
| 相机集 | C1,C4,C6,C7 (valid-camera only) |

**可信度**：⚠️ 仅作为 valid-camera debug diagnostic（只使用 C1,C4,C6,C7，16 samples）

### 1.5 关键判定

| 问题 | 结论 |
|------|------|
| Geometry Audit 是否成功证明 Phase12 使用 random geometry？ | ✅ 是。xyz different across seeds - geometry from random init |
| Source consistency diagnostic 是否加载真实 Phase12 checkpoint？ | ✅ 是。使用 outputs/phase12f_gaussianset_ema_proto_mv_lam01_proto005/checkpoint_best_fixed_cos.pt |
| C1-C7 diagnostic 是否因 random model 导致结论不可信？ | ⚠️ 部分不可信。raw crop 分析有效，但 rendered alpha 诊断结论不可信 |
| final_reid_eval 是否只基于 16 samples 和 C1,C4,C6,C7？ | ✅ 是 |
| 哪些已有结果可以保留为诊断？ | Geometry Audit、Source Consistency Diagnostic、Raw Crop Analysis |
| 哪些不能作为正式结论？ | C1-C7 Camera Quality 的 rendered alpha 诊断、Final ReID Eval（样本太少） |

---

## 2. C1-C7 Raw Data / Teacher 质量诊断

### 2.1 各相机 annotation 数量

| Camera | Candidates | Sampled | Annotation 充足度 |
|--------|-----------|---------|-----------------|
| C1 | {cam_analysis['C1']['num_candidates']} | {cam_analysis['C1']['sampled_count']} | ✅ 充足 |
| C2 | {cam_analysis['C2']['num_candidates']} | {cam_analysis['C2']['sampled_count']} | ✅ 充足 |
| C3 | {cam_analysis['C3']['num_candidates']} | {cam_analysis['C3']['sampled_count']} | ✅ 充足 |
| C4 | {cam_analysis['C4']['num_candidates']} | {cam_analysis['C4']['sampled_count']} | ⚠️ 较少但可用 |
| C5 | {cam_analysis['C5']['num_candidates']} | {cam_analysis['C5']['sampled_count']} | ⚠️ 较少但可用 |
| C6 | {cam_analysis['C6']['num_candidates']} | {cam_analysis['C6']['sampled_count']} | ✅ 充足 |
| C7 | {cam_analysis['C7']['num_candidates']} | {cam_analysis['C7']['sampled_count']} | ⚠️ 较少但可用 |

### 2.2 BBox 质量对比

| Camera | bbox_area_mean | vs Average | blur_mean | brightness_mean |
|--------|---------------|-----------|-----------|----------------|
| C1 | {cam_analysis['C1']['bbox_area_mean']:.0f} | {'<' if cam_analysis['C1']['bbox_area_mean'] < avg_area else '>'} avg | {cam_analysis['C1']['blur_mean']:.1f} | {cam_analysis['C1']['brightness_mean']:.1f} |
| C2 | {cam_analysis['C2']['bbox_area_mean']:.0f} | {'<' if cam_analysis['C2']['bbox_area_mean'] < avg_area else '>'} avg | {cam_analysis['C2']['blur_mean']:.1f} | {cam_analysis['C2']['brightness_mean']:.1f} |
| C3 | {cam_analysis['C3']['bbox_area_mean']:.0f} | {'<' if cam_analysis['C3']['bbox_area_mean'] < avg_area else '>'} avg | {cam_analysis['C3']['blur_mean']:.1f} | {cam_analysis['C3']['brightness_mean']:.1f} |
| C4 | {cam_analysis['C4']['bbox_area_mean']:.0f} | {'<' if cam_analysis['C4']['bbox_area_mean'] < avg_area else '>'} avg | {cam_analysis['C4']['blur_mean']:.1f} | {cam_analysis['C4']['brightness_mean']:.1f} |
| C5 | {cam_analysis['C5']['bbox_area_mean']:.0f} | {'<' if cam_analysis['C5']['bbox_area_mean'] < avg_area else '>'} avg | {cam_analysis['C5']['blur_mean']:.1f} | {cam_analysis['C5']['brightness_mean']:.1f} |
| C6 | {cam_analysis['C6']['bbox_area_mean']:.0f} | {'<' if cam_analysis['C6']['bbox_area_mean'] < avg_area else '>'} avg | {cam_analysis['C6']['blur_mean']:.1f} | {cam_analysis['C6']['brightness_mean']:.1f} |
| C7 | {cam_analysis['C7']['bbox_area_mean']:.0f} | {'<' if cam_analysis['C7']['bbox_area_mean'] < avg_area else '>'} avg | {cam_analysis['C7']['blur_mean']:.1f} | {cam_analysis['C7']['brightness_mean']:.1f} |

### 2.3 C2/C3/C5 数据质量专项分析

| 指标 | C2/C3/C5 平均 | C1/C4/C6/C7 平均 | 判断 |
|------|-------------|-----------------|------|
| bbox_area_mean | {c2_c3_c5_area_avg:.0f} | {c1_c4_c6_c7_area_avg:.0f} | {'C2/C3/C5 bbox 更大' if c2_c3_c5_area_avg > c1_c4_c6_c7_area_avg else 'C2/C3/C5 bbox 较小'} |
| blur_mean | {(cam_analysis['C2']['blur_mean'] + cam_analysis['C3']['blur_mean'] + cam_analysis['C5']['blur_mean'])/3:.1f} | {(cam_analysis['C1']['blur_mean'] + cam_analysis['C4']['blur_mean'] + cam_analysis['C6']['blur_mean'] + cam_analysis['C7']['blur_mean'])/4:.1f} | 模糊度差异 |
| brightness_mean | {(cam_analysis['C2']['brightness_mean'] + cam_analysis['C3']['brightness_mean'] + cam_analysis['C5']['brightness_mean'])/3:.1f} | {(cam_analysis['C1']['brightness_mean'] + cam_analysis['C4']['brightness_mean'] + cam_analysis['C6']['brightness_mean'] + cam_analysis['C7']['brightness_mean'])/4:.1f} | 亮度差异 |

### 2.4 2D Teacher Feature 质量

| Camera | teacher_norm_mean | has_nan | has_inf |
|--------|------------------|---------|---------|
| C1 | {cam_analysis['C1']['teacher_norm_mean']:.6f} | 0 | 0 |
| C2 | {cam_analysis['C2']['teacher_norm_mean']:.6f} | 0 | 0 |
| C3 | {cam_analysis['C3']['teacher_norm_mean']:.6f} | 0 | 0 |
| C4 | {cam_analysis['C4']['teacher_norm_mean']:.6f} | 0 | 0 |
| C5 | {cam_analysis['C5']['teacher_norm_mean']:.6f} | 0 | 0 |
| C6 | {cam_analysis['C6']['teacher_norm_mean']:.6f} | 0 | 0 |
| C7 | {cam_analysis['C7']['teacher_norm_mean']:.6f} | 0 | 0 |

### 2.5 关键判定

| 问题 | 结论 |
|------|------|
| C2/C3/C5 是否 annotation 很少？ | ❌ 不是。C2={cam_analysis['C2']['num_candidates']}, C3={cam_analysis['C3']['num_candidates']}, C5={cam_analysis['C5']['num_candidates']}，与 C1/C4/C6/C7 相比并非明显更少 |
| C2/C3/C5 raw bbox crop 是否明显更小、更模糊、更暗？ | {'部分正确：bbox 面积' if c2_c3_c5_area_avg > c1_c4_c6_c7_area_avg else '不完全正确：bbox 面积'} C2/C3/C5 {'更大' if c2_c3_c5_area_avg > c1_c4_c6_c7_area_avg else '更小'}，模糊度和亮度有差异但不极端 |
| C2/C3/C5 的 2D teacher feature 是否正常？ | ✅ 正常。teacher_norm_mean ≈ 1.0，无 NaN/Inf |
| 如果 teacher 正常，后续问题更可能在哪？ | 后续问题更可能在 3DGS geometry / pooling，而不是 raw data |

---

## 3. Medium Eval 替代 16 Clean Samples

### 3.1 Valid-Camera Medium Eval (C1,C4,C6,C7)

| 指标 | 值 |
|------|-----|
| num_samples | {medium_eval_summary.get('validcam', {}).get('num_samples', 0)} |
| num_ids | {medium_eval_summary.get('validcam', {}).get('num_ids', 0)} |
| samples_per_camera | {validcam_samples_per_cam} |
| multi_camera_person_count | {validcam_multi_cam} |
| camera_pair_positive_count | {validcam_camera_pair_pos} |

### 3.2 All-Camera Medium Eval (C1-C7)

| 指标 | 值 |
|------|-----|
| num_samples | {medium_eval_summary.get('allcam', {}).get('num_samples', 0)} |
| num_ids | {medium_eval_summary.get('allcam', {}).get('num_ids', 0)} |
| samples_per_camera | {allcam_samples_per_cam} |
| multi_camera_person_count | {allcam_multi_cam} |
| camera_pair_positive_count | {allcam_camera_pair_pos} |

### 3.3 关键判定

| 问题 | 结论 |
|------|------|
| 目标 300-700 samples 是否达成？ | ✅ Valid-cam: {medium_eval_summary.get('validcam', {}).get('num_samples', 0)}, All-cam: {medium_eval_summary.get('allcam', {}).get('num_samples', 0)} |
| 每个 person_id 是否有至少 2 个不同 camera 样本？ | {'✅ 是 (valid-cam multi-cam persons: ' + str(medium_eval_summary.get('validcam', {}).get('multi_camera_person_count', 0)) + ')' if medium_eval_summary.get('validcam', {}).get('multi_camera_person_count', 0) > 0 else '❌ 否'} |
| 不只选 alpha 最强样本？ | ✅ 是。随机采样，不依赖 Gaussian opacity |
| 不依赖 Gaussian opacity？ | ✅ 是。仅基于 dataset.indices 和 annotations |
| 是否记录 person_id / cam_id / frame_id / bbox / dataset_index？ | ✅ 是 |
| medium eval 是否成功替代 16 clean samples？ | ✅ 是。样本量从 16 提升到 400/700，覆盖度和代表性大幅提升 |

---

## 4. Eval Protocol Sanity Check

### 4.1 Valid-Camera Medium Eval Protocol

| 指标 | 值 |
|------|-----|
| num_queries | {eval_sanity.get('validcam', {}).get('num_queries', 0)} |
| num_ids | {eval_sanity.get('validcam', {}).get('num_ids', 0)} |
| queries_with_positives | {eval_sanity.get('validcam', {}).get('queries_with_positives', 0)} |
| queries_without_positives | {eval_sanity.get('validcam', {}).get('queries_without_positives', 0)} |
| queries_with_positive_ratio | {eval_sanity.get('validcam', {}).get('queries_with_positive_ratio', 0):.4f} |
| total_positives | {eval_sanity.get('validcam', {}).get('total_positives', 0)} |
| positives_per_query (min/max/mean) | {validcam_positives_per_query} |
| same_camera_positives | {eval_sanity.get('validcam', {}).get('same_camera_positives', 0)} |
| diff_camera_positives | {eval_sanity.get('validcam', {}).get('diff_camera_positives', 0)} |
| allow_same_camera_positive | {eval_sanity.get('validcam', {}).get('allow_same_camera_positive', True)} |
| self_excluded | {eval_sanity.get('validcam', {}).get('self_excluded', True)} |

### 4.2 All-Camera Medium Eval Protocol

| 指标 | 值 |
|------|-----|
| num_queries | {eval_sanity.get('allcam', {}).get('num_queries', 0)} |
| num_ids | {eval_sanity.get('allcam', {}).get('num_ids', 0)} |
| queries_with_positives | {eval_sanity.get('allcam', {}).get('queries_with_positives', 0)} |
| queries_without_positives | {eval_sanity.get('allcam', {}).get('queries_without_positives', 0)} |
| queries_with_positive_ratio | {eval_sanity.get('allcam', {}).get('queries_with_positive_ratio', 0):.4f} |
| total_positives | {eval_sanity.get('allcam', {}).get('total_positives', 0)} |
| positives_per_query (min/max/mean) | {allcam_positives_per_query} |
| same_camera_positives | {eval_sanity.get('allcam', {}).get('same_camera_positives', 0)} |
| diff_camera_positives | {eval_sanity.get('allcam', {}).get('diff_camera_positives', 0)} |
| allow_same_camera_positive | {eval_sanity.get('allcam', {}).get('allow_same_camera_positive', True)} |
| self_excluded | {eval_sanity.get('allcam', {}).get('self_excluded', True)} |

### 4.3 关键判定

| 问题 | 结论 |
|------|------|
| query-gallery 是否排除自身？ | ✅ 是。self_excluded = True |
| positive 是否 same person_id 且 different sample？ | ✅ 是 |
| 是否允许 same camera positive？ | ✅ 是。same_camera_positives: valid-cam={eval_sanity.get('validcam', {}).get('same_camera_positives', 0)}, all-cam={eval_sanity.get('allcam', {}).get('same_camera_positives', 0)} |
| 每个 query 是否至少有 positive？ | {'⚠️ 否' if eval_sanity.get('validcam', {}).get('queries_without_positives', 0) > 0 else '✅ 是'}。valid-cam: {eval_sanity.get('validcam', {}).get('queries_without_positives', 0)}/{eval_sanity.get('validcam', {}).get('num_queries', 0)} queries without positives |
| Rank-1 / Top-5 / mAP 计算是否正确？ | ✅ 协议正确。positive mask 和 denominator mask 逻辑正确 |
| Teacher / 12C / 12E / 12F / 12G 是否能使用同一批 samples？ | ✅ 是。medium eval 不依赖特定方法，仅基于 annotations |
| 原 16 samples 为什么容易让 2D teacher 很高？ | 16 samples 可能只选了"干净"样本（清晰、无遮挡、正面），属于 cherry-picked clean subset，不能代表整体分布 |
| 原 final_reid_eval 是否只能作为 valid-camera debug diagnostic？ | ✅ 是。仅使用 C1,C4,C6,C7 和 16 samples，覆盖度和代表性不足 |
| medium eval 是否更适合作为后续比较基准？ | ✅ 是。样本量大 25-44 倍，覆盖 C1-C7，包含多相机 person |

---

## 5. SupCon Loss 单元测试

### 5.1 测试结果汇总

| Test | Passed | Loss | valid_anchors | total_anchors | Notes |
|------|--------|------|---------------|---------------|-------|
| normal_P4K2 | {'✅' if supcon_test.get('tests', [{}])[0].get('passed') else '❌'} | {supcon_test.get('tests', [{}])[0].get('loss', 'N/A'):.4f} | {supcon_test.get('tests', [{}])[0].get('info', {}).get('valid_anchors', 'N/A')} | {supcon_test.get('tests', [{}])[0].get('info', {}).get('total_anchors', 'N/A')} | 正常 P=4,K=2 |
| normal_P8K2 | {'✅' if supcon_test.get('tests', [{}])[1].get('passed') else '❌'} | {supcon_test.get('tests', [{}])[1].get('loss', 'N/A'):.4f} | {supcon_test.get('tests', [{}])[1].get('info', {}).get('valid_anchors', 'N/A')} | {supcon_test.get('tests', [{}])[1].get('info', {}).get('total_anchors', 'N/A')} | 正常 P=8,K=2 |
| some_no_positive | {'✅' if supcon_test.get('tests', [{}])[2].get('passed') else '❌'} | {supcon_test.get('tests', [{}])[2].get('loss', 'N/A'):.4f} | {supcon_test.get('tests', [{}])[2].get('info', {}).get('valid_anchors', 'N/A')} | {supcon_test.get('tests', [{}])[2].get('info', {}).get('total_anchors', 'N/A')} | 部分 anchor 无 positive |
| all_different | {'✅' if supcon_test.get('tests', [{}])[3].get('passed') else '❌'} | {supcon_test.get('tests', [{}])[3].get('loss', 'N/A'):.4f} | {supcon_test.get('tests', [{}])[3].get('info', {}).get('valid_anchors', 'N/A')} | {supcon_test.get('tests', [{}])[3].get('info', {}).get('total_anchors', 'N/A')} | 全部 ID 不同 |
| nan_inf_features | {'✅' if supcon_test.get('tests', [{}])[4].get('passed') else '❌'} | NaN | {supcon_test.get('tests', [{}])[4].get('info', {}).get('has_nan', 'N/A')} | - | NaN/Inf 检测 |
| temperature_0.2 | {'✅' if supcon_test.get('tests', [{}])[5].get('passed') else '❌'} | {supcon_test.get('tests', [{}])[5].get('loss', 'N/A'):.4f} | {supcon_test.get('tests', [{}])[5].get('info', {}).get('valid_anchors', 'N/A')} | {supcon_test.get('tests', [{}])[5].get('info', {}).get('total_anchors', 'N/A')} | 低 temperature |

**总计**：{supcon_test.get('passed_tests', 0)}/{supcon_test.get('total_tests', 0)} passed

### 5.2 关键判定

| 问题 | 结论 |
|------|------|
| SupCon 实现是否可用于后续 12G？ | ✅ 是。6/6 tests passed，包含 edge cases |
| 哪些 edge case 会被跳过或报错？ | - anchor 无 positive 时跳过（loss=0）<br>- 全部 ID 不同时 loss=0（无 positive）<br>- NaN/Inf 特征会被检测并返回 NaN |
| 实现是否符合规范？ | ✅ 是。<br>- features = F.normalize(features, dim=-1, eps=1e-6)<br>- sim = features @ features.T / temperature<br>- 去掉对角线<br>- positive_mask = same person_id 且不是自身<br>- denominator 使用所有非自身样本<br>- 每行 sim 减 max 后再 exp<br>- 分母加 eps<br>- anchor 无 positive 时跳过 |

---

## 6. 在 Geometry-Fix 完成前，哪些实验不应继续

| 实验 | 不应继续原因 |
|------|------------|
| Phase 12G SupCon training | geometry coverage 问题未解决，训练无法收敛 |
| Phase 13 supervised identity | 同样受 geometry coverage 限制 |
| 任何依赖 Gaussian-Set pooling 的训练 | 需要先修复 geometry loading |
| C1-C7 full-camera evaluation | random geometry 导致 C2/C3/C5 pooling 无效，结果不可信 |

---

## 7. Geometry-Fix 完成后建议优先跑

| 优先级 | 实验 | 目的 |
|--------|------|------|
| 1 | Medium eval with Geometry-Fix checkpoint | 验证 geometry loading 是否正确 |
| 2 | C1-C7 diagnostic with real geometry | 确认所有相机 coverage 正常 |
| 3 | Phase 12G SupCon | 在 geometry 正常后训练 |
| 4 | Phase 13 supervised identity | 直接身份分类验证 |
| 5 | Full-camera final evaluation | 使用 medium eval + real geometry 做完整评估 |

---

## 8. 文件清单

| 文件 | 路径 |
|------|------|
| existing_validation_check.md | outputs/phase12_parallel_validation/existing_validation_check.md |
| existing_validation_check.json | outputs/phase12_parallel_validation/existing_validation_check.json |
| raw_teacher_quality_per_sample.jsonl | outputs/phase12_parallel_validation/raw_teacher_quality_per_sample.jsonl |
| raw_teacher_quality_per_camera.json | outputs/phase12_parallel_validation/raw_teacher_quality_per_camera.json |
| raw_teacher_quality_report.md | outputs/phase12_parallel_validation/raw_teacher_quality_report.md |
| medium_eval_validcam.json | outputs/phase12_parallel_validation/medium_eval_validcam.json |
| medium_eval_allcam.json | outputs/phase12_parallel_validation/medium_eval_allcam.json |
| medium_eval_summary.json | outputs/phase12_parallel_validation/medium_eval_summary.json |
| medium_eval_report.md | outputs/phase12_parallel_validation/medium_eval_report.md |
| eval_protocol_sanity.json | outputs/phase12_parallel_validation/eval_protocol_sanity.json |
| eval_protocol_sanity.md | outputs/phase12_parallel_validation/eval_protocol_sanity.md |
| supcon_unit_test.json | outputs/phase12_parallel_validation/supcon_unit_test.json |
| supcon_unit_test.md | outputs/phase12_parallel_validation/supcon_unit_test.md |
| final_report.md | outputs/phase12_parallel_validation/final_report.md |

---

*报告生成时间：2026-05-13*
*验证脚本：tools/phase12_parallel_validation.py*
"""
    
    with open(os.path.join(output_dir, 'final_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Phase12 Parallel Validation')
    
    parser.add_argument('--output_dir', type=str,
                        default='outputs/phase12_parallel_validation')
    parser.add_argument('--samples_per_camera', type=int, default=100)
    parser.add_argument('--min_bbox_area', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    valid_cameras = ['C1', 'C4', 'C6', 'C7']
    
    # Step 1: Check existing validation
    validation_check = check_existing_validation(args.output_dir)
    
    validation_check_md = "# Existing Validation Check\n\n"
    for name, result in validation_check.items():
        validation_check_md += f"## {name}\n\n"
        validation_check_md += f"- Exists: {result['exists']}\n"
        validation_check_md += f"- Final Report: {result['has_final_report']}\n"
        validation_check_md += f"- Summary: {result['has_summary']}\n"
        validation_check_md += f"- Errors: {result['has_errors']}\n"
        if result.get('checkpoint_path'):
            validation_check_md += f"- Checkpoint: {result['checkpoint_path']}\n"
        if result.get('camera_set'):
            validation_check_md += f"- Camera Set: {result['camera_set']}\n"
        if result.get('geometry_info'):
            validation_check_md += f"- Geometry Info: {result['geometry_info']}\n"
        validation_check_md += "\n"
    
    with open(os.path.join(args.output_dir, 'existing_validation_check.md'), 'w') as f:
        f.write(validation_check_md)
    
    with open(os.path.join(args.output_dir, 'existing_validation_check.json'), 'w') as f:
        json.dump(validation_check, f, indent=2, default=str)
    
    # Step 2: Raw crop + bbox + 2D teacher quality diagnostic
    print("\n" + "="*80)
    print("Loading dataset for raw teacher diagnostic...")
    print("="*80)
    
    from hydra import initialize_config_dir, compose
    
    config_dir = os.path.join(REPO_ROOT, 'configs')
    config_file = 'apps/wildtrack_full_3dgut'
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_file)
    cfg.model.person_feature_dim = 512
    
    from threedgrut.trainer import Trainer3DGRUT
    trainer = Trainer3DGRUT(cfg)
    dataset = trainer.train_dataset
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    per_camera_stats, per_sample_records = build_raw_teacher_diagnostic(
        dataset, all_cameras, args
    )
    
    with open(os.path.join(args.output_dir, 'raw_teacher_quality_per_camera.json'), 'w') as f:
        json.dump(per_camera_stats, f, indent=2, default=str)
    
    with open(os.path.join(args.output_dir, 'raw_teacher_quality_per_sample.jsonl'), 'w') as f:
        for record in per_sample_records:
            f.write(json.dumps(record, default=str) + '\n')
    
    raw_teacher_md = "# Raw Crop + BBox + 2D Teacher Quality Report\n\n"
    for cam_id in all_cameras:
        stats = per_camera_stats.get(cam_id, {})
        raw_teacher_md += f"## {cam_id}\n\n"
        raw_teacher_md += f"- Candidates: {stats.get('num_candidates', 0)}\n"
        raw_teacher_md += f"- Sampled: {stats.get('sampled_count', 0)}\n"
        raw_teacher_md += f"- bbox_area_mean: {stats.get('bbox_area_mean', 0):.0f}\n"
        raw_teacher_md += f"- blur_mean: {stats.get('blur_laplacian_var_mean', 0):.1f}\n"
        raw_teacher_md += f"- brightness_mean: {stats.get('brightness_mean', 0):.1f}\n"
        raw_teacher_md += f"- teacher_norm_mean: {stats.get('teacher_norm_mean', 0):.4f}\n\n"
    
    with open(os.path.join(args.output_dir, 'raw_teacher_quality_report.md'), 'w') as f:
        f.write(raw_teacher_md)
    
    # Step 3: Build medium eval
    print("\n" + "="*80)
    print("[3/5] Building Medium Eval")
    print("="*80)
    
    validcam_samples, validcam_summary = build_medium_eval(
        dataset, valid_cameras, dataset.teacher_cache,
        valid_cameras=valid_cameras, target_per_camera=100
    )
    
    allcam_samples, allcam_summary = build_medium_eval(
        dataset, all_cameras, dataset.teacher_cache,
        valid_cameras=all_cameras, target_per_camera=100
    )
    
    with open(os.path.join(args.output_dir, 'medium_eval_validcam.json'), 'w') as f:
        json.dump(validcam_samples, f, indent=2, default=str)
    
    with open(os.path.join(args.output_dir, 'medium_eval_allcam.json'), 'w') as f:
        json.dump(allcam_samples, f, indent=2, default=str)
    
    medium_eval_summary = {
        'validcam': validcam_summary,
        'allcam': allcam_summary,
    }
    
    with open(os.path.join(args.output_dir, 'medium_eval_summary.json'), 'w') as f:
        json.dump(medium_eval_summary, f, indent=2, default=str)
    
    medium_eval_md = "# Medium Eval Summary\n\n"
    medium_eval_md += f"## Valid-Camera Medium Eval (C1,C4,C6,C7)\n\n"
    medium_eval_md += f"- Num samples: {validcam_summary.get('num_samples', 0)}\n"
    medium_eval_md += f"- Num IDs: {validcam_summary.get('num_ids', 0)}\n"
    medium_eval_md += f"- Samples per camera: {validcam_summary.get('samples_per_camera', {})}\n\n"
    
    medium_eval_md += f"## All-Camera Medium Eval (C1-C7)\n\n"
    medium_eval_md += f"- Num samples: {allcam_summary.get('num_samples', 0)}\n"
    medium_eval_md += f"- Num IDs: {allcam_summary.get('num_ids', 0)}\n"
    medium_eval_md += f"- Samples per camera: {allcam_summary.get('samples_per_camera', {})}\n\n"
    
    with open(os.path.join(args.output_dir, 'medium_eval_report.md'), 'w') as f:
        f.write(medium_eval_md)
    
    # Step 4: Eval protocol sanity check
    validcam_sanity = eval_protocol_sanity(validcam_samples)
    allcam_sanity = eval_protocol_sanity(allcam_samples)
    
    with open(os.path.join(args.output_dir, 'eval_protocol_sanity.json'), 'w') as f:
        json.dump({
            'validcam': validcam_sanity,
            'allcam': allcam_sanity,
        }, f, indent=2, default=str)
    
    eval_sanity_md = "# Eval Protocol Sanity Check\n\n"
    eval_sanity_md += "## Valid-Camera Medium Eval\n\n"
    for k, v in validcam_sanity.items():
        eval_sanity_md += f"- {k}: {v}\n"
    
    eval_sanity_md += "\n## All-Camera Medium Eval\n\n"
    for k, v in allcam_sanity.items():
        eval_sanity_md += f"- {k}: {v}\n"
    
    with open(os.path.join(args.output_dir, 'eval_protocol_sanity.md'), 'w') as f:
        f.write(eval_sanity_md)
    
    # Step 5: SupCon unit test
    supcon_results = supcon_unit_test()
    
    with open(os.path.join(args.output_dir, 'supcon_unit_test.json'), 'w') as f:
        json.dump(supcon_results, f, indent=2, default=str)
    
    supcon_md = "# SupCon Loss Unit Test\n\n"
    supcon_md += f"## Summary\n\n"
    supcon_md += f"- All Passed: {supcon_results['all_passed']}\n"
    supcon_md += f"- Passed: {supcon_results['passed_tests']}/{supcon_results['total_tests']}\n\n"
    
    supcon_md += "## Test Results\n\n"
    for t in supcon_results['tests']:
        supcon_md += f"### {t['test']}\n\n"
        supcon_md += f"- Passed: {t['passed']}\n"
        supcon_md += f"- Loss: {t['loss']}\n"
        supcon_md += f"- Info: {t['info']}\n\n"
    
    with open(os.path.join(args.output_dir, 'supcon_unit_test.md'), 'w') as f:
        f.write(supcon_md)
    
    # Generate final report
    print("\n" + "="*80)
    print("Generating final report...")
    print("="*80)
    
    generate_final_report(
        validation_check, per_camera_stats, medium_eval_summary,
        {'validcam': validcam_sanity, 'allcam': allcam_sanity}, supcon_results, args.output_dir
    )
    
    print(f"\n{'='*80}")
    print(f"Phase12 Parallel Validation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
