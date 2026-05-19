#!/usr/bin/env python3
"""
Phase 12B: Gaussian-Set Progressive Overfit

Test Gaussian-Set pooling from 1 ROI to 2/4/8 ROIs.
Uses direct Gaussian xyz projection with bbox cropping (Phase 12A method).

Goal: Verify that Gaussian-Set can stably overfit with multiple ROIs and observe cross-view gap.
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
    
    This directly projects Gaussian centers to image plane, filters by bbox,
    and pools person_feature weighted by opacity.
    Same approach as Phase 12A which achieved cosine 0.07->1.0 with 19 Gaussians.
    
    Returns: G (pooled feature), debug_info
    """
    x1, y1, x2, y2 = bbox
    
    try:
        # 1. Read Gaussian parameters
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
        
        # 2. Get intrinsics from batch
        intrinsics = gpu_batch.intrinsics  # [fx, fy, cx, cy]
        if intrinsics is None or len(intrinsics) < 4:
            return None, {
                'num_gaussians_in_bbox': 0,
                'weight_sum': 0.0,
                'failure_reason': 'no_intrinsics',
            }
        
        fx, fy, cx, cy = intrinsics
        
        # 3. Get extrinsics from batch T_to_world [B, 4, 4]
        # T_to_world transforms from camera to world
        # We need world to camera for projection
        T_to_world = gpu_batch.T_to_world[0]  # [4, 4]
        R_world_to_cam = T_to_world[:3, :3].t()  # [3, 3]
        t_world_to_cam = -R_world_to_cam @ T_to_world[:3, 3]  # [3]
        
        # 4. Project Gaussian xyz to image plane
        # x_cam = R @ x_world + t
        xyz_cam = (R_world_to_cam @ xyz.t()).t() + t_world_to_cam  # [N, 3]
        
        # depth > 0
        depth = xyz_cam[:, 2]  # [N]
        valid_depth = depth > 0
        
        # 5. Project to pixel coordinates
        x_img = fx * xyz_cam[:, 0] / xyz_cam[:, 2] + cx  # [N]
        y_img = fy * xyz_cam[:, 1] / xyz_cam[:, 2] + cy  # [N]
        
        # Get image dimensions from rendered feature map
        h_img = gpu_batch.rays_dir.shape[1]
        w_img = gpu_batch.rays_dir.shape[2]
        
        # 6. Filter: depth>0, x/y finite, x/y in image bounds
        x_finite = torch.isfinite(x_img)
        y_finite = torch.isfinite(y_img)
        x_in_bounds = (x_img >= 0) & (x_img < w_img)
        y_in_bounds = (y_img >= 0) & (y_img < h_img)
        opacity_positive = opacity > 0
        
        valid = valid_depth & x_finite & y_finite & x_in_bounds & y_in_bounds & opacity_positive
        
        # 7. Select Gaussians whose projection center falls within bbox
        inside_bbox = (x_img >= x1) & (x_img < x2) & (y_img >= y1) & (y_img < y2)
        inside = valid & inside_bbox
        
        if inside.sum() == 0:
            return None, {
                'num_gaussians_in_bbox': 0,
                'weight_sum': 0.0,
                'failure_reason': 'no_gaussians_in_bbox',
            }
        
        # 8. Pool: G = sum(weights[:,None] * z) / (sum(weights)+denom_eps)
        weights = opacity[inside]  # [M]
        z = person_feature[inside]  # [M, D]
        
        weight_sum = weights.sum()
        denom = weight_sum.clamp(min=args.denom_eps)
        weighted_sum = (weights[:, None] * z).sum(dim=0)  # [D]
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
        import traceback as tb
        return None, {
            'num_gaussians_in_bbox': 0,
            'weight_sum': 0.0,
            'failure_reason': f'{str(e)[:60]}',
        }


class PkMultiViewSampler:
    def __init__(self, dataset, num_person=4, num_views=2, max_retries=50):
        self.dataset = dataset
        self.num_person = num_person
        self.num_views = num_views
        self.max_retries = max_retries
        self.valid_timestamps = []
        self.valid_persons = set()
        self.timestamp_to_persons = defaultdict(set)
        self.person_cam_at_ts = defaultdict(lambda: defaultdict(set))
        self.cam_frame_to_index = {}

        for idx in range(len(dataset)):
            ds_cam_id, frame_idx = dataset.indices[idx]
            anns = dataset.annotations.get(int(frame_idx), [])
            for ann in anns:
                ann_cam_id = ann.get('camera_id')
                if ann_cam_id is None:
                    continue
                cam_id_str = f"C{ann_cam_id + 1}"
                self.cam_frame_to_index[(cam_id_str, int(frame_idx))] = idx
                pid = ann.get('new_id')
                if pid is None:
                    continue
                self.timestamp_to_persons[int(frame_idx)].add(pid)
                self.person_cam_at_ts[pid][int(frame_idx)].add(cam_id_str)

        for ts in sorted(self.timestamp_to_persons.keys()):
            pids = self.timestamp_to_persons[ts]
            valid_at_ts = [
                pid for pid in pids
                if len(self.person_cam_at_ts[pid].get(ts, set())) >= num_views
            ]
            if len(valid_at_ts) >= num_person:
                self.valid_timestamps.append(ts)
                self.valid_persons.update(valid_at_ts)

        print(f"[PkMultiViewSampler] {len(self.valid_timestamps)} valid timestamps, "
              f"{len(self.valid_persons)} valid persons, {len(self.cam_frame_to_index)} entries")

    def sample_batch(self):
        for _ in range(self.max_retries):
            if not self.valid_timestamps:
                return None
            ts = random.choice(self.valid_timestamps)
            available = list(self.timestamp_to_persons[ts])
            if len(available) < self.num_person:
                continue
            chosen_pids = random.sample(available, self.num_person)
            mv_samples = []
            valid = True
            for pid in chosen_pids:
                cams = sorted(self.person_cam_at_ts[pid].get(ts, set()))
                if len(cams) < self.num_views:
                    valid = False
                    break
                sel_cams = random.sample(cams, self.num_views)
                mv_samples.append({
                    'person_id': pid,
                    'views': [(c, ts) for c in sel_cams],
                })
            if valid and len(mv_samples) == self.num_person:
                return mv_samples
        return None

    def get_dataset_index(self, cam_id, frame_idx):
        return self.cam_frame_to_index.get((cam_id, int(frame_idx)))


def select_clean_rois(args, trainer, sampler, batch_builder, allowed_cameras=None, roi_count=1, same_person=False):
    """Select ROIs for Phase 12B overfitting."""
    print(f"\nSelecting {roi_count} ROIs...")
    
    selected_samples = []
    attempts = 0
    max_attempts = roi_count * 300
    
    used_person_cam_frames = set()
    
    while len(selected_samples) < roi_count and attempts < max_attempts:
        attempts += 1
        
        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue
        
        for ps in mv_samples:
            if len(selected_samples) >= roi_count:
                break
            
            if same_person and len(selected_samples) > 0:
                first_pid = selected_samples[0]['person_id']
                if ps['person_id'] != first_pid:
                    continue
            
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                if len(selected_samples) >= roi_count:
                    break
                
                if allowed_cameras and cam_id not in allowed_cameras:
                    continue
                
                pid_key = (pid, cam_id, int(frame_idx))
                if pid_key in used_person_cam_frames:
                    continue
                
                gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, int(frame_idx))
                if gpu_batch is None:
                    continue
                
                inst = None
                for i in gpu_batch.instances:
                    if i.get('train_id') == pid:
                        inst = i
                        break
                if inst is None:
                    continue
                
                teacher_emb = inst.get('teacher_embedding')
                if teacher_emb is None:
                    continue
                
                bbox = inst['bbox_xyxy']
                if bbox is None:
                    continue
                
                try:
                    if hasattr(bbox, 'tolist'):
                        bbox = bbox.tolist()
                    if len(bbox) < 4:
                        continue
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                except Exception:
                    continue
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area < args.min_bbox_area:
                    continue
                
                G, debug_info = gaussian_set_pooling(
                    trainer.model, gpu_batch, [x1, y1, x2, y2], cam_id, int(frame_idx), args, trainer.device
                )
                
                if G is None or debug_info['num_gaussians_in_bbox'] == 0:
                    continue
                
                selected_samples.append({
                    'person_id': pid,
                    'train_id': pid,
                    'cam_id': cam_id,
                    'frame_idx': int(frame_idx),
                    'bbox': [x1, y1, x2, y2],
                    'bbox_area': bbox_area,
                    'teacher_emb': teacher_emb.cpu().numpy().tolist() if hasattr(teacher_emb, 'cpu') else teacher_emb,
                    'num_gaussians_in_bbox': debug_info['num_gaussians_in_bbox'],
                    'weight_sum': debug_info['weight_sum'],
                })
                
                used_person_cam_frames.add(pid_key)
                
                if len(selected_samples) >= roi_count:
                    break
    
    if len(selected_samples) < roi_count:
        print(f"WARNING: Only found {len(selected_samples)} candidates (requested {roi_count})")
    
    print(f"Selected {len(selected_samples)} ROIs from {attempts} attempts")
    
    unique_persons = set(s['person_id'] for s in selected_samples)
    person_count = len(unique_persons)
    camera_count = len(set(s['cam_id'] for s in selected_samples))
    print(f"  {person_count} unique persons, {camera_count} unique cameras")
    
    return selected_samples, person_count, camera_count


def run_phase12b(args, trainer, sampler, batch_builder, allowed_cameras=None):
    """Phase 12B: Gaussian-Set Progressive Overfit."""
    roi_count = args.progressive_roi_count
    same_person = getattr(args, 'same_person_cross_view', False)
    
    print(f"\n{'='*70}")
    print(f"PHASE 12B: GAUSSIAN-SET PROGRESSIVE OVERFIT")
    print(f"{'='*70}")
    print(f"ROI count: {roi_count}")
    print(f"Same person cross-view: {same_person}")
    print(f"Allowed cameras: {allowed_cameras}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    selected_samples, person_count, camera_count = select_clean_rois(
        args, trainer, sampler, batch_builder, allowed_cameras, roi_count, same_person
    )
    
    if len(selected_samples) == 0:
        print("ERROR: No valid ROIs selected")
        return False
    
    with open(os.path.join(args.output_dir, 'selected_samples.json'), 'w') as f:
        json.dump(selected_samples, f, indent=2, default=str)
    
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    metrics_log = []
    
    pf = trainer.model.get_person_feature()
    pf_before = pf.clone().detach()
    
    optimizer = torch.optim.Adam(
        [trainer.model._person_feature],
        lr=args.person_feature_lr,
    )
    
    best_cos_mean = -1.0
    best_step = 0
    total_nan = 0
    total_inf = 0
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {len(selected_samples)} ROIs, {person_count} persons, {camera_count} cameras")
    print(f"{'='*70}")
    
    for step in range(args.num_steps):
        step_start = time.time()
        
        optimizer.zero_grad()
        
        losses = []
        cos_values = []
        gaussianset_features = []
        num_gaussians_list = []
        weight_sum_list = []
        invalid_roi_count = 0
        
        for sample in selected_samples:
            gpu_batch = batch_builder.get_batch_by_cam_frame(sample['cam_id'], sample['frame_idx'])
            if gpu_batch is None:
                invalid_roi_count += 1
                continue
            
            G, debug_info = gaussian_set_pooling(
                trainer.model, gpu_batch, sample['bbox'], sample['cam_id'], sample['frame_idx'], args, trainer.device
            )
            
            if G is None:
                invalid_roi_count += 1
                continue
            
            T = torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=trainer.device)
            T = normalize_feat(T)
            
            cos_sim = F.cosine_similarity(G.unsqueeze(0), T.unsqueeze(0)).item()
            loss_i = 1.0 - F.cosine_similarity(G.unsqueeze(0), T.unsqueeze(0))
            
            gaussianset_features.append(G)
            losses.append(loss_i)
            cos_values.append(cos_sim)
            num_gaussians_list.append(debug_info['num_gaussians_in_bbox'])
            weight_sum_list.append(debug_info['weight_sum'])
        
        valid_roi_count = len(losses)
        if valid_roi_count == 0:
            print(f"Step {step}: WARNING - no valid ROIs")
            continue
        
        loss = torch.stack(losses).mean()
        loss.backward()
        
        grad_norm_before_clip = None
        grad_norm_after_clip = None
        if trainer.model._person_feature.grad is not None:
            grad_norm_before_clip = trainer.model._person_feature.grad.norm().item()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([trainer.model._person_feature], args.grad_clip_norm)
            grad_norm_after_clip = trainer.model._person_feature.grad.norm().item()
        
        optimizer.step()
        
        pf_after = trainer.model.get_person_feature()
        param_delta_tensor = pf_after - pf_before
        param_delta_norm = param_delta_tensor.norm().item()
        param_delta_max = param_delta_tensor.abs().max().item()
        pf_before = pf_after.clone().detach()
        
        cos_mean = float(np.mean(cos_values))
        cos_min = float(np.min(cos_values))
        cos_max = float(np.max(cos_values))
        
        same_cos = None
        diff_cos = None
        gap = None
        positive_pair_count = 0
        negative_pair_count = 0
        
        if person_count >= 2 and len(gaussianset_features) >= 2:
            same_cos_list = []
            diff_cos_list = []
            
            person_to_indices = {}
            for idx, s in enumerate(selected_samples[:len(gaussianset_features)]):
                pid = s['person_id']
                if pid not in person_to_indices:
                    person_to_indices[pid] = []
                person_to_indices[pid].append(idx)
            
            for pid, indices in person_to_indices.items():
                if len(indices) >= 2:
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            idx_i, idx_j = indices[i], indices[j]
                            if idx_i < len(gaussianset_features) and idx_j < len(gaussianset_features):
                                sc = F.cosine_similarity(
                                    gaussianset_features[idx_i].unsqueeze(0),
                                    gaussianset_features[idx_j].unsqueeze(0)
                                ).item()
                                same_cos_list.append(sc)
                                positive_pair_count += 1
            
            all_indices = list(range(len(gaussianset_features)))
            neg_checked = 0
            for _ in range(min(100, len(all_indices) * (len(all_indices) - 1) // 2)):
                i, j = random.sample(all_indices, 2)
                pid_i = selected_samples[i]['person_id']
                pid_j = selected_samples[j]['person_id']
                if pid_i != pid_j:
                    dc = F.cosine_similarity(
                        gaussianset_features[i].unsqueeze(0),
                        gaussianset_features[j].unsqueeze(0)
                    ).item()
                    diff_cos_list.append(dc)
                    negative_pair_count += 1
                    neg_checked += 1
                    if neg_checked >= 50:
                        break
            
            if same_cos_list:
                same_cos = float(np.mean(same_cos_list))
            if diff_cos_list:
                diff_cos = float(np.mean(diff_cos_list))
            if same_cos is not None and diff_cos is not None:
                gap = same_cos - diff_cos
        
        if cos_mean > best_cos_mean:
            best_cos_mean = cos_mean
            best_step = step
        
        nan_count = int(torch.isnan(trainer.model._person_feature).sum().item())
        inf_count = int(torch.isinf(trainer.model._person_feature).sum().item())
        total_nan += nan_count
        total_inf += inf_count
        
        step_time = time.time() - step_start
        
        if step % args.log_interval == 0:
            log_line = (f"[PHASE12B] Step {step:5d}: "
                       f"loss={loss.item():.4f} "
                       f"cos_mean={cos_mean:.4f} "
                       f"cos_min={cos_min:.4f} "
                       f"cos_max={cos_max:.4f} "
                       f"grad={grad_norm_before_clip:.4e}"
                       f"->{grad_norm_after_clip:.4e} "
                       f"delta={param_delta_norm:.6e} "
                       f"num_gauss={np.mean(num_gaussians_list):.0f} "
                       f"valid_roi={valid_roi_count}/{len(selected_samples)} "
                       f"t={step_time:.2f}s")
            print(log_line)
            if same_cos is not None:
                print(f"  same_cos={same_cos:.4f}, diff_cos={diff_cos:.4f}, gap={gap:.4f}")
        
        metrics_log.append({
            'step': step,
            'loss_teacher': float(loss.item()),
            'cos_mean': float(cos_mean),
            'cos_min': float(cos_min),
            'cos_max': float(cos_max),
            'per_roi_cos': cos_values,
            'num_gaussians_min': int(np.min(num_gaussians_list)) if num_gaussians_list else 0,
            'num_gaussians_mean': float(np.mean(num_gaussians_list)) if num_gaussians_list else 0,
            'num_gaussians_max': int(np.max(num_gaussians_list)) if num_gaussians_list else 0,
            'weight_sum_min': float(np.min(weight_sum_list)) if weight_sum_list else 0,
            'weight_sum_mean': float(np.mean(weight_sum_list)) if weight_sum_list else 0,
            'weight_sum_max': float(np.max(weight_sum_list)) if weight_sum_list else 0,
            'same_cos': same_cos,
            'diff_cos': diff_cos,
            'cross_view_gap': gap,
            'positive_pair_count': positive_pair_count,
            'negative_pair_count': negative_pair_count,
            'grad_norm_before_clip': float(grad_norm_before_clip) if grad_norm_before_clip is not None else 0,
            'grad_norm_after_clip': float(grad_norm_after_clip) if grad_norm_after_clip is not None else 0,
            'param_delta_norm': float(param_delta_norm),
            'param_delta_max': float(param_delta_max),
            'invalid_roi_count': invalid_roi_count,
            'nan_count': nan_count,
            'inf_count': inf_count,
        })
    
    with open(metrics_path, 'w') as f:
        for r in metrics_log:
            f.write(json.dumps(r, default=str) + "\n")
    
    first_cos = metrics_log[0]['cos_mean'] if metrics_log else 0
    last_cos = metrics_log[-1]['cos_mean'] if metrics_log else 0
    cos_delta = last_cos - first_cos
    
    first_loss = metrics_log[0]['loss_teacher'] if metrics_log else 0
    last_loss = metrics_log[-1]['loss_teacher'] if metrics_log else 0
    loss_delta = last_loss - first_loss
    
    first_gap = metrics_log[0]['cross_view_gap'] if metrics_log else None
    last_gap = metrics_log[-1]['cross_view_gap'] if metrics_log else None
    gap_delta = (last_gap - first_gap) if (first_gap is not None and last_gap is not None) else None
    
    param_deltas = [m['param_delta_norm'] for m in metrics_log if m['param_delta_norm'] > 0]
    
    per_roi_final_cos = []
    for s in selected_samples:
        idx = selected_samples.index(s)
        if idx < len(metrics_log[-1]['per_roi_cos']):
            per_roi_final_cos.append(metrics_log[-1]['per_roi_cos'][idx])
    
    if roi_count == 1:
        if last_cos > 0.99 and loss_delta < 0:
            verdict = "phase12b_1roi_success"
        elif cos_delta > 0.5:
            verdict = "phase12b_1roi_success"
        else:
            verdict = "phase12b_1roi_partial"
    elif roi_count == 2:
        if last_cos > 0.95 and (gap_delta is None or gap_delta > -0.1):
            verdict = "phase12b_2roi_crossview_success"
        else:
            verdict = "phase12b_2roi_partial"
    elif roi_count == 4:
        if last_cos > 0.90 and (last_gap is not None and last_gap > 0):
            verdict = "phase12b_4roi_multiid_success"
        else:
            verdict = "phase12b_4roi_partial"
    elif roi_count == 8:
        if last_cos > 0.85 and (gap_delta is not None and gap_delta > 0) and total_nan == 0:
            verdict = "phase12b_8roi_ready_for_random_training"
        else:
            verdict = "phase12b_8roi_partial"
    else:
        verdict = "phase12b_unknown"
    
    summary = {
        'roi_count': roi_count,
        'person_count': person_count,
        'camera_count': camera_count,
        'first_cos_mean': float(first_cos),
        'best_cos_mean': float(best_cos_mean),
        'last_cos_mean': float(last_cos),
        'cos_delta': float(cos_delta),
        'first_loss': float(first_loss),
        'last_loss': float(last_loss),
        'loss_delta': float(loss_delta),
        'first_gap': float(first_gap) if first_gap is not None else None,
        'last_gap': float(last_gap) if last_gap is not None else None,
        'gap_delta': float(gap_delta) if gap_delta is not None else None,
        'max_grad_norm': max(m['grad_norm_before_clip'] for m in metrics_log) if metrics_log else 0,
        'mean_param_delta_norm': float(np.mean(param_deltas)) if param_deltas else 0,
        'nan_total': total_nan,
        'inf_total': total_inf,
        'per_roi_final_cos': per_roi_final_cos,
        'verdict': verdict,
    }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"PHASE 12B RESULT: {roi_count} ROI")
    print(f"{'='*70}")
    print(f"Best cos: {best_cos_mean:.4f}")
    print(f"Last cos: {last_cos:.4f}")
    print(f"Cos delta: {cos_delta:+.4f}")
    print(f"Loss delta: {loss_delta:+.4f}")
    if last_gap is not None:
        print(f"Last gap: {last_gap:.4f}")
    print(f"Mean param delta: {np.mean(param_deltas) if param_deltas else 0:.6e}")
    print(f"NaN/Inf: {total_nan}/{total_inf}")
    print(f"VERDICT: {verdict}")
    print(f"SUCCESS: {'success' in verdict}")
    
    return "success" in verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/apps/wildtrack_full_3dgut.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs/phase12b_gaussianset_progressive')
    parser.add_argument('--person_feature_lr', type=float, default=1e-3)
    parser.add_argument('--num_steps', type=int, default=300)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--progressive_roi_count', type=int, default=1)
    parser.add_argument('--same_person_cross_view', action='store_true', default=False)
    parser.add_argument('--allowed_cameras', type=str, default='C1,C4,C6,C7')
    parser.add_argument('--min_alpha_sum', type=float, default=0.1)
    parser.add_argument('--min_bbox_area', type=int, default=100)
    parser.add_argument('--grad_clip_norm', type=float, default=10.0)
    parser.add_argument('--denom_eps', type=float, default=1e-8)
    parser.add_argument('--P', type=int, default=4)
    parser.add_argument('--K', type=int, default=2)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    allowed_cameras = [c.strip() for c in args.allowed_cameras.split(',')] if args.allowed_cameras else None
    
    config_dir = os.path.join(REPO_ROOT, 'configs')
    config_file = 'apps/' + os.path.basename(args.config).replace('.yaml', '')
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_file)
    
    cfg.model.person_feature_dim = 512
    
    from threedgrut.trainer import Trainer3DGRUT
    
    trainer = Trainer3DGRUT(cfg)
    
    sampler = PkMultiViewSampler(trainer.train_dataset, num_person=args.P, num_views=args.K)
    
    class BatchBuilder:
        def __init__(self, trainer, sampler):
            self.trainer = trainer
            self.sampler = sampler
        
        def get_batch_by_cam_frame(self, cam_id, frame_idx):
            idx = self.sampler.get_dataset_index(cam_id, frame_idx)
            if idx is None:
                return None
            raw_batch = self.trainer.train_dataset[idx]
            return self.trainer.train_dataset.get_gpu_batch_with_intrinsics(raw_batch)
    
    batch_builder = BatchBuilder(trainer, sampler)
    
    return run_phase12b(args, trainer, sampler, batch_builder, allowed_cameras)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
