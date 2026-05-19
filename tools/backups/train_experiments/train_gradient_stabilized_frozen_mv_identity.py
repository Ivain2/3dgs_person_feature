#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 11B v4: Gradient-stabilized Frozen Multi-view Identity Training.

Diagnostic mode: isolate gradient explosion source through 5 tests.
Train mode: stable training with L_teacher + L_mv (L_proto optional).

DO NOT modify any existing training scripts.
"""

import argparse
import os
import sys
import json
import time
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT


def normalize_feat(x, eps=1e-6):
    return F.normalize(x.float(), p=2, dim=0, eps=eps)


def compute_supcon_loss(features, labels, temperature=0.2):
    device = features.device
    N = features.shape[0]
    if N < 2:
        return torch.zeros(1, device=device)
    sim_matrix = features @ features.T / temperature
    labels_expanded = labels.unsqueeze(1)
    positive_mask = (labels_expanded == labels_expanded.T).float()
    positive_mask.fill_diagonal_(0)
    num_positives = positive_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
    exp_sim = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0])
    exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)
    pos_exp = exp_sim * positive_mask
    log_prob = pos_exp.sum(dim=1, keepdim=True) / num_positives
    log_prob = log_prob - torch.log(exp_sim_sum.clamp(min=1e-8))
    return -log_prob.mean()


def opacity_roi_pooling(feature_map, alpha_map, bbox_xyxy, denom_eps=1e-2, detach_opacity_weight=True, skip_normalize=False):
    """
    Opacity-aware ROI pooling with diagnostic info.
    Returns: pooled feature or None, stats dict
    """
    D, H, W = feature_map.shape
    xmin = max(0, int(bbox_xyxy[0].item()))
    ymin = max(0, int(bbox_xyxy[1].item()))
    xmax = min(W, max(xmin + 1, int(bbox_xyxy[2].item())))
    ymax = min(H, max(ymin + 1, int(bbox_xyxy[3].item())))

    bbox_area = (xmax - xmin) * (ymax - ymin)
    if bbox_area < 1:
        return None, {"valid": False, "reason": "bbox_area_too_small", "bbox_area": bbox_area}

    region = feature_map[:, ymin:ymax, xmin:xmax]
    alpha_region = alpha_map[ymin:ymax, xmin:xmax]

    if detach_opacity_weight:
        weight = alpha_region.detach()
    else:
        weight = alpha_region

    alpha_sum = weight.sum()
    denom = alpha_sum.clamp(min=denom_eps)
    clamped = (alpha_sum < denom_eps)

    weighted_sum = (region * weight.unsqueeze(0)).sum(dim=(1, 2))
    pooled = weighted_sum / denom
    if not skip_normalize:
        pooled = normalize_feat(pooled)

    stats = {
        "valid": True,
        "alpha_sum": alpha_sum.item(),
        "denom": denom.item(),
        "clamped": clamped.item(),
        "bbox_area": bbox_area,
        "pooled_norm": pooled.norm(p=2).item(),
    }
    return pooled, stats


def gaussian_set_pooling(model, gpu_batch, bbox, cam_id, frame_id, args, device):
    """
    Gaussian-Set pooling: directly aggregate Gaussians inside bbox.
    Uses rendered feature map and opacity map for pooling.
    
    Input: model, gpu_batch, bbox=[x1,y1,x2,y2], cam_id, frame_id, args, device
    Output: G (pooled feature), debug_info dict
    """
    x1, y1, x2, y2 = bbox
    
    try:
        render_out = model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
        person_feature_map = render_out.get('person_feature_map')
        person_opacity_map = render_out.get('person_opacity_map')
        
        if person_feature_map is None or person_opacity_map is None:
            return None, {
                'num_gaussians_in_bbox': 0,
                'weight_sum': 0.0,
                'error': 'no_feature_map',
                'bbox': bbox,
                'cam_id': cam_id,
                'frame_id': frame_id,
            }
        
        D, H, W = person_feature_map.shape
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(W, x2), min(H, y2)
        
        if x2_c <= x1_c or y2_c <= y1_c:
            return None, {
                'num_gaussians_in_bbox': 0,
                'weight_sum': 0.0,
                'error': 'bbox_outside_image',
                'bbox': bbox,
                'cam_id': cam_id,
                'frame_id': frame_id,
            }
        
        region_features = person_feature_map[:, y1_c:y2_c, x1_c:x2_c]
        region_opacity = person_opacity_map[y1_c:y2_c, x1_c:x2_c]
        
        weight = region_opacity
        alpha_sum = weight.sum()
        
        if alpha_sum < args.denom_eps:
            return None, {
                'num_gaussians_in_bbox': 0,
                'weight_sum': float(alpha_sum.item()),
                'error': 'low_alpha',
                'bbox': bbox,
                'cam_id': cam_id,
                'frame_id': frame_id,
            }
        
        denom = alpha_sum.clamp(min=args.denom_eps)
        
        weighted_sum = (region_features * weight.unsqueeze(0)).sum(dim=(1, 2))
        G = weighted_sum / denom
        G = normalize_feat(G)
        
        debug_info = {
            'num_gaussians_in_bbox': int((region_opacity > 0.01).sum().item()),
            'weight_sum': float(alpha_sum.item()),
            'weight_min': float(weight.min().item()),
            'weight_mean': float(weight.mean().item()),
            'weight_max': float(weight.max().item()),
            'bbox': bbox,
            'cam_id': cam_id,
            'frame_id': frame_id,
        }
        
        return G, debug_info
        
    except Exception as e:
        print(f"gaussian_set_pooling error: {e}")
        return None, {
            'num_gaussians_in_bbox': 0,
            'weight_sum': 0.0,
            'error': str(e),
            'bbox': bbox,
            'cam_id': cam_id,
            'frame_id': frame_id,
        }


def get_grad_stats(tensor):
    if tensor.grad is None:
        return {"grad_norm": 0.0, "grad_max": 0.0, "grad_nonzero": 0, "grad_nonzero_ratio": 0.0, "nan_count": 0, "inf_count": 0}
    g = tensor.grad
    return {
        "grad_norm": g.norm().item(),
        "grad_max": g.abs().max().item(),
        "grad_nonzero": (g.abs() > 1e-12).sum().item(),
        "grad_nonzero_ratio": (g.abs() > 1e-12).sum().item() / g.numel() * 100,
        "nan_count": torch.isnan(g).sum().item(),
        "inf_count": torch.isinf(g).sum().item(),
    }


class PkMultiViewSampler:
    def __init__(self, dataset, num_person, num_views, max_retries=10):
        self.dataset = dataset
        self.num_person = num_person
        self.num_views = num_views
        self.max_retries = max_retries
        self.annotations = dataset.annotations
        self.teacher_cache = dataset.teacher_cache

        self.cam_frame_to_index = {}
        valid_frame_set = set()
        for idx, (cam_id, frame_idx) in enumerate(dataset.indices):
            fi = int(frame_idx)
            self.cam_frame_to_index[(cam_id, fi)] = idx
            valid_frame_set.add(fi)

        self.timestamp_to_persons = defaultdict(set)
        self.person_cam_at_ts = defaultdict(lambda: defaultdict(set))

        for frame_id, annots in self.annotations.items():
            if not isinstance(annots, list):
                continue
            fi = int(frame_id)
            if fi not in valid_frame_set:
                continue
            for p in annots:
                pid = p.get('train_id') or p.get('new_id')
                if pid is None:
                    continue
                annot_cam_id = p.get('camera_id')
                if annot_cam_id is None:
                    continue
                cam_id = f"C{annot_cam_id + 1}"
                self.person_cam_at_ts[pid][fi].add(cam_id)
                self.timestamp_to_persons[fi].add(pid)

        valid_persons = set()
        for pid in self.person_cam_at_ts:
            for ts, cams in self.person_cam_at_ts[pid].items():
                if len(cams) >= self.num_views:
                    valid_persons.add(pid)
                    break

        filtered_person_cam = defaultdict(lambda: defaultdict(set))
        filtered_ts_persons = defaultdict(set)
        for pid in valid_persons:
            for ts, cams in self.person_cam_at_ts[pid].items():
                if len(cams) >= self.num_views:
                    filtered_person_cam[pid][ts] = cams
                    filtered_ts_persons[ts].add(pid)

        self.person_cam_at_ts = filtered_person_cam
        self.timestamp_to_persons = filtered_ts_persons
        self.all_timestamps = sorted([ts for ts, pids in self.timestamp_to_persons.items() if len(pids) >= self.num_person])

        print(f"[PkMultiViewSampler] {len(self.all_timestamps)} valid timestamps, "
              f"{len(valid_persons)} valid persons, {len(self.cam_frame_to_index)} entries")

    def sample_batch(self):
        for attempt in range(self.max_retries):
            if not self.all_timestamps:
                return None
            timestamp = random.choice(self.all_timestamps)
            available_persons = list(self.timestamp_to_persons.get(timestamp, set()))
            if len(available_persons) < self.num_person:
                continue
            selected_persons = random.sample(available_persons, self.num_person)
            mv_samples = []
            valid = True
            for pid in selected_persons:
                cams_at_ts = self.person_cam_at_ts.get(pid, {}).get(timestamp, set())
                if len(cams_at_ts) < self.num_views:
                    valid = False
                    break
                chosen_cams = random.sample(sorted(cams_at_ts), self.num_views)
                mv_samples.append({
                    'person_id': pid,
                    'views': [(cam_id, timestamp) for cam_id in chosen_cams],
                })
            if not valid or len(mv_samples) < self.num_person:
                continue
            return mv_samples
        return None

    def get_dataset_index(self, cam_id, frame_idx):
        return self.cam_frame_to_index.get((cam_id, int(frame_idx)))


class FrozenMVBatchBuilder:
    def __init__(self, trainer, sampler):
        self.trainer = trainer
        self.sampler = sampler

    def get_batch_by_cam_frame(self, cam_id, frame_idx):
        idx = self.sampler.get_dataset_index(cam_id, frame_idx)
        if idx is None:
            return None
        raw_batch = self.trainer.train_dataset[idx]
        return self.trainer.train_dataset.get_gpu_batch_with_intrinsics(raw_batch)

    def process_mv_samples(self, mv_samples, denom_eps=1e-2, detach_opacity_weight=True, debug_grad_chain=False):
        view_groups = defaultdict(list)
        for ps in mv_samples:
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                view_groups[(cam_id, int(frame_idx))].append(pid)

        unique_renders = len(view_groups)
        observations = []
        skipped_low_alpha = 0
        diag_info_list = []

        for view_idx, ((cam_id, frame_idx), pids) in enumerate(view_groups.items()):
            gpu_batch = self.get_batch_by_cam_frame(cam_id, frame_idx)
            if gpu_batch is None:
                continue

            if debug_grad_chain:
                print(f"  [process_mv] View {view_idx}: rendering cam={cam_id}, frame={frame_idx}")
                print(f"    person_feature.requires_grad before render: {self.trainer.model._person_feature.requires_grad}")

            render_out = self.trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            if debug_grad_chain:
                print(f"    feature_map.requires_grad after render: {person_feature_map.requires_grad}")
                print(f"    feature_map.mean: {person_feature_map.mean().item():.6f}")
                if person_opacity_map is not None:
                    print(f"    opacity_map.requires_grad after render: {person_opacity_map.requires_grad}")

            for pid_idx, pid in enumerate(pids):
                inst = None
                for i in gpu_batch.instances:
                    if i.get('train_id') == pid:
                        inst = i
                        break
                if inst is None:
                    continue

                teacher_emb = inst.get('teacher_embedding')
                if teacher_emb is None:
                    skipped_low_alpha += 1
                    continue

                bbox = inst['bbox_xyxy']
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=self.trainer.device)

                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=denom_eps, detach_opacity_weight=detach_opacity_weight,
                )

                if debug_grad_chain and f_v is not None:
                    print(f"    Person {pid} (idx={pid_idx}): f_v.requires_grad={f_v.requires_grad}, "
                          f"grad_fn={f_v.grad_fn}, norm={f_v.norm().item():.4f}")

                if f_v is None:
                    skipped_low_alpha += 1
                    continue

                observations.append({
                    'person_id': pid,
                    'feature': f_v,
                    'teacher_feature': teacher_emb,
                    'cam_id': cam_id,
                    'frame_idx': frame_idx,
                })
                diag_info_list.append(pool_stats)

        if debug_grad_chain and observations:
            f_stack_test = torch.stack([o['feature'] for o in observations])
            print(f"  [process_mv] After stack: f_stack.requires_grad={f_stack_test.requires_grad}, "
                  f"grad_fn={f_stack_test.grad_fn}")
            # Test backward immediately
            f_stack_test.retain_grad()
            test_loss = f_stack_test.sum()
            test_loss.backward()
            pf_test = self.trainer.model.get_person_feature()
            if pf_test.grad is not None:
                print(f"  [process_mv] After immediate backward: person_feature.grad_norm={pf_test.grad.norm().item():.6e}")
            else:
                print(f"  [process_mv] After immediate backward: person_feature.grad is None")
            # Clear for next test
            self.trainer.model.zero_grad()

        return observations, skipped_low_alpha, unique_renders, diag_info_list


def run_diagnostic(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC MODE")
    print("=" * 70)

    pf = trainer.model.get_person_feature()
    print(f"\n[Pre-check] person_feature:")
    print(f"  shape: {pf.shape}")
    print(f"  requires_grad: {pf.requires_grad}")
    print(f"  grad before: {pf.grad}")

    # Test 0: Direct gradient test on person_feature
    trainer.model.zero_grad()
    test_val = pf.sum() * 0.0001
    test_val.backward()
    print(f"  grad after direct backward: norm={pf.grad.norm().item():.6e}, nonzero_ratio={(pf.grad.abs()>1e-12).sum().item()/pf.grad.numel()*100:.4f}%")
    trainer.model.zero_grad()

    mv_samples = sampler.sample_batch()
    if mv_samples is None:
        print("ERROR: Cannot sample valid batch")
        return False

    observations, skipped, n_renders, diag_infos = batch_builder.process_mv_samples(
        mv_samples, denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight
    )
    num_valid = len(observations)
    print(f"\nSampled: {len(mv_samples)} persons x {len(mv_samples[0]['views'])} views, "
          f"valid={num_valid}, skipped={skipped}, renders={n_renders}")

    if num_valid < 4:
        print("ERROR: Not enough valid features for diagnostic")
        return False

    person_ids = [o['person_id'] for o in observations]
    p_ids_tensor = torch.tensor(person_ids, device=trainer.device)
    f_stack = torch.stack([o['feature'] for o in observations])

    alpha_sums = [d.get('alpha_sum', 0) for d in diag_infos if d.get('valid')]
    denoms = [d.get('denom', 0) for d in diag_infos if d.get('valid')]
    pooled_norms = [d.get('pooled_norm', 0) for d in diag_infos if d.get('valid')]

    # Check gradient chain
    print(f"\n[Gradient chain check]")
    print(f"  f_stack requires_grad: {f_stack.requires_grad}")
    print(f"  f_stack grad_fn: {f_stack.grad_fn}")
    print(f"  f_stack[0] grad_fn: {f_stack[0].grad_fn}")

    results = []

    test_defs = [
        ("test1_pooled_feature_sum", "renderer+ROI_pooling_gradient"),
        ("test2_teacher_cosine", "teacher_cosine_alignment"),
        ("test3_teacher_mse", "teacher_mse_alignment"),
        ("test4_supcon", "supervised_contrastive_loss"),
        ("test5_proto_ce", "prototype_classification"),
        ("test6_direct_render_check", "direct_render_gradient_no_roi"),
        ("test7_roi_pooling_gradient_check", "verify_roi_pooling_gradient_chain"),
    ]

    for test_name, test_desc in test_defs:
        trainer.model.zero_grad()
        trainer.model.optimizer.zero_grad()

        try:
            if test_name == "test7_roi_pooling_gradient_check":
                # Detailed ROI pooling gradient chain test - each sub-test needs fresh render
                test_results = {}
                
                # Sub-test a: Direct feature map region sum
                ps = mv_samples[0]
                pid = ps['person_id']
                cam_id, frame_idx = ps['views'][0]
                idx = sampler.get_dataset_index(cam_id, frame_idx)
                raw_batch = trainer.train_dataset[idx]
                gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(raw_batch)
                render_out = trainer.model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
                fmap = render_out['person_feature_map']
                amap = render_out.get('person_opacity_map')
                inst = None
                for i in gpu_batch.instances:
                    if i.get('train_id') == pid:
                        inst = i
                        break
                if inst is None:
                    record = {"test_name": test_name, "error": "person not in instances"}
                    results.append(record)
                    continue
                bbox = inst['bbox_xyxy']
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)
                D, H, W = fmap.shape
                xmin, ymin = max(0, int(bbox[0])), max(0, int(bbox[1]))
                xmax, ymax = min(W, int(bbox[2])), min(H, int(bbox[3]))
                
                trainer.model.zero_grad()
                region_direct = fmap[:, ymin:ymax, xmin:xmax]
                loss_direct = region_direct.sum()
                loss_direct.backward()
                grad_direct = get_grad_stats(pf)
                test_results['grad_direct_region_sum'] = grad_direct['grad_norm']
                print(f"  [test7a] Direct region sum: grad_norm={grad_direct['grad_norm']:.6e}")
                
                # Sub-test b: ROI pooling with detach_opacity_weight=False (fresh render)
                trainer.model.zero_grad()
                gpu_batch2 = trainer.train_dataset.get_gpu_batch_with_intrinsics(trainer.train_dataset[idx])
                render_out2 = trainer.model(gpu_batch2, train=False, frame_id=0, render_person_feature=True)
                fmap2 = render_out2['person_feature_map']
                amap2 = render_out2.get('person_opacity_map')
                f_v_no_detach, _ = opacity_roi_pooling(fmap2, amap2, bbox_t, denom_eps=1e-2, detach_opacity_weight=False)
                loss_nd = f_v_no_detach.sum()
                loss_nd.backward()
                grad_nd = get_grad_stats(pf)
                test_results['grad_roi_no_detach'] = grad_nd['grad_norm']
                print(f"  [test7b] ROI pool (detach=False): grad_norm={grad_nd['grad_norm']:.6e}")
                
                # Sub-test c: ROI pooling with detach_opacity_weight=True (fresh render)
                trainer.model.zero_grad()
                gpu_batch3 = trainer.train_dataset.get_gpu_batch_with_intrinsics(trainer.train_dataset[idx])
                render_out3 = trainer.model(gpu_batch3, train=False, frame_id=0, render_person_feature=True)
                fmap3 = render_out3['person_feature_map']
                amap3 = render_out3.get('person_opacity_map')
                f_v_detach, _ = opacity_roi_pooling(fmap3, amap3, bbox_t, denom_eps=1e-2, detach_opacity_weight=True)
                loss_dt = f_v_detach.sum()
                loss_dt.backward()
                grad_dt = get_grad_stats(pf)
                test_results['grad_roi_with_detach'] = grad_dt['grad_norm']
                print(f"  [test7c] ROI pool (detach=True): grad_norm={grad_dt['grad_norm']:.6e}")
                
                # Sub-test d: Without normalize (fresh render)
                trainer.model.zero_grad()
                gpu_batch4 = trainer.train_dataset.get_gpu_batch_with_intrinsics(trainer.train_dataset[idx])
                render_out4 = trainer.model(gpu_batch4, train=False, frame_id=0, render_person_feature=True)
                fmap4 = render_out4['person_feature_map']
                amap4 = render_out4.get('person_opacity_map')
                region4 = fmap4[:, ymin:ymax, xmin:xmax]
                alpha4 = amap4[ymin:ymax, xmin:xmax].detach()
                wsum = (region4 * alpha4.unsqueeze(0)).sum(dim=(1,2))
                a_sum = alpha4.sum()
                denom = a_sum.clamp(min=1e-2)
                pooled_no_norm = wsum / denom
                loss_pnn = pooled_no_norm.sum()
                loss_pnn.backward()
                grad_pnn = get_grad_stats(pf)
                test_results['grad_roi_without_normalize'] = grad_pnn['grad_norm']
                print(f"  [test7d] ROI pool without normalize: grad_norm={grad_pnn['grad_norm']:.6e}")
                
                # Sub-test e: Check alpha values
                alpha_sum = amap[ymin:ymax, xmin:xmax].sum().item()
                test_results['alpha_sum'] = alpha_sum
                print(f"  [test7e] Alpha sum: {alpha_sum:.6f}")
                
                record = {"test_name": test_name, **test_results}
                results.append(record)
                continue

            debug_this_test = (test_name in ["test1_pooled_feature_sum", "test6_direct_render_check", "test7_roi_pooling_gradient_check"])
            
            observations_fresh, _, _, diag_infos_fresh = batch_builder.process_mv_samples(
                mv_samples, denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                debug_grad_chain=debug_this_test
            )
            f_stack_fresh = torch.stack([o['feature'] for o in observations_fresh])
            num_valid_fresh = len(observations_fresh)

            alpha_sums_f = [d.get('alpha_sum', 0) for d in diag_infos_fresh if d.get('valid')]
            denoms_f = [d.get('denom', 0) for d in diag_infos_fresh if d.get('valid')]
            pooled_norms_f = [d.get('pooled_norm', 0) for d in diag_infos_fresh if d.get('valid')]

            if test_name == "test1_pooled_feature_sum":
                loss = f_stack_fresh.sum()
            elif test_name == "test2_teacher_cosine":
                loss = (1 - torch.stack([
                    torch.dot(f, normalize_feat(torch.as_tensor(o['teacher_feature'], dtype=torch.float32, device=trainer.device).squeeze()))
                    for f, o in zip(f_stack_fresh, observations_fresh)
                ]).mean())
            elif test_name == "test3_teacher_mse":
                loss = F.mse_loss(
                    f_stack_fresh,
                    torch.stack([normalize_feat(torch.as_tensor(o['teacher_feature'], dtype=torch.float32, device=trainer.device).squeeze()) for o in observations_fresh])
                )
            elif test_name == "test4_supcon":
                loss = compute_supcon_loss(f_stack_fresh, p_ids_tensor, temperature=args.tau_mv)
            elif test_name == "test5_proto_ce":
                loss = F.cross_entropy(f_stack_fresh @ prototypes.T / args.tau_proto, p_ids_tensor)
            elif test_name == "test6_direct_render_check":
                # Direct render without ROI pooling
                obs0 = observations_fresh[0]
                # Re-render to get the full feature map
                cam_id, frame_idx = obs0['cam_id'], obs0['frame_idx']
                idx = sampler.get_dataset_index(cam_id, frame_idx)
                raw_batch = trainer.train_dataset[idx]
                gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(raw_batch)
                render_out = trainer.model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
                fmap = render_out['person_feature_map']
                # Use a small region's sum as loss
                loss = fmap[:, 100:110, 100:110].sum()

            loss.backward()
            grad_info = get_grad_stats(pf)

            # Check if person_feature.grad is allocated
            pf_grad_allocated = (pf.grad is not None)
            pf_grad_device = str(pf.grad.device) if pf_grad_allocated else None

            teacher_norms = [
                torch.as_tensor(o['teacher_feature'], dtype=torch.float32, device=trainer.device).squeeze().norm().item()
                for o in observations_fresh
            ]
            student_teacher_cos = [
                torch.dot(f, normalize_feat(torch.as_tensor(o['teacher_feature'], dtype=torch.float32, device=trainer.device).squeeze())).item()
                for f, o in zip(f_stack_fresh, observations_fresh)
            ]

            record = {
                "test_name": test_name,
                "loss_value": loss.item(),
                "num_valid_features": num_valid_fresh,
                "num_persons": len(mv_samples),
                "num_views": len(mv_samples[0]['views']),
                "alpha_sum_min": min(alpha_sums_f) if alpha_sums_f else 0,
                "alpha_sum_mean": np.mean(alpha_sums_f) if alpha_sums_f else 0,
                "alpha_sum_max": max(alpha_sums_f) if alpha_sums_f else 0,
                "denominator_min": min(denoms_f) if denoms_f else 0,
                "denominator_mean": np.mean(denoms_f) if denoms_f else 0,
                "denominator_max": max(denoms_f) if denoms_f else 0,
                "denominator_clamp_ratio": sum(1 for d in diag_infos_fresh if d.get('clamped', False)) / max(len(diag_infos_fresh), 1),
                "pooled_feature_norm_min": min(pooled_norms_f) if pooled_norms_f else 0,
                "pooled_feature_norm_mean": np.mean(pooled_norms_f) if pooled_norms_f else 0,
                "pooled_feature_norm_max": max(pooled_norms_f) if pooled_norms_f else 0,
                "teacher_feature_norm_min": min(teacher_norms) if teacher_norms else 0,
                "teacher_feature_norm_mean": np.mean(teacher_norms) if teacher_norms else 0,
                "teacher_feature_norm_max": max(teacher_norms) if teacher_norms else 0,
                "student_teacher_cos_mean": np.mean(student_teacher_cos) if student_teacher_cos else 0,
                "similarity_logit_min": f_stack_fresh.min().item(),
                "similarity_logit_mean": f_stack_fresh.mean().item(),
                "similarity_logit_max": f_stack_fresh.max().item(),
                "person_feature_grad_allocated": pf_grad_allocated,
                "person_feature_grad_device": pf_grad_device,
                "person_feature_grad_norm_before_clip": grad_info["grad_norm"],
                "person_feature_grad_max_before_clip": grad_info["grad_max"],
                "person_feature_grad_nonzero_count": grad_info["grad_nonzero"],
                "person_feature_grad_nonzero_ratio_global": grad_info["grad_nonzero_ratio"],
                "nan_count": grad_info["nan_count"],
                "inf_count": grad_info["inf_count"],
            }
            results.append(record)

            print(f"\n  [{test_name}] ({test_desc})")
            print(f"    loss = {loss.item():.6f}")
            print(f"    grad_allocated = {pf_grad_allocated}")
            print(f"    grad_norm = {grad_info['grad_norm']:.6e}")
            print(f"    grad_max = {grad_info['grad_max']:.6e}")
            print(f"    grad_nz_ratio = {grad_info['grad_nonzero_ratio']:.4f}%")
            print(f"    nan/inf = {grad_info['nan_count']}/{grad_info['inf_count']}")
            print(f"    alpha_sum = [{min(alpha_sums_f):.6f}, {np.mean(alpha_sums_f):.6f}, {max(alpha_sums_f):.6f}]")
            print(f"    pooled_norm = [{min(pooled_norms_f):.6f}, {np.mean(pooled_norms_f):.6f}, {max(pooled_norms_f):.6f}]")
            print(f"    student_teacher_cos = {np.mean(student_teacher_cos):.4f}")

        except Exception as e:
            print(f"\n  [{test_name}] ({test_desc}) ERROR: {e}")
            import traceback
            traceback.print_exc()
            record = {"test_name": test_name, "error": str(e)}
            results.append(record)

    log_path = os.path.join(args.output_dir, "diagnostic_results.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(log_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Diagnostic log saved: {log_path}")

    print(f"\n--- Gradient Explosion Summary ---")
    for r in results:
        if 'person_feature_grad_norm_before_clip' in r:
            gn = r['person_feature_grad_norm_before_clip']
            status = "EXPLOSION" if gn > 1e6 else ("HIGH" if gn > 100 else ("NORMAL" if gn > 1 else "LOW"))
            print(f"  {r['test_name']:30s}: grad_norm={gn:.6e} [{status}]")

    return True


def run_train_sequential_safe(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("TRAIN SEQUENTIAL SAFE MODE")
    print("=" * 70)

    train_log = []
    t_start = time.time()

    for step in range(args.num_steps):
        t_iter_start = time.time()

        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue

        view_groups = defaultdict(list)
        for ps in mv_samples:
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                view_groups[(cam_id, int(frame_idx))].append(pid)

        unique_render_count = len(view_groups)
        trainer.model.optimizer.zero_grad()

        total_loss_teacher = 0.0
        total_valid = 0
        all_alpha_sums = []
        all_denoms = []
        all_clamped = []
        all_pooled_norms = []
        all_cos_fv_tv = []

        for view_idx, ((cam_id, frame_idx), pids) in enumerate(view_groups.items()):
            gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
            if gpu_batch is None:
                continue

            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            view_loss = 0.0
            num_valid_in_view = 0

            for pid in pids:
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
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)

                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )

                if f_v is None:
                    continue

                all_alpha_sums.append(pool_stats.get('alpha_sum', 0))
                all_denoms.append(pool_stats.get('denom', 0))
                if pool_stats.get('clamped', False):
                    all_clamped.append(1)
                else:
                    all_clamped.append(0)
                all_pooled_norms.append(pool_stats.get('pooled_norm', 0))

                teacher_feat = normalize_feat(
                    torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                )
                cos_sim = torch.dot(f_v, teacher_feat)
                all_cos_fv_tv.append(cos_sim.item())

                if args.teacher_loss_type == 'cosine':
                    l_t = 1 - cos_sim
                else:
                    l_t = F.mse_loss(f_v, teacher_feat)

                view_loss = view_loss + l_t
                num_valid_in_view += 1

            if num_valid_in_view > 0:
                view_loss = view_loss / num_valid_in_view
                scaled_loss = args.lambda_teacher * view_loss / unique_render_count
                scaled_loss.backward()

                total_loss_teacher += view_loss.item()
                total_valid += num_valid_in_view

        pf = trainer.model.get_person_feature()
        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]

        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
        grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0

        trainer.model.optimizer.step()

        if total_valid < 6:
            if step % args.log_interval == 0:
                print(f"[SEQ_SAFE] Step {step:5d}: skipped (valid={total_valid} < 6)")
            continue

        denom_clamp_ratio = sum(all_clamped) / max(len(all_clamped), 1)

        nan_count = grad_info["nan_count"]
        inf_count = grad_info["inf_count"]

        iter_time = time.time() - t_iter_start

        record = {
            "step": step,
            "loss_total": total_loss_teacher / max(1, unique_render_count),
            "loss_teacher": total_loss_teacher / max(1, unique_render_count),
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "alpha_sum_min": min(all_alpha_sums) if all_alpha_sums else 0,
            "alpha_sum_mean": np.mean(all_alpha_sums) if all_alpha_sums else 0,
            "alpha_sum_max": max(all_alpha_sums) if all_alpha_sums else 0,
            "denom_clamp_ratio": denom_clamp_ratio,
            "pooled_feature_norm_min": min(all_pooled_norms) if all_pooled_norms else 0,
            "pooled_feature_norm_mean": np.mean(all_pooled_norms) if all_pooled_norms else 0,
            "pooled_feature_norm_max": max(all_pooled_norms) if all_pooled_norms else 0,
            "cos_fv_tv_mean": np.mean(all_cos_fv_tv) if all_cos_fv_tv else 0,
            "valid_feature_count": total_valid,
            "unique_render_count": unique_render_count,
            "missing_count": 0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "iter_time": iter_time,
        }
        train_log.append(record)

        if step % args.log_interval == 0:
            print(f"[SEQ_SAFE] Step {step:5d}: "
                  f"loss={record['loss_total']:.4f} "
                  f"grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                  f"cos_tv={record['cos_fv_tv_mean']:.4f} "
                  f"valid={total_valid} renders={unique_render_count} "
                  f"clamp_ratio={denom_clamp_ratio:.3f} "
                  f"nan/inf={nan_count}/{inf_count} "
                  f"t={iter_time:.2f}s")

        if step > 0 and args.save_interval > 0 and step % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_v4_seq_step_{step}.pth")
            torch.save({
                'step': step,
                'person_feature': trainer.model.get_person_feature().detach().cpu(),
                'optimizer_state': trainer.model.optimizer.state_dict(),
            }, ckpt_path)

    final_path = os.path.join(args.output_dir, "checkpoint_v4_seq_final.pth")
    torch.save({
        'step': args.num_steps - 1,
        'person_feature': trainer.model.get_person_feature().detach().cpu(),
        'optimizer_state': trainer.model.optimizer.state_dict(),
    }, final_path)

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r) + "\n")

    print(f"\nTotal training time: {time.time() - t_start:.1f}s")
    print(f"Final checkpoint: {final_path}")
    print(f"Metrics: {metrics_path}")
    return True


def run_train_sequential_mv_stopgrad(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("TRAIN SEQUENTIAL MV STOPGRAD MODE")
    print("=" * 70)

    train_log = []
    t_start = time.time()

    for step in range(args.num_steps):
        t_iter_start = time.time()

        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue

        view_groups = defaultdict(list)
        for ps in mv_samples:
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                view_groups[(cam_id, int(frame_idx))].append(pid)

        unique_render_count = len(view_groups)

        target_bank = {}
        with torch.no_grad():
            for (cam_id, frame_idx), pids in view_groups.items():
                gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
                if gpu_batch is None:
                    continue

                render_out = trainer.model(
                    gpu_batch, train=False, frame_id=0, render_person_feature=True
                )
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')

                for pid in pids:
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
                    bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)

                    f_v, pool_stats = opacity_roi_pooling(
                        person_feature_map, person_opacity_map, bbox_t,
                        denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                    )

                    if f_v is None:
                        continue

                    key = (pid, cam_id, int(frame_idx))
                    target_bank[key] = {
                        'feature': f_v.detach().clone(),
                        'teacher_feature': teacher_emb,
                        'bbox': bbox_t,
                    }

        trainer.model.optimizer.zero_grad()

        total_loss_teacher = 0.0
        total_loss_mv = 0.0
        total_valid = 0
        skipped_mv = 0
        all_alpha_sums = []
        all_denoms = []
        all_clamped = []
        all_pooled_norms = []
        all_cos_fv_tv = []
        cross_view_same_cos = []
        cross_view_diff_cos = []
        positive_count = 0
        negative_count = 0

        for view_idx, ((cam_id, frame_idx), pids) in enumerate(view_groups.items()):
            gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
            if gpu_batch is None:
                continue

            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            view_loss_teacher = 0.0
            view_loss_mv = 0.0
            num_valid_in_view = 0

            anchor_features = []
            anchor_pids = []

            for pid in pids:
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
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)

                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )

                if f_v is None:
                    continue

                all_alpha_sums.append(pool_stats.get('alpha_sum', 0))
                all_denoms.append(pool_stats.get('denom', 0))
                if pool_stats.get('clamped', False):
                    all_clamped.append(1)
                else:
                    all_clamped.append(0)
                all_pooled_norms.append(pool_stats.get('pooled_norm', 0))

                teacher_feat = normalize_feat(
                    torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                )
                cos_sim = torch.dot(f_v, teacher_feat)
                all_cos_fv_tv.append(cos_sim.item())

                if args.teacher_loss_type == 'cosine':
                    l_t = 1 - cos_sim
                else:
                    l_t = F.mse_loss(f_v, teacher_feat)

                view_loss_teacher = view_loss_teacher + l_t

                anchor_features.append(f_v)
                anchor_pids.append(pid)
                num_valid_in_view += 1

            if num_valid_in_view > 0 and args.lambda_mv > 0 and anchor_features:
                anchor_stack = torch.stack(anchor_features)

                target_features = []
                target_labels = []

                for i, pid in enumerate(anchor_pids):
                    same_person_other_views = []
                    different_person_features = []

                    for (t_pid, t_cam, t_frame), t_data in target_bank.items():
                        if t_cam == cam_id and t_frame == frame_idx:
                            continue
                        if t_pid == pid:
                            same_person_other_views.append(t_data['feature'])
                        else:
                            different_person_features.append(t_data['feature'])

                    if same_person_other_views:
                        positive_count += len(same_person_other_views)
                        for pos_f in same_person_other_views:
                            target_features.append(pos_f)
                            target_labels.append(1)
                    else:
                        skipped_mv += 1

                    if different_person_features:
                        negative_count += len(different_person_features)
                        for neg_f in different_person_features:
                            target_features.append(neg_f)
                            target_labels.append(0)

                if target_features:
                    target_stack = torch.stack(target_features).detach()
                    labels_tensor = torch.tensor(target_labels, dtype=torch.float32, device=trainer.device)

                    sim_matrix = anchor_stack @ target_stack.T / args.tau_mv
                    pos_mask = (labels_tensor == 1).float().unsqueeze(0)  # [1, num_targets]
                    neg_mask = (labels_tensor == 0).float().unsqueeze(0)  # [1, num_targets]

                    pos_sim = (sim_matrix * pos_mask).sum(dim=1)  # [num_anchors]
                    pos_count = pos_mask.sum(dim=1).clamp(min=1e-8)  # [1]
                    neg_sim = (sim_matrix * neg_mask).sum(dim=1)  # [num_anchors]

                    exp_pos = torch.exp(pos_sim / pos_count.squeeze())
                    exp_neg = torch.exp(neg_sim)

                    l_mv = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8)).mean()
                    view_loss_mv = l_mv

            if num_valid_in_view > 0:
                view_loss = view_loss_teacher / num_valid_in_view
                if view_loss_mv > 0:
                    view_loss = view_loss + args.lambda_mv * view_loss_mv

                scaled_loss = args.lambda_teacher * view_loss / unique_render_count
                scaled_loss.backward()

                total_loss_teacher += view_loss_teacher.item() / num_valid_in_view
                if view_loss_mv > 0:
                    total_loss_mv += view_loss_mv.item()
                total_valid += num_valid_in_view

                for i, pid in enumerate(anchor_pids):
                    for (t_pid, t_cam, t_frame), t_data in target_bank.items():
                        if t_cam == cam_id and t_frame == frame_idx:
                            continue
                        cos_ij = torch.dot(anchor_features[i], t_data['feature']).item()
                        if t_pid == pid:
                            cross_view_same_cos.append(cos_ij)
                        else:
                            cross_view_diff_cos.append(cos_ij)

        pf = trainer.model.get_person_feature()
        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]

        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
        grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0

        trainer.model.optimizer.step()

        if total_valid < 6:
            if step % args.log_interval == 0:
                print(f"[SEQ_MV] Step {step:5d}: skipped (valid={total_valid} < 6)")
            continue

        denom_clamp_ratio = sum(all_clamped) / max(len(all_clamped), 1)
        nan_count = grad_info["nan_count"]
        inf_count = grad_info["inf_count"]

        iter_time = time.time() - t_iter_start

        same_cos_mean = np.mean(cross_view_same_cos) if cross_view_same_cos else 0
        diff_cos_mean = np.mean(cross_view_diff_cos) if cross_view_diff_cos else 0

        record = {
            "step": step,
            "loss_total": total_loss_teacher / max(1, unique_render_count) + args.lambda_mv * total_loss_mv / max(1, unique_render_count),
            "loss_teacher": total_loss_teacher / max(1, unique_render_count),
            "loss_mv_stopgrad": total_loss_mv / max(1, unique_render_count),
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "alpha_sum_min": min(all_alpha_sums) if all_alpha_sums else 0,
            "alpha_sum_mean": np.mean(all_alpha_sums) if all_alpha_sums else 0,
            "alpha_sum_max": max(all_alpha_sums) if all_alpha_sums else 0,
            "denom_clamp_ratio": denom_clamp_ratio,
            "pooled_feature_norm_min": min(all_pooled_norms) if all_pooled_norms else 0,
            "pooled_feature_norm_mean": np.mean(all_pooled_norms) if all_pooled_norms else 0,
            "pooled_feature_norm_max": max(all_pooled_norms) if all_pooled_norms else 0,
            "cos_fv_tv_mean": np.mean(all_cos_fv_tv) if all_cos_fv_tv else 0,
            "cross_view_same_cos": same_cos_mean,
            "cross_view_diff_cos": diff_cos_mean,
            "cross_view_gap": same_cos_mean - diff_cos_mean,
            "valid_feature_count": total_valid,
            "unique_render_count": unique_render_count,
            "positive_pair_count": positive_count,
            "negative_pair_count": negative_count,
            "skipped_mv_count": skipped_mv,
            "tau_mv": args.tau_mv,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "iter_time": iter_time,
        }
        train_log.append(record)

        if step % args.log_interval == 0:
            print(f"[SEQ_MV] Step {step:5d}: "
                  f"loss={record['loss_total']:.4f} "
                  f"(teacher={record['loss_teacher']:.4f}, mv={record['loss_mv_stopgrad']:.4f}) "
                  f"grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                  f"cos_tv={record['cos_fv_tv_mean']:.4f} "
                  f"gap={record['cross_view_gap']:.4f} "
                  f"(same={same_cos_mean:.4f}, diff={diff_cos_mean:.4f}) "
                  f"valid={total_valid} pos={positive_count} neg={negative_count} skip={skipped_mv} "
                  f"t={iter_time:.2f}s")

        if step > 0 and args.save_interval > 0 and step % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_v4_seq_mv_step_{step}.pth")
            torch.save({
                'step': step,
                'person_feature': trainer.model.get_person_feature().detach().cpu(),
                'optimizer_state': trainer.model.optimizer.state_dict(),
            }, ckpt_path)

    final_path = os.path.join(args.output_dir, "checkpoint_v4_seq_mv_final.pth")
    torch.save({
        'step': args.num_steps - 1,
        'person_feature': trainer.model.get_person_feature().detach().cpu(),
        'optimizer_state': trainer.model.optimizer.state_dict(),
    }, final_path)

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r) + "\n")

    print(f"\nTotal training time: {time.time() - t_start:.1f}s")
    print(f"Final checkpoint: {final_path}")
    print(f"Metrics: {metrics_path}")
    return True


def run_ablate_denom_eps(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("ABLATE DENOM EPS MODE")
    print("=" * 70)

    denom_values = [1e-3, 1e-2, 1e-1]
    results_summary = []

    for denom_eps in denom_values:
        print(f"\n{'=' * 50}")
        print(f"Testing denom_eps = {denom_eps}")
        print(f"{'=' * 50}")

        output_dir = args.output_dir.rstrip('/') + f"_denom{denom_eps:.0e}".replace('-', 'm')
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            args_dict = vars(args).copy()
            args_dict['denom_eps'] = denom_eps
            json.dump(args_dict, f, indent=2)

        args.denom_eps = denom_eps
        args.output_dir = output_dir

        train_log = []
        t_start = time.time()

        for step in range(args.num_steps):
            t_iter_start = time.time()

            mv_samples = sampler.sample_batch()
            if mv_samples is None:
                continue

            view_groups = defaultdict(list)
            for ps in mv_samples:
                pid = ps['person_id']
                for cam_id, frame_idx in ps['views']:
                    view_groups[(cam_id, int(frame_idx))].append(pid)

            unique_render_count = len(view_groups)
            trainer.model.optimizer.zero_grad()

            total_loss_teacher = 0.0
            total_valid = 0
            all_alpha_sums = []
            all_denoms = []
            all_clamped = []
            all_pooled_norms = []
            all_cos_fv_tv = []

            for view_idx, ((cam_id, frame_idx), pids) in enumerate(view_groups.items()):
                gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
                if gpu_batch is None:
                    continue

                render_out = trainer.model(
                    gpu_batch, train=False, frame_id=0, render_person_feature=True
                )
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')

                view_loss = 0.0
                num_valid_in_view = 0

                for pid in pids:
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
                    bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)

                    f_v, pool_stats = opacity_roi_pooling(
                        person_feature_map, person_opacity_map, bbox_t,
                        denom_eps=denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                    )

                    if f_v is None:
                        continue

                    all_alpha_sums.append(pool_stats.get('alpha_sum', 0))
                    all_denoms.append(pool_stats.get('denom', 0))
                    if pool_stats.get('clamped', False):
                        all_clamped.append(1)
                    else:
                        all_clamped.append(0)
                    all_pooled_norms.append(pool_stats.get('pooled_norm', 0))

                    teacher_feat = normalize_feat(
                        torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                    )
                    cos_sim = torch.dot(f_v, teacher_feat)
                    all_cos_fv_tv.append(cos_sim.item())

                    if args.teacher_loss_type == 'cosine':
                        l_t = 1 - cos_sim
                    else:
                        l_t = F.mse_loss(f_v, teacher_feat)

                    view_loss = view_loss + l_t
                    num_valid_in_view += 1

                if num_valid_in_view > 0:
                    view_loss = view_loss / num_valid_in_view
                    scaled_loss = args.lambda_teacher * view_loss / unique_render_count
                    scaled_loss.backward()

                    total_loss_teacher += view_loss.item()
                    total_valid += num_valid_in_view

            pf = trainer.model.get_person_feature()
            grad_info = get_grad_stats(pf)
            grad_norm_before = grad_info["grad_norm"]

            torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
            grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0

            trainer.model.optimizer.step()

            if total_valid < 6:
                if step % args.log_interval == 0:
                    print(f"[ABLATEdenom={denom_eps}] Step {step:5d}: skipped (valid={total_valid} < 6)")
                continue

            denom_clamp_ratio = sum(all_clamped) / max(len(all_clamped), 1)
            nan_count = grad_info["nan_count"]
            inf_count = grad_info["inf_count"]

            iter_time = time.time() - t_iter_start

            record = {
                "step": step,
                "loss_total": total_loss_teacher / max(1, unique_render_count),
                "loss_teacher": total_loss_teacher / max(1, unique_render_count),
                "grad_norm_before_clip": grad_norm_before,
                "grad_norm_after_clip": grad_norm_after,
                "alpha_sum_min": min(all_alpha_sums) if all_alpha_sums else 0,
                "alpha_sum_mean": np.mean(all_alpha_sums) if all_alpha_sums else 0,
                "alpha_sum_max": max(all_alpha_sums) if all_alpha_sums else 0,
                "denom_clamp_ratio": denom_clamp_ratio,
                "pooled_feature_norm_min": min(all_pooled_norms) if all_pooled_norms else 0,
                "pooled_feature_norm_mean": np.mean(all_pooled_norms) if all_pooled_norms else 0,
                "pooled_feature_norm_max": max(all_pooled_norms) if all_pooled_norms else 0,
                "cos_fv_tv_mean": np.mean(all_cos_fv_tv) if all_cos_fv_tv else 0,
                "valid_feature_count": total_valid,
                "unique_render_count": unique_render_count,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "iter_time": iter_time,
            }
            train_log.append(record)

            if step % args.log_interval == 0:
                print(f"[ABLATEdenom={denom_eps}] Step {step:5d}: "
                      f"loss={record['loss_total']:.4f} "
                      f"grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                      f"cos_tv={record['cos_fv_tv_mean']:.4f} "
                      f"valid={total_valid} clamp_ratio={denom_clamp_ratio:.3f} "
                      f"t={iter_time:.2f}s")

        metrics_path = os.path.join(output_dir, "metrics.jsonl")
        with open(metrics_path, 'w') as f:
            for r in train_log:
                f.write(json.dumps(r) + "\n")

        last_10_grad = np.mean([r['grad_norm_before_clip'] for r in train_log[-10:]]) if len(train_log) >= 10 else 0
        last_10_cos = np.mean([r['cos_fv_tv_mean'] for r in train_log[-10:]]) if len(train_log) >= 10 else 0
        last_10_loss = np.mean([r['loss_teacher'] for r in train_log[-10:]]) if len(train_log) >= 10 else 0
        avg_clamp = np.mean([r['denom_clamp_ratio'] for r in train_log]) if train_log else 0

        results_summary.append({
            "denom_eps": denom_eps,
            "output_dir": output_dir,
            "last_10_grad_norm": last_10_grad,
            "last_10_cos_tv": last_10_cos,
            "last_10_loss_teacher": last_10_loss,
            "avg_clamp_ratio": avg_clamp,
            "total_steps": len(train_log),
        })

        print(f"\n  Summary for denom_eps={denom_eps}:")
        print(f"    grad_norm (last 10) = {last_10_grad:.6e}")
        print(f"    cos_tv (last 10) = {last_10_cos:.4f}")
        print(f"    loss_teacher (last 10) = {last_10_loss:.4f}")
        print(f"    clamp_ratio (avg) = {avg_clamp:.4f}")

    print(f"\n{'=' * 70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'denom_eps':<12} {'grad_norm':<15} {'cos_tv':<10} {'loss_teacher':<12} {'clamp_ratio':<12}")
    for r in results_summary:
        print(f"{r['denom_eps']:<12.0e} {r['last_10_grad_norm']:<15.6e} {r['last_10_cos_tv']:<10.4f} "
              f"{r['last_10_loss_teacher']:<12.4f} {r['avg_clamp_ratio']:<12.4f}")

    summary_path = os.path.join(args.output_dir.rstrip('/').rsplit('_denom', 1)[0] if '_denom' in args.output_dir else args.output_dir,
                                "ablation_summary.json")
    os.makedirs(os.path.dirname(summary_path) if os.path.dirname(summary_path) else '.', exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    return True


def run_train(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("TRAIN MODE")
    print("=" * 70)

    train_log = []
    t_start = time.time()

    for step in range(args.num_steps):
        t_iter_start = time.time()

        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue

        observations, skipped, n_renders, diag_infos = batch_builder.process_mv_samples(
            mv_samples, denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight
        )
        num_valid = len(observations)

        if num_valid < 6:
            if step % args.log_interval == 0:
                print(f"[V4] Step {step:5d}: skipped (valid={num_valid} < 6)")
            continue

        person_ids = [o['person_id'] for o in observations]
        p_ids_tensor = torch.tensor(person_ids, device=trainer.device)
        f_stack = torch.stack([o['feature'] for o in observations])

        loss_teacher = (1 - torch.stack([
            torch.dot(f, normalize_feat(torch.as_tensor(o['teacher_feature'], dtype=torch.float32, device=trainer.device).squeeze()))
            for f, o in zip(f_stack, observations)
        ]).mean())

        loss_mv = compute_supcon_loss(f_stack, p_ids_tensor, temperature=args.tau_mv)

        loss_proto = torch.zeros(1, device=trainer.device)
        if args.enable_proto_after_steps >= 0 and step > args.enable_proto_after_steps:
            loss_proto = F.cross_entropy(f_stack @ prototypes.T / args.tau_proto, p_ids_tensor)

        loss_total = (args.lambda_teacher * loss_teacher +
                      args.lambda_mv * loss_mv +
                      args.lambda_proto * loss_proto)

        trainer.model.zero_grad()
        loss_total.backward()

        pf = trainer.model.get_person_feature()
        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]

        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
        grad_norm_after = pf.grad.norm().item()

        trainer.model.optimizer.step()

        alpha_sums = [d.get('alpha_sum', 0) for d in diag_infos if d.get('valid')]
        denoms = [d.get('denom', 0) for d in diag_infos if d.get('valid')]
        clamp_ratio = sum(1 for d in diag_infos if d.get('clamped', False)) / max(len(diag_infos), 1)

        student_teacher_cos = [
            torch.dot(f, normalize_feat(torch.as_tensor(o['teacher_feature'], dtype=torch.float32, device=trainer.device).squeeze())).item()
            for f, o in zip(f_stack, observations)
        ]

        same_cos, diff_cos = [], []
        for i in range(len(f_stack)):
            for j in range(i + 1, len(f_stack)):
                cos_ij = torch.dot(f_stack[i], f_stack[j]).item()
                if person_ids[i] == person_ids[j]:
                    same_cos.append(cos_ij)
                else:
                    diff_cos.append(cos_ij)

        labels_expanded = p_ids_tensor.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        positive_mask.fill_diagonal_(0)
        n_pos = int(positive_mask.sum().item() / 2)
        n_neg = int((1 - positive_mask - torch.eye(num_valid, device=trainer.device)).sum().item() / 2)

        iter_time = time.time() - t_iter_start

        record = {
            "step": step,
            "loss_total": loss_total.item(),
            "loss_teacher": loss_teacher.item(),
            "loss_mv": loss_mv.item(),
            "loss_proto": loss_proto.item(),
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "alpha_sum_min": min(alpha_sums) if alpha_sums else 0,
            "alpha_sum_mean": np.mean(alpha_sums) if alpha_sums else 0,
            "alpha_sum_max": max(alpha_sums) if alpha_sums else 0,
            "denom_clamp_ratio": clamp_ratio,
            "cos_fv_tv_mean": np.mean(student_teacher_cos) if student_teacher_cos else 0,
            "cross_view_same_cos": np.mean(same_cos) if same_cos else 0,
            "cross_view_diff_cos": np.mean(diff_cos) if diff_cos else 0,
            "cross_view_gap": (np.mean(same_cos) - np.mean(diff_cos)) if same_cos and diff_cos else 0,
            "valid_feature_count": num_valid,
            "positive_pair_count": n_pos,
            "negative_pair_count": n_neg,
            "iter_time": iter_time,
        }
        train_log.append(record)

        if step % args.log_interval == 0:
            print(f"[V4] Step {step:5d}: "
                  f"loss={loss_total.item():.4f} "
                  f"(teacher={loss_teacher.item():.4f}, mv={loss_mv.item():.4f}, proto={loss_proto.item():.4f}) "
                  f"grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                  f"cos_tv={np.mean(student_teacher_cos):.4f} "
                  f"gap={(np.mean(same_cos) - np.mean(diff_cos)):.4f} "
                  f"valid={num_valid} pos={n_pos} neg={n_neg} "
                  f"t={iter_time:.2f}s")

        if step > 0 and args.save_interval > 0 and step % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{step}.pth")
            torch.save({
                'step': step,
                'person_feature': trainer.model.get_person_feature().detach().cpu(),
                'optimizer_state': trainer.model.optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(args.output_dir, "checkpoint_final.pth")
    torch.save({
        'step': args.num_steps - 1,
        'person_feature': trainer.model.get_person_feature().detach().cpu(),
        'optimizer_state': trainer.model.optimizer.state_dict(),
    }, final_path)

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r) + "\n")

    print(f"\nTotal training time: {time.time() - t_start:.1f}s")
    print(f"Final checkpoint: {final_path}")
    print(f"Metrics: {metrics_path}")
    return True


def run_optimizer_sanity(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("OPTIMIZER SANITY MODE")
    print("=" * 70)

    pf = trainer.model.get_person_feature()
    pf_id = id(pf)
    pf_requires_grad = pf.requires_grad

    optimizer_has_pf = False
    for group in trainer.model.optimizer.param_groups:
        for p in group['params']:
            if id(p) == pf_id:
                optimizer_has_pf = True
                break

    trainable_names = []
    trainable_count = 0
    for n, p in trainer.model.named_parameters():
        if p.requires_grad:
            trainable_names.append(n)
            trainable_count += p.numel()

    print(f"person_feature id: {pf_id}")
    print(f"person_feature requires_grad: {pf_requires_grad}")
    print(f"optimizer contains person_feature: {optimizer_has_pf}")
    print(f"trainable_params: {trainable_names}")
    print(f"trainable_param_count: {trainable_count}")

    if not optimizer_has_pf:
        print("[WARNING] optimizer does NOT contain person_feature!")
    if not pf_requires_grad:
        print("[WARNING] person_feature requires_grad is False!")

    train_log = []
    t_start = time.time()

    for step in range(args.num_steps):
        t_iter_start = time.time()

        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue

        view_groups = defaultdict(list)
        for ps in mv_samples:
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                view_groups[(cam_id, int(frame_idx))].append(pid)

        unique_render_count = len(view_groups)
        trainer.model.optimizer.zero_grad()

        total_loss_teacher = 0.0
        total_valid = 0
        all_alpha_sums = []
        all_denoms = []
        all_clamped = []
        all_pooled_norms = []
        all_cos_fv_tv = []

        for view_idx, ((cam_id, frame_idx), pids) in enumerate(view_groups.items()):
            gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
            if gpu_batch is None:
                continue

            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            view_loss = 0.0
            num_valid_in_view = 0

            for pid in pids:
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
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)

                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )

                if f_v is None:
                    continue

                all_alpha_sums.append(pool_stats.get('alpha_sum', 0))
                all_denoms.append(pool_stats.get('denom', 0))
                if pool_stats.get('clamped', False):
                    all_clamped.append(1)
                else:
                    all_clamped.append(0)
                all_pooled_norms.append(pool_stats.get('pooled_norm', 0))

                teacher_feat = normalize_feat(
                    torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                )
                cos_sim = torch.dot(f_v, teacher_feat)
                all_cos_fv_tv.append(cos_sim.item())

                if args.teacher_loss_type == 'cosine':
                    l_t = 1 - cos_sim
                else:
                    l_t = F.mse_loss(f_v, teacher_feat)

                view_loss = view_loss + l_t
                num_valid_in_view += 1

            if num_valid_in_view > 0:
                view_loss = view_loss / num_valid_in_view
                scaled_loss = args.lambda_teacher * view_loss / unique_render_count
                scaled_loss.backward()

                total_loss_teacher += view_loss.item()
                total_valid += num_valid_in_view

        pf = trainer.model.get_person_feature()
        pf_before = pf.detach().clone()
        param_norm_before = pf.norm().item()

        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]
        grad_max_before = grad_info["grad_max"]

        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
        grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0

        trainer.model.optimizer.step()

        param_norm_after = pf.norm().item()
        delta = pf.detach() - pf_before
        param_delta_norm = delta.norm().item()
        param_delta_max = delta.abs().max().item()
        relative_param_delta = param_delta_norm / max(param_norm_before, 1e-12)

        if grad_norm_before > 1e-12 and param_delta_norm < 1e-20:
            print(f"[WARNING] Step {step}: grad_norm={grad_norm_before:.6e} but param_delta_norm={param_delta_norm:.6e}")

        nan_count = grad_info["nan_count"]
        inf_count = grad_info["inf_count"]

        iter_time = time.time() - t_iter_start

        record = {
            "step": step,
            "loss_teacher": total_loss_teacher / max(1, unique_render_count),
            "cos_fv_tv_mean": np.mean(all_cos_fv_tv) if all_cos_fv_tv else 0,
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "grad_max_before_clip": grad_max_before,
            "param_norm_before": param_norm_before,
            "param_norm_after": param_norm_after,
            "param_delta_norm": param_delta_norm,
            "param_delta_max": param_delta_max,
            "relative_param_delta": relative_param_delta,
            "optimizer_contains_person_feature": optimizer_has_pf,
            "person_feature_requires_grad": pf_requires_grad,
            "trainable_param_names": trainable_names,
            "trainable_param_count": trainable_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "iter_time": iter_time,
        }
        train_log.append(record)

        if step % args.log_interval == 0:
            print(f"[OPTIMIZER] Step {step:5d}: "
                  f"loss={record['loss_teacher']:.4f} "
                  f"cos_tv={record['cos_fv_tv_mean']:.4f} "
                  f"grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                  f"delta_norm={param_delta_norm:.6e} "
                  f"relative_delta={relative_param_delta:.6e} "
                  f"nan/inf={nan_count}/{inf_count}")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r) + "\n")

    summary = {
        "first10_cos_tv": np.mean([r['cos_fv_tv_mean'] for r in train_log[:10]]) if train_log else 0,
        "last10_cos_tv": np.mean([r['cos_fv_tv_mean'] for r in train_log[-10:]]) if train_log else 0,
        "first10_loss": np.mean([r['loss_teacher'] for r in train_log[:10]]) if train_log else 0,
        "last10_loss": np.mean([r['loss_teacher'] for r in train_log[-10:]]) if train_log else 0,
        "mean_grad_norm": np.mean([r['grad_norm_before_clip'] for r in train_log]) if train_log else 0,
        "mean_param_delta": np.mean([r['param_delta_norm'] for r in train_log]) if train_log else 0,
        "mean_relative_delta": np.mean([r['relative_param_delta'] for r in train_log]) if train_log else 0,
        "optimizer_contains_person_feature": optimizer_has_pf,
        "person_feature_requires_grad": pf_requires_grad,
        "total_steps": len(train_log),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    return True


def run_overfit_teacher_single_batch(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("OVERFIT TEACHER SINGLE BATCH MODE")
    print("=" * 70)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mv_samples = sampler.sample_batch()
    if mv_samples is None:
        print("ERROR: Cannot sample valid batch")
        return False

    fixed_samples_path = os.path.join(args.output_dir, "fixed_samples.json")
    fixed_samples_serializable = []
    for ps in mv_samples:
        entry = {
            'person_id': ps['person_id'],
            'views': list(ps['views']),
        }
        fixed_samples_serializable.append(entry)
    with open(fixed_samples_path, 'w') as f:
        json.dump(fixed_samples_serializable, f, indent=2)
    print(f"Fixed samples saved to: {fixed_samples_path}")

    view_groups = defaultdict(list)
    for ps in mv_samples:
        pid = ps['person_id']
        for cam_id, frame_idx in ps['views']:
            view_groups[(cam_id, int(frame_idx))].append(pid)

    print(f"Fixed batch: {len(view_groups)} unique renders, {len(mv_samples)} persons x {len(mv_samples[0]['views'])} views")

    train_log = []
    t_start = time.time()

    for step in range(args.num_steps):
        t_iter_start = time.time()

        trainer.model.optimizer.zero_grad()

        total_loss_teacher = 0.0
        total_valid = 0
        all_alpha_sums = []
        all_denoms = []
        all_clamped = []
        all_pooled_norms = []
        all_cos_fv_tv = []
        all_teacher_norms = []

        for (cam_id, frame_idx), pids in view_groups.items():
            gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
            if gpu_batch is None:
                continue

            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            view_loss = 0.0
            num_valid_in_view = 0

            for pid in pids:
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
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)

                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )

                if f_v is None:
                    continue

                all_alpha_sums.append(pool_stats.get('alpha_sum', 0))
                all_denoms.append(pool_stats.get('denom', 0))
                if pool_stats.get('clamped', False):
                    all_clamped.append(1)
                else:
                    all_clamped.append(0)
                all_pooled_norms.append(pool_stats.get('pooled_norm', 0))

                teacher_feat = normalize_feat(
                    torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                )
                all_teacher_norms.append(teacher_feat.norm().item())
                cos_sim = torch.dot(f_v, teacher_feat)
                all_cos_fv_tv.append(cos_sim.item())

                if args.teacher_loss_type == 'cosine':
                    l_t = 1 - cos_sim
                else:
                    l_t = F.mse_loss(f_v, teacher_feat)

                view_loss = view_loss + l_t
                num_valid_in_view += 1

            if num_valid_in_view > 0:
                view_loss = view_loss / num_valid_in_view
                scaled_loss = args.lambda_teacher * view_loss / len(view_groups)
                scaled_loss.backward()

                total_loss_teacher += view_loss.item()
                total_valid += num_valid_in_view

        pf = trainer.model.get_person_feature()
        pf_before = pf.detach().clone()
        param_norm_before = pf.norm().item()

        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]

        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
        grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0

        trainer.model.optimizer.step()

        param_norm_after = pf.norm().item()
        delta = pf.detach() - pf_before
        param_delta_norm = delta.norm().item()
        param_delta_max = delta.abs().max().item()
        relative_param_delta = param_delta_norm / max(param_norm_before, 1e-12)

        nan_count = grad_info["nan_count"]
        inf_count = grad_info["inf_count"]
        denom_clamp_ratio = sum(all_clamped) / max(len(all_clamped), 1)

        iter_time = time.time() - t_iter_start

        record = {
            "step": step,
            "loss_teacher": total_loss_teacher / max(1, len(view_groups)),
            "cos_fv_tv_mean": np.mean(all_cos_fv_tv) if all_cos_fv_tv else 0,
            "cos_fv_tv_min": min(all_cos_fv_tv) if all_cos_fv_tv else 0,
            "cos_fv_tv_max": max(all_cos_fv_tv) if all_cos_fv_tv else 0,
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "param_delta_norm": param_delta_norm,
            "param_delta_max": param_delta_max,
            "relative_param_delta": relative_param_delta,
            "alpha_sum_min": min(all_alpha_sums) if all_alpha_sums else 0,
            "alpha_sum_mean": np.mean(all_alpha_sums) if all_alpha_sums else 0,
            "alpha_sum_max": max(all_alpha_sums) if all_alpha_sums else 0,
            "denom_clamp_ratio": denom_clamp_ratio,
            "pooled_feature_norm_min": min(all_pooled_norms) if all_pooled_norms else 0,
            "pooled_feature_norm_mean": np.mean(all_pooled_norms) if all_pooled_norms else 0,
            "pooled_feature_norm_max": max(all_pooled_norms) if all_pooled_norms else 0,
            "teacher_feature_norm_min": min(all_teacher_norms) if all_teacher_norms else 0,
            "teacher_feature_norm_mean": np.mean(all_teacher_norms) if all_teacher_norms else 0,
            "teacher_feature_norm_max": max(all_teacher_norms) if all_teacher_norms else 0,
            "valid_feature_count": total_valid,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "iter_time": iter_time,
        }
        train_log.append(record)

        if step % args.log_interval == 0:
            print(f"[OVERFIT] Step {step:5d}: "
                  f"loss={record['loss_teacher']:.4f} "
                  f"cos_tv={record['cos_fv_tv_mean']:.4f} "
                  f"grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                  f"delta={param_delta_norm:.6e} "
                  f"valid={total_valid} "
                  f"nan/inf={nan_count}/{inf_count} "
                  f"t={iter_time:.2f}s")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r) + "\n")

    first_cos = np.mean([r['cos_fv_tv_mean'] for r in train_log[:10]]) if len(train_log) >= 10 else 0
    last_cos = np.mean([r['cos_fv_tv_mean'] for r in train_log[-10:]]) if len(train_log) >= 10 else 0
    first_loss = np.mean([r['loss_teacher'] for r in train_log[:10]]) if len(train_log) >= 10 else 0
    last_loss = np.mean([r['loss_teacher'] for r in train_log[-10:]]) if len(train_log) >= 10 else 0
    mean_delta = np.mean([r['param_delta_norm'] for r in train_log]) if train_log else 0
    total_nan = sum(r['nan_count'] for r in train_log)
    total_inf = sum(r['inf_count'] for r in train_log)

    overfit_success = (last_cos > first_cos + 0.05) and (last_loss < first_loss - 0.05) and (mean_delta > 1e-12) and (total_nan == 0) and (total_inf == 0)

    summary = {
        "first10_cos_tv": first_cos,
        "last10_cos_tv": last_cos,
        "cos_tv_delta": last_cos - first_cos,
        "first10_loss": first_loss,
        "last10_loss": last_loss,
        "loss_delta": last_loss - first_loss,
        "mean_param_delta": mean_delta,
        "total_nan": total_nan,
        "total_inf": total_inf,
        "overfit_success": overfit_success,
        "total_steps": len(train_log),
    }
    if not overfit_success:
        summary["warning"] = (
            f"Single-batch overfit FAILED. "
            f"cos_tv delta={last_cos-first_cos:+.4f} (need >+0.05), "
            f"loss delta={last_loss-first_loss:+.4f} (need <-0.05), "
            f"mean_param_delta={mean_delta:.6e} (need >1e-12), "
            f"nan={total_nan}, inf={total_inf}. "
            f"Do NOT continue to multi-view or Proto CE."
        )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"OVERFIT RESULT SUMMARY")
    print(f"{'='*70}")
    print(f"cos_tv: first10={first_cos:.4f}, last10={last_cos:.4f}, delta={last_cos-first_cos:+.4f}")
    print(f"loss:   first10={first_loss:.4f}, last10={last_loss:.4f}, delta={last_loss-first_loss:+.4f}")
    print(f"mean_param_delta: {mean_delta:.6e}")
    print(f"nan={total_nan}, inf={total_inf}")
    print(f"OVERFIT SUCCESS: {overfit_success}")
    if not overfit_success:
        print(summary["warning"])

    print(f"\nMetrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    return True


def run_train_teacher_fixed_eval(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("TRAIN TEACHER FIXED EVAL MODE")
    print("=" * 70)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    eval_samples = sampler.sample_batch()
    if eval_samples is None:
        print("ERROR: Cannot sample eval batch")
        return False

    fixed_eval_path = os.path.join(args.output_dir, "fixed_eval_samples.json")
    fixed_eval_serializable = []
    for ps in eval_samples:
        entry = {
            'person_id': ps['person_id'],
            'views': list(ps['views']),
        }
        fixed_eval_serializable.append(entry)
    with open(fixed_eval_path, 'w') as f:
        json.dump(fixed_eval_serializable, f, indent=2)
    print(f"Fixed eval samples saved to: {fixed_eval_path}")

    eval_view_groups = defaultdict(list)
    for ps in eval_samples:
        pid = ps['person_id']
        for cam_id, frame_idx in ps['views']:
            eval_view_groups[(cam_id, int(frame_idx))].append(pid)

    train_log = []
    t_start = time.time()

    for step in range(args.num_steps):
        t_iter_start = time.time()

        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue

        train_view_groups = defaultdict(list)
        for ps in mv_samples:
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                train_view_groups[(cam_id, int(frame_idx))].append(pid)

        unique_render_count = len(train_view_groups)
        trainer.model.optimizer.zero_grad()

        total_loss_teacher = 0.0
        total_valid = 0
        all_cos_fv_tv = []

        for (cam_id, frame_idx), pids in train_view_groups.items():
            gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
            if gpu_batch is None:
                continue

            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            view_loss = 0.0
            num_valid_in_view = 0

            for pid in pids:
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
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)

                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )

                if f_v is None:
                    continue

                teacher_feat = normalize_feat(
                    torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                )
                cos_sim = torch.dot(f_v, teacher_feat)
                all_cos_fv_tv.append(cos_sim.item())

                if args.teacher_loss_type == 'cosine':
                    l_t = 1 - cos_sim
                else:
                    l_t = F.mse_loss(f_v, teacher_feat)

                view_loss = view_loss + l_t
                num_valid_in_view += 1

            if num_valid_in_view > 0:
                view_loss = view_loss / num_valid_in_view
                scaled_loss = args.lambda_teacher * view_loss / unique_render_count
                scaled_loss.backward()

                total_loss_teacher += view_loss.item()
                total_valid += num_valid_in_view

        pf = trainer.model.get_person_feature()
        pf_before = pf.detach().clone()

        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]

        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
        grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0

        trainer.model.optimizer.step()

        delta = pf.detach() - pf_before
        param_delta_norm = delta.norm().item()

        nan_count = grad_info["nan_count"]
        inf_count = grad_info["inf_count"]

        iter_time = time.time() - t_iter_start

        record = {
            "step": step,
            "train_loss_teacher": total_loss_teacher / max(1, unique_render_count),
            "train_cos_fv_tv_mean": np.mean(all_cos_fv_tv) if all_cos_fv_tv else 0,
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "param_delta_norm": param_delta_norm,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "iter_time": iter_time,
        }

        if step % args.eval_interval == 0 or step == args.num_steps - 1:
            with torch.no_grad():
                eval_total_loss = 0.0
                eval_valid = 0
                eval_cos_fv_tv = []
                eval_alpha_sums = []
                eval_denoms = []
                eval_clamped = 0

                for (cam_id, frame_idx), pids in eval_view_groups.items():
                    gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
                    if gpu_batch is None:
                        continue

                    render_out = trainer.model(
                        gpu_batch, train=False, frame_id=0, render_person_feature=True
                    )
                    person_feature_map = render_out['person_feature_map']
                    person_opacity_map = render_out.get('person_opacity_map')

                    for pid in pids:
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
                        bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)

                        f_v, pool_stats = opacity_roi_pooling(
                            person_feature_map, person_opacity_map, bbox_t,
                            denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                        )

                        if f_v is None:
                            continue

                        teacher_feat = normalize_feat(
                            torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                        )
                        cos_sim = torch.dot(f_v, teacher_feat)
                        eval_cos_fv_tv.append(cos_sim.item())
                        eval_alpha_sums.append(pool_stats.get('alpha_sum', 0))
                        eval_denoms.append(pool_stats.get('denom', 0))
                        if pool_stats.get('clamped', False):
                            eval_clamped += 1
                        eval_valid += 1

                eval_loss = (1 - np.mean(eval_cos_fv_tv)) if eval_cos_fv_tv else 0
                record['fixed_eval_loss_teacher'] = eval_loss
                record['fixed_eval_cos_fv_tv_mean'] = np.mean(eval_cos_fv_tv) if eval_cos_fv_tv else 0
                record['fixed_eval_cos_fv_tv_min'] = min(eval_cos_fv_tv) if eval_cos_fv_tv else 0
                record['fixed_eval_cos_fv_tv_max'] = max(eval_cos_fv_tv) if eval_cos_fv_tv else 0
                record['fixed_eval_valid_feature_count'] = eval_valid
                record['fixed_eval_alpha_sum_min'] = min(eval_alpha_sums) if eval_alpha_sums else 0
                record['fixed_eval_alpha_sum_mean'] = np.mean(eval_alpha_sums) if eval_alpha_sums else 0
                record['fixed_eval_alpha_sum_max'] = max(eval_alpha_sums) if eval_alpha_sums else 0
                record['fixed_eval_denom_clamp_ratio'] = eval_clamped / max(eval_valid, 1)

        train_log.append(record)

        if step % args.log_interval == 0:
            log_msg = (f"[FIXED_EVAL] Step {step:5d}: "
                       f"train_loss={record['train_loss_teacher']:.4f} "
                       f"train_cos={record['train_cos_fv_tv_mean']:.4f} "
                       f"grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                       f"delta={param_delta_norm:.6e} "
                       f"valid={total_valid}")
            if 'fixed_eval_cos_fv_tv_mean' in record:
                log_msg += (f" | eval_loss={record['fixed_eval_loss_teacher']:.4f} "
                           f"eval_cos={record['fixed_eval_cos_fv_tv_mean']:.4f}")
            log_msg += f" t={iter_time:.2f}s"
            print(log_msg)

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r) + "\n")

    eval_records = [r for r in train_log if 'fixed_eval_cos_fv_tv_mean' in r]
    if eval_records:
        first_eval_cos = eval_records[0]['fixed_eval_cos_fv_tv_mean']
        last_eval_cos = eval_records[-1]['fixed_eval_cos_fv_tv_mean']
        eval_cos_delta = last_eval_cos - first_eval_cos
        print(f"\nFixed eval cos_tv: first={first_eval_cos:.4f}, last={last_eval_cos:.4f}, delta={eval_cos_delta:+.4f}")
        if eval_cos_delta > 0.02:
            print("  -> Positive learning trend detected!")
        else:
            print("  -> No significant learning trend.")

    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        "total_steps": len(train_log),
        "eval_records_count": len(eval_records),
        "first_eval_cos": eval_records[0]['fixed_eval_cos_fv_tv_mean'] if eval_records else 0,
        "last_eval_cos": eval_records[-1]['fixed_eval_cos_fv_tv_mean'] if eval_records else 0,
        "eval_cos_delta": eval_cos_delta if eval_records else 0,
        "learning_trend_positive": eval_records[-1]['fixed_eval_cos_fv_tv_mean'] > eval_records[0]['fixed_eval_cos_fv_tv_mean'] + 0.02 if len(eval_records) >= 2 else False,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    return True


def run_teacher_gap_diagnostic(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("TEACHER GAP DIAGNOSTIC MODE")
    print("=" * 70)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_same_cos = []
    all_diff_cos = []
    positive_count = 0
    negative_count = 0
    valid_teacher_count = 0

    for step in range(args.num_steps):
        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue

        view_groups = defaultdict(list)
        for ps in mv_samples:
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                view_groups[(cam_id, int(frame_idx))].append(pid)

        person_view_features = []
        person_view_ids = []
        person_view_pids = []

        for (cam_id, frame_idx), pids in view_groups.items():
            gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
            if gpu_batch is None:
                continue

            for pid in pids:
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

                teacher_feat = normalize_feat(
                    torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                )
                person_view_features.append(teacher_feat)
                person_view_ids.append((pid, cam_id, int(frame_idx)))
                person_view_pids.append(pid)
                valid_teacher_count += 1

        if len(person_view_features) < 2:
            continue

        for i in range(len(person_view_features)):
            for j in range(i + 1, len(person_view_features)):
                cos_ij = torch.dot(person_view_features[i], person_view_features[j]).item()
                if person_view_pids[i] == person_view_pids[j]:
                    if person_view_ids[i][1] != person_view_ids[j][1] or person_view_ids[i][2] != person_view_ids[j][2]:
                        all_same_cos.append(cos_ij)
                        positive_count += 1
                else:
                    all_diff_cos.append(cos_ij)
                    negative_count += 1

    same_mean = np.mean(all_same_cos) if all_same_cos else 0
    same_min = min(all_same_cos) if all_same_cos else 0
    same_max = max(all_same_cos) if all_same_cos else 0
    diff_mean = np.mean(all_diff_cos) if all_diff_cos else 0
    diff_min = min(all_diff_cos) if all_diff_cos else 0
    diff_max = max(all_diff_cos) if all_diff_cos else 0
    gap = same_mean - diff_mean

    print(f"\n{'='*70}")
    print(f"TEACHER GAP DIAGNOSTIC RESULT")
    print(f"{'='*70}")
    print(f"Valid teacher features: {valid_teacher_count}")
    print(f"Same-person pairs: {positive_count}")
    print(f"Diff-person pairs: {negative_count}")
    print(f"Same cos: mean={same_mean:.4f}, min={same_min:.4f}, max={same_max:.4f}")
    print(f"Diff cos: mean={diff_mean:.4f}, min={diff_min:.4f}, max={diff_max:.4f}")
    print(f"Teacher GAP = same - diff = {gap:+.4f}")

    if gap > 0:
        print("  -> Teacher has cross-view identity discriminative signal. OK to align.")
    else:
        print("  -> Teacher noise is high. Consider ID prototype / teacher center instead of direct per-view alignment.")

    summary = {
        "teacher_same_cos_mean": float(same_mean),
        "teacher_same_cos_min": float(same_min),
        "teacher_same_cos_max": float(same_max),
        "teacher_diff_cos_mean": float(diff_mean),
        "teacher_diff_cos_min": float(diff_min),
        "teacher_diff_cos_max": float(diff_max),
        "teacher_gap": float(gap),
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "valid_teacher_feature_count": valid_teacher_count,
        "teacher_gap_positive": bool(gap > 0),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        json.dump(summary, f)

    print(f"\nSummary: {summary_path}")
    print(f"Metrics: {metrics_path}")
    return True


def collect_clean_candidates(args, trainer, sampler, batch_builder, allowed_cameras=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    candidates = []
    invalid_reasons = defaultdict(int)
    n_candidates = 0

    for _ in range(args.candidate_multiplier * max(args.P, 4)):
        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue
        for ps in mv_samples:
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                if allowed_cameras and cam_id not in allowed_cameras:
                    continue
                gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, int(frame_idx))
                if gpu_batch is None:
                    invalid_reasons['no_batch'] += 1
                    continue
                inst = None
                for i in gpu_batch.instances:
                    if i.get('train_id') == pid:
                        inst = i
                        break
                if inst is None:
                    invalid_reasons['no_instance'] += 1
                    continue
                teacher_emb = inst.get('teacher_embedding')
                if teacher_emb is None:
                    invalid_reasons['no_teacher'] += 1
                    continue
                bbox = inst['bbox_xyxy']
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if bbox_area < args.min_bbox_area:
                    invalid_reasons['bbox_too_small'] += 1
                    continue
                render_out = trainer.model(
                    gpu_batch, train=False, frame_id=0, render_person_feature=True
                )
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)
                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )
                if f_v is None or torch.isnan(f_v).any() or torch.isinf(f_v).any():
                    invalid_reasons['invalid_pooled'] += 1
                    continue
                teacher_feat = normalize_feat(
                    torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                )
                if torch.isnan(teacher_feat).any() or torch.isinf(teacher_feat).any():
                    invalid_reasons['invalid_teacher'] += 1
                    continue
                alpha_sum = pool_stats.get('alpha_sum', 0)
                denom = pool_stats.get('denom', 0)
                clamped = pool_stats.get('clamped', False)
                if args.require_unclamped and clamped:
                    invalid_reasons['clamped'] += 1
                    continue
                if alpha_sum < args.min_alpha_sum:
                    invalid_reasons['alpha_too_small'] += 1
                    continue
                n_candidates += 1
                candidates.append({
                    'person_id': pid,
                    'cam_id': cam_id,
                    'frame_idx': int(frame_idx),
                    'bbox': list(bbox),
                    'bbox_area': float(bbox_area),
                    'alpha_sum': float(alpha_sum),
                    'denom': float(denom),
                    'clamped': bool(clamped),
                    'pooled_norm': float(pool_stats.get('pooled_norm', 0)),
                    'teacher_norm': float(teacher_feat.norm().item()),
                    'cos_fv_tv': float(torch.dot(f_v, teacher_feat).item()),
                    'teacher_emb': teacher_emb,
                })

    return candidates, dict(invalid_reasons), n_candidates


def run_single_roi_overfit(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("SINGLE ROI OVERFIT MODE")
    print("=" * 70)

    print("Collecting clean ROI candidates...")
    candidates, invalid_reasons, n_candidates = collect_clean_candidates(args, trainer, sampler, batch_builder)
    print(f"Candidates: {n_candidates}, Invalid reasons: {invalid_reasons}")

    if not candidates:
        print("ERROR: No clean ROI candidates found!")
        return False

    best_roi = candidates[0]
    print(f"Selected ROI: person={best_roi['person_id']}, cam={best_roi['cam_id']}, "
          f"frame={best_roi['frame_idx']}, alpha_sum={best_roi['alpha_sum']:.4f}, "
          f"cos={best_roi['cos_fv_tv']:.4f}")

    train_log = []
    t_start = time.time()

    for step in range(args.num_steps):
        t_iter_start = time.time()

        cam_id = best_roi['cam_id']
        frame_idx = best_roi['frame_idx']

        gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
        if gpu_batch is None:
            continue

        render_out = trainer.model(
            gpu_batch, train=False, frame_id=0, render_person_feature=True
        )
        person_feature_map = render_out['person_feature_map']
        person_opacity_map = render_out.get('person_opacity_map')

        bbox_t = torch.tensor(best_roi['bbox'], dtype=torch.float32, device=trainer.device)
        f_v, pool_stats = opacity_roi_pooling(
            person_feature_map, person_opacity_map, bbox_t,
            denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
        )

        if f_v is None:
            continue

        teacher_feat = normalize_feat(
            torch.as_tensor(best_roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
        )
        cos_sim = torch.dot(f_v, teacher_feat)

        if args.teacher_loss_type == 'cosine':
            loss = 1 - cos_sim
        else:
            loss = F.mse_loss(f_v, teacher_feat)

        loss_scaled = args.lambda_teacher * loss
        loss_scaled.backward()

        pf = trainer.model.get_person_feature()
        pf_before = pf.detach().clone()
        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]
        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
        grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0
        trainer.model.optimizer.step()

        delta = pf.detach() - pf_before
        param_delta_norm = delta.norm().item()
        nan_count = grad_info["nan_count"]
        inf_count = grad_info["inf_count"]

        record = {
            "step": step,
            "loss_teacher": float(loss.item()),
            "cos_fv_tv": float(cos_sim.item()),
            "alpha_sum": float(pool_stats.get('alpha_sum', 0)),
            "denom": float(pool_stats.get('denom', 0)),
            "clamped": bool(pool_stats.get('clamped', False)),
            "pooled_feature_norm": float(pool_stats.get('pooled_norm', 0)),
            "teacher_feature_norm": float(teacher_feat.norm().item()),
            "grad_norm_before_clip": float(grad_norm_before),
            "grad_norm_after_clip": float(grad_norm_after),
            "param_delta_norm": float(param_delta_norm),
            "nan_count": int(nan_count),
            "inf_count": int(inf_count),
            "iter_time": float(time.time() - t_iter_start),
        }
        train_log.append(record)

        if step % args.log_interval == 0:
            print(f"[SINGLE_ROI] Step {step:5d}: loss={record['loss_teacher']:.4f} "
                  f"cos={record['cos_fv_tv']:.4f} grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                  f"delta={param_delta_norm:.6e} alpha={record['alpha_sum']:.4f} "
                  f"nan/inf={nan_count}/{inf_count} t={record['iter_time']:.2f}s")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r) + "\n")

    first_cos = float(np.mean([r['cos_fv_tv'] for r in train_log[:10]])) if len(train_log) >= 10 else 0
    last_cos = float(np.mean([r['cos_fv_tv'] for r in train_log[-10:]])) if len(train_log) >= 10 else 0
    first_loss = float(np.mean([r['loss_teacher'] for r in train_log[:10]])) if len(train_log) >= 10 else 0
    last_loss = float(np.mean([r['loss_teacher'] for r in train_log[-10:]])) if len(train_log) >= 10 else 0
    mean_delta = float(np.mean([r['param_delta_norm'] for r in train_log])) if train_log else 0
    total_nan = int(sum(r['nan_count'] for r in train_log))
    total_inf = int(sum(r['inf_count'] for r in train_log))
    success = bool((last_cos > first_cos + 0.05) and (last_loss < first_loss - 0.05) and (mean_delta > 1e-12) and (total_nan == 0) and (total_inf == 0))

    summary = {
        "roi_person_id": int(best_roi['person_id']),
        "roi_cam_id": str(best_roi['cam_id']),
        "roi_frame_idx": int(best_roi['frame_idx']),
        "roi_alpha_sum": float(best_roi['alpha_sum']),
        "candidate_count": int(n_candidates),
        "invalid_reasons": invalid_reasons,
        "first10_cos": first_cos,
        "last10_cos": last_cos,
        "cos_delta": float(last_cos - first_cos),
        "first10_loss": first_loss,
        "last10_loss": last_loss,
        "loss_delta": float(last_loss - first_loss),
        "mean_param_delta": mean_delta,
        "total_nan": total_nan,
        "total_inf": total_inf,
        "overfit_success": success,
        "total_steps": len(train_log),
    }
    if not success:
        summary["warning"] = (
            f"Single-ROI overfit FAILED. cos_delta={last_cos-first_cos:+.4f} (need >+0.05), "
            f"loss_delta={last_loss-first_loss:+.4f} (need <-0.05). "
            f"Check: teacher/bbox/camera alignment, person_feature parameter binding, "
            f"ROI pooling dependency on person_feature, normalize/projection."
        )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SINGLE ROI OVERFIT RESULT")
    print(f"{'='*70}")
    print(f"ROI: person={best_roi['person_id']}, cam={best_roi['cam_id']}, alpha={best_roi['alpha_sum']:.4f}")
    print(f"cos: first10={first_cos:.4f}, last10={last_cos:.4f}, delta={last_cos-first_cos:+.4f}")
    print(f"loss: first10={first_loss:.4f}, last10={last_loss:.4f}, delta={last_loss-first_loss:+.4f}")
    print(f"mean_param_delta: {mean_delta:.6e}")
    print(f"SUCCESS: {success}")
    if not success:
        print(summary.get("warning", ""))

    return True


def run_progressive_overfit(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("PROGRESSIVE OVERFIT MODE")
    print("=" * 70)

    stages = [
        {"name": "1_roi", "n_rois": 1},
        {"name": "2_roi_same_person", "n_rois": 2},
        {"name": "4_roi_2persons", "n_rois": 4},
        {"name": "8_roi_4persons", "n_rois": 8},
    ]

    print("Collecting clean ROI candidates...")
    candidates, invalid_reasons, n_candidates = collect_clean_candidates(args, trainer, sampler, batch_builder)
    print(f"Candidates: {n_candidates}")

    if len(candidates) < 8:
        print(f"WARNING: Only {len(candidates)} candidates, reducing stage requirements")

    stage_summaries = []

    for stage in stages:
        n_rois = min(stage["n_rois"], len(candidates))
        stage_candidates = candidates[:n_rois]

        print(f"\n{'='*50}")
        print(f"Stage: {stage['name']} ({n_rois} ROIs)")
        print(f"{'='*50}")

        stage_output_dir = args.output_dir.rstrip('/') + f"_{stage['name']}"
        os.makedirs(stage_output_dir, exist_ok=True)

        stage_log = []
        for step in range(args.num_steps):
            t_iter_start = time.time()
            trainer.model.optimizer.zero_grad()

            stage_loss = 0.0
            stage_cos = []
            stage_valid = 0

            for roi in stage_candidates:
                gpu_batch = batch_builder.get_batch_by_cam_frame(roi['cam_id'], roi['frame_idx'])
                if gpu_batch is None:
                    continue
                render_out = trainer.model(
                    gpu_batch, train=False, frame_id=0, render_person_feature=True
                )
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')
                bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )
                if f_v is None:
                    continue
                teacher_feat = normalize_feat(
                    torch.as_tensor(roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
                )
                cos_sim = torch.dot(f_v, teacher_feat)
                if args.teacher_loss_type == 'cosine':
                    l = 1 - cos_sim
                else:
                    l = F.mse_loss(f_v, teacher_feat)
                stage_loss = stage_loss + l
                stage_cos.append(float(cos_sim.item()))
                stage_valid += 1

            if stage_valid == 0:
                continue

            avg_loss = stage_loss / stage_valid
            avg_loss.backward()

            pf = trainer.model.get_person_feature()
            pf_before = pf.detach().clone()
            grad_info = get_grad_stats(pf)
            grad_norm_before = grad_info["grad_norm"]
            torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
            grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0
            trainer.model.optimizer.step()
            delta = pf.detach() - pf_before
            param_delta_norm = float(delta.norm().item())

            record = {
                "step": step,
                "loss_teacher": float(avg_loss.item()),
                "cos_fv_tv_mean": float(np.mean(stage_cos)) if stage_cos else 0,
                "grad_norm_before_clip": float(grad_norm_before),
                "grad_norm_after_clip": float(grad_norm_after),
                "param_delta_norm": param_delta_norm,
                "valid_feature_count": stage_valid,
                "nan_count": int(grad_info["nan_count"]),
                "inf_count": int(grad_info["inf_count"]),
                "iter_time": float(time.time() - t_iter_start),
            }
            stage_log.append(record)

            if step % args.log_interval == 0:
                print(f"[{stage['name'].upper()}] Step {step:5d}: loss={record['loss_teacher']:.4f} "
                      f"cos={record['cos_fv_tv_mean']:.4f} grad={grad_norm_before:.4e} "
                      f"delta={param_delta_norm:.6e} valid={stage_valid}")

        stage_metrics_path = os.path.join(stage_output_dir, "metrics.jsonl")
        with open(stage_metrics_path, 'w') as f:
            for r in stage_log:
                f.write(json.dumps(r) + "\n")

        first_cos = float(np.mean([r['cos_fv_tv_mean'] for r in stage_log[:10]])) if len(stage_log) >= 10 else 0
        last_cos = float(np.mean([r['cos_fv_tv_mean'] for r in stage_log[-10:]])) if len(stage_log) >= 10 else 0
        first_loss = float(np.mean([r['loss_teacher'] for r in stage_log[:10]])) if len(stage_log) >= 10 else 0
        last_loss = float(np.mean([r['loss_teacher'] for r in stage_log[-10:]])) if len(stage_log) >= 10 else 0
        mean_delta = float(np.mean([r['param_delta_norm'] for r in stage_log])) if stage_log else 0
        success = bool((last_cos > first_cos + 0.05) and (last_loss < first_loss - 0.05))

        failure_reason = ""
        if not success:
            if last_cos <= first_cos + 0.05:
                failure_reason = "cos not improving"
            if last_loss >= first_loss - 0.05:
                failure_reason += (" and " if failure_reason else "") + "loss not decreasing"

        stage_summary = {
            "stage_name": stage['name'],
            "requested_count": int(stage['n_rois']),
            "valid_count": int(stage_valid) if stage_log else 0,
            "cos_start": first_cos,
            "cos_end": last_cos,
            "cos_delta": float(last_cos - first_cos),
            "loss_start": first_loss,
            "loss_end": last_loss,
            "loss_delta": float(last_loss - first_loss),
            "mean_param_delta": mean_delta,
            "success": success,
            "failure_reason": failure_reason,
        }
        stage_summaries.append(stage_summary)

        stage_summary_path = os.path.join(stage_output_dir, "summary.json")
        with open(stage_summary_path, 'w') as f:
            json.dump(stage_summary, f, indent=2)

        print(f"  -> cos: {first_cos:.4f} -> {last_cos:.4f} (delta={last_cos-first_cos:+.4f})")
        print(f"  -> loss: {first_loss:.4f} -> {last_loss:.4f} (delta={last_loss-first_loss:+.4f})")
        print(f"  -> SUCCESS: {success}")

    print(f"\n{'='*70}")
    print(f"PROGRESSIVE OVERFIT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Stage':<25} {'n_rois':<8} {'cos_delta':<12} {'loss_delta':<12} {'Success':<10}")
    for s in stage_summaries:
        print(f"{s['stage_name']:<25} {s['requested_count']:<8} {s['cos_delta']:+.4f}    "
              f"{s['loss_delta']:+.4f}    {s['success']}")

    overall_summary_path = os.path.join(args.output_dir, "summary.json")
    with open(overall_summary_path, 'w') as f:
        json.dump({"stages": stage_summaries}, f, indent=2)

    return True


def run_clean_8roi_overfit(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("CLEAN 8 ROI OVERFIT MODE")
    print("=" * 70)

    print("Collecting clean ROI candidates...")
    target = args.target_valid_count or (args.P * args.K)
    orig_mult = args.candidate_multiplier
    args.candidate_multiplier = max(orig_mult, 20)
    candidates, invalid_reasons, n_candidates = collect_clean_candidates(args, trainer, sampler, batch_builder)
    args.candidate_multiplier = orig_mult
    print(f"Candidates: {n_candidates}, Invalid reasons: {invalid_reasons}")

    candidates_sorted = sorted(candidates, key=lambda c: c['alpha_sum'], reverse=True)
    if len(candidates_sorted) < target:
        print(f"WARNING: Only {len(candidates_sorted)} clean candidates, using all of them")
        selected = candidates_sorted
    else:
        selected = candidates_sorted[:target]

    print(f"Selected {len(selected)} ROIs for overfit")
    for i, roi in enumerate(selected):
        print(f"  [{i}] person={roi['person_id']}, cam={roi['cam_id']}, "
              f"frame={roi['frame_idx']}, alpha={roi['alpha_sum']:.4f}, cos={roi['cos_fv_tv']:.4f}")

    fixed_samples_path = os.path.join(args.output_dir, "fixed_samples.json")
    with open(fixed_samples_path, 'w') as f:
        json.dump(selected, f, indent=2, default=str)

    quality = {
        "selected_count": len(selected),
        "candidate_count": n_candidates,
        "invalid_reason_counts": invalid_reasons,
        "alpha_sum": {
            "min": min(r['alpha_sum'] for r in selected) if selected else 0,
            "max": max(r['alpha_sum'] for r in selected) if selected else 0,
            "mean": np.mean([r['alpha_sum'] for r in selected]) if selected else 0,
        },
        "bbox_area": {
            "min": min(r['bbox_area'] for r in selected) if selected else 0,
            "max": max(r['bbox_area'] for r in selected) if selected else 0,
            "mean": np.mean([r['bbox_area'] for r in selected]) if selected else 0,
        },
        "teacher_norm": {
            "min": min(r['teacher_norm'] for r in selected) if selected else 0,
            "max": max(r['teacher_norm'] for r in selected) if selected else 0,
            "mean": np.mean([r['teacher_norm'] for r in selected]) if selected else 0,
        },
        "pooled_norm": {
            "min": min(r['pooled_norm'] for r in selected) if selected else 0,
            "max": max(r['pooled_norm'] for r in selected) if selected else 0,
            "mean": np.mean([r['pooled_norm'] for r in selected]) if selected else 0,
        },
    }
    quality_path = os.path.join(args.output_dir, "fixed_samples_quality.json")
    with open(quality_path, 'w') as f:
        json.dump(quality, f, indent=2)

    train_log = []
    t_start = time.time()

    for step in range(args.num_steps):
        t_iter_start = time.time()
        trainer.model.optimizer.zero_grad()

        step_loss = 0.0
        step_cos = []
        step_alpha = []
        step_denom = []
        step_clamped = []
        valid_count = 0
        invalid_count = 0
        nan_count_total = 0
        inf_count_total = 0

        for roi in selected:
            gpu_batch = batch_builder.get_batch_by_cam_frame(roi['cam_id'], roi['frame_idx'])
            if gpu_batch is None:
                invalid_count += 1
                continue

            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
            f_v, pool_stats = opacity_roi_pooling(
                person_feature_map, person_opacity_map, bbox_t,
                denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
            )

            if f_v is None:
                invalid_count += 1
                continue

            teacher_feat = normalize_feat(
                torch.as_tensor(roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
            )
            cos_sim = torch.dot(f_v, teacher_feat)
            step_cos.append(float(cos_sim.item()))
            step_alpha.append(float(pool_stats.get('alpha_sum', 0)))
            step_denom.append(float(pool_stats.get('denom', 0)))
            step_clamped.append(int(pool_stats.get('clamped', False)))

            if args.teacher_loss_type == 'cosine':
                l = 1 - cos_sim
            else:
                l = F.mse_loss(f_v, teacher_feat)
            step_loss = step_loss + l
            valid_count += 1

        if valid_count == 0:
            if step % args.log_interval == 0:
                print(f"[CLEAN_8ROI] Step {step}: WARNING: no valid ROI, skipping backward")
            continue

        avg_loss = step_loss / valid_count
        avg_loss.backward()

        pf = trainer.model.get_person_feature()
        pf_before = pf.detach().clone()
        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]
        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
        grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0
        trainer.model.optimizer.step()
        delta = pf.detach() - pf_before
        param_delta_norm = float(delta.norm().item())

        denom_clamp_ratio = sum(step_clamped) / max(len(step_clamped), 1)

        record = {
            "step": step,
            "loss_teacher": float(avg_loss.item()),
            "cos_fv_tv_mean": float(np.mean(step_cos)) if step_cos else 0,
            "cos_fv_tv_min": float(min(step_cos)) if step_cos else 0,
            "cos_fv_tv_max": float(max(step_cos)) if step_cos else 0,
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "param_delta_norm": param_delta_norm,
            "alpha_sum_min": min(step_alpha) if step_alpha else 0,
            "alpha_sum_mean": float(np.mean(step_alpha)) if step_alpha else 0,
            "alpha_sum_max": max(step_alpha) if step_alpha else 0,
            "denom_clamp_ratio": denom_clamp_ratio,
            "valid_feature_count": valid_count,
            "invalid_feature_count": invalid_count,
            "nan_count": int(grad_info["nan_count"]),
            "inf_count": int(grad_info["inf_count"]),
            "iter_time": float(time.time() - t_iter_start),
        }
        train_log.append(record)

        if step % args.log_interval == 0:
            print(f"[CLEAN_8ROI] Step {step:5d}: loss={record['loss_teacher']:.4f} "
                  f"cos={record['cos_fv_tv_mean']:.4f} "
                  f"grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                  f"delta={param_delta_norm:.6e} valid={valid_count}/{len(selected)} "
                  f"nan/inf={record['nan_count']}/{record['inf_count']} "
                  f"t={record['iter_time']:.2f}s")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r) + "\n")

    first_cos = float(np.mean([r['cos_fv_tv_mean'] for r in train_log[:10]])) if len(train_log) >= 10 else 0
    last_cos = float(np.mean([r['cos_fv_tv_mean'] for r in train_log[-10:]])) if len(train_log) >= 10 else 0
    first_loss = float(np.mean([r['loss_teacher'] for r in train_log[:10]])) if len(train_log) >= 10 else 0
    last_loss = float(np.mean([r['loss_teacher'] for r in train_log[-10:]])) if len(train_log) >= 10 else 0
    mean_delta = float(np.mean([r['param_delta_norm'] for r in train_log])) if train_log else 0
    total_nan = int(sum(r['nan_count'] for r in train_log))
    total_inf = int(sum(r['inf_count'] for r in train_log))
    success = bool((last_cos > first_cos + 0.05) and (last_loss < first_loss - 0.05) and
                   (total_nan == 0) and (total_inf == 0))

    summary = {
        "selected_count": len(selected),
        "candidate_count": n_candidates,
        "invalid_reasons": invalid_reasons,
        "first10_cos": first_cos,
        "last10_cos": last_cos,
        "cos_delta": float(last_cos - first_cos),
        "first10_loss": first_loss,
        "last10_loss": last_loss,
        "loss_delta": float(last_loss - first_loss),
        "mean_param_delta": mean_delta,
        "total_nan": total_nan,
        "total_inf": total_inf,
        "overfit_success": success,
        "total_steps": len(train_log),
    }
    if not success:
        summary["warning"] = (
            f"Clean 8-ROI overfit FAILED. cos_delta={last_cos-first_cos:+.4f} (need >+0.05), "
            f"loss_delta={last_loss-first_loss:+.4f} (need <-0.05). "
            f"Check: multi-ROI gradient conflict, sample quality, alpha_sum too low, teacher conflict."
        )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"CLEAN 8 ROI OVERFIT RESULT")
    print(f"{'='*70}")
    print(f"Selected ROIs: {len(selected)}")
    print(f"cos: first10={first_cos:.4f}, last10={last_cos:.4f}, delta={last_cos-first_cos:+.4f}")
    print(f"loss: first10={first_loss:.4f}, last10={last_loss:.4f}, delta={last_loss-first_loss:+.4f}")
    print(f"mean_param_delta: {mean_delta:.6e}")
    print(f"SUCCESS: {success}")
    if not success:
        print(summary.get("warning", ""))

    return True


def run_roi_quality_scan(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("ROI QUALITY SCAN MODE")
    print("=" * 70)

    num_scan = getattr(args, 'num_scan_samples', 5000)
    print(f"Scanning {num_scan} ROI samples...")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    roi_metrics = []
    invalid_reasons = defaultdict(int)
    valid_count = 0
    total_count = 0

    cam_stats = defaultdict(lambda: {"valid": 0, "total": 0})
    person_stats = defaultdict(lambda: {"valid": 0, "total": 0})

    scan_iters = 0
    max_iters = num_scan * 5

    while total_count < num_scan and scan_iters < max_iters:
        scan_iters += 1
        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue

        for ps in mv_samples:
            if total_count >= num_scan:
                break
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                if total_count >= num_scan:
                    break

                gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, int(frame_idx))
                if gpu_batch is None:
                    total_count += 1
                    invalid_reasons['no_batch'] += 1
                    cam_stats[cam_id]['total'] += 1
                    continue

                inst = None
                for i in gpu_batch.instances:
                    if i.get('train_id') == pid:
                        inst = i
                        break
                if inst is None:
                    total_count += 1
                    invalid_reasons['no_instance'] += 1
                    continue

                teacher_emb = inst.get('teacher_embedding')
                bbox = inst['bbox_xyxy']
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                render_out = trainer.model(
                    gpu_batch, train=False, frame_id=0, render_person_feature=True
                )
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')

                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)
                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )

                cam_stats[cam_id]['total'] += 1
                person_stats[pid]['total'] += 1

                total_count += 1

                if f_v is None:
                    invalid_reasons['roi_pool_failed'] += 1
                    roi_metrics.append({
                        'cam_id': cam_id, 'frame_idx': int(frame_idx), 'person_id': pid,
                        'bbox_area': float(bbox_area), 'alpha_sum': 0, 'denom': 0,
                        'clamped': True, 'pooled_feature_norm': 0, 'teacher_feature_norm': 0,
                        'valid': False, 'invalid_reason': 'roi_pool_failed',
                    })
                    continue

                teacher_feat = normalize_feat(
                    torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                ) if teacher_emb is not None else None

                alpha_sum = float(pool_stats.get('alpha_sum', 0))
                denom = float(pool_stats.get('denom', 0))
                clamped = bool(pool_stats.get('clamped', False))
                pooled_norm = float(pool_stats.get('pooled_norm', 0))
                teacher_norm = float(teacher_feat.norm().item()) if teacher_feat is not None else 0
                cos_val = float(torch.dot(f_v, teacher_feat).item()) if teacher_feat is not None else 0

                valid_flag = True
                reason = ""
                if alpha_sum < 1e-3:
                    valid_flag = False
                    reason = "alpha_too_small_1e-3"
                elif alpha_sum < 1e-2:
                    valid_flag = False
                    reason = "alpha_too_small_1e-2"
                elif alpha_sum < 5e-2:
                    valid_flag = False
                    reason = "alpha_too_small_5e-2"
                elif alpha_sum < 1e-1:
                    valid_flag = False
                    reason = "alpha_too_small_1e-1"

                if valid_flag:
                    valid_count += 1
                    cam_stats[cam_id]['valid'] += 1
                    person_stats[pid]['valid'] += 1
                else:
                    invalid_reasons[reason] += 1

                roi_metrics.append({
                    'cam_id': cam_id, 'frame_idx': int(frame_idx), 'person_id': pid,
                    'bbox_area': float(bbox_area), 'alpha_sum': alpha_sum, 'denom': denom,
                    'clamped': clamped, 'pooled_feature_norm': pooled_norm,
                    'teacher_feature_norm': teacher_norm, 'cos_fv_tv': cos_val,
                    'valid': valid_flag, 'invalid_reason': reason,
                })

    metrics_path = os.path.join(args.output_dir, "roi_quality_metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for m in roi_metrics:
            f.write(json.dumps(m) + "\n")

    alpha_sums = [m['alpha_sum'] for m in roi_metrics]
    bbox_areas = [m['bbox_area'] for m in roi_metrics]

    alpha_thresholds = {1e-1: 0, 5e-2: 0, 1e-2: 0, 1e-3: 0}
    for a in alpha_sums:
        for t in alpha_thresholds:
            if a >= t:
                alpha_thresholds[t] += 1

    valid_ratio_by_alpha = {
        str(t): alpha_thresholds[t] / max(len(alpha_sums), 1) for t in [1e-1, 5e-2, 1e-2, 1e-3]
    }

    valid_ratio_by_cam = {}
    for cam_id, stats in cam_stats.items():
        valid_ratio_by_cam[cam_id] = stats['valid'] / max(stats['total'], 1)

    valid_ratio_by_person = {}
    for pid, stats in list(person_stats.items())[:50]:
        valid_ratio_by_person[str(pid)] = stats['valid'] / max(stats['total'], 1)

    summary = {
        "total_roi": total_count,
        "valid_roi_count": valid_count,
        "valid_ratio": valid_count / max(total_count, 1),
        "invalid_reason_counts": dict(invalid_reasons),
        "alpha_sum": {
            "mean": float(np.mean(alpha_sums)) if alpha_sums else 0,
            "median": float(np.median(alpha_sums)) if alpha_sums else 0,
            "p10": float(np.percentile(alpha_sums, 10)) if alpha_sums else 0,
            "p25": float(np.percentile(alpha_sums, 25)) if alpha_sums else 0,
            "p75": float(np.percentile(alpha_sums, 75)) if alpha_sums else 0,
            "p90": float(np.percentile(alpha_sums, 90)) if alpha_sums else 0,
            "max": float(max(alpha_sums)) if alpha_sums else 0,
        },
        "bbox_area": {
            "mean": float(np.mean(bbox_areas)) if bbox_areas else 0,
            "median": float(np.median(bbox_areas)) if bbox_areas else 0,
            "min": float(min(bbox_areas)) if bbox_areas else 0,
            "max": float(max(bbox_areas)) if bbox_areas else 0,
        },
        "valid_ratio_by_camera": valid_ratio_by_cam,
        "valid_ratio_by_person_sample": valid_ratio_by_person,
        "valid_ratio_by_alpha_threshold": valid_ratio_by_alpha,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ROI QUALITY SCAN SUMMARY")
    print(f"{'='*70}")
    print(f"Total ROIs scanned: {total_count}")
    print(f"Valid ROIs: {valid_count} ({summary['valid_ratio']*100:.1f}%)")
    print(f"\nInvalid reason counts:")
    for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"  {reason}: {count}")
    print(f"\nAlpha sum stats:")
    for k, v in summary['alpha_sum'].items():
        print(f"  {k}: {v:.6f}")
    print(f"\nValid ratio by alpha threshold:")
    for t, ratio in summary['valid_ratio_by_alpha_threshold'].items():
        print(f"  >= {t}: {ratio*100:.1f}%")
    print(f"\nValid ratio by camera:")
    for cam, ratio in sorted(valid_ratio_by_cam.items()):
        print(f"  {cam}: {ratio*100:.1f}%")

    print(f"\nMetrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    return True


def build_candidate_pool(args, trainer, sampler, batch_builder, allowed_cameras=None):
    pool_path = getattr(args, 'candidate_pool_path', None)
    if pool_path and os.path.exists(pool_path):
        print(f"Loading candidate pool from {pool_path}...")
        with open(pool_path, 'r') as f:
            pool = json.load(f)
        print(f"Loaded {len(pool)} candidates from pool")
        return pool

    print("\n" + "=" * 70)
    print("BUILDING CANDIDATE POOL (one-time)")
    print("=" * 70)

    pool_size = getattr(args, 'pre_collect_pool_size', 1000)
    print(f"Target pool size: {pool_size}")
    print(f"Allowed cameras: {allowed_cameras}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pool = []
    invalid_reasons = defaultdict(int)
    attempts = 0
    max_attempts = pool_size * 20

    per_cam_sampled = defaultdict(int)
    per_cam_valid = defaultdict(int)

    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue

        for ps in mv_samples:
            if len(pool) >= pool_size:
                break
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                if len(pool) >= pool_size:
                    break
                if allowed_cameras and cam_id not in allowed_cameras:
                    continue
                per_cam_sampled[cam_id] += 1

                gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, int(frame_idx))
                if gpu_batch is None:
                    invalid_reasons['no_batch'] += 1
                    continue

                try:
                    render_out = trainer.model(
                        gpu_batch, train=False, frame_id=0, render_person_feature=True
                    )
                    person_feature_map = render_out['person_feature_map']
                    person_opacity_map = render_out.get('person_opacity_map')
                except Exception:
                    invalid_reasons['render_failed'] += 1
                    continue

                inst = None
                for i in gpu_batch.instances:
                    if i.get('train_id') == pid:
                        inst = i
                        break
                if inst is None:
                    invalid_reasons['no_instance'] += 1
                    continue

                bbox = inst['bbox_xyxy']
                bbox_area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                if bbox_area < args.min_bbox_area:
                    invalid_reasons['bbox_too_small'] += 1
                    continue

                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)
                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )
                if f_v is None:
                    invalid_reasons['roi_pool_failed'] += 1
                    continue

                alpha_sum = float(pool_stats.get('alpha_sum', 0))
                if alpha_sum < args.min_alpha_sum:
                    invalid_reasons['alpha_too_small'] += 1
                    continue
                if pool_stats.get('clamped', False):
                    invalid_reasons['denom_clamped'] += 1
                    continue

                per_cam_valid[cam_id] += 1
                teacher_emb = inst.get('teacher_embedding')
                if teacher_emb is not None:
                    teacher_emb = np.asarray(teacher_emb).tolist()

                dataset_idx = sampler.get_dataset_index(cam_id, int(frame_idx))

                pool.append({
                    'person_id': pid,
                    'train_id': pid,
                    'cam_id': cam_id,
                    'frame_idx': int(frame_idx),
                    'dataset_index': dataset_idx,
                    'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox),
                    'bbox_area': bbox_area,
                    'alpha_sum': alpha_sum,
                    'teacher_emb': teacher_emb,
                })

    print(f"Pool built: {len(pool)} candidates from {attempts} attempts")
    print(f"Invalid reasons: {dict(invalid_reasons)}")

    alphas = [c['alpha_sum'] for c in pool]
    print(f"Alpha sum: min={min(alphas):.4f}, mean={np.mean(alphas):.4f}, max={max(alphas):.4f}")
    print(f"Per-camera sampled: {dict(per_cam_sampled)}")
    print(f"Per-camera valid: {dict(per_cam_valid)}")

    pool_save_path = os.path.join(args.output_dir, "candidate_pool.json")
    if getattr(args, 'save_candidate_pool', True):
        with open(pool_save_path, 'w') as f:
            json.dump(pool, f, default=str)
        print(f"Pool saved to {pool_save_path}")

    pool_meta = {
        'pool_size': len(pool),
        'attempts': attempts,
        'invalid_reasons': dict(invalid_reasons),
        'alpha_sum': {
            'min': float(min(alphas)) if alphas else 0,
            'mean': float(np.mean(alphas)) if alphas else 0,
            'max': float(max(alphas)) if alphas else 0,
        },
        'per_camera_sampled': dict(per_cam_sampled),
        'per_camera_valid': dict(per_cam_valid),
        'per_camera_valid_ratio': {
            cam: per_cam_valid.get(cam, 0) / max(per_cam_sampled.get(cam, 1), 1)
            for cam in set(list(per_cam_sampled.keys()) + list(per_cam_valid.keys()))
        },
    }
    return pool, pool_meta


def run_train_teacher_clean_fixed_eval(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("TRAIN TEACHER CLEAN FIXED EVAL MODE")
    print("=" * 70)

    allowed_cameras = getattr(args, 'allowed_cameras', None)
    if allowed_cameras:
        if isinstance(allowed_cameras, str):
            allowed_cameras = [c.strip() for c in allowed_cameras.split(',')]
        print(f"Camera-aware mode: allowed_cameras={allowed_cameras}")
    else:
        allowed_cameras = None
        print("Using all cameras (no allowed_cameras filter)")

    use_pool = getattr(args, 'use_pre_collected_pool', False)
    train_pool = None
    pool_meta = None

    if use_pool:
        print("\nUsing pre-collected candidate pool mode")
        result = build_candidate_pool(args, trainer, sampler, batch_builder, allowed_cameras=allowed_cameras)
        if isinstance(result, tuple):
            train_pool, pool_meta = result
        else:
            train_pool = result
        print(f"Train pool size: {len(train_pool)} candidates")

        pool_save_path = os.path.join(args.output_dir, "candidate_pool.json")
        if getattr(args, 'save_candidate_pool', True) and pool_meta is not None:
            with open(pool_save_path, 'w') as f:
                json.dump({'pool': train_pool, 'meta': pool_meta}, f, indent=2, default=str)
            print(f"Pool + meta saved to {pool_save_path}")
    else:
        print("Using per-step candidate collection mode (legacy)")

    print("Collecting fixed eval samples...")
    orig_mult = args.candidate_multiplier
    args.candidate_multiplier = max(orig_mult, 20)
    eval_candidates, eval_invalid, eval_n_cand = collect_clean_candidates(
        args, trainer, sampler, batch_builder, allowed_cameras=allowed_cameras)
    args.candidate_multiplier = orig_mult

    eval_candidates = sorted(eval_candidates, key=lambda c: c['alpha_sum'], reverse=True)
    target = args.target_valid_count or (args.P * args.K)
    if len(eval_candidates) < target:
        print(f"WARNING: Only {len(eval_candidates)} eval candidates (requested {target})")
        eval_selected = eval_candidates
    else:
        eval_selected = eval_candidates[:target]

    fixed_eval_path = os.path.join(args.output_dir, "fixed_eval_samples.json")
    with open(fixed_eval_path, 'w') as f:
        json.dump(eval_selected, f, indent=2, default=str)
    print(f"Fixed eval samples: {len(eval_selected)} ROIs")

    # Optimizer sanity check
    pf = trainer.model.get_person_feature()
    optimizer_has_pf = False
    for group in trainer.model.optimizer.param_groups:
        for p in group['params']:
            if p is pf:
                optimizer_has_pf = True
                break
    if not optimizer_has_pf:
        print("WARNING: person_feature is not in optimizer param_groups.")
    if not pf.requires_grad:
        print("WARNING: person_feature.requires_grad is False.")

    train_log = []
    t_start = time.time()
    max_grad_norm = 0
    total_nan = 0
    total_inf = 0
    param_deltas = []
    param_norms = []
    best_fixed_cos = -999
    best_step = 0

    per_cam_train_stats = defaultdict(lambda: {"count": 0, "valid": 0, "cos": []})
    per_cam_eval_stats = defaultdict(lambda: {"count": 0, "valid": 0, "cos": []})

    for step in range(args.num_steps):
        t_iter_start = time.time()

        if use_pool and train_pool is not None:
            train_selected = random.sample(train_pool, min(args.P * args.K, len(train_pool)))
            n_cand = len(train_pool)
        else:
            args.candidate_multiplier = max(getattr(args, 'candidate_multiplier', 10), 10)
            train_candidates, train_invalid, n_cand = collect_clean_candidates(
                args, trainer, sampler, batch_builder, allowed_cameras=allowed_cameras)
            train_candidates = sorted(train_candidates, key=lambda c: c['alpha_sum'], reverse=True)
            target_train = args.P * args.K
            if len(train_candidates) < target_train:
                train_selected = train_candidates
            else:
                train_selected = train_candidates[:target_train]

        step_per_cam_valid = defaultdict(int)
        step_per_cam_cos = defaultdict(list)
        for roi in train_selected:
            step_per_cam_valid[roi['cam_id']] += 1

        trainer.model.optimizer.zero_grad()

        step_loss = 0.0
        step_cos = []
        step_alpha = []
        valid_count = 0
        invalid_count = 0
        denom_clamped = 0

        for roi in train_selected:
            gpu_batch = batch_builder.get_batch_by_cam_frame(roi['cam_id'], roi['frame_idx'])
            if gpu_batch is None:
                invalid_count += 1
                continue

            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
            f_v, pool_stats = opacity_roi_pooling(
                person_feature_map, person_opacity_map, bbox_t,
                denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
            )

            if f_v is None:
                invalid_count += 1
                continue

            teacher_feat = normalize_feat(
                torch.as_tensor(roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
            )
            cos_sim = torch.dot(f_v, teacher_feat)
            step_cos.append(float(cos_sim.item()))
            step_alpha.append(float(pool_stats.get('alpha_sum', 0)))
            step_per_cam_cos[roi['cam_id']].append(float(cos_sim.item()))

            if pool_stats.get('clamped', False):
                denom_clamped += 1

            if args.teacher_loss_type == 'cosine':
                l = 1 - cos_sim
            else:
                l = F.mse_loss(f_v, teacher_feat)
            step_loss = step_loss + l
            valid_count += 1

        if valid_count == 0:
            continue

        avg_loss = step_loss / valid_count
        avg_loss.backward()

        pf = trainer.model.get_person_feature()
        pf_before = pf.detach().clone()
        param_norm_before = float(pf_before.norm().item())
        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]
        grad_max_before = float(pf.grad.abs().max().item()) if pf.grad is not None else 0.0
        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
        grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0
        trainer.model.optimizer.step()
        delta = pf.detach() - pf_before
        param_delta_norm = float(delta.norm().item())
        param_delta_max = float(delta.abs().max().item())
        param_norm_after = float(pf.detach().norm().item())
        relative_param_delta = param_delta_norm / max(param_norm_before, 1e-8)

        param_deltas.append(param_delta_norm)
        param_norms.append(param_norm_before)
        max_grad_norm = max(max_grad_norm, grad_norm_before)
        total_nan += int(grad_info["nan_count"])
        total_inf += int(grad_info["inf_count"])

        if grad_norm_before > 0 and param_delta_norm < 1e-12:
            print(f"WARNING: Step {step}: Gradient exists (norm={grad_norm_before:.4e}) but person_feature is not updated.")

        record = {
            "step": step,
            "train_loss_teacher": float(avg_loss.item()),
            "train_cos_fv_tv_mean": float(np.mean(step_cos)) if step_cos else 0,
            "train_cos_fv_tv_min": float(min(step_cos)) if step_cos else 0,
            "train_cos_fv_tv_max": float(max(step_cos)) if step_cos else 0,
            "valid_feature_count": valid_count,
            "invalid_feature_count": invalid_count,
            "clean_candidate_count": n_cand,
            "alpha_sum_min": min(step_alpha) if step_alpha else 0,
            "alpha_sum_mean": float(np.mean(step_alpha)) if step_alpha else 0,
            "alpha_sum_max": max(step_alpha) if step_alpha else 0,
            "denom_clamp_ratio": denom_clamped / max(valid_count, 1),
            "grad_norm_before_clip": grad_norm_before,
            "grad_max_before_clip": grad_max_before,
            "grad_norm_after_clip": grad_norm_after,
            "param_delta_norm": param_delta_norm,
            "param_delta_max": param_delta_max,
            "relative_param_delta": relative_param_delta,
            "param_norm_before": param_norm_before,
            "param_norm_after": param_norm_after,
            "optimizer_contains_person_feature": optimizer_has_pf,
            "person_feature_requires_grad": pf.requires_grad,
            "nan_count": int(grad_info["nan_count"]),
            "inf_count": int(grad_info["inf_count"]),
            "iter_time": float(time.time() - t_iter_start),
            "per_camera_valid_count": dict(step_per_cam_valid),
            "per_camera_cos_mean": {cam: float(np.mean(cs)) for cam, cs in step_per_cam_cos.items() if cs},
        }

        if step % args.eval_interval == 0 or step == args.num_steps - 1:
            with torch.no_grad():
                eval_cos = []
                eval_loss_sum = 0.0
                eval_valid = 0
                eval_alpha = []
                eval_denom_clamped = 0
                eval_per_cam_cos = defaultdict(list)
                eval_per_cam_valid = defaultdict(int)
                for roi in eval_selected:
                    gpu_batch = batch_builder.get_batch_by_cam_frame(roi['cam_id'], roi['frame_idx'])
                    if gpu_batch is None:
                        continue
                    render_out = trainer.model(
                        gpu_batch, train=False, frame_id=0, render_person_feature=True
                    )
                    person_feature_map = render_out['person_feature_map']
                    person_opacity_map = render_out.get('person_opacity_map')
                    bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
                    f_v, pool_stats = opacity_roi_pooling(
                        person_feature_map, person_opacity_map, bbox_t,
                        denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                    )
                    if f_v is None:
                        continue
                    teacher_feat = normalize_feat(
                        torch.as_tensor(roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
                    )
                    c = float(torch.dot(f_v, teacher_feat).item())
                    eval_cos.append(c)
                    eval_alpha.append(float(pool_stats.get('alpha_sum', 0)))
                    eval_per_cam_cos[roi['cam_id']].append(c)
                    eval_per_cam_valid[roi['cam_id']] += 1
                    if pool_stats.get('clamped', False):
                        eval_denom_clamped += 1
                    if args.teacher_loss_type == 'cosine':
                        eval_loss_sum += 1 - c
                    eval_valid += 1

                record['fixed_loss_teacher'] = float(eval_loss_sum / eval_valid) if eval_valid > 0 else 0
                record['fixed_cos_fv_tv_mean'] = float(np.mean(eval_cos)) if eval_cos else 0
                record['fixed_cos_fv_tv_min'] = float(min(eval_cos)) if eval_cos else 0
                record['fixed_cos_fv_tv_max'] = float(max(eval_cos)) if eval_cos else 0
                record['fixed_valid_feature_count'] = eval_valid
                record['fixed_alpha_sum_min'] = min(eval_alpha) if eval_alpha else 0
                record['fixed_alpha_sum_mean'] = float(np.mean(eval_alpha)) if eval_alpha else 0
                record['fixed_alpha_sum_max'] = max(eval_alpha) if eval_alpha else 0
                record['fixed_denom_clamp_ratio'] = eval_denom_clamped / max(eval_valid, 1)
                record['fixed_per_camera_cos_mean'] = {cam: float(np.mean(cs)) for cam, cs in eval_per_cam_cos.items() if cs}
                record['fixed_per_camera_valid_count'] = dict(eval_per_cam_valid)

                if eval_cos:
                    fc = float(np.mean(eval_cos))
                    if fc > best_fixed_cos:
                        best_fixed_cos = fc
                        best_step = step

        train_log.append(record)

        if step % args.log_interval == 0:
            msg = (f"[CLEAN_FIXED] Step {step:5d}: train_loss={record['train_loss_teacher']:.4f} "
                   f"train_cos={record['train_cos_fv_tv_mean']:.4f} "
                   f"grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                   f"delta={param_delta_norm:.6e} valid={valid_count}")
            if 'fixed_cos_fv_tv_mean' in record:
                msg += (f" | eval_loss={record['fixed_loss_teacher']:.4f} "
                       f"eval_cos={record['fixed_cos_fv_tv_mean']:.4f}")
            msg += f" t={record['iter_time']:.2f}s"
            print(msg)

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r, default=str) + "\n")

    eval_records = [r for r in train_log if 'fixed_cos_fv_tv_mean' in r]
    if eval_records:
        first_cos = eval_records[0]['fixed_cos_fv_tv_mean']
        last_cos = eval_records[-1]['fixed_cos_fv_tv_mean']
        first_loss = eval_records[0].get('fixed_loss_teacher', 0)
        last_loss = eval_records[-1].get('fixed_loss_teacher', 0)
        cos_delta = last_cos - first_cos
        loss_delta = last_loss - first_loss
    else:
        first_cos = last_cos = cos_delta = first_loss = last_loss = loss_delta = 0

    # Determine verdict
    if cos_delta > 0.05 and loss_delta < 0 and total_nan == 0 and total_inf == 0:
        verdict = "roi_validcam_teacher_alignment_success"
    elif np.mean(param_deltas) < 1e-10:
        verdict = "optimizer_update_failure"
    elif cos_delta < 0.01:
        verdict = "roi_teacher_alignment_not_learned"
    else:
        verdict = "roi_opacity_signal_failure"

    summary = {
        "allowed_cameras": allowed_cameras,
        "use_pre_collected_pool": use_pool,
        "pool_meta": pool_meta,
        "lr": args.person_feature_lr,
        "num_steps": len(train_log),
        "first_eval_fixed_cos": first_cos,
        "best_eval_fixed_cos": best_fixed_cos,
        "last_eval_fixed_cos": last_cos,
        "fixed_cos_delta": cos_delta,
        "first_eval_loss": first_loss,
        "last_eval_loss": last_loss,
        "loss_delta": loss_delta,
        "max_grad_norm": max_grad_norm,
        "mean_param_delta_norm": float(np.mean(param_deltas)) if param_deltas else 0,
        "nan_total": total_nan,
        "inf_total": total_inf,
        "verdict": verdict,
        "per_camera_summary": {
            "train": {cam: {"count": s["count"], "valid": s["valid"]} for cam, s in per_cam_train_stats.items()},
            "eval": {cam: {"count": s["count"], "valid": s["valid"]} for cam, s in per_cam_eval_stats.items()},
        },
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"CLEAN FIXED EVAL RESULT")
    print(f"{'='*70}")
    print(f"Allowed cameras: {allowed_cameras}")
    print(f"Fixed eval cos: first={first_cos:.4f}, best={best_fixed_cos:.4f}, last={last_cos:.4f}, delta={cos_delta:+.4f}")
    print(f"Fixed eval loss: first={first_loss:.4f}, last={last_loss:.4f}, delta={loss_delta:+.4f}")
    print(f"Mean param delta: {np.mean(param_deltas):.6e}")
    print(f"Max grad norm: {max_grad_norm:.4e}")
    print(f"NaN/Inf: {total_nan}/{total_inf}")
    print(f"VERDICT: {verdict}")
    print(f"SUCCESS: {'roi_validcam_teacher_alignment_success' in verdict}")

    return True


def run_train_teacher_alpha_curriculum(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("TRAIN TEACHER ALPHA CURRICULUM MODE")
    print("=" * 70)

    curriculum_stages = [
        {"stage": 1, "min_alpha_sum": 1e-1, "steps": 500},
        {"stage": 2, "min_alpha_sum": 5e-2, "steps": 500},
        {"stage": 3, "min_alpha_sum": 1e-2, "steps": 500},
        {"stage": 4, "min_alpha_sum": 1e-3, "steps": 500},
    ]

    print("Collecting fixed eval samples (alpha >= 1e-1)...")
    orig_alpha = args.min_alpha_sum
    args.min_alpha_sum = 1e-1
    orig_mult = args.candidate_multiplier
    args.candidate_multiplier = max(orig_mult, 20)
    eval_candidates, _, _ = collect_clean_candidates(args, trainer, sampler, batch_builder)
    args.candidate_multiplier = orig_mult
    args.min_alpha_sum = orig_alpha

    eval_candidates = sorted(eval_candidates, key=lambda c: c['alpha_sum'], reverse=True)
    target = args.target_valid_count or (args.P * args.K)
    eval_selected = eval_candidates[:target] if len(eval_candidates) >= target else eval_candidates
    print(f"Fixed eval samples: {len(eval_selected)} ROIs")

    train_log = []
    t_start = time.time()
    global_step = 0

    for stage_def in curriculum_stages:
        stage = stage_def["stage"]
        min_alpha = stage_def["min_alpha_sum"]
        n_steps = stage_def["steps"]

        print(f"\n{'='*50}")
        print(f"Stage {stage}: min_alpha_sum={min_alpha}, steps={n_steps}")
        print(f"{'='*50}")

        args.min_alpha_sum = min_alpha

        for step in range(n_steps):
            t_iter_start = time.time()

            train_candidates, train_invalid, n_cand = collect_clean_candidates(
                args, trainer, sampler, batch_builder)
            train_selected = train_candidates[:args.P * args.K] if len(train_candidates) >= args.P * args.K else train_candidates

            trainer.model.optimizer.zero_grad()

            step_loss = 0.0
            step_cos = []
            step_alpha = []
            valid_count = 0
            invalid_count = 0

            for roi in train_selected:
                gpu_batch = batch_builder.get_batch_by_cam_frame(roi['cam_id'], roi['frame_idx'])
                if gpu_batch is None:
                    invalid_count += 1
                    continue

                render_out = trainer.model(
                    gpu_batch, train=False, frame_id=0, render_person_feature=True
                )
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')

                bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )

                if f_v is None:
                    invalid_count += 1
                    continue

                teacher_feat = normalize_feat(
                    torch.as_tensor(roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
                )
                cos_sim = torch.dot(f_v, teacher_feat)
                step_cos.append(float(cos_sim.item()))
                step_alpha.append(float(pool_stats.get('alpha_sum', 0)))

                if args.teacher_loss_type == 'cosine':
                    l = 1 - cos_sim
                else:
                    l = F.mse_loss(f_v, teacher_feat)
                step_loss = step_loss + l
                valid_count += 1

            if valid_count == 0:
                continue

            avg_loss = step_loss / valid_count
            avg_loss.backward()

            pf = trainer.model.get_person_feature()
            pf_before = pf.detach().clone()
            grad_info = get_grad_stats(pf)
            grad_norm_before = grad_info["grad_norm"]
            torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
            grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0
            trainer.model.optimizer.step()
            delta = pf.detach() - pf_before
            param_delta_norm = float(delta.norm().item())

            record = {
                "step": global_step,
                "stage": stage,
                "current_min_alpha_sum": min_alpha,
                "train_valid_ratio": valid_count / max(valid_count + invalid_count, 1),
                "train_loss_teacher": float(avg_loss.item()),
                "train_cos_fv_tv_mean": float(np.mean(step_cos)) if step_cos else 0,
                "alpha_sum_min": min(step_alpha) if step_alpha else 0,
                "alpha_sum_mean": float(np.mean(step_alpha)) if step_alpha else 0,
                "alpha_sum_max": max(step_alpha) if step_alpha else 0,
                "invalid_reason_counts": train_invalid,
                "grad_norm_before_clip": grad_norm_before,
                "grad_norm_after_clip": grad_norm_after,
                "param_delta_norm": param_delta_norm,
                "nan_count": int(grad_info["nan_count"]),
                "inf_count": int(grad_info["inf_count"]),
                "iter_time": float(time.time() - t_iter_start),
            }

            if global_step % 50 == 0 or global_step == 0:
                with torch.no_grad():
                    eval_cos = []
                    eval_loss_sum = 0.0
                    eval_valid = 0
                    for roi in eval_selected:
                        gpu_batch = batch_builder.get_batch_by_cam_frame(roi['cam_id'], roi['frame_idx'])
                        if gpu_batch is None:
                            continue
                        render_out = trainer.model(
                            gpu_batch, train=False, frame_id=0, render_person_feature=True
                        )
                        person_feature_map = render_out['person_feature_map']
                        person_opacity_map = render_out.get('person_opacity_map')
                        bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
                        f_v, pool_stats = opacity_roi_pooling(
                            person_feature_map, person_opacity_map, bbox_t,
                            denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                        )
                        if f_v is None:
                            continue
                        teacher_feat = normalize_feat(
                            torch.as_tensor(roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
                        )
                        c = float(torch.dot(f_v, teacher_feat).item())
                        eval_cos.append(c)
                        if args.teacher_loss_type == 'cosine':
                            eval_loss_sum += 1 - c
                        eval_valid += 1

                    record['fixed_loss_teacher'] = float(eval_loss_sum / eval_valid) if eval_valid > 0 else 0
                    record['fixed_cos_fv_tv_mean'] = float(np.mean(eval_cos)) if eval_cos else 0

                msg = (f"[CURRICULUM S{stage}] Step {global_step:5d}: "
                       f"alpha>={min_alpha:.0e} train_loss={record['train_loss_teacher']:.4f} "
                       f"train_cos={record['train_cos_fv_tv_mean']:.4f} "
                       f"eval_cos={record['fixed_cos_fv_tv_mean']:.4f} "
                       f"grad={grad_norm_before:.4e} valid={valid_count}")
                print(msg)

            train_log.append(record)
            global_step += 1

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r) + "\n")

    stage_summaries = []
    for s in range(1, 5):
        stage_records = [r for r in train_log if r.get('stage') == s]
        if stage_records:
            eval_recs = [r for r in stage_records if 'fixed_cos_fv_tv_mean' in r]
            if eval_recs:
                first_cos = eval_recs[0]['fixed_cos_fv_tv_mean']
                last_cos = eval_recs[-1]['fixed_cos_fv_tv_mean']
            else:
                first_cos = last_cos = 0
            stage_summaries.append({
                "stage": s,
                "min_alpha_sum": stage_records[0]['current_min_alpha_sum'],
                "steps": len(stage_records),
                "first_eval_cos": first_cos,
                "last_eval_cos": last_cos,
                "cos_delta": last_cos - first_cos,
            })

    summary = {
        "total_steps": len(train_log),
        "stage_summaries": stage_summaries,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ALPHA CURRICULUM SUMMARY")
    print(f"{'='*70}")
    print(f"{'Stage':<8} {'min_alpha':<12} {'eval_cos_start':<16} {'eval_cos_end':<14} {'delta':<10}")
    for ss in stage_summaries:
        print(f"{ss['stage']:<8} {ss['min_alpha']:<12.0e} {ss['first_eval_cos']:<16.4f} "
              f"{ss['last_eval_cos']:<14.4f} {ss['cos_delta']:+.4f}")

    return True


def run_train_clean_teacher_mv_fixed_eval(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("TRAIN CLEAN TEACHER MV FIXED EVAL MODE")
    print("=" * 70)

    print("Collecting fixed eval samples...")
    orig_mult = args.candidate_multiplier
    args.candidate_multiplier = max(orig_mult, 20)
    eval_candidates, _, _ = collect_clean_candidates(args, trainer, sampler, batch_builder)
    args.candidate_multiplier = orig_mult

    eval_candidates = sorted(eval_candidates, key=lambda c: c['alpha_sum'], reverse=True)
    target = args.target_valid_count or (args.P * args.K)
    eval_selected = eval_candidates[:target] if len(eval_candidates) >= target else eval_candidates

    fixed_eval_path = os.path.join(args.output_dir, "fixed_eval_samples.json")
    with open(fixed_eval_path, 'w') as f:
        json.dump(eval_selected, f, indent=2, default=str)
    print(f"Fixed eval samples: {len(eval_selected)} ROIs")

    train_log = []
    t_start = time.time()

    for step in range(args.num_steps):
        t_iter_start = time.time()

        train_candidates, _, n_cand = collect_clean_candidates(args, trainer, sampler, batch_builder)
        train_selected = train_candidates[:args.P * args.K] if len(train_candidates) >= args.P * args.K else train_candidates

        view_groups = defaultdict(list)
        for roi in train_selected:
            view_groups[(roi['cam_id'], roi['frame_idx'])].append(roi)

        target_bank = {}
        with torch.no_grad():
            for (cam_id, frame_idx), rois in view_groups.items():
                gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
                if gpu_batch is None:
                    continue
                render_out = trainer.model(
                    gpu_batch, train=False, frame_id=0, render_person_feature=True
                )
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')
                for roi in rois:
                    bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
                    f_v, _ = opacity_roi_pooling(
                        person_feature_map, person_opacity_map, bbox_t,
                        denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                    )
                    if f_v is not None:
                        key = (roi['person_id'], cam_id, frame_idx)
                        target_bank[key] = {'feature': f_v.detach().clone(), 'bbox': roi['bbox']}

        trainer.model.optimizer.zero_grad()

        total_loss_teacher = 0.0
        total_loss_mv = 0.0
        total_valid = 0
        all_cos = []
        cross_same = []
        cross_diff = []
        pos_count = 0
        neg_count = 0

        for (cam_id, frame_idx), rois in view_groups.items():
            gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id, frame_idx)
            if gpu_batch is None:
                continue

            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            anchor_feats = []
            anchor_pids = []
            view_loss_t = 0.0
            num_valid = 0

            for roi in rois:
                bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
                f_v, _ = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )
                if f_v is None:
                    continue

                teacher_feat = normalize_feat(
                    torch.as_tensor(roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
                )
                cos_sim = torch.dot(f_v, teacher_feat)
                all_cos.append(float(cos_sim.item()))

                if args.teacher_loss_type == 'cosine':
                    l = 1 - cos_sim
                else:
                    l = F.mse_loss(f_v, teacher_feat)
                view_loss_t += l
                num_valid += 1

                anchor_feats.append(f_v)
                anchor_pids.append(roi['person_id'])

            if num_valid == 0:
                continue

            total_loss_teacher += view_loss_t.item() / num_valid
            total_valid += num_valid

            if args.lambda_mv > 0 and anchor_feats:
                anchor_stack = torch.stack(anchor_feats)
                target_feats = []
                target_labels = []

                for i, pid in enumerate(anchor_pids):
                    same_views = []
                    diff_views = []
                    for (t_pid, t_cam, t_frame), t_data in target_bank.items():
                        if t_cam == cam_id and t_frame == frame_idx:
                            continue
                        if t_pid == pid:
                            same_views.append(t_data['feature'])
                        else:
                            diff_views.append(t_data['feature'])

                    if same_views:
                        pos_count += len(same_views)
                        for sf in same_views:
                            target_feats.append(sf)
                            target_labels.append(1)
                    if diff_views:
                        neg_count += len(diff_views)
                        for df in diff_views:
                            target_feats.append(df)
                            target_labels.append(0)

                if target_feats:
                    target_stack = torch.stack(target_feats).detach()
                    labels_t = torch.tensor(target_labels, dtype=torch.float32, device=trainer.device)
                    sim_mat = anchor_stack @ target_stack.T / args.tau_mv
                    pos_mask = (labels_t == 1).float().unsqueeze(0)
                    neg_mask = (labels_t == 0).float().unsqueeze(0)
                    pos_sim = (sim_mat * pos_mask).sum(dim=1)
                    pos_cnt = pos_mask.sum(dim=1).clamp(min=1e-8)
                    neg_sim = (sim_mat * neg_mask).sum(dim=1)
                    l_mv = -torch.log(torch.exp(pos_sim / pos_cnt.squeeze()) /
                                     (torch.exp(pos_sim / pos_cnt.squeeze()) + torch.exp(neg_sim) + 1e-8)).mean()
                    total_loss_mv += l_mv.item()

            for i, pid in enumerate(anchor_pids):
                for (t_pid, t_cam, t_frame), t_data in target_bank.items():
                    if t_cam == cam_id and t_frame == frame_idx:
                        continue
                    c_ij = float(torch.dot(anchor_feats[i], t_data['feature']).item())
                    if t_pid == pid:
                        cross_same.append(c_ij)
                    else:
                        cross_diff.append(c_ij)

        if total_valid == 0:
            continue

        loss_total = args.lambda_teacher * (total_loss_teacher / len(view_groups))
        if args.lambda_mv > 0:
            loss_total += args.lambda_mv * total_loss_mv / max(len(view_groups), 1)

        loss_total.backward()

        pf = trainer.model.get_person_feature()
        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]
        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
        grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0
        trainer.model.optimizer.step()

        record = {
            "step": step,
            "loss_teacher": total_loss_teacher / max(len(view_groups), 1),
            "loss_mv": total_loss_mv / max(len(view_groups), 1),
            "fixed_cos_fv_tv": 0,
            "cross_view_same_cos": float(np.mean(cross_same)) if cross_same else 0,
            "cross_view_diff_cos": float(np.mean(cross_diff)) if cross_diff else 0,
            "cross_view_gap": (float(np.mean(cross_same)) - float(np.mean(cross_diff))) if cross_same and cross_diff else 0,
            "positive_pair_count": pos_count,
            "negative_pair_count": neg_count,
            "grad_norm": grad_norm_before,
            "nan_count": int(grad_info["nan_count"]),
            "inf_count": int(grad_info["inf_count"]),
            "iter_time": float(time.time() - t_iter_start),
        }

        if step % args.eval_interval == 0 or step == args.num_steps - 1:
            with torch.no_grad():
                eval_cos = []
                for roi in eval_selected:
                    gpu_batch = batch_builder.get_batch_by_cam_frame(roi['cam_id'], roi['frame_idx'])
                    if gpu_batch is None:
                        continue
                    render_out = trainer.model(
                        gpu_batch, train=False, frame_id=0, render_person_feature=True
                    )
                    person_feature_map = render_out['person_feature_map']
                    person_opacity_map = render_out.get('person_opacity_map')
                    bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
                    f_v, _ = opacity_roi_pooling(
                        person_feature_map, person_opacity_map, bbox_t,
                        denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                    )
                    if f_v is None:
                        continue
                    teacher_feat = normalize_feat(
                        torch.as_tensor(roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
                    )
                    eval_cos.append(float(torch.dot(f_v, teacher_feat).item()))

                record['fixed_cos_fv_tv'] = float(np.mean(eval_cos)) if eval_cos else 0

        train_log.append(record)

        if step % args.log_interval == 0:
            print(f"[MV_FIXED] Step {step:5d}: loss_t={record['loss_teacher']:.4f} "
                  f"loss_mv={record['loss_mv']:.4f} "
                  f"eval_cos={record['fixed_cos_fv_tv']:.4f} "
                  f"gap={record['cross_view_gap']:.4f} "
                  f"grad={grad_norm_before:.4e} "
                  f"pos={pos_count} neg={neg_count}")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r) + "\n")

    eval_recs = [r for r in train_log if 'fixed_cos_fv_tv' in r]
    if eval_recs:
        first_cos = eval_recs[0]['fixed_cos_fv_tv']
        last_cos = eval_recs[-1]['fixed_cos_fv_tv']
        cos_delta = last_cos - first_cos
    else:
        first_cos = last_cos = cos_delta = 0

    summary = {
        "total_steps": len(train_log),
        "first_eval_cos": first_cos,
        "last_eval_cos": last_cos,
        "eval_cos_delta": cos_delta,
        "lambda_mv": args.lambda_mv,
        "tau_mv": args.tau_mv,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"MV FIXED EVAL RESULT")
    print(f"{'='*70}")
    print(f"lambda_mv={args.lambda_mv}, tau_mv={args.tau_mv}")
    print(f"Fixed eval cos: first={first_cos:.4f}, last={last_cos:.4f}, delta={cos_delta:+.4f}")

    return True


def run_camera_roi_diagnostic(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("CAMERA ROI DIAGNOSTIC MODE")
    print("=" * 70)

    samples_per_camera = getattr(args, 'samples_per_camera', 100)
    save_debug_images = getattr(args, 'save_debug_images', False)
    debug_images_per_camera = getattr(args, 'debug_images_per_camera', 5)

    all_cameras = sorted(sampler.cam_frame_to_index.keys(), key=lambda x: x[0])
    unique_cams = sorted(set(cam for cam, _ in all_cameras))
    print(f"Cameras in dataset: {unique_cams}")
    print(f"Samples per camera: {samples_per_camera}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("[WARNING] matplotlib not available, debug images will not be saved")

    if save_debug_images and HAS_MATPLOTLIB:
        debug_dir = os.path.join(args.output_dir, "debug_images")
        os.makedirs(debug_dir, exist_ok=True)

    all_metrics = []
    camera_summary = {}

    for cam_id in unique_cams:
        print(f"\n{'='*50}")
        print(f"Camera: {cam_id}")
        print(f"{'='*50}")

        cam_samples = 0
        cam_metrics = []
        cam_invalid_reasons = defaultdict(int)
        render_success = 0
        debug_saved = 0

        attempts = 0
        max_attempts = samples_per_camera * 10

        while cam_samples < samples_per_camera and attempts < max_attempts:
            attempts += 1
            mv_samples = sampler.sample_batch()
            if mv_samples is None:
                continue

            for ps in mv_samples:
                if cam_samples >= samples_per_camera:
                    break
                pid = ps['person_id']

                views_for_cam = [(c, f) for c, f in ps['views'] if c == cam_id]
                if not views_for_cam:
                    continue

                cam_id_view, frame_idx = views_for_cam[0]

                gpu_batch = batch_builder.get_batch_by_cam_frame(cam_id_view, int(frame_idx))
                dataset_idx = sampler.get_dataset_index(cam_id_view, int(frame_idx))
                cam_frame_key_found = (dataset_idx is not None)

                if gpu_batch is None:
                    cam_samples += 1
                    cam_invalid_reasons['no_batch'] += 1
                    cam_metrics.append({
                        'cam_id': cam_id_view, 'frame_idx': int(frame_idx), 'person_id': pid,
                        'bbox': None, 'bbox_area': 0, 'image_width': 0, 'image_height': 0,
                        'bbox_in_bounds': False, 'bbox_clamped': False,
                        'alpha_sum': 0, 'alpha_max': 0, 'alpha_mean': 0,
                        'denom': 0, 'clamped': True, 'pooled_feature_norm': 0,
                        'teacher_feature_norm': 0, 'valid': False,
                        'invalid_reason': 'no_batch', 'dataset_index': dataset_idx,
                        'cam_frame_key_found': cam_frame_key_found, 'render_success': False,
                    })
                    continue

                try:
                    render_out = trainer.model(
                        gpu_batch, train=False, frame_id=0, render_person_feature=True
                    )
                    person_feature_map = render_out['person_feature_map']
                    person_opacity_map = render_out.get('person_opacity_map')
                    render_ok = True
                except Exception as e:
                    render_ok = False
                    person_feature_map = None
                    person_opacity_map = None

                if not render_ok:
                    cam_samples += 1
                    cam_invalid_reasons['render_failed'] += 1
                    cam_metrics.append({
                        'cam_id': cam_id_view, 'frame_idx': int(frame_idx), 'person_id': pid,
                        'bbox': None, 'bbox_area': 0, 'image_width': 0, 'image_height': 0,
                        'bbox_in_bounds': False, 'bbox_clamped': False,
                        'alpha_sum': 0, 'alpha_max': 0, 'alpha_mean': 0,
                        'denom': 0, 'clamped': True, 'pooled_feature_norm': 0,
                        'teacher_feature_norm': 0, 'valid': False,
                        'invalid_reason': 'render_failed', 'dataset_index': dataset_idx,
                        'cam_frame_key_found': cam_frame_key_found, 'render_success': False,
                    })
                    continue

                render_success += 1

                D, H, W = person_feature_map.shape
                inst = None
                for i in gpu_batch.instances:
                    if i.get('train_id') == pid:
                        inst = i
                        break

                if inst is None:
                    cam_samples += 1
                    cam_invalid_reasons['no_instance'] += 1
                    cam_metrics.append({
                        'cam_id': cam_id_view, 'frame_idx': int(frame_idx), 'person_id': pid,
                        'bbox': None, 'bbox_area': 0, 'image_width': W, 'image_height': H,
                        'bbox_in_bounds': False, 'bbox_clamped': False,
                        'alpha_sum': 0, 'alpha_max': 0, 'alpha_mean': 0,
                        'denom': 0, 'clamped': True, 'pooled_feature_norm': 0,
                        'teacher_feature_norm': 0, 'valid': False,
                        'invalid_reason': 'no_instance', 'dataset_index': dataset_idx,
                        'cam_frame_key_found': cam_frame_key_found, 'render_success': True,
                    })
                    continue

                bbox = inst['bbox_xyxy']
                bbox_raw = list(bbox)
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                xmin_raw, ymin_raw = int(bbox[0]), int(bbox[1])
                xmax_raw, ymax_raw = int(bbox[2]), int(bbox[3])
                xmin = max(0, xmin_raw)
                ymin = max(0, ymin_raw)
                xmax = min(W, max(xmin + 1, xmax_raw))
                ymax = min(H, max(ymin + 1, ymax_raw))
                bbox_clamped = (xmin != xmin_raw or ymin != ymin_raw or xmax != xmax_raw or ymax != ymax_raw)
                bbox_in_bounds = (xmin_raw >= 0 and ymin_raw >= 0 and xmax_raw <= W and ymax_raw <= H)

                if person_opacity_map is not None:
                    alpha_region = person_opacity_map[ymin:ymax, xmin:xmax]
                    alpha_sum = float(alpha_region.sum().item())
                    alpha_max = float(alpha_region.max().item())
                    alpha_mean = float(alpha_region.mean().item())
                else:
                    alpha_sum = 0
                    alpha_max = 0
                    alpha_mean = 0

                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)
                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )

                teacher_emb = inst.get('teacher_embedding')
                teacher_feat = normalize_feat(
                    torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                ) if teacher_emb is not None else None
                teacher_norm = float(teacher_feat.norm().item()) if teacher_feat is not None else 0

                denom = float(pool_stats.get('denom', 0)) if pool_stats else 0
                clamped = bool(pool_stats.get('clamped', False)) if pool_stats else True
                pooled_norm = float(pool_stats.get('pooled_norm', 0)) if pool_stats and f_v is not None else 0

                valid = (f_v is not None and alpha_sum > 1e-3 and not clamped)
                invalid_reason = ""
                if f_v is None:
                    invalid_reason = "roi_pool_failed"
                    cam_invalid_reasons['roi_pool_failed'] += 1
                elif alpha_sum < 1e-3:
                    invalid_reason = "alpha_too_small_1e-3"
                    cam_invalid_reasons['alpha_too_small'] += 1
                elif clamped:
                    invalid_reason = "denom_clamped"
                    cam_invalid_reasons['denom_clamped'] += 1
                else:
                    pass

                cam_samples += 1
                metric = {
                    'cam_id': cam_id_view, 'frame_idx': int(frame_idx), 'person_id': pid,
                    'bbox': bbox_raw, 'bbox_area': float(bbox_area),
                    'image_width': W, 'image_height': H,
                    'bbox_in_bounds': bbox_in_bounds, 'bbox_clamped': bbox_clamped,
                    'alpha_sum': alpha_sum, 'alpha_max': alpha_max, 'alpha_mean': alpha_mean,
                    'denom': denom, 'clamped': clamped,
                    'pooled_feature_norm': pooled_norm, 'teacher_feature_norm': teacher_norm,
                    'valid': valid, 'invalid_reason': invalid_reason,
                    'dataset_index': dataset_idx, 'cam_frame_key_found': cam_frame_key_found,
                    'render_success': True,
                }
                cam_metrics.append(metric)

                if save_debug_images and HAS_MATPLOTLIB and debug_saved < debug_images_per_camera:
                    try:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                        if person_opacity_map is not None:
                            alpha_np = person_opacity_map.detach().cpu().numpy()
                            axes[0].imshow(alpha_np, cmap='viridis')
                            axes[0].set_title(f"Opacity Map\nalpha_sum={alpha_sum:.2f}, max={alpha_max:.4f}")
                            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           linewidth=2, edgecolor='red', facecolor='none')
                            axes[0].add_patch(rect)
                        else:
                            axes[0].text(0.5, 0.5, "No opacity map", ha='center', va='center')

                        if f_v is not None:
                            axes[1].text(0.5, 0.5, f"Valid ROI\npooled_norm={pooled_norm:.4f}\nteacher_norm={teacher_norm:.4f}",
                                        ha='center', va='center', fontsize=12)
                        else:
                            axes[1].text(0.5, 0.5, f"Invalid ROI\nreason={invalid_reason}",
                                        ha='center', va='center', fontsize=12, color='red')

                        for ax in axes:
                            ax.axis('off')

                        plt.tight_layout()
                        img_path = os.path.join(debug_dir, f"{cam_id}_sample{debug_saved}.png")
                        plt.savefig(img_path, dpi=100)
                        plt.close()

                        debug_info = {
                            'cam_id': cam_id_view, 'frame_idx': int(frame_idx), 'person_id': pid,
                            'bbox_raw': bbox_raw, 'bbox_clamped': [xmin, ymin, xmax, ymax],
                            'image_size': [W, H], 'alpha_sum': alpha_sum, 'alpha_max': alpha_max,
                            'alpha_mean': alpha_mean, 'valid': valid, 'invalid_reason': invalid_reason,
                            'dataset_index': dataset_idx, 'teacher_exists': teacher_emb is not None,
                        }
                        info_path = os.path.join(debug_dir, f"{cam_id}_sample{debug_saved}_info.json")
                        with open(info_path, 'w') as f:
                            json.dump(debug_info, f, indent=2)

                        debug_saved += 1
                    except Exception as e:
                        print(f"  [WARNING] Failed to save debug image: {e}")

        cam_valid = sum(1 for m in cam_metrics if m['valid'])
        alphas = [m['alpha_sum'] for m in cam_metrics]
        alpha_maxs = [m['alpha_max'] for m in cam_metrics]
        bbox_areas = [m['bbox_area'] for m in cam_metrics if m['bbox_area'] > 0]
        pooled_norms = [m['pooled_feature_norm'] for m in cam_metrics if m['pooled_feature_norm'] > 0]
        teacher_norms = [m['teacher_feature_norm'] for m in cam_metrics if m['teacher_feature_norm'] > 0]
        in_bounds_count = sum(1 for m in cam_metrics if m['bbox_in_bounds'])

        summary = {
            "sampled_count": len(cam_metrics),
            "render_success_count": render_success,
            "valid_count": cam_valid,
            "valid_ratio": cam_valid / max(len(cam_metrics), 1),
            "invalid_reason_counts": dict(cam_invalid_reasons),
            "alpha_sum": {
                "mean": float(np.mean(alphas)) if alphas else 0,
                "median": float(np.median(alphas)) if alphas else 0,
                "min": float(min(alphas)) if alphas else 0,
                "max": float(max(alphas)) if alphas else 0,
                "p10": float(np.percentile(alphas, 10)) if alphas else 0,
                "p90": float(np.percentile(alphas, 90)) if alphas else 0,
            },
            "alpha_max": {
                "mean": float(np.mean(alpha_maxs)) if alpha_maxs else 0,
                "median": float(np.median(alpha_maxs)) if alpha_maxs else 0,
                "max": float(max(alpha_maxs)) if alpha_maxs else 0,
            },
            "bbox_area": {
                "mean": float(np.mean(bbox_areas)) if bbox_areas else 0,
                "median": float(np.median(bbox_areas)) if bbox_areas else 0,
                "min": float(min(bbox_areas)) if bbox_areas else 0,
                "max": float(max(bbox_areas)) if bbox_areas else 0,
            },
            "bbox_in_bounds_ratio": in_bounds_count / max(len(cam_metrics), 1),
            "pooled_norm": {
                "mean": float(np.mean(pooled_norms)) if pooled_norms else 0,
                "median": float(np.median(pooled_norms)) if pooled_norms else 0,
                "max": float(max(pooled_norms)) if pooled_norms else 0,
            },
            "teacher_norm": {
                "mean": float(np.mean(teacher_norms)) if teacher_norms else 0,
                "median": float(np.median(teacher_norms)) if teacher_norms else 0,
                "max": float(max(teacher_norms)) if teacher_norms else 0,
            },
        }
        camera_summary[cam_id] = summary
        all_metrics.extend(cam_metrics)

        print(f"  Sampled: {len(cam_metrics)}, Valid: {cam_valid} ({summary['valid_ratio']*100:.1f}%)")
        print(f"  Render success: {render_success}")
        print(f"  Invalid reasons: {dict(cam_invalid_reasons)}")
        print(f"  Alpha sum: mean={summary['alpha_sum']['mean']:.4f}, median={summary['alpha_sum']['median']:.4f}")
        print(f"  Alpha max: mean={summary['alpha_max']['mean']:.4f}, max={summary['alpha_max']['max']:.4f}")
        print(f"  BBox in-bounds ratio: {summary['bbox_in_bounds_ratio']*100:.1f}%")
        if save_debug_images:
            print(f"  Debug images saved: {debug_saved}")

    metrics_path = os.path.join(args.output_dir, "camera_roi_metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for m in all_metrics:
            f.write(json.dumps(m, default=str) + "\n")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump({"per_camera": camera_summary}, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("CAMERA ROI DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    print(f"{'Camera':<10} {'Valid':<8} {'Ratio':<10} {'AlphaMean':<12} {'AlphaMax':<12} {'InBounds':<10}")
    for cam_id in sorted(camera_summary.keys()):
        s = camera_summary[cam_id]
        print(f"{cam_id:<10} {s['valid_count']:<8} {s['valid_ratio']*100:<10.1f} "
              f"{s['alpha_sum']['mean']:<12.4f} {s['alpha_max']['mean']:<12.4f} "
              f"{s['bbox_in_bounds_ratio']*100:<10.1f}%")

    print(f"\nMetrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    if save_debug_images and HAS_MATPLOTLIB:
        print(f"Debug images: {debug_dir}")
    return True


def run_cam_frame_mapping_check(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("CAM-FRAME MAPPING CHECK MODE")
    print("=" * 70)

    annotation_cam_frames = []
    for frame_id, annots in sampler.annotations.items():
        if not isinstance(annots, list):
            continue
        fi = int(frame_id)
        for p in annots:
            pid = p.get('train_id') or p.get('new_id')
            if pid is None:
                continue
            annot_cam_id = p.get('camera_id')
            if annot_cam_id is None:
                continue
            cam_id = f"C{annot_cam_id + 1}"
            annotation_cam_frames.append({
                'annotation_cam_id': cam_id,
                'annotation_frame_idx': fi,
                'person_id': pid,
            })

    random.shuffle(annotation_cam_frames)
    check_samples = annotation_cam_frames[:500]

    mapping_results = []
    warnings = []

    for sample in check_samples:
        cam_id = sample['annotation_cam_id']
        frame_idx = sample['annotation_frame_idx']
        person_id = sample['person_id']

        dataset_idx = sampler.get_dataset_index(cam_id, frame_idx)
        key_found = (cam_id, frame_idx) in sampler.cam_frame_to_index

        dataset_cam_name = None
        dataset_frame = None
        image_path = None

        if dataset_idx is not None:
            raw_batch = trainer.train_dataset[dataset_idx]
            if hasattr(raw_batch, 'camera_name'):
                dataset_cam_name = raw_batch.camera_name
            elif hasattr(raw_batch, 'cam_id'):
                dataset_cam_name = raw_batch.cam_id
            if hasattr(raw_batch, 'frame_id'):
                dataset_frame = raw_batch.frame_id
            elif hasattr(raw_batch, 'timestamp'):
                dataset_frame = raw_batch.timestamp
            if hasattr(raw_batch, 'image_path'):
                image_path = raw_batch.image_path

        mismatch = False
        warning_msg = None
        if not key_found:
            warning_msg = f"Key not found: ({cam_id}, {frame_idx}) for person {person_id}"
            mismatch = True
        elif dataset_cam_name is not None and dataset_cam_name != cam_id:
            warning_msg = f"Cam mismatch: annotation={cam_id}, dataset={dataset_cam_name}"
            mismatch = True

        if mismatch and warning_msg:
            warnings.append(warning_msg)

        mapping_results.append({
            'annotation_cam_id': cam_id,
            'annotation_frame_idx': frame_idx,
            'person_id': person_id,
            'cam_frame_key_found': key_found,
            'dataset_index': dataset_idx,
            'dataset_camera_name': dataset_cam_name,
            'dataset_frame': str(dataset_frame) if dataset_frame is not None else None,
            'image_path': image_path,
            'mismatch': mismatch,
            'warning': warning_msg,
        })

    total_checked = len(mapping_results)
    total_found = sum(1 for r in mapping_results if r['cam_frame_key_found'])
    total_mismatch = sum(1 for r in mapping_results if r['mismatch'])

    cam_check_stats = defaultdict(lambda: {"total": 0, "found": 0, "mismatch": 0})
    for r in mapping_results:
        cam = r['annotation_cam_id']
        cam_check_stats[cam]["total"] += 1
        if r['cam_frame_key_found']:
            cam_check_stats[cam]["found"] += 1
        if r['mismatch']:
            cam_check_stats[cam]["mismatch"] += 1

    summary = {
        "total_checked": total_checked,
        "total_found": total_found,
        "total_mismatch": total_mismatch,
        "found_ratio": total_found / max(total_checked, 1),
        "mismatch_ratio": total_mismatch / max(total_checked, 1),
        "per_camera_stats": {cam: dict(stats) for cam, stats in cam_check_stats.items()},
        "warnings": warnings[:50],
        "total_warnings": len(warnings),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    results_path = os.path.join(args.output_dir, "mapping_results.jsonl")
    with open(results_path, 'w') as f:
        for r in mapping_results:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"\nTotal checked: {total_checked}")
    print(f"Key found: {total_found} ({summary['found_ratio']*100:.1f}%)")
    print(f"Mismatches: {total_mismatch} ({summary['mismatch_ratio']*100:.1f}%)")
    print(f"\nPer-camera stats:")
    for cam in sorted(cam_check_stats.keys()):
        stats = cam_check_stats[cam]
        found_ratio = stats['found'] / max(stats['total'], 1) * 100
        print(f"  {cam}: total={stats['total']}, found={stats['found']} ({found_ratio:.1f}%), mismatch={stats['mismatch']}")

    if warnings:
        print(f"\nWarnings (first 10):")
        for w in warnings[:10]:
            print(f"  {w}")

    print(f"\nSummary: {summary_path}")
    return True


def run_train_teacher_camera_aware_alpha_curriculum(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("TRAIN TEACHER CAMERA-AWARE ALPHA CURRICULUM MODE")
    print("=" * 70)

    allowed_cams_str = getattr(args, 'allowed_cameras', 'C1,C4,C6,C7')
    allowed_cameras = [c.strip() for c in allowed_cams_str.split(',')] if allowed_cams_str else None
    if allowed_cameras:
        print(f"Allowed cameras: {allowed_cameras}")

    curriculum_stages = [
        {"stage": 1, "min_alpha_sum": 1e-1, "allowed_cameras": allowed_cameras, "steps": 500},
        {"stage": 2, "min_alpha_sum": 5e-2, "allowed_cameras": allowed_cameras, "steps": 500},
        {"stage": 3, "min_alpha_sum": 1e-2, "allowed_cameras": allowed_cameras, "steps": 500},
        {"stage": 4, "min_alpha_sum": 1e-1, "allowed_cameras": None, "steps": 500},
    ]

    print("Collecting fixed eval samples (alpha >= 1e-1, allowed cameras only)...")
    orig_alpha = args.min_alpha_sum
    orig_mult = args.candidate_multiplier
    args.min_alpha_sum = 1e-1
    args.candidate_multiplier = max(orig_mult, 20)

    if allowed_cameras:
        eval_candidates_filtered = []
        eval_candidates, _, _ = collect_clean_candidates(args, trainer, sampler, batch_builder)
        for c in eval_candidates:
            if c['cam_id'] in allowed_cameras:
                eval_candidates_filtered.append(c)
        eval_candidates = eval_candidates_filtered
    else:
        eval_candidates, _, _ = collect_clean_candidates(args, trainer, sampler, batch_builder)

    args.candidate_multiplier = orig_mult
    args.min_alpha_sum = orig_alpha

    eval_candidates = sorted(eval_candidates, key=lambda c: c['alpha_sum'], reverse=True)
    target = args.target_valid_count or (args.P * args.K)
    eval_selected = eval_candidates[:target] if len(eval_candidates) >= target else eval_candidates
    print(f"Fixed eval samples: {len(eval_selected)} ROIs")

    fixed_eval_path = os.path.join(args.output_dir, "fixed_eval_samples.json")
    with open(fixed_eval_path, 'w') as f:
        json.dump(eval_selected, f, indent=2, default=str)

    train_log = []
    t_start = time.time()
    global_step = 0

    for stage_def in curriculum_stages:
        stage = stage_def["stage"]
        min_alpha = stage_def["min_alpha_sum"]
        stage_allowed = stage_def["allowed_cameras"]
        n_steps = stage_def["steps"]

        stage_allowed_str = ",".join(stage_allowed) if stage_allowed else "ALL"
        print(f"\n{'='*50}")
        print(f"Stage {stage}: min_alpha_sum={min_alpha}, allowed_cameras={stage_allowed_str}, steps={n_steps}")
        print(f"{'='*50}")

        args.min_alpha_sum = min_alpha

        stage_cam_stats = defaultdict(lambda: {"sampled": 0, "valid": 0, "invalid_reasons": defaultdict(int)})

        for step in range(n_steps):
            t_iter_start = time.time()

            train_candidates, train_invalid, n_cand = collect_clean_candidates(
                args, trainer, sampler, batch_builder)

            if stage_allowed:
                train_candidates = [c for c in train_candidates if c['cam_id'] in stage_allowed]

            train_selected = train_candidates[:args.P * args.K] if len(train_candidates) >= args.P * args.K else train_candidates

            for roi in train_selected:
                stage_cam_stats[roi['cam_id']]["sampled"] += 1

            trainer.model.optimizer.zero_grad()

            step_loss = 0.0
            step_cos = []
            step_alpha = []
            valid_count = 0
            invalid_count = 0

            for roi in train_selected:
                gpu_batch = batch_builder.get_batch_by_cam_frame(roi['cam_id'], roi['frame_idx'])
                if gpu_batch is None:
                    invalid_count += 1
                    stage_cam_stats[roi['cam_id']]["invalid_reasons"]['no_batch'] += 1
                    continue

                render_out = trainer.model(
                    gpu_batch, train=False, frame_id=0, render_person_feature=True
                )
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')

                bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
                f_v, pool_stats = opacity_roi_pooling(
                    person_feature_map, person_opacity_map, bbox_t,
                    denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                )

                if f_v is None:
                    invalid_count += 1
                    stage_cam_stats[roi['cam_id']]["invalid_reasons"]['roi_pool_failed'] += 1
                    continue

                teacher_feat = normalize_feat(
                    torch.as_tensor(roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
                )
                cos_sim = torch.dot(f_v, teacher_feat)
                step_cos.append(float(cos_sim.item()))
                step_alpha.append(float(pool_stats.get('alpha_sum', 0)))

                if args.teacher_loss_type == 'cosine':
                    l = 1 - cos_sim
                else:
                    l = F.mse_loss(f_v, teacher_feat)
                step_loss = step_loss + l
                valid_count += 1
                stage_cam_stats[roi['cam_id']]["valid"] += 1

            if valid_count == 0:
                continue

            avg_loss = step_loss / valid_count
            avg_loss.backward()

            pf = trainer.model.get_person_feature()
            pf_before = pf.detach().clone()
            grad_info = get_grad_stats(pf)
            grad_norm_before = grad_info["grad_norm"]
            torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
            grad_norm_after = pf.grad.norm().item() if pf.grad is not None else 0.0
            trainer.model.optimizer.step()
            delta = pf.detach() - pf_before
            param_delta_norm = float(delta.norm().item())

            record = {
                "global_step": global_step,
                "stage": stage,
                "min_alpha_sum": min_alpha,
                "allowed_cameras": stage_allowed_str,
                "train_loss_teacher": float(avg_loss.item()),
                "train_cos_fv_tv_mean": float(np.mean(step_cos)) if step_cos else 0,
                "valid_feature_count": valid_count,
                "invalid_feature_count": invalid_count,
                "alpha_sum_min": min(step_alpha) if step_alpha else 0,
                "alpha_sum_mean": float(np.mean(step_alpha)) if step_alpha else 0,
                "alpha_sum_max": max(step_alpha) if step_alpha else 0,
                "grad_norm_before_clip": grad_norm_before,
                "grad_norm_after_clip": grad_norm_after,
                "param_delta_norm": param_delta_norm,
                "nan_count": int(grad_info["nan_count"]),
                "inf_count": int(grad_info["inf_count"]),
                "iter_time": float(time.time() - t_iter_start),
            }

            if step % 50 == 0 or step == n_steps - 1:
                with torch.no_grad():
                    eval_cos = []
                    eval_loss_sum = 0.0
                    eval_valid = 0
                    for roi in eval_selected:
                        gpu_batch = batch_builder.get_batch_by_cam_frame(roi['cam_id'], roi['frame_idx'])
                        if gpu_batch is None:
                            continue
                        render_out = trainer.model(
                            gpu_batch, train=False, frame_id=0, render_person_feature=True
                        )
                        person_feature_map = render_out['person_feature_map']
                        person_opacity_map = render_out.get('person_opacity_map')
                        bbox_t = torch.tensor(roi['bbox'], dtype=torch.float32, device=trainer.device)
                        f_v, _ = opacity_roi_pooling(
                            person_feature_map, person_opacity_map, bbox_t,
                            denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
                        )
                        if f_v is None:
                            continue
                        teacher_feat = normalize_feat(
                            torch.as_tensor(roi['teacher_emb'], dtype=torch.float32, device=trainer.device).squeeze()
                        )
                        c = float(torch.dot(f_v, teacher_feat).item())
                        eval_cos.append(c)
                        if args.teacher_loss_type == 'cosine':
                            eval_loss_sum += 1 - c
                        eval_valid += 1

                    record['fixed_loss_teacher'] = float(eval_loss_sum / eval_valid) if eval_valid > 0 else 0
                    record['fixed_cos_fv_tv_mean'] = float(np.mean(eval_cos)) if eval_cos else 0
                    record['fixed_cos_fv_tv_min'] = float(min(eval_cos)) if eval_cos else 0
                    record['fixed_cos_fv_tv_max'] = float(max(eval_cos)) if eval_cos else 0

            valid_ratio_by_cam = {}
            sample_count_by_cam = {}
            invalid_reason_by_cam = {}
            for cam_id, stats in stage_cam_stats.items():
                valid_ratio_by_cam[cam_id] = stats['valid'] / max(stats['sampled'], 1)
                sample_count_by_cam[cam_id] = stats['sampled']
                invalid_reason_by_cam[cam_id] = dict(stats['invalid_reasons'])

            record['valid_ratio_by_camera'] = valid_ratio_by_cam
            record['sample_count_by_camera'] = sample_count_by_cam
            record['invalid_reason_by_camera'] = invalid_reason_by_cam

            train_log.append(record)

            if step % args.log_interval == 0:
                msg = (f"[CAM_ALPHA] Stage={stage} Step={step:5d}: "
                       f"train_loss={record['train_loss_teacher']:.4f} "
                       f"train_cos={record['train_cos_fv_tv_mean']:.4f} "
                       f"grad={grad_norm_before:.4e}->{grad_norm_after:.4f} "
                       f"valid={valid_count}")
                if 'fixed_cos_fv_tv_mean' in record:
                    msg += f" | eval_cos={record['fixed_cos_fv_tv_mean']:.4f}"
                msg += f" t={record['iter_time']:.2f}s"
                print(msg)

            global_step += 1

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r, default=str) + "\n")

    eval_records = [r for r in train_log if 'fixed_cos_fv_tv_mean' in r]
    if eval_records:
        first_cos = eval_records[0]['fixed_cos_fv_tv_mean']
        last_cos = eval_records[-1]['fixed_cos_fv_tv_mean']
        cos_delta = last_cos - first_cos
    else:
        first_cos = last_cos = cos_delta = 0

    summary = {
        "total_steps": len(train_log),
        "eval_records_count": len(eval_records),
        "first_eval_cos": first_cos,
        "last_eval_cos": last_cos,
        "eval_cos_delta": cos_delta,
        "learning_trend_positive": cos_delta > 0.02,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"CAMERA-AWARE ALPHA CURRICULUM RESULT")
    print(f"{'='*70}")
    print(f"Fixed eval cos: first={first_cos:.4f}, last={last_cos:.4f}, delta={cos_delta:+.4f}")
    print(f"SUCCESS: {cos_delta > 0.02}")
    return True


def run_phase12a_gaussianset_single_roi_overfit(args, trainer, sampler, batch_builder, prototypes, valid_mask):
    print("\n" + "=" * 70)
    print("PHASE 12A: GAUSSIAN-SET SINGLE ROI OVERFIT")
    print("=" * 70)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    allowed_cameras = getattr(args, 'allowed_cameras', None)
    if allowed_cameras:
        if isinstance(allowed_cameras, str):
            allowed_cameras = [c.strip() for c in allowed_cameras.split(',')]
    else:
        allowed_cameras = ['C1', 'C4', 'C6', 'C7']

    print(f"Selecting clean ROI from cameras: {allowed_cameras}")
    print("Collecting candidates...")
    candidates, invalid_reasons, n_cand = collect_clean_candidates(
        args, trainer, sampler, batch_builder, allowed_cameras=allowed_cameras)

    if len(candidates) == 0:
        print("ERROR: No clean candidates found. Cannot proceed.")
        return False

    candidates = sorted(candidates, key=lambda c: c['alpha_sum'], reverse=True)
    selected_roi = candidates[0]

    print(f"\nSelected ROI:")
    print(f"  cam_id: {selected_roi['cam_id']}")
    print(f"  frame_idx: {selected_roi['frame_idx']}")
    print(f"  person_id: {selected_roi['person_id']}")
    print(f"  alpha_sum: {selected_roi['alpha_sum']:.4f}")
    print(f"  bbox_area: {selected_roi['bbox_area']:.1f}")
    print(f"  bbox: {selected_roi['bbox']}")

    sample_save_path = os.path.join(args.output_dir, "selected_sample.json")
    with open(sample_save_path, 'w') as f:
        json.dump(selected_roi, f, indent=2, default=str)

    teacher_embedding = torch.as_tensor(
        selected_roi['teacher_emb'], dtype=torch.float32, device=trainer.device
    ).squeeze()
    teacher_embedding = normalize_feat(teacher_embedding)

    print(f"\nFreezing geometry, optimizing person_feature only")
    pf = trainer.model.get_person_feature()
    pf.requires_grad_(True)

    gaussians_xyz = trainer.model.positions.detach()
    N = gaussians_xyz.shape[0]
    print(f"Total Gaussians: {N}")

    gpu_batch = batch_builder.get_batch_by_cam_frame(
        selected_roi['cam_id'], selected_roi['frame_idx'])
    if gpu_batch is None:
        print("ERROR: Could not get batch for selected ROI")
        return False

    if hasattr(gpu_batch, 'C2W'):
        c2w = gpu_batch.C2W
    elif hasattr(gpu_batch, 'T_to_world'):
        c2w = gpu_batch.T_to_world
    else:
        print("ERROR: Cannot get camera extrinsics (C2W)")
        print(f"Available batch attributes: {[a for a in dir(gpu_batch) if not a.startswith('_')]}")
        return False

    if isinstance(c2w, torch.Tensor):
        if c2w.dim() == 3:
            c2w = c2w[0]
        elif c2w.dim() == 4:
            c2w = c2w[0, 0]
        world_to_cam = torch.inverse(c2w)
    else:
        print("ERROR: C2W is not a tensor")
        return False

    if hasattr(gpu_batch, 'intrinsics'):
        intr = gpu_batch.intrinsics
    else:
        print("ERROR: Cannot get camera intrinsics")
        print(f"Available batch attributes: {[a for a in dir(gpu_batch) if not a.startswith('_')]}")
        return False

    if isinstance(intr, torch.Tensor):
        if intr.dim() == 2:
            intr = intr[0]
        elif intr.dim() == 3:
            intr = intr[0, 0]
        fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
        camera_matrix = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=trainer.device)
    elif isinstance(intr, (list, tuple)):
        fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
        camera_matrix = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=trainer.device)
    elif isinstance(intr, dict):
        fx = intr.get('fx', intr.get('focal_length', None))
        fy = intr.get('fy', fx if isinstance(fx, (int, float)) else intr.get('focal_length', None))
        cx = intr.get('cx', intr.get('principal_point', [None, None])[0] if isinstance(intr.get('principal_point'), (list, tuple)) else None)
        cy = intr.get('cy', intr.get('principal_point', [None, None])[1] if isinstance(intr.get('principal_point'), (list, tuple)) else None)
        if fx is None or fy is None or cx is None or cy is None:
            print(f"ERROR: Cannot parse intrinsics dict: {intr}")
            return False
        camera_matrix = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=trainer.device)
    else:
        print(f"ERROR: Unknown intrinsics format: {type(intr)}")
        return False

    bbox = selected_roi['bbox']
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    print(f"\nProjecting {N} Gaussians to 2D...")
    with torch.no_grad():
        xyz_hom = torch.cat([gaussians_xyz, torch.ones(N, 1, device=trainer.device)], dim=1)
        xyz_cam = (world_to_cam @ xyz_hom.T).T
        xyz_cam = xyz_cam[:, :3]

        z = xyz_cam[:, 2]
        depth_mask = z > 0.01

        x_proj = (camera_matrix[0, 0] * xyz_cam[:, 0] / z) + camera_matrix[0, 2]
        y_proj = (camera_matrix[1, 1] * xyz_cam[:, 1] / z) + camera_matrix[1, 2]

        inside_bbox_mask = (
            (x_proj >= x1) & (x_proj <= x2) &
            (y_proj >= y1) & (y_proj <= y2) &
            depth_mask
        )

    num_inside = int(inside_bbox_mask.sum().item())
    print(f"Gaussians inside bbox: {num_inside}")

    if num_inside == 0:
        print("WARNING: No Gaussians inside bbox. Using fallback: render + opacity-based selection")
        with torch.no_grad():
            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            if person_opacity_map is not None:
                H, W = person_opacity_map.shape
                x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                x2_int, y2_int = min(W, int(x2)), min(H, int(y2))
                roi_opacity = person_opacity_map[y1_int:y2_int, x1_int:x2_int]
                opacity_threshold = float(roi_opacity.max().item()) * 0.1
                opacity_mask_2d = roi_opacity > opacity_threshold

                inside_indices = []
                for gy in range(y1_int, y2_int):
                    for gx in range(x1_int, x2_int):
                        if gy < H and gx < W and opacity_mask_2d[gy - y1_int, gx - x1_int]:
                            inside_indices.append((gx, gy))

                print(f"Fallback: {len(inside_indices)} high-opacity pixels in ROI")

                if len(inside_indices) == 0:
                    print("ERROR: No high-opacity pixels found. Cannot proceed.")
                    return False

                bbox_center_x = (x1 + x2) / 2.0
                bbox_center_y = (y1 + y2) / 2.0
                max_dist = max(x2 - x1, y2 - y1) / 2.0

                gaussian_indices = []
                weights = []
                for gx, gy in inside_indices:
                    dist = ((gx - bbox_center_x) ** 2 + (gy - bbox_center_y) ** 2) ** 0.5
                    weight = float(person_opacity_map[gy, gx].item())
                    decay = max(0.0, 1.0 - dist / max_dist)
                    weights.append(weight * decay)

                print(f"Fallback mode: using pixel-based approximation")
                print(f"Weight sum: {sum(weights):.4f}")

                train_log = []
                t_start = time.time()
                best_cos = -999
                best_step = 0
                param_deltas = []
                total_nan = 0
                total_inf = 0

                for step in range(args.num_steps):
                    t_iter_start = time.time()
                    trainer.model.optimizer.zero_grad()

                    with torch.no_grad():
                        render_out = trainer.model(
                            gpu_batch, train=False, frame_id=0, render_person_feature=True
                        )
                        person_feature_map = render_out['person_feature_map']

                        gaussianset_features = []
                        gaussianset_weights = []
                        for idx, (gx, gy) in enumerate(inside_indices):
                            feat = person_feature_map[:, gy, gx]
                            gaussianset_features.append(feat)
                            gaussianset_weights.append(weights[idx])

                        gaussianset_features = torch.stack(gaussianset_features)
                        gaussianset_weights = torch.tensor(gaussianset_weights, device=trainer.device)
                        weight_sum = gaussianset_weights.sum() + args.denom_eps
                        student_embedding = (gaussianset_weights[:, None] * gaussianset_features).sum(dim=0) / weight_sum
                        student_embedding = normalize_feat(student_embedding)

                    cos_sim = torch.dot(student_embedding, teacher_embedding)
                    loss_teacher = 1 - cos_sim

                    loss_teacher.backward()

                    pf_before = pf.detach().clone()
                    grad_info = get_grad_stats(pf)
                    grad_norm_before = grad_info["grad_norm"]
                    grad_norm_after = 0.0
                    if pf.grad is not None:
                        torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
                        grad_norm_after = pf.grad.norm().item()
                    trainer.model.optimizer.step()

                    delta = pf.detach() - pf_before
                    param_delta_norm = float(delta.norm().item())
                    param_delta_max = float(delta.abs().max().item())

                    param_deltas.append(param_delta_norm)
                    total_nan += int(grad_info["nan_count"])
                    total_inf += int(grad_info["inf_count"])

                    if cos_sim.item() > best_cos:
                        best_cos = cos_sim.item()
                        best_step = step

                    record = {
                        "step": step,
                        "loss_teacher": float(loss_teacher.item()),
                        "cos_gaussianset_teacher": float(cos_sim.item()),
                        "num_gaussians_in_bbox": len(inside_indices),
                        "weight_sum": float(weight_sum.item()),
                        "weight_min": float(min(weights)),
                        "weight_mean": float(np.mean(weights)),
                        "weight_max": float(max(weights)),
                        "grad_norm_before_clip": grad_norm_before,
                        "grad_norm_after_clip": grad_norm_after,
                        "param_delta_norm": param_delta_norm,
                        "param_delta_max": param_delta_max,
                        "nan_count": int(grad_info["nan_count"]),
                        "inf_count": int(grad_info["inf_count"]),
                        "iter_time": float(time.time() - t_iter_start),
                    }
                    train_log.append(record)

                    if step % args.log_interval == 0 or step == 0:
                        print(f"[GAUSSIANSET] Step {step:4d}: loss={loss_teacher.item():.4f} "
                              f"cos={cos_sim.item():.4f} delta={param_delta_norm:.6e} "
                              f"t={record['iter_time']:.2f}s")

                metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
                with open(metrics_path, 'w') as f:
                    for r in train_log:
                        f.write(json.dumps(r, default=str) + "\n")

                if train_log:
                    first_cos = train_log[0]['cos_gaussianset_teacher']
                    last_cos = train_log[-1]['cos_gaussianset_teacher']
                    cos_delta = last_cos - first_cos
                else:
                    first_cos = last_cos = cos_delta = 0

                if cos_delta > 0.05:
                    verdict = "gaussianset_promising"
                elif num_inside == 0 and len(inside_indices) == 0:
                    verdict = "gaussianset_projection_failure"
                elif np.mean(param_deltas) < 1e-10:
                    verdict = "gaussianset_no_learning"
                else:
                    verdict = "gaussianset_promising"

                summary = {
                    "mode": "gaussianset_single_roi_overfit_fallback",
                    "best_cos": best_cos,
                    "first_cos": first_cos,
                    "last_cos": last_cos,
                    "cos_delta": cos_delta,
                    "convergence_steps_to_cos_threshold": best_step,
                    "num_gaussians_in_bbox": len(inside_indices),
                    "optimizer_delta": {
                        "mean_param_delta_norm": float(np.mean(param_deltas)) if param_deltas else 0,
                        "max_param_delta_norm": float(max(param_deltas)) if param_deltas else 0,
                    },
                    "nan_total": total_nan,
                    "inf_total": total_inf,
                    "verdict": verdict,
                }

                summary_path = os.path.join(args.output_dir, "summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)

                print(f"\n{'='*70}")
                print(f"GAUSSIAN-SET SINGLE ROI OVERFIT RESULT (FALLBACK)")
                print(f"{'='*70}")
                print(f"Best cos: {best_cos:.4f}")
                print(f"Cos delta: {cos_delta:+.4f}")
                print(f"Mean param delta: {np.mean(param_deltas):.6e}")
                print(f"NaN/Inf: {total_nan}/{total_inf}")
                print(f"VERDICT: {verdict}")
                print(f"SUCCESS: {'promising' in verdict}")

                return True

    gaussians_inside_indices = torch.where(inside_bbox_mask)[0]

    z_inside = xyz_cam[gaussians_inside_indices, 2]
    depth_valid = z_inside > 0.01

    try:
        with torch.no_grad():
            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_opacity_map = render_out.get('person_opacity_map')

        if person_opacity_map is not None:
            H, W = person_opacity_map.shape
            x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
            x2_int, y2_int = min(W, int(x2)), min(H, int(y2))

            opacity_values = []
            for idx in gaussians_inside_indices:
                px = int(x_proj[idx].item())
                py = int(y_proj[idx].item())
                if 0 <= px < W and 0 <= py < H:
                    opacity_values.append(float(person_opacity_map[py, px].item()))
                else:
                    opacity_values.append(0.0)

            opacity_tensor = torch.tensor(opacity_values, dtype=torch.float32, device=trainer.device)
            opacity_valid = opacity_tensor > 0.05
            final_mask = depth_valid & opacity_valid
        else:
            final_mask = depth_valid
    except Exception as e:
        print(f"WARNING: Opacity filtering failed ({e}), using depth-only filtering")
        final_mask = depth_valid

    final_indices = gaussians_inside_indices[final_mask]
    num_final = len(final_indices)
    print(f"Gaussians inside bbox after filtering: {num_final}")

    if num_final == 0:
        print("WARNING: No valid Gaussians after filtering. Using all inside-bbox Gaussians.")
        final_indices = gaussians_inside_indices[depth_valid]
        num_final = len(final_indices)
        if num_final == 0:
            print("ERROR: Still no valid Gaussians. Cannot proceed.")
            return False

    bbox_center_x = (x1 + x2) / 2.0
    bbox_center_y = (y1 + y2) / 2.0

    x_inside = x_proj[final_indices]
    y_inside = y_proj[final_indices]
    distances = torch.sqrt((x_inside - bbox_center_x) ** 2 + (y_inside - bbox_center_y) ** 2)

    max_bbox_dim = max(x2 - x1, y2 - y1)
    distance_decay = torch.clamp(1.0 - distances / max_bbox_dim, min=0.0)

    z_inside_filtered = z_inside[final_mask]
    depth_weights = torch.exp(-torch.clamp(z_inside_filtered - z_inside_filtered.min(), max=5.0))

    try:
        with torch.no_grad():
            render_out = trainer.model(
                gpu_batch, train=False, frame_id=0, render_person_feature=True
            )
            person_opacity_map = render_out.get('person_opacity_map')

        if person_opacity_map is not None:
            H, W = person_opacity_map.shape
            opacity_values = []
            for idx in final_indices:
                px = int(x_proj[idx].item())
                py = int(y_proj[idx].item())
                if 0 <= px < W and 0 <= py < H:
                    opacity_values.append(float(person_opacity_map[py, px].item()))
                else:
                    opacity_values.append(0.0)
            opacity_weights = torch.tensor(opacity_values, dtype=torch.float32, device=trainer.device)
        else:
            opacity_weights = torch.ones(num_final, dtype=torch.float32, device=trainer.device)
    except Exception:
        opacity_weights = torch.ones(num_final, dtype=torch.float32, device=trainer.device)

    weights = distance_decay * depth_weights * opacity_weights
    weight_sum = weights.sum() + args.denom_eps

    print(f"\nGaussian-set aggregation config:")
    print(f"  num_gaussians: {num_final}")
    print(f"  weight_sum: {weight_sum.item():.4f}")
    print(f"  weight_min: {weights.min().item():.6f}")
    print(f"  weight_mean: {weights.mean().item():.6f}")
    print(f"  weight_max: {weights.max().item():.6f}")

    train_log = []
    t_start = time.time()
    best_cos = -999
    best_step = 0
    param_deltas = []
    total_nan = 0
    total_inf = 0

    final_indices_cpu = final_indices.cpu()

    for step in range(args.num_steps):
        t_iter_start = time.time()
        trainer.model.optimizer.zero_grad()

        person_features = pf[final_indices_cpu].to(trainer.device)

        student_embedding = (weights[:, None] * person_features).sum(dim=0) / weight_sum
        student_embedding = normalize_feat(student_embedding)

        cos_sim = torch.dot(student_embedding, teacher_embedding)
        loss_teacher = 1 - cos_sim

        loss_teacher.backward()

        pf_before = pf.detach().clone()
        grad_info = get_grad_stats(pf)
        grad_norm_before = grad_info["grad_norm"]
        grad_norm_after = 0.0
        if pf.grad is not None:
            torch.nn.utils.clip_grad_norm_(pf, max_norm=args.grad_clip_norm)
            grad_norm_after = pf.grad.norm().item()
        trainer.model.optimizer.step()

        delta = pf.detach() - pf_before
        param_delta_norm = float(delta.norm().item())
        param_delta_max = float(delta.abs().max().item())

        param_deltas.append(param_delta_norm)
        total_nan += int(grad_info["nan_count"])
        total_inf += int(grad_info["inf_count"])

        if cos_sim.item() > best_cos:
            best_cos = cos_sim.item()
            best_step = step

        record = {
            "step": step,
            "loss_teacher": float(loss_teacher.item()),
            "cos_gaussianset_teacher": float(cos_sim.item()),
            "num_gaussians_in_bbox": num_final,
            "weight_sum": float(weight_sum.item()),
            "weight_min": float(weights.min().item()),
            "weight_mean": float(weights.mean().item()),
            "weight_max": float(weights.max().item()),
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "param_delta_norm": param_delta_norm,
            "param_delta_max": param_delta_max,
            "nan_count": int(grad_info["nan_count"]),
            "inf_count": int(grad_info["inf_count"]),
            "iter_time": float(time.time() - t_iter_start),
        }
        train_log.append(record)

        if step % args.log_interval == 0 or step == 0:
            print(f"[GAUSSIANSET] Step {step:4d}: loss={loss_teacher.item():.4f} "
                  f"cos={cos_sim.item():.4f} delta={param_delta_norm:.6e} "
                  f"t={record['iter_time']:.2f}s")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in train_log:
            f.write(json.dumps(r, default=str) + "\n")

    if train_log:
        first_cos = train_log[0]['cos_gaussianset_teacher']
        last_cos = train_log[-1]['cos_gaussianset_teacher']
        cos_delta = last_cos - first_cos
    else:
        first_cos = last_cos = cos_delta = 0

    if cos_delta > 0.05:
        verdict = "gaussianset_promising"
    elif num_final == 0:
        verdict = "gaussianset_projection_failure"
    elif np.mean(param_deltas) < 1e-10:
        verdict = "gaussianset_no_learning"
    else:
        verdict = "gaussianset_promising"

    summary = {
        "mode": "gaussianset_single_roi_overfit",
        "best_cos": best_cos,
        "first_cos": first_cos,
        "last_cos": last_cos,
        "cos_delta": cos_delta,
        "convergence_steps_to_cos_threshold": best_step,
        "num_gaussians_in_bbox": num_final,
        "optimizer_delta": {
            "mean_param_delta_norm": float(np.mean(param_deltas)) if param_deltas else 0,
            "max_param_delta_norm": float(max(param_deltas)) if param_deltas else 0,
        },
        "nan_total": total_nan,
        "inf_total": total_inf,
        "verdict": verdict,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"GAUSSIAN-SET SINGLE ROI OVERFIT RESULT")
    print(f"{'='*70}")
    print(f"Best cos: {best_cos:.4f}")
    print(f"Cos delta: {cos_delta:+.4f}")
    print(f"Mean param delta: {np.mean(param_deltas):.6e}")
    print(f"NaN/Inf: {total_nan}/{total_inf}")
    print(f"VERDICT: {verdict}")
    print(f"SUCCESS: {'promising' in verdict}")

    return True


def run_phase12b_gaussianset_progressive_overfit(args, trainer, sampler, batch_builder, allowed_cameras=None):
    """
    Phase 12B: Gaussian-Set Progressive Overfit.
    Test 1/2/4/8 ROI overfitting using direct Gaussian aggregation within bbox.
    """
    roi_count = args.progressive_roi_count
    same_person = getattr(args, 'same_person_cross_view', False)
    
    print(f"\n{'='*70}")
    print(f"PHASE 12B: GAUSSIAN-SET PROGRESSIVE OVERFIT")
    print(f"{'='*70}")
    print(f"ROI count: {roi_count}")
    print(f"Same person cross-view: {same_person}")
    print(f"Allowed cameras: {allowed_cameras}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    selected_samples = []
    attempts = 0
    max_attempts = roi_count * 200
    
    used_person_cam_frames = set()
    
    print("Selecting ROIs...")
    
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
                
                if attempts % 50 == 0:
                    print(f"  Attempt {attempts}: found {len(selected_samples)}/{roi_count} ROIs")
        
        if attempts % 100 == 0:
            print(f"  Attempt {attempts}: found {len(selected_samples)}/{roi_count} ROIs")
    
    if len(selected_samples) < roi_count:
        print(f"WARNING: Only found {len(selected_samples)} candidates (requested {roi_count})")
        return False
    
    print(f"Selected {len(selected_samples)} ROIs from {attempts} attempts")
    
    with open(os.path.join(args.output_dir, 'selected_samples.json'), 'w') as f:
        json.dump(selected_samples, f, indent=2, default=str)
    
    unique_persons = set(s['person_id'] for s in selected_samples)
    person_count = len(unique_persons)
    camera_count = len(set(s['cam_id'] for s in selected_samples))
    
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    metrics_log = []
    
    pf = trainer.model.get_person_feature()
    pf_before = pf.clone().detach()
    
    optimizer = torch.optim.Adam(
        [trainer.model._person_feature],
        lr=args.person_feature_lr,
    )
    
    start_time = time.time()
    
    best_cos_mean = -1.0
    best_step = 0
    
    for step in range(args.num_steps):
        step_start = time.time()
        
        optimizer.zero_grad()
        
        losses = []
        cos_values = []
        gaussianset_features = []
        teacher_features = []
        num_gaussians_list = []
        weight_sum_list = []
        
        for sample in selected_samples:
            gpu_batch = batch_builder.get_batch_by_cam_frame(sample['cam_id'], sample['frame_idx'])
            if gpu_batch is None:
                continue
            
            G, debug_info = gaussian_set_pooling(
                trainer.model, gpu_batch, sample['bbox'], sample['cam_id'], sample['frame_idx'], args, trainer.device
            )
            
            if G is None:
                continue
            
            T = torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=trainer.device)
            T = normalize_feat(T)
            
            cos_sim = F.cosine_similarity(G.unsqueeze(0), T.unsqueeze(0)).item()
            loss_i = 1.0 - F.cosine_similarity(G.unsqueeze(0), T.unsqueeze(0))
            
            gaussianset_features.append(G)
            teacher_features.append(T)
            losses.append(loss_i)
            cos_values.append(cos_sim)
            num_gaussians_list.append(debug_info['num_gaussians_in_bbox'])
            weight_sum_list.append(debug_info['weight_sum'])
        
        if len(losses) == 0:
            continue
        
        loss = torch.stack(losses).mean()
        
        loss.backward()
        
        grad_norm_before_clip = None
        if trainer.model._person_feature.grad is not None:
            grad_norm_before_clip = trainer.model._person_feature.grad.norm().item()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([trainer.model._person_feature], args.grad_clip_norm)
        
        optimizer.step()
        
        pf_after = trainer.model.get_person_feature()
        param_delta = (pf_after - pf_before).norm().item()
        pf_before = pf_after.clone().detach()
        
        cos_mean = np.mean(cos_values) if cos_values else 0.0
        cos_min = np.min(cos_values) if cos_values else 0.0
        cos_max = np.max(cos_values) if cos_values else 0.0
        
        same_cos = None
        diff_cos = None
        gap = None
        positive_pair_count = 0
        negative_pair_count = 0
        
        if person_count >= 2:
            same_cos_list = []
            diff_cos_list = []
            
            person_to_indices = {}
            for idx, s in enumerate(selected_samples):
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
            neg_pairs_checked = 0
            for _ in range(min(100, len(all_indices) * (len(all_indices) - 1) // 2)):
                i, j = random.sample(all_indices, 2)
                if i >= len(gaussianset_features) or j >= len(gaussianset_features):
                    continue
                pid_i = selected_samples[i]['person_id']
                pid_j = selected_samples[j]['person_id']
                if pid_i != pid_j:
                    dc = F.cosine_similarity(
                        gaussianset_features[i].unsqueeze(0),
                        gaussianset_features[j].unsqueeze(0)
                    ).item()
                    diff_cos_list.append(dc)
                    negative_pair_count += 1
                    neg_pairs_checked += 1
                    if neg_pairs_checked >= 50:
                        break
            
            if same_cos_list:
                same_cos = np.mean(same_cos_list)
            if diff_cos_list:
                diff_cos = np.mean(diff_cos_list)
            if same_cos is not None and diff_cos is not None:
                gap = same_cos - diff_cos
        
        if cos_mean > best_cos_mean:
            best_cos_mean = cos_mean
            best_step = step
        
        grad_norm_after_clip = None
        if trainer.model._person_feature.grad is not None:
            grad_norm_after_clip = trainer.model._person_feature.grad.norm().item()
        
        nan_count = torch.isnan(trainer.model._person_feature).sum().item()
        inf_count = torch.isinf(trainer.model._person_feature).sum().item()
        
        step_time = time.time() - step_start
        
        if step % args.log_interval == 0:
            print(f"[CLEAN_GAUSSIANSET] Step {step:5d}: "
                  f"loss={loss.item():.4f} "
                  f"cos_mean={cos_mean:.4f} "
                  f"cos_min={cos_min:.4f} "
                  f"cos_max={cos_max:.4f} "
                  f"grad={grad_norm_before_clip:.4e}"
                  f"->{grad_norm_after_clip:.4e} "
                  f"delta={param_delta:.6e} "
                  f"num_gauss={np.mean(num_gaussians_list):.0f} "
                  f"t={step_time:.2f}s")
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
            'same_cos': float(same_cos) if same_cos is not None else None,
            'diff_cos': float(diff_cos) if diff_cos is not None else None,
            'gap': float(gap) if gap is not None else None,
            'positive_pair_count': positive_pair_count,
            'negative_pair_count': negative_pair_count,
            'grad_norm_before_clip': float(grad_norm_before_clip) if grad_norm_before_clip is not None else 0,
            'grad_norm_after_clip': float(grad_norm_after_clip) if grad_norm_after_clip is not None else 0,
            'param_delta_norm': float(param_delta),
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
    
    first_gap = metrics_log[0]['gap'] if metrics_log else None
    last_gap = metrics_log[-1]['gap'] if metrics_log else None
    gap_delta = (last_gap - first_gap) if (first_gap is not None and last_gap is not None) else None
    
    param_deltas = [m['param_delta_norm'] for m in metrics_log if m['param_delta_norm'] > 0]
    total_nan = sum(m['nan_count'] for m in metrics_log)
    total_inf = sum(m['inf_count'] for m in metrics_log)
    
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
        if last_cos > 0.90 and (gap is not None and gap > 0):
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
    if gap is not None:
        print(f"Gap: {gap:.4f}")
    print(f"Mean param delta: {np.mean(param_deltas):.6e}")
    print(f"NaN/Inf: {total_nan}/{total_inf}")
    print(f"VERDICT: {verdict}")
    print(f"SUCCESS: {'success' in verdict}")
    
    return "success" in verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/apps/wildtrack_full_3dgut.yaml')
    parser.add_argument('--mode', type=str, default='diagnostic',
                        choices=['diagnostic', 'train', 'train_sequential_safe', 'ablate_denom_eps',
                                 'train_sequential_mv_stopgrad', 'optimizer_sanity',
                                 'overfit_teacher_single_batch', 'train_teacher_fixed_eval',
                                 'teacher_gap_diagnostic', 'single_roi_overfit', 'progressive_overfit',
                                 'clean_8roi_overfit', 'roi_quality_scan',
                                 'train_teacher_clean_fixed_eval', 'train_teacher_alpha_curriculum',
                                 'train_clean_teacher_mv_fixed_eval',
                                 'camera_roi_diagnostic', 'cam_frame_mapping_check',
                                 'train_teacher_camera_aware_alpha_curriculum',
                                 'phase12a_gaussianset_single_roi_overfit',
                                 'phase12b_gaussianset_progressive_overfit'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/phase11B_v4_gradient_stabilized')
    parser.add_argument('--person_feature_dim', type=int, default=512)
    parser.add_argument('--person_feature_lr', type=float, default=1e-5)
    parser.add_argument('--P', type=int, default=4)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--denom_eps', type=float, default=1e-1)
    parser.add_argument('--detach_opacity_weight', action='store_true', default=True)
    parser.add_argument('--teacher_loss_type', type=str, default='cosine', choices=['cosine', 'mse'])
    parser.add_argument('--lambda_teacher', type=float, default=1.0)
    parser.add_argument('--lambda_mv', type=float, default=0.5)
    parser.add_argument('--lambda_proto', type=float, default=0.05)
    parser.add_argument('--enable_proto_after_steps', type=int, default=-1)
    parser.add_argument('--tau_mv', type=float, default=0.2)
    parser.add_argument('--tau_proto', type=float, default=0.5)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prototype_path', type=str,
                        default='/data02/zhangrunxiang/data/Wildtrack/reid_teacher_prototypes.pt')
    parser.add_argument('--clean_batch', action='store_true', default=False)
    parser.add_argument('--candidate_multiplier', type=int, default=10)
    parser.add_argument('--min_alpha_sum', type=float, default=1e-1)
    parser.add_argument('--min_bbox_area', type=float, default=100)
    parser.add_argument('--require_unclamped', action='store_true', default=False)
    parser.add_argument('--target_valid_count', type=int, default=None)
    parser.add_argument('--num_scan_samples', type=int, default=5000)
    parser.add_argument('--samples_per_camera', type=int, default=100)
    parser.add_argument('--save_debug_images', action='store_true', default=False)
    parser.add_argument('--debug_images_per_camera', type=int, default=5)
    parser.add_argument('--allowed_cameras', type=str, default=None,
                        help='Comma-separated list of allowed camera IDs, e.g. C1,C4,C6,C7')
    parser.add_argument('--use_pre_collected_pool', action='store_true', default=False)
    parser.add_argument('--pre_collect_pool_size', type=int, default=1000)
    parser.add_argument('--candidate_pool_path', type=str, default=None)
    parser.add_argument('--save_candidate_pool', action='store_true', default=True)
    parser.add_argument('--progressive_roi_count', type=int, default=1)
    parser.add_argument('--same_person_cross_view', action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.output_dir = args.output_dir.rstrip('/')
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 70)
    print("Phase 11B v4: Gradient-stabilized Frozen Multi-view Identity Training")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:30s} = {v}")

    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)

    conf.model.person_feature_dim = args.person_feature_dim
    conf.model.person_feature_lr = args.person_feature_lr
    conf.loss.use_reid = True
    conf.loss.lambda_reid = 0.0

    print("\nInitializing trainer...")
    trainer = Trainer3DGRUT(conf)
    print(f"Gaussians: {trainer.model.num_gaussians}")

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=trainer.device)
        if '_person_feature' in ckpt['model_state_dict']:
            ckpt_dim = ckpt['model_state_dict']['_person_feature'].shape[1]
            if ckpt_dim != args.person_feature_dim:
                print(f"  Removing person_feature (ckpt_dim={ckpt_dim}, model_dim={args.person_feature_dim})")
                del ckpt['model_state_dict']['_person_feature']
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        ckpt_candidates = [
            'runs/phase10B_opacity_lam005_lr1e4/latest.pth',
            'runs/phase10C_topk_detach_lam005_lr1e4_stable/latest.pth',
        ]
        for ckpt_path in ckpt_candidates:
            if os.path.exists(ckpt_path):
                print(f"Loading stable checkpoint: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=trainer.device)
                if '_person_feature' in ckpt['model_state_dict']:
                    ckpt_dim = ckpt['model_state_dict']['_person_feature'].shape[1]
                    if ckpt_dim != args.person_feature_dim:
                        print(f"  Removing person_feature (ckpt_dim={ckpt_dim}, model_dim={args.person_feature_dim})")
                        del ckpt['model_state_dict']['_person_feature']
                trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
                break
        else:
            print("[Warning] No stable checkpoint found, using default initialization")

    freeze_params = [
        trainer.model.positions, trainer.model.rotation, trainer.model.scale,
        trainer.model.density, trainer.model.features_albedo, trainer.model.features_specular,
    ]
    for param in freeze_params:
        if param is not None:
            param.requires_grad_(False)

    trainer.model._person_feature.requires_grad_(True)

    person_feature_param = trainer.model._person_feature
    trainer.model.optimizer = torch.optim.Adam(
        [{'params': person_feature_param, 'lr': args.person_feature_lr}],
        eps=1e-8, weight_decay=0,
    )

    trainable_params = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\n[Frozen Geometry] Trainable parameters: {len(trainable_params)}")
    print(f"  Names: {trainable_params}")
    print(f"  Total: {trainable_count:,}")

    if 'positions' in ' '.join(trainable_params) or 'scale' in ' '.join(trainable_params):
        print("[ERROR] Geometry parameters are still trainable!")
        return False

    proto_data = torch.load(args.prototype_path, map_location=trainer.device, weights_only=False)
    prototypes = proto_data['prototypes'].to(trainer.device)
    valid_mask = proto_data['valid_mask'].to(trainer.device)
    print(f"\nLoaded prototypes: {prototypes.shape}, valid IDs: {valid_mask.sum().item()}")

    sampler = PkMultiViewSampler(trainer.train_dataset, num_person=args.P, num_views=args.K)
    batch_builder = FrozenMVBatchBuilder(trainer, sampler)

    allowed_cameras = None
    if getattr(args, 'allowed_cameras', None):
        allowed_cameras = [c.strip() for c in args.allowed_cameras.split(',')]

    if args.mode == 'diagnostic':
        return run_diagnostic(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'train_sequential_safe':
        return run_train_sequential_safe(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'ablate_denom_eps':
        return run_ablate_denom_eps(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'train_sequential_mv_stopgrad':
        return run_train_sequential_mv_stopgrad(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'optimizer_sanity':
        return run_optimizer_sanity(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'overfit_teacher_single_batch':
        return run_overfit_teacher_single_batch(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'train_teacher_fixed_eval':
        return run_train_teacher_fixed_eval(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'teacher_gap_diagnostic':
        return run_teacher_gap_diagnostic(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'single_roi_overfit':
        return run_single_roi_overfit(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'progressive_overfit':
        return run_progressive_overfit(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'clean_8roi_overfit':
        return run_clean_8roi_overfit(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'roi_quality_scan':
        return run_roi_quality_scan(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'train_teacher_clean_fixed_eval':
        return run_train_teacher_clean_fixed_eval(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'train_teacher_alpha_curriculum':
        return run_train_teacher_alpha_curriculum(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'train_clean_teacher_mv_fixed_eval':
        return run_train_clean_teacher_mv_fixed_eval(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'camera_roi_diagnostic':
        return run_camera_roi_diagnostic(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'cam_frame_mapping_check':
        return run_cam_frame_mapping_check(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'train_teacher_camera_aware_alpha_curriculum':
        return run_train_teacher_camera_aware_alpha_curriculum(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'phase12a_gaussianset_single_roi_overfit':
        return run_phase12a_gaussianset_single_roi_overfit(args, trainer, sampler, batch_builder, prototypes, valid_mask)
    elif args.mode == 'phase12b_gaussianset_progressive_overfit':
        return run_phase12b_gaussianset_progressive_overfit(args, trainer, sampler, batch_builder, allowed_cameras)
    else:
        return run_train(args, trainer, sampler, batch_builder, prototypes, valid_mask)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
