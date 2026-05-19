#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 11B v3: Frozen-geometry multi-view identity feature training.

Fixed data loading: uses exact (cam_id, frame_idx) indexing from dataset.indices.
L_reid = 1.0 * L_proto + 0.5 * L_mv + 0.1 * L_teacher
"""

import argparse
import os
import sys
import json
import time
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.roi_pooling import roi_pool


def save_checkpoint(model, optimizer, step, path):
    state = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def compute_supcon_loss(features, labels, temperature=0.2):
    device = features.device
    N = features.shape[0]
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


class PkMultiViewSampler:
    """Random timestamp + P persons + K views per person sampler."""

    def __init__(self, dataset, num_person, num_views, max_retries=10):
        self.dataset = dataset
        self.num_person = num_person
        self.num_views = num_views
        self.max_retries = max_retries

        self.annotations = dataset.annotations
        self.teacher_cache = dataset.teacher_cache

        # Step 1: Build (cam_id, frame_idx) -> dataset index from dataset.indices
        self.cam_frame_to_index = {}
        valid_frame_set = set()
        for idx, (cam_id, frame_idx) in enumerate(dataset.indices):
            fi = int(frame_idx)
            self.cam_frame_to_index[(cam_id, fi)] = idx
            valid_frame_set.add(fi)

        # Step 2: Build visibility index from annotations, only for valid frames
        # Step 3: Build set of (pid, frame_idx, cam_id) that have teacher embeddings
        self.timestamp_to_persons = defaultdict(set)
        self.person_cam_at_ts = defaultdict(lambda: defaultdict(set))
        self.valid_observations = set()  # (pid, frame_idx, cam_id) with teacher embedding

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
                
                # Check if teacher embedding exists
                bbox_dict = p.get('bbox', {})
                bbox_xyxy = [
                    bbox_dict.get('xmin', -1), bbox_dict.get('ymin', -1),
                    bbox_dict.get('xmax', -1), bbox_dict.get('ymax', -1)
                ]
                
                has_teacher = False
                if self.teacher_cache is not None:
                    from threedgrut.datasets.cache_key import make_cache_key
                    cache_key = make_cache_key(fi, cam_id, pid, bbox_xyxy)
                    cache_entry = self.teacher_cache.get(cache_key)
                    has_teacher = cache_entry is not None
                
                if has_teacher:
                    self.valid_observations.add((pid, fi, cam_id))
                    self.person_cam_at_ts[pid][fi].add(cam_id)
                    self.timestamp_to_persons[fi].add(pid)

        self.all_timestamps = sorted(self.timestamp_to_persons.keys())
        self.person_ids = sorted(self.person_cam_at_ts.keys())

        # Remove persons/timestamps with insufficient views
        valid_persons = set()
        for pid in self.person_ids:
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
              f"{len(valid_persons)} valid persons, {len(self.cam_frame_to_index)} entries, "
              f"{len(self.valid_observations)} teacher-obs")

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
    """Given multi-view observations, build and execute rendering batches."""

    def __init__(self, trainer, sampler):
        self.trainer = trainer
        self.sampler = sampler

    def get_batch_by_cam_frame(self, cam_id, frame_idx):
        """Get a GPU batch for the specified camera and frame."""
        idx = self.sampler.get_dataset_index(cam_id, frame_idx)
        if idx is None:
            return None
        raw_batch = self.trainer.train_dataset[idx]
        return self.trainer.train_dataset.get_gpu_batch_with_intrinsics(raw_batch)

    def process_mv_samples(self, mv_samples):
        """
        Process multi-view samples:
        1. Group by unique (cam_id, frame_idx)
        2. Render each unique view once
        3. ROI pool each person's bbox
        Returns list of (person_id, feature, teacher_feature, cam_id, frame_idx)
        """
        view_groups = defaultdict(list)
        for ps in mv_samples:
            pid = ps['person_id']
            for cam_id, frame_idx in ps['views']:
                view_groups[(cam_id, int(frame_idx))].append(pid)

        unique_renders = len(view_groups)
        observations = []
        skipped_low_alpha = 0

        for (cam_id, frame_idx), pids in view_groups.items():
            gpu_batch = self.get_batch_by_cam_frame(cam_id, frame_idx)
            if gpu_batch is None:
                continue

            render_out = self.trainer.model(
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
                    skipped_low_alpha += 1
                    continue

                bbox = inst['bbox_xyxy']
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=self.trainer.device)

                f_v, pool_stats = roi_pool(
                    person_feature_map, bbox_t,
                    opacity_map=person_opacity_map,
                    pooling='opacity',
                    min_alpha_sum=1e-6,
                    detach_opacity_weight=True,
                )
                if f_v is None and person_opacity_map is not None:
                    f_v, pool_stats = roi_pool(
                        person_feature_map, bbox_t,
                        pooling='mean',
                    )
                    if f_v is not None:
                        skipped_low_alpha += 0.5

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

        return observations, skipped_low_alpha, unique_renders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--warmup_iters', type=int, default=0)
    parser.add_argument('--train_iters', type=int, default=10000)
    parser.add_argument('--lambda_reid', type=float, default=0.05)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--ckpt_dir', type=str, default='runs/phase11B_v3')
    parser.add_argument('--log_path', type=str, default='tools/phase11B_v3_log.json')
    parser.add_argument('--pooling', type=str, default='opacity')
    parser.add_argument('--detach_opacity_weight', action='store_true', default=True)
    parser.add_argument('--no_detach_opacity_weight', action='store_false', dest='detach_opacity_weight')
    parser.add_argument('--person_feature_lr', type=float, default=1e-4)
    parser.add_argument('--loss_type', type=str, default='frozen_mv_identity')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--mv_tau', type=float, default=0.2)
    parser.add_argument('--mv_num_person', type=int, default=4)
    parser.add_argument('--mv_num_views', type=int, default=2)
    parser.add_argument('--mv_loss_weight', type=float, default=0.5)
    parser.add_argument('--teacher_loss_weight', type=float, default=0.1)
    parser.add_argument('--experiment_name', type=str, default='phase11B_v3')
    parser.add_argument('--prototype_path', type=str,
                        default='/data02/zhangrunxiang/data/Wildtrack/reid_teacher_prototypes.pt')
    parser.add_argument('--debug_sampler', action='store_true')
    parser.add_argument('--checkpoint_every', type=int, default=500)
    args = parser.parse_args()

    if args.experiment_name:
        args.ckpt_dir = f'runs/{args.experiment_name}'
        args.log_path = f'tools/{args.experiment_name}_log.json'

    print("=" * 70)
    print("Phase 11B v3: Frozen-geometry Multi-view Identity Training")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:30s} = {v}")

    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)

    conf.model.person_feature_dim = 64  # Reduced from 512 to avoid gradient explosion in 3DGS feature rendering
    conf.model.person_feature_lr = args.person_feature_lr
    conf.loss.use_reid = True
    conf.loss.lambda_reid = args.lambda_reid

    print("\nInitializing trainer...")
    trainer = Trainer3DGRUT(conf)
    print(f"Gaussians: {trainer.model.num_gaussians}")

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=trainer.device)
        # Remove person_feature from checkpoint if dimension mismatch
        if '_person_feature' in ckpt['model_state_dict']:
            ckpt_dim = ckpt['model_state_dict']['_person_feature'].shape[1]
            if ckpt_dim != conf.model.person_feature_dim:
                print(f"  Removing person_feature from checkpoint (ckpt_dim={ckpt_dim}, model_dim={conf.model.person_feature_dim})")
                del ckpt['model_state_dict']['_person_feature']
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded from step {ckpt.get('step', 'unknown')}")
    else:
        ckpt_candidates = [
            'runs/phase10B_opacity_lam005_lr1e4/latest.pth',
            'runs/phase10C_topk_detach_lam005_lr1e4_stable/latest.pth',
        ]
        for ckpt_path in ckpt_candidates:
            if os.path.exists(ckpt_path):
                print(f"Loading stable checkpoint: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=trainer.device)
                # Remove person_feature from checkpoint if dimension mismatch
                if '_person_feature' in ckpt['model_state_dict']:
                    ckpt_dim = ckpt['model_state_dict']['_person_feature'].shape[1]
                    if ckpt_dim != conf.model.person_feature_dim:
                        print(f"  Removing person_feature from checkpoint (ckpt_dim={ckpt_dim}, model_dim={conf.model.person_feature_dim})")
                        del ckpt['model_state_dict']['_person_feature']
                trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
                break
        else:
            print("[Warning] No stable checkpoint found, using default initialization")

    # Freeze all geometry and appearance parameters
    freeze_params = [
        trainer.model.positions,
        trainer.model.rotation,
        trainer.model.scale,
        trainer.model.density,
        trainer.model.features_albedo,
        trainer.model.features_specular,
    ]
    for param in freeze_params:
        if param is not None:
            param.requires_grad_(False)
    
    # Ensure person_feature is trainable
    trainer.model._person_feature.requires_grad_(True)
    
    # Create projection head for teacher features (512D -> 64D)
    teacher_proj_dim = conf.model.person_feature_dim
    trainer.teacher_projection = torch.nn.Linear(512, teacher_proj_dim, bias=False).to(trainer.device)
    # Initialize with random orthogonal projection
    with torch.no_grad():
        torch.manual_seed(42)
        weight = torch.randn(512, teacher_proj_dim, device=trainer.device)
        u, _, vh = torch.linalg.svd(weight, full_matrices=False)
        trainer.teacher_projection.weight.data = u[:, :teacher_proj_dim].T
    trainer.teacher_projection.requires_grad_(False)  # Freeze projection
    
    # CRITICAL: Create a new optimizer that ONLY optimizes person_feature
    # This avoids issues with frozen parameters in the original optimizer
    person_feature_param = trainer.model._person_feature
    trainer.model.optimizer = torch.optim.Adam(
        [{'params': person_feature_param, 'lr': args.person_feature_lr}],
        eps=1e-8,
        weight_decay=0,
    )
    print("[Frozen Geometry] Created person_feature-only optimizer")
    print(f"  person_feature: requires_grad={person_feature_param.requires_grad}, shape={person_feature_param.shape}, device={person_feature_param.device}")
    print(f"  Teacher projection: 512D -> {teacher_proj_dim}D (frozen)")
    
    # Quick gradient test
    trainer.model.zero_grad()
    dummy_loss = person_feature_param.sum() * 0.0001
    dummy_loss.backward()
    has_grad = person_feature_param.grad is not None
    grad_norm = person_feature_param.grad.abs().mean().item() if has_grad else 0
    trainer.model.zero_grad()  # Clear after test
    print(f"  Gradient test: has_grad={has_grad}, grad_norm={grad_norm:.6f}")
    if not has_grad or grad_norm == 0:
        print("[ERROR] person_feature cannot receive gradients! Aborting.")
        return False

    # Load prototypes and project to match person_feature_dim
    proto_data = torch.load(args.prototype_path, map_location=trainer.device, weights_only=False)
    prototypes_512 = proto_data['prototypes'].to(trainer.device)
    valid_mask = proto_data['valid_mask'].to(trainer.device)
    
    # Project prototypes from 512D to 64D using PCA-like random projection
    # This is needed because we're training with 64D features
    proj_dim = conf.model.person_feature_dim
    if prototypes_512.shape[1] != proj_dim:
        print(f"  Projecting prototypes from {prototypes_512.shape[1]}D to {proj_dim}D")
        # Use a simple learned projection: initialize random projection matrix
        torch.manual_seed(42)
        proj_matrix = torch.randn(prototypes_512.shape[1], proj_dim, device=trainer.device)
        proj_matrix = proj_matrix / torch.norm(proj_matrix, dim=0, keepdim=True)
        prototypes = prototypes_512 @ proj_matrix
        prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)
        print(f"  Prototypes projected to: {prototypes.shape}")
    else:
        prototypes = prototypes_512
    
    print(f"\nLoaded prototypes: {prototypes.shape}, valid IDs: {valid_mask.sum().item()}")

    # Create sampler and batch builder
    sampler = PkMultiViewSampler(
        trainer.train_dataset,
        num_person=args.mv_num_person,
        num_views=args.mv_num_views,
    )
    batch_builder = FrozenMVBatchBuilder(trainer, sampler)

    # Debug mode
    if args.debug_sampler:
        print("\n" + "=" * 70)
        print("DEBUG SAMPLER MODE (200 samples)")
        print("=" * 70)

        success_count = 0
        person_counts = []
        view_counts = []
        unique_renders_list = []
        positive_counts = []
        negative_counts = []
        valid_features_list = []
        cam_duplicate_count = 0
        diag_false_count = 0

        for i in range(200):
            result = sampler.sample_batch()
            if result is None:
                continue
            success_count += 1

            person_counts.append(len(result))
            obs, skipped, n_renders = batch_builder.process_mv_samples(result)
            unique_renders_list.append(n_renders)
            valid_features_list.append(len(obs))

            for ps in result:
                view_counts.append(len(ps['views']))
                cams = [v[0] for v in ps['views']]
                if len(cams) != len(set(cams)):
                    cam_duplicate_count += 1

            # Build SupCon mask analysis
            p_ids = [o['person_id'] for o in obs]
            if len(p_ids) >= 2:
                labels = torch.tensor(p_ids)
                labels_expanded = labels.unsqueeze(1)
                positive_mask = (labels_expanded == labels_expanded.T).float()
                positive_mask.fill_diagonal_(0)

                diag_val = torch.diag(positive_mask).sum().item()
                if diag_val == 0:
                    diag_false_count += 1

                n_pos = positive_mask.sum().item() / 2
                n_neg = (1 - positive_mask - torch.eye(len(labels))).sum().item() / 2
                positive_counts.append(int(n_pos))
                negative_counts.append(int(n_neg))

            if i < 5 or i % 50 == 0:
                print(f"  Sample {i}: persons={len(result)}, valid_feats={len(obs)}, "
                      f"unique_renders={n_renders}, skip={skipped}")

        print(f"\n--- Debug Summary ---")
        print(f"  Success rate: {success_count}/200 = {success_count/200*100:.1f}%")
        print(f"  Mean persons: {np.mean(person_counts):.1f}" if person_counts else "  N/A")
        print(f"  Mean views/person: {np.mean(view_counts):.1f}" if view_counts else "  N/A")
        print(f"  Mean unique renders: {np.mean(unique_renders_list):.1f}" if unique_renders_list else "  N/A")
        print(f"  Mean valid features: {np.mean(valid_features_list):.1f}" if valid_features_list else "  N/A")
        print(f"  Mean positive pairs: {np.mean(positive_counts):.1f}" if positive_counts else "  N/A")
        print(f"  Mean negative pairs: {np.mean(negative_counts):.1f}" if negative_counts else "  N/A")
        print(f"  Cam duplicates: {cam_duplicate_count}")
        print(f"  Diag all False: {diag_false_count}/{success_count}")

        # Pass/fail
        checks = {
            'success_rate_ge_80': success_count / 200 >= 0.8,
            'mean_persons_ge_4': np.mean(person_counts) >= 4 if person_counts else False,
            'mean_views_ge_2': np.mean(view_counts) >= 2 if view_counts else False,
            'positive_pairs_gt_0': np.mean(positive_counts) > 0 if positive_counts else False,
            'negative_pairs_gt_0': np.mean(negative_counts) > 0 if negative_counts else False,
            'diag_all_false': diag_false_count == success_count,
            'mean_valid_feats_ge_6': np.mean(valid_features_list) >= 6 if valid_features_list else False,
        }
        print(f"\n--- Checks ---")
        for name, passed in checks.items():
            print(f"  {name:30s}: {'PASS' if passed else 'FAIL'}")

        return True

    # Training loop
    total_iters = args.warmup_iters + args.train_iters
    warmup_log = {'L_rgb': [], 'feat_nz': []}
    train_log = {
        'L_rgb': [], 'L_proto': [], 'L_mv': [], 'L_teacher': [], 'L_reid_total': [],
        'cos_fv_Pi': [], 'cos_fv_tv': [],
        'cross_view_same_cos': [], 'cross_view_diff_cos': [], 'cross_view_gap': [],
        'valid_inst': [], 'skipped_low_alpha': [], 'invalid_mv_batch': [],
        'num_unique_renders': [], 'num_observations': [], 'num_valid_features': [],
        'positive_pair_count': [], 'negative_pair_count': [],
        'grad_mean': [], 'grad_max': [], 'grad_nz': [],
        'nan_inf': [], 'iter_time': [],
    }

    trainer.model.train()
    t_start = time.time()
    any_nan_inf = False

    for step in range(total_iters):
        t_iter_start = time.time()

        # Warmup: use standard dataloader
        if step < args.warmup_iters:
            train_iter = iter(trainer.train_dataloader)
            try:
                batch_data = next(train_iter)
            except StopIteration:
                train_iter = iter(trainer.train_dataloader)
                batch_data = next(train_iter)
            gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)
            trainer.model.zero_grad()
            render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
            pred_rgb = render_out['pred_rgb']
            L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)
            L_rgb.backward()
            trainer.model.optimizer.step()
            warmup_log['L_rgb'].append(L_rgb.item())
            if step < 3 or step % 200 == 0:
                print(f"[WARMUP] Step {step}: L_rgb={L_rgb.item():.4f}")
            continue

        # Sample multi-view batch
        mv_samples = sampler.sample_batch()
        if mv_samples is None:
            continue

        observations, skipped_low_alpha, unique_renders = batch_builder.process_mv_samples(mv_samples)
        num_obs = len(observations)
        num_valid = num_obs

        if num_valid < 6:
            train_log['invalid_mv_batch'].append(1)
            train_log['skipped_low_alpha'].append(skipped_low_alpha)
            train_log['num_observations'].append(num_obs)
            train_log['num_valid_features'].append(num_valid)
            train_log['num_unique_renders'].append(unique_renders)
            train_log['positive_pair_count'].append(0)
            train_log['negative_pair_count'].append(0)
            for key in ['L_rgb', 'L_proto', 'L_mv', 'L_teacher', 'L_reid_total',
                        'cos_fv_Pi', 'cos_fv_tv', 'cross_view_same_cos', 'cross_view_diff_cos',
                        'cross_view_gap', 'valid_inst', 'grad_mean', 'grad_max', 'grad_nz',
                        'nan_inf', 'iter_time']:
                train_log[key].append(0)
            continue

        # Collect features and labels
        f_list = []
        p_ids = []
        t_emb_list = []
        for obs in observations:
            f_list.append(obs['feature'])
            p_ids.append(obs['person_id'])
            if obs['teacher_feature'] is not None:
                t_emb_list.append((obs['feature'], obs['teacher_feature']))

        f_stack = torch.stack(f_list)
        p_ids_tensor = torch.tensor(p_ids, device=trainer.device)

        # L_proto: CE loss
        logits = f_stack @ prototypes.T / args.tau
        L_proto = F.cross_entropy(logits, p_ids_tensor)

        # L_mv: supervised contrastive loss
        L_mv = compute_supcon_loss(f_stack, p_ids_tensor, temperature=args.mv_tau)

        # L_teacher: cosine loss (project teacher features from 512D to feature_dim)
        if t_emb_list:
            teacher_losses = []
            for f_v, t_v in t_emb_list:
                # Project teacher feature from 512D to feature_dim
                t_v_64 = trainer.teacher_projection(torch.as_tensor(t_v, dtype=torch.float32, device=trainer.device).squeeze())
                t_v_norm = F.normalize(t_v_64, p=2, dim=0, eps=1e-6)
                cos_t = torch.dot(f_v, t_v_norm)
                teacher_losses.append(1 - cos_t)
            L_teacher = torch.stack(teacher_losses).mean()
        else:
            L_teacher = torch.zeros(1, device=trainer.device)

        # Total ReID loss
        L_reid = 1.0 * L_proto + args.mv_loss_weight * L_mv + args.teacher_loss_weight * L_teacher

        # Backward (only person_feature gets gradients due to frozen params)
        trainer.model.zero_grad()
        step_nan = torch.isnan(L_reid)
        if not step_nan:
            (args.lambda_reid * L_reid).backward()
            
            # Check gradient BEFORE step
            pf = trainer.model.get_person_feature()
            if pf.grad is None:
                grad_mean = grad_max = grad_nz = 0.0
                if step < 5 or step % 100 == 0:
                    print(f"[V3] Step {step}: WARNING - No gradient on person_feature!")
            else:
                # Gradient clipping to prevent explosion
                grad_norm_before = pf.grad.norm().item()
                torch.nn.utils.clip_grad_norm_(pf, max_norm=1.0)
                grad_norm_after = pf.grad.norm().item()
                
                grad_mean = pf.grad.abs().mean().item()
                grad_max = pf.grad.abs().max().item()
                grad_nz = (pf.grad.abs() > 1e-12).sum().item() / pf.grad.numel() * 100
                
                if step < 5 or step % 100 == 0:
                    print(f"[V3] Step {step}: grad_mean={grad_mean:.6f}, grad_max={grad_max:.6f}, "
                          f"grad_nz={grad_nz:.4f}%, clip={grad_norm_before:.2f}->{grad_norm_after:.2f}")
            
            trainer.model.optimizer.step()
        else:
            any_nan_inf = True
            grad_mean = grad_max = grad_nz = 0.0

        # Cosine metrics
        cos_Pi_list = []
        cos_tv_list = []
        for obs in observations:
            pid = obs['person_id']
            if valid_mask[pid]:
                P_i = prototypes[pid].squeeze()
                cos_Pi_list.append(torch.dot(obs['feature'], P_i).item())
            if obs['teacher_feature'] is not None:
                t_v = torch.as_tensor(obs['teacher_feature'], dtype=torch.float32, device=trainer.device).squeeze()
                t_v_proj = trainer.teacher_projection(t_v)
                t_v_norm = F.normalize(t_v_proj, p=2, dim=0, eps=1e-6)
                cos_tv_list.append(torch.dot(obs['feature'], t_v_norm).item())

        # Cross-view metrics
        same_cos, diff_cos = [], []
        for i in range(len(f_list)):
            for j in range(i + 1, len(f_list)):
                cos_ij = torch.dot(f_list[i], f_list[j]).item()
                if p_ids[i] == p_ids[j]:
                    same_cos.append(cos_ij)
                else:
                    diff_cos.append(cos_ij)

        cross_same = np.mean(same_cos) if same_cos else 0
        cross_diff = np.mean(diff_cos) if diff_cos else 0
        cross_gap = cross_same - cross_diff

        # SupCon mask stats
        labels_expanded = p_ids_tensor.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        positive_mask.fill_diagonal_(0)
        n_pos = int(positive_mask.sum().item() / 2)
        n_neg = int((1 - positive_mask - torch.eye(num_valid, device=trainer.device)).sum().item() / 2)

        iter_time = time.time() - t_iter_start

        train_log['L_rgb'].append(0)
        train_log['L_proto'].append(L_proto.item())
        train_log['L_mv'].append(L_mv.item())
        train_log['L_teacher'].append(L_teacher.item())
        train_log['L_reid_total'].append(L_reid.item())
        train_log['cos_fv_Pi'].append(np.mean(cos_Pi_list) if cos_Pi_list else 0)
        train_log['cos_fv_tv'].append(np.mean(cos_tv_list) if cos_tv_list else 0)
        train_log['cross_view_same_cos'].append(cross_same)
        train_log['cross_view_diff_cos'].append(cross_diff)
        train_log['cross_view_gap'].append(cross_gap)
        train_log['valid_inst'].append(num_valid)
        train_log['skipped_low_alpha'].append(skipped_low_alpha)
        train_log['invalid_mv_batch'].append(0)
        train_log['num_unique_renders'].append(unique_renders)
        train_log['num_observations'].append(num_obs)
        train_log['num_valid_features'].append(num_valid)
        train_log['positive_pair_count'].append(n_pos)
        train_log['negative_pair_count'].append(n_neg)
        train_log['grad_mean'].append(grad_mean)
        train_log['grad_max'].append(grad_max)
        train_log['grad_nz'].append(grad_nz)
        train_log['nan_inf'].append(bool(step_nan))
        train_log['iter_time'].append(iter_time)

        should_print = (step < 3 or step % 100 == 0
                        or step == total_iters - 1)
        if should_print:
            print(f"[V3] Step {step:5d}: L_proto={L_proto.item():.4f}, L_mv={L_mv.item():.4f}, "
                  f"L_teacher={L_teacher.item():.4f}, L_reid={L_reid.item():.4f}, "
                  f"cos_Pi={np.mean(cos_Pi_list) if cos_Pi_list else 0:.4f}, "
                  f"cos_tv={np.mean(cos_tv_list) if cos_tv_list else 0:.4f}, "
                  f"cross_same={cross_same:.4f}, cross_diff={cross_diff:.4f}, "
                  f"cross_gap={cross_gap:.6f}, valid={num_valid}, obs={num_obs}, "
                  f"pos={n_pos}, neg={n_neg}, renders={unique_renders}, "
                  f"skip={skipped_low_alpha}, grad_nz={grad_nz:.4f}%, t={iter_time:.2f}s")
            sys.stdout.flush()

        reid_step = step - args.warmup_iters
        if reid_step > 0 and reid_step % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"step_{step}.pth")
            save_checkpoint(trainer.model, trainer.model.optimizer, step, ckpt_path)
            sys.stdout.flush()

    final_path = os.path.join(args.ckpt_dir, "latest.pth")
    save_checkpoint(trainer.model, trainer.model.optimizer, total_iters - 1, final_path)
    sys.stdout.flush()

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Results summary
    r = train_log
    if r['L_proto'] and any(v > 0 for v in r['L_proto']):
        nz_proto = [i for i, v in enumerate(r['L_proto']) if v > 0]
        if nz_proto:
            n = len(nz_proto)
            w = min(50, max(10, n // 10))
            print(f"\n--- ReID Training ({n} valid steps) ---")
            for k in ['L_proto', 'L_mv', 'L_teacher', 'L_reid_total']:
                vals = [r[k][i] for i in nz_proto]
                print(f"  {k:20s}: first{w}={np.mean(vals[:w]):.4f}  last{w}={np.mean(vals[-w:]):.4f}")
            for k in ['cos_fv_Pi', 'cos_fv_tv', 'cross_view_same_cos', 'cross_view_diff_cos', 'cross_view_gap']:
                vals = [r[k][i] for i in nz_proto]
                print(f"  {k:20s}: first{w}={np.mean(vals[:w]):.4f}  last{w}={np.mean(vals[-w:]):.6f}")
            print(f"  valid_inst:          mean={np.mean([r['valid_inst'][i] for i in nz_proto]):.1f}")
            print(f"  NaN/Inf:             {sum(r['nan_inf'])}/{len(r['nan_inf'])}")

    log_path = os.path.join(REPO_ROOT, args.log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump({
            'warmup': warmup_log,
            'train': train_log,
            'config': vars(args),
            'total_time_s': total_time,
        }, f, indent=2)
    print(f"\nLog saved: {log_path}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
