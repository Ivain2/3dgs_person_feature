#!/usr/bin/env python3
"""
Debug script for Phase 11B MultiViewSampler and SupCon loss.

Tests:
  1. Sampler quality: person count, view count, camera diversity, timestamp consistency
  2. SupCon mask: positive/negative pairs, diagonal correctness
  3. Feature statistics: norms, diversity
"""

import sys
import os
import random
from collections import defaultdict, Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
import numpy as np

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.roi_pooling import roi_pool


class MultiViewSampler:
    """Copy of the sampler from train_phase11B_mv_supcon.py"""

    def __init__(self, dataset, num_person=4, num_views=2, max_retries=5):
        self.dataset = dataset
        self.num_person = num_person
        self.num_views = num_views
        self.max_retries = max_retries
        self.camera_ids = dataset.camera_ids
        self.annotations = dataset.annotations
        self.timestamp_to_persons = defaultdict(set)
        self.person_to_frames = defaultdict(list)
        self.all_timestamps = set()
        for frame_id, annots in self.annotations.items():
            if not isinstance(annots, list):
                continue
            person_ids = [p.get('train_id') or p.get('new_id') for p in annots
                          if p.get('train_id') is not None or p.get('new_id') is not None]
            if person_ids:
                self.timestamp_to_persons[frame_id] = set(person_ids)
                self.all_timestamps.add(frame_id)
                for pid in person_ids:
                    for cam_id in self.camera_ids:
                        self.person_to_frames[pid].append((cam_id, frame_id))
        self.all_timestamps = sorted(self.all_timestamps)
        self.person_ids = sorted(self.person_to_frames.keys())
        self.person_to_cam_frames = defaultdict(lambda: defaultdict(list))
        for pid in self.person_to_frames:
            for cam_id, frame_id in self.person_to_frames[pid]:
                self.person_to_cam_frames[pid][cam_id].append(frame_id)

    def sample_batch(self):
        for _ in range(self.max_retries):
            timestamp = random.choice(self.all_timestamps)
            available_persons = list(self.timestamp_to_persons.get(timestamp, set()))
            if len(available_persons) < self.num_person:
                continue
            selected_persons = random.sample(available_persons, self.num_person)
            mv_samples = []
            for pid in selected_persons:
                valid_views = []
                cam_frames = self.person_to_cam_frames.get(pid, {})
                for cam_id, frames in cam_frames.items():
                    if timestamp in frames:
                        valid_views.append(cam_id)
                if len(valid_views) < self.num_views:
                    valid_views = []
                    for cam_id in self.camera_ids:
                        if cam_id in cam_frames:
                            valid_views.append(cam_id)
                    if len(valid_views) < self.num_views:
                        continue
                chosen_views = random.sample(valid_views, self.num_views)
                mv_samples.append({
                    'person_id': pid,
                    'views': [(cam_id, timestamp) for cam_id in chosen_views],
                })
            if len(mv_samples) >= 2:
                return mv_samples
        timestamp = random.choice(self.all_timestamps)
        available_persons = list(self.timestamp_to_persons.get(timestamp, set()))
        if len(available_persons) < 2:
            return None
        selected_persons = random.sample(available_persons, min(self.num_person, len(available_persons)))
        mv_samples = []
        for pid in selected_persons:
            cam_frames = self.person_to_cam_frames.get(pid, {})
            valid_views = list(cam_frames.keys())
            if len(valid_views) < 2:
                continue
            chosen_views = random.sample(valid_views, min(self.num_views, len(valid_views)))
            chosen_ts = []
            for cam_id in chosen_views:
                ts_list = cam_frames[cam_id]
                chosen_ts.append((cam_id, random.choice(ts_list)))
            mv_samples.append({
                'person_id': pid,
                'views': chosen_ts,
            })
        return mv_samples if len(mv_samples) >= 2 else None


def compute_supcon_loss(features, labels, temperature=0.07):
    device = features.device
    N = features.shape[0]
    sim_matrix = features @ features.T / temperature
    labels_expanded = labels.unsqueeze(1)
    positive_mask = (labels_expanded == labels_expanded.T).float()
    positive_mask.fill_diagonal_(0)
    num_positives = positive_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
    exp_sim = torch.exp(sim_matrix)
    exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)
    pos_exp = exp_sim * positive_mask
    log_prob = pos_exp.sum(dim=1, keepdim=True) / num_positives
    log_prob = log_prob - torch.log(exp_sim_sum)
    log_prob = log_prob.mean()
    return -log_prob


def main():
    print("=" * 80)
    print("Phase 11B MultiViewSampler + SupCon Debug")
    print("=" * 80)

    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = "apps/wildtrack_full_3dgut"
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)
    conf.model.person_feature_dim = 512
    conf.model.person_feature_lr = 1e-4

    print("\nInitializing trainer (this may take a minute)...")
    trainer = Trainer3DGRUT(conf)
    print(f"Device: {trainer.device}")
    print(f"Gaussians: {trainer.model.num_gaussians}")

    proto_data = torch.load('/data02/zhangrunxiang/data/Wildtrack/reid_teacher_prototypes.pt',
                            map_location=trainer.device, weights_only=False)
    prototypes = proto_data['prototypes'].to(trainer.device)
    valid_mask = proto_data['valid_mask'].to(trainer.device)

    sampler = MultiViewSampler(trainer.train_dataset, num_person=4, num_views=2, max_retries=5)
    print(f"\nSampler: {len(sampler.all_timestamps)} timestamps, {len(sampler.person_ids)} persons")

    # Test sampler quality
    print("\n" + "=" * 80)
    print("SAMPLER QUALITY TEST (200 samples)")
    print("=" * 80)

    success_count = 0
    person_counts = []
    view_counts_per_person = []
    all_cam_ids_used = []
    all_timestamps_used = []
    all_pids_used = []
    same_cam_positive = 0
    total_positive = 0
    duplicate_views = 0
    teacher_norms = []

    train_iter = iter(trainer.train_dataloader)
    def get_next_batch():
        nonlocal train_iter
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(trainer.train_dataloader)
            batch_data = next(train_iter)
        return trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)

    sample_batches = []

    for i in range(200):
        result = sampler.sample_batch()
        if result is None or len(result) < 2:
            continue
        success_count += 1

        person_counts.append(len(result))
        person_features_this_batch = []
        person_ids_this_batch = []
        cam_ids_this_batch = []
        timestamps_this_batch = []

        for ps in result:
            pid = ps['person_id']
            all_pids_used.append(pid)
            views = ps['views']
            view_counts_per_person.append(len(views))

            cam_ids = [v[0] for v in views]
            ts_ids = [v[1] for v in views]
            all_cam_ids_used.extend(cam_ids)
            all_timestamps_used.extend(ts_ids)
            cam_ids_this_batch.extend(cam_ids)
            timestamps_this_batch.extend(ts_ids)

            # Check for duplicate (cam_id, timestamp) pairs
            if len(views) != len(set(views)):
                duplicate_views += 1

            # Check for same camera in positive pair
            if len(cam_ids) >= 2:
                if cam_ids[0] == cam_ids[1]:
                    same_cam_positive += 1
                total_positive += 1

        if len(sample_batches) < 3:
            sample_batches.append({
                'batch_idx': i,
                'persons': result,
                'cam_ids': cam_ids_this_batch,
                'timestamps': timestamps_this_batch,
            })

    print(f"Success rate: {success_count}/200 = {success_count/200*100:.1f}%")
    print(f"Mean persons per batch: {np.mean(person_counts):.1f}" if person_counts else "N/A")
    print(f"Mean views per person: {np.mean(view_counts_per_person):.1f}" if view_counts_per_person else "N/A")
    print(f"Same-cam positive pairs: {same_cam_positive}/{total_positive}" if total_positive > 0 else "N/A")
    print(f"Duplicate view pairs: {duplicate_views}")
    print(f"Unique timestamps used: {len(set(all_timestamps_used))}")
    print(f"Unique persons used: {len(set(all_pids_used))}")
    print(f"Unique cameras used: {len(set(all_cam_ids_used))}")
    print(f"Camera distribution: {dict(Counter(all_cam_ids_used))}" if all_cam_ids_used else "N/A")

    # Check SupCon mask quality for 3 sample batches
    print("\n" + "=" * 80)
    print("SUPCON MASK AND FEATURE TEST (3 sample batches)")
    print("=" * 80)

    for sb_idx, sb in enumerate(sample_batches):
        print(f"\n--- Sample Batch {sb_idx + 1} (sampler iteration {sb['batch_idx']}) ---")
        print(f"  Persons: {len(sb['persons'])}")
        for ps_idx, ps in enumerate(sb['persons']):
            print(f"  Person {ps_idx}: pid={ps['person_id']}, views={ps['views']}")

        # Render and pool
        all_features = []
        all_pids = []
        for ps in sb['persons']:
            pid = ps['person_id']
            for cam_id, timestamp in ps['views']:
                frame_idx = trainer.train_dataset.camera_ids.index(cam_id) if cam_id in trainer.train_dataset.camera_ids else 0
                gpu_batch = get_next_batch()
                with torch.no_grad():
                    render_out = trainer.model(gpu_batch, train=False, frame_id=frame_idx, render_person_feature=True)
                    person_feature_map = render_out['person_feature_map']
                    person_opacity_map = render_out.get('person_opacity_map')

                inst = None
                for inst_data in gpu_batch.instances:
                    if inst_data.get('train_id') == pid and inst_data.get('valid', False):
                        inst = inst_data
                        break
                if inst is None:
                    continue

                bbox = inst['bbox_xyxy']
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)

                f_v, pool_stats = roi_pool(
                    person_feature_map, bbox_t,
                    opacity_map=person_opacity_map,
                    pooling='opacity',
                    min_alpha_sum=0.01,
                    detach_opacity_weight=True,
                )
                if f_v is None:
                    continue

                all_features.append(f_v)
                all_pids.append(pid)
                t_emb = inst.get('teacher_embedding')
                if t_emb is not None:
                    t_v = torch.as_tensor(t_emb, dtype=torch.float32, device=trainer.device).squeeze()
                    teacher_norms.append(torch.norm(t_v).item())

        if len(all_features) < 2:
            print(f"  Not enough features ({len(all_features)}) after pooling")
            continue

        f_stack = torch.stack(all_features)
        p_ids = torch.tensor(all_pids, device=trainer.device)

        print(f"  Features collected: {len(all_features)}")
        print(f"  Person IDs: {all_pids}")
        print(f"  Feature norms: {[f'{f.norm().item():.4f}' for f in all_features]}")
        print(f"  Feature pairwise cosine:")
        N = len(all_features)
        for ii in range(N):
            row = []
            for jj in range(N):
                cos = torch.dot(all_features[ii], all_features[jj]).item()
                row.append(f"{cos:.3f}")
            print(f"    [{', '.join(row)}]")

        # SupCon mask
        labels_expanded = p_ids.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        positive_mask.fill_diagonal_(0)

        print(f"  SupCon positive_mask:")
        pm_np = positive_mask.cpu().numpy()
        for ii in range(N):
            row = [int(x) for x in pm_np[ii]]
            print(f"    {row}")

        diag_val = torch.diag(positive_mask)
        print(f"  positive_mask diagonal: {diag_val.tolist()} (must be all 0)")

        num_pos = positive_mask.sum(dim=1).tolist()
        num_neg = (1 - positive_mask - torch.eye(N, device=trainer.device)).sum(dim=1).tolist()
        print(f"  Positives per anchor: {num_pos}")
        print(f"  Negatives per anchor: {num_neg}")

        # Compute SupCon loss
        supcon = compute_supcon_loss(f_stack, p_ids, temperature=0.2)
        print(f"  SupCon loss (tau=0.2): {supcon.item():.4f}")

    # Teacher feature norms
    if teacher_norms:
        print(f"\n--- Teacher Feature Norms ({len(teacher_norms)} samples) ---")
        print(f"  Mean: {np.mean(teacher_norms):.4f}")
        print(f"  Std:  {np.std(teacher_norms):.4f}")
        print(f"  Min:  {np.min(teacher_norms):.4f}")
        print(f"  Max:  {np.max(teacher_norms):.4f}")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    checks = {
        'success_rate_ge_80': success_count / 200 >= 0.8,
        'mean_persons_ge_4': np.mean(person_counts) >= 4 if person_counts else False,
        'mean_views_ge_2': np.mean(view_counts_per_person) >= 2 if view_counts_per_person else False,
        'no_same_cam_positive': same_cam_positive == 0,
        'no_duplicate_views': duplicate_views == 0,
    }
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s}: {status}")


if __name__ == "__main__":
    main()
