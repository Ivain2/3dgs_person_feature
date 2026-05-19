#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 6B: Evaluate identity discrimination of rendered person features.

Load checkpoint, render person_feature_map for multiple views,
ROI pool to get f_v, then evaluate:
  - cos(f_v, t_v) vs teacher
  - same-id cosine vs diff-id cosine
  - retrieval top1/top5 accuracy

Usage:
    python tools/phase6_eval_identity_features.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --checkpoint runs/phase6_reid_main/latest.pth \
        --num_batches 50
"""

import argparse
import os
import sys
import json

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.roi_pooling import roi_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_batches', type=int, default=50)
    parser.add_argument('--output', type=str, default='tools/phase6_eval_log.json')
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 6B: Identity Feature Evaluation")
    print("=" * 70)

    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)

    conf.model.person_feature_dim = 512
    conf.loss.use_reid = True

    print("\nInitializing trainer...")
    trainer = Trainer3DGRUT(conf)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded from step {ckpt.get('step', 'unknown')}")

    trainer.model.eval()

    # Collect features
    all_f_v = []
    all_t_v = []
    all_train_ids = []
    all_camera_ids = []
    all_frame_ids = []

    print(f"\nCollecting features from {args.num_batches} batches...")
    train_iter = iter(trainer.train_dataloader)

    with torch.no_grad():
        for i in range(args.num_batches):
            try:
                batch_data = next(train_iter)
            except StopIteration:
                train_iter = iter(trainer.train_dataloader)
                batch_data = next(train_iter)

            gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)

            render_out = trainer.model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
            person_feature_map = render_out['person_feature_map']

            for inst in gpu_batch.instances:
                if not inst.get('valid', False):
                    continue
                teacher_emb = inst.get('teacher_embedding')
                if teacher_emb is None:
                    continue

                bbox = inst['bbox_xyxy']
                f_v = roi_pool(person_feature_map, torch.tensor(bbox, dtype=torch.float32, device=trainer.device))
                f_v_norm = F.normalize(f_v, p=2, dim=0, eps=1e-6)
                t_v = torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                t_v_norm = F.normalize(t_v, p=2, dim=0, eps=1e-6)

                all_f_v.append(f_v_norm.cpu().numpy())
                all_t_v.append(t_v_norm.cpu().numpy())
                all_train_ids.append(inst.get('train_id', -1))
                all_camera_ids.append(inst.get('camera_id', -1))
                all_frame_ids.append(inst.get('frame_id', -1))

            if (i + 1) % 10 == 0:
                print(f"  Collected {len(all_f_v)} instances from {i+1} batches")

    if len(all_f_v) == 0:
        print("❌ No valid instances collected!")
        return False

    f_v_matrix = np.array(all_f_v)  # [N, 512]
    t_v_matrix = np.array(all_t_v)  # [N, 512]
    train_ids = np.array(all_train_ids)
    camera_ids = np.array(all_camera_ids)
    frame_ids = np.array(all_frame_ids)

    N = len(f_v_matrix)
    print(f"\nTotal instances collected: {N}")
    print(f"Unique train_ids: {len(np.unique(train_ids))}")

    # ===== Evaluation =====
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # 1. cos(f_v, t_v)
    cos_teacher = np.sum(f_v_matrix * t_v_matrix, axis=1)
    print(f"\n1. cos(f_v, t_v):")
    print(f"   mean={np.mean(cos_teacher):.4f}, std={np.std(cos_teacher):.4f}")
    print(f"   min={np.min(cos_teacher):.4f}, max={np.max(cos_teacher):.4f}")

    # 2. Same-id vs diff-id cosine
    print(f"\n2. Same-id vs Diff-id cosine (f_v-f_v):")
    cos_matrix = f_v_matrix @ f_v_matrix.T  # [N, N]

    same_id_cos = []
    diff_id_cos = []

    for i in range(N):
        for j in range(i + 1, N):
            if train_ids[i] == train_ids[j] and train_ids[i] >= 0:
                same_id_cos.append(cos_matrix[i, j])
            elif train_ids[i] != train_ids[j] and train_ids[i] >= 0 and train_ids[j] >= 0:
                diff_id_cos.append(cos_matrix[i, j])

    if same_id_cos:
        same_mean = np.mean(same_id_cos)
        same_std = np.std(same_id_cos)
        print(f"   same-id:  mean={same_mean:.4f}, std={same_std:.4f}, count={len(same_id_cos)}")
    else:
        same_mean = 0
        same_std = 0
        print(f"   same-id:  no pairs found")

    if diff_id_cos:
        diff_mean = np.mean(diff_id_cos)
        diff_std = np.std(diff_id_cos)
        print(f"   diff-id:  mean={diff_mean:.4f}, std={diff_std:.4f}, count={len(diff_id_cos)}")
    else:
        diff_mean = 0
        diff_std = 0
        print(f"   diff-id:  no pairs found")

    gap = same_mean - diff_mean
    print(f"   gap (same - diff): {gap:.4f}")
    print(f"   {'✅ same > diff' if gap > 0 else '❌ same <= diff'}")

    # 3. Retrieval evaluation
    print(f"\n3. Retrieval (gallery = all other instances):")
    top1_correct = 0
    top5_correct = 0
    valid_queries = 0

    for i in range(N):
        if train_ids[i] < 0:
            continue

        query_id = train_ids[i]
        scores = cos_matrix[i].copy()
        scores[i] = -999  # exclude self

        top_indices = np.argsort(scores)[::-1][:5]
        top_ids = train_ids[top_indices]

        if top_ids[0] == query_id:
            top1_correct += 1
        if query_id in top_ids:
            top5_correct += 1
        valid_queries += 1

    if valid_queries > 0:
        top1_acc = top1_correct / valid_queries
        top5_acc = top5_correct / valid_queries
        random_top1 = 1.0 / len(np.unique(train_ids))
        random_top5 = min(5.0 / len(np.unique(train_ids)), 1.0)

        print(f"   top1_acc = {top1_acc:.4f} (random={random_top1:.4f})")
        print(f"   top5_acc = {top5_acc:.4f} (random={random_top5:.4f})")
        print(f"   {'✅ top1 > random' if top1_acc > random_top1 else '❌ top1 <= random'}")
        print(f"   {'✅ top5 > random' if top5_acc > random_top5 else '❌ top5 <= random'}")
    else:
        top1_acc = 0
        top5_acc = 0
        random_top1 = 0
        random_top5 = 0

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  cos(f_v, t_v): {np.mean(cos_teacher):.4f}")
    print(f"  same-id cosine: {same_mean:.4f}")
    print(f"  diff-id cosine: {diff_mean:.4f}")
    print(f"  gap: {gap:.4f}")
    print(f"  top1_acc: {top1_acc:.4f} (random={random_top1:.4f})")
    print(f"  top5_acc: {top5_acc:.4f} (random={random_top5:.4f})")

    # Checklist
    print(f"\n  CHECKLIST:")
    checks = {}
    checks['cos_teacher_positive'] = np.mean(cos_teacher) > 0
    checks['same_gt_diff'] = gap > 0
    checks['top1_gt_random'] = top1_acc > random_top1
    checks['top5_gt_random'] = top5_acc > random_top5
    checks['no_nan'] = not (np.isnan(f_v_matrix).any() or np.isnan(cos_teacher).any())

    for k, v in checks.items():
        print(f"    {'✅' if v else '❌'} {k}: {v}")

    all_pass = all(checks.values())
    print(f"\n  {'🎉 EVALUATION PASSED' if all_pass else '⚠️  EVALUATION FAILED'}")

    # Save log
    log_path = os.path.join(REPO_ROOT, args.output)
    save_data = {
        'num_instances': int(N),
        'num_unique_ids': int(len(np.unique(train_ids))),
        'cos_teacher_mean': float(np.mean(cos_teacher)),
        'cos_teacher_std': float(np.std(cos_teacher)),
        'same_id_cosine_mean': float(same_mean),
        'same_id_cosine_std': float(same_std),
        'same_id_count': int(len(same_id_cos)),
        'diff_id_cosine_mean': float(diff_mean),
        'diff_id_cosine_std': float(diff_std),
        'diff_id_count': int(len(diff_id_cos)),
        'gap': float(gap),
        'top1_acc': float(top1_acc),
        'top5_acc': float(top5_acc),
        'random_top1': float(random_top1),
        'random_top5': float(random_top5),
        'checks': {k: bool(v) for k, v in checks.items()},
    }
    with open(log_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Log saved: {log_path}")

    return all_pass


if __name__ == "__main__":
    main()
