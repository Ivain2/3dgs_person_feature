#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 7: Teacher vs Student same-protocol evaluation.

Evaluate both teacher t_v and student f_v on the SAME instance set,
then compare retrieval (top1/top5/mAP) and same/diff cosine.

Usage:
    # Dry run
    python tools/phase7_teacher_vs_student_eval.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --checkpoint runs/phase6_reid_main/latest.pth \
        --num_batches 5 \
        --output tools/phase7_teacher_vs_student_dryrun.json

    # Full eval
    python tools/phase7_teacher_vs_student_eval.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --checkpoint runs/phase6_reid_main/latest.pth \
        --num_batches 50 \
        --output tools/phase7_teacher_vs_student_eval.json \
        --csv_output tools/phase7_teacher_vs_student_embeddings.csv
"""

import argparse
import os
import sys
import json
import csv
import traceback
import time

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.roi_pooling import roi_pool


def compute_retrieval(features, labels):
    """
    All-vs-all retrieval, exclude self-match.
    Positive = same train_id, negative = different train_id.
    Skip queries with no same-id positive.
    mAP uses full ranking (not just top-5).
    """
    cos_sim = features @ features.T
    N = len(features)
    top1_hits = 0
    top5_hits = 0
    ap_sum = 0.0
    skipped = 0

    for q in range(N):
        q_label = labels[q]
        if q_label < 0:
            skipped += 1
            continue

        positive_mask = labels == q_label
        positive_mask[q] = False
        n_pos = positive_mask.sum()
        if n_pos == 0:
            skipped += 1
            continue

        scores = cos_sim[q].copy()
        scores[q] = -999
        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = labels[sorted_indices]

        if sorted_labels[0] == q_label:
            top1_hits += 1
        if q_label in sorted_labels[:5]:
            top5_hits += 1

        n_found = 0
        precisions = []
        for rank_i, r_label in enumerate(sorted_labels):
            if r_label == q_label:
                n_found += 1
                precisions.append(n_found / (rank_i + 1))
        ap = np.mean(precisions) if precisions else 0.0
        ap_sum += ap

    n_valid = N - skipped
    if n_valid == 0:
        return {"top1": 0.0, "top5": 0.0, "mAP": 0.0, "skipped": skipped}

    return {
        "top1": top1_hits / n_valid,
        "top5": top5_hits / n_valid,
        "mAP": ap_sum / n_valid,
        "skipped": skipped,
    }


def compute_same_diff_cosine(features, labels):
    """Compute same-id and diff-id mean cosine."""
    cos_sim = features @ features.T
    N = len(features)
    same_cos = []
    diff_cos = []

    for i in range(N):
        for j in range(i + 1, N):
            if labels[i] < 0 or labels[j] < 0:
                continue
            if labels[i] == labels[j]:
                same_cos.append(cos_sim[i, j])
            else:
                diff_cos.append(cos_sim[i, j])

    return {
        "same_mean": float(np.mean(same_cos)) if same_cos else 0.0,
        "same_std": float(np.std(same_cos)) if same_cos else 0.0,
        "diff_mean": float(np.mean(diff_cos)) if diff_cos else 0.0,
        "diff_std": float(np.std(diff_cos)) if diff_cos else 0.0,
        "same_count": len(same_cos),
        "diff_count": len(diff_cos),
    }


def camera_str_to_int(cam_str):
    if isinstance(cam_str, str):
        digits = ''.join(c for c in cam_str if c.isdigit())
        return int(digits) if digits else 0
    return int(cam_str) if cam_str is not None else -1


def extract_batch_meta(batch_data):
    cam_raw = batch_data.get("camera_id", None)
    frame_raw = batch_data.get("frame_idx", None)

    if isinstance(cam_raw, (list, tuple)):
        cam_raw = cam_raw[0]
    if isinstance(frame_raw, (list, tuple)):
        frame_raw = frame_raw[0]
    if hasattr(frame_raw, 'item'):
        frame_raw = frame_raw.item()

    cam_int = camera_str_to_int(cam_raw)
    frame_int = int(frame_raw) if frame_raw is not None else -1
    cam_str = str(cam_raw) if cam_raw is not None else "unknown"

    return cam_int, frame_int, cam_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_batches', type=int, default=50)
    parser.add_argument('--output', type=str, default='tools/phase7_teacher_vs_student_eval.json')
    parser.add_argument('--csv_output', type=str, default=None)
    parser.add_argument('--max_instances', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_prototype', action='store_true', default=False,
                        help='Enable teacher prototype sanity check (optional)')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'opacity', 'topk_opacity'])
    parser.add_argument('--topk_ratio', type=float, default=0.3)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 7: Teacher vs Student Same-Protocol Evaluation")
    print("=" * 70)
    print(f"  config           = {args.config}")
    print(f"  checkpoint       = {args.checkpoint}")
    print(f"  num_batches      = {args.num_batches}")
    print(f"  enable_prototype = {args.enable_prototype}")
    print(f"  seed             = {args.seed}")
    print(f"  split            = train")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)

    conf.model.person_feature_dim = 512

    print("\nInitializing trainer...")
    trainer = Trainer3DGRUT(conf)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"  Loaded from step {ckpt.get('step', 'unknown')}")

    trainer.model.eval()

    all_f_v = []
    all_t_v = []
    all_train_ids = []
    all_raw_ids = []
    all_camera_ids = []
    all_camera_strs = []
    all_frame_ids = []
    all_bboxes = []

    print(f"\nCollecting features from {args.num_batches} batches (split=train)...")
    train_iter = iter(trainer.train_dataloader)
    skipped_instances = 0
    nan_count = 0
    t_start = time.time()

    output = None

    try:
        with torch.no_grad():
            for i in range(args.num_batches):
                try:
                    batch_data = next(train_iter)
                except StopIteration:
                    train_iter = iter(trainer.train_dataloader)
                    batch_data = next(train_iter)

                cam_int, frame_int, cam_str = extract_batch_meta(batch_data)

                gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)

                render_out = trainer.model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')

                batch_nan = False
                if torch.isnan(person_feature_map).any() or torch.isinf(person_feature_map).any():
                    nan_count += 1
                    batch_nan = True

                for inst in gpu_batch.instances:
                    if not inst.get('valid', False):
                        skipped_instances += 1
                        continue

                    teacher_emb = inst.get('teacher_embedding')
                    if teacher_emb is None:
                        skipped_instances += 1
                        continue

                    bbox = inst['bbox_xyxy']
                    try:
                        f_v, _ = roi_pool(person_feature_map,
                                          torch.tensor(bbox, dtype=torch.float32, device=trainer.device),
                                          opacity_map=person_opacity_map,
                                          pooling=args.pooling,
                                          topk_ratio=args.topk_ratio)
                        if f_v is None:
                            skipped_instances += 1
                            continue
                    except Exception:
                        skipped_instances += 1
                        continue

                    if torch.isnan(f_v).any() or torch.isinf(f_v).any():
                        nan_count += 1
                        skipped_instances += 1
                        continue

                    f_v_norm = F.normalize(f_v, p=2, dim=0, eps=1e-6)
                    t_v = torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                    t_v_norm = F.normalize(t_v, p=2, dim=0, eps=1e-6)

                    all_f_v.append(f_v_norm.cpu().numpy())
                    all_t_v.append(t_v_norm.cpu().numpy())
                    all_train_ids.append(inst.get('train_id', -1))
                    all_raw_ids.append(inst.get('raw_id', -1))
                    all_camera_ids.append(cam_int)
                    all_camera_strs.append(cam_str)
                    all_frame_ids.append(frame_int)
                    all_bboxes.append(list(bbox) if not isinstance(bbox, list) else bbox)

                    if args.max_instances and len(all_f_v) >= args.max_instances:
                        break

                if args.max_instances and len(all_f_v) >= args.max_instances:
                    break

                elapsed = time.time() - t_start
                print(f"  batch {i+1}/{args.num_batches}: collected {len(all_f_v)} instances, "
                      f"skipped {skipped_instances}, NaN={nan_count}, "
                      f"elapsed {elapsed:.1f}s")

        if len(all_f_v) == 0:
            print("ERROR: No valid instances collected!")
            output = {'error': 'no_valid_instances', 'num_instances': 0}
            output_path = os.path.join(REPO_ROOT, args.output)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            return False

        f_v_matrix = np.array(all_f_v)
        t_v_matrix = np.array(all_t_v)
        train_ids = np.array(all_train_ids, dtype=np.int64)
        raw_ids = np.array(all_raw_ids, dtype=np.int64)
        camera_ids = np.array(all_camera_ids, dtype=np.int64)
        frame_ids = np.array(all_frame_ids, dtype=np.int64)
        bboxes = np.array(all_bboxes, dtype=np.float32)

        N = len(f_v_matrix)
        n_ids = len(np.unique(train_ids[train_ids >= 0]))
        n_cameras = len(np.unique(camera_ids[camera_ids >= 0]))
        unique_cam_strs = sorted(set(all_camera_strs))

        print(f"\nTotal instances: {N}")
        print(f"Unique train_ids: {n_ids}")
        print(f"Unique cameras: {n_cameras} ({unique_cam_strs})")
        print(f"Skipped instances: {skipped_instances}")
        print(f"NaN/Inf count: {nan_count}")

        cos_student_teacher = np.sum(f_v_matrix * t_v_matrix, axis=1)
        mean_cos_ft = float(np.mean(cos_student_teacher))
        print(f"\nmean cos(f_v, t_v) = {mean_cos_ft:.4f}")

        print("\n--- Teacher evaluation ---")
        t_result = compute_retrieval(t_v_matrix, train_ids)
        t_sd = compute_same_diff_cosine(t_v_matrix, train_ids)
        t_gap = t_sd['same_mean'] - t_sd['diff_mean']
        print(f"  same_cos = {t_sd['same_mean']:.4f}")
        print(f"  diff_cos = {t_sd['diff_mean']:.4f}")
        print(f"  gap      = {t_gap:.4f}")
        print(f"  top1     = {t_result['top1']:.4f}")
        print(f"  top5     = {t_result['top5']:.4f}")
        print(f"  mAP      = {t_result['mAP']:.4f}")
        print(f"  skipped  = {t_result['skipped']}")

        print("\n--- Student evaluation ---")
        s_result = compute_retrieval(f_v_matrix, train_ids)
        s_sd = compute_same_diff_cosine(f_v_matrix, train_ids)
        s_gap = s_sd['same_mean'] - s_sd['diff_mean']
        print(f"  same_cos = {s_sd['same_mean']:.4f}")
        print(f"  diff_cos = {s_sd['diff_mean']:.4f}")
        print(f"  gap      = {s_gap:.4f}")
        print(f"  top1     = {s_result['top1']:.4f}")
        print(f"  top5     = {s_result['top5']:.4f}")
        print(f"  mAP      = {s_result['mAP']:.4f}")
        print(f"  skipped  = {s_result['skipped']}")

        random_top1 = 1.0 / n_ids if n_ids > 1 else 0.0
        random_top5 = min(5.0 / n_ids, 1.0)
        print(f"\n--- Random baseline ---")
        print(f"  top1 = {random_top1:.4f}")
        print(f"  top5 = {random_top5:.4f}")

        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)
        print(f"{'method':>10} | {'top1':>6} | {'top5':>6} | {'mAP':>6} | {'same_cos':>8} | {'diff_cos':>8} | {'gap':>6} | {'N':>5}")
        print("-" * 70)
        print(f"{'teacher':>10} | {t_result['top1']:6.4f} | {t_result['top5']:6.4f} | {t_result['mAP']:6.4f} | "
              f"{t_sd['same_mean']:8.4f} | {t_sd['diff_mean']:8.4f} | {t_gap:6.4f} | {N:5d}")
        print(f"{'student':>10} | {s_result['top1']:6.4f} | {s_result['top5']:6.4f} | {s_result['mAP']:6.4f} | "
              f"{s_sd['same_mean']:8.4f} | {s_sd['diff_mean']:8.4f} | {s_gap:6.4f} | {N:5d}")
        print(f"{'random':>10} | {random_top1:6.4f} | {random_top5:6.4f} | {'N/A':>6} | "
              f"{'N/A':>8} | {'N/A':>8} | {'N/A':>6} | {N:5d}")

        top1_ratio = s_result['top1'] / max(t_result['top1'], 1e-8)
        top5_ratio = s_result['top5'] / max(t_result['top5'], 1e-8)
        gap_ratio = s_gap / max(abs(t_gap), 1e-8)
        if t_gap < 0:
            gap_ratio = -gap_ratio
        print(f"\n--- Ratios ---")
        print(f"  student/teacher top1 ratio: {top1_ratio:.4f}")
        print(f"  student/teacher top5 ratio: {top5_ratio:.4f}")
        print(f"  student/teacher gap ratio:  {gap_ratio:.4f}")

        print(f"\n" + "=" * 70)
        print("JUDGMENT")
        print("=" * 70)

        if top1_ratio >= 0.9 and s_gap > 0:
            judgment = "CLOSE"
            print(f"  Student接近teacher")
        elif top1_ratio < 0.5 or s_gap <= 0:
            judgment = "BELOW"
            print(f"  Student明显低于teacher")
        else:
            judgment = "INTERMEDIATE"
            print(f"  Student处于中间水平")

        print(f"  student/teacher top1 = {top1_ratio:.2%}")
        print(f"  student/teacher gap  = {gap_ratio:.2%}")
        print(f"  student gap = {s_gap:.4f} ({'positive' if s_gap > 0 else 'NEGATIVE - no identity cluster'})")

        output = {
            'num_instances': int(N),
            'num_unique_ids': int(n_ids),
            'num_cameras': int(n_cameras),
            'camera_list': unique_cam_strs,
            'split': 'train',
            'mean_cos_fv_tv': mean_cos_ft,
            'skipped_instances': int(skipped_instances),
            'nan_inf_count': int(nan_count),
            'teacher': {
                'same_cos': t_sd['same_mean'],
                'diff_cos': t_sd['diff_mean'],
                'gap': t_gap,
                'top1': t_result['top1'],
                'top5': t_result['top5'],
                'mAP': t_result['mAP'],
                'skipped_queries': t_result['skipped'],
            },
            'student': {
                'same_cos': s_sd['same_mean'],
                'diff_cos': s_sd['diff_mean'],
                'gap': s_gap,
                'top1': s_result['top1'],
                'top5': s_result['top5'],
                'mAP': s_result['mAP'],
                'skipped_queries': s_result['skipped'],
            },
            'random': {
                'top1': random_top1,
                'top5': random_top5,
            },
            'ratios': {
                'top1_student_teacher': top1_ratio,
                'top5_student_teacher': top5_ratio,
                'gap_student_teacher': gap_ratio,
            },
            'judgment': judgment,
        }

        if args.enable_prototype:
            print(f"\n--- Teacher prototype sanity (optional) ---")
            try:
                unique_ids = np.unique(train_ids[train_ids >= 0])
                prototype_t = np.zeros((len(unique_ids), t_v_matrix.shape[1]))
                for idx, uid in enumerate(unique_ids):
                    mask = (train_ids == uid)
                    if mask.sum() > 0:
                        prototype_t[idx] = t_v_matrix[mask].mean(axis=0)
                prototype_t = prototype_t / (np.linalg.norm(prototype_t, axis=1, keepdims=True) + 1e-6)

                id_to_idx = {uid: idx for idx, uid in enumerate(unique_ids)}
                proto_idx = np.array([id_to_idx.get(tid, -1) for tid in train_ids], dtype=np.int64)

                valid_proto = proto_idx >= 0
                cos_t_proto = np.sum(t_v_matrix[valid_proto] * prototype_t[proto_idx[valid_proto]], axis=1)
                cos_f_proto = np.sum(f_v_matrix[valid_proto] * prototype_t[proto_idx[valid_proto]], axis=1)

                mean_cos_t_proto = float(np.mean(cos_t_proto))
                mean_cos_f_proto = float(np.mean(cos_f_proto))

                print(f"  mean cos(t_v, P_train_id) = {mean_cos_t_proto:.4f}")
                print(f"  mean cos(f_v, P_train_id) = {mean_cos_f_proto:.4f}")

                output['prototype'] = {
                    'mean_cos_tv_P': mean_cos_t_proto,
                    'mean_cos_fv_P': mean_cos_f_proto,
                }
            except Exception as e:
                print(f"  WARNING: Prototype computation failed: {e}")
                traceback.print_exc()
                output['prototype_error'] = str(e)

    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        traceback.print_exc()
        if output is None:
            output = {'error': str(e)}

    if output is not None:
        output_path = os.path.join(REPO_ROOT, args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved: {output_path}")

    if args.csv_output and len(all_f_v) > 0:
        try:
            csv_path = os.path.join(REPO_ROOT, args.csv_output)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'instance_idx', 'train_id', 'raw_id', 'camera_id', 'camera_str', 'frame_id',
                    'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
                    'cos_fv_tv',
                ])
                for idx in range(N):
                    bbox = bboxes[idx]
                    writer.writerow([
                        idx, int(train_ids[idx].item()), int(raw_ids[idx].item()),
                        int(camera_ids[idx].item()), all_camera_strs[idx], int(frame_ids[idx].item()),
                        float(bbox[0].item()), float(bbox[1].item()), float(bbox[2].item()), float(bbox[3].item()),
                        float(cos_student_teacher[idx].item()),
                    ])
            print(f"CSV saved: {csv_path}")
        except Exception as e:
            print(f"WARNING: CSV save failed: {e}")

    if output is not None and 'error' not in output:
        judgment = output.get('judgment', 'UNKNOWN')
        s_gap = output['student']['gap']
        top1_ratio = output['ratios']['top1_student_teacher']

        print(f"\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print(f"  MVP有效: student feature有identity signal (top1 > random)")
        print(f"  Student离teacher差距: top1={top1_ratio:.2%} of teacher")
        print(f"  Student gap: {s_gap:.4f} ({'positive' if s_gap > 0 else 'NEGATIVE'})")
        print(f"  Judgment: {judgment}")

        if judgment == "BELOW":
            print(f"\n  下一步建议:")
            print(f"    1. 优先加入 teacher prototype loss (identity-level supervision)")
            print(f"    2. Multi-view consistency loss")
            print(f"    3. 更长训练 (20k-50k iters)")
        elif judgment == "CLOSE":
            print(f"\n  下一步建议:")
            print(f"    1. 更长训练 + prototype loss")
            print(f"    2. Cross-camera-only retrieval")
        else:
            print(f"\n  下一步建议:")
            print(f"    1. 更长训练")
            print(f"    2. Prototype loss")
            print(f"    3. Multi-view consistency")

    return output is not None and 'error' not in output


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
