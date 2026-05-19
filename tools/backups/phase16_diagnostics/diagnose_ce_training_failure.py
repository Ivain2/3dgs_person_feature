#!/usr/bin/env python3
"""Phase 16 Diagnostics: CE Failure Root Cause Analysis.

Experiments:
  Task 0: Metadata / eval fix
  Task 1: Teacher embedding classifier-only upper bound
  Task 2: Cached student feature classifier-only
  Task 3: 5-ID balanced tiny CE overfit (with full renderer gradient)
  Task 4: Ablations (if needed)
  Task 5: Final report
"""

import argparse
import csv
import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.datasets import make as make_dataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import roi_pool, scale_bbox_to_render

# ─── Paths ───────────────────────────────────────────────────────────────────
TEACHER_ONLY_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt"
REID_INIT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase14_clean_geometry/full_soft_reset_30k/reid_init/reid_init_ckpt.pt"
DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
OUTPUT_DIR = "/data02/zhangrunxiang/3dgrut/outputs/phase16_ce_diagnostics"
DEVICE = "cuda"
RANDOM_SEED = 42
IMG_W, IMG_H = 1920, 1088  # original Wildtrack resolution (padded)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def setup_model(state, device):
    """Load geometry + person_feature from checkpoint, freeze everything except person_feature."""
    reid_state = load_ckpt(REID_INIT_CKPT)
    conf = reid_state.get("config", None)
    conf.model.person_feature_dim = 512
    scene_extent = reid_state.get("scene_extent", 1.0)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent)
    for key in ["positions", "density", "scale", "rotation", "features_albedo", "features_specular"]:
        if key in state:
            getattr(model, key).data = state[key].to(device)
    pf_key = "_person_feature" if "_person_feature" in state else "person_feature"
    model._person_feature = torch.nn.Parameter(state[pf_key].to(device))
    model = model.to(device)
    return model, conf


def freeze_all_except(model, trainable_names=None):
    """Freeze all parameters except those whose names contain trainable_names substrings."""
    for name, param in model.named_parameters():
        if trainable_names is None or not any(t in name for t in trainable_names):
            param.requires_grad = False


def make_classifier(in_dim, num_classes, use_layernorm=False):
    if use_layernorm:
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )


class BalancedSampler:
    """Mini-batch balanced sampler: each batch tries to cover all classes equally."""
    def __init__(self, data_list, batch_size):
        self.data_list = data_list
        self.batch_size = batch_size
        self.by_label = defaultdict(list)
        for i, (_, label) in enumerate(data_list):
            self.by_label[label].append(i)
        self.labels = sorted(self.by_label.keys())
        self.rng = np.random.RandomState(RANDOM_SEED)

    def __iter__(self):
        n_per_class = max(1, self.batch_size // len(self.labels))
        indices = []
        for label in self.labels:
            pool = self.by_label[label]
            if len(pool) >= n_per_class:
                indices.extend(self.rng.choice(pool, size=n_per_class, replace=False).tolist())
            else:
                indices.extend(pool)
                indices.extend(self.rng.choice(pool, size=n_per_class - len(pool), replace=True).tolist())
        self.rng.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return max(1, len(self.data_list) // self.batch_size)


# ─── Task 0: Collect metadata with correct cam_id / frame_id ─────────────────
def collect_metadata(model, dataset, device, max_frames=300):
    """Collect ROI samples with full metadata, ensuring cam_id is correct."""
    model.eval()
    all_samples = []
    t0 = time.time()

    with torch.no_grad():
        for idx in range(min(len(dataset), max_frames)):
            if idx % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [metadata] frame {idx}/{min(len(dataset), max_frames)}, samples={len(all_samples)}, {elapsed:.0f}s")

            batch = dataset[idx]
            # ── IMPORTANT: cam_id and frame_idx are at batch level, not in gpu_batch ──
            cam_id = batch.get("camera_id", "unknown")
            frame_idx = batch.get("frame_idx", -1)

            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            gpu_instances = gpu_batch.instances
            if not gpu_instances:
                continue

            pf_map = model(gpu_batch, train=False, frame_id=0, render_person_feature=True).get("person_feature_map")
            if pf_map is None:
                continue
            _, h, w = pf_map.shape

            for inst in gpu_instances:
                if not inst.get("valid", False):
                    continue
                train_id = inst.get("train_id")
                raw_id = inst.get("raw_id")
                if train_id is None:
                    continue
                bbox_orig = inst.get("bbox_xyxy_original")
                inst_w = inst.get("img_width_original", IMG_W)
                inst_h = inst.get("img_height_original", IMG_H)
                if bbox_orig is None:
                    continue
                bbox_r = scale_bbox_to_render(bbox_orig, src_w=inst_w, src_h=inst_h, dst_w=w, dst_h=h)
                x1 = int(torch.clamp(bbox_r[0], 0, w - 1).item())
                y1 = int(torch.clamp(bbox_r[1], 0, h - 1).item())
                x2 = int(torch.clamp(bbox_r[2], x1 + 1, w).item())
                y2 = int(torch.clamp(bbox_r[3], y1 + 1, h).item())
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox_c = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device)
                f_v, _ = roi_pool(pf_map, bbox_c)
                if f_v is None:
                    continue

                sample = {
                    "sample_id": f"{idx}_{train_id}_{raw_id}",
                    "idx": idx,
                    "cam_id": cam_id,
                    "frame_id": int(frame_idx),
                    "train_id": int(train_id),
                    "person_id": int(raw_id),
                    "bbox_xyxy_original": list(bbox_orig),
                    "bbox_render": [x1, y1, x2, y2],
                    "feature_norm": f_v.norm().item(),
                    "teacher_available": inst.get("teacher_embedding") is not None,
                }
                all_samples.append(sample)

    print(f"  [metadata] done: {len(all_samples)} samples in {time.time() - t0:.1f}s")
    return all_samples


def save_metadata_reports(all_samples, out_dir):
    """Save Task 0 metadata reports."""
    os.makedirs(out_dir, exist_ok=True)

    # cam_id distribution
    cam_counts = defaultdict(int)
    for s in all_samples:
        cam_counts[s["cam_id"]] += 1
    with open(os.path.join(out_dir, "cam_id_distribution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cam_id", "count"])
        for cam, cnt in sorted(cam_counts.items()):
            w.writerow([cam, cnt])

    # ID distribution
    id_counts = defaultdict(int)
    for s in all_samples:
        id_counts[s["train_id"]] += 1
    with open(os.path.join(out_dir, "id_distribution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_id", "count"])
        for tid, cnt in sorted(id_counts.items()):
            w.writerow([tid, cnt])

    # Sample debug (first 200)
    with open(os.path.join(out_dir, "identity_sample_debug.jsonl"), "w") as f:
        for s in all_samples[:200]:
            f.write(json.dumps(s) + "\n")

    # Pair construction analysis
    id_cams = defaultdict(set)
    id_frames = defaultdict(set)
    for s in all_samples:
        id_cams[s["train_id"]].add(s["cam_id"])
        id_frames[s["train_id"]].add(s["frame_id"])

    cross_cam_ids = sum(1 for tid in id_cams if len(id_cams[tid]) > 1)
    multi_frame_ids = sum(1 for tid in id_frames if len(id_frames[tid]) > 1)

    # Count cross-camera same-ID pairs
    cross_cam_same_pairs = 0
    for tid in id_cams:
        cams = sorted(id_cams[tid])
        if len(cams) > 1:
            samples_by_cam = defaultdict(list)
            for s in all_samples:
                if s["train_id"] == tid:
                    samples_by_cam[s["cam_id"]].append(s["sample_id"])
            for i in range(len(cams)):
                for j in range(i + 1, len(cams)):
                    cross_cam_same_pairs += len(samples_by_cam[cams[i]]) * len(samples_by_cam[cams[j]])

    report = (
        f"# Metadata Report\n\n"
        f"## Summary\n\n"
        f"- Total samples: {len(all_samples)}\n"
        f"- Total unique IDs: {len(id_counts)}\n"
        f"- Unique cameras: {len(cam_counts)}\n\n"
        f"## cam_id Coverage\n\n"
    )
    for cam, cnt in sorted(cam_counts.items()):
        report += f"- {cam}: {cnt} samples\n"

    report += (
        f"\n## ID Stats\n\n"
        f"- IDs with multi-camera samples: {cross_cam_ids}\n"
        f"- IDs with multi-frame samples: {multi_frame_ids}\n"
        f"- Cross-camera same-ID pairs: {cross_cam_same_pairs}\n\n"
        f"## Per-ID Multi-Camera IDs\n\n"
    )
    for tid in sorted(id_cams.keys()):
        if len(id_cams[tid]) > 1:
            report += f"- ID {tid}: cameras={sorted(id_cams[tid])}, frames={len(id_frames[tid])}\n"

    with open(os.path.join(out_dir, "metadata_report.md"), "w") as f:
        f.write(report)

    print(f"  [metadata report] cam_ids: {sorted(cam_counts.keys())}, cross_cam_ids: {cross_cam_ids}, cross_cam_same_pairs: {cross_cam_same_pairs}")
    return len(cam_counts) > 1, cross_cam_same_pairs > 0


# ─── Task 1: Teacher embedding classifier-only upper bound ────────────────────
def task1_teacher_classifier(dataset, all_samples, out_dir, num_ids=30, min_samples=20, steps=500, batch_size=64):
    """Train classifier on teacher embeddings to verify label mapping is learnable."""
    print("\n" + "=" * 60)
    print("Task 1: Teacher Embedding Classifier-Only Upper Bound")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    # Select IDs with enough samples
    id_counts = defaultdict(int)
    for s in all_samples:
        id_counts[s["train_id"]] += 1
    valid_ids = [tid for tid, cnt in id_counts.items() if cnt >= min_samples]
    valid_ids.sort(key=lambda tid: id_counts[tid], reverse=True)
    selected_ids = valid_ids[:num_ids]
    class_map = {tid: i for i, tid in enumerate(sorted(selected_ids))}
    num_classes = len(class_map)

    # Build training data from teacher embeddings (cached in dataset)
    train_data = []
    missing_teacher = 0
    for s in all_samples:
        if s["train_id"] not in class_map:
            continue
        # Get teacher embedding from dataset
        batch = dataset[s["idx"]]
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
        for inst in gpu_batch.instances:
            if inst.get("train_id") == s["train_id"] and inst.get("teacher_embedding") is not None:
                te = inst["teacher_embedding"]
                if isinstance(te, np.ndarray):
                    te = torch.from_numpy(te).float()
                train_data.append((te, class_map[s["train_id"]]))
                break
        else:
            missing_teacher += 1

    if len(train_data) < num_classes * 5:
        print(f"  [WARN] Only {len(train_data)} samples with teacher embedding, need more")
        # Fallback: use collected features (they were pooled from render, not teacher)
        # Actually for Task 1 we need teacher embeddings specifically
        print(f"  Missing teacher: {missing_teacher}")

    # Balanced dataset
    samples_by_label = defaultdict(list)
    for feat, label in train_data:
        samples_by_label[label].append(feat)

    # Trim to balanced count
    min_per_label = min(len(v) for v in samples_by_label.values()) if samples_by_label else 0
    train_list = []
    for label, feats in samples_by_label.items():
        for f in feats[:min_per_label]:
            train_list.append((f, label))

    if len(train_list) < num_classes * 2:
        print(f"  [FAIL] Not enough balanced samples: {len(train_list)}")
        return False

    # Train classifier
    classifier = make_classifier(512, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-3)
    sampler = BalancedSampler(train_list, batch_size)

    all_metrics = []
    t0 = time.time()
    for epoch in range(steps):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx in sampler:
            feats = torch.stack([train_data[i][0].to(DEVICE) for i in batch_idx], dim=0)
            labels = torch.tensor([train_data[i][1] for i in batch_idx], dtype=torch.long, device=DEVICE)
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
            epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
            epoch_total += len(labels)

        avg_loss = epoch_loss / max(1, epoch_total)
        acc = epoch_correct / max(1, epoch_total)
        all_metrics.append({"step": epoch, "ce_loss": avg_loss, "acc": acc})

        if epoch % 50 == 0 or epoch == steps - 1:
            print(f"  [Task1] epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}, n={epoch_total}, time={time.time()-t0:.0f}s")

    # Save outputs
    with open(os.path.join(out_dir, "metrics.jsonl"), "w") as f:
        for m in all_metrics:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(out_dir, "ce_loss_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "ce_loss"])
        w.writeheader()
        w.writerows([{"step": m["step"], "ce_loss": m["ce_loss"]} for m in all_metrics])
    with open(os.path.join(out_dir, "train_acc_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "acc"])
        w.writeheader()
        w.writerows([{"step": m["step"], "acc": m["acc"]} for m in all_metrics])

    init_loss = all_metrics[0]["ce_loss"]
    final_loss = all_metrics[-1]["ce_loss"]
    final_acc = all_metrics[-1]["acc"]

    passed = final_acc > 0.8 and final_loss < init_loss * 0.5
    print(f"  [Task1 RESULT] init_loss={init_loss:.4f}, final_loss={final_loss:.4f}, final_acc={final_acc:.4f}, passed={passed}")

    with open(os.path.join(out_dir, "final_report.md"), "w") as f:
        f.write(f"# Task 1: Teacher Classifier-Only\n\n"
                f"- Selected IDs: {num_classes}\n"
                f"- Training samples: {len(train_list)}\n"
                f"- Initial loss: {init_loss:.4f}\n"
                f"- Final loss: {final_loss:.4f}\n"
                f"- Final accuracy: {final_acc:.4f}\n"
                f"- Missing teacher embeddings: {missing_teacher}\n"
                f"- **PASS**: {passed}\n")
    return passed


# ─── Task 2: Cached student feature classifier-only ──────────────────────────
def task2_cached_student_classifier(model, dataset, all_samples, class_map, out_dir, steps=500, batch_size=64):
    """Train classifier on cached (frozen) student ROI features to check if they are separable."""
    print("\n" + "=" * 60)
    print("Task 2: Cached Student Feature Classifier-Only")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    num_classes = len(class_map)

    # Cache student features
    print("  [cache] Collecting student ROI features...")
    model.eval()
    cached_features = []
    cached_samples_info = []
    t0 = time.time()

    with torch.no_grad():
        for idx in range(min(len(dataset), 300)):
            batch = dataset[idx]
            cam_id = batch.get("camera_id", "unknown")
            frame_idx = batch.get("frame_idx", -1)
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            gpu_instances = gpu_batch.instances
            if not gpu_instances:
                continue

            pf_map = model(gpu_batch, train=False, frame_id=0, render_person_feature=True).get("person_feature_map")
            if pf_map is None:
                continue
            _, h, w = pf_map.shape

            for inst in gpu_instances:
                if not inst.get("valid", False):
                    continue
                train_id = inst.get("train_id")
                if train_id is None or train_id not in class_map:
                    continue
                bbox_orig = inst.get("bbox_xyxy_original")
                if bbox_orig is None:
                    continue
                bbox_r = scale_bbox_to_render(bbox_orig, src_w=IMG_W, src_h=IMG_H, dst_w=w, dst_h=h)
                x1 = int(torch.clamp(bbox_r[0], 0, w - 1).item())
                y1 = int(torch.clamp(bbox_r[1], 0, h - 1).item())
                x2 = int(torch.clamp(bbox_r[2], x1 + 1, w).item())
                y2 = int(torch.clamp(bbox_r[3], y1 + 1, h).item())
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox_c = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=DEVICE)
                f_v, _ = roi_pool(pf_map, bbox_c)
                if f_v is None:
                    continue

                cached_features.append(f_v.detach().cpu())
                cached_samples_info.append({
                    "idx": idx, "cam_id": cam_id, "frame_id": int(frame_idx),
                    "train_id": int(train_id), "class_id": class_map[train_id],
                    "feature_norm": f_v.norm().item(),
                })

    print(f"  [cache] {len(cached_features)} features in {time.time()-t0:.1f}s")

    if len(cached_features) < num_classes * 5:
        print(f"  [FAIL] Not enough cached features: {len(cached_features)}")
        return False

    # Build training list
    train_list = []
    for i, info in enumerate(cached_samples_info):
        train_list.append((cached_features[i], info["class_id"]))

    # Balance
    samples_by_label = defaultdict(list)
    for feat, label in train_list:
        samples_by_label[label].append((feat, label))
    min_per_label = min(len(v) for v in samples_by_label.values())
    balanced_list = []
    for label in samples_by_label:
        balanced_list.extend(samples_by_label[label][:min_per_label])

    # Train classifier (student features are FROZEN, only train classifier)
    classifier = make_classifier(512, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-3)
    sampler = BalancedSampler(balanced_list, batch_size)

    all_metrics = []
    feature_norms = [s["feature_norm"] for s in cached_samples_info]
    t0 = time.time()

    for epoch in range(steps):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx in sampler:
            feats = torch.stack([balanced_list[i][0].to(DEVICE) for i in batch_idx], dim=0)
            labels = torch.tensor([balanced_list[i][1] for i in batch_idx], dtype=torch.long, device=DEVICE)
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
            epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
            epoch_total += len(labels)

        avg_loss = epoch_loss / max(1, epoch_total)
        acc = epoch_correct / max(1, epoch_total)
        all_metrics.append({"step": epoch, "ce_loss": avg_loss, "acc": acc})

        if epoch % 50 == 0 or epoch == steps - 1:
            print(f"  [Task2] epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}, n={epoch_total}, time={time.time()-t0:.0f}s")

    # Save outputs
    torch.save({"features": torch.stack(cached_features), "info": cached_samples_info},
               os.path.join(out_dir, "cached_features.pt"))
    with open(os.path.join(out_dir, "cached_samples.jsonl"), "w") as f:
        for s in cached_samples_info:
            f.write(json.dumps(s) + "\n")
    with open(os.path.join(out_dir, "metrics.jsonl"), "w") as f:
        for m in all_metrics:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(out_dir, "ce_loss_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "ce_loss"])
        w.writeheader()
        w.writerows([{"step": m["step"], "ce_loss": m["ce_loss"]} for m in all_metrics])
    with open(os.path.join(out_dir, "train_acc_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "acc"])
        w.writeheader()
        w.writerows([{"step": m["step"], "acc": m["acc"]} for m in all_metrics])

    # Pairwise cosine stats
    features = torch.stack(cached_features)
    features_normed = F.normalize(features, p=2, dim=-1)
    n = min(500, len(features_normed))
    cos_matrix = (features_normed[:n] @ features_normed[:n].T).numpy()
    np.fill_diagonal(cos_matrix, np.nan)
    valid_cosines = cos_matrix[~np.isnan(cos_matrix)]

    # Same/diff cosine
    id_to_indices = defaultdict(list)
    for i, info in enumerate(cached_samples_info):
        id_to_indices[info["train_id"]].append(i)

    same_cosines, diff_cosines = [], []
    ids_list = list(id_to_indices.keys())
    for i, id_a in enumerate(ids_list):
        for j in range(len(id_to_indices[id_a])):
            for k in range(j + 1, len(id_to_indices[id_a])):
                idx_a, idx_b = id_to_indices[id_a][j], id_to_indices[id_a][k]
                cos = F.cosine_similarity(features_normed[idx_a:idx_a+1], features_normed[idx_b:idx_b+1]).item()
                same_cosines.append(cos)
        for id_b in ids_list[i+1:]:
            for idx_a in id_to_indices[id_a]:
                for idx_b in id_to_indices[id_b]:
                    cos = F.cosine_similarity(features_normed[idx_a:idx_a+1], features_normed[idx_b:idx_b+1]).item()
                    diff_cosines.append(cos)

    same_mean = np.mean(same_cosines) if same_cosines else 0
    diff_mean = np.mean(diff_cosines) if diff_cosines else 0

    with open(os.path.join(out_dir, "pairwise_cosine_stats.json"), "w") as f:
        json.dump({
            "mean": float(np.nanmean(valid_cosines)), "std": float(np.nanstd(valid_cosines)),
            "min": float(np.nanmin(valid_cosines)), "max": float(np.nanmax(valid_cosines)),
            "dup_ratio_0.999": float(np.mean(valid_cosines > 0.999)),
            "same_id_mean": float(same_mean), "diff_id_mean": float(diff_mean),
            "same_diff_gap": float(same_mean - diff_mean),
            "same_pairs": len(same_cosines), "diff_pairs": len(diff_cosines),
        }, f, indent=2)

    with open(os.path.join(out_dir, "identity_gap_eval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["same_id_cosine", "diff_id_cosine", "gap", "same_pairs", "diff_pairs"])
        w.writerow([same_mean, diff_mean, same_mean - diff_mean, len(same_cosines), len(diff_cosines)])

    init_loss = all_metrics[0]["ce_loss"]
    final_loss = all_metrics[-1]["ce_loss"]
    final_acc = all_metrics[-1]["acc"]

    passed = final_acc > 0.5 and final_loss < init_loss * 0.7
    print(f"  [Task2 RESULT] init_loss={init_loss:.4f}, final_loss={final_loss:.4f}, final_acc={final_acc:.4f}")
    print(f"  [Task2 RESULT] same_cos={same_mean:.4f}, diff_cos={diff_mean:.4f}, gap={same_mean-diff_mean:.4f}")
    print(f"  [Task2 RESULT] passed={passed}")

    with open(os.path.join(out_dir, "final_report.md"), "w") as f:
        f.write(f"# Task 2: Cached Student Classifier-Only\n\n"
                f"- Cached features: {len(cached_features)}\n"
                f"- Feature norm mean: {np.mean(feature_norms):.4f}\n"
                f"- Initial loss: {init_loss:.4f}\n"
                f"- Final loss: {final_loss:.4f}\n"
                f"- Final accuracy: {final_acc:.4f}\n"
                f"- Same-id cosine: {same_mean:.4f}\n"
                f"- Diff-id cosine: {diff_mean:.4f}\n"
                f"- Gap: {same_mean - diff_mean:.4f}\n"
                f"- **PASS** (features are separable): {passed}\n")
    return passed


# ─── Task 3: 5-ID balanced tiny CE overfit ────────────────────────────────────
def task3_tiny_ce(model, dataset, all_samples, out_dir, num_ids=5, min_samples=20, max_samples=50, steps=1000, batch_size=16):
    """Minimal CE overfit with 5 IDs, both person_feature and classifier trainable."""
    print("\n" + "=" * 60)
    print("Task 3: 5-ID Balanced Tiny CE Overfit")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    # Select 5 IDs with most samples and multi-camera coverage
    id_counts = defaultdict(int)
    id_cams = defaultdict(set)
    for s in all_samples:
        id_counts[s["train_id"]] += 1
        id_cams[s["train_id"]].add(s["cam_id"])

    # Prefer IDs with multi-camera coverage
    valid_ids = [tid for tid, cnt in id_counts.items() if cnt >= min_samples]
    multi_cam_ids = [tid for tid in valid_ids if len(id_cams[tid]) > 1]
    single_cam_ids = [tid for tid in valid_ids if len(id_cams[tid]) == 1]

    # Sort: multi-cam first, then by count
    multi_cam_ids.sort(key=lambda tid: id_counts[tid], reverse=True)
    single_cam_ids.sort(key=lambda tid: id_counts[tid], reverse=True)

    selected_ids = multi_cam_ids[:num_ids]
    if len(selected_ids) < num_ids:
        selected_ids.extend(single_cam_ids[:num_ids - len(selected_ids)])
    selected_ids = selected_ids[:num_ids]

    class_map = {tid: i for i, tid in enumerate(sorted(selected_ids))}
    num_classes = len(class_map)

    print(f"  Selected IDs: {selected_ids}")
    print(f"  Class map: {class_map}")

    # Filter samples
    selected_samples = [s for s in all_samples if s["train_id"] in class_map]
    # Limit samples per ID
    samples_per_id = defaultdict(list)
    for s in selected_samples:
        samples_per_id[s["train_id"]].append(s)
    balanced_samples = []
    for tid in selected_ids:
        samples = samples_per_id[tid]
        balanced_samples.extend(samples[:max_samples])

    print(f"  Total training samples: {len(balanced_samples)}")

    # Save selected info
    with open(os.path.join(out_dir, "selected_ids.json"), "w") as f:
        json.dump({"ids": selected_ids, "class_map": {str(k): v for k, v in class_map.items()}}, f, indent=2)
    with open(os.path.join(out_dir, "selected_samples.jsonl"), "w") as f:
        for s in balanced_samples:
            f.write(json.dumps(s) + "\n")

    # Subset stats
    id_cam_counts = defaultdict(set)
    cam_counts = defaultdict(int)
    for s in balanced_samples:
        id_cam_counts[s["train_id"]].add(s["cam_id"])
        cam_counts[s["cam_id"]] += 1

    with open(os.path.join(out_dir, "subset_stats.md"), "w") as f:
        f.write(f"# 5-ID CE Subset Stats\n\n")
        f.write(f"- ID count: {num_classes}\n")
        f.write(f"- Total samples: {len(balanced_samples)}\n\n")
        for tid in sorted(selected_ids):
            f.write(f"- ID {tid}: {len(samples_per_id[tid])} samples, cameras: {sorted(id_cam_counts[tid])}\n")
        f.write(f"\n## Camera Distribution\n\n")
        for cam, cnt in sorted(cam_counts.items()):
            f.write(f"- {cam}: {cnt} samples\n")

    # ── Training: model.train(), person_feature + classifier both trainable ──
    model.train()
    freeze_all_except(model, ["person_feature"])
    assert model._person_feature.requires_grad, "person_feature must be trainable"

    classifier = make_classifier(512, num_classes).to(DEVICE)
    person_feature_lr = 0.01
    classifier_lr = 0.003

    optimizer = torch.optim.Adam([
        {"params": model._person_feature, "lr": person_feature_lr},
        {"params": classifier.parameters(), "lr": classifier_lr},
    ])

    # Build indexed training list
    train_list = []
    for s in balanced_samples:
        train_list.append((s, class_map[s["train_id"]]))

    sampler = BalancedSampler(train_list, batch_size)

    # Track initial person_feature state
    initial_pf = model._person_feature.detach().clone()

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    all_metrics = []
    per_cam_data = defaultdict(lambda: {"correct": 0, "total": 0})
    per_id_data = defaultdict(lambda: {"correct": 0, "total": 0})

    t0 = time.time()
    for step in range(steps):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        step_roi_count = 0

        for batch_idx in sampler:
            batch_samples = [train_list[i] for i in batch_idx]

            # Group by frame to avoid re-rendering
            frame_groups = defaultdict(list)
            for sample, label in batch_samples:
                key = sample["idx"]
                frame_groups[key].append((sample, label))

            loss_terms = []
            for idx, items in frame_groups.items():
                batch = dataset[idx % len(dataset)]
                cam_id = batch.get("camera_id", "unknown")
                gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
                pf_map = model(gpu_batch, train=True, frame_id=step, render_person_feature=True).get("person_feature_map")
                if pf_map is None:
                    continue
                _, h, w = pf_map.shape

                features = []
                labels = []
                for sample, label in items:
                    bbox_orig = sample.get("bbox_xyxy_original")
                    if bbox_orig is None:
                        continue
                    bbox_r = scale_bbox_to_render(bbox_orig, src_w=IMG_W, src_h=IMG_H, dst_w=w, dst_h=h)
                    x1 = int(torch.clamp(bbox_r[0], 0, w - 1).item())
                    y1 = int(torch.clamp(bbox_r[1], 0, h - 1).item())
                    x2 = int(torch.clamp(bbox_r[2], x1 + 1, w).item())
                    y2 = int(torch.clamp(bbox_r[3], y1 + 1, h).item())
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bbox_c = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=DEVICE)
                    f_v, _ = roi_pool(pf_map, bbox_c)
                    if f_v is None:
                        continue
                    features.append(f_v)
                    labels.append(label)

                if len(features) == 0:
                    continue

                features = torch.stack(features, dim=0)
                labels_t = torch.tensor(labels, dtype=torch.long, device=DEVICE)
                logits = classifier(features)
                loss = F.cross_entropy(logits, labels_t)
                loss_terms.append(loss)

                pred = logits.argmax(dim=-1)
                for i, l in enumerate(labels_t):
                    per_cam_data[cam_id]["total"] += 1
                    per_id_data[labels[i]]["total"] += 1
                    if pred[i].item() == l.item():
                        per_cam_data[cam_id]["correct"] += 1
                        per_id_data[labels[i]]["correct"] += 1
                    epoch_correct += 1
                epoch_total += len(labels)
                step_roi_count += len(labels)

            if loss_terms:
                loss = torch.stack(loss_terms).mean()
                optimizer.zero_grad()
                loss.backward()

                # Grad clipping
                torch.nn.utils.clip_grad_norm_(
                    list(model._person_feature.parameters()) if hasattr(model._person_feature, 'parameters') else [model._person_feature],
                    max_norm=1.0
                )

                grad_pf = model._person_feature.grad.norm().item() if model._person_feature.grad is not None else 0.0
                grad_clf = sum(p.grad.norm().item() for p in classifier.parameters() if p.grad is not None)
                optimizer.step()
                epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, step_roi_count)
        acc = epoch_correct / max(1, epoch_total)
        grad_pf = model._person_feature.grad.norm().item() if model._person_feature.grad is not None else 0.0
        pf_delta = (model._person_feature - initial_pf).norm().item()
        geometry_delta = model.positions.norm().item()  # should be constant

        all_metrics.append({
            "step": step, "ce_loss": avg_loss, "acc": acc,
            "grad_norm_person_feature": grad_pf,
            "person_feature_delta": pf_delta,
            "valid_roi": epoch_total,
        })

        if step % 100 == 0 or step == steps - 1:
            elapsed = time.time() - t0
            print(f"  [Task3] step {step}: loss={avg_loss:.4f}, acc={acc:.4f}, grad_pf={grad_pf:.6f}, pf_delta={pf_delta:.4f}, roi={epoch_total}, time={elapsed:.0f}s")

        # Save checkpoints
        if step == 499:
            torch.save({
                "_person_feature": model._person_feature.detach().cpu(),
                "classifier": classifier.state_dict(),
                "global_step": 500,
            }, os.path.join(out_dir, "checkpoint_500.pt"))
        if step == 999:
            torch.save({
                "_person_feature": model._person_feature.detach().cpu(),
                "classifier": classifier.state_dict(),
                "global_step": 1000,
            }, os.path.join(out_dir, "checkpoint_1000.pt"))

    # Save metrics
    with open(metrics_path, "w") as f:
        for m in all_metrics:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(out_dir, "ce_loss_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "ce_loss"])
        w.writeheader()
        w.writerows([{"step": m["step"], "ce_loss": m["ce_loss"]} for m in all_metrics])
    with open(os.path.join(out_dir, "train_acc_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "acc"])
        w.writeheader()
        w.writerows([{"step": m["step"], "acc": m["acc"]} for m in all_metrics])

    # Per-ID accuracy
    with open(os.path.join(out_dir, "per_id_acc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_id", "class_id", "correct", "total", "accuracy"])
        for tid in sorted(selected_ids):
            d = per_id_data.get(class_map[tid], {"correct": 0, "total": 0})
            a = d["correct"] / max(1, d["total"])
            w.writerow([tid, class_map[tid], d["correct"], d["total"], f"{a:.4f}"])

    # Per-camera accuracy
    with open(os.path.join(out_dir, "per_camera_acc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "correct", "total", "accuracy"])
        for cam, d in sorted(per_cam_data.items()):
            a = d["correct"] / max(1, d["total"])
            w.writerow([cam, d["correct"], d["total"], f"{a:.4f}"])

    # Identity eval after CE
    model.eval()
    all_features = []
    with torch.no_grad():
        for idx in range(min(len(dataset), 300)):
            batch = dataset[idx]
            cam_id = batch.get("camera_id", "unknown")
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            gpu_instances = gpu_batch.instances
            if not gpu_instances:
                continue
            pf_map = model(gpu_batch, train=False, frame_id=0, render_person_feature=True).get("person_feature_map")
            if pf_map is None:
                continue
            _, h, w = pf_map.shape
            for inst in gpu_instances:
                if not inst.get("valid", False):
                    continue
                train_id = inst.get("train_id")
                if train_id is None or train_id not in class_map:
                    continue
                bbox_orig = inst.get("bbox_xyxy_original")
                if bbox_orig is None:
                    continue
                bbox_r = scale_bbox_to_render(bbox_orig, src_w=IMG_W, src_h=IMG_H, dst_w=w, dst_h=h)
                x1 = int(torch.clamp(bbox_r[0], 0, w - 1).item())
                y1 = int(torch.clamp(bbox_r[1], 0, h - 1).item())
                x2 = int(torch.clamp(bbox_r[2], x1 + 1, w).item())
                y2 = int(torch.clamp(bbox_r[3], y1 + 1, h).item())
                if x2 <= x1 or y2 <= y1:
                    continue
                bbox_c = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=DEVICE)
                f_v, _ = roi_pool(pf_map, bbox_c)
                if f_v is None:
                    continue
                all_features.append({
                    "train_id": int(train_id), "cam_id": cam_id,
                    "feature": F.normalize(f_v.detach().cpu().squeeze(0), p=2, dim=-1),
                })

    # Compute pairwise cosine stats
    if len(all_features) >= 10:
        features = torch.stack([f["feature"] for f in all_features])
        n = min(300, len(features))
        cos_matrix = (features[:n] @ features[:n].T).numpy()
        np.fill_diagonal(cos_matrix, np.nan)
        valid_cosines = cos_matrix[~np.isnan(cos_matrix)]

        id_to_features = defaultdict(list)
        for f in all_features:
            id_to_features[f["train_id"]].append(f)

        same_cos, diff_cos = [], []
        ids_list = list(id_to_features.keys())
        for i, id_a in enumerate(ids_list):
            entries_a = id_to_features[id_a]
            for j in range(1, len(entries_a)):
                same_cos.append(F.cosine_similarity(entries_a[0]["feature"].unsqueeze(0), entries_a[j]["feature"].unsqueeze(0)).item())
            for id_b in ids_list[i+1:]:
                for ea in entries_a:
                    for eb in id_to_features[id_b]:
                        diff_cos.append(F.cosine_similarity(ea["feature"].unsqueeze(0), eb["feature"].unsqueeze(0)).item())

        same_mean = np.mean(same_cos) if same_cos else 0
        diff_mean = np.mean(diff_cos) if diff_cos else 0
        dup_ratio = float(np.mean(valid_cosines > 0.999))
    else:
        same_mean = diff_mean = dup_ratio = 0

    init_loss = all_metrics[0]["ce_loss"]
    final_loss = all_metrics[-1]["ce_loss"]
    final_acc = all_metrics[-1]["acc"]
    final_grad_pf = all_metrics[-1]["grad_norm_person_feature"]
    pf_delta = all_metrics[-1]["person_feature_delta"]

    # Pass criteria
    passed = (
        final_acc > 0.8
        and final_loss < init_loss * 0.5
        and final_grad_pf > 0.001
        and pf_delta > 0.01
        and dup_ratio < 0.3
    )

    print(f"\n  [Task3 RESULT]")
    print(f"    init_loss={init_loss:.4f}, final_loss={final_loss:.4f}")
    print(f"    final_acc={final_acc:.4f}")
    print(f"    final_grad_pf={final_grad_pf:.6f}")
    print(f"    person_feature_delta={pf_delta:.4f}")
    print(f"    dup_ratio={dup_ratio:.4f}")
    print(f"    same_cos={same_mean:.4f}, diff_cos={diff_mean:.4f}, gap={same_mean-diff_mean:.4f}")
    print(f"    **PASS**: {passed}")

    with open(os.path.join(out_dir, "final_report.md"), "w") as f:
        f.write(f"# Task 3: 5-ID Balanced Tiny CE Overfit\n\n"
                f"## Config\n\n"
                f"- IDs: {selected_ids}\n"
                f"- Samples: {len(balanced_samples)}\n"
                f"- Classes: {num_classes}\n"
                f"- Steps: {steps}\n"
                f"- Batch size: {batch_size}\n"
                f"- person_feature lr: {person_feature_lr}\n"
                f"- classifier lr: {classifier_lr}\n\n"
                f"## Results\n\n"
                f"- Initial CE loss: {init_loss:.4f}\n"
                f"- Final CE loss: {final_loss:.4f}\n"
                f"- Final accuracy: {final_acc:.4f}\n"
                f"- Final grad_pf: {final_grad_pf:.6f}\n"
                f"- person_feature delta: {pf_delta:.4f}\n"
                f"- Duplicate ratio (>0.999): {dup_ratio:.4f}\n"
                f"- Same-id cosine: {same_mean:.4f}\n"
                f"- Diff-id cosine: {diff_mean:.4f}\n"
                f"- Same/diff gap: {same_mean - diff_mean:.4f}\n\n"
                f"## Pass Criteria\n\n"
                f"- acc > 0.80: {'YES' if final_acc > 0.8 else 'NO'} ({final_acc:.4f})\n"
                f"- loss decrease > 50%: {'YES' if final_loss < init_loss * 0.5 else 'NO'} ({final_loss/init_loss:.2f}x)\n"
                f"- grad_pf > 0.001: {'YES' if final_grad_pf > 0.001 else 'NO'} ({final_grad_pf:.6f})\n"
                f"- pf_delta > 0.01: {'YES' if pf_delta > 0.01 else 'NO'} ({pf_delta:.4f})\n"
                f"- dup_ratio < 0.3: {'YES' if dup_ratio < 0.3 else 'NO'} ({dup_ratio:.4f})\n\n"
                f"## Per-ID Accuracy\n\n")
        for tid in sorted(selected_ids):
            d = per_id_data.get(class_map[tid], {"correct": 0, "total": 0})
            a = d["correct"] / max(1, d["total"])
            f.write(f"- ID {tid}: {d['correct']}/{d['total']} = {a:.4f}\n")

        f.write(f"\n## Per-Camera Accuracy\n\n")
        for cam, d in sorted(per_cam_data.items()):
            a = d["correct"] / max(1, d["total"])
            f.write(f"- {cam}: {d['correct']}/{d['total']} = {a:.4f}\n")

        f.write(f"\n## Decision: {'PASS' if passed else 'FAIL'}\n")

    return passed


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_ckpt", default=TEACHER_ONLY_CKPT)
    parser.add_argument("--num_ids_teacher", type=int, default=30)
    parser.add_argument("--num_ids_tiny", type=int, default=5)
    parser.add_argument("--steps_task1", type=int, default=500)
    parser.add_argument("--steps_task3", type=int, default=1000)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    os.environ["TORCH_EXTENSIONS_DIR"] = "/data02/zhangrunxiang/.cache/torch_extensions/py311_cu118"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    print("\n" + "=" * 60)
    print("Loading checkpoint and model")
    print("=" * 60)
    state = load_ckpt(args.init_ckpt)
    print(f"  positions: {state['positions'].shape}")
    print(f"  _person_feature: {state['_person_feature'].shape}")

    model, conf = setup_model(state, device)
    freeze_all_except(model, ["person_feature"])
    print(f"  person_feature requires_grad: {model._person_feature.requires_grad}")

    # ── Load dataset ──
    print("\n" + "=" * 60)
    print("Loading dataset")
    print("=" * 60)
    train_ds, _ = make_dataset("wildtrack", conf, ray_jitter=None)
    print(f"  Dataset size: {len(train_ds)}")

    # ── Task 0: Metadata ──
    print("\n" + "=" * 60)
    print("Task 0: Metadata / Eval Fix")
    print("=" * 60)
    all_samples = collect_metadata(model, train_ds, device, max_frames=300)
    print(f"  Collected {len(all_samples)} valid samples")
    cam_ok, pair_ok = save_metadata_reports(all_samples, os.path.join(OUTPUT_DIR, "metadata_debug"))

    # ── Select IDs for Tasks 1-2 ──
    id_counts = defaultdict(int)
    for s in all_samples:
        id_counts[s["train_id"]] += 1
    valid_ids = [tid for tid, cnt in id_counts.items() if cnt >= 20]
    valid_ids.sort(key=lambda tid: id_counts[tid], reverse=True)
    selected_ids_task1 = valid_ids[:args.num_ids_teacher]
    class_map_task1 = {tid: i for i, tid in enumerate(sorted(selected_ids_task1))}

    # ── Task 1: Teacher classifier-only ──
    task1_pass = task1_teacher_classifier(
        train_ds, all_samples, os.path.join(OUTPUT_DIR, "teacher_classifier_only"),
        num_ids=args.num_ids_teacher, min_samples=20,
        steps=args.steps_task1, batch_size=64
    )

    # ── Task 2: Cached student classifier-only ──
    task2_pass = task2_cached_student_classifier(
        model, train_ds, all_samples, class_map_task1,
        os.path.join(OUTPUT_DIR, "student_cached_classifier_only"),
        steps=args.steps_task1, batch_size=64
    )

    # ── Task 3: 5-ID tiny CE ──
    # Clear GPU cache before Task 3 to avoid OOM
    torch.cuda.empty_cache()

    # Reload model fresh to avoid any state corruption from previous tasks
    print("\nReloading model for Task 3...")
    state = load_ckpt(args.init_ckpt)
    model, conf = setup_model(state, device)
    freeze_all_except(model, ["person_feature"])

    task3_pass = task3_tiny_ce(
        model, train_ds, all_samples, os.path.join(OUTPUT_DIR, "ce_tiny_5id"),
        num_ids=args.num_ids_tiny, min_samples=20, max_samples=50,
        steps=args.steps_task3, batch_size=8
    )

    # ── Task 5: Final Report ──
    print("\n" + "=" * 60)
    print("Task 5: Final Report")
    print("=" * 60)

    if task1_pass and task2_pass and task3_pass:
        decision = "A"
        decision_text = "All diagnostics PASS. Teacher upper bound OK, student features are separable, 5-ID CE overfit works. Next: 30-ID balanced CE."
    elif task1_pass and not task2_pass:
        decision = "B"
        decision_text = "Teacher upper bound PASS but student cached features NOT separable. Teacher-only student features lack identity info. Try reid_init + CE or Teacher+CE, NOT SupCon."
    elif task1_pass and task2_pass and not task3_pass:
        decision = "C"
        decision_text = "Teacher PASS, student cached PASS, but train-with-PF CE FAIL. Issue in renderer/PF gradient path, ROI normalize, or optimizer settings. Fix training pipeline before scaling."
    elif not task1_pass:
        decision = "D"
        decision_text = "Teacher upper bound FAIL. Label mapping, subset selection, or classifier architecture is broken. Fix these before any CE experiment."
    else:
        decision = "E"
        decision_text = "cam_id / metadata issue. Fix eval metadata before proceeding."

    with open(os.path.join(OUTPUT_DIR, "final_report.md"), "w") as f:
        f.write(f"# Phase 16 CE Diagnostics - Final Report\n\n"
                f"## Summary\n\n"
                f"- Metadata / cam_id: {'FIXED' if cam_ok and pair_ok else 'ISSUES REMAIN'}\n"
                f"- Task 1 (Teacher classifier-only): {'PASS' if task1_pass else 'FAIL'}\n"
                f"- Task 2 (Cached student classifier-only): {'PASS' if task2_pass else 'FAIL'}\n"
                f"- Task 3 (5-ID tiny CE overfit): {'PASS' if task3_pass else 'FAIL'}\n\n"
                f"## Decision: {decision}\n\n"
                f"{decision_text}\n\n"
                f"## Next Steps\n\n"
                f"- If A: Enter 30-ID balanced CE small overfit\n"
                f"- If B: Try reid_init + CE or Teacher+CE, NOT SupCon\n"
                f"- If C: Fix renderer/PF gradient path, ROI normalize, or optimizer\n"
                f"- If D: Fix label mapping / subset / classifier\n"
                f"- If E: Fix metadata eval before any training\n")

    print(f"\nFinal decision: {decision}")
    print(f"Report: {OUTPUT_DIR}/final_report.md")


if __name__ == "__main__":
    main()
