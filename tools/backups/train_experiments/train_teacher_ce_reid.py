#!/usr/bin/env python3
"""Phase 17: Teacher+CE ReID Training with P×K Sampler.

Unified training script supporting:
  - Teacher+CE combined loss
  - P×K balanced sampler (P identities × K samples)
  - LayerNorm neck + linear classifier
  - 30-ID or all-ID modes
  - Comprehensive identity/cross-camera evaluation
  - Auto anomaly monitoring
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.datasets import make as make_dataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import roi_pool, scale_bbox_to_render

REID_INIT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase14_clean_geometry/full_soft_reset_30k/reid_init/reid_init_ckpt.pt"
TEACHER_ONLY_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt"
DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
OUTPUT_ROOT = "/data02/zhangrunxiang/3dgrut/outputs/phase17_teacher_ce_experiments"
IMG_W, IMG_H = 1920, 1088
RANDOM_SEED = 42


# ─── Model Setup ──────────────────────────────────────────────────────────────
def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def setup_model(init_ckpt, device, reid_init_ckpt=REID_INIT_CKPT):
    reid_state = load_ckpt(reid_init_ckpt)
    conf = reid_state.get("config", None)
    conf.model.person_feature_dim = 512
    scene_extent = reid_state.get("scene_extent", 1.0)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent)

    state = load_ckpt(init_ckpt)
    for key in ["positions", "density", "scale", "rotation", "features_albedo", "features_specular"]:
        if key in state:
            getattr(model, key).data = state[key].to(device)
    pf_key = "_person_feature" if "_person_feature" in state else "person_feature"
    if pf_key in state:
        model._person_feature = torch.nn.Parameter(state[pf_key].to(device))
    model = model.to(device)

    for name, param in model.named_parameters():
        if "person_feature" not in name:
            param.requires_grad = False
    return model, conf, state


# ─── Neck + Classifier ────────────────────────────────────────────────────────
class ReIDHead(nn.Module):
    def __init__(self, in_dim, num_classes, neck_dim=256, use_layernorm=True):
        super().__init__()
        self.neck = nn.Sequential(
            nn.Linear(in_dim, neck_dim),
            nn.LayerNorm(neck_dim) if use_layernorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(neck_dim, num_classes)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        neck_feat = self.neck(x)
        logits = self.classifier(neck_feat)
        return logits, neck_feat


# ─── P×K Sampler ──────────────────────────────────────────────────────────────
class PxKSampler:
    """Sample P identities, K samples each, frame-aware grouping."""
    def __init__(self, samples_by_id, P=8, K=4, seed=RANDOM_SEED):
        self.P = P
        self.K = K
        self.ids = sorted(samples_by_id.keys())
        self.samples_by_id = samples_by_id
        self.rng = np.random.RandomState(seed)
        self.step_count = 0

    def sample_batch(self):
        if len(self.ids) < self.P:
            chosen_ids = self.ids
        else:
            chosen_ids = list(self.rng.choice(self.ids, size=self.P, replace=False))

        batch = []
        for tid in chosen_ids:
            pool = self.samples_by_id[tid]
            k = min(self.K, len(pool))
            indices = self.rng.choice(len(pool), size=k, replace=False)
            for i in indices:
                batch.append(pool[i])
        self.rng.shuffle(batch)
        self.step_count += 1
        return batch


# ─── Sample Collection ────────────────────────────────────────────────────────
def collect_samples(model, dataset, device, max_frames=400, class_map=None, min_samples=5):
    """Collect ROI samples with teacher embeddings and full metadata."""
    model.eval()
    all_samples = []
    t0 = time.time()

    with torch.no_grad():
        for idx in range(min(len(dataset), max_frames)):
            if idx % 50 == 0:
                print(f"  [collect] frame {idx}/{min(len(dataset), max_frames)}, samples={len(all_samples)}, {time.time()-t0:.0f}s")
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
                if train_id is None:
                    continue
                if class_map is not None and train_id not in class_map:
                    continue
                bbox_orig = inst.get("bbox_xyxy_original")
                if bbox_orig is None:
                    continue
                teacher_emb = inst.get("teacher_embedding")
                if teacher_emb is None:
                    continue

                bbox_r = scale_bbox_to_render(bbox_orig, src_w=IMG_W, src_h=IMG_H, dst_w=w, dst_h=h)
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

                te = torch.tensor(teacher_emb, dtype=torch.float32) if isinstance(teacher_emb, np.ndarray) else teacher_emb
                if isinstance(te, torch.Tensor):
                    te = te.float()

                all_samples.append({
                    "idx": idx, "cam_id": cam_id, "frame_id": int(frame_idx),
                    "train_id": int(train_id), "class_id": -1,
                    "bbox_xyxy_original": list(bbox_orig),
                    "bbox_render": [x1, y1, x2, y2],
                    "teacher_embedding": te,
                    "student_feature_norm": f_v.norm().item(),
                })

    print(f"  [collect] done: {len(all_samples)} samples in {time.time()-t0:.0f}s")
    return all_samples


def select_ids(all_samples, num_ids=30, min_samples=20, max_samples=50):
    """Select top IDs by sample count, preferring multi-camera IDs."""
    id_counts = defaultdict(int)
    id_cams = defaultdict(set)
    for s in all_samples:
        id_counts[s["train_id"]] += 1
        id_cams[s["train_id"]].add(s["cam_id"])

    valid_ids = [tid for tid, cnt in id_counts.items() if cnt >= min_samples]
    multi_cam = sorted([tid for tid in valid_ids if len(id_cams[tid]) > 1],
                       key=lambda t: id_counts[t], reverse=True)
    single_cam = sorted([tid for tid in valid_ids if len(id_cams[tid]) == 1],
                        key=lambda t: id_counts[t], reverse=True)
    selected = (multi_cam + single_cam)[:num_ids]
    class_map = {tid: i for i, tid in enumerate(sorted(selected))}
    return selected, class_map


def prepare_samples(all_samples, class_map, max_per_id=50):
    """Assign class_id and limit samples per ID."""
    samples_by_id = defaultdict(list)
    for s in all_samples:
        if s["train_id"] not in class_map:
            continue
        s["class_id"] = class_map[s["train_id"]]
        samples_by_id[s["train_id"]].append(s)

    for tid in samples_by_id:
        samples_by_id[tid] = samples_by_id[tid][:max_per_id]
    return samples_by_id


# ─── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, dataset, device, class_map, head, out_dir=None, prefix="eval"):
    """Comprehensive identity/cross-camera evaluation."""
    model.eval()
    all_feats = []
    t0 = time.time()

    with torch.no_grad():
        for idx in range(min(len(dataset), 400)):
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
                bbox_c = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device)
                f_v, _ = roi_pool(pf_map, bbox_c)
                if f_v is None:
                    continue
                te = inst.get("teacher_embedding")
                te_tensor = None
                if te is not None:
                    te_tensor = torch.tensor(te, dtype=torch.float32) if isinstance(te, np.ndarray) else te.float()

                all_feats.append({
                    "train_id": int(train_id), "cam_id": cam_id, "frame_id": int(frame_idx),
                    "feature": F.normalize(f_v.detach().cpu().squeeze(0), p=2, dim=-1),
                    "teacher_embedding": F.normalize(te_tensor, p=2, dim=-1) if te_tensor is not None else None,
                })

    if len(all_feats) < 10:
        return {"same_diff_gap": 0, "cross_camera_gap": 0, "pairwise_mean": 0, "dup_ratio": 1.0}

    features = torch.stack([f["feature"] for f in all_feats])
    n = min(500, len(features))
    cos_matrix = (features[:n] @ features[:n].T).numpy()
    np.fill_diagonal(cos_matrix, np.nan)
    valid_cosines = cos_matrix[~np.isnan(cos_matrix)]

    id_to_feats = defaultdict(list)
    for f in all_feats:
        id_to_feats[f["train_id"]].append(f)

    same_cos, diff_cos, cross_same, cross_diff = [], [], [], []
    ids_list = list(id_to_feats.keys())
    for i, id_a in enumerate(ids_list):
        entries_a = id_to_feats[id_a]
        for j in range(1, len(entries_a)):
            cos = F.cosine_similarity(entries_a[0]["feature"].unsqueeze(0), entries_a[j]["feature"].unsqueeze(0)).item()
            same_cos.append(cos)
            if entries_a[0]["cam_id"] != entries_a[j]["cam_id"]:
                cross_same.append(cos)
        for id_b in ids_list[i + 1:]:
            for ea in entries_a:
                for eb in id_to_feats[id_b]:
                    cos = F.cosine_similarity(ea["feature"].unsqueeze(0), eb["feature"].unsqueeze(0)).item()
                    diff_cos.append(cos)
                    if ea["cam_id"] != eb["cam_id"]:
                        cross_diff.append(cos)

    same_mean = np.mean(same_cos) if same_cos else 0
    diff_mean = np.mean(diff_cos) if diff_cos else 0
    cross_same_mean = np.mean(cross_same) if cross_same else 0
    cross_diff_mean = np.mean(cross_diff) if cross_diff else 0
    dup_ratio = float(np.mean(valid_cosines > 0.999))

    # Teacher cosine
    teacher_cosines = []
    for f in all_feats:
        if f["teacher_embedding"] is not None:
            teacher_cosines.append(F.cosine_similarity(f["feature"].unsqueeze(0), f["teacher_embedding"].unsqueeze(0)).item())
    teacher_cos_mean = np.mean(teacher_cosines) if teacher_cosines else 0

    # Retrieval
    rank1, rank5, total_q = 0, 0, 0
    for tid, entries in id_to_feats.items():
        if len(entries) < 2:
            continue
        query = entries[0]
        gallery = []
        for tid2, entries2 in id_to_feats.items():
            for e in entries2:
                if e["frame_id"] != query["frame_id"] or e["cam_id"] != query["cam_id"]:
                    gallery.append((e, tid2 == tid))
        if not gallery:
            continue
        sims = sorted([(F.cosine_similarity(query["feature"].unsqueeze(0), e["feature"].unsqueeze(0)).item(), is_same)
                        for e, is_same in gallery], key=lambda x: -x[0])
        rank1 += sims[0][1]
        rank5 += any(s[1] for s in sims[:min(5, len(sims))])
        total_q += 1

    result = {
        "pairwise_mean": float(np.nanmean(valid_cosines)),
        "pairwise_std": float(np.nanstd(valid_cosines)),
        "pairwise_min": float(np.nanmin(valid_cosines)),
        "pairwise_max": float(np.nanmax(valid_cosines)),
        "dup_ratio": dup_ratio,
        "same_id_cosine": float(same_mean),
        "diff_id_cosine": float(diff_mean),
        "same_diff_gap": float(same_mean - diff_mean),
        "cross_camera_same": float(cross_same_mean),
        "cross_camera_diff": float(cross_diff_mean),
        "cross_camera_gap": float(cross_same_mean - cross_diff_mean),
        "cross_camera_same_count": len(cross_same),
        "cross_camera_diff_count": len(cross_diff),
        "teacher_cosine_mean": float(teacher_cos_mean),
        "rank1": rank1 / max(1, total_q),
        "rank5": rank5 / max(1, total_q),
        "total_queries": total_q,
        "num_features": len(all_feats),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{prefix}_pairwise_cosine_stats.json"), "w") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(out_dir, f"{prefix}_identity_gap_eval.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["same_id_cosine", "diff_id_cosine", "gap", "cross_cam_same", "cross_cam_diff", "cross_gap"])
            w.writerow([same_mean, diff_mean, same_mean - diff_mean, cross_same_mean, cross_diff_mean, cross_same_mean - cross_diff_mean])
        with open(os.path.join(out_dir, f"{prefix}_cross_camera_eval.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cross_cam_same_mean", "cross_cam_diff_mean", "cross_cam_gap", "same_count", "diff_count"])
            w.writerow([cross_same_mean, cross_diff_mean, cross_same_mean - cross_diff_mean, len(cross_same), len(cross_diff)])

    print(f"  [eval] same={same_mean:.4f}, diff={diff_mean:.4f}, gap={same_mean-diff_mean:.4f}, "
          f"cross_gap={cross_same_mean-cross_diff_mean:.4f}, dup={dup_ratio:.4f}, "
          f"teacher_cos={teacher_cos_mean:.4f}, rank1={result['rank1']:.4f}, {time.time()-t0:.0f}s")
    return result


# ─── Anomaly Monitor ──────────────────────────────────────────────────────────
class AnomalyMonitor:
    def __init__(self, out_dir, max_zero_grad=10, max_dup_ratio=0.9):
        self.out_dir = out_dir
        self.max_zero_grad = max_zero_grad
        self.max_dup_ratio = max_dup_ratio
        self.zero_grad_count = 0
        self.recent_metrics = []

    def check(self, step, metrics):
        self.recent_metrics.append(metrics)
        if len(self.recent_metrics) > 50:
            self.recent_metrics = self.recent_metrics[-50:]

        failures = []
        if np.isnan(metrics.get("loss_total", 0)) or np.isinf(metrics.get("loss_total", 0)):
            failures.append(f"loss_total is NaN/Inf: {metrics.get('loss_total')}")
        if metrics.get("grad_pf", 0) == 0:
            self.zero_grad_count += 1
            if self.zero_grad_count >= self.max_zero_grad:
                failures.append(f"grad_pf=0 for {self.zero_grad_count} consecutive steps")
        else:
            self.zero_grad_count = 0
        if metrics.get("valid_roi_count", 1) < 2:
            failures.append(f"valid_roi_count too low: {metrics.get('valid_roi_count')}")
        if metrics.get("dup_ratio", 0) > self.max_dup_ratio:
            failures.append(f"dup_ratio > {self.max_dup_ratio}: {metrics.get('dup_ratio'):.4f}")

        if failures:
            self.write_failure(step, failures)
            return True
        return False

    def write_failure(self, step, reasons):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "failure_report.md"), "w") as f:
            f.write(f"# Failure Report\n\n## Step: {step}\n\n## Reasons\n\n")
            for r in reasons:
                f.write(f"- {r}\n")
            f.write(f"\n## Recent 50 Metrics\n\n```\n")
            for m in self.recent_metrics:
                f.write(json.dumps(m) + "\n")
            f.write("```\n\n## Possible Causes\n\n")
            f.write("- Check ROI pooling output\n- Check teacher embedding availability\n")
            f.write("- Check learning rate\n- Check feature collapse\n")
            f.write("- Check CUDA memory\n")


# ─── Main Training Loop ───────────────────────────────────────────────────────
def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Save run command
    with open(os.path.join(args.output_dir, "run_command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    with open(os.path.join(args.output_dir, "running_job.pid"), "w") as f:
        f.write(str(os.getpid()) + "\n")

    # Load model
    print("=" * 60)
    print(f"Phase 17: Teacher+CE Training")
    print(f"  init_ckpt: {args.init_ckpt}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  num_ids: {args.num_ids}")
    print(f"  steps: {args.steps}")
    print("=" * 60)

    model, conf, state = setup_model(args.init_ckpt, device)
    initial_positions = model.positions.detach().clone()
    print(f"  positions: {model.positions.shape}, _person_feature: {model._person_feature.shape}")
    print(f"  person_feature requires_grad: {model._person_feature.requires_grad}")

    # Load dataset
    train_ds, _ = make_dataset("wildtrack", conf, ray_jitter=None)
    print(f"  dataset size: {len(train_ds)}")

    # Collect samples
    print("\n[1/4] Collecting samples...")
    all_samples = collect_samples(model, train_ds, device, max_frames=400)

    # Select IDs
    if args.num_ids > 0 and args.num_ids < len(set(s["train_id"] for s in all_samples)):
        selected_ids, class_map = select_ids(all_samples, num_ids=args.num_ids, min_samples=args.min_samples)
    else:
        all_ids = sorted(set(s["train_id"] for s in all_samples))
        class_map = {tid: i for i, tid in enumerate(all_ids)}
        selected_ids = all_ids
    num_classes = len(class_map)
    print(f"  Selected {len(selected_ids)} IDs, {num_classes} classes")

    # Prepare samples
    samples_by_id = prepare_samples(all_samples, class_map, max_per_id=args.max_per_id)
    total_samples = sum(len(v) for v in samples_by_id.values())
    print(f"  Total training samples: {total_samples}")

    # Save subset info
    with open(os.path.join(args.output_dir, "selected_ids.json"), "w") as f:
        json.dump({"ids": selected_ids, "class_map": {str(k): v for k, v in class_map.items()}, "num_classes": num_classes}, f, indent=2)

    # Setup training
    print("\n[2/4] Setting up training...")
    model.train()
    for name, param in model.named_parameters():
        if "person_feature" not in name:
            param.requires_grad = False

    head = ReIDHead(512, num_classes, neck_dim=256, use_layernorm=True).to(device)
    optimizer = torch.optim.Adam([
        {"params": model._person_feature, "lr": args.lr_pf},
        {"params": head.parameters(), "lr": args.lr_head},
    ])

    sampler = PxKSampler(samples_by_id, P=args.P, K=args.K, seed=RANDOM_SEED)
    monitor = AnomalyMonitor(args.output_dir)

    initial_pf = model._person_feature.detach().clone()

    # Training loop
    print(f"\n[3/4] Training {args.steps} steps...")
    torch.cuda.empty_cache()
    log_path = os.path.join(args.output_dir, "train.log")
    log_f = open(log_path, "w")
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    metrics_f = open(metrics_path, "w")

    all_metrics = []
    t_start = time.time()
    grad_accum_steps = 0

    for step in range(args.steps):
        t_step = time.time()
        batch_samples = sampler.sample_batch()

        # Group by frame to minimize renders
        frame_groups = defaultdict(list)
        for s in batch_samples:
            frame_groups[s["idx"]].append(s)

        loss_teacher_total = 0.0
        loss_ce_total = 0.0
        cos_to_teacher_list = []
        feat_norms = []
        valid_roi = 0
        missing_teacher = 0
        ce_correct = 0
        ce_total = 0
        n_frames = len(frame_groups)

        for idx, items in frame_groups.items():
            batch = train_ds[idx % len(train_ds)]
            cam_id = batch.get("camera_id", "unknown")
            gpu_batch = train_ds.get_gpu_batch_with_intrinsics(batch)
            pf_map = model(gpu_batch, train=True, frame_id=step, render_person_feature=True).get("person_feature_map")
            if pf_map is None:
                continue
            _, h, w = pf_map.shape

            # Build instance lookup for teacher embeddings
            inst_by_train_id = {}
            for inst in gpu_batch.instances:
                if inst.get("valid", False) and inst.get("teacher_embedding") is not None:
                    inst_by_train_id[inst.get("train_id")] = inst

            features = []
            labels = []
            teacher_embs = []

            for s in items:
                bbox_orig = s.get("bbox_xyxy_original")
                if bbox_orig is None:
                    continue
                bbox_r = scale_bbox_to_render(bbox_orig, src_w=IMG_W, src_h=IMG_H, dst_w=w, dst_h=h)
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

                # Get teacher embedding from instance lookup
                te = inst_by_train_id.get(s["train_id"], {}).get("teacher_embedding")
                if te is None:
                    te = s.get("teacher_embedding")
                    if te is not None:
                        te = te.to(device) if isinstance(te, torch.Tensor) else torch.tensor(te, dtype=torch.float32, device=device)
                    else:
                        missing_teacher += 1
                        continue

                if isinstance(te, np.ndarray):
                    te = torch.tensor(te, dtype=torch.float32, device=device)
                elif isinstance(te, torch.Tensor):
                    te = te.to(device).float()

                features.append(f_v)
                labels.append(s["class_id"])
                teacher_embs.append(te)
                feat_norms.append(f_v.norm().item())
                valid_roi += 1

            if len(features) == 0:
                del pf_map
                continue

            features = torch.stack(features, dim=0)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)
            teacher_stack = torch.stack(teacher_embs, dim=0)

            # Teacher loss
            student_normed = F.normalize(features, p=2, dim=-1)
            teacher_normed = F.normalize(teacher_stack, p=2, dim=-1)
            cos_sim = F.cosine_similarity(student_normed, teacher_normed, dim=-1)
            loss_teacher = (1 - cos_sim).mean()
            loss_teacher_total += loss_teacher.item()
            cos_to_teacher_list.extend(cos_sim.detach().cpu().tolist())

            # CE loss
            logits, neck_feat = head(student_normed)
            loss_ce = F.cross_entropy(logits, labels_t)
            loss_ce_total += loss_ce.item()

            pred = logits.argmax(dim=-1)
            ce_correct += (pred == labels_t).sum().item()
            ce_total += len(labels_t)

            # Per-frame backward with gradient accumulation
            frame_loss = args.lambda_teacher * loss_teacher + args.lambda_ce * loss_ce
            frame_loss = frame_loss / n_frames  # normalize by number of frames
            frame_loss.backward()
            grad_accum_steps += 1

            del pf_map, features, student_normed, teacher_normed, cos_sim, loss_teacher, loss_ce, frame_loss

        if grad_accum_steps == 0:
            continue

        # Grad clipping and optimizer step
        torch.nn.utils.clip_grad_norm_([model._person_feature], max_norm=args.grad_clip)
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip)

        grad_pf = model._person_feature.grad.norm().item() if model._person_feature.grad is not None else 0.0
        grad_head = sum(p.grad.norm().item() for p in head.parameters() if p.grad is not None)

        optimizer.step()
        optimizer.zero_grad()
        grad_accum_steps = 0
        torch.cuda.empty_cache()

        # Metrics
        cos_teacher_mean = np.mean(cos_to_teacher_list) if cos_to_teacher_list else 0
        ce_acc = ce_correct / max(1, ce_total)
        pf_delta = (model._person_feature - initial_pf).norm().item()
        feat_norm_mean = np.mean(feat_norms) if feat_norms else 0
        geo_delta = (model.positions - initial_positions).norm().item()

        metrics = {
            "step": step,
            "loss_total": (loss_teacher_total * args.lambda_teacher + loss_ce_total * args.lambda_ce) / max(1, n_frames),
            "loss_teacher": loss_teacher_total / max(1, n_frames),
            "loss_ce": loss_ce_total / max(1, n_frames),
            "cosine_to_teacher": cos_teacher_mean,
            "ce_acc_top1": ce_acc,
            "grad_pf": grad_pf,
            "grad_head": grad_head,
            "valid_roi_count": valid_roi,
            "missing_teacher_count": missing_teacher,
            "feature_norm_mean": feat_norm_mean,
            "person_feature_delta": pf_delta,
            "geometry_delta": geo_delta,
            "cuda_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            "seconds_per_iter": time.time() - t_step,
        }

        all_metrics.append(metrics)
        metrics_f.write(json.dumps(metrics) + "\n")
        metrics_f.flush()

        # Anomaly check
        if monitor.check(step, metrics):
            print(f"  [ANOMALY] Training stopped at step {step}")
            break

        # Logging
        if step % args.log_interval == 0 or step == args.steps - 1:
            msg = (f"[step {step}] loss={metrics['loss_total']:.4f} "
                   f"(t={metrics['loss_teacher']:.4f} ce={metrics['loss_ce']:.4f}) "
                   f"cos_t={metrics['cosine_to_teacher']:.4f} "
                   f"acc={metrics['ce_acc_top1']:.4f} "
                   f"grad_pf={metrics['grad_pf']:.6f} "
                   f"roi={metrics['valid_roi_count']} "
                   f"pf_delta={metrics['person_feature_delta']:.4f} "
                   f"time={metrics['seconds_per_iter']:.1f}s")
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

        # Eval
        if step > 0 and step % args.eval_interval == 0:
            print(f"\n  [eval at step {step}]")
            eval_result = evaluate(model, train_ds, device, class_map, head, args.output_dir, prefix=f"step{step}")
            metrics.update(eval_result)
            model.train()

        # Save checkpoints
        if step > 0 and step % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
            torch.save({
                "_person_feature": model._person_feature.detach().cpu(),
                "head": head.state_dict(),
                "class_map": {str(k): v for k, v in class_map.items()},
                "global_step": step,
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    # Final eval
    print("\n[4/4] Final evaluation...")
    final_eval = evaluate(model, train_ds, device, class_map, head, args.output_dir, prefix="final")

    # Save final checkpoint
    torch.save({
        "_person_feature": model._person_feature.detach().cpu(),
        "head": head.state_dict(),
        "class_map": {str(k): v for k, v in class_map.items()},
        "global_step": args.steps,
    }, os.path.join(args.output_dir, "checkpoint_final.pt"))

    # Write CSV outputs
    with open(os.path.join(args.output_dir, "loss_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss_total", "loss_teacher", "loss_ce"])
        w.writeheader()
        w.writerows([{"step": m["step"], "loss_total": m["loss_total"], "loss_teacher": m["loss_teacher"], "loss_ce": m["loss_ce"]} for m in all_metrics])
    with open(os.path.join(args.output_dir, "teacher_cosine_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "cosine_to_teacher"])
        w.writeheader()
        w.writerows([{"step": m["step"], "cosine_to_teacher": m["cosine_to_teacher"]} for m in all_metrics])
    with open(os.path.join(args.output_dir, "ce_acc_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "ce_acc_top1"])
        w.writeheader()
        w.writerows([{"step": m["step"], "ce_acc_top1": m["ce_acc_top1"]} for m in all_metrics])

    # Final report
    init_loss = all_metrics[0]["loss_total"] if all_metrics else 0
    final_loss = all_metrics[-1]["loss_total"] if all_metrics else 0
    init_acc = all_metrics[0]["ce_acc_top1"] if all_metrics else 0
    final_acc = all_metrics[-1]["ce_acc_top1"] if all_metrics else 0
    final_grad_pf = all_metrics[-1]["grad_pf"] if all_metrics else 0
    total_time = time.time() - t_start

    passed = (
        final_acc > 0.3
        and final_eval.get("same_diff_gap", 0) > 0.01
        and final_eval.get("dup_ratio", 1) < 0.5
        and final_grad_pf > 0
        and final_eval.get("cross_camera_gap", 0) != 0
    )

    with open(os.path.join(args.output_dir, "final_report.md"), "w") as f:
        f.write(f"# Phase 17: Teacher+CE Training Report\n\n")
        f.write(f"## Config\n\n")
        f.write(f"- init_ckpt: `{args.init_ckpt}`\n")
        f.write(f"- num_ids: {num_classes}\n")
        f.write(f"- total_samples: {total_samples}\n")
        f.write(f"- steps: {args.steps}\n")
        f.write(f"- P×K: {args.P}×{args.K}\n")
        f.write(f"- lambda_teacher: {args.lambda_teacher}\n")
        f.write(f"- lambda_ce: {args.lambda_ce}\n")
        f.write(f"- lr_pf: {args.lr_pf}\n")
        f.write(f"- lr_head: {args.lr_head}\n")
        f.write(f"- grad_clip: {args.grad_clip}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"- Initial loss: {init_loss:.4f}\n")
        f.write(f"- Final loss: {final_loss:.4f}\n")
        f.write(f"- Initial acc: {init_acc:.4f}\n")
        f.write(f"- Final acc: {final_acc:.4f}\n")
        f.write(f"- Final grad_pf: {final_grad_pf:.6f}\n")
        f.write(f"- Total time: {total_time:.0f}s ({total_time/3600:.1f}h)\n\n")
        f.write(f"## Identity Eval\n\n")
        f.write(f"- same_id_cosine: {final_eval.get('same_id_cosine', 0):.4f}\n")
        f.write(f"- diff_id_cosine: {final_eval.get('diff_id_cosine', 0):.4f}\n")
        f.write(f"- same_diff_gap: {final_eval.get('same_diff_gap', 0):.4f}\n")
        f.write(f"- cross_camera_gap: {final_eval.get('cross_camera_gap', 0):.4f}\n")
        f.write(f"- dup_ratio: {final_eval.get('dup_ratio', 1):.4f}\n")
        f.write(f"- teacher_cosine: {final_eval.get('teacher_cosine_mean', 0):.4f}\n")
        f.write(f"- rank1: {final_eval.get('rank1', 0):.4f}\n")
        f.write(f"- rank5: {final_eval.get('rank5', 0):.4f}\n\n")
        f.write(f"## Decision: {'PASS' if passed else 'FAIL'}\n")

    log_f.close()
    metrics_f.close()

    print(f"\n{'='*60}")
    print(f"Training complete. PASS={passed}")
    print(f"  Final acc: {final_acc:.4f}, gap: {final_eval.get('same_diff_gap', 0):.4f}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"{'='*60}")

    return passed, final_eval


# ─── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_ids", type=int, default=30)
    parser.add_argument("--min_samples", type=int, default=20)
    parser.add_argument("--max_per_id", type=int, default=50)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--P", type=int, default=8, help="identities per batch")
    parser.add_argument("--K", type=int, default=4, help="samples per identity")
    parser.add_argument("--lambda_teacher", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=0.5)
    parser.add_argument("--lr_pf", type=float, default=0.005)
    parser.add_argument("--lr_head", type=float, default=0.003)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.environ["TORCH_EXTENSIONS_DIR"] = "/data02/zhangrunxiang/.cache/torch_extensions/py311_cu118"
    train(args)


if __name__ == "__main__":
    main()
