#!/usr/bin/env python3
"""Phase 16: CE Small Overfit - Identity Discrimination Validation."""

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

INIT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt"
REID_INIT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase14_clean_geometry/full_soft_reset_30k/reid_init/reid_init_ckpt.pt"
DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
OUTPUT_DIR = "/data02/zhangrunxiang/3dgrut/outputs/phase16_ce_small_overfit"
DEVICE = "cuda"
RANDOM_SEED = 42
NUM_IDS = 30
MIN_SAMPLES_PER_ID = 5


def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def setup_model(state, device):
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
    for name, param in model.named_parameters():
        if "person_feature" not in name:
            param.requires_grad = False
    assert model.positions.shape[0] == model._person_feature.shape[0]
    assert model._person_feature.shape[1] == 512
    return model, conf


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.head(x)


def collect_valid_samples(model, dataset, device, max_frames=200):
    model.eval()
    all_samples = []
    t0 = time.time()
    with torch.no_grad():
        for idx in range(min(len(dataset), max_frames)):
            if idx % 20 == 0:
                print(f"  [collect] frame {idx}/{min(len(dataset), max_frames)}, samples={len(all_samples)}, elapsed={time.time()-t0:.1f}s")
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
                bbox_orig = inst.get("bbox_xyxy_original")
                orig_w = inst.get("img_width_original", 1920)
                orig_h = inst.get("img_height_original", 1088)
                if bbox_orig is None:
                    continue
                bbox_r = scale_bbox_to_render(bbox_orig, src_w=orig_w, src_h=orig_h, dst_w=w, dst_h=h)
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
                f_v_cpu = f_v.detach().cpu().squeeze(0)
                sample = {
                    "idx": idx, "cam_id": cam_id, "frame_id": int(frame_idx),
                    "train_id": int(train_id), "feature": f_v_cpu,
                    "feature_norm": f_v_cpu.norm().item(),
                    "bbox_w": x2 - x1, "bbox_h": y2 - y1,
                    "_bbox_orig": list(bbox_orig),
                }
                all_samples.append(sample)
    print(f"  [collect] done: {len(all_samples)} samples in {time.time()-t0:.1f}s")
    return all_samples


def select_ids(all_samples, num_ids=30, min_samples=5):
    id_counts = defaultdict(int)
    id_cams = defaultdict(set)
    for s in all_samples:
        id_counts[s["train_id"]] += 1
        id_cams[s["train_id"]].add(s["cam_id"])
    valid_ids = [tid for tid, cnt in id_counts.items() if cnt >= min_samples]
    valid_ids.sort(key=lambda tid: id_counts[tid], reverse=True)
    selected_ids = valid_ids[:num_ids]
    selected_samples = [s for s in all_samples if s["train_id"] in selected_ids]
    class_map = {tid: i for i, tid in enumerate(sorted(selected_ids))}
    return selected_ids, selected_samples, class_map


def save_eval_fix_reports(all_samples, selected_samples, class_map, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cam_counts = defaultdict(int)
    for s in all_samples:
        cam_counts[s["cam_id"]] += 1
    with open(os.path.join(out_dir, "cam_id_distribution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cam_id", "count"])
        for cam, cnt in sorted(cam_counts.items()):
            w.writerow([cam, cnt])
    id_counts = defaultdict(int)
    for s in all_samples:
        id_counts[s["train_id"]] += 1
    with open(os.path.join(out_dir, "id_distribution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_id", "count", "selected"])
        for tid, cnt in sorted(id_counts.items()):
            w.writerow([tid, cnt, tid in class_map])
    with open(os.path.join(out_dir, "identity_sample_debug.jsonl"), "w") as f:
        for s in all_samples[:200]:
            serializable = {k: v for k, v in s.items() if not isinstance(v, torch.Tensor)}
            f.write(json.dumps(serializable) + "\n")
    id_cams = defaultdict(set)
    for s in all_samples:
        id_cams[s["train_id"]].add(s["cam_id"])
    cross_cam_ids = sum(1 for tid in id_cams if len(id_cams[tid]) > 1)
    with open(os.path.join(out_dir, "pair_construction_report.md"), "w") as f:
        f.write(f"# Pair Construction Report\n\n")
        f.write(f"## cam_id Coverage\n\n")
        for cam, cnt in sorted(cam_counts.items()):
            f.write(f"- {cam}: {cnt} samples\n")
        f.write(f"\n## ID Stats\n\n")
        f.write(f"- Total unique IDs: {len(id_counts)}\n")
        f.write(f"- IDs with multi-camera samples: {cross_cam_ids}\n")
        f.write(f"- Selected IDs: {len(class_map)}\n")
        f.write(f"- Selected samples: {len(selected_samples)}\n")


def save_subset_stats(selected_ids, selected_samples, class_map, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "selected_ids.json"), "w") as f:
        json.dump({"ids": selected_ids, "class_map": {str(k): v for k, v in class_map.items()}}, f, indent=2)
    with open(os.path.join(out_dir, "selected_samples.jsonl"), "w") as f:
        for s in selected_samples:
            serializable = {k: v for k, v in s.items() if not isinstance(v, torch.Tensor)}
            f.write(json.dumps(serializable) + "\n")
    id_counts = defaultdict(int)
    id_cams = defaultdict(set)
    cam_counts = defaultdict(int)
    for s in selected_samples:
        id_counts[s["train_id"]] += 1
        id_cams[s["train_id"]].add(s["cam_id"])
        cam_counts[s["cam_id"]] += 1
    with open(os.path.join(out_dir, "subset_stats.md"), "w") as f:
        f.write(f"# CE Subset Stats\n\n")
        f.write(f"- ID count: {len(selected_ids)}\n")
        f.write(f"- Total samples: {len(selected_samples)}\n\n")
        f.write(f"## Per-ID Sample Count\n\n")
        for tid in sorted(selected_ids):
            f.write(f"- ID {tid}: {id_counts[tid]} samples, cameras: {sorted(id_cams[tid])}\n")
        f.write(f"\n## Per-Camera Distribution\n\n")
        for cam, cnt in sorted(cam_counts.items()):
            f.write(f"- {cam}: {cnt} samples\n")


def train_ce(model, dataset, device, selected_samples, class_map, num_classes, steps=500, log_interval=20, out_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    samples_by_id = defaultdict(list)
    for s in selected_samples:
        if s["train_id"] in class_map:
            samples_by_id[s["train_id"]].append(s)
    train_list = []
    for tid, samples in samples_by_id.items():
        for s in samples:
            train_list.append((s, class_map[tid]))
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    classifier = SimpleClassifier(512, num_classes).to(device)
    optimizer = torch.optim.Adam([
        {"params": model._person_feature, "lr": 0.001},
        {"params": classifier.parameters(), "lr": 0.01},
    ])
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "w")
    all_metrics = []
    all_acc = []
    per_cam_data = defaultdict(lambda: {"correct": 0, "total": 0})
    per_id_data = defaultdict(lambda: {"correct": 0, "total": 0})

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Starting Phase 16 CE small overfit: {steps} steps")
    log(f"Classes: {num_classes}, Samples: {len(train_list)}")

    for step in range(steps):
        batch_indices = np.random.choice(len(train_list), size=min(4, len(train_list)), replace=False)
        batch_samples = [train_list[i] for i in batch_indices]
        frame_groups = defaultdict(list)
        for sample, label in batch_samples:
            key = (sample["idx"], sample["cam_id"], sample["frame_id"])
            frame_groups[key].append((sample, label))
        loss_terms = []
        total = 0

        for (idx, cam_id, frame_id), items in frame_groups.items():
            batch = dataset[idx % len(dataset)]
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            pf_map = model(gpu_batch, train=True, frame_id=step, render_person_feature=True).get("person_feature_map")
            if pf_map is None:
                continue
            _, h, w = pf_map.shape
            features = []
            labels = []
            for sample, label in items:
                bbox_orig = sample.get("_bbox_orig")
                if bbox_orig is None:
                    continue
                bbox_r = scale_bbox_to_render(bbox_orig, src_w=1920, src_h=1088, dst_w=w, dst_h=h)
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
                features.append(f_v)
                labels.append(label)
            if len(features) == 0:
                continue
            features = torch.stack(features, dim=0)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)
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
                total += 1

        if not loss_terms:
            continue
        loss = torch.stack(loss_terms).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_pf = model._person_feature.grad.norm().item() if model._person_feature.grad is not None else 0.0
        grad_clf = sum(p.grad.norm().item() for p in classifier.parameters() if p.grad is not None)
        optimizer.step()
        avg_loss = loss.item()
        acc_vals = [d["correct"] / d["total"] for d in per_cam_data.values() if d["total"] > 0]
        acc = np.mean(acc_vals) if acc_vals else 0.0
        all_metrics.append({"step": step, "ce_loss": avg_loss, "train_acc_top1": acc, "grad_norm_person_feature": grad_pf, "grad_norm_classifier": grad_clf, "valid_roi": total})
        all_acc.append({"step": step, "acc": acc})
        if step % log_interval == 0 or step == steps - 1:
            log(f"Step {step}: ce_loss={avg_loss:.4f}, acc={acc:.4f}, roi={total}, grad_pf={grad_pf:.4f}")
        if step == 499:
            save_ckpt(model, classifier, os.path.join(out_dir, "checkpoint_500.pt"), 500)
            log("Saved checkpoint_500.pt")
        if step == 999:
            save_ckpt(model, classifier, os.path.join(out_dir, "checkpoint_1000.pt"), 1000)
            log("Saved checkpoint_1000.pt")

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
        w.writerows(all_acc)
    with open(os.path.join(out_dir, "per_camera_acc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "correct", "total", "accuracy"])
        for cam, data in sorted(per_cam_data.items()):
            a = data["correct"] / data["total"] if data["total"] > 0 else 0
            w.writerow([cam, data["correct"], data["total"], f"{a:.4f}"])
    with open(os.path.join(out_dir, "per_id_acc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_id", "correct", "total", "accuracy"])
        for tid, data in sorted(per_id_data.items()):
            a = data["correct"] / data["total"] if data["total"] > 0 else 0
            w.writerow([tid, data["correct"], data["total"], f"{a:.4f}"])
    log_f.close()
    return all_metrics


def save_ckpt(model, classifier, path, step):
    torch.save({
        "positions": model.positions.detach().cpu(), "density": model.density.detach().cpu(),
        "scale": model.scale.detach().cpu(), "rotation": model.rotation.detach().cpu(),
        "features_albedo": model.features_albedo.detach().cpu(),
        "features_specular": model.features_specular.detach().cpu(),
        "_person_feature": model._person_feature.detach().cpu(),
        "person_feature": model._person_feature.detach().cpu(),
        "classifier": classifier.state_dict(),
        "global_step": step, "scene_extent": model.scene_extent,
    }, path)


def identity_eval_after_ce(model, dataset, device, class_map, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    all_features = []
    with torch.no_grad():
        for idx in range(min(len(dataset), 500)):
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
                bbox_r = scale_bbox_to_render(bbox_orig, src_w=1920, src_h=1088, dst_w=w, dst_h=h)
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
                all_features.append({
                    "train_id": int(train_id), "cam_id": cam_id, "frame_id": int(frame_idx),
                    "feature": F.normalize(f_v.detach().cpu().squeeze(0), p=2, dim=-1),
                })
    if len(all_features) < 10:
        print("Too few features for eval")
        return {"same": 0, "diff": 0, "gap": 0}
    features = torch.stack([f["feature"] for f in all_features])
    n = min(200, len(features))
    cos_matrix = (features[:n] @ features[:n].T).numpy()
    np.fill_diagonal(cos_matrix, np.nan)
    valid_cosines = cos_matrix[~np.isnan(cos_matrix)]
    id_to_features = defaultdict(list)
    for f in all_features:
        id_to_features[f["train_id"]].append(f)
    same_cosines, diff_cosines, cross_cam_same, cross_cam_diff = [], [], [], []
    ids = list(id_to_features.keys())
    for i, id_a in enumerate(ids):
        entries_a = id_to_features[id_a]
        for j, id_b in enumerate(ids):
            if j <= i:
                continue
            for ea in entries_a:
                for eb in id_to_features[id_b]:
                    cos = F.cosine_similarity(ea["feature"].unsqueeze(0), eb["feature"].unsqueeze(0)).item()
                    diff_cosines.append(cos)
                    if ea["cam_id"] != eb["cam_id"]:
                        cross_cam_diff.append(cos)
        for idx_e in range(1, len(entries_a)):
            cos = F.cosine_similarity(entries_a[0]["feature"].unsqueeze(0), entries_a[idx_e]["feature"].unsqueeze(0)).item()
            same_cosines.append(cos)
            if entries_a[0]["cam_id"] != entries_a[idx_e]["cam_id"]:
                cross_cam_same.append(cos)
    same_mean = np.mean(same_cosines) if same_cosines else 0.0
    diff_mean = np.mean(diff_cosines) if diff_cosines else 0.0
    cross_same_mean = np.mean(cross_cam_same) if cross_cam_same else 0.0
    cross_diff_mean = np.mean(cross_cam_diff) if cross_cam_diff else 0.0
    with open(os.path.join(out_dir, "identity_diagnostic_after_ce.md"), "w") as f:
        f.write(f"# Identity Diagnostic After CE\n\n"
                f"## Pairwise Cosine\n\n- Mean: {np.nanmean(valid_cosines):.4f}\n- Std: {np.nanstd(valid_cosines):.4f}\n"
                f"- Min: {np.nanmin(valid_cosines):.4f}\n- Max: {np.nanmax(valid_cosines):.4f}\n"
                f"- Duplicate ratio (>0.999): {np.sum(valid_cosines > 0.999) / len(valid_cosines):.4f}\n\n"
                f"## Identity Gap\n\n- Same-id: {same_mean:.4f} ({len(same_cosines)} pairs)\n"
                f"- Diff-id: {diff_mean:.4f} ({len(diff_cosines)} pairs)\n- Gap: {same_mean - diff_mean:.4f}\n\n"
                f"## Cross-camera Gap\n\n- Cross same: {cross_same_mean:.4f} ({len(cross_cam_same)} pairs)\n"
                f"- Cross diff: {cross_diff_mean:.4f} ({len(cross_cam_diff)} pairs)\n- Gap: {cross_same_mean - cross_diff_mean:.4f}\n")
    with open(os.path.join(out_dir, "identity_eval_after_ce.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["same_id_cosine_mean", "diff_id_cosine_mean", "same_diff_gap", "same_pair_count", "diff_pair_count"])
        w.writerow([same_mean, diff_mean, same_mean - diff_mean, len(same_cosines), len(diff_cosines)])
    with open(os.path.join(out_dir, "cross_camera_eval_after_ce.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cross_cam_same_mean", "cross_cam_diff_mean", "cross_cam_gap", "cross_cam_same_count", "cross_cam_diff_count"])
        w.writerow([cross_same_mean, cross_diff_mean, cross_same_mean - cross_diff_mean, len(cross_cam_same), len(cross_cam_diff)])
    with open(os.path.join(out_dir, "pairwise_cosine_stats_after_ce.json"), "w") as f:
        json.dump({"pairwise_mean": float(np.nanmean(valid_cosines)), "pairwise_std": float(np.nanstd(valid_cosines)),
                   "pairwise_min": float(np.nanmin(valid_cosines)), "pairwise_max": float(np.nanmax(valid_cosines)),
                   "dup_ratio": float(np.sum(valid_cosines > 0.999) / len(valid_cosines)),
                   "same_id_cosine_mean": float(same_mean), "diff_id_cosine_mean": float(diff_mean),
                   "same_diff_gap": float(same_mean - diff_mean)}, f, indent=2)
    # Retrieval
    rank1_correct, rank5_correct, total_queries = 0, 0, 0
    for tid, entries in id_to_features.items():
        if len(entries) < 2:
            continue
        query = entries[0]
        gallery = []
        for tid2, entries2 in id_to_features.items():
            for e in entries2:
                if e["frame_id"] != query["frame_id"] or e["cam_id"] != query["cam_id"]:
                    gallery.append((e, tid2 == tid))
        if not gallery:
            continue
        sims = sorted([(F.cosine_similarity(query["feature"].unsqueeze(0), e["feature"].unsqueeze(0)).item(), is_same) for e, is_same in gallery], key=lambda x: -x[0])
        rank1_correct += sims[0][1]
        rank5_correct += any(s[1] for s in sims[:min(5, len(sims))])
        total_queries += 1
    with open(os.path.join(out_dir, "retrieval_eval_after_ce.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank1", "rank5", "total_queries"])
        w.writerow([rank1_correct / total_queries if total_queries > 0 else 0, rank5_correct / total_queries if total_queries > 0 else 0, total_queries])
    print(f"  [POST-CE EVAL] same={same_mean:.4f}, diff={diff_mean:.4f}, gap={same_mean - diff_mean:.4f}")
    print(f"  [POST-CE RETRIEVAL] rank1={rank1_correct / total_queries:.4f} ({total_queries} queries)")
    return {"same": same_mean, "diff": diff_mean, "gap": same_mean - diff_mean}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_ckpt", default=INIT_CKPT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num_ids", type=int, default=NUM_IDS)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()
    os.environ["TORCH_EXTENSIONS_DIR"] = "/data02/zhangrunxiang/.cache/torch_extensions/py311_cu118"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("\n=== Loading checkpoint ===")
    state = load_ckpt(args.init_ckpt)
    print(f"positions: {state['positions'].shape}")
    print(f"_person_feature: {state['_person_feature'].shape}")
    print("\n=== Setup model ===")
    model, conf = setup_model(state, device)
    print("\n=== Load dataset ===")
    train_ds, _ = make_dataset("wildtrack", conf, ray_jitter=None)
    print(f"Dataset size: {len(train_ds)}")
    print("\n=== Collect valid samples ===")
    all_samples = collect_valid_samples(model, train_ds, device)
    print(f"Collected {len(all_samples)} valid samples")
    print("\n=== Task 1: Eval Fix Reports ===")
    selected_ids, selected_samples, class_map = select_ids(all_samples, num_ids=args.num_ids)
    save_eval_fix_reports(all_samples, selected_samples, class_map, os.path.join(OUTPUT_DIR, "eval_fix"))
    print("\n=== Task 2: Select IDs ===")
    print(f"Selected {len(selected_ids)} IDs, {len(selected_samples)} samples")
    save_subset_stats(selected_ids, selected_samples, class_map, os.path.join(OUTPUT_DIR, "ce_subset"))
    print("\n=== Task 3: CE Training ===")
    num_classes = len(class_map)
    metrics = train_ce(model, train_ds, device, selected_samples, class_map, num_classes=num_classes, steps=args.steps, out_dir=OUTPUT_DIR)
    print("\n=== Task 5: Identity Eval After CE ===")
    eval_result = identity_eval_after_ce(model, train_ds, device, class_map, os.path.join(OUTPUT_DIR, "identity_eval"))
    print("\n=== Task 6: Final Report ===")
    init_loss = metrics[0]["ce_loss"] if metrics else 0
    final_loss = metrics[-1]["ce_loss"] if metrics else 0
    init_acc = metrics[0]["train_acc_top1"] if metrics else 0
    final_acc = metrics[-1]["train_acc_top1"] if metrics else 0
    if final_loss < init_loss and final_acc > 0.3:
        decision, dec_text = "A", "CE small overfit PASS. Loss decreased, accuracy improved, same/diff gap evaluated."
    elif final_loss < init_loss:
        decision, dec_text = "B", "CE loss decreased but accuracy low. Check label mapping / classifier."
    else:
        decision, dec_text = "D", "CE loss did not decrease. Check ROI features, learning rate."
    with open(os.path.join(OUTPUT_DIR, "final_report.md"), "w") as f:
        f.write(f"# Phase 16: CE Small Overfit - Final Report\n\n"
                f"## Init checkpoint: `{args.init_ckpt}`\n\n"
                f"## Config\n\n- Geometry N: {state['positions'].shape[0]}\n"
                f"- _person_feature: [{state['_person_feature'].shape[0]}, {state['_person_feature'].shape[1]}]\n"
                f"- Selected IDs: {len(selected_ids)}\n- Selected samples: {len(selected_samples)}\n"
                f"- Classes: {num_classes}\n- Steps: {args.steps}\n\n"
                f"## Training Results\n\n- Initial CE loss: {init_loss:.4f}\n- Final CE loss: {final_loss:.4f}\n"
                f"- Loss delta: {final_loss - init_loss:+.4f}\n\n- Initial accuracy: {init_acc:.4f}\n"
                f"- Final accuracy: {final_acc:.4f}\n\n"
                f"## Identity Gap After CE\n\n- Same-id cosine: {eval_result['same']:.4f}\n"
                f"- Diff-id cosine: {eval_result['diff']:.4f}\n- Gap: {eval_result['gap']:.4f}\n\n"
                f"## Decision: {decision}\n\n{dec_text}\n")
    print(f"Final report: {OUTPUT_DIR}/final_report.md")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
