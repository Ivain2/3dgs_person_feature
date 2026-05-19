#!/usr/bin/env python3
"""V2.1: Pure 2D ReID / Appearance Baseline (full/balanced evaluation).

Uses pre-extracted ClipReID teacher embeddings from reid_teacher_cache.
Evaluates cross-camera retrieval, pairwise separability, and association proxy.

NO 3D TRAINING. NO TRACKER CHANGES. NO BBOX SCALE CHANGES.

Coordinate protocol (frozen from V0):
  - annotation bbox: 1920x1080 original space
  - downsample_factor=4, padding 1080->1088
  - render size: WxH=480x272
  - scale_bbox_to_render: sx=sy=0.25
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.datasets.dataset_wildtrack import WildtrackDataset
from threedgrut.datasets.reid_teacher_cache import ReidTeacherCache
from threedgrut.datasets.cache_key import make_cache_key

DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"


def collect_all_detections(max_per_cam=-1, sample_mode="all", seed=42):
    teacher_cache = ReidTeacherCache(DATASET_PATH)
    if teacher_cache is None:
        print("[ERROR] Teacher cache not loaded!")
        return [], {}

    ann_dir = os.path.join(DATASET_PATH, "annotations_remapped")
    id_map_path = os.path.join(DATASET_PATH, "id_map.json")

    with open(id_map_path, "r") as f:
        id_map_data = json.load(f)
    raw_to_new = id_map_data.get("raw_to_new", id_map_data)

    cam_id_map = {0: "C1", 1: "C2", 2: "C3", 3: "C4", 4: "C5", 5: "C6", 6: "C7"}
    cam_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    all_candidates = []
    attempted = 0
    cache_hit = 0
    cache_miss = 0

    for ann_file in sorted(os.listdir(ann_dir)):
        if not ann_file.endswith(".json"):
            continue
        frame_id = int(ann_file.replace(".json", ""))

        with open(os.path.join(ann_dir, ann_file), "r") as f:
            ann = json.load(f)

        for inst in ann:
            cam_id = cam_id_map.get(inst.get("camera_id", -1))
            if cam_id is None:
                continue

            raw_id = inst.get("raw_id", -1)
            if str(raw_id) not in raw_to_new:
                continue

            train_id = raw_to_new[str(raw_id)]
            bbox_dict = inst.get("bbox", None)
            if bbox_dict is None:
                continue

            bbox_xyxy_original = [int(bbox_dict["xmin"]), int(bbox_dict["ymin"]),
                                  int(bbox_dict["xmax"]), int(bbox_dict["ymax"])]

            attempted += 1
            cache_key = make_cache_key(frame_id, cam_id, train_id, bbox_xyxy_original)
            cache_entry = teacher_cache.get(cache_key)

            if cache_entry is None:
                cache_miss += 1
                continue

            embedding = cache_entry.get("embedding")
            if embedding is None:
                cache_miss += 1
                continue

            cache_hit += 1

            if isinstance(embedding, np.ndarray):
                emb = embedding.astype(np.float32)
            else:
                emb = np.array(embedding, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)

            bbox_render = [int(v / 4) for v in bbox_xyxy_original]
            roi_w = bbox_render[2] - bbox_render[0]
            roi_h = bbox_render[3] - bbox_render[1]
            border_flag = (bbox_render[0] <= 2 or bbox_render[1] <= 2 or
                           bbox_render[2] >= 478 or bbox_render[3] >= 270)
            small_bbox = (roi_w < 10 or roi_h < 20)

            all_candidates.append({
                "camera_id": cam_id,
                "frame_id": frame_id,
                "person_id": int(raw_id),
                "train_id": train_id,
                "bbox_xyxy_original": bbox_xyxy_original,
                "bbox_render": bbox_render,
                "crop_w": bbox_xyxy_original[2] - bbox_xyxy_original[0],
                "crop_h": bbox_xyxy_original[3] - bbox_xyxy_original[1],
                "border_flag": border_flag,
                "small_bbox": small_bbox,
                "embedding": emb,
            })

    cache_stats = {
        "attempted": attempted,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "cache_hit_ratio": cache_hit / max(1, attempted),
    }

    # Sampling
    if sample_mode == "all":
        detections = all_candidates
    elif sample_mode == "balanced":
        rng = np.random.RandomState(seed)
        # Group by (camera_id, person_id)
        cam_pid_dets = defaultdict(list)
        for d in all_candidates:
            cam_pid_dets[(d["camera_id"], d["person_id"])].append(d)

        # Target: equal samples per (cam, pid), limited by min count
        # Also limit total per camera
        per_cam_limit = max_per_cam if max_per_cam > 0 else 999999
        # Find min samples per (cam, pid) that have >= 1
        counts = [len(v) for v in cam_pid_dets.values() if len(v) >= 1]
        target_per_cam_pid = min(counts) if counts else 1
        target_per_cam_pid = max(1, min(target_per_cam_pid, 20))

        detections = []
        cam_count = defaultdict(int)
        for (cam, pid), dets in sorted(cam_pid_dets.items()):
            if cam_count[cam] >= per_cam_limit:
                continue
            rng.shuffle(dets)
            take = min(target_per_cam_pid, len(dets), per_cam_limit - cam_count[cam])
            for d in dets[:take]:
                detections.append(d)
                cam_count[cam] += 1
    elif sample_mode == "random":
        rng = np.random.RandomState(seed)
        rng.shuffle(all_candidates)
        per_cam_limit = max_per_cam if max_per_cam > 0 else len(all_candidates)
        cam_count = defaultdict(int)
        detections = []
        for d in all_candidates:
            if cam_count[d["camera_id"]] >= per_cam_limit:
                continue
            detections.append(d)
            cam_count[d["camera_id"]] += 1
    else:  # "first"
        per_cam_limit = max_per_cam if max_per_cam > 0 else len(all_candidates)
        cam_count = defaultdict(int)
        detections = []
        for d in all_candidates:
            if cam_count[d["camera_id"]] >= per_cam_limit:
                continue
            detections.append(d)
            cam_count[d["camera_id"]] += 1

    return detections, cache_stats


def evaluate_retrieval(detections, out_dir):
    print("\n[Eval 1] Cross-camera retrieval...")

    emb_matrix = np.stack([d["embedding"] for d in detections])
    cam_ids = [d["camera_id"] for d in detections]
    person_ids = [d["person_id"] for d in detections]
    n = len(detections)

    # Chunked similarity for large n
    chunk_size = 5000
    sim_full = np.zeros((n, n), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim_full[start:end] = emb_matrix[start:end] @ emb_matrix.T

    # Build camera index for fast gallery lookup
    cam_indices = defaultdict(list)
    for i, c in enumerate(cam_ids):
        cam_indices[c].append(i)

    ranks = []
    aps = []
    skipped = 0
    per_cam_ranks = defaultdict(list)
    per_cam_aps = defaultdict(list)
    per_pair_ranks = defaultdict(list)
    per_pair_aps = defaultdict(list)
    per_pair_counts = defaultdict(int)
    border_ranks, border_aps = [], []
    nonborder_ranks, nonborder_aps = [], []
    small_ranks, small_aps = [], []

    for i in range(n):
        q_cam = cam_ids[i]
        q_pid = person_ids[i]

        gallery_indices = [j for c in cam_indices if c != q_cam for j in cam_indices[c]]
        positives = [j for j in gallery_indices if person_ids[j] == q_pid]
        if not positives:
            skipped += 1
            continue

        sims = sim_full[i, gallery_indices]
        sorted_idx = np.argsort(-sims)
        sorted_pids = [person_ids[gallery_indices[j]] for j in sorted_idx]

        rank = None
        for r_idx, pid in enumerate(sorted_pids):
            if pid == q_pid:
                rank = r_idx + 1
                break

        if rank is not None:
            ranks.append(rank)
            per_cam_ranks[q_cam].append(rank)
            g_idx = gallery_indices[sorted_idx[0]] if rank == 1 else gallery_indices[sorted_idx[rank - 1]]
            g_cam = cam_ids[g_idx]
            per_pair_ranks[(q_cam, g_cam)].append(rank)
            per_pair_counts[(q_cam, g_cam)] += 1
            if detections[i]["border_flag"]:
                border_ranks.append(rank)
            else:
                nonborder_ranks.append(rank)
            if detections[i]["small_bbox"]:
                small_ranks.append(rank)

        # AP
        hits = 0
        precision_sum = 0.0
        for r_idx, j in enumerate(sorted_idx):
            if person_ids[gallery_indices[j]] == q_pid:
                hits += 1
                precision_sum += hits / (r_idx + 1)
        ap = precision_sum / max(1, len(positives))
        aps.append(ap)
        per_cam_aps[q_cam].append(ap)
        per_pair_aps[(q_cam, g_cam if rank else q_cam)].append(ap)
        if detections[i]["border_flag"]:
            border_aps.append(ap)
        else:
            nonborder_aps.append(ap)
        if detections[i]["small_bbox"]:
            small_aps.append(ap)

    rank1 = np.mean([1 if r <= 1 else 0 for r in ranks]) if ranks else 0
    rank5 = np.mean([1 if r <= 5 else 0 for r in ranks]) if ranks else 0
    rank10 = np.mean([1 if r <= 10 else 0 for r in ranks]) if ranks else 0
    mAP = np.mean(aps) if aps else 0

    print(f"  Overall: mAP={mAP:.4f}, Rank-1={rank1:.4f}, Rank-5={rank5:.4f}, Rank-10={rank10:.4f}")
    print(f"  Skipped: {skipped}, Border Rank-1: {np.mean([1 if r<=1 else 0 for r in border_ranks]) if border_ranks else 0:.4f}, Non-border Rank-1: {np.mean([1 if r<=1 else 0 for r in nonborder_ranks]) if nonborder_ranks else 0:.4f}")

    overall_metrics = {
        "mAP": float(mAP), "rank1": float(rank1), "rank5": float(rank5), "rank10": float(rank10),
        "num_queries": len(ranks), "skipped_queries": skipped,
        "border_rank1": float(np.mean([1 if r<=1 else 0 for r in border_ranks])) if border_ranks else None,
        "nonborder_rank1": float(np.mean([1 if r<=1 else 0 for r in nonborder_ranks])) if nonborder_ranks else None,
        "small_bbox_rank1": float(np.mean([1 if r<=1 else 0 for r in small_ranks])) if small_ranks else None,
        "border_count": len(border_ranks), "nonborder_count": len(nonborder_ranks),
    }
    with open(os.path.join(out_dir, "retrieval_metrics.json"), "w") as f:
        json.dump(overall_metrics, f, indent=2)

    with open(os.path.join(out_dir, "retrieval_by_cam.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "count", "mAP", "rank1", "rank5", "rank10"])
        for cam in sorted(per_cam_ranks.keys()):
            cr = per_cam_ranks[cam]
            ca = per_cam_aps[cam]
            w.writerow([cam, len(cr), f"{np.mean(ca):.4f}",
                        f"{np.mean([1 if r<=1 else 0 for r in cr]):.4f}",
                        f"{np.mean([1 if r<=5 else 0 for r in cr]):.4f}",
                        f"{np.mean([1 if r<=10 else 0 for r in cr]):.4f}"])

    with open(os.path.join(out_dir, "retrieval_by_cam_pair.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_cam", "gallery_cam", "count", "mAP", "rank1", "rank5", "low_stat"])
        for (qc, gc) in sorted(per_pair_ranks.keys()):
            pr = per_pair_ranks[(qc, gc)]
            pa = per_pair_aps.get((qc, gc), [0])
            cnt = per_pair_counts.get((qc, gc), 0)
            low_stat = "YES" if cnt < 50 else "NO"
            w.writerow([qc, gc, len(pr), f"{np.mean(pa):.4f}",
                        f"{np.mean([1 if r<=1 else 0 for r in pr]):.4f}",
                        f"{np.mean([1 if r<=5 else 0 for r in pr]):.4f}",
                        low_stat])

    return overall_metrics


def evaluate_pairwise(detections, out_dir, max_pos=1000000, max_neg=1000000):
    print("\n[Eval 2] Pairwise separability...")

    emb_matrix = np.stack([d["embedding"] for d in detections])
    person_ids = np.array([d["person_id"] for d in detections])
    cam_ids_list = [d["camera_id"] for d in detections]

    pid_indices = defaultdict(list)
    for i, d in enumerate(detections):
        pid_indices[d["person_id"]].append(i)

    pos_sims = []
    for pid, indices in pid_indices.items():
        cams = defaultdict(list)
        for i in indices:
            cams[cam_ids_list[i]].append(i)
        cam_list = list(cams.keys())
        for ci in range(len(cam_list)):
            for cj in range(ci + 1, len(cam_list)):
                for i in cams[cam_list[ci]][:10]:
                    for j in cams[cam_list[cj]][:10]:
                        pos_sims.append(float(np.dot(emb_matrix[i], emb_matrix[j])))
                        if len(pos_sims) >= max_pos:
                            break
                    if len(pos_sims) >= max_pos:
                        break
                if len(pos_sims) >= max_pos:
                    break
            if len(pos_sims) >= max_pos:
                break

    # Negative pairs via random sampling
    rng = np.random.RandomState(42)
    n = len(detections)
    neg_sims = []
    attempts = 0
    while len(neg_sims) < min(len(pos_sims) * 2, max_neg) and attempts < max_neg * 10:
        i, j = rng.randint(0, n, 2)
        if i != j and person_ids[i] != person_ids[j] and cam_ids_list[i] != cam_ids_list[j]:
            neg_sims.append(float(np.dot(emb_matrix[i], emb_matrix[j])))
        attempts += 1

    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)

    print(f"  Positive: {len(pos_sims)}, Negative: {len(neg_sims)}")
    print(f"  Pos mean={pos_sims.mean():.4f}, Neg mean={neg_sims.mean():.4f}, Gap={pos_sims.mean()-neg_sims.mean():.4f}")

    labels = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
    scores = np.concatenate([pos_sims, neg_sims])

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
        fpr, tpr, thresholds_roc = roc_curve(labels, scores)
        precision, recall, thresholds_pr = precision_recall_curve(labels, scores)
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = float(fpr[eer_idx])
        eer_threshold = float(thresholds_roc[eer_idx])
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = float(f1_scores[best_f1_idx])
        best_f1_threshold = float(thresholds_pr[min(best_f1_idx, len(thresholds_pr) - 1)])
        print(f"  ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, EER={eer:.4f}, BestF1={best_f1:.4f}")

        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(pos_sims, bins=50, alpha=0.6, label=f"Pos (mean={pos_sims.mean():.3f})", density=True)
        ax.hist(neg_sims, bins=50, alpha=0.6, label=f"Neg (mean={neg_sims.mean():.3f})", density=True)
        ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Density")
        ax.set_title("Positive vs Negative Cosine Similarity"); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "pos_neg_similarity_hist.png"), dpi=100); plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})"); ax.plot([0,1],[0,1],"k--")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=100); plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(recall, precision, label=f"PR (AUC={pr_auc:.4f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curve"); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "pr_curve.png"), dpi=100); plt.close()
    except ImportError:
        roc_auc = pr_auc = eer = eer_threshold = best_f1 = best_f1_threshold = None

    pairwise_metrics = {
        "num_positive_pairs": len(pos_sims), "num_negative_pairs": len(neg_sims),
        "pos_cosine_mean": float(pos_sims.mean()), "pos_cosine_std": float(pos_sims.std()),
        "neg_cosine_mean": float(neg_sims.mean()), "neg_cosine_std": float(neg_sims.std()),
        "pos_neg_gap": float(pos_sims.mean() - neg_sims.mean()),
        "roc_auc": roc_auc, "pr_auc": pr_auc, "eer": eer, "eer_threshold": eer_threshold,
        "best_f1": best_f1, "best_f1_threshold": best_f1_threshold,
    }
    with open(os.path.join(out_dir, "pairwise_metrics.json"), "w") as f:
        json.dump(pairwise_metrics, f, indent=2)

    return pairwise_metrics


def evaluate_association_proxy(detections, out_dir):
    print("\n[Eval 3] Cross-camera association proxy...")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("  [WARN] scipy not available, skipping")
        return None

    frame_dets = defaultdict(list)
    for d in detections:
        frame_dets[d["frame_id"]].append(d)

    cam_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    cam_pairs = [(ci, cj) for ci in cam_names for cj in cam_names if ci < cj]
    thresholds = np.arange(0.0, 1.0, 0.05)
    pair_results = defaultdict(lambda: {"tp": [], "fp": [], "fn": []})

    for frame_id, dets in frame_dets.items():
        cam_dets = defaultdict(list)
        for d in dets:
            cam_dets[d["camera_id"]].append(d)

        for ci, cj in cam_pairs:
            di = cam_dets.get(ci, [])
            dj = cam_dets.get(cj, [])
            if not di or not dj:
                continue

            emb_i = np.stack([d["embedding"] for d in di])
            emb_j = np.stack([d["embedding"] for d in dj])
            sim = emb_i @ emb_j.T
            cost = 1 - sim
            row_ind, col_ind = linear_sum_assignment(cost)

            gt_pairs = set()
            for ri, d_ri in enumerate(di):
                for ci2, d_ci2 in enumerate(dj):
                    if d_ri["person_id"] == d_ci2["person_id"]:
                        gt_pairs.add((ri, ci2))

            for thr_idx, thr in enumerate(thresholds):
                tp = fp = 0
                matched_set = set()
                for r, c in zip(row_ind, col_ind):
                    if sim[r, c] >= thr:
                        matched_set.add((r, c))
                        if (r, c) in gt_pairs:
                            tp += 1
                        else:
                            fp += 1
                fn = len(gt_pairs) - len(gt_pairs & matched_set)
                pair_results[(ci, cj)]["tp"].append(tp)
                pair_results[(ci, cj)]["fp"].append(fp)
                pair_results[(ci, cj)]["fn"].append(fn)

    pair_summary = {}
    with open(os.path.join(out_dir, "association_by_cam_pair.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cam_pair", "best_f1", "best_threshold", "precision", "recall", "num_frames"])
        for pair in sorted(pair_results.keys()):
            pr = pair_results[pair]
            n_frames = len(pr["tp"]) // len(thresholds) if pr["tp"] else 0
            best_f1 = 0; best_thr = 0; best_prec = 0; best_rec = 0
            for t_idx in range(len(thresholds)):
                tp = sum(pr["tp"][t_idx + f * len(thresholds)] for f in range(n_frames))
                fp = sum(pr["fp"][t_idx + f * len(thresholds)] for f in range(n_frames))
                fn = sum(pr["fn"][t_idx + f * len(thresholds)] for f in range(n_frames))
                prec = tp / max(1, tp + fp)
                rec = tp / max(1, tp + fn)
                f1 = 2 * prec * rec / max(1e-8, prec + rec)
                if f1 > best_f1:
                    best_f1 = f1; best_thr = thresholds[t_idx]; best_prec = prec; best_rec = rec
            w.writerow([f"{pair[0]}-{pair[1]}", f"{best_f1:.4f}", f"{best_thr:.2f}",
                        f"{best_prec:.4f}", f"{best_rec:.4f}", n_frames])
            pair_summary[f"{pair[0]}-{pair[1]}"] = {
                "best_f1": best_f1, "best_threshold": best_thr,
                "precision": best_prec, "recall": best_rec, "num_frames": n_frames,
            }

    with open(os.path.join(out_dir, "association_proxy_metrics.json"), "w") as f:
        json.dump(pair_summary, f, indent=2)

    # Overall best-F1
    all_f1s = [v["best_f1"] for v in pair_summary.values()]
    overall_f1 = np.mean(all_f1s) if all_f1s else 0
    print(f"  Overall mean best-F1: {overall_f1:.4f} across {len(pair_summary)} pairs")
    return pair_summary


def generate_final_report(detections, cache_stats, retrieval_metrics, pairwise_metrics,
                          association_metrics, out_dir, sample_mode):
    r = f"# V2.1 Pure 2D ReID Baseline Report ({sample_mode})\n\n"
    r += f"## Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n"

    r += "## 1. Configuration\n\n"
    r += "- Dataset: Wildtrack\n"
    r += "- Feature backend: ClipReID (ViT-B-16), fine-tuned on Wildtrack, pre-extracted\n"
    r += "- Feature dim: 512, L2-normalized\n"
    r += "- Crop source: Original RGB image + bbox_xyxy_original (1920×1080)\n"
    r += "- Evaluation: Multi-camera / cross-view ReID (query from one camera, gallery from other cameras)\n"
    r += f"- Sample mode: {sample_mode}\n"
    r += "- No 3D features used\n\n"

    r += "## 2. Cache Hit Statistics\n\n"
    r += f"- Attempted detections: {cache_stats['attempted']}\n"
    r += f"- Cache hit: {cache_stats['cache_hit']}\n"
    r += f"- Cache miss: {cache_stats['cache_miss']}\n"
    r += f"- Cache hit ratio: {cache_stats['cache_hit_ratio']:.4f}\n\n"

    r += "## 3. Detection Statistics\n\n"
    n = len(detections)
    n_ids = len(set(d["person_id"] for d in detections))
    cam_counts = defaultdict(int)
    cam_id_counts = defaultdict(set)
    for d in detections:
        cam_counts[d["camera_id"]] += 1
        cam_id_counts[d["camera_id"]].add(d["person_id"])
    r += f"- Used detections: {n}\n"
    r += f"- Unique person IDs: {n_ids}\n"
    r += f"- Per-camera detections: {dict(sorted(cam_counts.items()))}\n"
    r += f"- Per-camera ID count: {dict(sorted((k, len(v)) for k, v in cam_id_counts.items()))}\n\n"

    # Cross-camera positive pairs
    pid_cams = defaultdict(set)
    for d in detections:
        pid_cams[d["person_id"]].add(d["camera_id"])
    cross_cam_pos = sum(max(0, len(cams) * (len(cams) - 1) // 2) for cams in pid_cams.values())
    r += f"- Cross-camera positive pair groups: {cross_cam_pos}\n\n"

    r += "## 4. Overall Retrieval\n\n"
    if retrieval_metrics:
        r += "| Metric | Value |\n|--------|-------|\n"
        for k in ["mAP", "rank1", "rank5", "rank10", "skipped_queries",
                   "border_rank1", "nonborder_rank1", "small_bbox_rank1",
                   "border_count", "nonborder_count"]:
            v = retrieval_metrics.get(k)
            if v is not None:
                r += f"| {k} | {v:.4f} |\n"
    r += "\n"

    r += "## 5. Per-Camera Retrieval\n\nSee `retrieval_by_cam.csv`\n\n"
    r += "## 6. Per-Camera-Pair Retrieval\n\nSee `retrieval_by_cam_pair.csv`\n"
    r += "- Pairs with count < 50 are marked `low_stat=YES`\n\n"

    r += "## 7. Pairwise Separability\n\n"
    if pairwise_metrics:
        r += "| Metric | Value |\n|--------|-------|\n"
        for k in ["pos_cosine_mean", "neg_cosine_mean", "pos_neg_gap",
                   "roc_auc", "pr_auc", "eer", "best_f1"]:
            v = pairwise_metrics.get(k)
            if v is not None:
                r += f"| {k} | {v:.4f} |\n"
    r += "\n"

    r += "## 8. Association Proxy\n\n"
    if association_metrics:
        # C2/C5 related pairs
        c2_pairs = {k: v for k, v in association_metrics.items() if "C2" in k}
        c5_pairs = {k: v for k, v in association_metrics.items() if "C5" in k}
        all_f1s = [v["best_f1"] for v in association_metrics.values()]
        r += f"- Overall mean best-F1: {np.mean(all_f1s):.4f}\n"
        r += f"- Best threshold range: {min(v['best_threshold'] for v in association_metrics.values()):.2f} - {max(v['best_threshold'] for v in association_metrics.values()):.2f}\n\n"
        r += "### C2-related pairs\n\n"
        r += "| Pair | Best-F1 | Threshold | Precision | Recall |\n|------|---------|-----------|-----------|--------|\n"
        for k in sorted(c2_pairs.keys()):
            v = c2_pairs[k]
            r += f"| {k} | {v['best_f1']:.4f} | {v['best_threshold']:.2f} | {v['precision']:.4f} | {v['recall']:.4f} |\n"
        r += "\n### C5-related pairs\n\n"
        r += "| Pair | Best-F1 | Threshold | Precision | Recall |\n|------|---------|-----------|-----------|--------|\n"
        for k in sorted(c5_pairs.keys()):
            v = c5_pairs[k]
            r += f"| {k} | {v['best_f1']:.4f} | {v['best_threshold']:.2f} | {v['precision']:.4f} | {v['recall']:.4f} |\n"
    r += "\n"

    r += "## 9. C2/C5 Analysis\n\n"
    if retrieval_metrics:
        r += f"- C2 border Rank-1: {retrieval_metrics.get('border_rank1', 'N/A')}\n"
        r += f"- C2 non-border Rank-1: {retrieval_metrics.get('nonborder_rank1', 'N/A')}\n"
        r += f"- Border count: {retrieval_metrics.get('border_count', 0)}\n"
    r += "\n"

    # Decision
    if retrieval_metrics:
        mAP = retrieval_metrics.get("mAP", 0)
        rank1 = retrieval_metrics.get("rank1", 0)
        if mAP > 0.3 and rank1 > 0.5:
            decision = "PASS"
        elif mAP > 0.1 or rank1 > 0.2:
            decision = "UNCERTAIN"
        else:
            decision = "FAIL"
    else:
        decision = "FAIL"

    r += f"## 10. Decision: {decision}\n\n"
    r += "## 11. Next Steps\n\n"
    if decision == "PASS":
        r += "- V2.1 PASS: Can serve as formal 2D baseline for V3 comparison.\n"
    elif decision == "UNCERTAIN":
        r += "- V2.1 UNCERTAIN: Moderate performance. Check per-camera-pair.\n"
    else:
        r += "- V2.1 FAIL: 2D baseline too weak.\n"

    with open(os.path.join(out_dir, "final_report.md"), "w") as f:
        f.write(r)
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_mode", default="all", choices=["all", "first", "random", "balanced"])
    parser.add_argument("--max_per_cam", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="outputs/v2_1_2d_reid_full")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print(f"V2.1: Pure 2D ReID Baseline (mode={args.sample_mode})")
    print("=" * 60)

    print("\n[1/5] Collecting detections...")
    t0 = time.time()
    detections, cache_stats = collect_all_detections(
        max_per_cam=args.max_per_cam, sample_mode=args.sample_mode, seed=args.seed)
    print(f"  Collected {len(detections)} detections in {time.time()-t0:.0f}s")
    print(f"  Cache: attempted={cache_stats['attempted']}, hit={cache_stats['cache_hit']}, "
          f"miss={cache_stats['cache_miss']}, ratio={cache_stats['cache_hit_ratio']:.4f}")

    if not detections:
        print("[ERROR] No detections!")
        return

    # Save features
    print("\n[Saving] Features and detections...")
    emb_matrix = np.stack([d["embedding"] for d in detections])
    np.savez(os.path.join(args.out_dir, "features.npz"), features=emb_matrix)

    with open(os.path.join(args.out_dir, "detections.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["camera_id", "frame_id", "person_id", "train_id",
                                           "bbox_xyxy_original", "bbox_render", "crop_w", "crop_h",
                                           "border_flag", "small_bbox"])
        w.writeheader()
        for d in detections:
            w.writerow({k: d[k] for k in w.fieldnames})

    with open(os.path.join(args.out_dir, "cache_stats.json"), "w") as f:
        json.dump(cache_stats, f, indent=2)

    # Eval
    print("\n[2/5] Cross-camera retrieval...")
    retrieval_metrics = evaluate_retrieval(detections, args.out_dir)

    print("\n[3/5] Pairwise separability...")
    pairwise_metrics = evaluate_pairwise(detections, args.out_dir)

    print("\n[4/5] Association proxy...")
    association_metrics = evaluate_association_proxy(detections, args.out_dir)

    print("\n[5/5] Final report...")
    decision = generate_final_report(detections, cache_stats, retrieval_metrics, pairwise_metrics,
                                      association_metrics, args.out_dir, args.sample_mode)

    print(f"\n{'='*60}")
    print(f"V2.1 Complete. Decision: {decision}")
    print(f"Report: {args.out_dir}/final_report.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
