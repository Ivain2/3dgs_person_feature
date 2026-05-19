#!/usr/bin/env python3
"""V3.0.1: Evaluate 3D Aggregated ReID Features (Feature Collapse Diagnosis).

Loads aggregated 3D Gaussian features, renders person_feature_map for selected
frames with optional valid_mask, extracts ROI features using topk_opacity pooling,
and evaluates using V2.1 protocol (retrieval, pairwise, association).

NO TRAINING. NO GEOMETRY CHANGES. NO TRACKER/BBOX SCALE CHANGES.

Coordinate protocol (frozen from V0):
  - annotation bbox: 1920x1080, padding 1080->1088, downsample=4
  - render size: WxH=480x272, scale_bbox_to_render: sx=sy=0.25
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.datasets.dataset_wildtrack import WildtrackDataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import scale_bbox_to_render, _clamp_bbox, roi_pool_topk_opacity

DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
REID_INIT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase14_clean_geometry/full_soft_reset_30k/reid_init/reid_init_ckpt.pt"
DEFAULT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt"
FEAT_DIM = 512


def setup_model(ckpt_path, device):
    reid_state = torch.load(REID_INIT_CKPT, map_location="cpu", weights_only=False)
    conf = reid_state.get("config")
    conf.model.person_feature_dim = FEAT_DIM
    scene_extent = reid_state.get("scene_extent", 1.0)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ["positions", "density", "scale", "rotation", "features_albedo", "features_specular"]:
        if key in state:
            getattr(model, key).data = state[key].to(device)
    pf_key = "_person_feature" if "_person_feature" in state else "person_feature"
    if pf_key in state:
        model._person_feature = torch.nn.Parameter(state[pf_key].to(device))
    else:
        model._person_feature = torch.nn.Parameter(
            torch.randn(model.positions.shape[0], FEAT_DIM, device=device) * 0.01)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, conf


def load_v2_data(detections_csv, features_npz):
    dets = []
    with open(detections_csv, "r") as f:
        for row in csv.DictReader(f):
            row["frame_id"] = int(row["frame_id"])
            row["person_id"] = int(row["person_id"])
            row["train_id"] = int(row["train_id"])
            row["border_flag"] = row["border_flag"] == "True"
            row["small_bbox"] = row["small_bbox"] == "True"
            bbox_str = row["bbox_xyxy_original"]
            row["bbox_xyxy_original"] = [int(x.strip()) for x in bbox_str.strip("[]").split(",")]
            dets.append(row)
    features = np.load(features_npz)["features"]
    for i, d in enumerate(dets):
        d["teacher_embedding"] = features[i].astype(np.float32)
    return dets


def build_frame_cam_index(dataset):
    mapping = {}
    for idx in range(len(dataset)):
        batch = dataset[idx]
        fid = batch.get("frame_idx", -1)
        cid = batch.get("camera_id", "unknown")
        mapping[(fid, cid)] = idx
    return mapping


def build_render_mask(agg, mode, beta_eps, purity_thr, device):
    beta = agg["beta"]
    valid_mask = agg["valid_gaussian_mask"]

    if mode == "all":
        return None, {"render_mask_mode": "all", "valid_gaussian_count": int(valid_mask.sum()),
                       "valid_gaussian_ratio": float(valid_mask.float().mean()),
                       "kept_beta_mass_ratio": 1.0, "dominant_ratio_thr": None, "beta_eps": beta_eps}

    if mode == "beta":
        mask = beta > beta_eps
        total_beta = beta.sum()
        kept_beta = beta[mask].sum()
        stats = {"render_mask_mode": "beta", "valid_gaussian_count": int(mask.sum()),
                 "valid_gaussian_ratio": float(mask.float().mean()),
                 "kept_beta_mass_ratio": float(kept_beta / (total_beta + 1e-6)),
                 "dominant_ratio_thr": None, "beta_eps": beta_eps}
        return mask.to(device), stats

    if mode == "purity":
        dominant_ratio = agg["dominant_ratio"]
        mask = (beta > beta_eps) & (dominant_ratio >= purity_thr)
        total_beta = beta.sum()
        kept_beta = beta[mask].sum()
        stats = {"render_mask_mode": "purity", "valid_gaussian_count": int(mask.sum()),
                 "valid_gaussian_ratio": float(mask.float().mean()),
                 "kept_beta_mass_ratio": float(kept_beta / (total_beta + 1e-6)),
                 "dominant_ratio_thr": purity_thr, "beta_eps": beta_eps}
        return mask.to(device), stats

    return None, {}


def extract_v3_features(model, ds, fc_index, v2_dets, selected_frames, device,
                        valid_mask, pooling, topk_ratio):
    print(f"\n  Extracting V3 features via rendering + ROI pooling (pooling={pooling}, topk={topk_ratio})...")

    view_dets = defaultdict(list)
    for d in v2_dets:
        view_dets[(d["frame_id"], d["camera_id"])].append(d)

    v3_dets = []
    total_views = 0
    zero_roi = 0
    feature_norms = []
    t0 = time.time()

    for frame_id in selected_frames:
        for cam_id in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            key = (frame_id, cam_id)
            if key not in view_dets or key not in fc_index:
                continue

            dets = view_dets[key]
            if not dets:
                continue

            batch = ds[fc_index[key]]
            gpu_batch = ds.get_gpu_batch_with_intrinsics(batch)
            H, W = gpu_batch.rays_ori.shape[1], gpu_batch.rays_ori.shape[2]

            with torch.no_grad():
                pf_map, opacity_map = model.render_person_feature_map(
                    gpu_batch, train=False, frame_id=0, valid_mask=valid_mask,
                    linearize_feature=args.linearize_feature, linearize_mode=args.linearize_mode)

            pf_nonzero = (pf_map.abs().sum() > 0).item()
            opacity_nonzero = (opacity_map.sum() > 0).item()

            for d in dets:
                bbox_orig = d["bbox_xyxy_original"]
                bbox_render = scale_bbox_to_render(bbox_orig, src_w=1920, src_h=1088, dst_w=W, dst_h=H)

                if pooling == "topk_opacity":
                    feat, _ = roi_pool_topk_opacity(pf_map, opacity_map, bbox_render,
                                                    topk_ratio=topk_ratio, detach_opacity_weight=True)
                else:
                    from threedgrut.utils.roi_pooling import roi_pool_mean
                    feat = roi_pool_mean(pf_map, bbox_render)

                if feat is None:
                    feat_np = np.zeros(FEAT_DIM, dtype=np.float32)
                else:
                    feat_np = feat.cpu().numpy()

                feat_norm = np.linalg.norm(feat_np)
                feature_norms.append(feat_norm)

                if feat_norm < 1e-6:
                    zero_roi += 1

                v3_dets.append({
                    "camera_id": cam_id,
                    "frame_id": frame_id,
                    "person_id": d["person_id"],
                    "train_id": d["train_id"],
                    "bbox_xyxy_original": bbox_orig,
                    "border_flag": d["border_flag"],
                    "small_bbox": d["small_bbox"],
                    "embedding": feat_np,
                    "teacher_embedding": d["teacher_embedding"],
                    "feature_norm": float(feat_norm),
                    "render_pf_nonzero": pf_nonzero,
                    "render_opacity_nonzero": opacity_nonzero,
                })

            total_views += 1
            if total_views % 20 == 0:
                elapsed = time.time() - t0
                print(f"    Views: {total_views}, Dets: {len(v3_dets)}, Zero-ROI: {zero_roi}, {elapsed:.0f}s")

            del pf_map, opacity_map
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Extracted {len(v3_dets)} V3 features in {elapsed:.0f}s (zero-ROI: {zero_roi})")
    return v3_dets, feature_norms


def compute_diagnostics(v3_dets, feature_norms):
    n = len(v3_dets)
    if n == 0:
        return {}

    emb_matrix = np.stack([d["embedding"] for d in v3_dets])
    feature_norms = np.array(feature_norms)

    roi_zero_ratio = np.mean(feature_norms < 1e-6)
    fn_mean = float(np.mean(feature_norms))
    fn_median = float(np.median(feature_norms))

    l2_emb = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
    sim_matrix = l2_emb @ l2_emb.T
    np.fill_diagonal(sim_matrix, -1)

    rng = np.random.RandomState(42)
    sample_size = min(10000, n * (n - 1))
    all_idx = np.arange(n)
    rand_pairs = rng.choice(n, size=(sample_size, 2), replace=True)
    rand_sims = np.array([float(np.dot(l2_emb[i], l2_emb[j])) for i, j in rand_pairs])
    mean_pairwise = float(np.mean(rand_sims))

    channel_std = np.std(emb_matrix, axis=0).mean()

    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(l2_emb[:min(5000, n)])
        top1_explained = float(pca.explained_variance_ratio_[0])
    except Exception:
        top1_explained = None

    return {
        "roi_zero_ratio": float(roi_zero_ratio),
        "feature_norm_mean": fn_mean,
        "feature_norm_median": fn_median,
        "mean_pairwise_cosine": mean_pairwise,
        "feature_channel_std_mean": float(channel_std),
        "pca_top1_explained_ratio": top1_explained,
    }


def evaluate_teacher_alignment(v3_dets, out_dir):
    print("\n  Computing teacher alignment...")
    alignments = []
    cam_alignments = defaultdict(list)

    for d in v3_dets:
        v3_emb = d["embedding"]
        teacher_emb = d["teacher_embedding"]
        v3_norm = np.linalg.norm(v3_emb)
        teacher_norm = np.linalg.norm(teacher_emb)

        if v3_norm > 1e-6 and teacher_norm > 1e-6:
            cos_sim = float(np.dot(v3_emb, teacher_emb) / (v3_norm * teacher_norm))
        else:
            cos_sim = 0.0

        alignments.append(cos_sim)
        cam_alignments[d["camera_id"]].append(cos_sim)
        d["teacher_cos_sim"] = cos_sim

    mean_align = float(np.mean(alignments)) if alignments else 0
    median_align = float(np.median(alignments)) if alignments else 0
    print(f"  Teacher alignment: mean={mean_align:.4f}, median={median_align:.4f}")

    with open(os.path.join(out_dir, "teacher_alignment.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["det_idx", "camera_id", "frame_id", "person_id", "teacher_cos_sim",
                     "feature_norm", "border_flag", "small_bbox"])
        for i, d in enumerate(v3_dets):
            w.writerow([i, d["camera_id"], d["frame_id"], d["person_id"],
                        f"{d['teacher_cos_sim']:.6f}", f"{d['feature_norm']:.6f}",
                        d["border_flag"], d["small_bbox"]])

    cam_summary = {}
    for cam in sorted(cam_alignments.keys()):
        vals = cam_alignments[cam]
        cam_summary[cam] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "count": len(vals),
        }

    return {"mean": mean_align, "median": median_align, "per_camera": cam_summary}


def evaluate_retrieval(detections, out_dir):
    print("\n  Cross-camera retrieval...")

    emb_matrix = np.stack([d["embedding"] for d in detections])
    l2_emb = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)

    cam_ids = [d["camera_id"] for d in detections]
    person_ids = [d["person_id"] for d in detections]
    n = len(detections)

    sim_full = l2_emb @ l2_emb.T

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

        hits = 0
        precision_sum = 0.0
        for r_idx, j in enumerate(sorted_idx):
            if person_ids[gallery_indices[j]] == q_pid:
                hits += 1
                precision_sum += hits / (r_idx + 1)
        ap = precision_sum / max(1, len(positives))
        aps.append(ap)
        per_cam_aps[q_cam].append(ap)
        if rank is not None:
            per_pair_aps[(q_cam, g_cam)].append(ap)

    rank1 = np.mean([1 if r <= 1 else 0 for r in ranks]) if ranks else 0
    rank5 = np.mean([1 if r <= 5 else 0 for r in ranks]) if ranks else 0
    rank10 = np.mean([1 if r <= 10 else 0 for r in ranks]) if ranks else 0
    mAP = np.mean(aps) if aps else 0

    print(f"  mAP={mAP:.4f}, R1={rank1:.4f}, R5={rank5:.4f}, R10={rank10:.4f}, skipped={skipped}")

    overall_metrics = {
        "mAP": float(mAP), "rank1": float(rank1), "rank5": float(rank5), "rank10": float(rank10),
        "num_queries": len(ranks), "skipped_queries": skipped,
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


def evaluate_pairwise(detections, out_dir, max_pos=500000, max_neg=500000):
    print("\n  Pairwise separability...")

    emb_matrix = np.stack([d["embedding"] for d in detections])
    l2_emb = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
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
                        pos_sims.append(float(np.dot(l2_emb[i], l2_emb[j])))
                        if len(pos_sims) >= max_pos:
                            break
                    if len(pos_sims) >= max_pos:
                        break
                if len(pos_sims) >= max_pos:
                    break
            if len(pos_sims) >= max_pos:
                break

    rng = np.random.RandomState(42)
    n = len(detections)
    neg_sims = []
    attempts = 0
    while len(neg_sims) < min(len(pos_sims) * 2, max_neg) and attempts < max_neg * 10:
        i, j = rng.randint(0, n, 2)
        if i != j and person_ids[i] != person_ids[j] and cam_ids_list[i] != cam_ids_list[j]:
            neg_sims.append(float(np.dot(l2_emb[i], l2_emb[j])))
        attempts += 1

    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)

    print(f"  Pos: {len(pos_sims)}, Neg: {len(neg_sims)}")
    print(f"  Pos mean={pos_sims.mean():.4f}, Neg mean={neg_sims.mean():.4f}, Gap={pos_sims.mean()-neg_sims.mean():.4f}")

    roc_auc = pr_auc = eer = best_f1 = best_f1_threshold = None
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
        labels = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
        scores = np.concatenate([pos_sims, neg_sims])
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
        fpr, tpr, thresholds_roc = roc_curve(labels, scores)
        precision, recall, thresholds_pr = precision_recall_curve(labels, scores)
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = float(fpr[eer_idx])
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = float(f1_scores[best_f1_idx])
        best_f1_threshold = float(thresholds_pr[min(best_f1_idx, len(thresholds_pr) - 1)])
        print(f"  ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, BestF1={best_f1:.4f}")
    except ImportError:
        print("  [WARN] sklearn not available")

    pairwise_metrics = {
        "num_positive_pairs": len(pos_sims), "num_negative_pairs": len(neg_sims),
        "pos_cosine_mean": float(pos_sims.mean()), "pos_cosine_std": float(pos_sims.std()),
        "neg_cosine_mean": float(neg_sims.mean()), "neg_cosine_std": float(neg_sims.std()),
        "pos_neg_gap": float(pos_sims.mean() - neg_sims.mean()),
        "roc_auc": roc_auc, "pr_auc": pr_auc, "eer": eer,
        "best_f1": best_f1, "best_f1_threshold": best_f1_threshold,
    }
    with open(os.path.join(out_dir, "pairwise_metrics.json"), "w") as f:
        json.dump(pairwise_metrics, f, indent=2)

    return pairwise_metrics


def evaluate_association_proxy(detections, out_dir):
    print("\n  Cross-camera association proxy...")
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
            l2_i = emb_i / (np.linalg.norm(emb_i, axis=1, keepdims=True) + 1e-8)
            l2_j = emb_j / (np.linalg.norm(emb_j, axis=1, keepdims=True) + 1e-8)
            sim = l2_i @ l2_j.T
            cost = 1 - sim
            row_ind, col_ind = linear_sum_assignment(cost)

            gt_pairs = set()
            for ri, d_ri in enumerate(di):
                for ci2, d_ci2 in enumerate(dj):
                    if d_ri["person_id"] == d_ci2["person_id"]:
                        gt_pairs.add((ri, ci2))

            for thr in thresholds:
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
                tp = sum(pr["tp"][t_idx + f_idx * len(thresholds)] for f_idx in range(n_frames))
                fp = sum(pr["fp"][t_idx + f_idx * len(thresholds)] for f_idx in range(n_frames))
                fn = sum(pr["fn"][t_idx + f_idx * len(thresholds)] for f_idx in range(n_frames))
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

    all_f1s = [v["best_f1"] for v in pair_summary.values()]
    overall_f1 = np.mean(all_f1s) if all_f1s else 0
    print(f"  Overall mean best-F1: {overall_f1:.4f} across {len(pair_summary)} pairs")
    return pair_summary


def generate_final_report(v3_dets, alignment_metrics, retrieval_metrics,
                          pairwise_metrics, association_metrics, diagnostics,
                          mask_stats, out_dir, variant, agg_config):
    r = "# V3.0.1 Evaluation Report\n\n"
    r += f"## Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
    r += "## Configuration\n\n"
    r += f"- feature_variant: {variant}\n"
    r += f"- render_mask_mode: {mask_stats.get('render_mask_mode', 'N/A')}\n"
    r += f"- aggregation mask_policy: {agg_config.get('mask_policy', 'N/A')}\n"
    r += f"- aggregation shrink_ratio: {agg_config.get('shrink_ratio', 'N/A')}\n"
    r += f"- aggregation overlap_policy: {agg_config.get('overlap_policy', 'N/A')}\n"
    r += f"- aggregation max_frames: {agg_config.get('max_frames', 'N/A')}\n"
    r += f"- feat_dim: {agg_config.get('feat_dim', FEAT_DIM)}\n\n"

    r += "## Detection Statistics\n\n"
    n = len(v3_dets)
    n_ids = len(set(d["person_id"] for d in v3_dets))
    zero_roi = sum(1 for d in v3_dets if d["feature_norm"] < 1e-6)
    cam_counts = defaultdict(int)
    for d in v3_dets:
        cam_counts[d["camera_id"]] += 1
    r += f"- Total detections: {n}\n"
    r += f"- Unique person IDs: {n_ids}\n"
    r += f"- Zero-ROI detections: {zero_roi} ({zero_roi/max(1,n)*100:.1f}%)\n"
    r += f"- Per-camera: {dict(sorted(cam_counts.items()))}\n\n"

    r += "## Mask Statistics\n\n"
    r += "| Metric | Value |\n|--------|-------|\n"
    for k, v in mask_stats.items():
        if isinstance(v, float):
            r += f"| {k} | {v:.4f} |\n"
        else:
            r += f"| {k} | {v} |\n"
    r += "\n"

    r += "## Collapse Diagnostics\n\n"
    r += "| Metric | Value |\n|--------|-------|\n"
    for k, v in diagnostics.items():
        if isinstance(v, float):
            r += f"| {k} | {v:.6f} |\n"
        else:
            r += f"| {k} | {v} |\n"
    r += "\n"

    r += "## Teacher Alignment\n\n"
    r += f"- Mean cosine(V3, teacher): {alignment_metrics['mean']:.4f}\n"
    r += f"- Median cosine(V3, teacher): {alignment_metrics['median']:.4f}\n\n"
    r += "| Camera | Mean Alignment | Median Alignment | Count |\n|--------|---------------|-----------------|-------|\n"
    for cam in sorted(alignment_metrics["per_camera"].keys()):
        cs = alignment_metrics["per_camera"][cam]
        r += f"| {cam} | {cs['mean']:.4f} | {cs['median']:.4f} | {cs['count']} |\n"
    r += "\n"

    r += "## Retrieval Metrics\n\n"
    if retrieval_metrics:
        r += "| Metric | Value |\n|--------|-------|\n"
        for k in ["mAP", "rank1", "rank5", "rank10", "num_queries", "skipped_queries"]:
            v = retrieval_metrics.get(k)
            if v is not None:
                if isinstance(v, float):
                    r += f"| {k} | {v:.4f} |\n"
                else:
                    r += f"| {k} | {v} |\n"
    r += "\n"

    r += "## Pairwise Separability\n\n"
    if pairwise_metrics:
        r += "| Metric | Value |\n|--------|-------|\n"
        for k in ["pos_cosine_mean", "neg_cosine_mean", "pos_neg_gap",
                   "roc_auc", "pr_auc", "best_f1"]:
            v = pairwise_metrics.get(k)
            if v is not None:
                if isinstance(v, float):
                    r += f"| {k} | {v:.4f} |\n"
                else:
                    r += f"| {k} | {v} |\n"
    r += "\n"

    r += "## Association Proxy\n\n"
    if association_metrics:
        all_f1s = [v["best_f1"] for v in association_metrics.values()]
        r += f"- Overall mean best-F1: {np.mean(all_f1s):.4f}\n\n"
        c2_pairs = {k: v for k, v in association_metrics.items() if "C2" in k}
        c5_pairs = {k: v for k, v in association_metrics.items() if "C5" in k}
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

    pass_conditions = []
    fail_conditions = []

    if alignment_metrics["mean"] > 0.01:
        pass_conditions.append(f"Teacher alignment positive ({alignment_metrics['mean']:.4f})")
    elif alignment_metrics["mean"] <= 0:
        fail_conditions.append(f"Teacher alignment <= 0 ({alignment_metrics['mean']:.4f})")
    else:
        pass_conditions.append(f"Teacher alignment weakly positive ({alignment_metrics['mean']:.4f})")

    if retrieval_metrics:
        if retrieval_metrics["mAP"] > 0.01:
            pass_conditions.append(f"mAP above random ({retrieval_metrics['mAP']:.4f})")
        else:
            fail_conditions.append(f"mAP at random level ({retrieval_metrics['mAP']:.4f})")

    if zero_roi / max(1, n) > 0.5:
        fail_conditions.append(f"High zero-ROI ratio ({zero_roi}/{n})")

    cam_dead = [cam for cam in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
                if cam_counts.get(cam, 0) == 0]
    if cam_dead:
        fail_conditions.append(f"Dead cameras: {cam_dead}")

    if pairwise_metrics and pairwise_metrics.get("pos_neg_gap", 0) < 0.01:
        fail_conditions.append(f"Pos-neg cosine gap near zero ({pairwise_metrics['pos_neg_gap']:.6f})")

    if fail_conditions:
        decision = "FAIL"
    elif len(pass_conditions) >= 2 and not fail_conditions:
        decision = "PASS"
    else:
        decision = "UNCERTAIN"

    r += f"## Decision: {decision}\n\n"
    r += "### Pass conditions\n"
    for c in pass_conditions:
        r += f"- {c}\n"
    if fail_conditions:
        r += "\n### Fail conditions\n"
        for c in fail_conditions:
            r += f"- {c}\n"

    with open(os.path.join(out_dir, "final_report.md"), "w") as f:
        f.write(r)

    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--aggregated_features", required=True)
    parser.add_argument("--feature_variant", default="raw", choices=["raw", "l2"])
    parser.add_argument("--render_mask_mode", default="all", choices=["all", "beta", "purity"])
    parser.add_argument("--beta_eps", type=float, default=1e-6)
    parser.add_argument("--purity_thr", type=float, default=0.7)
    parser.add_argument("--pooling", default="topk_opacity", choices=["mean", "topk_opacity"])
    parser.add_argument("--topk_ratio", type=float, default=0.3)
    parser.add_argument("--linearize_feature", action="store_true")
    parser.add_argument("--linearize_mode", default="sh_offset", choices=["sh_offset"])
    parser.add_argument("--eval_detections", default="outputs/v2_1_2d_reid_full/detections.csv")
    parser.add_argument("--eval_teacher_features", default="outputs/v2_1_2d_reid_full/features.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print(f"V3.0.1: Evaluate 3D Aggregated ReID (variant={args.feature_variant}, mask={args.render_mask_mode})")
    print("=" * 60)

    print("\n[1/7] Loading aggregated features...")
    agg = torch.load(args.aggregated_features, map_location="cpu", weights_only=False)
    agg_config = agg.get("config", {})
    selected_frames = agg.get("selected_frames", [])
    print(f"  Feature shape: raw={agg['gaussian_features_raw'].shape}, l2={agg['gaussian_features_l2'].shape}")
    print(f"  Valid Gaussians: {agg['valid_gaussian_mask'].sum()}/{agg['valid_gaussian_mask'].shape[0]}")
    print(f"  Selected frames: {len(selected_frames)}")

    if args.feature_variant == "l2":
        gaussian_features = agg["gaussian_features_l2"]
        print(f"  Using L2-normalized features")
    else:
        gaussian_features = agg["gaussian_features_raw"]
        print(f"  Using raw features")

    print("\n[2/7] Building render mask...")
    valid_mask, mask_stats = build_render_mask(
        agg, args.render_mask_mode, args.beta_eps, args.purity_thr, device)
    print(f"  Mask mode: {args.render_mask_mode}")
    if valid_mask is not None:
        print(f"  Valid Gaussians after mask: {valid_mask.sum().item()}")
    else:
        print(f"  No mask (all Gaussians)")

    print("\n[3/7] Loading model and overriding person features...")
    model, conf = setup_model(args.ckpt, device)
    N = model.positions.shape[0]
    print(f"  N_gaussians={N}")

    override_feat = gaussian_features.to(device)
    model._person_feature = torch.nn.Parameter(override_feat)
    print(f"  Overridden _person_feature with {args.feature_variant} features: {override_feat.shape}")
    print(f"  Feature norm (mean): {override_feat.norm(dim=1).mean().item():.4f}")
    print(f"  Non-zero rows: {(override_feat.norm(dim=1) > 1e-6).sum().item()}/{N}")

    print("\n[4/7] Loading dataset and V2 data...")
    ds = WildtrackDataset(dataset_path=DATASET_PATH, downsample_factor=4, load_teacher_cache=False)
    v2_dets = load_v2_data(args.eval_detections, args.eval_teacher_features)
    print(f"  Dataset size: {len(ds)}, V2 detections: {len(v2_dets)}")

    print("\n[5/7] Building frame-camera index...")
    fc_index = build_frame_cam_index(ds)
    print(f"  Indexed {len(fc_index)} (frame,cam) pairs")

    print("\n[6/7] Extracting V3 features...")
    v3_dets, feature_norms = extract_v3_features(
        model, ds, fc_index, v2_dets, selected_frames, device,
        valid_mask, args.pooling, args.topk_ratio)

    if not v3_dets:
        print("[ERROR] No V3 features extracted!")
        return

    print("\n[7/7] Evaluating...")

    emb_matrix = np.stack([d["embedding"] for d in v3_dets])
    np.savez(os.path.join(args.out_dir, "features.npz"), features=emb_matrix)

    diagnostics = compute_diagnostics(v3_dets, feature_norms)

    alignment_metrics = evaluate_teacher_alignment(v3_dets, args.out_dir)

    with open(os.path.join(args.out_dir, "detections.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["camera_id", "frame_id", "person_id", "train_id",
                                           "bbox_xyxy_original", "border_flag", "small_bbox",
                                           "feature_norm", "teacher_cos_sim"])
        w.writeheader()
        for d in v3_dets:
            w.writerow({k: d.get(k, "") for k in w.fieldnames})

    retrieval_metrics = evaluate_retrieval(v3_dets, args.out_dir)
    pairwise_metrics = evaluate_pairwise(v3_dets, args.out_dir)
    association_metrics = evaluate_association_proxy(v3_dets, args.out_dir)

    with open(os.path.join(args.out_dir, "mask_stats.json"), "w") as f:
        json.dump(mask_stats, f, indent=2)

    decision = generate_final_report(
        v3_dets, alignment_metrics, retrieval_metrics,
        pairwise_metrics, association_metrics, diagnostics,
        mask_stats, args.out_dir, args.feature_variant, agg_config)

    print(f"\n{'='*60}")
    print(f"V3.0.1 Evaluation Complete. Decision: {decision}")
    print(f"Mask: {args.render_mask_mode}, Variant: {args.feature_variant}")
    print(f"Teacher alignment: {alignment_metrics['mean']:.4f}")
    if retrieval_metrics:
        print(f"mAP={retrieval_metrics['mAP']:.4f}, R1={retrieval_metrics['rank1']:.4f}")
    if pairwise_metrics and pairwise_metrics.get("roc_auc"):
        print(f"ROC-AUC={pairwise_metrics['roc_auc']:.4f}")
    if association_metrics:
        all_f1s = [v["best_f1"] for v in association_metrics.values()]
        print(f"AssocF1={np.mean(all_f1s):.4f}")
    print(f"Pos-Neg gap: {pairwise_metrics.get('pos_neg_gap', 0):.6f}")
    print(f"Diagnostics: roi_zero={diagnostics.get('roi_zero_ratio', 0):.4f}, "
          f"mean_pairwise_cos={diagnostics.get('mean_pairwise_cosine', 0):.4f}, "
          f"pca_top1={diagnostics.get('pca_top1_explained_ratio', 'N/A')}")
    print(f"Report: {args.out_dir}/final_report.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
