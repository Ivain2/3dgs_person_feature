#!/usr/bin/env python3
"""V3.0.2 Task C: Direct 3D Evaluation of Pure Registered Features.

Bypasses render_person_feature_map. For each detection, projects Gaussian
centers to the view, selects those falling inside the bbox, and pools their
registered features using opacity/density weights.

This tests whether the registered Gaussian features have ReID discriminative
power WITHOUT going through the 2D renderer.

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
from threedgrut.utils.roi_pooling import scale_bbox_to_render, _clamp_bbox

DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
REID_INIT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase14_clean_geometry/full_soft_reset_30k/reid_init/reid_init_ckpt.pt"
DEFAULT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt"


def setup_model(ckpt_path, device, feature_dim):
    reid_state = torch.load(REID_INIT_CKPT, map_location="cpu", weights_only=False)
    conf = reid_state.get("config")
    conf.model.person_feature_dim = feature_dim
    scene_extent = reid_state.get("scene_extent", 1.0)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ["positions", "density", "scale", "rotation", "features_albedo", "features_specular"]:
        if key in state:
            getattr(model, key).data = state[key].to(device)
    model._person_feature = torch.nn.Parameter(
        torch.randn(model.positions.shape[0], feature_dim, device=device) * 0.01)
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


@torch.no_grad()
def direct_3d_pool(model, gpu_batch, bbox_render, valid_mask, device):
    """Pool Gaussian features by projecting centers to view and selecting inside bbox.

    Uses the V0/Phase12 gaussian_set_pooling approach:
    1. Project Gaussian positions to image plane
    2. Select those inside the bbox
    3. Pool with density/opacity weights
    4. L2 normalize
    """
    x1, y1, x2, y2 = bbox_render

    xyz = model.positions
    opacity = model.get_density().squeeze(-1)
    person_feature = model.get_person_feature()

    N = xyz.shape[0]
    if N == 0:
        return None, 0

    intrinsics = gpu_batch.intrinsics
    if intrinsics is None or len(intrinsics) < 4:
        return None, 0

    fx, fy, cx, cy = intrinsics
    T_to_world = gpu_batch.T_to_world[0]
    R_world_to_cam = T_to_world[:3, :3].t()
    t_world_to_cam = -R_world_to_cam @ T_to_world[:3, 3]

    xyz_cam = (R_world_to_cam @ xyz.t()).t() + t_world_to_cam
    depth = xyz_cam[:, 2]
    valid_depth = depth > 0

    x_img = fx * xyz_cam[:, 0] / xyz_cam[:, 2] + cx
    y_img = fy * xyz_cam[:, 1] / xyz_cam[:, 2] + cy
    h_img, w_img = gpu_batch.rays_dir.shape[1], gpu_batch.rays_dir.shape[2]

    x_finite = torch.isfinite(x_img)
    y_finite = torch.isfinite(y_img)
    x_in_bounds = (x_img >= 0) & (x_img < w_img)
    y_in_bounds = (y_img >= 0) & (y_img < h_img)

    valid = valid_depth & x_finite & y_finite & x_in_bounds & y_in_bounds
    if valid_mask is not None:
        valid = valid & valid_mask.to(device)

    inside_bbox = (x_img >= x1) & (x_img < x2) & (y_img >= y1) & (y_img < y2)
    inside = valid & inside_bbox

    if inside.sum() == 0:
        return None, 0

    weights = opacity[inside]
    z = person_feature[inside]
    weight_sum = weights.sum()

    if weight_sum < 1e-8:
        return None, int(inside.sum().item())

    weighted_sum = (weights[:, None] * z).sum(dim=0)
    G = weighted_sum / (weight_sum + 1e-8)
    G_norm = G.norm()
    if G_norm > 1e-6:
        G = G / G_norm

    return G, int(inside.sum().item())


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
                     "feature_norm"])
        for i, d in enumerate(v3_dets):
            w.writerow([i, d["camera_id"], d["frame_id"], d["person_id"],
                        f"{d['teacher_cos_sim']:.6f}", f"{d['feature_norm']:.6f}"])

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

        hits = 0
        precision_sum = 0.0
        for r_idx, j in enumerate(sorted_idx):
            if person_ids[gallery_indices[j]] == q_pid:
                hits += 1
                precision_sum += hits / (r_idx + 1)
        ap = precision_sum / max(1, len(positives))
        aps.append(ap)

    rank1 = np.mean([1 if r <= 1 else 0 for r in ranks]) if ranks else 0
    rank5 = np.mean([1 if r <= 5 else 0 for r in ranks]) if ranks else 0
    rank10 = np.mean([1 if r <= 10 else 0 for r in ranks]) if ranks else 0
    mAP = np.mean(aps) if aps else 0

    print(f"  mAP={mAP:.4f}, R1={rank1:.4f}, R5={rank5:.4f}, R10={rank10:.4f}, skipped={skipped}")

    return {
        "mAP": float(mAP), "rank1": float(rank1), "rank5": float(rank5), "rank10": float(rank10),
        "num_queries": len(ranks), "skipped_queries": skipped,
    }


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
        pass

    return {
        "num_positive_pairs": len(pos_sims), "num_negative_pairs": len(neg_sims),
        "pos_cosine_mean": float(pos_sims.mean()), "pos_cosine_std": float(pos_sims.std()),
        "neg_cosine_mean": float(neg_sims.mean()), "neg_cosine_std": float(neg_sims.std()),
        "pos_neg_gap": float(pos_sims.mean() - neg_sims.mean()),
        "roc_auc": roc_auc, "pr_auc": pr_auc, "eer": eer,
        "best_f1": best_f1, "best_f1_threshold": best_f1_threshold,
    }


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
        pair_summary[f"{pair[0]}-{pair[1]}"] = {
            "best_f1": best_f1, "best_threshold": best_thr,
            "precision": best_prec, "recall": best_rec, "num_frames": n_frames,
        }

    all_f1s = [v["best_f1"] for v in pair_summary.values()]
    overall_f1 = np.mean(all_f1s) if all_f1s else 0
    print(f"  Overall mean best-F1: {overall_f1:.4f} across {len(pair_summary)} pairs")
    return pair_summary


def generate_final_report(v3_dets, alignment_metrics, retrieval_metrics,
                          pairwise_metrics, association_metrics, diagnostics,
                          out_dir, feature_set, method_name):
    r = f"# V3.0.2 Direct 3D Evaluation Report ({method_name})\n\n"
    r += f"## Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
    r += f"## Configuration\n\n"
    r += f"- feature_set: {feature_set}\n"
    r += f"- method: direct 3D pooling (no renderer)\n\n"

    r += "## Detection Statistics\n\n"
    n = len(v3_dets)
    n_ids = len(set(d["person_id"] for d in v3_dets))
    zero_roi = sum(1 for d in v3_dets if d["feature_norm"] < 1e-6)
    r += f"- Total detections: {n}\n"
    r += f"- Unique person IDs: {n_ids}\n"
    r += f"- Zero-norm detections: {zero_roi} ({zero_roi/max(1,n)*100:.1f}%)\n\n"

    r += "## Collapse Diagnostics\n\n"
    r += "| Metric | Value |\n|--------|-------|\n"
    for k, v in diagnostics.items():
        if isinstance(v, float):
            r += f"| {k} | {v:.6f} |\n"
        else:
            r += f"| {k} | {v} |\n"
    r += "\n"

    r += "## Teacher Alignment\n\n"
    r += f"- Mean: {alignment_metrics['mean']:.4f}\n"
    r += f"- Median: {alignment_metrics['median']:.4f}\n\n"

    r += "## Retrieval Metrics\n\n"
    if retrieval_metrics:
        r += f"- mAP: {retrieval_metrics['mAP']:.4f}\n"
        r += f"- R1: {retrieval_metrics['rank1']:.4f}\n"
        r += f"- R5: {retrieval_metrics['rank5']:.4f}\n"
        r += f"- R10: {retrieval_metrics['rank10']:.4f}\n\n"

    r += "## Pairwise Separability\n\n"
    if pairwise_metrics:
        r += f"- Pos cosine mean: {pairwise_metrics['pos_cosine_mean']:.4f}\n"
        r += f"- Neg cosine mean: {pairwise_metrics['neg_cosine_mean']:.4f}\n"
        r += f"- Pos-Neg gap: {pairwise_metrics['pos_neg_gap']:.4f}\n"
        if pairwise_metrics.get("roc_auc"):
            r += f"- ROC-AUC: {pairwise_metrics['roc_auc']:.4f}\n"
        if pairwise_metrics.get("best_f1"):
            r += f"- AssocF1: {pairwise_metrics['best_f1']:.4f}\n"
    r += "\n"

    if association_metrics:
        all_f1s = [v["best_f1"] for v in association_metrics.values()]
        r += f"## Association Proxy\n\n"
        r += f"- Overall mean best-F1: {np.mean(all_f1s):.4f}\n\n"

    with open(os.path.join(out_dir, "final_report.md"), "w") as f:
        f.write(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--registered_features", required=True)
    parser.add_argument("--feature_set", default="beta", choices=["beta", "purity07"])
    parser.add_argument("--eval_detections", default="outputs/v2_1_2d_reid_full/detections.csv")
    parser.add_argument("--eval_teacher_features", default="outputs/v2_1_2d_reid_full/features.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print(f"V3.0.2: Direct 3D Evaluation (feature_set={args.feature_set})")
    print("=" * 60)

    print("\n[1/6] Loading registered features...")
    reg = torch.load(args.registered_features, map_location="cpu", weights_only=False)
    if args.feature_set == "beta":
        features = reg["features_beta"]
        valid_mask = reg["mask_beta"]
    else:
        features = reg["features_purity07"]
        valid_mask = reg["mask_purity07"]
    feat_dim = features.shape[1]
    N = features.shape[0]
    print(f"  Feature shape: {features.shape}")
    print(f"  Valid mask sum: {valid_mask.sum()}/{N}")

    print("\n[2/6] Loading model...")
    model, conf = setup_model(args.ckpt, device, feat_dim)
    model._person_feature = torch.nn.Parameter(features.to(device))
    print(f"  Overridden _person_feature: {features.shape}")

    print("\n[3/6] Loading dataset and V2 data...")
    ds = WildtrackDataset(dataset_path=DATASET_PATH, downsample_factor=4, load_teacher_cache=False)
    v2_dets = load_v2_data(args.eval_detections, args.eval_teacher_features)
    print(f"  Dataset size: {len(ds)}, V2 detections: {len(v2_dets)}")

    agg_config = reg.get("config", {})
    if "selected_frames" in reg:
        selected_frames = reg["selected_frames"]
    else:
        agg2 = torch.load(agg_config.get("aggregated_features", ""), map_location="cpu", weights_only=False)
        selected_frames = agg2.get("selected_frames", [])

    fc_mapping = {}
    for idx in range(len(ds)):
        batch = ds[idx]
        fid = batch.get("frame_idx", -1)
        cid = batch.get("camera_id", "unknown")
        fc_mapping[(fid, cid)] = idx

    view_dets = defaultdict(list)
    for d in v2_dets:
        view_dets[(d["frame_id"], d["camera_id"])].append(d)

    print("\n[4/6] Direct 3D pooling...")
    v3_dets = []
    no_roi = 0
    gaussian_counts = []
    feature_norms = []
    t0 = time.time()
    total_views = 0

    for frame_id in selected_frames:
        for cam_id in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            key = (frame_id, cam_id)
            if key not in view_dets or key not in fc_mapping:
                continue

            dets = view_dets[key]
            if not dets:
                continue

            batch = ds[fc_mapping[key]]
            gpu_batch = ds.get_gpu_batch_with_intrinsics(batch)
            H, W = gpu_batch.rays_ori.shape[1], gpu_batch.rays_ori.shape[2]

            for d in dets:
                bbox_orig = d["bbox_xyxy_original"]
                bbox_render = scale_bbox_to_render(bbox_orig, src_w=1920, src_h=1088, dst_w=W, dst_h=H)
                x1, y1, x2, y2 = _clamp_bbox(bbox_render, H, W)

                feat, gauss_count = direct_3d_pool(model, gpu_batch, (x1, y1, x2, y2), valid_mask, device)

                if feat is None:
                    feat_np = np.zeros(feat_dim, dtype=np.float32)
                    no_roi += 1
                else:
                    feat_np = feat.cpu().numpy()

                gauss_counts = int(gauss_count)
                gaussian_counts.append(gauss_counts)
                feat_norm = float(np.linalg.norm(feat_np))
                feature_norms.append(feat_norm)

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
                    "feature_norm": feat_norm,
                    "gaussian_count": gauss_counts,
                })

            total_views += 1
            if total_views % 20 == 0:
                elapsed = time.time() - t0
                print(f"    Views: {total_views}, Dets: {len(v3_dets)}, No-ROI: {no_roi}, {elapsed:.0f}s")

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Extracted {len(v3_dets)} features in {elapsed:.0f}s (no-ROI: {no_roi})")

    print("\n[5/6] Saving features...")
    emb_matrix = np.stack([d["embedding"] for d in v3_dets])
    np.savez(os.path.join(args.out_dir, "features.npz"), features=emb_matrix)

    with open(os.path.join(args.out_dir, "detections.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["camera_id", "frame_id", "person_id", "train_id",
                                           "bbox_xyxy_original", "border_flag", "small_bbox",
                                           "feature_norm", "gaussian_count"])
        w.writeheader()
        for d in v3_dets:
            w.writerow({k: d.get(k, "") for k in w.fieldnames})

    pooling_stats = {
        "no_gaussian_roi_ratio": no_roi / max(1, len(v3_dets)),
        "pooled_gaussian_count_mean": float(np.mean(gaussian_counts)),
        "pooled_gaussian_count_median": float(np.median(gaussian_counts)),
    }
    with open(os.path.join(args.out_dir, "pooling_stats.json"), "w") as f:
        json.dump(pooling_stats, f, indent=2)

    print("\n[6/6] Evaluating...")
    diagnostics = compute_diagnostics(v3_dets, feature_norms)
    alignment_metrics = evaluate_teacher_alignment(v3_dets, args.out_dir)
    retrieval_metrics = evaluate_retrieval(v3_dets, args.out_dir)
    pairwise_metrics = evaluate_pairwise(v3_dets, args.out_dir)
    association_metrics = evaluate_association_proxy(v3_dets, args.out_dir)

    generate_final_report(
        v3_dets, alignment_metrics, retrieval_metrics,
        pairwise_metrics, association_metrics, diagnostics,
        args.out_dir, args.feature_set, "direct3D")

    print(f"\n{'='*60}")
    print(f"V3.0.2 Direct 3D Evaluation Complete")
    print(f"Feature set: {args.feature_set}")
    print(f"Teacher alignment: {alignment_metrics['mean']:.4f}")
    if retrieval_metrics:
        print(f"mAP={retrieval_metrics['mAP']:.4f}, R1={retrieval_metrics['rank1']:.4f}")
    if pairwise_metrics and pairwise_metrics.get("roc_auc"):
        print(f"ROC-AUC={pairwise_metrics['roc_auc']:.4f}")
    if pairwise_metrics:
        print(f"Pos-Neg gap: {pairwise_metrics['pos_neg_gap']:.6f}")
    if association_metrics:
        all_f1s = [v["best_f1"] for v in association_metrics.values()]
        print(f"AssocF1={np.mean(all_f1s):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
