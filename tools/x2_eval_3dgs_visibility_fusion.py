#!/usr/bin/env python3
"""X2 Task 3+4+5: Evaluate 3DGS visibility-weighted fusion with leave-one-view-out.

Methods:
1. naive_average_feature (X1 baseline)
2. bbox_area_weighted_feature (X1 baseline)
3. visibility_projection
4. visibility_bbox_valid
5. visibility_3dgs_depth_ratio (torso)
6. visibility_3dgs_depth_ratio_multi_anchor (weighted foot+torso+head)
7. visibility_3dgs_gaussian_count (inverse: fewer = more visible)
8. visibility_3dgs_weighted_opacity (inverse: less opacity = more visible)
9. visibility_bbox_occlusion (inverse: less overlap = more visible)
10. visibility_3dgs_combined (depth_ratio * (1 - bbox_occlusion))

All methods use leave-one-view-out with one gallery entry per person.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

CAMERA_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance-table", default="outputs/x1_mv_fusion/instance_table.csv")
    p.add_argument("--features", default="outputs/v2_2d_reid_baseline/features.npz")
    p.add_argument("--detections", default="outputs/v2_2d_reid_baseline/detections.csv")
    p.add_argument("--visibility", default="outputs/x2_3dgs_visibility_fusion/visibility_scores.csv")
    p.add_argument("--output-dir", default="outputs/x2_3dgs_visibility_fusion")
    return p.parse_args()


def load_data(args):
    inst = []
    with open(args.instance_table) as f:
        for r in csv.DictReader(f):
            r["frame_id"] = int(r["frame_id"])
            r["person_id"] = int(r["person_id"])
            r["feature_index"] = int(r["feature_index"])
            r["bbox_valid"] = r["bbox_valid"] == "True"
            r["bbox_area"] = int(r["bbox_area"])
            r["bbox_height"] = int(r["bbox_height"])
            inst.append(r)

    features = np.load(args.features)["features"]

    vis = {}
    with open(args.visibility) as f:
        for r in csv.DictReader(f):
            key = (int(r["frame_id"]), int(r["person_id"]), r["camera_id"])
            vis[key] = {
                "v_projection": int(r["v_projection"]),
                "v_bbox_valid": int(r["v_bbox_valid"]),
                "v_depth_ratio_foot": float(r["v_depth_ratio_foot"]),
                "v_depth_ratio_torso": float(r["v_depth_ratio_torso"]),
                "v_depth_ratio_head": float(r["v_depth_ratio_head"]),
                "v_gaussian_count_torso": int(r["v_gaussian_count_torso"]),
                "v_weighted_opacity_torso": float(r["v_weighted_opacity_torso"]),
                "v_bbox_occlusion": float(r["v_bbox_occlusion"]),
                "v_weighted_3dgs": float(r["v_weighted_3dgs"]),
            }

    dets = {}
    with open(args.detections) as f:
        for r in csv.DictReader(f):
            dets[(int(r["frame_id"]), int(r["person_id"]), r["camera_id"])] = r

    return inst, features, vis, dets


def get_visibility_weight(vis_data, method):
    if method == "naive_average_feature":
        return 1.0
    elif method == "bbox_area_weighted_feature":
        return None
    elif method == "visibility_projection":
        return float(vis_data["v_projection"])
    elif method == "visibility_bbox_valid":
        return float(vis_data["v_bbox_valid"])
    elif method == "visibility_3dgs_depth_ratio":
        return max(vis_data["v_depth_ratio_torso"], 0.01)
    elif method == "visibility_3dgs_depth_ratio_multi_anchor":
        dr_f = max(vis_data["v_depth_ratio_foot"], 0.01)
        dr_t = max(vis_data["v_depth_ratio_torso"], 0.01)
        dr_h = max(vis_data["v_depth_ratio_head"], 0.01)
        return 0.2 * dr_f + 0.5 * dr_t + 0.3 * dr_h
    elif method == "visibility_3dgs_gaussian_count":
        gc = vis_data["v_gaussian_count_torso"]
        return 1.0 / (1.0 + gc)
    elif method == "visibility_3dgs_weighted_opacity":
        wo = vis_data["v_weighted_opacity_torso"]
        return 1.0 / (1.0 + wo)
    elif method == "visibility_bbox_occlusion":
        return 1.0 - vis_data["v_bbox_occlusion"]
    elif method == "visibility_3dgs_combined":
        dr_t = max(vis_data["v_depth_ratio_torso"], 0.01)
        occ = vis_data["v_bbox_occlusion"]
        return dr_t * (1.0 - occ)
    else:
        raise ValueError(f"Unknown method: {method}")


def fuse_features(obs_rows, features, method, vis, eps=1e-8):
    if method == "naive_average_feature":
        feats = np.array([features[r["feature_index"]] for r in obs_rows])
        fused = np.mean(feats, axis=0)
    elif method == "bbox_area_weighted_feature":
        feats = np.array([features[r["feature_index"]] for r in obs_rows])
        ws = np.array([float(r["bbox_area"]) for r in obs_rows], dtype=np.float32)
        ws = ws / (ws.sum() + eps)
        fused = np.sum(ws[:, None] * feats, axis=0)
    else:
        feats = np.array([features[r["feature_index"]] for r in obs_rows])
        ws = []
        for r in obs_rows:
            key = (r["frame_id"], r["person_id"], r["camera_id"])
            w = get_visibility_weight(vis.get(key, {}), method)
            ws.append(max(w, eps))
        ws = np.array(ws, dtype=np.float32)
        ws = ws / (ws.sum() + eps)
        fused = np.sum(ws[:, None] * feats, axis=0)

    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused


def evaluate_method(inst, features, vis, method):
    valid_rows = [r for r in inst if r["bbox_valid"]]
    fpg = defaultdict(list)
    for r in valid_rows:
        fpg[(r["frame_id"], r["person_id"])].append(r)
    fps = defaultdict(set)
    for r in valid_rows:
        fps[r["frame_id"]].add(r["person_id"])

    ranks = []
    aps = []
    skipped = 0
    gallery_sizes = []
    fallback_count = 0
    per_cam_ranks = defaultdict(list)
    per_cam_aps = defaultdict(list)

    for frame_id in sorted(fps.keys()):
        person_ids = sorted(fps[frame_id])
        for person_id in person_ids:
            group = fpg.get((frame_id, person_id), [])
            if len(group) < 2:
                continue
            for query_row in group:
                q_cam = query_row["camera_id"]
                q_feat = features[query_row["feature_index"]]
                other_views = [r for r in group if r["camera_id"] != q_cam]
                if not other_views:
                    skipped += 1
                    continue

                gallery = []
                for pid in person_ids:
                    pid_group = fpg.get((frame_id, pid), [])
                    pid_other = [r for r in pid_group if r["camera_id"] != q_cam]
                    if not pid_other:
                        continue
                    pid_fused = fuse_features(pid_other, features, method, vis)
                    gallery.append((pid, pid_fused))

                pos_in_gallery = [g for g in gallery if g[0] == person_id]
                if not pos_in_gallery:
                    skipped += 1
                    continue

                gallery_sizes.append(len(gallery))
                g_feats = np.array([g[1] for g in gallery])
                g_pids = [g[0] for g in gallery]
                sims = g_feats @ q_feat
                sorted_idx = np.argsort(-sims)
                sorted_pids = [g_pids[j] for j in sorted_idx]

                rank = None
                for r_idx, pid in enumerate(sorted_pids):
                    if pid == person_id:
                        rank = r_idx + 1
                        break
                if rank is not None:
                    ranks.append(rank)
                    per_cam_ranks[q_cam].append(rank)

                hits = 0
                ps = 0.0
                for r_idx, j in enumerate(sorted_idx):
                    if g_pids[j] == person_id:
                        hits += 1
                        ps += hits / (r_idx + 1)
                n_pos = len(pos_in_gallery)
                ap = ps / max(1, n_pos)
                aps.append(ap)
                per_cam_aps[q_cam].append(ap)

    return {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "Rank-1": float(np.mean([1 if r <= 1 else 0 for r in ranks])) if ranks else 0.0,
        "Rank-5": float(np.mean([1 if r <= 5 else 0 for r in ranks])) if ranks else 0.0,
        "Rank-10": float(np.mean([1 if r <= 10 else 0 for r in ranks])) if ranks else 0.0,
        "num_queries": len(ranks),
        "skipped_queries": skipped,
        "avg_gallery_size": float(np.mean(gallery_sizes)) if gallery_sizes else 0.0,
        "fallback_count": fallback_count,
        "per_camera": {
            cam: {
                "mAP": float(np.mean([a for a in per_cam_aps[cam]])) if per_cam_aps[cam] else 0.0,
                "Rank-1": float(np.mean([1 if r <= 1 else 0 for r in per_cam_ranks[cam]])) if per_cam_ranks[cam] else 0.0,
            }
            for cam in CAMERA_NAMES
        },
    }


def compute_per_query_aps(inst, features, vis, method):
    valid_rows = [r for r in inst if r["bbox_valid"]]
    fpg = defaultdict(list)
    for r in valid_rows:
        fpg[(r["frame_id"], r["person_id"])].append(r)
    fps = defaultdict(set)
    for r in valid_rows:
        fps[r["frame_id"]].add(r["person_id"])

    query_aps = []

    for frame_id in sorted(fps.keys()):
        person_ids = sorted(fps[frame_id])
        for person_id in person_ids:
            group = fpg.get((frame_id, person_id), [])
            if len(group) < 2:
                continue
            for query_row in group:
                q_cam = query_row["camera_id"]
                q_feat = features[query_row["feature_index"]]
                other_views = [r for r in group if r["camera_id"] != q_cam]
                if not other_views:
                    continue

                gallery = []
                for pid in person_ids:
                    pid_group = fpg.get((frame_id, pid), [])
                    pid_other = [r for r in pid_group if r["camera_id"] != q_cam]
                    if not pid_other:
                        continue
                    pid_fused = fuse_features(pid_other, features, method, vis)
                    gallery.append((pid, pid_fused))

                pos_in_gallery = [g for g in gallery if g[0] == person_id]
                if not pos_in_gallery:
                    continue

                g_feats = np.array([g[1] for g in gallery])
                g_pids = [g[0] for g in gallery]
                sims = g_feats @ q_feat
                sorted_idx = np.argsort(-sims)

                hits = 0
                ps = 0.0
                for r_idx, j in enumerate(sorted_idx):
                    if g_pids[j] == person_id:
                        hits += 1
                        ps += hits / (r_idx + 1)
                n_pos = len(pos_in_gallery)
                query_aps.append(ps / max(1, n_pos))

    return np.array(query_aps)


def paired_ttest(aps_a, aps_b, name_a, name_b):
    diffs = aps_a - aps_b
    t_stat, p_value = stats.ttest_rel(aps_a, aps_b)
    ci_lo = diffs.mean() - 1.96 * diffs.std() / np.sqrt(len(diffs))
    ci_hi = diffs.mean() + 1.96 * diffs.std() / np.sqrt(len(diffs))
    return {
        "comparison": f"{name_a} vs {name_b}",
        "mean_ap_diff": float(diffs.mean()),
        "p_value": float(p_value),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "significant_005": bool(p_value < 0.05),
        "a_wins": int((diffs > 0).sum()),
        "b_wins": int((diffs < 0).sum()),
        "ties": int((diffs == 0).sum()),
    }


def write_readme(all_results, ttest_results, subset_results, output_dir):
    lines = []
    lines.append("# X2 3DGS Visibility-Weighted Fusion")
    lines.append("")
    lines.append("## 1. Experiment Protocol")
    lines.append("Leave-one-view-out evaluation on WildTrack (7 cameras, 89 frames, 92 person IDs).")
    lines.append("Same protocol as X1 fair baseline: one gallery entry per person, query camera excluded.")
    lines.append("")
    lines.append("## 2. Visibility Metrics")
    lines.append("")
    lines.append("NOTE: Standard 3DGS transmittance is NOT used. The model was trained without person masking,")
    lines.append("so background Gaussians fill the entire scene including person locations, making T~0 everywhere.")
    lines.append("")
    lines.append("Instead, we use:")
    lines.append("- v_depth_ratio: ratio of nearest Gaussian depth to anchor depth at projected pixel")
    lines.append("  - 1.0 = no Gaussian in front of anchor (visible)")
    lines.append("  - <1.0 = Gaussian closer than anchor (partially occluded)")
    lines.append("- v_gaussian_count: number of Gaussians near the projected anchor pixel")
    lines.append("- v_weighted_opacity: sum of alpha contributions from nearby Gaussians")
    lines.append("- v_bbox_occlusion: max IoU with other persons' bboxes")
    lines.append("")
    lines.append("## 3. Results (All Queries)")
    lines.append("")
    lines.append("| Method | mAP | Rank-1 | Rank-5 | Rank-10 | #Queries |")
    lines.append("|--------|-----|--------|--------|---------|----------|")
    for method in all_results:
        r = all_results[method]
        lines.append(f"| {method} | {r['mAP']:.4f} | {r['Rank-1']:.4f} | {r['Rank-5']:.4f} | {r['Rank-10']:.4f} | {r['num_queries']} |")
    lines.append("")
    lines.append("## 4. Paired t-tests")
    lines.append("")
    for t in ttest_results:
        lines.append(f"### {t['comparison']}")
        lines.append(f"- Mean AP diff: {t['mean_ap_diff']:.6f}")
        lines.append(f"- p-value: {t['p_value']:.6f}")
        lines.append(f"- 95% CI: [{t['ci_95'][0]:.6f}, {t['ci_95'][1]:.6f}]")
        lines.append(f"- Significant @0.05: {t['significant_005']}")
        lines.append(f"- A wins / B wins / ties: {t['a_wins']} / {t['b_wins']} / {t['ties']}")
        lines.append("")
    lines.append("## 5. Subset Analysis")
    lines.append("")
    for subset_name, subset_methods in subset_results.items():
        lines.append(f"### {subset_name}")
        lines.append("| Method | mAP | Rank-1 | #Queries |")
        lines.append("|--------|-----|--------|----------|")
        for method, r in subset_methods.items():
            lines.append(f"| {method} | {r['mAP']:.4f} | {r['Rank-1']:.4f} | {r['num_queries']} |")
        lines.append("")
    lines.append("## 6. Conclusion")
    lines.append("")
    na = all_results.get("naive_average_feature", {})
    best_3dgs = None
    best_3dgs_map = 0
    for m in ["visibility_3dgs_depth_ratio", "visibility_3dgs_depth_ratio_multi_anchor",
              "visibility_3dgs_combined"]:
        if m in all_results and all_results[m]["mAP"] > best_3dgs_map:
            best_3dgs = m
            best_3dgs_map = all_results[m]["mAP"]
    if best_3dgs:
        delta = (best_3dgs_map - na["mAP"]) * 100
        if delta > 0:
            lines.append(f"Best 3DGS method ({best_3dgs}) beats naive_average by +{delta:.2f}% mAP")
        else:
            lines.append(f"Best 3DGS method ({best_3dgs}) does NOT beat naive_average ({delta:.2f}% mAP)")
    lines.append("")

    path = os.path.join(output_dir, "README_X2_3DGS_VISIBILITY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[SAVE] {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    inst, features, vis, dets = load_data(args)
    print(f"[LOAD] instance_table: {len(inst)}, features: {features.shape}, visibility: {len(vis)}")

    methods = [
        "naive_average_feature",
        "bbox_area_weighted_feature",
        "visibility_projection",
        "visibility_bbox_valid",
        "visibility_3dgs_depth_ratio",
        "visibility_3dgs_depth_ratio_multi_anchor",
        "visibility_3dgs_gaussian_count",
        "visibility_3dgs_weighted_opacity",
        "visibility_bbox_occlusion",
        "visibility_3dgs_combined",
    ]

    all_results = {}
    per_query_aps = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {method}")
        print(f"{'='*60}")
        result = evaluate_method(inst, features, vis, method)
        all_results[method] = result
        print(f"  mAP={result['mAP']:.4f}  Rank-1={result['Rank-1']:.4f}  "
              f"Rank-5={result['Rank-5']:.4f}  Rank-10={result['Rank-10']:.4f}  "
              f"queries={result['num_queries']}  skipped={result['skipped_queries']}")

    key_methods = ["naive_average_feature", "visibility_3dgs_depth_ratio",
                   "visibility_3dgs_depth_ratio_multi_anchor", "visibility_3dgs_combined"]
    for method in key_methods:
        print(f"Computing per-query APs for {method}...")
        per_query_aps[method] = compute_per_query_aps(inst, features, vis, method)

    ttest_results = []
    ttest_pairs = [
        ("visibility_3dgs_depth_ratio_multi_anchor", "naive_average_feature"),
        ("visibility_3dgs_depth_ratio_multi_anchor", "bbox_area_weighted_feature"),
        ("visibility_3dgs_depth_ratio", "naive_average_feature"),
        ("visibility_3dgs_combined", "naive_average_feature"),
    ]
    for a, b in ttest_pairs:
        if a in per_query_aps and b in per_query_aps:
            t = paired_ttest(per_query_aps[a], per_query_aps[b], a, b)
            ttest_results.append(t)
            print(f"  {t['comparison']}: diff={t['mean_ap_diff']:.6f}, p={t['p_value']:.6f}, "
                  f"sig={t['significant_005']}, wins={t['a_wins']}/{t['b_wins']}/{t['ties']}")

    # Subset analysis
    bbox_area_25 = np.percentile([r["bbox_area"] for r in inst if r["bbox_valid"]], 25)
    border_flags = {}
    with open(args.detections) as f:
        for r in csv.DictReader(f):
            key = (int(r["frame_id"]), int(r["person_id"]), r["camera_id"])
            border_flags[key] = r.get("border_flag", "0") == "1"
    small_flags = {}
    with open(args.detections) as f:
        for r in csv.DictReader(f):
            key = (int(r["frame_id"]), int(r["person_id"]), r["camera_id"])
            small_flags[key] = r.get("small_bbox", "0") == "1"

    subset_results = {}

    # Border queries
    border_inst = [r for r in inst if border_flags.get((r["frame_id"], r["person_id"], r["camera_id"]), False)]
    if border_inst:
        subset_results["border_queries"] = {}
        for method in ["naive_average_feature", "visibility_3dgs_depth_ratio", "visibility_3dgs_combined"]:
            result = evaluate_method(border_inst, features, vis, method)
            subset_results["border_queries"][method] = result

    # Small bbox queries
    small_inst = [r for r in inst if small_flags.get((r["frame_id"], r["person_id"], r["camera_id"]), False)]
    if small_inst:
        subset_results["small_bbox_queries"] = {}
        for method in ["naive_average_feature", "visibility_3dgs_depth_ratio", "visibility_3dgs_combined"]:
            result = evaluate_method(small_inst, features, vis, method)
            subset_results["small_bbox_queries"][method] = result

    # Low bbox area queries
    low_area_inst = [r for r in inst if r["bbox_valid"] and r["bbox_area"] < bbox_area_25]
    if low_area_inst:
        subset_results["low_area_queries"] = {}
        for method in ["naive_average_feature", "visibility_3dgs_depth_ratio", "visibility_3dgs_combined"]:
            result = evaluate_method(low_area_inst, features, vis, method)
            subset_results["low_area_queries"][method] = result

    # Save results
    csv_path = os.path.join(args.output_dir, "eval_3dgs_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mAP", "Rank-1", "Rank-5", "Rank-10",
                                                "num_queries", "skipped_queries", "avg_gallery_size", "fallback_count"])
        writer.writeheader()
        for method, m in all_results.items():
            writer.writerow({
                "method": method, "mAP": f"{m['mAP']:.6f}", "Rank-1": f"{m['Rank-1']:.6f}",
                "Rank-5": f"{m['Rank-5']:.6f}", "Rank-10": f"{m['Rank-10']:.6f}",
                "num_queries": m["num_queries"], "skipped_queries": m["skipped_queries"],
                "avg_gallery_size": f"{m['avg_gallery_size']:.2f}", "fallback_count": m["fallback_count"],
            })
    print(f"\n[SAVE] {csv_path}")

    json_path = os.path.join(args.output_dir, "eval_3dgs_results.json")
    with open(json_path, "w") as f:
        json.dump({"results": all_results, "ttests": ttest_results, "subsets": subset_results}, f, indent=2)
    print(f"[SAVE] {json_path}")

    write_readme(all_results, ttest_results, subset_results, args.output_dir)


if __name__ == "__main__":
    main()
