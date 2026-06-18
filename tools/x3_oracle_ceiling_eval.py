#!/usr/bin/env python3
"""Oracle Ceiling Experiment: theoretical upper bound of view-reliability weighting.

Measures how much GT-quality view selection/weighting can improve over naive average.
If even oracle cannot significantly beat naive, no real visibility signal can.

Methods:
  baseline_naive:         equal weight average (baseline)
  oracle_weighted_occ:    weighted by (1 - bbox_occlusion)
  oracle_weighted_area:   weighted by normalized bbox_area
  oracle_topk1:           keep only top-1 quality view (by s_occ)
  oracle_topk2:           keep only top-2 quality views (by s_occ)
  oracle_threshold:       discard views with s_occ < 0.5, then average
  oracle_best_quality:    keep single best quality view (strongest legal oracle)
  oracle_best_by_query:   [CHEATING] keep view most similar to query (upper bound reference)

Correctness rules:
  1. One fused gallery entry per person (fair granularity)
  2. Rank formula: [1 if r<=1 else 0 for r in ranks]
  3. Leave-one-view-out: query camera excluded from all gallery fusion
  4. feature_index == row index verified
  5. Oracle weights only use gallery view GT quality (except method E, labeled cheating)
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
    p.add_argument("--visibility", default="outputs/x2_3dgs_visibility_fusion_bugfix/visibility_scores.csv")
    p.add_argument("--output-dir", default="outputs/x3_oracle_ceiling")
    return p.parse_args()


def load_data(args):
    inst = []
    with open(args.instance_table) as f:
        for i, r in enumerate(csv.DictReader(f)):
            r["frame_id"] = int(r["frame_id"])
            r["person_id"] = int(r["person_id"])
            r["feature_index"] = int(r["feature_index"])
            r["bbox_valid"] = r["bbox_valid"] == "True"
            r["bbox_area"] = int(r["bbox_area"])
            r["bbox_height"] = int(r["bbox_height"])
            assert int(r["feature_index"]) == i, f"feature_index mismatch at row {i}"
            inst.append(r)

    features = np.load(args.features)["features"]
    assert features.shape[0] == len(inst), f"features {features.shape[0]} != instance_table {len(inst)}"

    vis = {}
    with open(args.visibility) as f:
        for r in csv.DictReader(f):
            key = (int(r["frame_id"]), int(r["person_id"]), r["camera_id"])
            vis[key] = {
                "v_bbox_occlusion": float(r["v_bbox_occlusion"]),
                "v_bbox_valid": int(r["v_bbox_valid"]),
                "bbox_area": int(r["bbox_area"]),
                "bbox_height": int(r["bbox_height"]),
            }

    return inst, features, vis


def get_oracle_quality(vis_data):
    """Compute oracle quality score for a gallery view."""
    s_occ = 1.0 - vis_data["v_bbox_occlusion"]
    s_area = vis_data["bbox_area"]
    s_height = vis_data["bbox_height"]
    s_valid = vis_data["v_bbox_valid"]
    return s_occ, s_area, s_height, s_valid


def fuse_features(obs_rows, features, method, vis, q_feat=None):
    """Fuse gallery view features according to method.

    q_feat is only used by oracle_best_by_query (cheating method).
    """
    if method == "baseline_naive":
        feats = np.array([features[r["feature_index"]] for r in obs_rows])
        fused = np.mean(feats, axis=0)

    elif method == "oracle_weighted_occ":
        feats = np.array([features[r["feature_index"]] for r in obs_rows])
        ws = []
        for r in obs_rows:
            key = (r["frame_id"], r["person_id"], r["camera_id"])
            vd = vis.get(key, {})
            s_occ = 1.0 - vd.get("v_bbox_occlusion", 0.0)
            ws.append(max(s_occ, 1e-8))
        ws = np.array(ws, dtype=np.float32)
        ws = ws / ws.sum()
        fused = np.sum(ws[:, None] * feats, axis=0)

    elif method == "oracle_weighted_area":
        feats = np.array([features[r["feature_index"]] for r in obs_rows])
        ws = []
        for r in obs_rows:
            key = (r["frame_id"], r["person_id"], r["camera_id"])
            vd = vis.get(key, {})
            ws.append(max(float(vd.get("bbox_area", 1)), 1e-8))
        ws = np.array(ws, dtype=np.float32)
        ws = ws / ws.sum()
        fused = np.sum(ws[:, None] * feats, axis=0)

    elif method in ("oracle_topk1", "oracle_topk2"):
        k = 1 if method == "oracle_topk1" else 2
        scored = []
        for r in obs_rows:
            key = (r["frame_id"], r["person_id"], r["camera_id"])
            vd = vis.get(key, {})
            s_occ = 1.0 - vd.get("v_bbox_occlusion", 0.0)
            s_area = float(vd.get("bbox_area", 0))
            scored.append((r, s_occ, s_area))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        selected = scored[:max(k, 1)]
        feats = np.array([features[s[0]["feature_index"]] for s in selected])
        fused = np.mean(feats, axis=0)

    elif method == "oracle_threshold":
        scored = []
        for r in obs_rows:
            key = (r["frame_id"], r["person_id"], r["camera_id"])
            vd = vis.get(key, {})
            s_occ = 1.0 - vd.get("v_bbox_occlusion", 0.0)
            scored.append((r, s_occ))
        filtered = [s for s in scored if s[1] >= 0.5]
        if not filtered:
            filtered = scored
        feats = np.array([features[s[0]["feature_index"]] for s in filtered])
        fused = np.mean(feats, axis=0)

    elif method == "oracle_best_quality":
        best_r = None
        best_score = -1
        for r in obs_rows:
            key = (r["frame_id"], r["person_id"], r["camera_id"])
            vd = vis.get(key, {})
            s_occ = 1.0 - vd.get("v_bbox_occlusion", 0.0)
            s_area = float(vd.get("bbox_area", 0))
            score = s_occ * 10000 + s_area
            if score > best_score:
                best_score = score
                best_r = r
        fused = features[best_r["feature_index"]].copy()

    elif method == "oracle_best_by_query":
        assert q_feat is not None, "q_feat required for oracle_best_by_query"
        best_sim = -2.0
        best_feat = None
        for r in obs_rows:
            feat = features[r["feature_index"]]
            sim = float(feat @ q_feat)
            if sim > best_sim:
                best_sim = sim
                best_feat = feat
        fused = best_feat.copy()

    else:
        raise ValueError(f"Unknown method: {method}")

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
                    pid_fused = fuse_features(pid_other, features, method, vis,
                                             q_feat=q_feat if method == "oracle_best_by_query" else None)
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
        "per_camera": {
            cam: {
                "mAP": float(np.mean(per_cam_aps[cam])) if per_cam_aps[cam] else 0.0,
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
                    pid_fused = fuse_features(pid_other, features, method, vis,
                                             q_feat=q_feat if method == "oracle_best_by_query" else None)
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    inst, features, vis = load_data(args)
    print(f"[LOAD] instance_table: {len(inst)}, features: {features.shape}, visibility: {len(vis)}")
    print(f"[VERIFY] feature_index alignment: OK (all rows match)")
    print(f"[VERIFY] L2 norms: mean={np.linalg.norm(features, axis=1).mean():.6f}")

    methods = [
        "baseline_naive",
        "oracle_weighted_occ",
        "oracle_weighted_area",
        "oracle_topk1",
        "oracle_topk2",
        "oracle_threshold",
        "oracle_best_quality",
        "oracle_best_by_query",
    ]

    all_results = {}
    per_query_aps = {}

    for method in methods:
        tag = ""
        if method == "oracle_best_by_query":
            tag = " [CHEATING - uses query info]"
        print(f"\n{'='*60}")
        print(f"EVALUATING: {method}{tag}")
        print(f"{'='*60}")
        result = evaluate_method(inst, features, vis, method)
        all_results[method] = result
        print(f"  mAP={result['mAP']:.4f}  Rank-1={result['Rank-1']:.4f}  "
              f"Rank-5={result['Rank-5']:.4f}  Rank-10={result['Rank-10']:.4f}  "
              f"queries={result['num_queries']}  avg_gallery={result['avg_gallery_size']:.1f}")

    for method in methods:
        print(f"Computing per-query APs for {method}...")
        per_query_aps[method] = compute_per_query_aps(inst, features, vis, method)

    ttest_results = []
    legal_methods = [m for m in methods if m != "oracle_best_by_query"]
    for method in legal_methods:
        if method == "baseline_naive":
            continue
        t = paired_ttest(per_query_aps[method], per_query_aps["baseline_naive"], method, "baseline_naive")
        ttest_results.append(t)
        print(f"  {t['comparison']}: diff={t['mean_ap_diff']:.6f}, p={t['p_value']:.6f}, "
              f"sig={t['significant_005']}, wins={t['a_wins']}/{t['b_wins']}/{t['ties']}")

    t_cheat = paired_ttest(per_query_aps["oracle_best_by_query"], per_query_aps["baseline_naive"],
                           "oracle_best_by_query", "baseline_naive")
    ttest_results.append(t_cheat)
    print(f"  {t_cheat['comparison']}: diff={t_cheat['mean_ap_diff']:.6f}, p={t_cheat['p_value']:.6f}, "
          f"sig={t_cheat['significant_005']}, wins={t_cheat['a_wins']}/{t_cheat['b_wins']}/{t_cheat['ties']}")

    csv_path = os.path.join(args.output_dir, "oracle_ceiling_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mAP", "Rank-1", "Rank-5", "Rank-10",
                                                "num_queries", "skipped_queries", "avg_gallery_size", "is_cheating"])
        writer.writeheader()
        for method, m in all_results.items():
            writer.writerow({
                "method": method, "mAP": f"{m['mAP']:.6f}", "Rank-1": f"{m['Rank-1']:.6f}",
                "Rank-5": f"{m['Rank-5']:.6f}", "Rank-10": f"{m['Rank-10']:.6f}",
                "num_queries": m["num_queries"], "skipped_queries": m["skipped_queries"],
                "avg_gallery_size": f"{m['avg_gallery_size']:.2f}",
                "is_cheating": method == "oracle_best_by_query",
            })
    print(f"\n[SAVE] {csv_path}")

    json_path = os.path.join(args.output_dir, "oracle_ceiling_results.json")
    with open(json_path, "w") as f:
        json.dump({"results": all_results, "ttests": ttest_results}, f, indent=2)
    print(f"[SAVE] {json_path}")

    naive_map = all_results["baseline_naive"]["mAP"]
    best_legal_map = 0
    best_legal_method = ""
    for m in legal_methods:
        if m == "baseline_naive":
            continue
        if all_results[m]["mAP"] > best_legal_map:
            best_legal_map = all_results[m]["mAP"]
            best_legal_method = m
    G = (best_legal_map - naive_map) * 100
    cheat_map = all_results["oracle_best_by_query"]["mAP"]
    cheat_G = (cheat_map - naive_map) * 100

    best_ttest = None
    for t in ttest_results:
        if t["comparison"].startswith(best_legal_method):
            best_ttest = t
            break

    print(f"\n{'='*60}")
    print(f"ORACLE CEILING SUMMARY")
    print(f"{'='*60}")
    print(f"baseline_naive mAP:          {naive_map:.4f}")
    print(f"Best legal oracle:           {best_legal_method} mAP={best_legal_map:.4f}")
    print(f"G (legal oracle - naive):    {G:+.2f}% mAP")
    if best_ttest:
        print(f"  p-value: {best_ttest['p_value']:.6f}, significant: {best_ttest['significant_005']}")
        print(f"  95% CI: [{best_ttest['ci_95'][0]:.6f}, {best_ttest['ci_95'][1]:.6f}]")
        print(f"  wins/losses/ties: {best_ttest['a_wins']}/{best_ttest['b_wins']}/{best_ttest['ties']}")
    print(f"")
    print(f"[CHEATING] oracle_best_by_query mAP: {cheat_map:.4f}")
    print(f"[CHEATING] headroom over naive:      {cheat_G:+.2f}% mAP")
    print(f"")
    print(f"=== VERDICT ===")
    if G < 1.0:
        print(f"G = {G:+.2f}% < 1%: view-reliability weighting ceiling is TOO LOW.")
        print(f"  No real visibility signal (3DGS, attention, part-based) can exceed this.")
        print(f"  Recommendation: pivot to cross-frame/cross-scene evaluation where ceiling is higher.")
    elif G < 3.0:
        print(f"G = {G:+.2f}% (1-3%): marginal ceiling. Weighting has limited potential.")
        print(f"  Any real method will be strictly below this oracle ceiling.")
        print(f"  Recommendation: consider if this gain justifies the complexity.")
    else:
        print(f"G = {G:+.2f}% (>3%): meaningful ceiling. Weighting has real potential.")
        print(f"  Oracle mAP = {best_legal_map:.4f} is the upper bound for any real method.")

    if cheat_G < 2.0:
        print(f"  [CHEATING] headroom = {cheat_G:+.2f}% is also small -> same-frame task near saturation.")


if __name__ == "__main__":
    main()
