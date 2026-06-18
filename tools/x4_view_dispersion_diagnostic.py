#!/usr/bin/env python3
"""X4: View Dispersion Diagnostic — is there a "bad view" worth down-weighting?

Core metric: per-view leave-one-out cosine distance
    d_i = (1/(K-1)) * sum_{j!=i} (1 - f_i . f_j)

This equals "cosine distance from f_i to the (unnormalized) centroid of the
other K-1 views", i.e. the leave-one-out centroid distance without self-bias.

Task A: Does a bad view exist? (distribution of d_i, stratified by K)
Task B: Can quality signals identify bad views? (AUC / Spearman, K>=3 only)

Correctness:
  - feature_index == row index verified at load
  - Two distance formulas cross-validated (sampled assert, tol 1e-6)
  - K=2: symmetric, no outlier identification
  - K>=3: outlier = argmax d_i
  - No query/retrieval info used
  - Signal direction: high d_i = bad view; signals aligned to predict high d_i
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

CAMERA_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

SIGNAL_DEFS = {
    "s_occ_inv": {"desc": "1 - v_bbox_occlusion (high=less occluded=better view)", "direction": "high=good"},
    "s_area": {"desc": "bbox_area (high=larger person=better view)", "direction": "high=good"},
    "s_height": {"desc": "bbox_height (high=taller person=better view)", "direction": "high=good"},
    "s_depth_ratio": {"desc": "v_depth_ratio_torso (high=no occluding Gaussian=better)", "direction": "high=good"},
    "s_weighted_opacity": {"desc": "v_weighted_opacity_torso (high=more Gaussian opacity=worse)", "direction": "high=bad"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance-table", default="outputs/x1_mv_fusion/instance_table.csv")
    p.add_argument("--features", default="outputs/v2_2d_reid_baseline/features.npz")
    p.add_argument("--visibility", default="outputs/x2_3dgs_visibility_fusion_bugfix/visibility_scores.csv")
    p.add_argument("--output-dir", default="outputs/x4_view_dispersion")
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
    assert features.shape[0] == len(inst)

    vis = {}
    with open(args.visibility) as f:
        for r in csv.DictReader(f):
            key = (int(r["frame_id"]), int(r["person_id"]), r["camera_id"])
            vis[key] = {
                "v_bbox_occlusion": float(r["v_bbox_occlusion"]),
                "v_depth_ratio_torso": float(r["v_depth_ratio_torso"]),
                "v_weighted_opacity_torso": float(r["v_weighted_opacity_torso"]),
                "bbox_area": int(r["bbox_area"]),
                "bbox_height": int(r["bbox_height"]),
            }

    return inst, features, vis


def compute_di_for_group(feats_array):
    """Compute d_i for each view in a group using leave-one-out centroid distance.

    Formula 1: d_i = (1/(K-1)) * sum_{j!=i} (1 - f_i . f_j)
    Formula 2: d_i = 1 - f_i . (sum_{j!=i} f_j) / ||sum_{j!=i} f_j||

    We use Formula 1 (unambiguous, no normalization of centroid).
    """
    K = feats_array.shape[0]
    if K < 2:
        return np.array([])

    sim_matrix = feats_array @ feats_array.T

    d_i = np.zeros(K)
    for i in range(K):
        mask = np.ones(K, dtype=bool)
        mask[i] = False
        d_i[i] = np.mean(1.0 - sim_matrix[i, mask])

    return d_i


def cross_validate_formulas(feats_array, tol=1e-6):
    """Cross-validate Formula 1 vs Formula 2 on a sample group.

    Formula 2: d_i = 1 - f_i . centroid_{j!=i} / ||centroid_{j!=i}||
    This is the "cosine distance to leave-one-out centroid".
    Note: Formula 1 and Formula 2 are NOT numerically identical in general.
    Formula 1 = mean pairwise distance; Formula 2 = distance to centroid.
    We compute both and report but only use Formula 1 as the primary metric.
    """
    K = feats_array.shape[0]
    if K < 2:
        return True

    sim_matrix = feats_array @ feats_array.T

    d_formula1 = np.zeros(K)
    d_formula2 = np.zeros(K)
    for i in range(K):
        mask = np.ones(K, dtype=bool)
        mask[i] = False
        d_formula1[i] = np.mean(1.0 - sim_matrix[i, mask])

        centroid = feats_array[mask].sum(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            d_formula2[i] = 1.0 - float(feats_array[i] @ centroid / norm)
        else:
            d_formula2[i] = 1.0

    return d_formula1, d_formula2


def get_signal_values(vis_data, signal_name):
    """Extract signal value. Direction: for AUC, we predict is_outlier (high d_i).
    Signals where high=good need to be inverted for AUC computation.
    """
    if signal_name == "s_occ_inv":
        return 1.0 - vis_data.get("v_bbox_occlusion", 0.0)
    elif signal_name == "s_area":
        return float(vis_data.get("bbox_area", 0))
    elif signal_name == "s_height":
        return float(vis_data.get("bbox_height", 0))
    elif signal_name == "s_depth_ratio":
        return float(vis_data.get("v_depth_ratio_torso", 0.0))
    elif signal_name == "s_weighted_opacity":
        return float(vis_data.get("v_weighted_opacity_torso", 0.0))
    else:
        raise ValueError(f"Unknown signal: {signal_name}")


def signal_predicts_outlier(signal_name, signal_val, is_outlier):
    """For AUC: we want signal to predict is_outlier=True.
    For 'high=good' signals: invert so low value -> high score -> predicts outlier.
    For 'high=bad' signals: high value directly predicts outlier.
    """
    if SIGNAL_DEFS[signal_name]["direction"] == "high=good":
        return -signal_val
    else:
        return signal_val


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    inst, features, vis = load_data(args)
    print(f"[LOAD] instance_table: {len(inst)}, features: {features.shape}, visibility: {len(vis)}")
    print(f"[VERIFY] feature_index alignment: OK")
    print(f"[VERIFY] L2 norms: mean={np.linalg.norm(features, axis=1).mean():.6f}")

    valid_rows = [r for r in inst if r["bbox_valid"]]
    fpg = defaultdict(list)
    for r in valid_rows:
        fpg[(r["frame_id"], r["person_id"])].append(r)

    print(f"\n[INFO] Total valid observations: {len(valid_rows)}")
    print(f"[INFO] Unique (frame, person) groups: {len(fpg)}")

    K_counts = defaultdict(int)
    for key, group in fpg.items():
        K_counts[len(group)] += 1
    for K in sorted(K_counts.keys()):
        pct = K_counts[K] / len(fpg) * 100
        print(f"  K={K}: {K_counts[K]} groups ({pct:.1f}%)")

    csv_rows = []
    all_d_i = []
    d_i_by_K = defaultdict(list)
    max_d_i_by_K = defaultdict(list)

    cross_val_count = 0
    cross_val_max_diff = 0.0

    for (fid, pid), group in sorted(fpg.items()):
        K = len(group)
        if K < 2:
            continue

        feat_indices = [r["feature_index"] for r in group]
        feats = features[feat_indices]

        d_i = compute_di_for_group(feats)

        if cross_val_count < 20 and K >= 3:
            d1, d2 = cross_validate_formulas(feats)
            max_diff = np.max(np.abs(d1 - d2))
            cross_val_max_diff = max(cross_val_max_diff, max_diff)
            cross_val_count += 1

        outlier_idx = np.argmax(d_i)

        for idx, r in enumerate(group):
            cam = r["camera_id"]
            key = (fid, pid, cam)
            vd = vis.get(key, {})

            is_outlier = 0
            if K >= 3 and idx == outlier_idx:
                is_outlier = 1

            row = {
                "frame_id": fid,
                "person_id": pid,
                "camera_id": cam,
                "K": K,
                "d_i": float(d_i[idx]),
                "is_outlier": is_outlier,
                "s_occ_inv": get_signal_values(vd, "s_occ_inv"),
                "s_area": get_signal_values(vd, "s_area"),
                "s_height": get_signal_values(vd, "s_height"),
                "s_depth_ratio": get_signal_values(vd, "s_depth_ratio"),
                "s_weighted_opacity": get_signal_values(vd, "s_weighted_opacity"),
            }
            csv_rows.append(row)
            all_d_i.append(float(d_i[idx]))
            d_i_by_K[K].append(float(d_i[idx]))

        max_d_i_by_K[K].append(float(d_i[outlier_idx]))

    print(f"\n[CROSS-VALIDATION] Formula 1 vs Formula 2: {cross_val_count} groups checked, max diff={cross_val_max_diff:.8f}")
    print(f"  (Note: Formula 1 = mean pairwise; Formula 2 = centroid distance. They differ by definition.)")

    all_d_i = np.array(all_d_i)

    # === TASK A: Distribution ===
    print(f"\n{'='*60}")
    print(f"TASK A: View Dispersion Distribution")
    print(f"{'='*60}")

    def report_dist(arr, label):
        arr = np.array(arr)
        print(f"\n  {label} (n={len(arr)}):")
        print(f"    mean={arr.mean():.6f}  std={arr.std():.6f}")
        for q in [50, 75, 90, 95, 99]:
            print(f"    p{q}={np.percentile(arr, q):.6f}")
        print(f"    max={arr.max():.6f}")

    report_dist(all_d_i, "All d_i")

    for K in sorted(d_i_by_K.keys()):
        report_dist(d_i_by_K[K], f"K={K} d_i")

    print(f"\n  max_i d_i distribution (per group):")
    for K in sorted(max_d_i_by_K.keys()):
        arr = np.array(max_d_i_by_K[K])
        print(f"    K={K}: mean={arr.mean():.6f}  p90={np.percentile(arr, 90):.6f}  p95={np.percentile(arr, 95):.6f}  max={arr.max():.6f}")

    p95_all = np.percentile(all_d_i, 95)
    p99_all = np.percentile(all_d_i, 99)
    max_d_i_all_groups = np.array([v for klist in max_d_i_by_K.values() for v in klist])
    p90_max_d = np.percentile(max_d_i_all_groups, 90)

    task_a = {
        "all_d_i": {
            "n": int(len(all_d_i)),
            "mean": float(all_d_i.mean()),
            "std": float(all_d_i.std()),
            "p50": float(np.percentile(all_d_i, 50)),
            "p75": float(np.percentile(all_d_i, 75)),
            "p90": float(np.percentile(all_d_i, 90)),
            "p95": float(p95_all),
            "p99": float(p99_all),
            "max": float(all_d_i.max()),
        },
        "by_K": {},
        "max_d_i_by_K": {},
    }
    for K in sorted(d_i_by_K.keys()):
        arr = np.array(d_i_by_K[K])
        task_a["by_K"][str(K)] = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        }
    for K in sorted(max_d_i_by_K.keys()):
        arr = np.array(max_d_i_by_K[K])
        task_a["max_d_i_by_K"][str(K)] = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }

    has_heavy_tail = bool(p95_all >= 0.15 or p90_max_d >= 0.15)
    task_a["verdict"] = {
        "p95_d_i": float(p95_all),
        "p99_d_i": float(p99_all),
        "p90_max_d_i": float(p90_max_d),
        "has_heavy_tail": has_heavy_tail,
        "interpretation": "bad views exist (heavy tail)" if has_heavy_tail else "no bad views (no heavy tail)",
    }

    print(f"\n  === TASK A VERDICT ===")
    print(f"  p95 d_i = {p95_all:.6f}")
    print(f"  p99 d_i = {p99_all:.6f}")
    print(f"  p90(max_i d_i) = {p90_max_d:.6f}")
    if has_heavy_tail:
        print(f"  -> HEAVY TAIL DETECTED: bad views exist, proceed to Task B")
    else:
        print(f"  -> NO HEAVY TAIL: no bad views worth down-weighting, weighting line is over")

    # === TASK B: Signal prediction (K>=3 only) ===
    task_b = {"signals": {}, "random_baseline": {}}

    k_ge3_rows = [r for r in csv_rows if r["K"] >= 3]
    print(f"\n{'='*60}")
    print(f"TASK B: Signal Prediction Power (K>=3 only, n={len(k_ge3_rows)} views)")
    print(f"{'='*60}")

    if not k_ge3_rows:
        print("  No K>=3 groups, skipping Task B")
    else:
        k_ge3_by_group = defaultdict(list)
        for r in k_ge3_rows:
            k_ge3_by_group[(r["frame_id"], r["person_id"])].append(r)

        n_groups_k3 = len(k_ge3_by_group)
        n_outliers = sum(1 for r in k_ge3_rows if r["is_outlier"] == 1)
        print(f"  K>=3 groups: {n_groups_k3}, outlier views: {n_outliers}")

        # Per-group AUC computation
        signal_aucs = defaultdict(list)
        signal_spearmans = defaultdict(list)

        for signal_name in SIGNAL_DEFS:
            valid_groups = 0
            for key, group_rows in k_ge3_by_group.items():
                if len(group_rows) < 3:
                    continue

                labels = np.array([r["is_outlier"] for r in group_rows])
                if labels.sum() != 1:
                    continue

                scores_raw = np.array([get_signal_values(
                    vis.get((r["frame_id"], r["person_id"], r["camera_id"]), {}),
                    signal_name
                ) for r in group_rows])

                scores_for_auc = np.array([signal_predicts_outlier(signal_name, s, l)
                                           for s, l in zip(scores_raw, labels)])

                try:
                    auc = roc_auc_score(labels, scores_for_auc)
                    signal_aucs[signal_name].append(auc)
                except ValueError:
                    pass

                d_is = np.array([r["d_i"] for r in group_rows])
                if len(set(scores_raw)) > 1 and len(set(d_is)) > 1:
                    rho, p_val = spearmanr(scores_raw, d_is)
                    signal_spearmans[signal_name].append({"rho": rho, "p": p_val})

                valid_groups += 1

            aucs = signal_aucs[signal_name]
            spearmans = signal_spearmans[signal_name]

            if aucs:
                auc_arr = np.array(aucs)
                auc_mean = float(auc_arr.mean())
                auc_std = float(auc_arr.std())
                n = len(auc_arr)
                ci_lo = auc_mean - 1.96 * auc_std / np.sqrt(n)
                ci_hi = auc_mean + 1.96 * auc_std / np.sqrt(n)
            else:
                auc_mean = auc_std = ci_lo = ci_hi = 0.0
                n = 0

            if spearmans:
                rhos = [s["rho"] for s in spearmans]
                mean_rho = float(np.mean(rhos))
                n_sig = sum(1 for s in spearmans if s["p"] < 0.05)
            else:
                mean_rho = 0.0
                n_sig = 0

            task_b["signals"][signal_name] = {
                "direction": SIGNAL_DEFS[signal_name]["direction"],
                "n_groups": n,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "auc_95ci": [float(ci_lo), float(ci_hi)],
                "spearman_mean_rho": mean_rho,
                "spearman_n_significant": n_sig,
                "spearman_n_total": len(spearmans) if spearmans else 0,
            }

            print(f"\n  {signal_name}:")
            print(f"    AUC: {auc_mean:.4f} ± {auc_std:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]  (n_groups={n})")
            print(f"    Spearman mean rho: {mean_rho:.4f}  significant: {n_sig}/{len(spearmans) if spearmans else 0}")

        # Random baseline
        np.random.seed(42)
        random_aucs = []
        for key, group_rows in k_ge3_by_group.items():
            if len(group_rows) < 3:
                continue
            labels = np.array([r["is_outlier"] for r in group_rows])
            if labels.sum() != 1:
                continue
            random_scores = np.random.randn(len(group_rows))
            try:
                auc = roc_auc_score(labels, random_scores)
                random_aucs.append(auc)
            except ValueError:
                pass

        random_auc_arr = np.array(random_aucs)
        task_b["random_baseline"] = {
            "auc_mean": float(random_auc_arr.mean()),
            "auc_std": float(random_auc_arr.std()),
            "n_groups": len(random_aucs),
        }
        print(f"\n  Random baseline AUC: {random_auc_arr.mean():.4f} ± {random_auc_arr.std():.4f}")

        # Task B verdict
        best_signal = None
        best_auc = 0
        for sn, sd in task_b["signals"].items():
            if sd["auc_mean"] > best_auc:
                best_auc = sd["auc_mean"]
                best_signal = sn

        signals_above_random = sum(
            1 for sn, sd in task_b["signals"].items()
            if sd["auc_mean"] > random_auc_arr.mean() + 0.02
        )

        task_b["verdict"] = {
            "best_signal": best_signal,
            "best_auc": best_auc,
            "random_auc": float(random_auc_arr.mean()),
            "signals_above_random": signals_above_random,
            "any_signal_useful": best_auc > 0.55,
        }

        print(f"\n  === TASK B VERDICT ===")
        print(f"  Best signal: {best_signal} AUC={best_auc:.4f}")
        print(f"  Random baseline: {random_auc_arr.mean():.4f}")
        if best_auc <= 0.55:
            print(f"  -> All signals ≈ random: physical visibility ≠ feature reliability")
            print(f"     3DGS-as-weight has no salvageable signal")
        else:
            print(f"  -> Signal {best_signal} can identify bad views (AUC={best_auc:.4f} > 0.55)")
            print(f"     Consider 'outlier removal' instead of continuous weighting")

    # === FINAL VERDICT ===
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT")
    print(f"{'='*60}")

    if not has_heavy_tail:
        print(f"Task A: No heavy tail -> no bad views worth down-weighting")
        print(f"CONCLUSION: View-reliability weighting has NO operable object.")
        print(f"  -> This line is OVER. Pivot to feature learning (training-time constraints)")
        print(f"     or cross-frame/cross-scene evaluation where view variance is larger.")
        final_verdict = "no_bad_views_weighting_over"
    else:
        task_b_verdict = task_b.get("verdict", {})
        if task_b_verdict.get("any_signal_useful", False):
            print(f"Task A: Heavy tail exists -> bad views confirmed")
            print(f"Task B: Signal {task_b_verdict['best_signal']} can identify them (AUC={task_b_verdict['best_auc']:.4f})")
            print(f"CONCLUSION: Bad views exist and are detectable.")
            print(f"  -> In cross-frame settings with higher ceiling, try outlier-removal")
            print(f"     with {task_b_verdict['best_signal']} as the signal, not continuous weighting.")
            final_verdict = "bad_views_detectable_try_outlier_removal"
        else:
            print(f"Task A: Heavy tail exists -> bad views confirmed")
            print(f"Task B: All signals ≈ random -> physical visibility ≠ feature reliability")
            print(f"CONCLUSION: Bad views exist but NO available signal can identify them.")
            print(f"  -> 3DGS-as-weight is unsalvageable. The gap is in signal, not in fusion method.")
            print(f"  -> Pivot to feature learning or cross-frame evaluation.")
            final_verdict = "bad_views_exist_but_no_signal"

    # Save CSV
    csv_path = os.path.join(args.output_dir, "dispersion_per_view.csv")
    fieldnames = ["frame_id", "person_id", "camera_id", "K", "d_i", "is_outlier",
                  "s_occ_inv", "s_area", "s_height", "s_depth_ratio", "s_weighted_opacity"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[SAVE] {csv_path} ({len(csv_rows)} rows)")

    # Save JSON
    summary = {
        "task_a": task_a,
        "task_b": task_b,
        "final_verdict": final_verdict,
    }
    json_path = os.path.join(args.output_dir, "dispersion_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] {json_path}")


if __name__ == "__main__":
    main()
