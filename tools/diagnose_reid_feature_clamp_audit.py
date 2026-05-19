#!/usr/bin/env python3
"""V3.0.4-A: Diagnose ReID Feature Clamp Range.

Audit whether real ReID features would trigger the CUDA clamp in
radianceFromSpH: rad = clamp(0.5 + SH_C0 * f, 0, 1).

If clamp ratio is negligible, no need to modify CUDA clamp.
If clamp ratio is significant, a no-clamp feature renderer is needed.

Usage:
  python tools/diagnose_reid_feature_clamp_range.py \
    --v2_detections outputs/v2_1_2d_reid_full/detections.csv \
    --v2_features outputs/v2_1_2d_reid_full/features.npz \
    --teacher_prototypes /data02/zhangrunxiang/data/Wildtrack/reid_teacher_prototypes.pt \
    --registered_features outputs/v3_0_2_pure_registration/registered_features.pt \
    --aggregated_features outputs/v3_0_1_small_full/aggregated_features.pt \
    --out_dir outputs/v3_0_4_clamp_audit/
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SH_C0 = 0.28209479177387814


def compute_clamp_stats(feat_array, name):
    """Compute clamp statistics for a feature array.

    x = 0.5 + SH_C0 * f
    Returns dict with clamp ratios and feature statistics.
    """
    x = 0.5 + SH_C0 * feat_array

    stats = {
        "name": name,
        "num_features": feat_array.shape[0],
        "feature_dim": feat_array.shape[1] if len(feat_array.shape) > 1 else 1,

        # Clamp statistics
        "clamp_low_ratio": float((x < 0).mean()),
        "clamp_high_ratio": float((x > 1).mean()),
        "clamp_total_ratio": float(((x < 0) | (x > 1)).mean()),

        # X statistics (pre-clamp radiance)
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "x_mean": float(x.mean()),
        "x_std": float(x.std()),
        "x_p1": float(np.percentile(x, 1)),
        "x_p5": float(np.percentile(x, 5)),
        "x_p50": float(np.percentile(x, 50)),
        "x_p95": float(np.percentile(x, 95)),
        "x_p99": float(np.percentile(x, 99)),

        # Feature statistics
        "f_min": float(feat_array.min()),
        "f_max": float(feat_array.max()),
        "f_mean": float(feat_array.mean()),
        "f_std": float(feat_array.std()),
        "f_p1": float(np.percentile(feat_array, 1)),
        "f_p5": float(np.percentile(feat_array, 5)),
        "f_p50": float(np.percentile(feat_array, 50)),
        "f_p95": float(np.percentile(feat_array, 95)),
        "f_p99": float(np.percentile(feat_array, 99)),

        # Per-channel std
        "channel_std_mean": float(feat_array.std(axis=0).mean()),
        "channel_std_min": float(feat_array.std(axis=0).min()),
        "channel_std_max": float(feat_array.std(axis=0).max()),

        # Feature norm
        "feature_norm_mean": float(np.linalg.norm(feat_array, axis=1).mean()),
        "feature_norm_std": float(np.linalg.norm(feat_array, axis=1).std()),
    }

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_detections", default="outputs/v2_1_2d_reid_full/detections.csv")
    parser.add_argument("--v2_features", default="outputs/v2_1_2d_reid_full/features.npz")
    parser.add_argument("--teacher_prototypes", default="/data02/zhangrunxiang/data/Wildtrack/reid_teacher_prototypes.pt")
    parser.add_argument("--registered_features", default="outputs/v3_0_2_pure_registration/registered_features.pt")
    parser.add_argument("--aggregated_features", default="outputs/v3_0_1_small_full/aggregated_features.pt")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("V3.0.4-A: ReID Feature Clamp Range Audit")
    print("=" * 60)
    print(f"  SH_C0 = {SH_C0}")
    print(f"  clamp: rad = clamp(0.5 + SH_C0 * f, 0, 1)")
    print()

    all_stats = []

    # 1. V2 detection embeddings
    print("[1/5] Loading V2 detection embeddings...")
    v2_feat = np.load(args.v2_features)["features"].astype(np.float32)
    print(f"  Shape: {v2_feat.shape}")
    stats = compute_clamp_stats(v2_feat, "v2_detection_embeddings")
    all_stats.append(stats)
    print(f"  clamp_low_ratio={stats['clamp_low_ratio']:.6f}, clamp_high_ratio={stats['clamp_high_ratio']:.6f}")
    print(f"  f_min={stats['f_min']:.4f}, f_max={stats['f_max']:.4f}")
    print()

    # 2. Teacher prototypes
    print("[2/5] Loading teacher prototypes...")
    try:
        proto_data = torch.load(args.teacher_prototypes, map_location="cpu", weights_only=False)
        if isinstance(proto_data, dict):
            # Try common keys
            for key in ["prototypes", "features", "teacher_features", "mean_features"]:
                if key in proto_data:
                    teacher_feat = proto_data[key].numpy()
                    break
            else:
                # Use the first tensor value
                for k, v in proto_data.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 2:
                        teacher_feat = v.numpy()
                        break
                else:
                    teacher_feat = None
        elif isinstance(proto_data, torch.Tensor):
            teacher_feat = proto_data.numpy()
        else:
            teacher_feat = None

        if teacher_feat is not None:
            print(f"  Shape: {teacher_feat.shape}")
            stats = compute_clamp_stats(teacher_feat, "teacher_prototypes")
            all_stats.append(stats)
            print(f"  clamp_low_ratio={stats['clamp_low_ratio']:.6f}, clamp_high_ratio={stats['clamp_high_ratio']:.6f}")
        else:
            print(f"  [WARN] Could not extract features from prototypes dict. Keys: {list(proto_data.keys()) if isinstance(proto_data, dict) else type(proto_data)}")
    except Exception as e:
        print(f"  [WARN] Failed to load teacher prototypes: {e}")
    print()

    # 3. Registered beta features
    print("[3/5] Loading registered features...")
    try:
        reg_data = torch.load(args.registered_features, map_location="cpu", weights_only=False)

        # Try different feature keys
        feature_keys = {}
        for key in ["features_beta", "features_purity07", "features", "registered_features"]:
            if key in reg_data:
                feature_keys[key] = reg_data[key].numpy()

        if not feature_keys:
            print(f"  [WARN] No feature keys found. Available: {list(reg_data.keys())}")
        else:
            for key, reg_feat in feature_keys.items():
                print(f"  {key} Shape: {reg_feat.shape}")
                # Stats for all
                stats = compute_clamp_stats(reg_feat, f"registered_{key}")
                all_stats.append(stats)
                print(f"  [{key}] clamp_low={stats['clamp_low_ratio']:.6f}, clamp_high={stats['clamp_high_ratio']:.6f}")

                # Also check valid subset (beta mask)
                if "beta" in reg_data and reg_feat.shape[0] == reg_data["beta"].shape[0]:
                    beta = reg_data["beta"].numpy()
                    beta_mask = beta > 1e-6
                    if beta_mask.sum() > 0:
                        beta_feat = reg_feat[beta_mask]
                        stats = compute_clamp_stats(beta_feat, f"registered_{key}_valid_beta")
                        all_stats.append(stats)
                        print(f"  [{key}_valid_beta] clamp_low={stats['clamp_low_ratio']:.6f}, clamp_high={stats['clamp_high_ratio']:.6f}")
    except Exception as e:
        print(f"  [WARN] Failed to load registered features: {e}")
    print()

    # 4. Aggregated gaussian_features_raw and l2
    print("[4/5] Loading aggregated features...")
    try:
        agg_data = torch.load(args.aggregated_features, map_location="cpu", weights_only=False)

        for key, name in [("gaussian_features_raw", "aggregated_raw"), ("gaussian_features_l2", "aggregated_l2")]:
            if key in agg_data:
                agg_feat = agg_data[key].numpy()
                print(f"  {key} Shape: {agg_feat.shape}")

                # All
                stats = compute_clamp_stats(agg_feat, name)
                all_stats.append(stats)
                print(f"  [{name}] clamp_low={stats['clamp_low_ratio']:.6f}, clamp_high={stats['clamp_high_ratio']:.6f}")

                # Valid mask subset
                if "valid_gaussian_mask" in agg_data and key == "gaussian_features_raw":
                    valid_mask = agg_data["valid_gaussian_mask"].numpy()
                    valid_feat = agg_feat[valid_mask]
                    if valid_feat.shape[0] > 0:
                        stats = compute_clamp_stats(valid_feat, f"{name}_valid_only")
                        all_stats.append(stats)
                        print(f"  [{name}_valid] clamp_low={stats['clamp_low_ratio']:.6f}, clamp_high={stats['clamp_high_ratio']:.6f}")
    except Exception as e:
        print(f"  [WARN] Failed to load aggregated features: {e}")
    print()

    # 5. Also test: what if features are L2-normalized per-row (like V2)?
    print("[5/5] Computing L2-normalized versions of V2 features...")
    v2_l2 = v2_feat / (np.linalg.norm(v2_feat, axis=1, keepdims=True) + 1e-8)
    stats = compute_clamp_stats(v2_l2, "v2_l2_normalized")
    all_stats.append(stats)
    print(f"  clamp_low_ratio={stats['clamp_low_ratio']:.6f}, clamp_high_ratio={stats['clamp_high_ratio']:.6f}")
    print()

    # Save results
    print("Saving results...")
    with open(os.path.join(args.out_dir, "clamp_audit_summary.json"), "w") as f:
        json.dump({s["name"]: {k: v for k, v in s.items() if k != "name"} for s in all_stats}, f, indent=2)

    # CSV
    with open(os.path.join(args.out_dir, "clamp_audit.csv"), "w", newline="") as f:
        fieldnames = ["name", "num_features", "feature_dim",
                      "clamp_low_ratio", "clamp_high_ratio", "clamp_total_ratio",
                      "x_min", "x_max", "x_p1", "x_p50", "x_p99",
                      "f_min", "f_max", "f_p1", "f_p50", "f_p99",
                      "channel_std_mean", "feature_norm_mean"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in all_stats:
            row = {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in s.items() if k in fieldnames}
            w.writerow(row)

    # Final report
    # Determine conclusion
    need_cuda_change = False
    reasons = []
    for s in all_stats:
        if s["clamp_low_ratio"] > 0.001 or s["clamp_high_ratio"] > 0.001:
            need_cuda_change = True
            reasons.append(f"{s['name']}: low={s['clamp_low_ratio']:.4f}, high={s['clamp_high_ratio']:.4f}")

    report = "# V3.0.4-A Clamp Audit Report\n\n"
    report += f"## Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
    report += "## Configuration\n\n"
    report += f"- SH_C0 = {SH_C0}\n"
    report += f"- Formula: x = 0.5 + SH_C0 * f\n"
    report += f"- Clamp: rad = clamp(x, 0, 1)\n\n"

    report += "## Clamp Statistics\n\n"
    report += "| Feature Set | Count | Clamp Low Ratio | Clamp High Ratio | f_min | f_max | Channel Std |\n"
    report += "|-------------|-------|-----------------|------------------|-------|-------|-------------|\n"
    for s in all_stats:
        report += f"| {s['name']} | {s['num_features']} | {s['clamp_low_ratio']:.6f} | {s['clamp_high_ratio']:.6f} | {s['f_min']:.4f} | {s['f_max']:.4f} | {s['channel_std_mean']:.6f} |\n"
    report += "\n"

    report += "## Percentile Analysis (X = 0.5 + SH_C0 * f)\n\n"
    report += "| Feature Set | x_min | x_p1 | x_p50 | x_p99 | x_max |\n"
    report += "|-------------|-------|------|-------|-------|-------|\n"
    for s in all_stats:
        report += f"| {s['name']} | {s['x_min']:.4f} | {s['x_p1']:.4f} | {s['x_p50']:.4f} | {s['x_p99']:.4f} | {s['x_max']:.4f} |\n"
    report += "\n"

    report += "## Conclusion\n\n"
    if need_cuda_change:
        report += f"**CLAMP DETECTED**: {len(reasons)} feature sets have clamp ratio > 0.001\n\n"
        for r in reasons:
            report += f"- {r}\n"
        report += "\n**Recommendation**: Consider no-clamp feature renderer for feature rendering.\n"
    else:
        report += "**NO SIGNIFICANT CLAMP**: All feature sets have clamp ratio < 0.001\n\n"
        report += "The CUDA clamp in radianceFromSpH is NOT the primary cause of feature collapse.\n"
        report += "The issue lies elsewhere (common component, global aggregation, or ROI pooling).\n"

    with open(os.path.join(args.out_dir, "final_report.md"), "w") as f:
        f.write(report)

    print("=" * 60)
    print("V3.0.4-A Clamp Audit Complete")
    print(f"  Need CUDA change: {need_cuda_change}")
    if need_cuda_change:
        for r in reasons:
            print(f"  - {r}")
    else:
        print("  No significant clamp detected. Focus on other causes.")
    print(f"  Report: {args.out_dir}/final_report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
