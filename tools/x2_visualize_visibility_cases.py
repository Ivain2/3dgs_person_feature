#!/usr/bin/env python3
"""X2 Task 6: Visualize visibility distributions and cases."""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAMERA_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance-table", default="outputs/x1_mv_fusion/instance_table.csv")
    p.add_argument("--features", default="outputs/v2_2d_reid_baseline/features.npz")
    p.add_argument("--visibility", default="outputs/x2_3dgs_visibility_fusion/visibility_scores.csv")
    p.add_argument("--eval-dir", default="outputs/x2_3dgs_visibility_fusion")
    p.add_argument("--output-dir", default="outputs/x2_3dgs_visibility_fusion/visualizations")
    return p.parse_args()


def load_visibility(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["v_depth_ratio_torso"] = float(r["v_depth_ratio_torso"])
            r["v_depth_ratio_foot"] = float(r["v_depth_ratio_foot"])
            r["v_depth_ratio_head"] = float(r["v_depth_ratio_head"])
            r["v_gaussian_count_torso"] = int(r["v_gaussian_count_torso"])
            r["v_weighted_opacity_torso"] = float(r["v_weighted_opacity_torso"])
            r["v_bbox_occlusion"] = float(r["v_bbox_occlusion"])
            r["v_weighted_3dgs"] = float(r["v_weighted_3dgs"])
            r["bbox_area"] = int(r["bbox_area"])
            r["bbox_height"] = int(r["bbox_height"])
            r["bbox_valid"] = r["bbox_valid"] == "True"
            rows.append(r)
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_visibility(args.visibility)
    valid = [r for r in rows if r["bbox_valid"] and r["v_depth_ratio_torso"] >= 0]
    print(f"[LOAD] {len(valid)} valid rows")

    # 1. Visibility distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    dr_torso = [r["v_depth_ratio_torso"] for r in valid]
    axes[0, 0].hist(dr_torso, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("v_depth_ratio_torso")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Depth Ratio Distribution (Torso)")
    axes[0, 0].axvline(np.mean(dr_torso), color="red", linestyle="--", label=f"mean={np.mean(dr_torso):.3f}")
    axes[0, 0].legend()

    gc = [r["v_gaussian_count_torso"] for r in valid]
    axes[0, 1].hist(gc, bins=range(0, max(gc) + 2), edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("v_gaussian_count_torso")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Gaussian Count Distribution (Torso)")

    wo = [r["v_weighted_opacity_torso"] for r in valid]
    axes[1, 0].hist(wo, bins=50, edgecolor="black", alpha=0.7)
    axes[1, 0].set_xlabel("v_weighted_opacity_torso")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Weighted Opacity Distribution (Torso)")

    dr_anchors = {
        "foot": [r["v_depth_ratio_foot"] for r in valid],
        "torso": [r["v_depth_ratio_torso"] for r in valid],
        "head": [r["v_depth_ratio_head"] for r in valid],
    }
    axes[1, 1].hist(dr_anchors["foot"], bins=50, alpha=0.5, label="foot")
    axes[1, 1].hist(dr_anchors["torso"], bins=50, alpha=0.5, label="torso")
    axes[1, 1].hist(dr_anchors["head"], bins=50, alpha=0.5, label="head")
    axes[1, 1].set_xlabel("v_depth_ratio")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Depth Ratio by Anchor")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "visibility_distribution.png"), dpi=150)
    plt.close()
    print(f"[SAVE] visibility_distribution.png")

    # 2. Visibility vs bbox_area
    fig, ax = plt.subplots(figsize=(8, 6))
    areas = [r["bbox_area"] for r in valid]
    drs = [r["v_depth_ratio_torso"] for r in valid]
    ax.scatter(areas, drs, alpha=0.3, s=10)
    ax.set_xlabel("bbox_area (px^2)")
    ax.set_ylabel("v_depth_ratio_torso")
    ax.set_title("Visibility vs Bbox Area")
    corr = np.corrcoef(areas, drs)[0, 1]
    ax.text(0.05, 0.95, f"r={corr:.3f}", transform=ax.transAxes, va="top")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "visibility_vs_bbox_area.png"), dpi=150)
    plt.close()
    print(f"[SAVE] visibility_vs_bbox_area.png")

    # 3. Per-camera visibility
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cam_dr_means = []
    cam_dr_stds = []
    cam_gc_means = []
    for cam in CAMERA_NAMES:
        cam_rows = [r for r in valid if r["camera_id"] == cam]
        cam_dr_means.append(np.mean([r["v_depth_ratio_torso"] for r in cam_rows]))
        cam_dr_stds.append(np.std([r["v_depth_ratio_torso"] for r in cam_rows]))
        cam_gc_means.append(np.mean([r["v_gaussian_count_torso"] for r in cam_rows]))

    axes[0].bar(CAMERA_NAMES, cam_dr_means, yerr=cam_dr_stds, capsize=5, edgecolor="black", alpha=0.7)
    axes[0].set_ylabel("v_depth_ratio_torso (mean ± std)")
    axes[0].set_title("Per-Camera Depth Ratio")

    axes[1].bar(CAMERA_NAMES, cam_gc_means, edgecolor="black", alpha=0.7)
    axes[1].set_ylabel("v_gaussian_count_torso (mean)")
    axes[1].set_title("Per-Camera Gaussian Count")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "per_camera_visibility.png"), dpi=150)
    plt.close()
    print(f"[SAVE] per_camera_visibility.png")

    print("[DONE] All visualizations generated")


if __name__ == "__main__":
    main()
