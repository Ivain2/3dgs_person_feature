#!/usr/bin/env python3
"""X2 Task 1: Compute visibility scores for each (frame, person, camera) triple.

Visibility metrics:
1. v_projection: anchor projects inside image (1/0)
2. v_bbox_valid: person has valid bbox in this camera (1/0)
3. v_3dgs_depth_ratio: ratio of nearest Gaussian depth to anchor depth at projected pixel
   - 1.0 = no Gaussian in front of anchor (fully visible)
   - <1.0 = Gaussian closer than anchor (partially occluded)
4. v_3dgs_gaussian_count: number of Gaussians near the projected anchor pixel
5. v_bbox_occlusion: 1 - max_IoU with other persons' bboxes (crowding measure)
6. v_weighted_3dgs: combined metric

NOTE: Standard 3DGS transmittance is NOT used because the model was trained
without person masking. Background Gaussians fill the entire scene including
person locations, making transmittance ≈ 0 everywhere. Instead, we use
depth-based and count-based metrics that measure occlusion from 3D geometry.

Coordinate system:
- WildTrack world coordinates in cm
- positionID = y_grid * NB_WIDTH(480) + x_grid
- world_x = ORIGINE_X(-3.0m) + x_grid * cell_size(2.5cm)
- world_y = ORIGINE_Y(-9.0m) + y_grid * cell_size(2.5cm)
- Person anchors: foot=(x,y,0), torso=(x,y,80), head=(x,y,170) cm
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch

CAMERA_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
VIEWNUM_TO_CAM = {i: c for i, c in enumerate(CAMERA_NAMES)}
CAM_TO_VIEWNUM = {c: i for i, c in enumerate(CAMERA_NAMES)}

POM_NB_WIDTH = 480
POM_ORIGINE_X = -3.0
POM_ORIGINE_Y = -9.0
POM_CELL_SIZE = 2.5 / 100.0

ANCHOR_HEIGHTS = {"foot": 0.0, "torso": 80.0, "head": 170.0}
PIXEL_RADIUS = 15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data02/zhangrunxiang/data/Wildtrack")
    p.add_argument("--instance-table", default="outputs/x1_mv_fusion/instance_table.csv")
    p.add_argument("--detections", default="outputs/v2_2d_reid_baseline/detections.csv")
    p.add_argument("--checkpoint", default="outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt")
    p.add_argument("--output-dir", default="outputs/x2_3dgs_visibility_fusion")
    return p.parse_args()


def load_calibrations(data_root):
    calibs = {}
    for cam in CAMERA_NAMES:
        fs = cv2.FileStorage(os.path.join(data_root, "calibrations/extrinsic/extr_{}.xml".format(cam)),
                             cv2.FILE_STORAGE_READ)
        R = fs.getNode("R").mat()
        T = fs.getNode("T").mat().flatten()
        fs.release()
        W2C = np.eye(4, dtype=np.float64)
        W2C[:3, :3] = R
        W2C[:3, 3] = T
        C2W = np.linalg.inv(W2C)
        fs = cv2.FileStorage(os.path.join(data_root, "calibrations/intrinsic_original/intr_{}.xml".format(cam)),
                             cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        fs.release()
        calibs[cam] = {"K": K.astype(np.float64), "W2C": W2C.astype(np.float64), "C2W": C2W.astype(np.float64)}
    return calibs


def decode_position_id(position_id):
    x_grid = position_id % POM_NB_WIDTH
    y_grid = position_id // POM_NB_WIDTH
    world_x = (POM_ORIGINE_X + x_grid * POM_CELL_SIZE) * 100.0
    world_y = (POM_ORIGINE_Y + y_grid * POM_CELL_SIZE) * 100.0
    return world_x, world_y


def load_3dgs_model(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    positions = state["positions"].numpy().astype(np.float64)
    density_raw = state["density"].numpy().astype(np.float64)
    scale_raw = state["scale"].numpy().astype(np.float64)
    opacities = 1.0 / (1.0 + np.exp(-density_raw)).flatten()
    scales = np.exp(scale_raw)
    print(f"[LOAD] 3DGS model: {positions.shape[0]} Gaussians")
    print(f"  Opacities: mean={opacities.mean():.4f}, median={np.median(opacities):.4f}")
    print(f"  Scales: mean={scales.mean():.2f}, median={np.median(scales):.2f} cm")
    return positions, opacities, scales


def precompute_gaussian_projections(positions, calibs):
    """For each camera, project all Gaussian centers to pixel coordinates and compute depths."""
    proj_data = {}
    for cam in CAMERA_NAMES:
        K = calibs[cam]["K"]
        W2C = calibs[cam]["W2C"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p_cam = (W2C[:3, :3] @ positions.T + W2C[:3, 3:4]).T
        in_front = p_cam[:, 2] > 0
        u = np.where(in_front, fx * p_cam[:, 0] / p_cam[:, 2] + cx, -1)
        v = np.where(in_front, fy * p_cam[:, 1] / p_cam[:, 2] + cy, -1)
        proj_data[cam] = {"u": u, "v": v, "depth": p_cam[:, 2], "in_front": in_front}
    return proj_data


def compute_3dgs_visibility_at_pixel(anchor_u, anchor_v, anchor_depth, proj_cam, opacities):
    """Compute 3DGS-based visibility at a projected anchor pixel location.

    Returns:
        depth_ratio: min(nearest Gaussian depth / anchor depth, 1.0)
                     1.0 = no occluding Gaussian, <1.0 = occluded
        gaussian_count: number of Gaussians within PIXEL_RADIUS of the pixel
        weighted_opacity: sum of alpha contributions from nearby Gaussians
    """
    du = proj_cam["u"] - anchor_u
    dv = proj_cam["v"] - anchor_v
    pixel_dist_sq = du ** 2 + dv ** 2

    near_mask = proj_cam["in_front"] & (pixel_dist_sq < PIXEL_RADIUS ** 2)
    if not near_mask.any():
        return 1.0, 0, 0.0

    near_depths = proj_cam["depth"][near_mask]
    near_opac = opacities[near_mask]
    near_pixel_dist = np.sqrt(pixel_dist_sq[near_mask])

    closer_mask = near_depths < anchor_depth
    if closer_mask.any():
        closer_depths = near_depths[closer_mask]
        depth_ratio = float(closer_depths.min() / anchor_depth)
    else:
        depth_ratio = 1.0

    gaussian_count = int(near_mask.sum())
    alpha_contrib = near_opac * np.exp(-0.5 * (near_pixel_dist / PIXEL_RADIUS) ** 2)
    weighted_opacity = float(alpha_contrib.sum())

    return depth_ratio, gaussian_count, weighted_opacity


def compute_bbox_occlusion(frame_annotations, camera_idx, person_id, person_bbox):
    """Compute max IoU of this person's bbox with other persons in same camera."""
    max_iou = 0.0
    x1, y1, x2, y2 = person_bbox
    area1 = max((x2 - x1) * (y2 - y1), 1)
    for other in frame_annotations:
        if other["personID"] == person_id:
            continue
        for v in other["views"]:
            if v["viewNum"] == camera_idx and v["xmin"] != -1:
                ox1, oy1, ox2, oy2 = v["xmin"], v["ymin"], v["xmax"], v["ymax"]
                ix1 = max(x1, ox1)
                iy1 = max(y1, oy1)
                ix2 = min(x2, ox2)
                iy2 = min(y2, oy2)
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area2 = (ox2 - ox1) * (oy2 - oy1)
                    iou = inter / (area1 + area2 - inter)
                    max_iou = max(max_iou, iou)
    return max_iou


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    calibs = load_calibrations(args.data_root)
    positions, opacities, scales = load_3dgs_model(args.checkpoint)
    proj_data = precompute_gaussian_projections(positions, calibs)

    inst = []
    with open(args.instance_table) as f:
        for r in csv.DictReader(f):
            r["frame_id"] = int(r["frame_id"])
            r["person_id"] = int(r["person_id"])
            r["feature_index"] = int(r["feature_index"])
            r["bbox_valid"] = r["bbox_valid"] == "True"
            r["bbox_area"] = int(r["bbox_area"])
            r["bbox_height"] = int(r["bbox_height"])
            r["position_id"] = int(r["position_id"])
            r["xmin"] = int(r["xmin"])
            r["ymin"] = int(r["ymin"])
            r["xmax"] = int(r["xmax"])
            r["ymax"] = int(r["ymax"])
            inst.append(r)

    print(f"[LOAD] instance_table: {len(inst)} rows")

    unique_fp = set()
    for r in inst:
        if r["bbox_valid"] and r["position_id"] > 0:
            unique_fp.add((r["frame_id"], r["person_id"]))
    print(f"[INFO] Unique (frame, person) with valid position: {len(unique_fp)}")

    fp_anchors = {}
    for fid, pid in unique_fp:
        rows = [r for r in inst if r["frame_id"] == fid and r["person_id"] == pid]
        pos_id = rows[0]["position_id"]
        wx, wy = decode_position_id(pos_id)
        fp_anchors[(fid, pid)] = {
            "foot": np.array([wx, wy, 0.0]),
            "torso": np.array([wx, wy, 80.0]),
            "head": np.array([wx, wy, 170.0]),
        }

    ann_cache = {}
    ann_dir = os.path.join(args.data_root, "annotations_positions")

    results = []
    total = len(inst)

    for idx, row in enumerate(inst):
        if (idx + 1) % 500 == 0:
            print(f"  Progress: {idx + 1}/{total}")

        fid = row["frame_id"]
        pid = row["person_id"]
        cam = row["camera_id"]
        feat_idx = row["feature_index"]
        bbox_valid = row["bbox_valid"]
        bbox_area = row["bbox_area"]
        bbox_height = row["bbox_height"]
        pos_id = row["position_id"]

        result_row = {
            "frame_id": fid, "person_id": pid, "camera_id": cam,
            "feature_index": feat_idx, "bbox_valid": bbox_valid,
            "bbox_area": bbox_area, "bbox_height": bbox_height,
            "position_id": pos_id,
        }

        if not bbox_valid or pos_id <= 0 or (fid, pid) not in fp_anchors:
            result_row.update({
                "world_x": -1, "world_y": -1, "world_z": -1,
                "proj_x_foot": -1, "proj_y_foot": -1,
                "proj_x_torso": -1, "proj_y_torso": -1,
                "proj_x_head": -1, "proj_y_head": -1,
                "v_projection": 0, "v_bbox_valid": 0,
                "v_depth_ratio_foot": -1, "v_depth_ratio_torso": -1, "v_depth_ratio_head": -1,
                "v_gaussian_count_torso": -1, "v_weighted_opacity_torso": -1,
                "v_bbox_occlusion": -1, "v_weighted_3dgs": -1,
            })
            results.append(result_row)
            continue

        anchors = fp_anchors[(fid, pid)]
        wx, wy = anchors["foot"][0], anchors["foot"][1]
        cal = calibs[cam]
        K = cal["K"]
        W2C = cal["W2C"]

        proj_coords = {}
        v_proj = 1
        for aname in ["foot", "torso", "head"]:
            p_cam = W2C[:3, :3] @ anchors[aname] + W2C[:3, 3]
            if p_cam[2] <= 0:
                proj_coords[aname] = (-1, -1, -1)
                v_proj = 0
            else:
                u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
                v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
                proj_coords[aname] = (u, v, p_cam[2])
                if not (0 <= u <= 1920 and 0 <= v <= 1080):
                    v_proj = 0

        depth_ratios = {}
        gauss_count_torso = -1
        weighted_opac_torso = -1
        for aname in ["foot", "torso", "head"]:
            u, v, depth = proj_coords[aname]
            if depth > 0:
                dr, gc, wo = compute_3dgs_visibility_at_pixel(
                    u, v, depth, proj_data[cam], opacities
                )
                depth_ratios[aname] = dr
                if aname == "torso":
                    gauss_count_torso = gc
                    weighted_opac_torso = wo
            else:
                depth_ratios[aname] = -1

        frame_key = f"{fid:08d}"
        if frame_key not in ann_cache:
            ann_path = os.path.join(ann_dir, f"{frame_key}.json")
            if os.path.exists(ann_path):
                with open(ann_path) as f:
                    ann_cache[frame_key] = json.load(f)
            else:
                ann_cache[frame_key] = []

        cam_idx = CAM_TO_VIEWNUM.get(cam, -1)
        if cam_idx >= 0 and ann_cache[frame_key]:
            bbox = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
            v_bbox_occ = compute_bbox_occlusion(ann_cache[frame_key], cam_idx, pid, bbox)
        else:
            v_bbox_occ = 0.0

        dr_foot = depth_ratios.get("foot", -1)
        dr_torso = depth_ratios.get("torso", -1)
        dr_head = depth_ratios.get("head", -1)

        valid_drs = [d for d in [dr_foot, dr_torso, dr_head] if d >= 0]
        if valid_drs:
            v_weighted = 0.2 * max(dr_foot, 0) + 0.5 * max(dr_torso, 0) + 0.3 * max(dr_head, 0)
        else:
            v_weighted = -1

        result_row.update({
            "world_x": wx, "world_y": wy, "world_z": 0.0,
            "proj_x_foot": proj_coords["foot"][0], "proj_y_foot": proj_coords["foot"][1],
            "proj_x_torso": proj_coords["torso"][0], "proj_y_torso": proj_coords["torso"][1],
            "proj_x_head": proj_coords["head"][0], "proj_y_head": proj_coords["head"][1],
            "v_projection": v_proj, "v_bbox_valid": 1,
            "v_depth_ratio_foot": dr_foot, "v_depth_ratio_torso": dr_torso, "v_depth_ratio_head": dr_head,
            "v_gaussian_count_torso": gauss_count_torso, "v_weighted_opacity_torso": weighted_opac_torso,
            "v_bbox_occlusion": v_bbox_occ, "v_weighted_3dgs": v_weighted,
        })
        results.append(result_row)

    csv_path = os.path.join(args.output_dir, "visibility_scores.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[SAVE] {csv_path} ({len(results)} rows)")

    valid_vis = [r for r in results if r["v_depth_ratio_torso"] >= 0]
    if valid_vis:
        diag = {
            "total_rows": len(results),
            "valid_rows": len(valid_vis),
            "note": "3DGS transmittance not used (model trained without person masking, T≈0 everywhere). "
                    "Using depth_ratio and gaussian_count instead.",
            "v_depth_ratio_torso_mean": float(np.mean([r["v_depth_ratio_torso"] for r in valid_vis])),
            "v_depth_ratio_torso_std": float(np.std([r["v_depth_ratio_torso"] for r in valid_vis])),
            "v_gaussian_count_torso_mean": float(np.mean([r["v_gaussian_count_torso"] for r in valid_vis])),
            "v_weighted_opacity_torso_mean": float(np.mean([r["v_weighted_opacity_torso"] for r in valid_vis])),
            "v_bbox_occlusion_mean": float(np.mean([r["v_bbox_occlusion"] for r in valid_vis])),
            "v_weighted_3dgs_mean": float(np.mean([r["v_weighted_3dgs"] for r in valid_vis])),
            "v_weighted_3dgs_std": float(np.std([r["v_weighted_3dgs"] for r in valid_vis])),
            "per_camera": {},
        }
        for cam in CAMERA_NAMES:
            cam_rows = [r for r in valid_vis if r["camera_id"] == cam]
            if cam_rows:
                diag["per_camera"][cam] = {
                    "v_depth_ratio_torso_mean": float(np.mean([r["v_depth_ratio_torso"] for r in cam_rows])),
                    "v_weighted_3dgs_mean": float(np.mean([r["v_weighted_3dgs"] for r in cam_rows])),
                    "v_bbox_occlusion_mean": float(np.mean([r["v_bbox_occlusion"] for r in cam_rows])),
                    "count": len(cam_rows),
                }

        diag_path = os.path.join(args.output_dir, "visibility_diagnostics.json")
        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2)
        print(f"[SAVE] {diag_path}")

        print(f"\n=== VISIBILITY SUMMARY ===")
        print(f"Valid rows: {len(valid_vis)}")
        print(f"v_depth_ratio_torso: mean={diag['v_depth_ratio_torso_mean']:.4f} std={diag['v_depth_ratio_torso_std']:.4f}")
        print(f"v_gaussian_count_torso: mean={diag['v_gaussian_count_torso_mean']:.1f}")
        print(f"v_weighted_opacity_torso: mean={diag['v_weighted_opacity_torso_mean']:.4f}")
        print(f"v_bbox_occlusion: mean={diag['v_bbox_occlusion_mean']:.4f}")
        print(f"v_weighted_3dgs: mean={diag['v_weighted_3dgs_mean']:.4f} std={diag['v_weighted_3dgs_std']:.4f}")

        occ_vals = [r["v_bbox_occlusion"] for r in valid_vis if r["v_bbox_occlusion"] >= 0]
        if occ_vals:
            occ_arr = np.array(occ_vals)
            n_zero = int((occ_arr == 0).sum())
            n_pos = int((occ_arr > 0).sum())
            n_near_one = int((occ_arr >= 0.99).sum())
            occ_diag = {
                "v_bbox_occlusion_mean": float(occ_arr.mean()),
                "v_bbox_occlusion_std": float(occ_arr.std()),
                "v_bbox_occlusion_min": float(occ_arr.min()),
                "v_bbox_occlusion_max": float(occ_arr.max()),
                "n_zero": n_zero,
                "pct_zero": float(n_zero / len(occ_arr) * 100),
                "n_positive": n_pos,
                "pct_positive": float(n_pos / len(occ_arr) * 100),
                "n_near_one": n_near_one,
                "pct_near_one": float(n_near_one / len(occ_arr) * 100),
                "per_camera_v_bbox_occlusion_mean": {},
            }
            for cam in CAMERA_NAMES:
                cam_occ = [r["v_bbox_occlusion"] for r in valid_vis
                           if r["camera_id"] == cam and r["v_bbox_occlusion"] >= 0]
                if cam_occ:
                    occ_diag["per_camera_v_bbox_occlusion_mean"][cam] = float(np.mean(cam_occ))

            positive_examples = [r for r in valid_vis if r["v_bbox_occlusion"] > 0]
            if positive_examples:
                np.random.seed(42)
                sample_indices = np.random.choice(len(positive_examples),
                                                  min(10, len(positive_examples)), replace=False)
                occ_diag["sample_positive_cases"] = []
                for si in sample_indices:
                    r = positive_examples[si]
                    occ_diag["sample_positive_cases"].append({
                        "frame_id": r["frame_id"],
                        "person_id": r["person_id"],
                        "camera_id": r["camera_id"],
                        "bbox_area": r["bbox_area"],
                        "v_bbox_occlusion": r["v_bbox_occlusion"],
                    })

            occ_diag_path = os.path.join(args.output_dir, "occlusion_bugfix_diagnostics.json")
            with open(occ_diag_path, "w") as f:
                json.dump(occ_diag, f, indent=2)
            print(f"[SAVE] {occ_diag_path}")

            print(f"\n=== OCCLUSION BUGFIX DIAGNOSTICS ===")
            print(f"v_bbox_occlusion: mean={occ_diag['v_bbox_occlusion_mean']:.4f} "
                  f"std={occ_diag['v_bbox_occlusion_std']:.4f} "
                  f"min={occ_diag['v_bbox_occlusion_min']:.4f} max={occ_diag['v_bbox_occlusion_max']:.4f}")
            print(f"  ==0: {n_zero} ({occ_diag['pct_zero']:.1f}%)  "
                  f">0: {n_pos} ({occ_diag['pct_positive']:.1f}%)  "
                  f">=0.99: {n_near_one} ({occ_diag['pct_near_one']:.1f}%)")

            if n_pos == 0:
                print("[ERROR] v_bbox_occlusion is ALL ZERO — cam mapping bug NOT fixed!")
                raise RuntimeError("v_bbox_occlusion all zero after bugfix — cam_idx mapping still broken")
            if n_near_one / len(occ_arr) > 0.9:
                print("[ERROR] v_bbox_occlusion almost all >=0.99 — self-exclusion bug NOT fixed!")
                raise RuntimeError("v_bbox_occlusion almost all >=0.99 — self not excluded in IoU computation")


if __name__ == "__main__":
    main()
