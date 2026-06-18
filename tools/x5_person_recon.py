#!/usr/bin/env python3
"""X5: Person-level 3DGS Reconstruction Feasibility Test.

Go/kill upstream validation: can 3DGS reconstruct a single person in WildTrack?

Task A: Candidate selection - find person-frames with most valid views
Task B: Per-person 3DGS reconstruction using simple differentiable renderer
Task C: Leave-one-view holdout evaluation

If even the most favorable person-frame cannot be reconstructed,
person-Gaussian approaches (IDR-GS etc.) have no foundation on WildTrack.
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

CAMERA_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
VIEWNUM_TO_CAM = {i: c for i, c in enumerate(CAMERA_NAMES)}
CAM_TO_VIEWNUM = {c: i for i, c in enumerate(CAMERA_NAMES)}
ORIGINE_X = -3.0
ORIGINE_Y = -9.0
NB_WIDTH = 480
CELL_SIZE = 0.025


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data02/zhangrunxiang/data/Wildtrack")
    p.add_argument("--output-dir", default="outputs/x5_person_recon")
    p.add_argument("--n-candidates", type=int, default=5)
    p.add_argument("--n-gaussians", type=int, default=500)
    p.add_argument("--n-iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--crop-pad", type=int, default=15)
    return p.parse_args()


def decode_position_id(position_id):
    x_grid = position_id % NB_WIDTH
    y_grid = position_id // NB_WIDTH
    world_x = (ORIGINE_X + x_grid * CELL_SIZE) * 100
    world_y = (ORIGINE_Y + y_grid * CELL_SIZE) * 100
    return world_x, world_y


def load_extrinsics(data_root):
    extrinsics = {}
    for cam in CAMERA_NAMES:
        path = os.path.join(data_root, "calibrations", "extrinsic", f"extr_{cam}.xml")
        tree = ET.parse(path)
        root = tree.getroot()
        R_data = [float(x) for x in root.find("R").find("data").text.split()]
        T_data = [float(x) for x in root.find("T").find("data").text.split()]
        R = np.array(R_data).reshape(3, 3)
        T = np.array(T_data).reshape(3, 1)
        C2W = np.eye(4)
        C2W[:3, :3] = R
        C2W[:3, 3] = T.flatten()
        extrinsics[cam] = C2W
    return extrinsics


def load_intrinsics(data_root):
    intrinsics = {}
    for cam in CAMERA_NAMES:
        path = os.path.join(data_root, "calibrations", "intrinsic_original", f"intr_{cam}.xml")
        tree = ET.parse(path)
        root = tree.getroot()
        K_data = [float(x) for x in root.find("camera_matrix").find("data").text.split()]
        dist_data = [float(x) for x in root.find("distortion_coefficients").find("data").text.split()]
        K = np.array(K_data).reshape(3, 3)
        intrinsics[cam] = {"K": K, "dist": np.array(dist_data)}
    return intrinsics


def load_annotations(data_root):
    ann_dir = os.path.join(data_root, "annotations_positions")
    all_anns = {}
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".json"):
            continue
        frame_id = int(fname.replace(".json", ""))
        with open(os.path.join(ann_dir, fname)) as f:
            persons = json.load(f)
        all_anns[frame_id] = persons
    return all_anns


def project_point(P_world, C2W, K):
    W2C = np.linalg.inv(C2W)
    P_cam = W2C[:3, :3] @ P_world + W2C[:3, 3:4]
    if P_cam[2] <= 0:
        return None
    p_pix = K @ P_cam
    return p_pix[:2] / p_cam[2]


def compute_angular_coverage(person_pos, C2W_dict, valid_cams):
    cam_positions = []
    for cam in valid_cams:
        cam_pos = C2W_dict[cam][:3, 3]
        cam_positions.append(cam_pos)
    if len(cam_positions) < 2:
        return 0.0
    cam_positions = np.array(cam_positions)
    person_pos_3d = np.array([person_pos[0], person_pos[1], 0.0])
    directions = cam_positions - person_pos_3d[None, :]
    directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    angles = []
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            cos_a = np.clip(np.dot(directions[i], directions[j]), -1, 1)
            angles.append(np.degrees(np.arccos(cos_a)))
    return max(angles) if angles else 0.0


# ============================================================
# Task A: Candidate Selection
# ============================================================

def task_a_select_candidates(args):
    print("=" * 60)
    print("TASK A: Candidate Selection")
    print("=" * 60)

    all_anns = load_annotations(args.data_root)
    extrinsics = load_extrinsics(args.data_root)
    intrinsics = load_intrinsics(args.data_root)

    K_dist = defaultdict(int)
    person_frame_info = []

    for frame_id in sorted(all_anns.keys()):
        persons = all_anns[frame_id]
        for person in persons:
            pid = person["personID"]
            pos_id = person["positionID"]
            wx, wy = decode_position_id(pos_id)

            valid_cams = []
            bbox_info = {}
            for v in person["views"]:
                cam = VIEWNUM_TO_CAM[v["viewNum"]]
                if v["xmin"] >= 0 and v["ymin"] >= 0:
                    valid_cams.append(cam)
                    w = v["xmax"] - v["xmin"]
                    h = v["ymax"] - v["ymin"]
                    bbox_info[cam] = {
                        "xmin": v["xmin"], "ymin": v["ymin"],
                        "xmax": v["xmax"], "ymax": v["ymax"],
                        "width": w, "height": h, "area": w * h,
                    }

            K = len(valid_cams)
            K_dist[K] += 1

            if K < 3:
                continue

            total_area = sum(bbox_info[c]["area"] for c in valid_cams)
            avg_area = total_area / K
            avg_height = sum(bbox_info[c]["height"] for c in valid_cams) / K
            ang_cov = compute_angular_coverage((wx, wy), extrinsics, valid_cams)

            person_frame_info.append({
                "frame_id": frame_id,
                "person_id": pid,
                "position_id": pos_id,
                "world_x": wx,
                "world_y": wy,
                "K": K,
                "valid_cams": valid_cams,
                "bbox_info": bbox_info,
                "avg_area": avg_area,
                "avg_height": avg_height,
                "angular_coverage": ang_cov,
            })

    total = sum(K_dist.values())
    print(f"\nValid view count distribution (total person-frames: {total}):")
    for k in sorted(K_dist.keys()):
        pct = K_dist[k] / total * 100
        print(f"  K={k}: {K_dist[k]} ({pct:.1f}%)")

    person_frame_info.sort(key=lambda x: (-x["K"], -x["avg_area"], -x["angular_coverage"]))
    candidates = person_frame_info[:args.n_candidates]

    print(f"\nTop {len(candidates)} candidates:")
    for i, c in enumerate(candidates):
        print(f"\n  Candidate {i+1}: frame={c['frame_id']}, person={c['person_id']}, "
              f"K={c['K']}, avg_area={c['avg_area']:.0f}, avg_height={c['avg_height']:.0f}, "
              f"angular_coverage={c['angular_coverage']:.1f} deg")
        for cam in c["valid_cams"]:
            bi = c["bbox_info"][cam]
            print(f"    {cam}: bbox=[{bi['xmin']},{bi['ymin']},{bi['xmax']},{bi['ymax']}] "
                  f"size={bi['width']}x{bi['height']} area={bi['area']}")

    stats = {
        "K_distribution": {str(k): v for k, v in sorted(K_dist.items())},
        "total_person_frames": total,
        "K_ge3_count": sum(1 for x in person_frame_info if x["K"] >= 3),
        "candidates": [],
    }
    for c in candidates:
        cd = {k: v for k, v in c.items() if k != "bbox_info"}
        cd["per_view_bbox"] = {cam: c["bbox_info"][cam] for cam in c["valid_cams"]}
        stats["candidates"].append(cd)

    stats_path = os.path.join(args.output_dir, "candidate_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[SAVE] {stats_path}")

    return candidates, extrinsics, intrinsics


# ============================================================
# Simple Differentiable Gaussian Splatting Renderer
# ============================================================

def quaternion_to_matrix(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def render_gaussians(means3D, scales, rotations, opacities, colors, W2C, K, img_size, device):
    """Render Gaussians via EWA splatting with alpha blending.

    means3D: [N, 3] world positions
    scales: [N, 3] scale (after exp, positive)
    rotations: [N, 4] quaternions (normalized)
    opacities: [N, 1] opacity (after sigmoid, [0,1])
    colors: [N, 3] RGB [0,1]
    W2C: [4, 4] world-to-camera
    K: [3, 3] intrinsic matrix
    img_size: (H, W)
    """
    N = means3D.shape[0]
    H, W = img_size

    R_wc = W2C[:3, :3].to(device)
    t_wc = W2C[:3, 3].to(device)
    K_dev = K.to(device)

    means_cam = (R_wc[None] @ means3D[:, :, None]).squeeze(-1) + t_wc[None]
    depths = means_cam[:, 2].clamp(min=0.01)

    proj = K_dev[None] @ means_cam[:, :, None]
    proj_xy = proj[:, :2, 0] / proj[:, 2:3, 0].clamp(min=0.01)
    valid_depth = means_cam[:, 2] > 0.1

    R_mat = quaternion_to_matrix(rotations)
    S_mat = torch.diag_embed(scales)
    cov3D = R_mat @ S_mat @ S_mat.transpose(1, 2) @ R_mat.transpose(1, 2)

    J = torch.zeros(N, 2, 3, device=device)
    fx, fy = K_dev[0, 0], K_dev[1, 1]
    z = depths
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * means_cam[:, 0] / (z * z)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * means_cam[:, 1] / (z * z)

    cov2D_full = J @ R_wc[None] @ cov3D @ R_wc[None].transpose(1, 2) @ J.transpose(1, 2)
    cov2D = cov2D_full[:, :2, :2]
    cov2D += 0.3 * torch.eye(2, device=device)[None]

    det = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] ** 2
    det = det.clamp(min=1e-6)
    inv_cov = torch.zeros_like(cov2D)
    inv_cov[:, 0, 0] = cov2D[:, 1, 1] / det
    inv_cov[:, 1, 1] = cov2D[:, 0, 0] / det
    inv_cov[:, 0, 1] = -cov2D[:, 0, 1] / det
    inv_cov[:, 1, 0] = -cov2D[:, 0, 1] / det

    power = 3.0
    sigma = torch.sqrt(det)
    radius = torch.ceil(torch.sqrt(det) * power).long().clamp(max=max(H, W))

    sort_idx = torch.argsort(depths)
    sorted_means2D = proj_xy[sort_idx]
    sorted_inv_cov = inv_cov[sort_idx]
    sorted_opacities = opacities[sort_idx]
    sorted_colors = colors[sort_idx]
    sorted_depths = depths[sort_idx]
    sorted_valid = valid_depth[sort_idx]
    sorted_radius = radius[sort_idx]
    sorted_det = det[sort_idx]

    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                             torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
    pixels = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    rendered = torch.zeros(H * W, 3, device=device)
    T_acc = torch.ones(H * W, device=device)

    for gi in range(N):
        if not sorted_valid[gi]:
            continue
        r = sorted_radius[gi].item()
        if r <= 0:
            continue

        cx, cy = sorted_means2D[gi, 0].item(), sorted_means2D[gi, 1].item()
        x_lo = max(0, int(cx - r))
        x_hi = min(W, int(cx + r + 1))
        y_lo = max(0, int(cy - r))
        y_hi = min(H, int(cy + r + 1))

        if x_lo >= x_hi or y_lo >= y_hi:
            continue

        local_pixels = pixels[y_lo * W + x_lo: y_lo * W + x_hi]
        for row in range(y_lo + 1, y_hi):
            local_pixels = torch.cat([local_pixels, pixels[row * W + x_lo: row * W + x_hi]])

        diff = local_pixels.float() - sorted_means2D[gi:gi+1]
        mahal = (diff[:, None, :] @ sorted_inv_cov[gi:gi+1] @ diff[:, :, None]).squeeze(-1).squeeze(-1)
        alpha = sorted_opacities[gi, 0] * torch.exp(-0.5 * mahal)
        alpha = alpha.clamp(max=0.99)

        mask = T_acc[y_lo * W + x_lo: y_lo * W + x_hi] if y_lo + 1 == y_hi else \
               T_acc[local_pixels[:, 1] * W + local_pixels[:, 0]]

        contrib = T_acc[local_pixels[:, 1] * W + local_pixels[:, 0]] * alpha
        rendered[local_pixels[:, 1] * W + local_pixels[:, 0]] += contrib[:, None] * sorted_colors[gi:gi+1]
        T_acc[local_pixels[:, 1] * W + local_pixels[:, 0]] *= (1 - alpha)

    return rendered.reshape(H, W, 3)


# ============================================================
# Task B+C: Per-person Reconstruction + Holdout Evaluation
# ============================================================

def crop_person_patch(img, bbox, pad):
    xmin, ymin, xmax, ymax = bbox
    H, W = img.shape[:2]
    xmin = max(0, xmin - pad)
    ymin = max(0, ymin - pad)
    xmax = min(W, xmax + pad)
    ymax = min(H, ymax + pad)
    return img[ymin:ymax, xmin:xmax].copy(), (xmin, ymin, xmax, ymax)


def adjust_intrinsics_for_crop(K, crop_origin):
    K_new = K.copy()
    K_new[0, 2] -= crop_origin[0]
    K_new[1, 2] -= crop_origin[1]
    return K_new


def undistort_image(img, K, dist):
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    undist = cv2.undistort(img, K, dist, None, new_K)
    return undist, new_K


def train_person_3dgs(candidate, extrinsics, intrinsics, args, device):
    """Train per-person 3DGS with leave-one-view holdout."""
    frame_id = candidate["frame_id"]
    pid = candidate["person_id"]
    valid_cams = candidate["valid_cams"]
    bbox_info = candidate["bbox_info"]
    wx, wy = candidate["world_x"], candidate["world_y"]

    img_dir = os.path.join(args.data_root, "Image_subsets")
    frame_str = f"{frame_id:08d}"

    view_data = {}
    for cam in valid_cams:
        img_path = os.path.join(img_dir, cam, f"{frame_str}.png")
        img = cv2.imread(img_path)
        if img is None:
            print(f"    WARNING: cannot read {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bi = bbox_info[cam]
        K_orig = intrinsics[cam]["K"]
        dist = intrinsics[cam]["dist"]

        img_undist, K_undist = undistort_image(img, K_orig, dist)

        crop, crop_box = crop_person_patch(img_undist, [bi["xmin"], bi["ymin"], bi["xmax"], bi["ymax"]], args.crop_pad)
        K_crop = adjust_intrinsics_for_crop(K_undist, (crop_box[0], crop_box[1]))

        crop_h, crop_w = crop.shape[:2]
        crop_float = (crop.astype(np.float32) / 255.0)

        C2W = extrinsics[cam]
        W2C = np.linalg.inv(C2W)

        view_data[cam] = {
            "image": crop_float,
            "K": K_crop,
            "W2C": W2C,
            "C2W": C2W,
            "crop_box": crop_box,
            "crop_h": crop_h,
            "crop_w": crop_w,
        }

    if len(view_data) < 3:
        print(f"    SKIP: only {len(view_data)} valid views after loading")
        return None

    results = {}

    for holdout_cam in valid_cams:
        if holdout_cam not in view_data:
            continue

        train_cams = [c for c in view_data if c != holdout_cam]
        if len(train_cams) < 2:
            continue

        print(f"\n  Holdout={holdout_cam}, train views={train_cams}")

        ref_view = view_data[train_cams[0]]
        ref_H, ref_W = ref_view["crop_h"], ref_view["crop_w"]
        render_size = (ref_H, ref_W)

        N = args.n_gaussians
        foot_z = 0.0
        torso_z = 80.0
        head_z = 170.0

        torch.manual_seed(42)
        np.random.seed(42)

        init_positions = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            t = i / N
            if t < 0.4:
                z = foot_z + (torso_z - foot_z) * (t / 0.4)
            elif t < 0.7:
                z = torso_z + (head_z - torso_z) * ((t - 0.4) / 0.3)
            else:
                z = head_z + np.random.randn() * 10
            dx = np.random.randn() * 25
            dy = np.random.randn() * 25
            dz = np.random.randn() * 10
            init_positions[i] = [wx + dx, wy + dy, z + dz]

        means3D = torch.nn.Parameter(torch.tensor(init_positions, dtype=torch.float32, device=device))
        log_scales = torch.nn.Parameter(torch.full((N, 3), -2.0, dtype=torch.float32, device=device))
        raw_rotations = torch.nn.Parameter(torch.tensor(
            np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32), device=device))
        raw_opacities = torch.nn.Parameter(torch.full((N, 1), 0.0, dtype=torch.float32, device=device))
        colors = torch.nn.Parameter(torch.tensor(
            np.random.rand(N, 3).astype(np.float32) * 0.5 + 0.3, device=device))

        optimizer = torch.optim.Adam([
            {"params": [means3D], "lr": args.lr},
            {"params": [log_scales], "lr": args.lr * 0.5},
            {"params": [raw_rotations], "lr": args.lr * 0.1},
            {"params": [raw_opacities], "lr": args.lr * 0.5},
            {"params": [colors], "lr": args.lr},
        ])

        train_gts = []
        train_W2Cs = []
        train_Ks = []
        for cam in train_cams:
            vd = view_data[cam]
            gt = torch.tensor(vd["image"], dtype=torch.float32, device=device)
            if gt.shape[0] != render_size[0] or gt.shape[1] != render_size[1]:
                gt = torch.nn.functional.interpolate(
                    gt.permute(2, 0, 1)[None], size=render_size, mode='bilinear', align_corners=False
                )[0].permute(1, 2, 0)
            train_gts.append(gt)
            train_W2Cs.append(torch.tensor(vd["W2C"], dtype=torch.float32, device=device))
            train_Ks.append(torch.tensor(vd["K"], dtype=torch.float32, device=device))

        loss_history = []
        psnr_history = []

        for iteration in range(args.n_iters):
            scales = log_scales.exp().clamp(max=100.0)
            rotations = F.normalize(raw_rotations, dim=1)
            opacities = torch.sigmoid(raw_opacities)

            total_loss = torch.tensor(0.0, device=device)

            for vi in range(len(train_cams)):
                rendered = render_gaussians(
                    means3D, scales, rotations, opacities, colors,
                    train_W2Cs[vi], train_Ks[vi], render_size, device
                )
                l1_loss = F.l1_loss(rendered, train_gts[vi])
                total_loss = total_loss + l1_loss

            total_loss = total_loss / len(train_cams)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_val = total_loss.item()
            loss_history.append(loss_val)

            if iteration % 500 == 0 or iteration == args.n_iters - 1:
                with torch.no_grad():
                    scales_e = log_scales.exp().clamp(max=100.0)
                    rotations_e = F.normalize(raw_rotations, dim=1)
                    opacities_e = torch.sigmoid(raw_opacities)
                    train_psnrs = []
                    for vi in range(len(train_cams)):
                        r = render_gaussians(
                            means3D, scales_e, rotations_e, opacities_e, colors,
                            train_W2Cs[vi], train_Ks[vi], render_size, device
                        )
                        mse = F.mse_loss(r, train_gts[vi])
                        psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
                        train_psnrs.append(psnr.item())
                    avg_train_psnr = np.mean(train_psnrs)
                    psnr_history.append(avg_train_psnr)
                    print(f"    iter {iteration}: loss={loss_val:.6f}, train_PSNR={avg_train_psnr:.2f}")

        with torch.no_grad():
            scales_final = log_scales.exp().clamp(max=100.0)
            rotations_final = F.normalize(raw_rotations, dim=1)
            opacities_final = torch.sigmoid(raw_opacities)

            holdout_W2C = torch.tensor(view_data[holdout_cam]["W2C"], dtype=torch.float32, device=device)
            holdout_K = torch.tensor(view_data[holdout_cam]["K"], dtype=torch.float32, device=device)
            holdout_gt = torch.tensor(view_data[holdout_cam]["image"], dtype=torch.float32, device=device)
            if holdout_gt.shape[0] != render_size[0] or holdout_gt.shape[1] != render_size[1]:
                holdout_gt = torch.nn.functional.interpolate(
                    holdout_gt.permute(2, 0, 1)[None], size=render_size, mode='bilinear', align_corners=False
                )[0].permute(1, 2, 0)

            holdout_rendered = render_gaussians(
                means3D, scales_final, rotations_final, opacities_final, colors,
                holdout_W2C, holdout_K, render_size, device
            )

            holdout_mse = F.mse_loss(holdout_rendered, holdout_gt)
            holdout_psnr = 10 * torch.log10(1.0 / holdout_mse.clamp(min=1e-10)).item()

            train_psnrs_final = []
            for vi in range(len(train_cams)):
                r = render_gaussians(
                    means3D, scales_final, rotations_final, opacities_final, colors,
                    train_W2Cs[vi], train_Ks[vi], render_size, device
                )
                mse = F.mse_loss(r, train_gts[vi])
                train_psnrs_final.append(10 * torch.log10(1.0 / mse.clamp(min=1e-10)).item())

            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            holdout_ssim = ssim_metric(holdout_rendered.permute(2, 0, 1)[None],
                                       holdout_gt.permute(2, 0, 1)[None]).item()

            pos_np = means3D.detach().cpu().numpy()
            pos_spread = {
                "x_range": [float(pos_np[:, 0].min()), float(pos_np[:, 0].max())],
                "y_range": [float(pos_np[:, 1].min()), float(pos_np[:, 1].max())],
                "z_range": [float(pos_np[:, 2].min()), float(pos_np[:, 2].max())],
                "x_std": float(pos_np[:, 0].std()),
                "y_std": float(pos_np[:, 1].std()),
                "z_std": float(pos_np[:, 2].std()),
            }

            result = {
                "holdout_cam": holdout_cam,
                "train_cams": train_cams,
                "n_train_views": len(train_cams),
                "holdout_psnr": holdout_psnr,
                "holdout_ssim": holdout_ssim,
                "train_psnr_mean": float(np.mean(train_psnrs_final)),
                "train_psnr_std": float(np.std(train_psnrs_final)),
                "loss_final": loss_history[-1],
                "gaussian_spread": pos_spread,
                "degradation": "billboard" if holdout_psnr < 15 else
                              ("partial" if holdout_psnr < 20 else "reasonable"),
            }
            results[holdout_cam] = result

            print(f"    -> holdout PSNR={holdout_psnr:.2f}, SSIM={holdout_ssim:.4f}, "
                  f"train PSNR={result['train_psnr_mean']:.2f}, "
                  f"degradation={result['degradation']}")

            save_dir = os.path.join(args.output_dir, f"person{pid}_frame{frame_id}")
            os.makedirs(save_dir, exist_ok=True)

            render_np = (holdout_rendered.detach().cpu().numpy() * 255).astype(np.uint8)
            gt_np = (holdout_gt.detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f"holdout_{holdout_cam}_render.png"),
                       cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, f"holdout_{holdout_cam}_gt.png"),
                       cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))

            for vi, cam in enumerate(train_cams):
                r = render_gaussians(
                    means3D, scales_final, rotations_final, opacities_final, colors,
                    train_W2Cs[vi], train_Ks[vi], render_size, device
                )
                r_np = (r.detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, f"train_{cam}_render.png"),
                           cv2.cvtColor(r_np, cv2.COLOR_RGB2BGR))

            loss_arr = np.array(loss_history)
            np.save(os.path.join(save_dir, f"loss_holdout_{holdout_cam}.npy"), loss_arr)

    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] device={device}")

    candidates, extrinsics, intrinsics = task_a_select_candidates(args)

    print(f"\n{'='*60}")
    print("TASK B+C: Per-person 3DGS Reconstruction + Holdout")
    print("=" * 60)

    all_results = {}

    for ci, candidate in enumerate(candidates):
        frame_id = candidate["frame_id"]
        pid = candidate["person_id"]
        print(f"\n{'='*40}")
        print(f"Candidate {ci+1}/{len(candidates)}: person={pid}, frame={frame_id}, K={candidate['K']}")
        print(f"{'='*40}")

        result = train_person_3dgs(candidate, extrinsics, intrinsics, args, device)
        if result is not None:
            all_results[f"p{pid}_f{frame_id}"] = result

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)

    summary = {"per_candidate": {}, "verdict": {}}

    for key, res in all_results.items():
        holdout_psnrs = [v["holdout_psnr"] for v in res.values()]
        train_psnrs = [v["train_psnr_mean"] for v in res.values()]
        degradations = [v["degradation"] for v in res.values()]

        avg_holdout = np.mean(holdout_psnrs)
        avg_train = np.mean(train_psnrs)
        gap = avg_train - avg_holdout

        print(f"\n  {key}:")
        print(f"    avg holdout PSNR: {avg_holdout:.2f}")
        print(f"    avg train PSNR:   {avg_train:.2f}")
        print(f"    gap (train-holdout): {gap:.2f}")
        print(f"    degradations: {degradations}")

        summary["per_candidate"][key] = {
            "avg_holdout_psnr": float(avg_holdout),
            "avg_train_psnr": float(avg_train),
            "train_holdout_gap": float(gap),
            "degradations": degradations,
            "per_holdout": res,
        }

    best_holdout = max(
        (np.mean([v["holdout_psnr"] for v in res.values()])
         for res in all_results.values()),
        default=0
    )

    if best_holdout < 15:
        verdict = "RECONSTRUCTION_FAILED"
        verdict_msg = ("WildTrack person-3DGS reconstruction NOT feasible. "
                       "Holdout PSNR < 15 even for best candidates. "
                       "View count insufficient + low resolution. "
                       "Person-Gaussian identity approaches have no foundation. "
                       "Next: (a) switch to reconstructable settings (camera motion/high-res/MultiviewX), "
                       "or (b) downgrade 3DGS to training-side data generator, not in WildTrack inference loop.")
    elif best_holdout < 20:
        verdict = "RECONSTRUCTION_DEGRADED"
        verdict_msg = ("Partial reconstruction with significant degradation. "
                       "Holdout PSNR 15-20: geometry partially constrained but unreliable. "
                       "Billboard-like degeneration likely. "
                       "Person-Gaussian approaches risky on WildTrack.")
    else:
        verdict = "RECONSTRUCTION_FEASIBLE"
        verdict_msg = ("Holdout PSNR >= 20: reconstruction shows promise. "
                       "Worth exploring IDR-GS style differentiable identity features.")

    summary["verdict"] = {
        "best_holdout_psnr": float(best_holdout),
        "verdict": verdict,
        "message": verdict_msg,
    }

    print(f"\n  BEST holdout PSNR: {best_holdout:.2f}")
    print(f"  VERDICT: {verdict}")
    print(f"  {verdict_msg}")

    summary_path = os.path.join(args.output_dir, "recon_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
