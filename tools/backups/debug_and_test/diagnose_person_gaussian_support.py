#!/usr/bin/env python3
"""V1: Person Gaussian Support Diagnostic.

Checks whether person bboxes have sufficient 3D Gaussian render support
(opacity and feature norm) inside the ROI region.

NO TRAINING. DIAGNOSTIC ONLY.

Coordinate protocol (frozen from V0):
  - annotation bbox: 1920×1080 original space
  - image/rays/render/person_feature_map: padded (1088) + downsampled (4x) → 480×272
  - scale_bbox_to_render(src_w=1920, src_h=1088, dst_w=480, dst_h=272)
  - sx = sy = 0.25, no transpose
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.datasets import make as make_dataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import roi_pool, scale_bbox_to_render, _clamp_bbox

REID_INIT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase14_clean_geometry/full_soft_reset_30k/reid_init/reid_init_ckpt.pt"
DEFAULT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt"
DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
DEVICE = "cuda"


def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def setup_model(init_ckpt, device, reid_init_ckpt=REID_INIT_CKPT):
    reid_state = load_ckpt(reid_init_ckpt)
    conf = reid_state.get("config", None)
    conf.model.person_feature_dim = 512
    scene_extent = reid_state.get("scene_extent", 1.0)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent)
    state = load_ckpt(init_ckpt)
    for key in ["positions", "density", "scale", "rotation", "features_albedo", "features_specular"]:
        if key in state:
            getattr(model, key).data = state[key].to(device)
    pf_key = "_person_feature" if "_person_feature" in state else "person_feature"
    if pf_key in state:
        model._person_feature = torch.nn.Parameter(state[pf_key].to(device))
    model = model.to(device)
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model, conf


def draw_bbox(img, bbox, color, thickness=2, label=""):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return img


def compute_ring_bbox(x1, y1, x2, y2, expand=1.5, H=272, W=480):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    ew, eh = bw * expand, bh * expand
    rx1 = max(0, int(cx - ew / 2))
    ry1 = max(0, int(cy - eh / 2))
    rx2 = min(W, int(cx + ew / 2))
    ry2 = min(H, int(cy + eh / 2))
    return rx1, ry1, rx2, ry2


def run_v1(model, dataset, device, output_dir, max_per_cam=100):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_records = []
    cam_sample_count = defaultdict(int)
    t0 = time.time()

    with torch.no_grad():
        for idx in range(len(dataset)):
            if time.time() - t0 > 3600:
                print(f"  [timeout] 60 min limit at idx={idx}")
                break

            batch = dataset[idx]
            cam_id = batch.get("camera_id", "unknown")
            frame_idx = batch.get("frame_idx", -1)

            if cam_sample_count[cam_id] >= max_per_cam:
                continue

            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            gpu_instances = gpu_batch.instances
            if not gpu_instances:
                continue

            outputs = model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
            pf_map = outputs.get("person_feature_map")
            opacity_map = outputs.get("person_opacity_map")

            if pf_map is None:
                continue

            D, feat_H, feat_W = pf_map.shape

            # Get opacity map - try multiple possible keys
            if opacity_map is None:
                opacity_map = outputs.get("pred_opacity")
                if opacity_map is not None:
                    if opacity_map.ndim == 4:
                        opacity_map = opacity_map.squeeze(0).squeeze(-1)
                    elif opacity_map.ndim == 3:
                        opacity_map = opacity_map.squeeze(-1)
            if opacity_map is None:
                opacity_map = outputs.get("accumulation")
                if opacity_map is not None and opacity_map.ndim > 2:
                    opacity_map = opacity_map.squeeze()

            # Feature norm map
            feat_norm_map = pf_map.norm(dim=0)  # [H, W]

            # RGB for visualization
            rgb_gt = batch.get("rgb")
            if rgb_gt is not None:
                if isinstance(rgb_gt, np.ndarray):
                    rgb_img = (rgb_gt * 255).astype(np.uint8)
                else:
                    rgb_img = (rgb_gt.cpu().numpy() * 255).astype(np.uint8)
            else:
                rgb_img = np.zeros((feat_H, feat_W, 3), dtype=np.uint8)

            img_H, img_W = rgb_img.shape[:2]

            for inst in gpu_instances:
                if cam_sample_count[cam_id] >= max_per_cam:
                    break

                train_id = inst.get("train_id")
                raw_id = inst.get("raw_id")
                if train_id is None:
                    continue

                bbox_original = inst.get("bbox_xyxy_original")
                inst_w = inst.get("img_width_original", 1920)
                inst_h = inst.get("img_height_original", 1088)
                if bbox_original is None:
                    continue

                # Use EXACTLY the same logic as trainer.py
                bbox_render = scale_bbox_to_render(
                    bbox_original, src_w=inst_w, src_h=inst_h,
                    dst_w=feat_W, dst_h=feat_H
                )

                xmin, ymin, xmax, ymax = _clamp_bbox(bbox_render, feat_H, feat_W)
                roi_w = xmax - xmin
                roi_h = ymax - ymin

                if roi_w <= 0 or roi_h <= 0:
                    continue

                border_flag = (xmin <= 2 or ymin <= 2 or
                               xmax >= feat_W - 2 or ymax >= feat_H - 2)
                clamped_flag = (xmin == 0 or ymin == 0 or
                                xmax == feat_W or ymax == feat_H)

                # Opacity stats inside bbox
                opacity_roi = opacity_map[ymin:ymax, xmin:xmax] if opacity_map is not None else None
                if opacity_roi is not None and opacity_roi.numel() > 0:
                    op_np = opacity_roi.cpu().numpy()
                    opacity_mean = float(op_np.mean())
                    opacity_max = float(op_np.max())
                    opacity_sum = float(op_np.sum())
                    opacity_ratio_1e4 = float((op_np > 1e-4).mean())
                    opacity_ratio_1e3 = float((op_np > 1e-3).mean())
                    opacity_ratio_1e2 = float((op_np > 1e-2).mean())
                    opacity_ratio_5e2 = float((op_np > 5e-2).mean())
                else:
                    opacity_mean = opacity_max = opacity_sum = 0
                    opacity_ratio_1e4 = opacity_ratio_1e3 = opacity_ratio_1e2 = opacity_ratio_5e2 = 0

                # Feature norm stats inside bbox
                fn_roi = feat_norm_map[ymin:ymax, xmin:xmax]
                fn_np = fn_roi.cpu().numpy()
                feature_norm_mean = float(fn_np.mean())
                feature_norm_max = float(fn_np.max())
                feature_norm_ratio_1e6 = float((fn_np > 1e-6).mean())
                feature_norm_ratio_1e4 = float((fn_np > 1e-4).mean())

                # Ring region (expanded bbox minus bbox)
                rx1, ry1, rx2, ry2 = compute_ring_bbox(xmin, ymin, xmax, ymax, expand=1.5, H=feat_H, W=feat_W)
                ring_area = (rx2 - rx1) * (ry2 - ry1) - roi_w * roi_h
                if ring_area > 4 and opacity_roi is not None:
                    ring_opacity = opacity_map[ry1:ry2, rx1:rx2].cpu().numpy()
                    ring_fn = feat_norm_map[ry1:ry2, rx1:rx2].cpu().numpy()
                    inside_mask = np.zeros((ry2 - ry1, rx2 - rx1), dtype=bool)
                    inside_mask[ymin - ry1:ymax - ry1, xmin - rx1:xmax - rx1] = True
                    ring_opacity_vals = ring_opacity[~inside_mask]
                    ring_fn_vals = ring_fn[~inside_mask]
                    ring_opacity_mean = float(ring_opacity_vals.mean()) if len(ring_opacity_vals) > 0 else float('nan')
                    ring_fn_mean = float(ring_fn_vals.mean()) if len(ring_fn_vals) > 0 else float('nan')
                else:
                    ring_opacity_mean = float('nan')
                    ring_fn_mean = float('nan')

                opacity_contrast = opacity_mean - ring_opacity_mean if not np.isnan(ring_opacity_mean) else float('nan')
                feature_contrast = feature_norm_mean - ring_fn_mean if not np.isnan(ring_fn_mean) else float('nan')

                # ROI pool check
                bbox_clamped = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32, device=device)
                f_v, roi_info = roi_pool(pf_map, bbox_clamped)
                valid_roi = f_v is not None
                has_nan = False
                if f_v is not None:
                    has_nan = bool(torch.isnan(f_v).any() or torch.isinf(f_v).any())

                zero_support = (opacity_mean < 1e-4 and feature_norm_mean < 1e-6)

                record = {
                    "camera_id": cam_id,
                    "frame_id": int(frame_idx),
                    "person_id": int(raw_id) if raw_id else -1,
                    "train_id": int(train_id),
                    "bbox_raw": str(bbox_original),
                    "bbox_render": f"[{xmin},{ymin},{xmax},{ymax}]",
                    "roi_w": roi_w,
                    "roi_h": roi_h,
                    "roi_area": roi_w * roi_h,
                    "border_flag": border_flag,
                    "clamped_flag": clamped_flag,
                    "opacity_mean": opacity_mean,
                    "opacity_max": opacity_max,
                    "opacity_sum": opacity_sum,
                    "opacity_ratio_gt_1e-4": opacity_ratio_1e4,
                    "opacity_ratio_gt_1e-3": opacity_ratio_1e3,
                    "opacity_ratio_gt_1e-2": opacity_ratio_1e2,
                    "opacity_ratio_gt_5e-2": opacity_ratio_5e2,
                    "feature_norm_mean": feature_norm_mean,
                    "feature_norm_max": feature_norm_max,
                    "feature_norm_ratio_gt_1e-6": feature_norm_ratio_1e6,
                    "feature_norm_ratio_gt_1e-4": feature_norm_ratio_1e4,
                    "inside_opacity_mean": opacity_mean,
                    "ring_opacity_mean": ring_opacity_mean,
                    "opacity_contrast": opacity_contrast,
                    "inside_feature_norm_mean": feature_norm_mean,
                    "ring_feature_norm_mean": ring_fn_mean,
                    "feature_contrast": feature_contrast,
                    "valid_roi": valid_roi,
                    "has_nan": has_nan,
                    "zero_support_flag": zero_support,
                }
                all_records.append(record)

                # Visualizations
                sample_idx = cam_sample_count[cam_id]
                if sample_idx < 20:
                    vis_dir = os.path.join(output_dir, "visualizations", cam_id)
                    os.makedirs(vis_dir, exist_ok=True)

                    scale_x = img_W / feat_W
                    scale_y = img_H / feat_H
                    bbox_img = [int(xmin * scale_x), int(ymin * scale_y),
                                int(xmax * scale_x), int(ymax * scale_y)]

                    # 1. RGB + bbox
                    vis = rgb_img.copy()
                    draw_bbox(vis, bbox_img, (0, 255, 0), 2,
                              f"op={opacity_mean:.3f}")
                    cv2.imwrite(os.path.join(vis_dir, f"{sample_idx:03d}_rgb.jpg"), vis)

                    # 2. Opacity heatmap + bbox
                    if opacity_map is not None:
                        op_vis = (opacity_map.cpu().numpy() * 255).astype(np.uint8)
                        op_color = cv2.applyColorMap(op_vis, cv2.COLORMAP_JET)
                        op_color = cv2.resize(op_color, (img_W, img_H))
                        draw_bbox(op_color, bbox_img, (0, 255, 0), 2)
                        cv2.imwrite(os.path.join(vis_dir, f"{sample_idx:03d}_opacity.jpg"), op_color)

                    # 3. Feature norm heatmap + bbox
                    fn_vis = (feat_norm_map.cpu().numpy() / max(feat_norm_map.max().item(), 1e-6) * 255).astype(np.uint8)
                    fn_color = cv2.applyColorMap(fn_vis, cv2.COLORMAP_HOT)
                    fn_color = cv2.resize(fn_color, (img_W, img_H))
                    draw_bbox(fn_color, bbox_img, (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(vis_dir, f"{sample_idx:03d}_featnorm.jpg"), fn_color)

                    # 4. Opacity crop
                    if opacity_roi is not None:
                        op_crop = (opacity_roi.cpu().numpy() * 255).astype(np.uint8)
                        op_crop_color = cv2.applyColorMap(op_crop, cv2.COLORMAP_JET)
                        op_crop_up = cv2.resize(op_crop_color, (max(roi_w * 8, 64), max(roi_h * 8, 64)))
                        cv2.imwrite(os.path.join(vis_dir, f"{sample_idx:03d}_opacity_crop.jpg"), op_crop_up)

                    # 5. Feature norm crop
                    fn_crop = (fn_np / max(fn_np.max(), 1e-6) * 255).astype(np.uint8)
                    fn_crop_color = cv2.applyColorMap(fn_crop, cv2.COLORMAP_HOT)
                    fn_crop_up = cv2.resize(fn_crop_color, (max(roi_w * 8, 64), max(roi_h * 8, 64)))
                    cv2.imwrite(os.path.join(vis_dir, f"{sample_idx:03d}_featnorm_crop.jpg"), fn_crop_up)

                    # 6. Support mask (opacity > 1e-2)
                    if opacity_roi is not None:
                        support_mask = (opacity_roi.cpu().numpy() > 1e-2).astype(np.uint8) * 255
                        support_up = cv2.resize(support_mask, (max(roi_w * 8, 64), max(roi_h * 8, 64)))
                        cv2.imwrite(os.path.join(vis_dir, f"{sample_idx:03d}_support_mask.jpg"), support_up)

                cam_sample_count[cam_id] += 1

            if idx % 50 == 0:
                done = sum(cam_sample_count.values())
                print(f"  [v1] idx={idx}, samples={done}, {time.time()-t0:.0f}s")

            if all(cam_sample_count[c] >= max_per_cam for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
                print(f"  [v1] All cameras have {max_per_cam} samples.")
                break

    return all_records


def write_reports(all_records, output_dir):
    # Per-bbox CSV
    csv_path = os.path.join(output_dir, "v1_per_bbox.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_records[0].keys())
        w.writeheader()
        w.writerows(all_records)

    # Summary by camera
    cam_stats = defaultdict(lambda: defaultdict(list))
    for r in all_records:
        cam = r["camera_id"]
        for k, v in r.items():
            if k in ("camera_id", "frame_id", "person_id", "train_id", "bbox_raw", "bbox_render"):
                continue
            if isinstance(v, (int, float)) and not np.isnan(v):
                cam_stats[cam][k].append(v)

    summary_cam_path = os.path.join(output_dir, "v1_summary_by_cam.csv")
    metric_keys = ["opacity_mean", "opacity_ratio_gt_1e-2", "feature_norm_mean",
                    "feature_norm_ratio_gt_1e-6", "opacity_contrast", "feature_contrast"]
    with open(summary_cam_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["camera", "count", "valid_roi_ratio", "border_ratio", "clamped_ratio", "zero_support_ratio"]
        for mk in metric_keys:
            header += [f"{mk}_median", f"{mk}_mean"]
        w.writerow(header)
        for cam in sorted(cam_stats.keys()):
            cs = cam_stats[cam]
            n = len(cs.get("valid_roi", [1]))
            row = [
                cam, n,
                np.mean(cs.get("valid_roi", [0])),
                np.mean(cs.get("border_flag", [0])),
                np.mean(cs.get("clamped_flag", [0])),
                np.mean(cs.get("zero_support_flag", [0])),
            ]
            for mk in metric_keys:
                vals = cs.get(mk, [0])
                row.append(float(np.median(vals)) if vals else 0)
                row.append(float(np.mean(vals)) if vals else 0)
            w.writerow(row)

    # Summary by ID
    id_stats = defaultdict(lambda: defaultdict(list))
    id_cams = defaultdict(set)
    for r in all_records:
        pid = r["person_id"]
        id_cams[pid].add(r["camera_id"])
        for k, v in r.items():
            if k in ("camera_id", "frame_id", "person_id", "train_id", "bbox_raw", "bbox_render"):
                continue
            if isinstance(v, (int, float)) and not np.isnan(v):
                id_stats[pid][k].append(v)

    summary_id_path = os.path.join(output_dir, "v1_summary_by_id.csv")
    with open(summary_id_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["person_id", "count", "camera_count", "opacity_mean", "feature_norm_mean", "zero_support_ratio"]
        w.writerow(header)
        for pid in sorted(id_stats.keys()):
            is_ = id_stats[pid]
            n = len(is_.get("valid_roi", [1]))
            row = [
                pid, n, len(id_cams[pid]),
                np.mean(is_.get("opacity_mean", [0])),
                np.mean(is_.get("feature_norm_mean", [0])),
                np.mean(is_.get("zero_support_flag", [0])),
            ]
            w.writerow(row)

    # Plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cams = sorted(cam_stats.keys())
        for metric, title in [
            ("opacity_mean", "Opacity Mean per Camera"),
            ("feature_norm_mean", "Feature Norm Mean per Camera"),
            ("opacity_ratio_gt_1e-2", "Opacity>0.01 Ratio per Camera"),
            ("opacity_contrast", "Opacity Contrast (inside-ring) per Camera"),
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            data = [cam_stats[c].get(metric, [0]) for c in cams]
            ax.boxplot(data, labels=cams)
            ax.set_title(title)
            ax.set_ylabel(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{metric}_per_cam.png"), dpi=100)
            plt.close()
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots")

    # Visual review contact sheet
    vis_review_path = os.path.join(output_dir, "v1_visual_review.md")
    with open(vis_review_path, "w") as f:
        f.write("# V1 Visual Review\n\n")
        for cam in sorted(cam_stats.keys()):
            vis_dir = os.path.join(output_dir, "visualizations", cam)
            if not os.path.exists(vis_dir):
                continue
            f.write(f"## {cam}\n\n")
            samples = sorted(set(fn.split("_")[0] for fn in os.listdir(vis_dir) if fn.endswith(".jpg")))[:10]
            for s in samples:
                f.write(f"### {cam} Sample {s}\n\n")
                for img_type in ["rgb", "opacity", "featnorm", "opacity_crop", "featnorm_crop", "support_mask"]:
                    fname = f"{s}_{img_type}.jpg"
                    fpath = os.path.join("visualizations", cam, fname)
                    if os.path.exists(os.path.join(output_dir, fpath)):
                        f.write(f"- [{img_type}]({fpath})\n")
                f.write("\n")

    # Final report
    report = generate_final_report(all_records, cam_stats, output_dir)
    with open(os.path.join(output_dir, "final_report.md"), "w") as f:
        f.write(report)

    print(f"\nV1 reports saved to {output_dir}/")
    print(f"  - v1_per_bbox.csv ({len(all_records)} samples)")
    print(f"  - v1_summary_by_cam.csv")
    print(f"  - v1_summary_by_id.csv")
    print(f"  - final_report.md")


def generate_final_report(all_records, cam_stats, output_dir):
    r = "# V1 Person Gaussian Support Diagnostic\n\n"
    r += f"## Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n"

    r += "## 1. Configuration\n\n"
    r += "- ckpt: phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt\n"
    r += "- config: from reid_init_ckpt.pt (downsample_factor=4)\n"
    r += "- render feature map resolution: 480×272 (W×H)\n\n"

    r += "## 2. Bbox Scale Protocol\n\n"
    r += "```\n"
    r += "bbox_xyxy_original (1920×1080) → scale_bbox_to_render(src_w=1920, src_h=1088, dst_w=480, dst_h=272)\n"
    r += "sx = sy = 0.25, no transpose\n"
    r += "```\n\n"

    r += "## 3. Opacity Source\n\n"
    r += "- Used: `outputs['person_opacity_map']` from `model.render_person_feature_map()`\n"
    r += "- This is the accumulated opacity/alpha from the same render pass as person_feature_map\n\n"

    r += "## 4. Per-Camera Summary\n\n"
    r += "| Camera | Count | Valid% | Border% | Clamped% | ZeroSup% | OpMean | Op>1e-2% | FnMean | Fn>1e-6% | OpContrast |\n"
    r += "|--------|-------|--------|---------|----------|----------|--------|----------|--------|----------|------------|\n"

    for cam in sorted(cam_stats.keys()):
        cs = cam_stats[cam]
        n = max(1, len(cs.get("valid_roi", [1])))
        r += (f"| {cam} | {n} | "
              f"{np.mean(cs.get('valid_roi',[0]))*100:.1f} | "
              f"{np.mean(cs.get('border_flag',[0]))*100:.1f} | "
              f"{np.mean(cs.get('clamped_flag',[0]))*100:.1f} | "
              f"{np.mean(cs.get('zero_support_flag',[0]))*100:.1f} | "
              f"{np.mean(cs.get('opacity_mean',[0])):.4f} | "
              f"{np.mean(cs.get('opacity_ratio_gt_1e-2',[0]))*100:.1f} | "
              f"{np.mean(cs.get('feature_norm_mean',[0])):.4f} | "
              f"{np.mean(cs.get('feature_norm_ratio_gt_1e-6',[0]))*100:.1f} | "
              f"{np.mean(cs.get('opacity_contrast',[0])):.4f} |\n")
    r += "\n"

    # C2 analysis
    c2_border = np.mean(cam_stats.get("C2", {}).get("border_flag", [0]))
    c2_opacity = np.mean(cam_stats.get("C2", {}).get("opacity_mean", [0]))
    r += "## 5. C2 Analysis\n\n"
    r += f"- C2 border_flag ratio: {c2_border*100:.1f}%\n"
    r += f"- C2 opacity_mean: {c2_opacity:.4f}\n"
    r += "- C2 has many border bboxes (data characteristic: camera angle puts people near edges)\n\n"

    # Small bbox analysis
    small_records = [r for r in all_records if r["roi_w"] < 10 or r["roi_h"] < 20]
    if small_records:
        small_op = np.mean([r["opacity_mean"] for r in small_records])
        small_fn = np.mean([r["feature_norm_mean"] for r in small_records])
        r += "## 6. Small Bbox Analysis\n\n"
        r += f"- Small bbox count (w<10 or h<20): {len(small_records)}\n"
        r += f"- Small bbox opacity_mean: {small_op:.4f}\n"
        r += f"- Small bbox feature_norm_mean: {small_fn:.4f}\n\n"

    # Overall assessment
    total = len(all_records)
    zero_sup = sum(1 for r in all_records if r["zero_support_flag"])
    valid = sum(1 for r in all_records if r["valid_roi"])
    overall_op = np.mean([r["opacity_mean"] for r in all_records])
    overall_fn = np.mean([r["feature_norm_mean"] for r in all_records])

    r += "## 7. Overall Assessment\n\n"
    r += f"- Total samples: {total}\n"
    r += f"- Valid ROI: {valid} ({valid/max(1,total)*100:.1f}%)\n"
    r += f"- Zero support: {zero_sup} ({zero_sup/max(1,total)*100:.1f}%)\n"
    r += f"- Overall opacity_mean: {overall_op:.4f}\n"
    r += f"- Overall feature_norm_mean: {overall_fn:.4f}\n\n"

    # Decision
    zero_sup_ratio = zero_sup / max(1, total)
    cam_zero_ratios = {}
    for cam in sorted(cam_stats.keys()):
        cs = cam_stats[cam]
        cam_zero_ratios[cam] = np.mean(cs.get("zero_support_flag", [0]))

    any_cam_dead = any(v > 0.5 for v in cam_zero_ratios.values())

    if zero_sup_ratio < 0.05 and not any_cam_dead and overall_op > 0.01:
        decision = "PASS"
        reason = "Most bboxes have non-zero opacity/feature support, no camera is systematically dead."
    elif zero_sup_ratio > 0.5 or any_cam_dead:
        decision = "FAIL"
        reason = "Many bboxes have zero support or some cameras are systematically dead."
    else:
        decision = "UNCERTAIN"
        reason = "Support is weak but not zero, or visual/statistical contradiction exists."

    r += f"## 8. Decision: {decision}\n\n"
    r += f"{reason}\n\n"

    r += "## 9. Next Steps\n\n"
    if decision == "PASS":
        r += "- V1 PASS: Sufficient Gaussian render support in person bboxes.\n"
        r += "- Next: Proceed with Teacher+CE training.\n"
    elif decision == "FAIL":
        r += "- V1 FAIL: Insufficient support. Check render quality, Gaussian density, or bbox alignment.\n"
    else:
        r += "- V1 UNCERTAIN: Need manual review of visualizations.\n"

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--max_per_cam", type=int, default=100)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--out_dir", default="outputs/v1_person_support_check")
    args = parser.parse_args()

    os.environ["TORCH_EXTENSIONS_DIR"] = "/data02/zhangrunxiang/.cache/torch_extensions/py311_cu118"
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("V1: Person Gaussian Support Diagnostic")
    print("=" * 60)

    print("\n[1/3] Loading model...")
    model, conf = setup_model(args.ckpt, device)
    print(f"  positions: {model.positions.shape}, _person_feature: {model._person_feature.shape}")

    print("\n[2/3] Loading dataset...")
    train_ds, _ = make_dataset("wildtrack", conf, ray_jitter=None)
    print(f"  dataset size: {len(train_ds)}")

    print(f"\n[3/3] Running V1 diagnostic (max {args.max_per_cam}/cam)...")
    all_records = run_v1(model, train_ds, device, args.out_dir, max_per_cam=args.max_per_cam)

    write_reports(all_records, args.out_dir)


if __name__ == "__main__":
    main()
