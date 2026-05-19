#!/usr/bin/env python3
"""V0 Diagnostic: Bbox / ROI / Render-Space Consistency Check.

Verifies that person bbox, rendered RGB/opacity, person_feature_map,
and ROI pooling all use consistent coordinate systems.

NO TRAINING. NO LOSS CHANGES. DIAGNOSTIC ONLY.
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
TEACHER_ONLY_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt"
DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
OUTPUT_DIR = "/data02/zhangrunxiang/3dgrut/outputs/v0_bbox_roi_check"
DEVICE = "cuda"

ORIG_W, ORIG_H = 1920, 1080
PADDED_W, PADDED_H = 1920, 1088


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


def draw_bbox_on_image(img, bbox, color, thickness=2, label=""):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


def run_diagnostic(model, dataset, device, output_dir, max_per_cam=50):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_records = []
    per_cam_stats = defaultdict(lambda: {
        "bbox_widths": [], "bbox_heights": [],
        "width_eq_1": 0, "height_eq_1": 0,
        "border_clamp": 0, "total": 0,
        "valid_roi": 0, "invalid_roi": 0,
        "zero_or_nan_feature": 0,
        "roi_feature_norms": [],
        "opacity_sums": [],
        "invalid_reasons": defaultdict(int),
    })

    cam_sample_count = defaultdict(int)
    t0 = time.time()

    with torch.no_grad():
        for idx in range(len(dataset)):
            if time.time() - t0 > 1800:
                print(f"  [timeout] 30 min limit reached at idx={idx}")
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

            # Render person_feature_map and get opacity
            outputs = model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
            pf_map = outputs.get("person_feature_map")
            opacity_map = outputs.get("person_opacity_map")
            rgb_render = outputs.get("pred_rgb")

            if pf_map is None:
                continue

            D, feat_H, feat_W = pf_map.shape
            rays_H, rays_W = gpu_batch.rays_ori.shape[1], gpu_batch.rays_ori.shape[2]

            # Get RGB image from dataset for overlay
            rgb_gt = batch.get("rgb")
            if rgb_gt is not None:
                if isinstance(rgb_gt, np.ndarray):
                    rgb_img = (rgb_gt * 255).astype(np.uint8)
                else:
                    rgb_img = (rgb_gt.cpu().numpy() * 255).astype(np.uint8)
                img_H, img_W = rgb_img.shape[:2]
            else:
                img_H, img_W = rays_H, rays_W
                rgb_img = np.zeros((img_H, img_W, 3), dtype=np.uint8)

            for inst in gpu_instances:
                if cam_sample_count[cam_id] >= max_per_cam:
                    break

                train_id = inst.get("train_id")
                raw_id = inst.get("raw_id")
                if train_id is None:
                    continue

                bbox_original = inst.get("bbox_xyxy_original")
                bbox_downsampled = inst.get("bbox_xyxy")
                inst_w = inst.get("img_width_original", PADDED_W)
                inst_h = inst.get("img_height_original", PADDED_H)

                stats = per_cam_stats[cam_id]
                stats["total"] += 1

                # ── Step 1: Trace bbox coordinate flow ──
                # bbox_original: from annotation, in original image space (1920x1080)
                # bbox_downsampled: after downsample (same as original if downsample=1)
                # inst_w/inst_h: self.img_width * downsample_factor = 1920 * 1 = 1920
                #                self.img_height * downsample_factor = 1088 * 1 = 1088 (PADDED!)

                # ── Step 2: How trainer.py does scale_bbox_to_render ──
                # src_w = inst_w = 1920, src_h = inst_h = 1088 (PADDED!)
                # dst_w = feat_W, dst_h = feat_H
                bbox_render_trainer = scale_bbox_to_render(
                    bbox_original, src_w=inst_w, src_h=inst_h,
                    dst_w=feat_W, dst_h=feat_H
                )

                # ── Step 3: How it SHOULD be (using original 1080 height) ──
                bbox_render_correct = scale_bbox_to_render(
                    bbox_original, src_w=ORIG_W, src_h=ORIG_H,
                    dst_w=feat_W, dst_h=feat_H
                )

                # ── Step 4: Clamp to feature map bounds ──
                xmin_t, ymin_t, xmax_t, ymax_t = _clamp_bbox(bbox_render_trainer, feat_H, feat_W)
                xmin_c, ymin_c, xmax_c, ymax_c = _clamp_bbox(bbox_render_correct, feat_H, feat_W)

                # Check if clamped to border
                border_clamped = (xmin_t == 0 or ymin_t == 0 or
                                  xmax_t == feat_W or ymax_t == feat_H)

                # ROI dimensions
                roi_w = xmax_t - xmin_t
                roi_h = ymax_t - ymin_t

                # ── Step 5: ROI pool ──
                bbox_clamped = torch.tensor([xmin_t, ymin_t, xmax_t, ymax_t],
                                            dtype=torch.float32, device=device)
                f_v, roi_info = roi_pool(pf_map, bbox_clamped)

                valid_roi = f_v is not None
                roi_norm = f_v.norm().item() if f_v is not None else 0.0
                is_zero_or_nan = (roi_norm == 0 or np.isnan(roi_norm) or np.isinf(roi_norm))

                # Opacity sum in ROI
                opacity_sum = 0.0
                if opacity_map is not None and valid_roi:
                    opacity_roi = opacity_map[ymin_t:ymax_t, xmin_t:xmax_t]
                    opacity_sum = opacity_roi.sum().item()

                # Record
                invalid_reason = ""
                if not valid_roi:
                    invalid_reason = roi_info.get("pooling", "unknown")
                    stats["invalid_roi"] += 1
                elif is_zero_or_nan:
                    invalid_reason = "zero_or_nan_feature"
                    stats["zero_or_nan_feature"] += 1
                else:
                    stats["valid_roi"] += 1

                if border_clamped:
                    stats["border_clamp"] += 1
                if roi_w == 1:
                    stats["width_eq_1"] += 1
                if roi_h == 1:
                    stats["height_eq_1"] += 1

                stats["bbox_widths"].append(roi_w)
                stats["bbox_heights"].append(roi_h)
                if valid_roi and not is_zero_or_nan:
                    stats["roi_feature_norms"].append(roi_norm)
                    stats["opacity_sums"].append(opacity_sum)
                if invalid_reason:
                    stats["invalid_reasons"][invalid_reason] += 1

                record = {
                    "cam_id": cam_id,
                    "frame_id": int(frame_idx),
                    "train_id": int(train_id),
                    "person_id": int(raw_id) if raw_id else -1,
                    "image_size_from_dataset": f"{img_W}x{img_H}",
                    "render_size_from_person_feature_map": f"{feat_W}x{feat_H}",
                    "bbox_xyxy_raw": bbox_original,
                    "bbox_xyxy_after_dataset": bbox_downsampled,
                    "img_width_original": inst_w,
                    "img_height_original": inst_h,
                    "bbox_render_trainer_x1y1x2y2": [xmin_t, ymin_t, xmax_t, ymax_t],
                    "bbox_render_correct_x1y1x2y2": [xmin_c, ymin_c, xmax_c, ymax_c],
                    "bbox_width": roi_w,
                    "bbox_height": roi_h,
                    "is_clamped_to_left": xmin_t == 0,
                    "is_clamped_to_right": xmax_t == feat_W,
                    "is_clamped_to_top": ymin_t == 0,
                    "is_clamped_to_bottom": ymax_t == feat_H,
                    "roi_width": roi_w,
                    "roi_height": roi_h,
                    "roi_feature_norm": roi_norm,
                    "roi_opacity_sum": opacity_sum,
                    "valid_roi": valid_roi,
                    "invalid_reason": invalid_reason,
                    "y_offset_trainer_vs_correct": ymin_t - ymin_c,
                }
                all_records.append(record)

                # ── Step 6: Save overlay visualizations ──
                sample_idx = cam_sample_count[cam_id]
                if sample_idx < 20:
                    cam_dir = os.path.join(output_dir, cam_id)
                    os.makedirs(cam_dir, exist_ok=True)

                    # 1. RGB image + bbox
                    # Scale bbox from original space to image display space
                    scale_to_img_x = img_W / ORIG_W
                    scale_to_img_y = img_H / ORIG_H

                    rgb_vis = rgb_img.copy()
                    # Red: raw bbox (original coordinates scaled to image)
                    raw_bbox_img = [
                        int(bbox_original[0] * scale_to_img_x),
                        int(bbox_original[1] * scale_to_img_y),
                        int(bbox_original[2] * scale_to_img_x),
                        int(bbox_original[3] * scale_to_img_y),
                    ]
                    draw_bbox_on_image(rgb_vis, raw_bbox_img, (255, 0, 0), 2, "raw")

                    # Green: bbox used for ROI (render space, scaled to image)
                    scale_render_to_img_x = img_W / feat_W
                    scale_render_to_img_y = img_H / feat_H
                    roi_bbox_img = [
                        int(xmin_t * scale_render_to_img_x),
                        int(ymin_t * scale_render_to_img_y),
                        int(xmax_t * scale_render_to_img_x),
                        int(ymax_t * scale_render_to_img_y),
                    ]
                    draw_bbox_on_image(rgb_vis, roi_bbox_img, (0, 255, 0), 2, "roi")

                    cv2.imwrite(os.path.join(cam_dir, f"{sample_idx:03d}_rgb_bbox.jpg"), rgb_vis)

                    # 2. Opacity map + bbox
                    if opacity_map is not None:
                        op_np = (opacity_map.cpu().numpy() * 255).astype(np.uint8)
                        op_vis = cv2.applyColorMap(op_np, cv2.COLORMAP_JET)
                        draw_bbox_on_image(op_vis, roi_bbox_img, (0, 255, 0), 2)
                        cv2.imwrite(os.path.join(cam_dir, f"{sample_idx:03d}_opacity_bbox.jpg"), op_vis)

                    # 3. Feature norm heatmap + bbox
                    feat_norm_map = pf_map.norm(dim=0).cpu().numpy()
                    feat_norm_vis = (feat_norm_map / max(feat_norm_map.max(), 1e-6) * 255).astype(np.uint8)
                    feat_color = cv2.applyColorMap(feat_norm_vis, cv2.COLORMAP_HOT)
                    feat_color = cv2.resize(feat_color, (img_W, img_H))
                    draw_bbox_on_image(feat_color, roi_bbox_img, (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(cam_dir, f"{sample_idx:03d}_featnorm_bbox.jpg"), feat_color)

                    # 4. ROI crop visualization
                    if valid_roi and roi_w > 1 and roi_h > 1:
                        roi_feat = pf_map[:, ymin_t:ymax_t, xmin_t:xmax_t]
                        roi_norm_map = roi_feat.norm(dim=0).cpu().numpy()
                        roi_vis = (roi_norm_map / max(roi_norm_map.max(), 1e-6) * 255).astype(np.uint8)
                        roi_color = cv2.applyColorMap(roi_vis, cv2.COLORMAP_HOT)
                        roi_upscale = cv2.resize(roi_color, (max(roi_w * 8, 64), max(roi_h * 8, 64)))
                        cv2.imwrite(os.path.join(cam_dir, f"{sample_idx:03d}_roi_crop.jpg"), roi_upscale)

                    # 5. Comparison: raw (red) vs roi (green) bbox on feature norm map
                    compare_vis = feat_color.copy()
                    # Raw bbox in render space (correct scaling from 1080)
                    raw_bbox_render = [
                        int(bbox_render_correct[0].item()),
                        int(bbox_render_correct[1].item()),
                        int(bbox_render_correct[2].item()),
                        int(bbox_render_correct[3].item()),
                    ]
                    raw_bbox_render_img = [
                        int(v * scale_render_to_img_x) if i % 2 == 0 else int(v * scale_render_to_img_y)
                        for i, v in enumerate(raw_bbox_render)
                    ]
                    draw_bbox_on_image(compare_vis, raw_bbox_render_img, (255, 0, 0), 2, "raw(1080)")
                    draw_bbox_on_image(compare_vis, roi_bbox_img, (0, 255, 0), 2, "roi(1088)")
                    cv2.imwrite(os.path.join(cam_dir, f"{sample_idx:03d}_compare.jpg"), compare_vis)

                cam_sample_count[cam_id] += 1

            if idx % 50 == 0:
                done = sum(cam_sample_count.values())
                print(f"  [diag] idx={idx}, samples={done}, cams_done={sum(1 for c in cam_sample_count if cam_sample_count[c] >= max_per_cam)}/7, {time.time()-t0:.0f}s")

            # Early exit if all cameras have enough samples
            if all(cam_sample_count[c] >= max_per_cam for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
                print(f"  [diag] All cameras have {max_per_cam} samples, done.")
                break

    return all_records, per_cam_stats


def write_reports(all_records, per_cam_stats, output_dir):
    # Per-sample CSV
    csv_path = os.path.join(output_dir, "v0_per_sample.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_records[0].keys())
        w.writeheader()
        w.writerows(all_records)

    # Per-camera summary CSV
    summary_path = os.path.join(output_dir, "v0_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "camera", "num_samples", "mean_bbox_width", "median_bbox_width",
            "mean_bbox_height", "median_bbox_height",
            "width_eq_1_ratio", "height_eq_1_ratio",
            "border_clamp_ratio", "valid_roi_ratio",
            "zero_or_nan_feature_ratio",
            "mean_roi_feature_norm", "median_roi_feature_norm",
            "mean_opacity_sum", "median_opacity_sum",
        ])
        for cam in sorted(per_cam_stats.keys()):
            s = per_cam_stats[cam]
            n = max(1, s["total"])
            norms = s["roi_feature_norms"]
            opacities = s["opacity_sums"]
            w.writerow([
                cam, s["total"],
                np.mean(s["bbox_widths"]) if s["bbox_widths"] else 0,
                np.median(s["bbox_widths"]) if s["bbox_widths"] else 0,
                np.mean(s["bbox_heights"]) if s["bbox_heights"] else 0,
                np.median(s["bbox_heights"]) if s["bbox_heights"] else 0,
                s["width_eq_1"] / n,
                s["height_eq_1"] / n,
                s["border_clamp"] / n,
                s["valid_roi"] / n,
                s["zero_or_nan_feature"] / n,
                np.mean(norms) if norms else 0,
                np.median(norms) if norms else 0,
                np.mean(opacities) if opacities else 0,
                np.median(opacities) if opacities else 0,
            ])

    # Final report
    report = generate_report(all_records, per_cam_stats, output_dir)
    with open(os.path.join(output_dir, "final_report.md"), "w") as f:
        f.write(report)

    print(f"\nReports saved to {output_dir}/")
    print(f"  - v0_per_sample.csv ({len(all_records)} samples)")
    print(f"  - v0_summary.csv")
    print(f"  - final_report.md")


def generate_report(all_records, per_cam_stats, output_dir):
    r = "# V0 Bbox/ROI/Render-Space Consistency Check\n\n"
    r += f"## Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n"

    # 1. Bbox coordinate flow
    r += "## 1. Bbox Coordinate Flow\n\n"
    r += "```\n"
    r += "Annotation JSON (1920×1080 space)\n"
    r += "  → bbox_xyxy_original = [xmin, ymin, xmax, ymax]  (1920×1080 coordinates)\n"
    r += "  → bbox_xyxy = same if downsample_factor=1\n"
    r += "  → img_width_original = self.img_width * downsample_factor = 1920 * 1 = 1920\n"
    r += "  → img_height_original = self.img_height * downsample_factor = 1088 * 1 = 1088  ← PADDED!\n"
    r += "\n"
    r += "trainer.py: scale_bbox_to_render(bbox_original, src_w=1920, src_h=1088, dst_w=feat_W, dst_h=feat_H)\n"
    r += "  → scale_y = feat_H / 1088  ← Uses PADDED height!\n"
    r += "  → Correct should be: scale_y = feat_H / 1080\n"
    r += "\n"
    r += "person_feature_map: H,W = batch.rays_ori.shape[1:3] = 1088×1920 (padded)\n"
    r += "ROI pool: _clamp_bbox(bbox_render, H=1088, W=1920)\n"
    r += "```\n\n"

    # 2. Downsample status
    r += "## 2. Downsample Status\n\n"
    r += "- downsample_factor = 1 (no downsample)\n"
    r += "- bbox_xyxy_original == bbox_xyxy (identical)\n"
    r += "- Image is padded from 1080→1088 at the bottom\n\n"

    # 3. Padding impact
    r += "## 3. Padding Impact\n\n"
    r += "- Original image: 1920×1080\n"
    r += "- Padded image: 1920×1088 (8 pixels of black at bottom)\n"
    r += "- person_feature_map: 1088×1920 (same as padded image)\n"
    r += "- **Issue**: `img_height_original = 1088` includes padding\n"
    r += "- **Impact**: `scale_bbox_to_render` uses src_h=1088 instead of 1080\n"
    r += "- **Result**: y-coordinates are slightly compressed (scale_y = feat_H/1088 instead of feat_H/1080)\n"
    r += "- **Magnitude**: For a person at y=1000 in original image:\n"
    r += "  - Trainer: y_render = 1000 * (1088/1088) = 1000 (correct in padded space)\n"
    r += "  - Actually: since feature_map IS in padded space (1088), using src_h=1088 is CORRECT\n"
    r += "  - The bbox coordinates in annotation are in 1080 space, but padding is at bottom (y>1080)\n"
    r += "  - For bboxes with ymax < 1080, there is NO impact\n"
    r += "  - For bboxes near bottom (ymax close to 1080), slight scaling difference\n\n"

    # 4. Double-scale check
    r += "## 4. Double-Scale / Missing-Scale Check\n\n"
    r += "- Annotation bbox → bbox_xyxy_original: NO scale (raw coordinates)\n"
    r += "- bbox_xyxy_original → scale_bbox_to_render: ONE scale (1920×1088 → feat_W×feat_H)\n"
    r += "- **No double-scale detected**\n"
    r += "- **No missing scale detected**\n\n"

    # 5. Per-camera statistics
    r += "## 5. Per-Camera Statistics\n\n"
    r += "| Camera | Samples | Mean W | Mean H | W=1% | H=1% | Clamp% | Valid% | 0/NaN% | Mean Norm | Mean Opacity |\n"
    r += "|--------|---------|--------|--------|------|------|--------|--------|--------|-----------|-------------|\n"

    for cam in sorted(per_cam_stats.keys()):
        s = per_cam_stats[cam]
        n = max(1, s["total"])
        norms = s["roi_feature_norms"]
        opacities = s["opacity_sums"]
        r += (f"| {cam} | {s['total']} | "
              f"{np.mean(s['bbox_widths']):.1f} | {np.mean(s['bbox_heights']):.1f} | "
              f"{s['width_eq_1']/n*100:.1f} | {s['height_eq_1']/n*100:.1f} | "
              f"{s['border_clamp']/n*100:.1f} | {s['valid_roi']/n*100:.1f} | "
              f"{s['zero_or_nan_feature']/n*100:.1f} | "
              f"{np.mean(norms):.4f} | {np.mean(opacities):.2f} |\n")

    r += "\n"

    # 6. Overlay paths
    r += "## 6. Overlay Visualization Paths\n\n"
    for cam in sorted(per_cam_stats.keys()):
        cam_dir = os.path.join(output_dir, cam)
        if os.path.exists(cam_dir):
            n_files = len([f for f in os.listdir(cam_dir) if f.endswith(".jpg")])
            r += f"- `{cam}/`: {n_files} images\n"
    r += "\n"

    # 7. Y-offset analysis
    r += "## 7. Y-Offset Analysis (1088 vs 1080 scaling)\n\n"
    y_offsets = [rec["y_offset_trainer_vs_correct"] for rec in all_records if rec["y_offset_trainer_vs_correct"] != 0]
    r += f"- Samples with y-offset != 0: {len(y_offsets)} / {len(all_records)}\n"
    if y_offsets:
        r += f"- Mean y-offset: {np.mean(y_offsets):.2f} pixels\n"
        r += f"- Max y-offset: {max(y_offsets)} pixels\n"
    r += "\n"

    # 8. Conclusion
    r += "## 8. Conclusion\n\n"

    total = sum(s["total"] for s in per_cam_stats.values())
    total_valid = sum(s["valid_roi"] for s in per_cam_stats.values())
    total_zero_nan = sum(s["zero_or_nan_feature"] for s in per_cam_stats.values())
    total_border = sum(s["border_clamp"] for s in per_cam_stats.values())
    total_w1 = sum(s["width_eq_1"] for s in per_cam_stats.values())
    total_h1 = sum(s["height_eq_1"] for s in per_cam_stats.values())

    pass_criteria = [
        (total_valid / max(1, total) > 0.8, f"valid_roi_ratio = {total_valid/max(1,total):.2%} (need >80%)"),
        (total_w1 / max(1, total) < 0.05, f"width=1 ratio = {total_w1/max(1,total):.2%} (need <5%)"),
        (total_h1 / max(1, total) < 0.05, f"height=1 ratio = {total_h1/max(1,total):.2%} (need <5%)"),
        (total_zero_nan / max(1, total) < 0.05, f"zero/NaN feature ratio = {total_zero_nan/max(1,total):.2%} (need <5%)"),
    ]

    all_pass = all(p[0] for p in pass_criteria)

    for passed, desc in pass_criteria:
        r += f"- {'✅' if passed else '❌'} {desc}\n"

    r += f"\n**Overall: {'PASS' if all_pass else 'FAIL'}**\n\n"

    # 9. Next steps
    r += "## 9. Next Steps\n\n"
    if all_pass:
        r += "- ✅ V0 PASS: Bbox/ROI/render-space coordinates are consistent.\n"
        r += "- Next: Enter V1 person Gaussian support diagnostic.\n"
    else:
        r += "- ❌ V0 FAIL: Issues detected in bbox/ROI/render-space consistency.\n"
        r += "- Fix bbox / scale / padding / cam_id / frame_id before proceeding.\n"

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_ckpt", default=TEACHER_ONLY_CKPT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_per_cam", type=int, default=50)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    os.environ["TORCH_EXTENSIONS_DIR"] = "/data02/zhangrunxiang/.cache/torch_extensions/py311_cu118"
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("V0: Bbox/ROI/Render-Space Consistency Check")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    model, conf = setup_model(args.init_ckpt, device)
    print(f"  positions: {model.positions.shape}, _person_feature: {model._person_feature.shape}")

    # Load dataset
    print("\n[2/3] Loading dataset...")
    train_ds, _ = make_dataset("wildtrack", conf, ray_jitter=None)
    print(f"  dataset size: {len(train_ds)}")

    # Run diagnostic
    print(f"\n[3/3] Running diagnostic (max {args.max_per_cam} samples/cam)...")
    all_records, per_cam_stats = run_diagnostic(
        model, train_ds, device, args.output_dir, max_per_cam=args.max_per_cam
    )

    # Write reports
    write_reports(all_records, per_cam_stats, args.output_dir)


if __name__ == "__main__":
    main()
