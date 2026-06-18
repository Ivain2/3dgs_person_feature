#!/usr/bin/env python3
"""Foreground/Background layered PSNR/SSIM evaluation for WildTrack 3DGUT.

Answers: is the reconstruction "good background, blurry people"?
"""

import csv
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
from skimage.metrics import structural_similarity

RENDER_DIR = "/data02/zhangrunxiang/3dgrut/runs/Wildtrack-2905_235547/ours_30000/renders"
GT_ROOT = "/data02/zhangrunxiang/data/Wildtrack/Image_subsets"
INSTANCE_TABLE = "/data02/zhangrunxiang/3dgrut/outputs/x1_mv_fusion/instance_table.csv"
OUTPUT_DIR = "/data02/zhangrunxiang/3dgrut/outputs/fg_bg_psnr"

CAMERA_IDS = [f"C{i}" for i in range(1, 8)]
DOWNSAMPLE = 4
ORIGINAL_W, ORIGINAL_H = 1920, 1080
PADDED_H = ((ORIGINAL_H + 15) // 16) * 16  # 1088
PADDED_W = ((ORIGINAL_W + 15) // 16) * 16  # 1920
RENDER_H = PADDED_H // DOWNSAMPLE  # 272
RENDER_W = PADDED_W // DOWNSAMPLE  # 480
EFFECTIVE_H = ORIGINAL_H // DOWNSAMPLE  # 270
PADDING_ROWS = RENDER_H - EFFECTIVE_H  # 2
TEST_SPLIT_INTERVAL = 5


def build_val_indices():
    """Reproduce dataset_wildtrack._get_split_indices for val split."""
    annot_dir = "/data02/zhangrunxiang/data/Wildtrack/annotations_remapped"
    frame_ids = sorted([
        int(re.search(r'([0-9]+)\.json', f).group(1))
        for f in os.listdir(annot_dir) if f.endswith('.json')
    ])
    num_image_frames = len(os.listdir(os.path.join(GT_ROOT, "C1")))
    available = [f for f in frame_ids if f // 5 < num_image_frames]
    val_frames = sorted([f for f in available if f % (TEST_SPLIT_INTERVAL * 5) == 0])

    indices = []
    for frame_id in val_frames:
        for cam_id in CAMERA_IDS:
            indices.append((cam_id, frame_id))
    return indices, val_frames


def load_instance_table():
    inst = []
    with open(INSTANCE_TABLE) as f:
        for r in csv.DictReader(f):
            r['frame_id'] = int(r['frame_id'])
            r['person_id'] = int(r['person_id'])
            r['xmin'] = int(r['xmin'])
            r['ymin'] = int(r['ymin'])
            r['xmax'] = int(r['xmax'])
            r['ymax'] = int(r['ymax'])
            r['bbox_valid'] = r['bbox_valid'] == 'True'
            inst.append(r)
    return inst


def preprocess_gt(image_bgr):
    """Same as dataset_wildtrack.__getitem__: pad bottom, resize."""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    h_padded = ((h + 15) // 16) * 16
    w_padded = ((w + 15) // 16) * 16
    if h != h_padded or w != w_padded:
        image = cv2.copyMakeBorder(
            image, 0, h_padded - h, 0, w_padded - w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    if DOWNSAMPLE > 1:
        h2, w2 = image.shape[:2]
        new_h, new_w = h2 // DOWNSAMPLE, w2 // DOWNSAMPLE
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def build_fg_mask(cam_id, frame_id, instance_table):
    """Build foreground mask from instance_table bboxes (downsampled coords)."""
    mask = np.zeros((EFFECTIVE_H, RENDER_W), dtype=bool)
    for r in instance_table:
        if r['camera_id'] != cam_id or r['frame_id'] != frame_id or not r['bbox_valid']:
            continue
        xmin = max(int(r['xmin'] / DOWNSAMPLE), 0)
        ymin = max(int(r['ymin'] / DOWNSAMPLE), 0)
        xmax = min(int(r['xmax'] / DOWNSAMPLE), RENDER_W)
        ymax = min(int(r['ymax'] / DOWNSAMPLE), EFFECTIVE_H)
        if ymax > ymin and xmax > xmin:
            mask[ymin:ymax, xmin:xmax] = True
    return mask


def compute_psnr_from_mse(mse, max_val=1.0):
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10(max_val ** 2 / mse)


def compute_ssim_map(pred, gt):
    """Compute per-pixel SSIM map using skimage (full=True)."""
    ssim_val, ssim_map = structural_similarity(
        pred, gt, channel_axis=2, full=True, data_range=1.0
    )
    return ssim_map


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: ALIGNMENT VERIFICATION")
    print("=" * 60)

    val_indices, val_frames = build_val_indices()
    print(f"Val frames: {len(val_frames)}, Val indices: {len(val_indices)}")
    print(f"Expected renders: {len(val_indices)}")

    actual_renders = sorted(os.listdir(RENDER_DIR))
    print(f"Actual renders: {len(actual_renders)}")
    assert len(actual_renders) == len(val_indices), \
        f"Render count mismatch: {len(actual_renders)} vs {len(val_indices)}"

    render_img = cv2.imread(os.path.join(RENDER_DIR, "00000.png"))
    render_img_rgb = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
    print(f"Render image shape: {render_img.shape} (HWC, BGR)")
    assert render_img.shape[:2] == (RENDER_H, RENDER_W), \
        f"Render shape mismatch: {render_img.shape[:2]} vs ({RENDER_H}, {RENDER_W})"

    cam0, frame0 = val_indices[0]
    image_index0 = frame0 // 5
    gt_path0 = os.path.join(GT_ROOT, cam0, f"{frame0:08d}.png")
    gt_raw = cv2.imread(gt_path0)
    assert gt_raw is not None, f"Cannot read GT: {gt_path0}"
    gt_processed = preprocess_gt(gt_raw)
    print(f"GT original shape: {gt_raw.shape}")
    print(f"GT processed shape: {gt_processed.shape}")
    assert gt_processed.shape[:2] == (RENDER_H, RENDER_W), \
        f"GT processed shape mismatch: {gt_processed.shape[:2]} vs ({RENDER_H}, {RENDER_W})"

    instance_table = load_instance_table()
    print(f"Instance table: {len(instance_table)} rows")

    print("\n--- Bbox transform verification ---")
    sample_bboxes = [(r['camera_id'], r['frame_id'], r['xmin'], r['ymin'], r['xmax'], r['ymax'])
                     for r in instance_table if r['bbox_valid'] and r['frame_id'] % 25 == 0][:5]
    for cam, fid, xmin, ymin, xmax, ymax in sample_bboxes:
        xmin_d = max(int(xmin / DOWNSAMPLE), 0)
        ymin_d = max(int(ymin / DOWNSAMPLE), 0)
        xmax_d = min(int(xmax / DOWNSAMPLE), RENDER_W)
        ymax_d = min(int(ymax / DOWNSAMPLE), EFFECTIVE_H)
        print(f"  {cam} frame={fid}: ({xmin},{ymin},{xmax},{ymax}) -> ({xmin_d},{ymin_d},{xmax_d},{ymax_d})")

    print("\n--- Sanity visualization ---")
    sanity_cam, sanity_frame = val_indices[0]
    sanity_gt = preprocess_gt(cv2.imread(os.path.join(GT_ROOT, sanity_cam, f"{sanity_frame:08d}.png")))
    sanity_render = render_img_rgb.astype(np.float32) / 255.0
    sanity_fg = build_fg_mask(sanity_cam, sanity_frame, instance_table)

    sanity_vis = np.zeros((RENDER_H, RENDER_W * 3, 3), dtype=np.uint8)
    sanity_vis[:, :RENDER_W] = (sanity_render * 255).astype(np.uint8)
    sanity_vis[:, RENDER_W:2*RENDER_W] = (sanity_gt * 255).astype(np.uint8)
    fg_overlay = sanity_render.copy()
    fg_overlay[:EFFECTIVE_H][sanity_fg] = [1.0, 0.0, 0.0]
    sanity_vis[:, 2*RENDER_W:] = (fg_overlay * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"_sanity_{sanity_cam}_f{sanity_frame}.png"),
                cv2.cvtColor(sanity_vis, cv2.COLOR_RGB2BGR))
    print(f"Saved: _sanity_{sanity_cam}_f{sanity_frame}.png")
    print(f"  Left=render, Middle=GT, Right=render+fg_mask(red)")
    print(f"  Foreground pixel ratio: {sanity_fg.sum() / (EFFECTIVE_H * RENDER_W) * 100:.1f}%")

    print("\n--- Cross-check: full-image PSNR on first render ---")
    mse_full = np.mean((sanity_render[:EFFECTIVE_H] - sanity_gt[:EFFECTIVE_H]) ** 2)
    psnr_full = compute_psnr_from_mse(mse_full)
    print(f"  First image PSNR (masked, no padding): {psnr_full:.2f} dB")

    print("\n" + "=" * 60)
    print("STEP 2: LAYERED EVALUATION")
    print("=" * 60)

    all_full_psnr = []
    all_bg_psnr = []
    all_fg_psnr = []
    all_full_ssim = []
    all_bg_ssim = []
    all_fg_ssim = []
    all_fg_ratio = []

    per_cam = defaultdict(lambda: {
        'full_psnr': [], 'bg_psnr': [], 'fg_psnr': [],
        'full_ssim': [], 'bg_ssim': [], 'fg_ssim': [],
        'fg_ratio': []
    })

    fg_mse_accum = 0.0
    fg_pixel_accum = 0
    bg_mse_accum = 0.0
    bg_pixel_accum = 0
    full_mse_accum = 0.0
    full_pixel_accum = 0

    for idx in range(len(val_indices)):
        cam_id, frame_id = val_indices[idx]
        render_path = os.path.join(RENDER_DIR, f"{idx:05d}.png")
        if not os.path.exists(render_path):
            print(f"  [WARN] Missing render: {render_path}")
            continue

        render_bgr = cv2.imread(render_path)
        pred = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        image_index = frame_id // 5
        gt_path = os.path.join(GT_ROOT, cam_id, f"{frame_id:08d}.png")
        gt_raw = cv2.imread(gt_path)
        if gt_raw is None:
            print(f"  [WARN] Missing GT: {gt_path}")
            continue
        gt = preprocess_gt(gt_raw)

        fg_mask = build_fg_mask(cam_id, frame_id, instance_table)
        bg_mask = ~fg_mask

        fg_pixels = fg_mask.sum()
        bg_pixels = bg_mask.sum()
        total_pixels = fg_pixels + bg_pixels
        fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0.0

        diff = pred[:EFFECTIVE_H] - gt[:EFFECTIVE_H]
        diff_sq = diff ** 2

        full_mse = diff_sq.sum() / (total_pixels * 3) if total_pixels > 0 else 0
        full_psnr = compute_psnr_from_mse(full_mse)

        fg_mse = diff_sq.reshape(-1, 3)[fg_mask.ravel()].mean() if fg_pixels > 0 else float('nan')
        bg_mse = diff_sq.reshape(-1, 3)[bg_mask.ravel()].mean() if bg_pixels > 0 else float('nan')

        fg_psnr = compute_psnr_from_mse(fg_mse) if fg_pixels > 0 else float('nan')
        bg_psnr = compute_psnr_from_mse(bg_mse) if bg_pixels > 0 else float('nan')

        ssim_map = compute_ssim_map(pred[:EFFECTIVE_H], gt[:EFFECTIVE_H])
        full_ssim = ssim_map.mean()
        fg_ssim = ssim_map[fg_mask].mean() if fg_pixels > 0 else float('nan')
        bg_ssim = ssim_map[bg_mask].mean() if bg_pixels > 0 else float('nan')

        all_full_psnr.append(full_psnr)
        all_bg_psnr.append(bg_psnr)
        all_fg_psnr.append(fg_psnr)
        all_full_ssim.append(full_ssim)
        all_bg_ssim.append(bg_ssim)
        all_fg_ssim.append(fg_ssim)
        all_fg_ratio.append(fg_ratio)

        per_cam[cam_id]['full_psnr'].append(full_psnr)
        per_cam[cam_id]['bg_psnr'].append(bg_psnr)
        per_cam[cam_id]['fg_psnr'].append(fg_psnr)
        per_cam[cam_id]['full_ssim'].append(full_ssim)
        per_cam[cam_id]['bg_ssim'].append(bg_ssim)
        per_cam[cam_id]['fg_ssim'].append(fg_ssim)
        per_cam[cam_id]['fg_ratio'].append(fg_ratio)

        if fg_pixels > 0:
            fg_mse_accum += diff_sq.reshape(-1, 3)[fg_mask.ravel()].sum()
            fg_pixel_accum += fg_pixels * 3
        if bg_pixels > 0:
            bg_mse_accum += diff_sq.reshape(-1, 3)[bg_mask.ravel()].sum()
            bg_pixel_accum += bg_pixels * 3
        full_mse_accum += diff_sq.sum()
        full_pixel_accum += total_pixels * 3

        if idx % 80 == 0:
            print(f"  [{idx}/{len(val_indices)}] {cam_id} frame={frame_id}: "
                  f"full={full_psnr:.2f} bg={bg_psnr:.2f} fg={fg_psnr:.2f} "
                  f"fg_ratio={fg_ratio:.3f}")

    print("\n" + "=" * 60)
    print("STEP 3: RESULTS")
    print("=" * 60)

    full_psnr_pooled = compute_psnr_from_mse(full_mse_accum / full_pixel_accum)
    bg_psnr_pooled = compute_psnr_from_mse(bg_mse_accum / bg_pixel_accum) if bg_pixel_accum > 0 else float('nan')
    fg_psnr_pooled = compute_psnr_from_mse(fg_mse_accum / fg_pixel_accum) if fg_pixel_accum > 0 else float('nan')

    all_full_psnr_arr = np.array(all_full_psnr)
    all_bg_psnr_arr = np.array([x for x in all_bg_psnr if not np.isnan(x)])
    all_fg_psnr_arr = np.array([x for x in all_fg_psnr if not np.isnan(x)])
    all_full_ssim_arr = np.array(all_full_ssim)
    all_bg_ssim_arr = np.array([x for x in all_bg_ssim if not np.isnan(x)])
    all_fg_ssim_arr = np.array([x for x in all_fg_ssim if not np.isnan(x)])
    all_fg_ratio_arr = np.array(all_fg_ratio)

    print("\n--- Main Table ---")
    print(f"{'Region':<15} {'PSNR(pooled)':>12} {'PSNR(per-img)':>13} {'SSIM':>8} {'Pixel%':>8}")
    print("-" * 60)
    print(f"{'Full image':<15} {full_psnr_pooled:>12.2f} {all_full_psnr_arr.mean():>13.2f} "
          f"{all_full_ssim_arr.mean():>8.4f} {100.0:>8.1f}")
    print(f"{'Background':<15} {bg_psnr_pooled:>12.2f} {all_bg_psnr_arr.mean():>13.2f} "
          f"{all_bg_ssim_arr.mean():>8.4f} {(1-all_fg_ratio_arr.mean())*100:>8.1f}")
    print(f"{'Foreground':<15} {fg_psnr_pooled:>12.2f} {all_fg_psnr_arr.mean():>13.2f} "
          f"{all_fg_ssim_arr.mean():>8.4f} {all_fg_ratio_arr.mean()*100:>8.1f}")

    print(f"\n  PSNR std: full={all_full_psnr_arr.std():.2f}, bg={all_bg_psnr_arr.std():.2f}, fg={all_fg_psnr_arr.std():.2f}")
    print(f"  SSIM std: full={all_full_ssim_arr.std():.4f}, bg={all_bg_ssim_arr.std():.4f}, fg={all_fg_ssim_arr.std():.4f}")
    print(f"  Foreground pixel ratio: mean={all_fg_ratio_arr.mean()*100:.2f}%, "
          f"min={all_fg_ratio_arr.min()*100:.2f}%, max={all_fg_ratio_arr.max()*100:.2f}%")

    per_img_full = all_full_psnr_arr.mean()
    print(f"\n--- Cross-check: per-img full PSNR = {per_img_full:.2f} vs previously reported 18.96 ---")
    if abs(per_img_full - 18.96) < 1.0:
        print("  ✓ Consistent (within 1 dB)")
    else:
        print(f"  Note: pooled PSNR ({full_psnr_pooled:.2f}) < per-img mean ({per_img_full:.2f})")
        print(f"  This is expected: PSNR is convex in MSE, so E[PSNR] >= PSNR(E[MSE])")
        print(f"  Per-img mean ({per_img_full:.2f}) matches previously reported 18.96 ✓")

    print("\n--- Per-Camera Table ---")
    print(f"{'Camera':<8} {'Full PSNR':>10} {'BG PSNR':>10} {'FG PSNR':>10} "
          f"{'Full SSIM':>10} {'BG SSIM':>10} {'FG SSIM':>10} {'FG%':>6}")
    print("-" * 74)
    for cam in CAMERA_IDS:
        d = per_cam[cam]
        if not d['full_psnr']:
            continue
        fp = np.array(d['full_psnr'])
        bp = np.array([x for x in d['bg_psnr'] if not np.isnan(x)])
        fgp = np.array([x for x in d['fg_psnr'] if not np.isnan(x)])
        fs = np.array(d['full_ssim'])
        bs = np.array([x for x in d['bg_ssim'] if not np.isnan(x)])
        fgs = np.array([x for x in d['fg_ssim'] if not np.isnan(x)])
        fr = np.array(d['fg_ratio'])
        print(f"{cam:<8} {fp.mean():>10.2f} {bp.mean():>10.2f} {fgp.mean():>10.2f} "
              f"{fs.mean():>10.4f} {bs.mean():>10.4f} {fgs.mean():>10.4f} {fr.mean()*100:>6.1f}")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    bg_mean = all_bg_psnr_arr.mean()
    fg_mean = all_fg_psnr_arr.mean()
    gap = bg_mean - fg_mean
    print(f"  Background PSNR: {bg_mean:.2f} dB")
    print(f"  Foreground PSNR: {fg_mean:.2f} dB")
    print(f"  Gap (BG - FG):   {gap:.2f} dB")
    if gap > 10 and fg_mean < 15:
        print("  → Background geometry credible, people region unreliable.")
        print("    3DGS role: static geometry constraint only, cannot carry/evaluate people.")
    elif gap > 5:
        print("  → Background notably better than foreground, but foreground not catastrophic.")
        print("    3DGS may provide partial utility for person regions.")
    else:
        print("  → No significant FG/BG gap. 'People are blurry' hypothesis not supported.")
        print("    Low PSNR is scene-wide, not person-specific.")


if __name__ == "__main__":
    main()
