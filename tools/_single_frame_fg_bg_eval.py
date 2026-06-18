#!/usr/bin/env python3
"""Single-frame fg/bg PSNR evaluation.

For each single-frame model, evaluate fg/bg PSNR/SSIM on its 7 training views.
Uses identical metric computation as _fg_bg_psnr_eval.py for fair comparison.
"""

import csv
import glob
import os
from collections import defaultdict

import cv2
import numpy as np
from skimage.metrics import structural_similarity

GT_ROOT = "/data02/zhangrunxiang/data/Wildtrack/Image_subsets"
INSTANCE_TABLE = "/data02/zhangrunxiang/3dgrut/outputs/x1_mv_fusion/instance_table.csv"
RUNS_ROOT = "/data02/zhangrunxiang/3dgrut/runs"
OUTPUT_DIR = "/data02/zhangrunxiang/3dgrut/outputs/fg_bg_psnr"

CAMERA_IDS = [f"C{i}" for i in range(1, 8)]
DOWNSAMPLE = 4
ORIGINAL_W, ORIGINAL_H = 1920, 1080
PADDED_H = ((ORIGINAL_H + 15) // 16) * 16
PADDED_W = ((ORIGINAL_W + 15) // 16) * 16
RENDER_H = PADDED_H // DOWNSAMPLE
RENDER_W = PADDED_W // DOWNSAMPLE
EFFECTIVE_H = ORIGINAL_H // DOWNSAMPLE

SINGLE_FRAMES = {
    100: {"label": "sparse", "persons": 5},
    275: {"label": "medium", "persons": 7},
    290: {"label": "medium", "persons": 9},
    15:  {"label": "crowded", "persons": 34},
}


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


def find_render_dir(frame_id):
    run_name = f"single_frame_f{frame_id}"
    run_dir = os.path.join(RUNS_ROOT, run_name)
    subdirs = glob.glob(os.path.join(run_dir, "Wildtrack-*", "ours_15000", "renders"))
    if subdirs:
        return subdirs[0]
    subdirs = glob.glob(os.path.join(run_dir, "Wildtrack-*", "ours_7000", "renders"))
    if subdirs:
        return subdirs[0]
    return None


def evaluate_single_frame(frame_id, instance_table):
    render_dir = find_render_dir(frame_id)
    if render_dir is None:
        print(f"  [SKIP] No renders found for frame_id={frame_id}")
        return None

    render_files = sorted([f for f in os.listdir(render_dir) if f.endswith('.png')])
    if len(render_files) != 7:
        print(f"  [WARN] Expected 7 renders, got {len(render_files)}")

    results = []
    fg_mse_accum = 0.0
    fg_pixel_accum = 0
    bg_mse_accum = 0.0
    bg_pixel_accum = 0
    full_mse_accum = 0.0
    full_pixel_accum = 0

    for idx, render_file in enumerate(render_files):
        cam_id = CAMERA_IDS[idx]

        render_bgr = cv2.imread(os.path.join(render_dir, render_file))
        pred = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

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

        ssim_map = structural_similarity(
            pred[:EFFECTIVE_H], gt[:EFFECTIVE_H], channel_axis=2, full=True, data_range=1.0
        )[1]
        full_ssim = ssim_map.mean()
        fg_ssim = ssim_map[fg_mask].mean() if fg_pixels > 0 else float('nan')
        bg_ssim = ssim_map[bg_mask].mean() if bg_pixels > 0 else float('nan')

        results.append({
            'cam_id': cam_id, 'full_psnr': full_psnr, 'bg_psnr': bg_psnr,
            'fg_psnr': fg_psnr, 'full_ssim': full_ssim, 'bg_ssim': bg_ssim,
            'fg_ssim': fg_ssim, 'fg_ratio': fg_ratio
        })

        if fg_pixels > 0:
            fg_mse_accum += diff_sq.reshape(-1, 3)[fg_mask.ravel()].sum()
            fg_pixel_accum += fg_pixels * 3
        if bg_pixels > 0:
            bg_mse_accum += diff_sq.reshape(-1, 3)[bg_mask.ravel()].sum()
            bg_pixel_accum += bg_pixels * 3
        full_mse_accum += diff_sq.sum()
        full_pixel_accum += total_pixels * 3

    if not results:
        return None

    full_psnr_pooled = compute_psnr_from_mse(full_mse_accum / full_pixel_accum)
    bg_psnr_pooled = compute_psnr_from_mse(bg_mse_accum / bg_pixel_accum) if bg_pixel_accum > 0 else float('nan')
    fg_psnr_pooled = compute_psnr_from_mse(fg_mse_accum / fg_pixel_accum) if fg_pixel_accum > 0 else float('nan')

    full_psnr_perimg = np.mean([r['full_psnr'] for r in results])
    bg_psnr_perimg = np.mean([r['bg_psnr'] for r in results if not np.isnan(r['bg_psnr'])])
    fg_psnr_perimg = np.mean([r['fg_psnr'] for r in results if not np.isnan(r['fg_psnr'])])

    full_ssim_mean = np.mean([r['full_ssim'] for r in results])
    bg_ssim_mean = np.mean([r['bg_ssim'] for r in results if not np.isnan(r['bg_ssim'])])
    fg_ssim_mean = np.mean([r['fg_ssim'] for r in results if not np.isnan(r['fg_ssim'])])
    fg_ratio_mean = np.mean([r['fg_ratio'] for r in results])

    return {
        'frame_id': frame_id,
        'full_psnr_pooled': full_psnr_pooled,
        'bg_psnr_pooled': bg_psnr_pooled,
        'fg_psnr_pooled': fg_psnr_pooled,
        'full_psnr_perimg': full_psnr_perimg,
        'bg_psnr_perimg': bg_psnr_perimg,
        'fg_psnr_perimg': fg_psnr_perimg,
        'full_ssim': full_ssim_mean,
        'bg_ssim': bg_ssim_mean,
        'fg_ssim': fg_ssim_mean,
        'fg_ratio': fg_ratio_mean,
        'per_cam': results,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    instance_table = load_instance_table()

    print("=" * 70)
    print("SINGLE-FRAME FG/BG PSNR EVALUATION")
    print("=" * 70)

    frame_results = {}
    for frame_id in sorted(SINGLE_FRAMES.keys()):
        info = SINGLE_FRAMES[frame_id]
        print(f"\n--- Frame {frame_id} ({info['label']}, {info['persons']} persons) ---")
        result = evaluate_single_frame(frame_id, instance_table)
        if result is None:
            continue
        frame_results[frame_id] = result
        print(f"  Full:  pooled={result['full_psnr_pooled']:.2f}, per-img={result['full_psnr_perimg']:.2f}, SSIM={result['full_ssim']:.4f}")
        print(f"  BG:    pooled={result['bg_psnr_pooled']:.2f}, per-img={result['bg_psnr_perimg']:.2f}, SSIM={result['bg_ssim']:.4f}")
        print(f"  FG:    pooled={result['fg_psnr_pooled']:.2f}, per-img={result['fg_psnr_perimg']:.2f}, SSIM={result['fg_ssim']:.4f}")
        print(f"  FG%:   {result['fg_ratio']*100:.2f}%")

        print(f"\n  Per-camera breakdown:")
        for r in result['per_cam']:
            fg_str = f"{r['fg_psnr']:.2f}" if not np.isnan(r['fg_psnr']) else "N/A"
            fg_ssim_str = f"{r['fg_ssim']:.4f}" if not np.isnan(r['fg_ssim']) else "N/A"
            print(f"    {r['cam_id']}: full={r['full_psnr']:.2f} bg={r['bg_psnr']:.2f} "
                  f"fg={fg_str} fg_ssim={fg_ssim_str} fg%={r['fg_ratio']*100:.1f}")

    if not frame_results:
        print("No results!")
        return

    print("\n" + "=" * 70)
    print("MAIN TABLE: Single-Frame Reconstruction")
    print("=" * 70)
    print(f"{'frame_id':<10} {'persons':>7} {'density':>8} | "
          f"{'Full PSNR':>10} {'BG PSNR':>10} {'FG PSNR':>10} | "
          f"{'FG SSIM':>8} {'FG%':>6} {'#Gauss':>8}")
    print("-" * 90)

    all_fg_psnr = []
    all_bg_psnr = []
    all_fg_ssim = []
    all_full_psnr = []

    for frame_id in sorted(frame_results.keys()):
        r = frame_results[frame_id]
        info = SINGLE_FRAMES[frame_id]
        all_fg_psnr.append(r['fg_psnr_perimg'])
        all_bg_psnr.append(r['bg_psnr_perimg'])
        all_fg_ssim.append(r['fg_ssim'])
        all_full_psnr.append(r['full_psnr_perimg'])

        run_dir = os.path.join(RUNS_ROOT, f"single_frame_f{frame_id}")
        ckpt_files = glob.glob(os.path.join(run_dir, "Wildtrack-*/ckpt_last.pt"))
        gauss_count = "?"
        if ckpt_files:
            import torch
            try:
                ckpt = torch.load(ckpt_files[0], map_location='cpu')
                if 'num_gaussians' in ckpt:
                    gauss_count = f"{ckpt['num_gaussians']:,}"
            except:
                pass

        print(f"{frame_id:<10} {info['persons']:>7} {info['label']:>8} | "
              f"{r['full_psnr_perimg']:>10.2f} {r['bg_psnr_perimg']:>10.2f} {r['fg_psnr_perimg']:>10.2f} | "
              f"{r['fg_ssim']:>8.4f} {r['fg_ratio']*100:>6.1f} {gauss_count:>8}")

    print("-" * 90)
    fg_mean = np.mean(all_fg_psnr)
    bg_mean = np.mean(all_bg_psnr)
    full_mean = np.mean(all_full_psnr)
    fg_ssim_mean = np.mean(all_fg_ssim)
    print(f"{'MEAN':<10} {'':>7} {'':>8} | "
          f"{full_mean:>10.2f} {bg_mean:>10.2f} {fg_mean:>10.2f} | "
          f"{fg_ssim_mean:>8.4f}")

    print("\n" + "=" * 70)
    print("CORE COMPARISON: Multi-frame vs Single-frame")
    print("=" * 70)
    print(f"{'Setting':<25} {'FG PSNR':>10} {'FG SSIM':>10} {'BG PSNR':>10} {'Full PSNR':>10}")
    print("-" * 65)
    print(f"{'Multi-frame (baseline)':<25} {'12.77':>10} {'0.31':>10} {'19.11':>10} {'18.96':>10}")
    print(f"{'Single-frame (mean)':<25} {fg_mean:>10.2f} {fg_ssim_mean:>10.4f} {bg_mean:>10.2f} {full_mean:>10.2f}")

    fg_delta = fg_mean - 12.77
    bg_delta = bg_mean - 19.11
    print(f"\n  FG PSNR improvement: +{fg_delta:.2f} dB")
    print(f"  BG PSNR improvement: +{bg_delta:.2f} dB")

    print("\n" + "=" * 70)
    print("DENSITY ANALYSIS")
    print("=" * 70)
    for frame_id in sorted(frame_results.keys()):
        r = frame_results[frame_id]
        info = SINGLE_FRAMES[frame_id]
        print(f"  Frame {frame_id} ({info['persons']} persons, {info['label']}): "
              f"FG PSNR={r['fg_psnr_perimg']:.2f}, BG PSNR={r['bg_psnr_perimg']:.2f}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if fg_mean > 20:
        print(f"  Single-frame FG PSNR = {fg_mean:.2f} >> multi-frame 12.77 dB")
        print(f"  → Cross-frame contradiction (cause a) is the DOMINANT cause of blurry people.")
        print(f"  → With single-frame training, 7 views CAN reconstruct people well.")
        print(f"  → View sparsity (cause b) is NOT the bottleneck at 7 views.")
    elif fg_mean > 15:
        print(f"  Single-frame FG PSNR = {fg_mean:.2f} > multi-frame 12.77 dB")
        print(f"  → Both causes contribute: cross-frame contradiction (a) + view sparsity (b).")
        print(f"  → Eliminating (a) helps significantly but doesn't fully solve the problem.")
    else:
        print(f"  Single-frame FG PSNR = {fg_mean:.2f} ≈ multi-frame 12.77 dB")
        print(f"  → View sparsity (cause b) is the DOMINANT cause.")
        print(f"  → Even without cross-frame contradiction, 7 views cannot reconstruct people.")


if __name__ == "__main__":
    main()
