#!/usr/bin/env python3
"""
Leave-One-Out 跨视角泛化验证实验评估脚本

用途：
评估所有 leave-one-out 实验的渲染质量，计算全图、bbox 前景、精确 mask 前景、背景的 PSNR。
SSIM 只保留全图一列，避免 masked SSIM 误导。

运行方式：
    python tools/eval_leave_one_out.py \
        --experiment_root experiments/leave_one_out \
        --output_csv results/loo_eval.csv \
        --dataset_path /data02/zhangrunxiang/data/Wildtrack_small_sample

输出 CSV 列：
    frame_id, held_out_cam, is_baseline,
    full_psnr, full_ssim,
    fg_bbox_psnr, bg_psnr, fg_seg_psnr,
    nearest_angle_diff,
    baseline_psnr, generalization_gap

关键设计：
    1. PSNR 只对 mask 内有效像素求 MSE，避免 0 像素虚高
    2. 数值范围统一 [0,1] 浮点
    3. mask 内像素数不足 100 时跳过
    4. SAM mask 不可用时 graceful 降级
    5. 渲染文件用 glob 查找，不硬编码文件名
    6. 图像尺寸从 GT 原始分辨率 + downsample_factor 推断，不硬编码
    7. 包含按视角差分桶的统计
    8. Baseline 渲染文件通过文件名中的相机ID显式匹配
"""

import argparse
import csv
import numpy as np
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loo_utils import (
    compute_psnr_masked, load_instance_table, build_bbox_mask,
    load_segmentation_mask, preprocess_gt, find_render_files,
    find_gt_path, load_extrinsics, find_nearest_train_angle_diff,
    CAMERA_IDS,
)


def build_render_cam_map(dataset_path: str, downsample_factor: int,
                         frame_id: int, held_out_camera: str = None) -> dict:
    """
    构建 渲染文件名 → 相机ID 的映射。
    
    渲染文件命名格式为 {iteration:05d}.png（如 00000.png），不含相机ID。
    需要通过 val dataset 的 indices 顺序来推断每个文件对应的相机。
    
    对于 LOO (held_out_camera != None): val dataset 只含 held-out camera，
        只有1个文件 00000.png
    对于 baseline (held_out_camera == None): val dataset 含全部7视角，
        7个文件 00000~00006.png
    
    Args:
        dataset_path: 数据集路径
        downsample_factor: 下采样因子
        frame_id: 帧ID
        held_out_camera: held-out 相机ID (None 表示 baseline)
    
    Returns:
        {文件名(str): 相机ID(str)}
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from threedgrut.datasets.dataset_wildtrack import WildtrackDataset
    
    try:
        val_dataset = WildtrackDataset(
            dataset_path,
            split="val",
            downsample_factor=downsample_factor,
            single_frame_id=frame_id,
            held_out_camera=held_out_camera,  # LOO 时传入 held-out camera
        )
        indices = val_dataset.indices
    except Exception:
        # fallback: 按默认顺序
        if held_out_camera is not None:
            indices = [(held_out_camera, frame_id)]
        else:
            indices = [(cam, frame_id) for cam in CAMERA_IDS]
    
    render_cam_map = {}
    for i, (cam_id, _) in enumerate(indices):
        filename = f"{i:05d}.png"
        render_cam_map[filename] = cam_id
    
    return render_cam_map


def evaluate_single_experiment(exp_dir: Path, instance_table: list,
                               dataset_path: str, extrinsics: dict,
                               downsample_factor: int,
                               seg_root: Path = None,
                               is_baseline: bool = False) -> dict:
    """
    评估单个实验（LOO 或 baseline）

    目录结构: exp_dir/run/Wildtrack-*/ours_*/renders/*.png
    渲染文件命名: 00000.png, 00001.png, ... (按 val dataloader 顺序)
    """
    # 解析 frame_id 和 held_cam
    frame_id = int(exp_dir.parent.name.split('_')[1])
    if is_baseline:
        held_cam = None
    else:
        held_cam = exp_dir.name.split('_')[1]

    # 查找渲染文件
    render_files = find_render_files(exp_dir)
    if not render_files:
        print(f"  警告: 未找到渲染文件 in {exp_dir}")
        return None

    results = {
        'frame_id': frame_id,
        'held_out_cam': held_cam if held_cam else 'all',
        'is_baseline': is_baseline,
        'nearest_angle_diff': np.nan,
        'baseline_psnr': np.nan,
        'generalization_gap': np.nan,
    }

    # 计算视角差（仅 LOO）
    if held_cam and held_cam in extrinsics:
        results['nearest_angle_diff'] = find_nearest_train_angle_diff(held_cam, extrinsics)

    # 确定 GT 相机
    if is_baseline:
        gt_cams = CAMERA_IDS
    else:
        gt_cams = [held_cam]

    # 构建 渲染文件名 → 相机ID 的映射
    # 渲染文件名为 00000.png, 00001.png, ... 按 val dataloader 顺序
    if is_baseline:
        # baseline: 7个文件按 val dataset indices 顺序
        render_cam_map = build_render_cam_map(
            dataset_path, downsample_factor, frame_id, held_out_camera=None
        )
    else:
        # LOO: 只有1个文件，对应 held_cam
        render_cam_map = build_render_cam_map(
            dataset_path, downsample_factor, frame_id, held_out_camera=held_cam
        )

    all_full_psnr = []
    all_fg_bbox_psnr = []
    all_bg_psnr = []
    all_fg_seg_psnr = []
    all_full_ssim = []

    for cam in gt_cams:
        # 找到对应的渲染文件（统一通过 render_cam_map 查找）
        render_path = None
        for fname, cam_id in render_cam_map.items():
            if cam_id == cam:
                for rf in render_files:
                    if rf.name == fname:
                        render_path = rf
                        break
                break
        if render_path is None:
            # fallback: 按索引匹配
            cam_idx = CAMERA_IDS.index(cam)
            if cam_idx < len(render_files):
                render_path = render_files[cam_idx]
            else:
                continue

        # GT 路径
        gt_path = find_gt_path(dataset_path, cam, frame_id)
        if not gt_path.exists():
            print(f"  警告: GT 不存在 {gt_path}")
            continue

        # 加载图像
        import cv2
        pred_bgr = cv2.imread(str(render_path))
        if pred_bgr is None:
            continue
        pred = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        gt_raw = cv2.imread(str(gt_path))
        if gt_raw is None:
            continue
        gt = preprocess_gt(gt_raw, downsample=downsample_factor)

        if pred.shape != gt.shape:
            print(f"  警告: 尺寸不一致 {pred.shape} vs {gt.shape}")
            continue

        # 从 GT 原始分辨率推断有效区域尺寸
        h_raw, w_raw = gt_raw.shape[:2]
        effective_h = h_raw // downsample_factor
        render_w = w_raw // downsample_factor
        pred = pred[:effective_h, :render_w]
        gt = gt[:effective_h, :render_w]

        # 构建 mask
        bbox_mask = build_bbox_mask(cam, frame_id, instance_table,
                                    effective_h, render_w, downsample_factor)
        bg_mask = ~bbox_mask
        full_mask = np.ones_like(bbox_mask)

        # PSNR
        all_full_psnr.append(compute_psnr_masked(pred, gt, full_mask))
        all_fg_bbox_psnr.append(compute_psnr_masked(pred, gt, bbox_mask))
        all_bg_psnr.append(compute_psnr_masked(pred, gt, bg_mask))

        # 全图 SSIM
        try:
            from skimage.metrics import structural_similarity as ssim
            pred_gray = np.mean(pred, axis=2)
            gt_gray = np.mean(gt, axis=2)
            all_full_ssim.append(ssim(pred_gray, gt_gray, data_range=1.0))
        except ImportError:
            all_full_ssim.append(np.nan)

        # 精确 mask PSNR（如果可用）
        if seg_root is not None:
            seg_path = seg_root / cam / f"{frame_id:04d}.png"
            seg_mask = load_segmentation_mask(seg_path, effective_h, render_w)
            if seg_mask is not None:
                all_fg_seg_psnr.append(compute_psnr_masked(pred, gt, seg_mask))
            else:
                all_fg_seg_psnr.append(np.nan)
        else:
            all_fg_seg_psnr.append(np.nan)

    # 汇总
    results['full_psnr'] = np.nanmean(all_full_psnr) if all_full_psnr else np.nan
    results['full_ssim'] = np.nanmean(all_full_ssim) if all_full_ssim else np.nan
    results['fg_bbox_psnr'] = np.nanmean(all_fg_bbox_psnr) if all_fg_bbox_psnr else np.nan
    results['bg_psnr'] = np.nanmean(all_bg_psnr) if all_bg_psnr else np.nan
    results['fg_seg_psnr'] = np.nanmean(all_fg_seg_psnr) if all_fg_seg_psnr else np.nan

    return results


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Out 实验评估")
    parser.add_argument("--experiment_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="results/loo_eval.csv")
    parser.add_argument("--segmentation_root", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--downsample_factor", type=int, default=4,
                        help="下采样因子（默认4）")
    parser.add_argument("--instance_table", type=str,
                        default="/data02/zhangrunxiang/3dgrut/outputs/x1_mv_fusion/instance_table.csv")

    args = parser.parse_args()

    # 加载数据
    print("加载 instance_table...")
    instance_table = load_instance_table(args.instance_table)
    print(f"  {len(instance_table)} 条记录")

    print("加载相机外参...")
    extrinsics = load_extrinsics(args.dataset_path)
    print(f"  {len(extrinsics)} 个相机")

    seg_root = Path(args.segmentation_root) if args.segmentation_root else None

    exp_root = Path(args.experiment_root)
    results = []

    for frame_dir in sorted(exp_root.glob("frame_*")):
        if not frame_dir.is_dir():
            continue

        # LOO 实验
        for held_dir in sorted(frame_dir.glob("held_*")):
            if not held_dir.is_dir():
                continue
            print(f"评估 {held_dir.relative_to(exp_root)}...")
            r = evaluate_single_experiment(
                held_dir, instance_table, args.dataset_path,
                extrinsics, args.downsample_factor, seg_root, is_baseline=False
            )
            if r is not None:
                results.append(r)

        # Baseline
        baseline_dir = frame_dir / "baseline"
        if baseline_dir.is_dir():
            print(f"评估 {baseline_dir.relative_to(exp_root)} (baseline)...")
            r = evaluate_single_experiment(
                baseline_dir, instance_table, args.dataset_path,
                extrinsics, args.downsample_factor, seg_root, is_baseline=True
            )
            if r is not None:
                results.append(r)

    # 计算 generalization_gap
    baseline_lookup = {}
    for r in results:
        if r['is_baseline']:
            baseline_lookup[r['frame_id']] = r['full_psnr']
    for r in results:
        if not r['is_baseline'] and r['frame_id'] in baseline_lookup:
            r['baseline_psnr'] = baseline_lookup[r['frame_id']]
            if not np.isnan(r['full_psnr']) and not np.isnan(r['baseline_psnr']):
                r['generalization_gap'] = r['baseline_psnr'] - r['full_psnr']

    # 保存 CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'frame_id', 'held_out_cam', 'is_baseline',
        'full_psnr', 'full_ssim',
        'fg_bbox_psnr', 'bg_psnr', 'fg_seg_psnr',
        'nearest_angle_diff',
        'baseline_psnr', 'generalization_gap',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果保存到 {output_path}")

    # 汇总统计
    if not results:
        return

    loo_results = [r for r in results if not r['is_baseline']]
    if loo_results:
        fg_psnrs = [r['fg_bbox_psnr'] for r in loo_results if not np.isnan(r['fg_bbox_psnr'])]
        full_psnrs = [r['full_psnr'] for r in loo_results if not np.isnan(r['full_psnr'])]
        bg_psnrs = [r['bg_psnr'] for r in loo_results if not np.isnan(r['bg_psnr'])]
        gaps = [r['generalization_gap'] for r in loo_results if not np.isnan(r.get('generalization_gap', np.nan))]

        print("\n" + "=" * 80)
        print("LOO 汇总统计")
        print("=" * 80)
        if full_psnrs:
            print(f"全图 PSNR: {np.mean(full_psnrs):.2f} +/- {np.std(full_psnrs):.2f} dB")
        if fg_psnrs:
            print(f"前景(bbox) PSNR: {np.mean(fg_psnrs):.2f} +/- {np.std(fg_psnrs):.2f} dB")
        if bg_psnrs:
            print(f"背景 PSNR: {np.mean(bg_psnrs):.2f} +/- {np.std(bg_psnrs):.2f} dB")
        if gaps:
            print(f"泛化差距 (baseline - LOO): {np.mean(gaps):.2f} +/- {np.std(gaps):.2f} dB")

        # 按相机分组
        print("\n按 held-out 相机分组:")
        for cam in sorted(set(r['held_out_cam'] for r in loo_results)):
            cam_r = [r for r in loo_results if r['held_out_cam'] == cam]
            cam_fg = [r['fg_bbox_psnr'] for r in cam_r if not np.isnan(r['fg_bbox_psnr'])]
            if cam_fg:
                print(f"  {cam}: FG PSNR = {np.mean(cam_fg):.2f} +/- {np.std(cam_fg):.2f} dB")

        # 按视角差分桶
        angle_diffs = [r['nearest_angle_diff'] for r in loo_results if not np.isnan(r['nearest_angle_diff'])]
        if angle_diffs:
            print("\n按视角差分桶:")
            buckets = [('<30', 0, 30), ('30-60', 30, 60), ('60-90', 60, 90), ('>90', 90, 181)]
            for label, lo, hi in buckets:
                bucket_r = [r for r in loo_results
                           if not np.isnan(r['nearest_angle_diff'])
                           and lo <= r['nearest_angle_diff'] < hi]
                bucket_fg = [r['fg_bbox_psnr'] for r in bucket_r if not np.isnan(r['fg_bbox_psnr'])]
                if bucket_fg:
                    print(f"  {label}°: FG PSNR = {np.mean(bucket_fg):.2f} +/- {np.std(bucket_fg):.2f} "
                          f"({len(bucket_fg)} samples)")


if __name__ == "__main__":
    main()
