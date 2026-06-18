#!/usr/bin/env python3
"""
Leave-One-Out 实验可视化脚本

用途：
为每个 (frame, held-out camera) 生成对比图：GT | 渲染 | 差异图 | mask覆盖

运行方式：
    python tools/visualize_loo.py \
        --experiment_root experiments/leave_one_out \
        --output_dir visualizations/loo \
        --dataset_path /data02/zhangrunxiang/data/Wildtrack_small_sample
"""

import argparse
import cv2
import numpy as np
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from loo_utils import (
    compute_psnr_masked, load_instance_table, build_bbox_mask,
    load_segmentation_mask, preprocess_gt, find_render_files,
    find_gt_path, CAMERA_IDS,
)
from eval_leave_one_out import build_render_cam_map


def create_comparison_image(gt: np.ndarray, pred: np.ndarray,
                           bbox_mask: np.ndarray, seg_mask: np.ndarray = None,
                           psnr: float = None) -> np.ndarray:
    """创建对比图：GT | 渲染 | 差异图(10x) | bbox mask覆盖 [精确mask覆盖]"""
    gt_uint8 = (gt * 255).astype(np.uint8)
    pred_uint8 = (pred * 255).astype(np.uint8)

    diff = np.clip(np.abs(gt - pred) * 10, 0, 1)
    diff_uint8 = (diff * 255).astype(np.uint8)

    bbox_overlay = gt_uint8.copy()
    bbox_overlay[bbox_mask, 0] = np.clip(
        bbox_overlay[bbox_mask, 0].astype(int) + 100, 0, 255
    ).astype(np.uint8)

    panels = [gt_uint8, pred_uint8, diff_uint8, bbox_overlay]

    if seg_mask is not None:
        seg_overlay = gt_uint8.copy()
        seg_overlay[seg_mask, 1] = np.clip(
            seg_overlay[seg_mask, 1].astype(int) + 100, 0, 255
        ).astype(np.uint8)
        panels.append(seg_overlay)

    return np.concatenate(panels, axis=1)


def visualize_single_experiment(exp_dir: Path, instance_table: list,
                               dataset_path: str, output_dir: Path,
                               downsample_factor: int,
                               seg_root: Path = None,
                               is_baseline: bool = False) -> bool:
    """可视化单个实验"""
    frame_id = int(exp_dir.parent.name.split('_')[1])
    if is_baseline:
        held_cam = 'all'
        gt_cams = CAMERA_IDS
    else:
        held_cam = exp_dir.name.split('_')[1]
        gt_cams = [held_cam]

    render_files = find_render_files(exp_dir)
    if not render_files:
        return False

    # 构建 渲染文件名 → 相机ID 的映射
    if is_baseline:
        render_cam_map = build_render_cam_map(
            dataset_path, downsample_factor, frame_id, held_out_camera=None
        )
    else:
        render_cam_map = build_render_cam_map(
            dataset_path, downsample_factor, frame_id, held_out_camera=held_cam
        )

    for cam in gt_cams:
        # 通过 render_cam_map 查找对应的渲染文件
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

        gt_path = find_gt_path(dataset_path, cam, frame_id)
        if not gt_path.exists():
            continue

        pred_bgr = cv2.imread(str(render_path))
        if pred_bgr is None:
            continue
        pred = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        gt_raw = cv2.imread(str(gt_path))
        if gt_raw is None:
            continue
        gt = preprocess_gt(gt_raw, downsample=downsample_factor)

        if pred.shape != gt.shape:
            continue

        # 从 GT 原始分辨率推断有效区域尺寸
        h_raw, w_raw = gt_raw.shape[:2]
        effective_h = h_raw // downsample_factor
        render_w = w_raw // downsample_factor
        pred = pred[:effective_h, :render_w]
        gt = gt[:effective_h, :render_w]

        bbox_mask = build_bbox_mask(cam, frame_id, instance_table,
                                    effective_h, render_w, downsample_factor)

        seg_mask = None
        if seg_root is not None:
            seg_path = seg_root / cam / f"{frame_id:04d}.png"
            seg_mask = load_segmentation_mask(seg_path, effective_h, render_w)

        psnr = compute_psnr_masked(pred, gt, bbox_mask)

        comparison = create_comparison_image(gt, pred, bbox_mask, seg_mask, psnr)

        subdir_name = f"frame_{frame_id}_held_{held_cam}" if not is_baseline else f"frame_{frame_id}_baseline"
        output_subdir = output_dir / subdir_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        output_path = output_subdir / f"{cam}.png"
        Image.fromarray(comparison).save(output_path)
        print(f"  保存 {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Out 实验可视化")
    parser.add_argument("--experiment_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="visualizations/loo")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--segmentation_root", type=str, default=None)
    parser.add_argument("--downsample_factor", type=int, default=4,
                        help="下采样因子（默认4）")
    parser.add_argument("--instance_table", type=str,
                        default="/data02/zhangrunxiang/3dgrut/outputs/x1_mv_fusion/instance_table.csv")

    args = parser.parse_args()

    print("加载 instance_table...")
    instance_table = load_instance_table(args.instance_table)

    exp_root = Path(args.experiment_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    seg_root = Path(args.segmentation_root) if args.segmentation_root else None

    success = 0
    total = 0

    for frame_dir in sorted(exp_root.glob("frame_*")):
        if not frame_dir.is_dir():
            continue
        for held_dir in sorted(frame_dir.glob("held_*")):
            if not held_dir.is_dir():
                continue
            total += 1
            print(f"可视化 {held_dir.relative_to(exp_root)}...")
            if visualize_single_experiment(held_dir, instance_table, args.dataset_path,
                                          output_root, args.downsample_factor, seg_root):
                success += 1

        baseline_dir = frame_dir / "baseline"
        if baseline_dir.is_dir():
            total += 1
            print(f"可视化 {baseline_dir.relative_to(exp_root)} (baseline)...")
            if visualize_single_experiment(baseline_dir, instance_table, args.dataset_path,
                                          output_root, args.downsample_factor, seg_root, is_baseline=True):
                success += 1

    print(f"\n可视化完成: {success}/{total}")


if __name__ == "__main__":
    main()
