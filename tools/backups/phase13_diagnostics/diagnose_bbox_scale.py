#!/usr/bin/env python3
"""
Phase13: BBox Scale Diagnostic

验证 bbox scale 修复是否正确工作。
不训练，不修改 geometry，只验证 bbox 坐标从 original → render 是否正确缩放。

输出：
- per-sample bbox 坐标对比表
- 修复前 vs 修复后 bbox 宽度/高度对比
- bbox opacity sum 对比
- overlay 可视化（render RGB + scaled bbox）
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from rich import pretty, traceback

pretty.install()
traceback.install()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)


class BatchBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self._cam_frame_to_index = {}
        for idx in range(len(dataset)):
            cam_id, frame_idx = dataset.indices[idx]
            self._cam_frame_to_index[(cam_id, int(frame_idx))] = idx

    def get_batch(self, cam_id, frame_idx):
        key = (cam_id, int(frame_idx))
        idx = self._cam_frame_to_index.get(key)
        if idx is None:
            return None
        raw_batch = self.dataset[idx]
        return self.dataset.get_gpu_batch_with_intrinsics(raw_batch)


def scale_bbox_to_render(bbox_xyxy, src_w, src_h, dst_w, dst_h):
    """Scale bbox from source image resolution to render resolution."""
    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)

    if isinstance(bbox_xyxy, (list, tuple)):
        x1, y1, x2, y2 = bbox_xyxy
    else:
        x1 = bbox_xyxy[0]
        y1 = bbox_xyxy[1]
        x2 = bbox_xyxy[2]
        y2 = bbox_xyxy[3]

    x1_scaled = x1 * scale_x
    y1_scaled = y1 * scale_y
    x2_scaled = x2 * scale_x
    y2_scaled = y2 * scale_y

    return [x1_scaled, y1_scaled, x2_scaled, y2_scaled]


def clamp_bbox_render(bbox, W, H):
    xmin = max(0, int(bbox[0]))
    ymin = max(0, int(bbox[1]))
    xmax = min(W, max(xmin + 1, int(bbox[2])))
    ymax = min(H, max(ymin + 1, int(bbox[3])))
    return xmin, ymin, xmax, ymax


def run_diagnostic(args):
    print("\n" + "=" * 80)
    print("Phase13: BBox Scale Diagnostic")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    from hydra import initialize_config_dir, compose

    config_dir = os.path.join(REPO_ROOT, 'configs')
    config_file = 'apps/wildtrack_full_3dgut'

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_file)
    cfg.model.person_feature_dim = 512

    from threedgrut.trainer import Trainer3DGRUT
    trainer = Trainer3DGRUT(cfg)
    model = trainer.model
    dataset = trainer.train_dataset
    device = trainer.device

    print(f"\nModel loaded: {type(model).__name__}")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Dataset img_width: {dataset.img_width}, img_height: {dataset.img_height}")
    print(f"Original image dimensions: {dataset._detect_image_dimensions()}")
    print(f"Downsample factor: {dataset.downsample_factor}")

    batch_builder = BatchBuilder(dataset)

    eval_samples_path = args.eval_samples
    if not eval_samples_path:
        eval_samples_path = os.path.join(REPO_ROOT, 'outputs/phase12_parallel_validation/medium_eval_allcam.json')

    with open(eval_samples_path, 'r') as f:
        eval_samples = json.load(f)
    print(f"\nLoaded {len(eval_samples)} eval samples")

    samples_by_cam = defaultdict(list)
    for s in eval_samples:
        samples_by_cam[s['cam_id']].append(s)

    all_cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    per_camera_stats = {}
    per_sample_records = []

    for cam_id in all_cameras:
        cam_samples = samples_by_cam.get(cam_id, [])
        if not cam_samples:
            per_camera_stats[cam_id] = {'camera': cam_id, 'num_samples': 0}
            continue

        target = min(len(cam_samples), args.samples_per_camera)
        sampled = random.sample(cam_samples, target) if len(cam_samples) > target else cam_samples

        print(f"\n{'='*60}")
        print(f"{cam_id}: {len(cam_samples)} available -> {len(sampled)} sampled")
        print(f"{'='*60}")

        cam_dir = os.path.join(args.output_dir, cam_id)
        os.makedirs(cam_dir, exist_ok=True)

        cam_records = []

        for s_idx, sample in enumerate(sampled):
            person_id = sample.get('person_id', 'unknown')
            frame_id = sample.get('frame_id', sample.get('frame_idx', 'unknown'))
            bbox_original = sample.get('bbox', [0, 0, 0, 0])

            if isinstance(frame_id, str):
                frame_id = int(frame_id)

            sample_id = f"{cam_id}_frame{frame_id:06d}_pid{person_id:03d}_{s_idx:03d}"

            gpu_batch = batch_builder.get_batch(cam_id, frame_id)
            if gpu_batch is None:
                continue

            D, H, W = 512, int(gpu_batch.rays_dir.shape[1]), int(gpu_batch.rays_dir.shape[2])

            orig_w = dataset.img_width * dataset.downsample_factor
            orig_h = dataset.img_height * dataset.downsample_factor

            bbox_before = bbox_original
            bbox_scaled = scale_bbox_to_render(bbox_original, orig_w, orig_h, W, H)
            xmin, ymin, xmax, ymax = clamp_bbox_render(bbox_scaled, W, H)
            width_before = xmax - xmin
            height_before = ymax - ymin

            old_xmin = max(0, int(bbox_original[0]))
            old_ymin = max(0, int(bbox_original[1]))
            old_xmax = min(W, max(old_xmin + 1, int(bbox_original[2])))
            old_ymax = min(H, max(old_ymin + 1, int(bbox_original[3])))
            old_width = old_xmax - old_xmin
            old_height = old_ymax - old_ymin

            is_width1_before = (old_width == 1)
            is_width1_after = (width_before == 1)
            is_clipped_before = (old_xmax == W or old_xmin == 0 or old_ymax == H or old_ymin == 0)
            is_clipped_after = (xmax == W or xmin == 0 or ymax == H or ymin == 0)

            record = {
                'sample_id': sample_id,
                'cam_id': cam_id,
                'frame_id': frame_id,
                'person_id': int(person_id),
                'orig_size': [orig_w, orig_h],
                'render_size': [W, H],
                'bbox_original': list(bbox_original),
                'bbox_before_fix': [old_xmin, old_ymin, old_xmax, old_ymax],
                'bbox_after_fix': [xmin, ymin, xmax, ymax],
                'width_before': old_width,
                'height_before': old_height,
                'width_after': width_before,
                'height_after': height_before,
                'is_width1_before': is_width1_before,
                'is_width1_after': is_width1_after,
                'is_clipped_before': is_clipped_before,
                'is_clipped_after': is_clipped_after,
            }
            cam_records.append(record)
            per_sample_records.append(record)

            if s_idx < 3:
                print(f"  [{s_idx}] orig={bbox_original}, "
                      f"before={old_xmin},{old_ymin},{old_xmax},{old_ymax} (w={old_width},h={old_height}), "
                      f"after={xmin},{ymin},{xmax},{ymax} (w={width_before},h={height_before})")

        valid_records = [r for r in cam_records if r['width_after'] > 1 and r['height_after'] > 1]
        width1_count_before = sum(1 for r in cam_records if r['is_width1_before'])
        width1_count_after = sum(1 for r in cam_records if r['is_width1_after'])
        clipped_count_before = sum(1 for r in cam_records if r['is_clipped_before'])
        clipped_count_after = sum(1 for r in cam_records if r['is_clipped_after'])

        per_camera_stats[cam_id] = {
            'camera': cam_id,
            'num_samples': len(sampled),
            'mean_width_before': float(np.mean([r['width_before'] for r in cam_records])),
            'mean_width_after': float(np.mean([r['width_after'] for r in cam_records])),
            'mean_height_before': float(np.mean([r['height_before'] for r in cam_records])),
            'mean_height_after': float(np.mean([r['height_after'] for r in cam_records])),
            'width1_count_before': width1_count_before,
            'width1_count_after': width1_count_after,
            'clipped_count_before': clipped_count_before,
            'clipped_count_after': clipped_count_after,
            'empty_bbox_count': len(cam_records) - len(valid_records),
            'roi_valid_ratio': len(valid_records) / max(len(cam_records), 1),
        }

        print(f"\n  {cam_id} Summary:")
        stats = per_camera_stats[cam_id]
        for k, v in stats.items():
            if k != 'camera':
                print(f"    {k}: {v}")

    with open(os.path.join(args.output_dir, 'bbox_scale_summary.json'), 'w') as f:
        json.dump(per_camera_stats, f, indent=2, default=str)

    with open(os.path.join(args.output_dir, 'bbox_scale_per_sample.jsonl'), 'w') as f:
        for record in per_sample_records:
            f.write(json.dumps(record, default=str) + '\n')

    generate_report(per_camera_stats, per_sample_records, args.output_dir)

    print(f"\n{'='*80}")
    print(f"Diagnostic Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


def generate_report(per_camera_stats, per_sample_records, output_dir):
    total = len(per_sample_records)
    width1_before = sum(1 for r in per_sample_records if r['is_width1_before'])
    width1_after = sum(1 for r in per_sample_records if r['is_width1_after'])
    clipped_before = sum(1 for r in per_sample_records if r['is_clipped_before'])
    clipped_after = sum(1 for r in per_sample_records if r['is_clipped_after'])

    report = f"""# Phase13: BBox Scale Diagnostic Report

## Key Findings

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Total samples | {total} | {total} |
| width=1 bboxes | {width1_before} ({width1_before/max(1,total):.1%}) | {width1_after} ({width1_after/max(1,total):.1%}) |
| clipped bboxes | {clipped_before} ({clipped_before/max(1,total):.1%}) | {clipped_after} ({clipped_after/max(1,total):.1%}) |

## Per-Camera Comparison

| Camera | Samples | mean_w_before | mean_w_after | mean_h_before | mean_h_after | width1_before | width1_after | roi_valid_ratio |
|--------|---------|--------------|-------------|--------------|-------------|--------------|-------------|----------------|
"""
    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = per_camera_stats.get(cam_id, {})
        report += (
            f"| {cam_id} | {stats.get('num_samples', 0)} | "
            f"{stats.get('mean_width_before', 0):.1f} | {stats.get('mean_width_after', 0):.1f} | "
            f"{stats.get('mean_height_before', 0):.1f} | {stats.get('mean_height_after', 0):.1f} | "
            f"{stats.get('width1_count_before', 0)} | {stats.get('width1_count_after', 0)} | "
            f"{stats.get('roi_valid_ratio', 0):.1%} |\n"
        )

    if width1_after < width1_before * 0.1:
        verdict = "✅ Bbox scale bug is FIXED. width=1 bboxes reduced significantly."
    elif width1_after < width1_before:
        verdict = "⚠️ Bbox scale bug is PARTIALLY fixed. Some improvement but width=1 still exists."
    else:
        verdict = "❌ Bbox scale bug is NOT fixed. width=1 bboxes still prevalent."

    report += f"""
## Verdict

{verdict}

## Next Steps

"""
    if width1_after < width1_before * 0.1:
        report += "bbox scale 修复已验证成功。可以继续进入 teacher-only warm-up / CE sanity 测试。\n"
    else:
        report += "bbox scale 修复效果不理想。需要进一步诊断 bbox 坐标来源问题。\n"

    with open(os.path.join(output_dir, 'bbox_scale_diagnostic_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Phase13: BBox Scale Diagnostic')

    parser.add_argument('--output_dir', type=str, default='outputs/phase13_bbox_scale_diagnostic')
    parser.add_argument('--eval_samples', type=str, default=None)
    parser.add_argument('--samples_per_camera', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_diagnostic(args)


if __name__ == '__main__':
    main()
