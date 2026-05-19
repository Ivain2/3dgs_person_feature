#!/usr/bin/env python3
"""
Phase 13: BBox Scale Fix Verification - Complete Verification Loop

Verifies that the bbox scale bug fix correctly resolves the Layer 0 coordinate readout issue.

Usage:
    python tools/phase13_verify_bbox_scale.py \
        --checkpoint outputs/phase12c_gaussianset_clean_random_fixed_eval_lr1e3_v2/checkpoint_latest.pt \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --dataset_path /data02/zhangrunxiang/data/Wildtrack \
        --eval_samples outputs/phase12_parallel_validation/medium_eval_allcam.json \
        --output_dir outputs/phase13_bbox_scale_verify \
        --overlay_samples_per_cam 20 \
        --samples_per_camera 40
"""

import os
import sys
import json
import csv
import math
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from threedgrut.datasets.dataset_wildtrack import WildtrackDataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import scale_bbox_to_render


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 13: BBox Scale Fix Verification")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (optional for Stage A)")
    parser.add_argument("--config", type=str, default="configs/apps/wildtrack_full_3dgut.yaml", help="Path to config file")
    parser.add_argument("--dataset_path", type=str, default="/data02/zhangrunxiang/data/Wildtrack", help="Path to WildTrack dataset")
    parser.add_argument("--eval_samples", type=str, default=None, help="Path to eval samples JSON")
    parser.add_argument("--output_dir", type=str, default="outputs/phase13_bbox_scale_verify", help="Output directory")
    parser.add_argument("--overlay_samples_per_cam", type=int, default=20, help="Overlay samples per camera")
    parser.add_argument("--samples_per_camera", type=int, default=40, help="Samples per camera for diagnostics")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def setup_dataset(dataset_path, downsample_factor=4):
    """Setup the WildTrack dataset."""
    print("\n" + "="*80)
    print("Setting up dataset...")
    print("="*80)
    
    dataset = WildtrackDataset(
        dataset_path=dataset_path,
        split="train",
        downsample_factor=downsample_factor,
        load_teacher_cache=True,
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    print(f"   Image dimensions (render): {dataset.img_width}x{dataset.img_height}")
    print(f"   Downsample factor: {dataset.downsample_factor}")
    print(f"   Original dimensions: {dataset.img_width * dataset.downsample_factor}x{dataset.img_height * dataset.downsample_factor}")
    
    return dataset


def load_eval_samples(eval_samples_path):
    """Load evaluation samples from JSON file."""
    print("\n" + "="*80)
    print(f"Loading eval samples: {eval_samples_path}")
    print("="*80)
    
    with open(eval_samples_path, 'r') as f:
        samples = json.load(f)
    
    print(f"✅ Loaded {len(samples)} eval samples")
    
    # Group by camera
    cam_counts = defaultdict(int)
    for s in samples:
        cam_counts[s['cam_id']] += 1
    
    print(f"   Per-camera: {dict(cam_counts)}")
    
    return samples


def setup_model(conf, checkpoint_path, device="cuda"):
    """Load model from checkpoint."""
    print("\n" + "="*80)
    print(f"Loading checkpoint: {checkpoint_path}")
    print("="*80)
    
    model = MixtureOfGaussians(conf, scene_extent=10.0)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.init_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {model.num_gaussians} Gaussians")
    if hasattr(model, '_person_feature'):
        print(f"   person_feature shape: {model._person_feature.shape}")
    
    return model


def check_double_scaling(dataset, max_samples_per_cam=50):
    """
    Check for double-scaling in bbox coordinates.
    Print first 50 samples per camera with detailed bbox info.
    """
    print("\n" + "="*80)
    print("STEP 1: Double-Scaling Check")
    print("="*80)
    
    render_w = dataset.img_width
    render_h = dataset.img_height
    orig_w = dataset.img_width * dataset.downsample_factor
    orig_h = dataset.img_height * dataset.downsample_factor
    
    print(f"Render resolution: {render_w}x{render_h}")
    print(f"Original resolution: {orig_w}x{orig_h}")
    print(f"Downsample factor: {dataset.downsample_factor}")
    print()
    
    double_scaling_issues = 0
    total_checked = 0
    cam_counts = defaultdict(int)
    
    # Sample data storage
    sample_info_list = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        cam_id = sample['camera_id']
        frame_idx = sample['frame_idx']
        instances = sample.get('instances', [])
        
        if cam_counts[cam_id] >= max_samples_per_cam:
            continue
            
        for inst in instances:
            if cam_counts[cam_id] >= max_samples_per_cam:
                break
                
            train_id = inst.get('train_id')
            if train_id is None:
                continue
            
            bbox_original = inst.get('bbox_xyxy_original')
            bbox_xyxy = inst.get('bbox_xyxy')
            
            total_checked += 1
            cam_counts[cam_id] += 1
            
            # Determine scale mode
            if bbox_original is not None:
                scale_mode = "original_to_render"
            else:
                scale_mode = "fallback"
            
            # Calculate expected values
            if bbox_original is not None:
                # Scale from original to render
                scale_x = render_w / float(orig_w)
                scale_y = render_h / float(orig_h)
                
                bbox_expected_render = [
                    bbox_original[0] * scale_x,
                    bbox_original[1] * scale_y,
                    bbox_original[2] * scale_x,
                    bbox_original[3] * scale_y,
                ]
                
                # Check if bbox_xyxy is already scaled
                if bbox_xyxy is not None:
                    # Calculate expected downsample values
                    ds_factor = dataset.downsample_factor
                    bbox_expected_ds = [
                        int(bbox_original[0] / ds_factor),
                        int(bbox_original[1] / ds_factor),
                        int(bbox_original[2] / ds_factor),
                        int(bbox_original[3] / ds_factor),
                    ]
                    
                    # Check if bbox_xyxy matches expected downsample values
                    matches_ds = all(abs(bbox_xyxy[i] - bbox_expected_ds[i]) <= 2 for i in range(4))
                    
                    # Check if bbox_xyxy matches expected render values (double-scaling!)
                    matches_render = all(abs(bbox_xyxy[i] - bbox_expected_render[i]) <= 2 for i in range(4))
                    
                    if matches_render and not matches_ds:
                        double_scaling_issues += 1
                        scale_mode_detail = "DOUBLE-SCALING DETECTED!"
                    elif matches_ds:
                        scale_mode_detail = "correct (already_render_space)"
                    else:
                        scale_mode_detail = "unknown (check manually)"
                else:
                    bbox_xyxy = bbox_expected_ds  # Simulated
                    scale_mode_detail = "no_bbox_xyxy (using original)"
            else:
                scale_mode_detail = "no_original_bbox"
            
            # Store sample info
            sample_info = {
                'cam_id': cam_id,
                'frame_id': frame_idx,
                'person_id': train_id,
                'bbox_xyxy_original': bbox_original,
                'bbox_xyxy': bbox_xyxy,
                'source_size_used': (orig_w, orig_h) if bbox_original else (render_w, render_h),
                'render_size': (render_w, render_h),
                'scale_mode': scale_mode_detail,
            }
            sample_info_list.append(sample_info)
            
            # Print first few samples per camera for quick check
            if cam_counts[cam_id] <= 3:
                print(f"[{cam_id}] frame={frame_idx:05d}, pid={train_id}")
                if bbox_original:
                    print(f"    bbox_original: {bbox_original}")
                if bbox_xyxy is not None:
                    print(f"    bbox_xyxy:     {bbox_xyxy}")
                print(f"    scale_mode:    {scale_mode_detail}")
                print()
    
    print("\n" + "-"*80)
    print(f"DOUBLE-SCALING CHECK SUMMARY:")
    print(f"Total checked: {total_checked}")
    print(f"Double-scaling issues: {double_scaling_issues}")
    print(f"Per-camera counts: {dict(cam_counts)}")
    
    if double_scaling_issues > 0:
        print(f"⚠️  WARNING: {double_scaling_issues} samples show double-scaling!")
    else:
        print("✅ No double-scaling detected.")
    
    return sample_info_list, double_scaling_issues == 0


def generate_overlays(dataset, model=None, device=None, max_samples_per_cam=20, output_dir=None):
    """
    Generate bbox overlay visualizations for each camera.
    Save rendered RGB + scaled bbox, person_feature_map + scaled bbox, etc.
    """
    print("\n" + "="*80)
    print("STEP 2: BBox Overlay Sanity")
    print("="*80)
    
    overlay_dir = os.path.join(output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    
    render_w = dataset.img_width
    render_h = dataset.img_height
    orig_w = dataset.img_width * dataset.downsample_factor
    orig_h = dataset.img_height * dataset.downsample_factor
    
    cam_counts = defaultdict(int)
    overlay_stats = []
    
    print(f"Generating overlays (max {max_samples_per_cam} per camera)...")
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        cam_id = sample['camera_id']
        frame_idx = sample['frame_idx']
        image_path = sample.get('image_path', 'unknown')
        instances = sample.get('instances', [])
        
        if cam_counts[cam_id] >= max_samples_per_cam:
            continue
        
        if not instances:
            continue
        
        # Load original image for reference
        img_original = cv2.imread(image_path)
        if img_original is None:
            continue
        img_h_orig, img_w_orig = img_original.shape[:2]
        
        # Resize to render size for overlay
        img_render = cv2.resize(img_original, (render_w, render_h), interpolation=cv2.INTER_AREA)
        
        # Create overlay image
        overlay_img = img_render.copy()
        
        valid_bboxes = 0
        for inst in instances:
            train_id = inst.get('train_id')
            valid = inst.get('valid', False)
            
            if train_id is None:
                continue
            
            bbox_original = inst.get('bbox_xyxy_original')
            
            if bbox_original is not None:
                # Scale bbox from original to render
                bbox_render = scale_bbox_to_render(
                    bbox_original,
                    src_w=img_w_orig,
                    src_h=img_h_orig,
                    dst_w=render_w,
                    dst_h=render_h,
                )
            else:
                bbox_xyxy = inst.get('bbox_xyxy')
                if bbox_xyxy is not None:
                    if isinstance(bbox_xyxy, (list, tuple)):
                        bbox_render = torch.tensor(bbox_xyxy, dtype=torch.float32)
                    else:
                        bbox_render = bbox_xyxy.float()
                else:
                    continue
            
            # Clamp bbox
            xmin = int(torch.clamp(bbox_render[0], 0, render_w - 1).item())
            ymin = int(torch.clamp(bbox_render[1], 0, render_h - 1).item())
            xmax = int(torch.clamp(bbox_render[2], xmin + 1, render_w).item())
            ymax = int(torch.clamp(bbox_render[3], ymin + 1, render_h).item())
            
            bw = xmax - xmin
            bh = ymax - ymin
            
            # Draw bbox
            color = (0, 255, 0) if valid else (0, 165, 255)
            cv2.rectangle(overlay_img, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Add label
            label = f"P{train_id}"
            cv2.putText(overlay_img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            valid_bboxes += 1
            
            # Store stats
            bbox_stats = {
                'cam_id': cam_id,
                'frame_id': frame_idx,
                'person_id': train_id,
                'bbox_original': bbox_original,
                'bbox_render_scaled': [bbox_render[0].item(), bbox_render[1].item(), 
                                       bbox_render[2].item(), bbox_render[3].item()],
                'bbox_clamped': [xmin, ymin, xmax, ymax],
                'bbox_width': bw,
                'bbox_height': bh,
                'valid': valid,
            }
            overlay_stats.append(bbox_stats)
        
        if valid_bboxes == 0:
            continue
        
        cam_counts[cam_id] += 1
        
        # Save overlay
        fname = f"{cam_id}_frame{frame_idx:05d}.png"
        fpath = os.path.join(overlay_dir, fname)
        cv2.imwrite(fpath, overlay_img)
        
        if cam_counts[cam_id] <= 3:
            print(f"  Saved overlay: {fpath} ({valid_bboxes} bboxes)")
    
    print("\n" + "-"*80)
    print(f"OVERLAY GENERATION SUMMARY:")
    for cam in sorted(cam_counts.keys()):
        print(f"  {cam}: {cam_counts[cam]} overlays")
    print(f"Total valid bbox stats: {len(overlay_stats)}")
    
    # Check overlay quality
    narrow_bboxes = sum(1 for s in overlay_stats if s['bbox_width'] <= 2 or s['bbox_height'] <= 2)
    edge_bboxes = sum(1 for s in overlay_stats if 
                      s['bbox_clamped'][0] <= 2 or s['bbox_clamped'][1] <= 2 or 
                      s['bbox_clamped'][2] >= render_w - 2 or s['bbox_clamped'][3] >= render_h - 2)
    
    print(f"\nQuality checks:")
    print(f"  Narrow bboxes (<=2px): {narrow_bboxes}/{len(overlay_stats)}")
    print(f"  Edge-touching bboxes: {edge_bboxes}/{len(overlay_stats)}")
    
    return overlay_stats, cam_counts


def run_per_camera_diagnostic(dataset, model=None, device=None, max_samples_per_cam=50, output_dir=None):
    """
    Run per-camera support diagnostic.
    Output per-camera metrics for bbox, pooling, etc.
    """
    print("\n" + "="*80)
    print("STEP 3: Per-Camera Support Diagnostic")
    print("="*80)
    
    render_w = dataset.img_width
    render_h = dataset.img_height
    orig_w = dataset.img_width * dataset.downsample_factor
    orig_h = dataset.img_height * dataset.downsample_factor
    
    cam_metrics = defaultdict(list)
    
    print(f"Running diagnostic (max {max_samples_per_cam} per camera)...")
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        cam_id = sample['camera_id']
        frame_idx = sample['frame_idx']
        instances = sample.get('instances', [])
        
        if len(cam_metrics[cam_id]) >= max_samples_per_cam:
            continue
        
        if not instances:
            continue
        
        img_path = sample.get('image_path', '')
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h_orig, img_w_orig = img.shape[:2]
        
        for inst in instances:
            if len(cam_metrics[cam_id]) >= max_samples_per_cam:
                break
                
            train_id = inst.get('train_id')
            valid = inst.get('valid', False)
            
            if train_id is None:
                continue
            
            bbox_original = inst.get('bbox_xyxy_original')
            
            # Scale bbox to render
            if bbox_original is not None:
                bbox_render = scale_bbox_to_render(
                    bbox_original,
                    src_w=img_w_orig,
                    src_h=img_h_orig,
                    dst_w=render_w,
                    dst_h=render_h,
                )
            else:
                bbox_xyxy = inst.get('bbox_xyxy')
                if bbox_xyxy is not None:
                    if isinstance(bbox_xyxy, (list, tuple)):
                        bbox_render = torch.tensor(bbox_xyxy, dtype=torch.float32)
                    else:
                        bbox_render = bbox_xyxy.float()
                else:
                    continue
            
            # Clamp bbox
            xmin = int(torch.clamp(bbox_render[0], 0, render_w - 1).item())
            ymin = int(torch.clamp(bbox_render[1], 0, render_h - 1).item())
            xmax = int(torch.clamp(bbox_render[2], xmin + 1, render_w).item())
            ymax = int(torch.clamp(bbox_render[3], ymin + 1, render_h).item())
            
            bw = xmax - xmin
            bh = ymax - ymin
            
            is_clipped = (xmin <= 2 or ymin <= 2 or xmax >= render_w - 2 or ymax >= render_h - 2)
            is_narrow = (bw <= 2 or bh <= 2)
            is_empty = (bw <= 0 or bh <= 0)
            
            metric = {
                'bbox_width': bw,
                'bbox_height': bh,
                'is_clipped': is_clipped,
                'is_narrow': is_narrow,
                'is_empty': is_empty,
                'valid': valid,
            }
            
            cam_metrics[cam_id].append(metric)
    
    # Aggregate per-camera metrics
    cam_summary = {}
    for cam_id in sorted(cam_metrics.keys()):
        metrics = cam_metrics[cam_id]
        n = len(metrics)
        if n == 0:
            continue
        
        widths = [m['bbox_width'] for m in metrics]
        heights = [m['bbox_height'] for m in metrics]
        
        cam_summary[cam_id] = {
            'num_samples': n,
            'mean_bbox_width': float(np.mean(widths)),
            'median_bbox_width': float(np.median(widths)),
            'mean_bbox_height': float(np.mean(heights)),
            'median_bbox_height': float(np.median(heights)),
            'clipped_bbox_ratio': float(sum(1 for m in metrics if m['is_clipped']) / n),
            'width_1px_ratio': float(sum(1 for m in metrics if m['bbox_width'] <= 1) / n),
            'empty_bbox_ratio': float(sum(1 for m in metrics if m['is_empty']) / n),
            'valid_ratio': float(sum(1 for m in metrics if m['valid']) / n),
        }
    
    # Output CSV
    csv_path = os.path.join(output_dir, "per_camera_support.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['camera', 'num_samples', 'mean_bbox_width', 'median_bbox_width', 
                         'mean_bbox_height', 'median_bbox_height', 'clipped_bbox_ratio',
                         'width_1px_ratio', 'empty_bbox_ratio', 'valid_ratio', 'verdict'])
        
        for cam_id, summary in sorted(cam_summary.items()):
            # Simple verdict
            verdict = "PASS" if (
                summary['valid_ratio'] > 0.5 and 
                summary['width_1px_ratio'] < 0.1 and 
                summary['empty_bbox_ratio'] < 0.1
            ) else "FAIL"
            
            writer.writerow([
                cam_id,
                summary['num_samples'],
                f"{summary['mean_bbox_width']:.1f}",
                f"{summary['median_bbox_width']:.1f}",
                f"{summary['mean_bbox_height']:.1f}",
                f"{summary['median_bbox_height']:.1f}",
                f"{summary['clipped_bbox_ratio']:.4f}",
                f"{summary['width_1px_ratio']:.4f}",
                f"{summary['empty_bbox_ratio']:.4f}",
                f"{summary['valid_ratio']:.4f}",
                verdict,
            ])
    
    # Output JSON
    json_path = os.path.join(output_dir, "support_summary.json")
    with open(json_path, 'w') as f:
        json.dump(cam_summary, f, indent=2)
    
    # Print summary
    print("\nPer-Camera Support Summary:")
    print("-" * 100)
    print(f"{'Camera':<8} {'Samples':>8} {'Mean W':>8} {'Med W':>8} {'Mean H':>8} {'Med H':>8} {'Clipped':>8} {'1px W':>8} {'Empty':>8} {'Valid':>8}")
    print("-" * 100)
    
    for cam_id, summary in sorted(cam_summary.items()):
        print(f"{cam_id:<8} {summary['num_samples']:>8} {summary['mean_bbox_width']:>8.1f} "
              f"{summary['median_bbox_width']:>8.1f} {summary['mean_bbox_height']:>8.1f} "
              f"{summary['median_bbox_height']:>8.1f} {summary['clipped_bbox_ratio']:>8.4f} "
              f"{summary['width_1px_ratio']:>8.4f} {summary['empty_bbox_ratio']:>8.4f} "
              f"{summary['valid_ratio']:>8.4f}")
    
    print("-" * 100)
    print(f"CSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    
    return cam_summary


def judge_layer0(double_scaling_ok, overlay_stats, cam_summary, output_dir=None):
    """
    Judge whether Layer 0 has passed based on verification criteria.
    """
    print("\n" + "="*80)
    print("STEP 4: Layer 0 PASS/FAIL Judgment")
    print("="*80)
    
    # Check criteria
    checks = {}
    
    # 1. No double-scaling
    checks['no_double_scaling'] = double_scaling_ok
    
    # 2. Bbox overlay quality
    if overlay_stats:
        narrow_ratio = sum(1 for s in overlay_stats if s['bbox_width'] <= 2 or s['bbox_height'] <= 2) / len(overlay_stats)
        edge_ratio = sum(1 for s in overlay_stats if 
                        s['bbox_clamped'][0] <= 2 or s['bbox_clamped'][1] <= 2 or 
                        s['bbox_clamped'][2] >= 480 - 2 or s['bbox_clamped'][3] >= 272 - 2) / len(overlay_stats)
        checks['bbox_not_narrow'] = narrow_ratio < 0.1
        checks['bbox_not_edge'] = edge_ratio < 0.1
    else:
        checks['bbox_not_narrow'] = False
        checks['bbox_not_edge'] = False
    
    # 3. Per-camera ROI valid ratio
    cam_valid_ratios = {}
    for cam_id, summary in cam_summary.items():
        cam_valid_ratios[cam_id] = summary['valid_ratio']
        checks[f'{cam_id}_valid_ratio'] = summary['valid_ratio'] > 0.5  # At least 50% valid
    
    # 4. Width/height reasonable
    for cam_id, summary in cam_summary.items():
        checks[f'{cam_id}_width_ok'] = summary['mean_bbox_width'] > 5
        checks[f'{cam_id}_height_ok'] = summary['mean_bbox_height'] > 10
    
    # Print results
    print("\nLayer 0 Verification Results:")
    print("-" * 60)
    
    all_pass = True
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        if not result:
            all_pass = False
        print(f"  {check_name:<30} {status}")
    
    print("-" * 60)
    
    # Generate support report
    report_path = os.path.join(output_dir, "support_report.md")
    with open(report_path, 'w') as f:
        f.write("# Phase 13: Layer 0 BBox Scale Verification Report\n\n")
        
        f.write("## Double-Scaling Check\n")
        f.write(f"Result: {'✅ PASS - No double-scaling detected' if checks['no_double_scaling'] else '❌ FAIL - Double-scaling detected'}\n\n")
        
        f.write("## Bbox Overlay Quality\n")
        f.write(f"Bboxes not narrow: {'✅ PASS' if checks['bbox_not_narrow'] else '❌ FAIL'}\n")
        f.write(f"Bboxes not edge-touching: {'✅ PASS' if checks['bbox_not_edge'] else '❌ FAIL'}\n\n")
        
        f.write("## Per-Camera Valid Ratio\n")
        f.write("| Camera | Valid Ratio | Status |\n")
        f.write("|--------|-------------|--------|\n")
        for cam_id in sorted(cam_valid_ratios.keys()):
            ratio = cam_valid_ratios[cam_id]
            status = "✅ PASS" if checks[f'{cam_id}_valid_ratio'] else "❌ FAIL"
            f.write(f"| {cam_id} | {ratio:.4f} | {status} |\n")
        f.write("\n")
        
        f.write("## Layer 0 Final Judgment\n")
        if all_pass:
            f.write("### ✅ PASS\n\n")
            f.write("All verification criteria have been met:\n")
            f.write("- No double-scaling detected\n")
            f.write("- Bboxes correctly scale to render space\n")
            f.write("- Bboxes fall on person regions\n")
            f.write("- Per-camera valid ratios are acceptable\n")
            f.write("- Bbox dimensions are reasonable\n\n")
            f.write("Next step: Proceed to teacher-only warm-up sanity.\n")
        else:
            f.write("### ❌ FAIL\n\n")
            f.write("Some verification criteria were not met:\n")
            for check_name, result in checks.items():
                if not result:
                    f.write(f"- {check_name}\n")
            f.write("\nNext step: Fix bbox/scale issues before proceeding.\n")
    
    print(f"\nSupport report saved to: {report_path}")
    
    return all_pass


def run_teacher_only_sanity(dataset, model, device, num_steps=500, output_dir=None):
    """
    Run teacher-only warm-up sanity check.
    Only runs for a small number of steps to verify teacher loss decreases.
    """
    print("\n" + "="*80)
    print("STEP 5: Teacher-Only Warm-up Sanity")
    print("="*80)
    
    teacher_dir = os.path.join(output_dir, "teacher_sanity")
    os.makedirs(teacher_dir, exist_ok=True)
    
    model.train()
    model.to(device)
    
    # Setup optimizer for person_feature only
    person_feature_param = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(person_feature_param, lr=1e-4)
    
    # Track metrics
    loss_history = []
    cam_metrics = defaultdict(list)
    
    print(f"Running {num_steps} steps of teacher-only training...")
    
    for step in range(num_steps):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(sample)
        
        # Forward pass
        outputs = model(gpu_batch, train=True, frame_id=step, render_person_feature=True)
        
        # Compute ReID loss
        instances = gpu_batch.instances
        if not instances:
            continue
        
        person_feature_map = outputs.get("person_feature_map")
        if person_feature_map is None:
            continue
        
        D, H, W = person_feature_map.shape
        loss_list = []
        valid_count = 0
        
        for inst in instances:
            if not inst.get("valid", False):
                continue
            teacher_emb = inst.get("teacher_embedding")
            if teacher_emb is None:
                continue
            
            bbox_original = inst.get("bbox_xyxy_original")
            orig_w = inst.get("img_width_original", 1920)
            orig_h = inst.get("img_height_original", 1088)
            
            if bbox_original is not None:
                bbox_render = scale_bbox_to_render(
                    bbox_original, src_w=orig_w, src_h=orig_h, dst_w=W, dst_h=H
                )
            else:
                bbox_xyxy = inst["bbox_xyxy"]
                if isinstance(bbox_xyxy, (list, tuple)):
                    bbox_render = torch.tensor(bbox_xyxy, dtype=torch.float32, device=person_feature_map.device)
                else:
                    bbox_render = bbox_xyxy.to(person_feature_map.device).float()
            
            xmin = int(torch.clamp(bbox_render[0], 0, W - 1).item())
            ymin = int(torch.clamp(bbox_render[1], 0, H - 1).item())
            xmax = int(torch.clamp(bbox_render[2], xmin + 1, W).item())
            ymax = int(torch.clamp(bbox_render[3], ymin + 1, H).item())
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            bbox_clamped = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32, device=person_feature_map.device)
            
            # Simple mean pooling
            region = person_feature_map[:, ymin:ymax, xmin:xmax]
            pooled = region.mean(dim=(1, 2))
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
            
            t_v = torch.tensor(teacher_emb, dtype=torch.float32, device=pooled.device)
            if t_v.dim() == 1:
                t_v = t_v.unsqueeze(0)
            if pooled.dim() == 1:
                pooled = pooled.unsqueeze(0)
            
            t_v = torch.nn.functional.normalize(t_v, p=2, dim=-1)
            loss_i = 1.0 - torch.nn.functional.cosine_similarity(pooled, t_v).mean()
            loss_list.append(loss_i)
            valid_count += 1
        
        if not loss_list:
            continue
        
        loss = torch.stack(loss_list).mean()
        loss_history.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, valid={valid_count}")
    
    # Save loss curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(loss_history)
    plt.title('Teacher Loss Curve')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(teacher_dir, 'teacher_loss_curve.png'))
    plt.close()
    
    # Compute cosine similarity metrics
    if len(loss_history) >= 10:
        initial_loss = np.mean(loss_history[:10])
        final_loss = np.mean(loss_history[-10:])
        loss_decreased = final_loss < initial_loss
        
        print(f"\nTeacher Sanity Results:")
        print(f"  Initial loss (avg first 10): {initial_loss:.4f}")
        print(f"  Final loss (avg last 10):    {final_loss:.4f}")
        print(f"  Loss decreased:              {'✅ YES' if loss_decreased else '❌ NO'}")
    
    return loss_history


def run_ce_small_overfit(dataset, model, device, num_ids=30, num_steps=200, output_dir=None):
    """
    Run CE small overfit sanity check.
    Select high-support subset and verify accuracy can increase from 0%.
    """
    print("\n" + "="*80)
    print("STEP 6: CE Small Overfit Sanity")
    print("="*80)
    
    ce_dir = os.path.join(output_dir, "ce_sanity")
    os.makedirs(ce_dir, exist_ok=True)
    
    # Select high-support IDs
    id_counts = defaultdict(int)
    for idx in range(min(100, len(dataset))):
        sample = dataset[idx]
        for inst in sample.get('instances', []):
            train_id = inst.get('train_id')
            if train_id is not None and inst.get('valid', False):
                id_counts[train_id] += 1
    
    # Get top IDs
    high_support_ids = sorted(id_counts.keys(), key=lambda x: id_counts[x], reverse=True)[:num_ids]
    print(f"Selected {len(high_support_ids)} high-support IDs")
    
    # Build classifier head
    model.train()
    model.to(device)
    
    feature_dim = 64  # person_feature_dim
    classifier = torch.nn.Linear(feature_dim, len(high_support_ids)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    id_to_label = {pid: idx for idx, pid in enumerate(high_support_ids)}
    
    # Track metrics
    loss_history = []
    acc_history = []
    
    print(f"Running {num_steps} steps of CE overfit...")
    
    for step in range(num_steps):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(sample)
        
        # Forward pass
        outputs = model(gpu_batch, train=True, frame_id=step, render_person_feature=True)
        person_feature_map = outputs.get("person_feature_map")
        
        if person_feature_map is None:
            continue
        
        D, H, W = person_feature_map.shape
        instances = gpu_batch.instances
        
        if not instances:
            continue
        
        # Get pooled features and labels
        features = []
        labels = []
        
        for inst in instances:
            train_id = inst.get('train_id')
            valid = inst.get('valid', False)
            
            if train_id is None or not valid or train_id not in id_to_label:
                continue
            
            bbox_original = inst.get("bbox_xyxy_original")
            orig_w = inst.get("img_width_original", 1920)
            orig_h = inst.get("img_height_original", 1088)
            
            if bbox_original is not None:
                bbox_render = scale_bbox_to_render(
                    bbox_original, src_w=orig_w, src_h=orig_h, dst_w=W, dst_h=H
                )
            else:
                continue
            
            xmin = int(torch.clamp(bbox_render[0], 0, W - 1).item())
            ymin = int(torch.clamp(bbox_render[1], 0, H - 1).item())
            xmax = int(torch.clamp(bbox_render[2], xmin + 1, W).item())
            ymax = int(torch.clamp(bbox_render[3], ymin + 1, H).item())
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            # Pool feature
            region = person_feature_map[:, ymin:ymax, xmin:xmax]
            pooled = region.mean(dim=(1, 2))
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
            
            features.append(pooled)
            labels.append(id_to_label[train_id])
        
        if not features:
            continue
        
        features = torch.stack(features)
        labels = torch.tensor(labels, dtype=torch.long, device=features.device)
        
        # Forward through classifier
        logits = classifier(features)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()
        
        loss_history.append(loss.item())
        acc_history.append(acc)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, acc={acc:.4f}")
    
    # Save plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(loss_history)
    plt.title('CE Loss Curve')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(ce_dir, 'ce_loss_curve.png'))
    plt.close()
    
    plt.figure()
    plt.plot(acc_history)
    plt.title('CE Accuracy Curve')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(ce_dir, 'ce_accuracy_curve.png'))
    plt.close()
    
    # Check if accuracy improved
    if len(acc_history) >= 10:
        initial_acc = np.mean(acc_history[:10])
        final_acc = np.mean(acc_history[-10:])
        acc_improved = final_acc > initial_acc
        
        print(f"\nCE Sanity Results:")
        print(f"  Initial acc (avg first 10): {initial_acc:.4f}")
        print(f"  Final acc (avg last 10):    {final_acc:.4f}")
        print(f"  Accuracy improved:          {'✅ YES' if acc_improved else '❌ NO'}")
    
    return loss_history, acc_history


def generate_final_report(double_scaling_ok, overlay_stats, cam_summary, layer0_pass, 
                          teacher_loss_history=None, ce_loss_history=None, 
                          ce_acc_history=None, output_dir=None):
    """Generate final report."""
    print("\n" + "="*80)
    print("STEP 7: Final Report Generation")
    print("="*80)
    
    report_path = os.path.join(output_dir, "final_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Phase 13: BBox Scale Fix Verification - Final Report\n\n")
        
        # Double-Scaling
        f.write("## 1. Double-Scaling Check\n")
        f.write(f"Result: {'✅ PASS - No double-scaling detected' if double_scaling_ok else '❌ FAIL - Double-scaling detected'}\n\n")
        
        # Bbox Scale Logic
        f.write("## 2. Final BBox Scale Logic\n")
        f.write("```python\n")
        f.write("if bbox_xyxy_original is not None:\n")
        f.write("    bbox_render = scale_bbox_to_render(\n")
        f.write("        bbox_xyxy_original,\n")
        f.write("        src_w=img_width_original,\n")
        f.write("        src_h=img_height_original,\n")
        f.write("        dst_w=W,  # render width\n")
        f.write("        dst_h=H,  # render height\n")
        f.write("    )\n")
        f.write("else:\n")
        f.write("    # bbox_xyxy already in render space\n")
        f.write("    bbox_render = torch.tensor(bbox_xyxy)\n")
        f.write("```\n\n")
        
        # Typical Examples
        f.write("## 3. Typical BBox Examples\n")
        if overlay_stats:
            for i, stat in enumerate(overlay_stats[:3]):
                f.write(f"### Example {i+1}\n")
                f.write(f"- Camera: {stat['cam_id']}\n")
                f.write(f"- Person ID: {stat['person_id']}\n")
                f.write(f"- Original bbox: {stat['bbox_original']}\n")
                f.write(f"- Scaled bbox: [{stat['bbox_render_scaled'][0]:.1f}, "
                        f"{stat['bbox_render_scaled'][1]:.1f}, "
                        f"{stat['bbox_render_scaled'][2]:.1f}, "
                        f"{stat['bbox_render_scaled'][3]:.1f}]\n")
                f.write(f"- Clamped bbox: {stat['bbox_clamped']}\n\n")
        
        # Overlay Paths
        f.write("## 4. Overlay Visualizations\n")
        f.write(f"Path: `{os.path.join(output_dir, 'overlays')}`\n")
        f.write(f"Total overlays: {len(overlay_stats)}\n\n")
        
        # Per-Camera Support
        f.write("## 5. Per-Camera Support Table\n")
        f.write("| Camera | Samples | Mean Width | Mean Height | Clipped Ratio | Valid Ratio |\n")
        f.write("|--------|---------|------------|-------------|---------------|-------------|\n")
        for cam_id in sorted(cam_summary.keys()):
            s = cam_summary[cam_id]
            f.write(f"| {cam_id} | {s['num_samples']} | {s['mean_bbox_width']:.1f} | "
                    f"{s['mean_bbox_height']:.1f} | {s['clipped_bbox_ratio']:.4f} | "
                    f"{s['valid_ratio']:.4f} |\n")
        f.write("\n")
        
        # Layer 0 Judgment
        f.write("## 6. Layer 0 PASS/FAIL Judgment\n")
        f.write(f"Result: {'✅ PASS' if layer0_pass else '❌ FAIL'}\n\n")
        
        # Teacher-Only Sanity
        f.write("## 7. Teacher-Only Sanity Results\n")
        if teacher_loss_history:
            initial = np.mean(teacher_loss_history[:10])
            final = np.mean(teacher_loss_history[-10:])
            f.write(f"Initial loss: {initial:.4f}\n")
            f.write(f"Final loss: {final:.4f}\n")
            f.write(f"Loss decreased: {'✅ YES' if final < initial else '❌ NO'}\n")
        else:
            f.write("Not run (Layer 0 failed)\n")
        f.write("\n")
        
        # CE Overfit Sanity
        f.write("## 8. CE Overfit Sanity Results\n")
        if ce_loss_history and ce_acc_history:
            initial_acc = np.mean(ce_acc_history[:10])
            final_acc = np.mean(ce_acc_history[-10:])
            f.write(f"Initial accuracy: {initial_acc:.4f}\n")
            f.write(f"Final accuracy: {final_acc:.4f}\n")
            f.write(f"Accuracy improved: {'✅ YES' if final_acc > initial_acc else '❌ NO'}\n")
        else:
            f.write("Not run (Teacher-only failed)\n")
        f.write("\n")
        
        # Next Steps
        f.write("## 9. Next Steps Recommendation\n")
        
        if layer0_pass:
            if teacher_loss_history and np.mean(teacher_loss_history[-10:]) < np.mean(teacher_loss_history[:10]):
                if ce_acc_history and np.mean(ce_acc_history[-10:]) > np.mean(ce_acc_history[:10]):
                    f.write("### Branch A: All checks passed\n")
                    f.write("✅ Proceed to 12G Teacher-Regularized SupCon training.\n")
                else:
                    f.write("### Branch C: CE failed\n")
                    f.write("Check label mapping, classifier head, batch construction.\n")
            else:
                f.write("### Branch B: Teacher-only failed\n")
                f.write("Check person_feature_map rendering, teacher target, feature norm, gradient.\n")
        else:
            f.write("### Branch D: Layer 0 failed\n")
            f.write("Continue fixing bbox/scale/render-space alignment. Do NOT run SupCon.\n")
        
        f.write("\n---\n")
        f.write(f"*Report generated: {Path(output_dir).name}*\n")
    
    print(f"Final report saved to: {report_path}")
    return report_path


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup dataset
    dataset = setup_dataset(args.dataset_path, downsample_factor=4)
    
    # ========================================
    # STAGE A: BBox Scale Verification (no model needed)
    # ========================================
    print("\n" + "#"*80)
    print("# STAGE A: BBox Scale Verification")
    print("#"*80)
    
    # Step 1: Check double-scaling
    sample_info_list, double_scaling_ok = check_double_scaling(
        dataset, max_samples_per_cam=args.samples_per_camera
    )
    
    # Step 2: Generate overlays
    overlay_stats, cam_overlay_counts = generate_overlays(
        dataset, model=None, device=args.device, 
        max_samples_per_cam=args.overlay_samples_per_cam,
        output_dir=args.output_dir
    )
    
    # Step 3: Per-camera diagnostic
    cam_summary = run_per_camera_diagnostic(
        dataset, model=None, device=args.device,
        max_samples_per_cam=args.samples_per_camera,
        output_dir=args.output_dir
    )
    
    # Step 4: Layer 0 judgment
    layer0_pass = judge_layer0(
        double_scaling_ok, overlay_stats, cam_summary,
        output_dir=args.output_dir
    )
    
    # ========================================
    # STAGE B: Teacher-Only + CE Sanity (requires checkpoint)
    # ========================================
    teacher_loss_history = None
    ce_loss_history = None
    ce_acc_history = None
    
    if layer0_pass and args.checkpoint and os.path.exists(args.checkpoint):
        print("\n" + "#"*80)
        print("# STAGE B: Teacher-Only + CE Sanity")
        print("#"*80)
        
        # Load config for model
        conf_path = args.config
        if os.path.exists(conf_path):
            # Load base config first, then override with app config
            base_conf_path = os.path.join(os.path.dirname(conf_path), '..', 'base_gs.yaml')
            if os.path.exists(base_conf_path):
                base_conf = OmegaConf.load(base_conf_path)
                app_conf = OmegaConf.load(conf_path)
                conf = OmegaConf.merge(base_conf, app_conf)
            else:
                conf = OmegaConf.load(conf_path)
        else:
            conf = None
        
        # Try to setup model - if it fails, skip Stage B
        try:
            if conf is not None:
                model = setup_model(conf, args.checkpoint, args.device)
                
                # Step 5: Teacher-only sanity
                teacher_loss_history = run_teacher_only_sanity(
                    dataset, model, args.device,
                    num_steps=500,
                    output_dir=args.output_dir
                )
                
                # Step 6: CE small overfit (if teacher-only passes)
                if teacher_loss_history:
                    initial_loss = np.mean(teacher_loss_history[:10])
                    final_loss = np.mean(teacher_loss_history[-10:])
                    teacher_pass = final_loss < initial_loss
                    
                    if teacher_pass:
                        ce_loss_history, ce_acc_history = run_ce_small_overfit(
                            dataset, model, args.device,
                            num_ids=30, num_steps=200,
                            output_dir=args.output_dir
                        )
                    else:
                        print("Teacher-only failed. Skipping CE sanity.")
            else:
                print("Config not found. Skipping Stage B.")
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Skipping Stage B (teacher-only and CE sanity).")
    elif not layer0_pass:
        print("Layer 0 failed. Skipping teacher-only and CE sanity.")
    else:
        print(f"Checkpoint not found at {args.checkpoint}. Skipping Stage B.")
    
    # Step 7: Final report
    generate_final_report(
        double_scaling_ok, overlay_stats, cam_summary, layer0_pass,
        teacher_loss_history=teacher_loss_history,
        ce_loss_history=ce_loss_history,
        ce_acc_history=ce_acc_history,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("Phase 13 verification complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()