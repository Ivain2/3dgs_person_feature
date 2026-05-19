#!/usr/bin/env python3
"""
Phase 13 Layer 0b: Geometry Support Verification

Verifies that bbox scale fix correctly reads rendered opacity / person_feature_map / Gaussian support.

Usage:
    python tools/phase13_layer0b_geometry_support_verify.py \
        --geometry_checkpoint runs/Wildtrack-2802_161501/ckpt_last.pt \
        --person_feature_checkpoint outputs/phase12c_gaussianset_clean_random_fixed_eval_lr1e3_v2/checkpoint_latest.pt \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --dataset_path /data02/zhangrunxiang/data/Wildtrack \
        --output_dir outputs/phase13_layer0b_geometry_support_verify \
        --samples_per_camera 40 \
        --overlay_samples_per_camera 20
"""

import os
import sys
import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from threedgrut.datasets.dataset_wildtrack import WildtrackDataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import scale_bbox_to_render


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry_checkpoint", type=str, required=True)
    parser.add_argument("--person_feature_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/apps/wildtrack_full_3dgut.yaml")
    parser.add_argument("--dataset_path", type=str, default="/data02/zhangrunxiang/data/Wildtrack")
    parser.add_argument("--output_dir", type=str, default="outputs/phase13_layer0b_geometry_support_verify")
    parser.add_argument("--samples_per_camera", type=int, default=40)
    parser.add_argument("--overlay_samples_per_camera", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_geometry_and_person_feature(geo_ckpt_path, pf_ckpt_path, device="cuda"):
    """Load full geometry + person_feature from checkpoints."""
    print(f"\nLoading geometry checkpoint: {geo_ckpt_path}")
    geo_ckpt = torch.load(geo_ckpt_path, map_location=device, weights_only=False)
    conf = geo_ckpt['config']
    print(f"  Using embedded config from checkpoint")

    n_geom = geo_ckpt['positions'].shape[0]
    print(f"  Geometry Gaussians: {n_geom}")

    pf_ckpt = None
    pf_dim = 512
    pf_metadata = None

    if pf_ckpt_path and os.path.exists(pf_ckpt_path):
        print(f"\nLoading person_feature checkpoint: {pf_ckpt_path}")
        pf_ckpt = torch.load(pf_ckpt_path, map_location=device, weights_only=False)
        if 'model_state_dict' in pf_ckpt and '_person_feature' in pf_ckpt['model_state_dict']:
            pf_tensor = pf_ckpt['model_state_dict']['_person_feature']
            n_pf_ckpt = pf_tensor.shape[0]
            pf_dim = pf_tensor.shape[1]

            if n_geom == n_pf_ckpt:
                ordering = "assumed_same"
            elif n_pf_ckpt < n_geom:
                ordering = "assumed_same (zero-padded)"
            else:
                ordering = "assumed_same (truncated)"

            pf_metadata = {
                'geometry_num_gaussians': n_geom,
                'person_feature_num_gaussians': n_pf_ckpt,
                'person_feature_dim': pf_dim,
                'same_count': min(n_geom, n_pf_ckpt),
                'ordering_verified': ordering,
            }
            print(f"  Geometry N: {n_geom}, PF N: {n_pf_ckpt}, dim: {pf_dim}")
            print(f"  ordering_verified: {ordering}")

    if n_geom != (pf_ckpt['model_state_dict']['_person_feature'].shape[0] if pf_ckpt else n_geom):
        print(f"\n⚠️  N mismatch but continuing with min(N_geom, N_pf)")

    OmegaConf.set_struct(conf, False)
    conf.model.person_feature_dim = pf_dim
    OmegaConf.set_struct(conf, True)
    print(f"  Set config model.person_feature_dim = {pf_dim}")

    model = MixtureOfGaussians(conf, scene_extent=geo_ckpt.get('scene_extent', 10.0))
    model.positions.data = geo_ckpt['positions'].to(device)
    model.rotation.data = geo_ckpt['rotation'].to(device)
    model.scale.data = geo_ckpt['scale'].to(device)
    model.density.data = geo_ckpt['density'].to(device)
    model.features_albedo.data = geo_ckpt['features_albedo'].to(device)
    model.features_specular.data = geo_ckpt['features_specular'].to(device)

    if pf_ckpt is not None and '_person_feature' in pf_ckpt['model_state_dict']:
        pf = pf_ckpt['model_state_dict']['_person_feature'].to(device)
        n_pf = pf.shape[0]
        if n_geom > model._person_feature.shape[0]:
            new_pf = torch.zeros(n_geom, pf_dim, device=device, dtype=torch.float32)
            new_pf[:min(n_geom, n_pf)] = pf[:min(n_geom, n_pf)]
            model._person_feature.data = new_pf
        else:
            model._person_feature.data = pf[:n_geom]

        OmegaConf.set_struct(model.conf, False)
        model.conf.model.person_feature_dim = pf_dim
        OmegaConf.set_struct(model.conf, True)

    print(f"\n  Model state:")
    print(f"    Positions: {model.positions.shape}")
    print(f"    Rotation: {model.rotation.shape}")
    print(f"    Scale: {model.scale.shape}")
    print(f"    Density: {model.density.shape}")
    print(f"    Features albedo: {model.features_albedo.shape}")
    print(f"    Features specular: {model.features_specular.shape}")
    print(f"    Person feature: {model._person_feature.shape}")

    return model, conf, pf_metadata


def render_and_verify(model, dataset, device, samples_per_cam=40, overlay_samples_per_cam=20, output_dir=None):
    """Render + compute bbox-internal support metrics."""
    print("\n" + "="*80)
    print("Starting Render + BBox Support Verification")
    print("="*80)

    model.eval().to(device)
    print("Building acceleration structures...")
    model.build_acc(rebuild=True)

    overlay_dir = os.path.join(output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    cam_metrics = defaultdict(list)
    sample_records = []
    cam_overlay_counts = defaultdict(int)
    total = 0
    valid_count = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        cam_id = sample['camera_id']
        frame_idx = sample['frame_idx']
        instances = sample.get('instances', [])

        if len(cam_metrics[cam_id]) >= samples_per_cam:
            continue
        if not instances:
            continue

        gpu_batch = dataset.get_gpu_batch_with_intrinsics(sample)

        with torch.no_grad():
            outputs = model(gpu_batch, train=False, frame_id=frame_idx, render_person_feature=True)

        pred_rgb = outputs.get("pred_rgb")
        pred_opacity = outputs.get("pred_opacity")
        person_feature_map = outputs.get("person_feature_map")

        if pred_rgb is None or pred_opacity is None:
            continue

        if pred_rgb.dim() == 4:
            pred_rgb = pred_rgb[0]
        
        # Handle various opacity shapes: [1,H,W,1], [1,H,W], [H,W,1], [H,W]
        if pred_opacity.dim() == 4:
            pred_opacity = pred_opacity[0, ..., 0]  # [1,H,W,1] -> [H,W]
        elif pred_opacity.dim() == 3:
            if pred_opacity.shape[0] == 1:
                pred_opacity = pred_opacity[0]  # [1,H,W] -> [H,W]
            elif pred_opacity.shape[-1] == 1:
                pred_opacity = pred_opacity[..., 0]  # [H,W,1] -> [H,W]

        # Handle pred_rgb shapes: [C,H,W] or [H,W,C]
        if pred_rgb.dim() == 3 and pred_rgb.shape[0] == 3:
            rgb_np = (pred_rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        elif pred_rgb.dim() == 3 and pred_rgb.shape[-1] == 3:
            rgb_np = (pred_rgb.cpu().numpy() * 255).astype(np.uint8)
        else:
            rgb_np = (pred_rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        if rgb_np.ndim == 2:
            rgb_np = np.stack([rgb_np] * 3, axis=-1)
        elif rgb_np.shape[2] > 3:
            rgb_np = rgb_np[:, :, :3]
        opacity_np = pred_opacity.cpu().numpy()  # [H, W]

        feature_norm_map = None
        if person_feature_map is not None:
            pfm = person_feature_map[0] if isinstance(person_feature_map, tuple) else person_feature_map
            feature_norm_map = pfm.norm(dim=0).cpu().numpy()

        H, W = rgb_np.shape[:2]
        overlay_img = np.ascontiguousarray(rgb_np.copy())

        for inst in instances:
            if len(cam_metrics[cam_id]) >= samples_per_cam:
                break

            train_id = inst.get('train_id')
            if train_id is None:
                continue

            bbox_original = inst.get('bbox_xyxy_original')
            img_w_orig = inst.get('img_width_original', 1920)
            img_h_orig = inst.get('img_height_original', 1088)

            if bbox_original is not None:
                # Debug: print first 3 samples to trace scaling issue
                if total < 3:
                    print(f"Debug bbox scaling:")
                    print(f"  bbox_original = {bbox_original}")
                    print(f"  img_w_orig = {img_w_orig}, img_h_orig = {img_h_orig}")
                    print(f"  W = {W}, H = {H}")
                
                bbox_render = scale_bbox_to_render(bbox_original, img_w_orig, img_h_orig, W, H)
                
                if total < 3:
                    print(f"  bbox_render = {bbox_render}")
            else:
                bbox_xyxy = inst.get('bbox_xyxy')
                if bbox_xyxy is not None:
                    bbox_render = torch.tensor(bbox_xyxy, dtype=torch.float32) if isinstance(bbox_xyxy, (list, tuple)) else bbox_xyxy.float()
                else:
                    continue

            xmin = int(torch.clamp(bbox_render[0], 0, W - 1).item())
            ymin = int(torch.clamp(bbox_render[1], 0, H - 1).item())
            xmax = int(torch.clamp(bbox_render[2], xmin + 1, W).item())
            ymax = int(torch.clamp(bbox_render[3], ymin + 1, H).item())
            bw, bh = xmax - xmin, ymax - ymin

            bbox_op = opacity_np[ymin:ymax, xmin:xmax]
            if bbox_op.size == 0:
                bbox_opacity_sum = 0.0
                bbox_opacity_mean = 0.0
                bbox_opacity_max = 0.0
            else:
                bbox_opacity_sum = float(bbox_op.sum())
                bbox_opacity_mean = float(bbox_op.mean())
                bbox_opacity_max = float(bbox_op.max())

            bbox_fn_mean = 0.0
            bbox_fn_max = 0.0
            pooled_fn = 0.0
            bg_fn_mean = 0.0
            bbox_to_bg = 0.0

            if feature_norm_map is not None:
                bbox_fn = feature_norm_map[ymin:ymax, xmin:xmax]
                bbox_fn_mean = float(bbox_fn.mean())
                bbox_fn_max = float(bbox_fn.max())

                if person_feature_map is not None:
                    pfm = person_feature_map[0] if isinstance(person_feature_map, tuple) else person_feature_map
                    region = pfm[:, ymin:ymax, xmin:xmax]
                    pooled = region.mean(dim=(1, 2))
                    pooled_fn = float(pooled.norm().item())

                margin = 10
                bg_vals = []
                if ymin > margin:
                    region = feature_norm_map[:min(margin, H), :]
                    if region.size > 0:
                        bg_vals.append(float(region.mean()))
                if ymax + margin < H:
                    region = feature_norm_map[max(ymax + margin, 0):, :]
                    if region.size > 0:
                        bg_vals.append(float(region.mean()))
                if bg_vals:
                    bg_fn_mean = float(np.mean(bg_vals))
                    bbox_to_bg = bbox_fn_mean / max(bg_fn_mean, 1e-6)

            roi_valid = True
            invalid_reason = None
            if bw <= 1 or bh <= 1:
                roi_valid = False
                invalid_reason = "bbox_too_small"
            elif bbox_opacity_sum < 0.01:
                roi_valid = False
                invalid_reason = "near_zero_opacity"
            elif feature_norm_map is not None and bbox_fn_mean < 0.001:
                roi_valid = False
                invalid_reason = "near_zero_feature"

            record = {
                'cam_id': cam_id, 'frame_id': frame_idx, 'person_id': train_id,
                'bbox_original': bbox_original,
                'bbox_render': [bbox_render[0].item(), bbox_render[1].item(), bbox_render[2].item(), bbox_render[3].item()],
                'bbox_width': bw, 'bbox_height': bh,
                'bbox_opacity_sum': bbox_opacity_sum, 'bbox_opacity_mean': bbox_opacity_mean,
                'bbox_opacity_max': bbox_opacity_max,
                'bbox_feature_norm_mean': bbox_fn_mean, 'bbox_feature_norm_max': bbox_fn_max,
                'bg_feature_norm_mean': bg_fn_mean, 'bbox_to_bg_feature_norm_ratio': bbox_to_bg,
                'pooled_feature_norm': pooled_fn, 'roi_valid': roi_valid, 'invalid_reason': invalid_reason,
            }
            cam_metrics[cam_id].append(record)
            sample_records.append(record)
            total += 1
            if roi_valid:
                valid_count += 1

            color = (0, 255, 0) if roi_valid else (0, 0, 255)
            cv2.rectangle(overlay_img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(overlay_img, f"P{train_id}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if cam_overlay_counts[cam_id] < overlay_samples_per_cam:
            cam_overlay_counts[cam_id] += 1
            fname = f"{cam_id}_frame{frame_idx:05d}.png"
            cv2.imwrite(os.path.join(overlay_dir, fname), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    print(f"\nProcessed {total} samples, {valid_count} valid")
    return cam_metrics, sample_records, cam_overlay_counts


def compute_per_camera_summary(cam_metrics):
    cam_summary = {}
    for cam_id in sorted(cam_metrics.keys()):
        metrics = cam_metrics[cam_id]
        n = len(metrics)
        if n == 0:
            continue

        op_sums = [m['bbox_opacity_sum'] for m in metrics]
        fn_means = [m['bbox_feature_norm_mean'] for m in metrics]
        pn_means = [m['pooled_feature_norm'] for m in metrics]

        roi_valid_count = sum(1 for m in metrics if m['roi_valid'])
        zero_feat = sum(1 for m in metrics if m['bbox_feature_norm_mean'] < 0.01)
        empty_support = sum(1 for m in metrics if m['bbox_opacity_sum'] < 0.01)

        cam_summary[cam_id] = {
            'camera': cam_id, 'num_samples': n,
            'roi_valid_ratio': roi_valid_count / n,
            'bbox_opacity_sum_mean': float(np.mean(op_sums)),
            'bbox_opacity_sum_median': float(np.median(op_sums)),
            'bbox_feature_norm_mean': float(np.mean(fn_means)),
            'pooled_feature_norm_mean': float(np.mean(pn_means)),
            'zero_feature_ratio': zero_feat / n,
            'empty_support_ratio': empty_support / n,
            'verdict': 'PASS' if roi_valid_count / n > 0.5 else 'FAIL',
        }
    return cam_summary


def judge_layer0b(cam_summary, geometry_loaded, renderer_works):
    checks = {'geometry_loaded': geometry_loaded, 'renderer_works': renderer_works}
    for cam_id, s in cam_summary.items():
        checks[f'{cam_id}_roi_valid'] = s['roi_valid_ratio'] > 0.5
        checks[f'{cam_id}_opacity_nonzero'] = s['bbox_opacity_sum_mean'] > 0.01
        checks[f'{cam_id}_feature_nonzero'] = s['bbox_feature_norm_mean'] > 0.01

    all_pass = all(checks.values())
    print("\nLayer 0b Verification Results:")
    print("-" * 60)
    for name, result in checks.items():
        print(f"  {name:<30} {'✅ PASS' if result else '❌ FAIL'}")
    print("-" * 60)
    print(f"\nLayer 0b: {'✅ PASS' if all_pass else '❌ FAIL'}")
    return all_pass


def generate_reports(cam_metrics, cam_summary, layer0b_pass, pf_metadata, output_dir, args):
    print("\n" + "="*80)
    print("Generating Reports")
    print("="*80)

    jsonl_path = os.path.join(output_dir, "sample_support.jsonl")
    with open(jsonl_path, 'w') as f:
        for cam_id in sorted(cam_metrics.keys()):
            for rec in cam_metrics[cam_id]:
                f.write(json.dumps(rec) + '\n')
    print(f"  Saved: {jsonl_path}")

    csv_path = os.path.join(output_dir, "per_camera_support.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['camera', 'num_samples', 'roi_valid_ratio',
                     'bbox_opacity_sum_mean', 'bbox_opacity_sum_median',
                     'bbox_feature_norm_mean', 'pooled_feature_norm_mean',
                     'zero_feature_ratio', 'empty_support_ratio', 'verdict'])
        for cam_id, s in sorted(cam_summary.items()):
            w.writerow([s['camera'], s['num_samples'],
                        f"{s['roi_valid_ratio']:.4f}", f"{s['bbox_opacity_sum_mean']:.6f}",
                        f"{s['bbox_opacity_sum_median']:.6f}", f"{s['bbox_feature_norm_mean']:.6f}",
                        f"{s['pooled_feature_norm_mean']:.6f}",
                        f"{s['zero_feature_ratio']:.4f}", f"{s['empty_support_ratio']:.4f}",
                        s['verdict']])
    print(f"  Saved: {csv_path}")

    report_path = os.path.join(output_dir, "layer0b_report.md")
    with open(report_path, 'w') as f:
        f.write("# Phase 13 Layer 0b: Geometry Support Verification Report\n\n")
        if pf_metadata:
            f.write("## Geometry / Person Feature Alignment\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| geometry_num_gaussians | {pf_metadata['geometry_num_gaussians']} |\n")
            f.write(f"| person_feature_num_gaussians | {pf_metadata['person_feature_num_gaussians']} |\n")
            f.write(f"| person_feature_dim | {pf_metadata['person_feature_dim']} |\n")
            f.write(f"| same_count | {pf_metadata['same_count']} |\n")
            f.write(f"| ordering_verified | {pf_metadata['ordering_verified']} |\n\n")
        f.write("## Per-Camera Support Summary\n")
        f.write("| Camera | Samples | ROI Valid | Opacity Sum | Feature Norm | Pooled Norm | Verdict |\n")
        f.write("|--------|---------|-----------|-------------|--------------|-------------|---------|\n")
        for cam_id, s in sorted(cam_summary.items()):
            f.write(f"| {cam_id} | {s['num_samples']} | {s['roi_valid_ratio']:.4f} | "
                    f"{s['bbox_opacity_sum_mean']:.4f} | {s['bbox_feature_norm_mean']:.4f} | "
                    f"{s['pooled_feature_norm_mean']:.4f} | {s['verdict']} |\n")
        f.write(f"\n## Layer 0b Verdict: {'✅ PASS' if layer0b_pass else '❌ FAIL'}\n")
    print(f"  Saved: {report_path}")

    final_path = os.path.join(output_dir, "final_report.md")
    with open(final_path, 'w') as f:
        f.write("# Phase 13 Layer 0b: Final Report\n\n")
        f.write("## 1. Checkpoint Paths\n\n")
        f.write(f"- **Geometry**: `{args.geometry_checkpoint}`\n")
        f.write(f"- **Person Feature**: `{args.person_feature_checkpoint}`\n\n")

        if pf_metadata:
            f.write("## 2. Geometry / Person Feature Alignment\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            for k, v in pf_metadata.items():
                f.write(f"| {k} | {v} |\n")
            f.write(f"\n**Note**: ordering_verified = {pf_metadata['ordering_verified']}. ")
            f.write("Person_feature checkpoint should be from the same geometry training pipeline.\n\n")

        f.write("## 3. Per-Camera Support Table\n")
        f.write("| Camera | Samples | ROI Valid | Opacity Sum | Feature Norm | Verdict |\n")
        f.write("|--------|---------|-----------|-------------|--------------|---------|\n")
        for cam_id, s in sorted(cam_summary.items()):
            f.write(f"| {cam_id} | {s['num_samples']} | {s['roi_valid_ratio']:.4f} | "
                    f"{s['bbox_opacity_sum_mean']:.4f} | {s['bbox_feature_norm_mean']:.4f} | {s['verdict']} |\n")
        f.write("\n")

        f.write("## 4. Overlay Visualizations\n")
        f.write(f"Path: `{os.path.join(output_dir, 'overlays')}`\n\n")

        f.write(f"## 5. Layer 0b Verdict: {'✅ PASS' if layer0b_pass else '❌ FAIL'}\n\n")

        if layer0b_pass:
            f.write("## 6. Teacher-Only Sanity: Pending (requires training loop)\n\n")
            f.write("## 7. CE Overfit Sanity: Pending (requires training loop)\n\n")
            f.write("## 8. Next Steps: Branch A\n")
            f.write("- Layer 0b PASS → proceed to teacher-only warm-up (500-1000 steps)\n")
            f.write("- If teacher-only passes → CE small overfit sanity\n")
            f.write("- If both pass → 12G Teacher-Regularized SupCon\n")
        else:
            f.write("## 6. Next Steps: Branch D\n")
            f.write("Layer 0b FAIL → Footprint-Aware / Rendered-Contribution Diagnostic\n")
    print(f"  Saved: {final_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1+2: Load geometry + person_feature
    print("\n" + "="*80)
    print("STEP 1+2: Loading Geometry + Person Feature")
    print("="*80)

    try:
        model, conf, pf_metadata = load_geometry_and_person_feature(
            args.geometry_checkpoint, args.person_feature_checkpoint, args.device
        )
        geometry_loaded = True
    except Exception as e:
        print(f"Failed to load geometry: {e}")
        import traceback
        traceback.print_exc()
        geometry_loaded = False
        model = None
        pf_metadata = None

    if not geometry_loaded:
        cam_summary = {}
        for cam in ['C1','C2','C3','C4','C5','C6','C7']:
            cam_summary[cam] = {'camera': cam, 'num_samples': 0, 'roi_valid_ratio': 0.0,
                                'bbox_opacity_sum_mean': 0.0, 'bbox_opacity_sum_median': 0.0,
                                'bbox_feature_norm_mean': 0.0, 'pooled_feature_norm_mean': 0.0,
                                'zero_feature_ratio': 1.0, 'empty_support_ratio': 1.0, 'verdict': 'FAIL'}
        generate_reports({}, cam_summary, False, None, output_dir=args.output_dir, args=args)
        return

    # Step 3: Dataset
    print("\n" + "="*80)
    print("STEP 3: Loading Dataset")
    print("="*80)
    dataset = WildtrackDataset(
        dataset_path=args.dataset_path, split="train",
        downsample_factor=4, load_teacher_cache=True,
    )
    print(f"Dataset loaded: {len(dataset)} samples, render: {dataset.img_width}x{dataset.img_height}")

    # Step 4: Render + verify
    cam_metrics, sample_records, cam_overlay_counts = render_and_verify(
        model, dataset, args.device,
        samples_per_cam=args.samples_per_camera,
        overlay_samples_per_cam=args.overlay_samples_per_camera,
        output_dir=args.output_dir,
    )

    # Step 5: Per-camera summary
    cam_summary = compute_per_camera_summary(cam_metrics)

    # Step 6: Judge
    layer0b_pass = judge_layer0b(cam_summary, geometry_loaded, renderer_works=True)

    # Step 7: Reports
    generate_reports(cam_metrics, cam_summary, layer0b_pass, pf_metadata, args.output_dir, args)

    print("\n" + "="*80)
    print("Layer 0b Verification Complete!")
    print(f"Results: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
