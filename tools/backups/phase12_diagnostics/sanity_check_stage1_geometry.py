#!/usr/bin/env python3
"""
Stage 1: Short All-Camera Geometry Sanity Run

Goal: Quick confirm that C1-C7 camera loader, render, opacity, projection have no obvious bugs.
Uses Phase11A real 3DGS geometry checkpoint.

Data scale: 10-20% frames, C1-C7 all cameras, time-uniform sampling, no clean sample filtering.

Output:
- original image
- rendered RGB
- rendered opacity/alpha
- RGB overlay
- opacity overlay
- per-camera render statistics
"""

import argparse
import json
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


def run_stage1(args):
    print("\n" + "=" * 80)
    print("Stage 1: Short All-Camera Geometry Sanity Run")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config and model
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
    print(f"Positions shape (before): {model.positions.shape}")

    # Load Phase11A geometry checkpoint
    geo_ckpt_path = args.geometry_checkpoint
    if not os.path.exists(geo_ckpt_path):
        print(f"\n❌ ERROR: Geometry checkpoint not found at {geo_ckpt_path}")
        return

    print(f"\nLoading geometry from: {geo_ckpt_path}")
    geo_ckpt = torch.load(geo_ckpt_path, map_location=device)
    model_state = geo_ckpt.get('model_state_dict', geo_ckpt)

    # Check geometry keys
    geo_keys = ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']
    found_keys = {}
    for k in geo_keys:
        if k in model_state:
            found_keys[k] = model_state[k].shape
            print(f"  Found: {k} shape={model_state[k].shape}")

    if not found_keys:
        print("\n❌ ERROR: No geometry keys found in checkpoint!")
        return

    # Load geometry into model
    num_gaussians = found_keys.get('positions', torch.Size([0]))[0]
    print(f"\nReal geometry: {num_gaussians} Gaussians")

    for k in ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']:
        if k in model_state:
            param_name = k if hasattr(model, k) else ('_' + k if hasattr(model, '_' + k) else None)
            if param_name and hasattr(model, param_name):
                getattr(model, param_name).data = model_state[k].to(device)
                getattr(model, param_name).requires_grad = False
                print(f"  Loaded & frozen: {param_name}")

    # Verify geometry
    print(f"\nPositions shape (after): {model.positions.shape}")
    print(f"Positions stats: mean={model.positions.mean().item():.3f}, std={model.positions.std().item():.3f}")
    density = model.get_density()
    print(f"Density stats: mean={density.mean().item():.4f}, max={density.max().item():.4f}")

    # Build camera-to-frame mapping
    cam_frames = defaultdict(set)
    for cam_id, frame_idx in dataset.indices:
        cam_frames[cam_id].add(int(frame_idx))

    all_cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    # Sample 15% of frames per camera, uniformly distributed in time
    sampled_frames = {}
    for cam_id in all_cameras:
        frames = sorted(cam_frames.get(cam_id, []))
        if not frames:
            continue
        n = max(5, int(len(frames) * 0.15))
        # Uniform time sampling
        step = len(frames) / n
        selected = [frames[int(i * step)] for i in range(n) if int(i * step) < len(frames)]
        sampled_frames[cam_id] = selected
        print(f"\n{cam_id}: {len(frames)} total frames -> {len(selected)} sampled")

    # Render diagnostic
    per_camera_stats = {}
    per_sample_records = []

    for cam_id in all_cameras:
        frames = sampled_frames.get(cam_id, [])
        if not frames:
            per_camera_stats[cam_id] = {
                'camera': cam_id, 'num_frames': 0,
                'render_success': False, 'error': 'no_frames',
            }
            continue

        cam_dir = os.path.join(args.output_dir, cam_id)
        os.makedirs(cam_dir, exist_ok=True)

        render_ok = 0
        render_fail = 0
        opacity_sums = []
        rgb_psnrs = []
        bbox_support = []

        for f_idx, frame_id in enumerate(frames):
            # Find dataset index for this camera+frame
            ds_idx = None
            for idx in range(len(dataset)):
                c, f = dataset.indices[idx]
                if c == cam_id and int(f) == frame_id:
                    ds_idx = idx
                    break

            if ds_idx is None:
                continue

            sample_id = f"{cam_id}_frame{frame_id:06d}_{f_idx:03d}"

            try:
                raw_batch = dataset[ds_idx]
                gpu_batch = dataset.get_gpu_batch_with_intrinsics(raw_batch)
            except Exception as e:
                render_fail += 1
                per_sample_records.append({
                    'sample_id': sample_id, 'cam_id': cam_id, 'frame_id': frame_id,
                    'render_success': False, 'error': str(e)[:100],
                })
                continue

            # Load original image
            image_path = None
            if cam_id in dataset.image_paths and frame_id < len(dataset.image_paths[cam_id]):
                image_path = dataset.image_paths[cam_id][frame_id]

            if image_path and os.path.exists(image_path):
                original_image = cv2.imread(image_path)
                if original_image is None:
                    original_image = None
            else:
                original_image = None

            # Render
            try:
                render_out = model(gpu_batch, train=False, frame_id=0, render_person_feature=False)

                if not isinstance(render_out, dict):
                    raise ValueError(f"Render output is not a dict: {type(render_out)}")

                # Get rendered RGB and opacity
                pred_rgb = render_out.get('pred_rgb', None)
                pred_opacity = render_out.get('pred_opacity', None)

                if pred_rgb is None and 'rgb' in render_out:
                    pred_rgb = render_out['rgb']
                if pred_opacity is None and 'alpha' in render_out:
                    pred_opacity = render_out['alpha']

                # Log available keys for first sample
                if f_idx == 0:
                    print(f"\n  {cam_id} render output keys: {list(render_out.keys())}")

                if pred_rgb is not None:
                    pred_rgb_np = (pred_rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    # Shape is [1, H, W, 3]
                    if pred_rgb_np.ndim == 4 and pred_rgb_np.shape[0] == 1:
                        pred_rgb_np = pred_rgb_np[0]
                    if pred_rgb_np.ndim == 3 and pred_rgb_np.shape[-1] != 3:
                        pred_rgb_np = pred_rgb_np.transpose(1, 2, 0)
                else:
                    pred_rgb_np = None

                if pred_opacity is not None:
                    opacity_np = pred_opacity.cpu().numpy()
                    # Shape is [1, H, W, 1]
                    if opacity_np.ndim == 4 and opacity_np.shape[0] == 1:
                        opacity_np = opacity_np[0]
                    if opacity_np.ndim == 3 and opacity_np.shape[-1] == 1:
                        opacity_np = opacity_np[:, :, 0]
                    elif opacity_np.ndim == 3 and opacity_np.shape[0] == 1:
                        opacity_np = opacity_np[0]
                    
                    if opacity_np.ndim == 2:
                        opacity_uint8 = (opacity_np * 255).clip(0, 255).astype(np.uint8)
                        opacity_colormap = cv2.applyColorMap(opacity_uint8, cv2.COLORMAP_JET)
                    else:
                        opacity_colormap = None
                else:
                    opacity_np = None
                    opacity_colormap = None

                # Statistics
                if pred_opacity is not None:
                    opacity_sum = float(pred_opacity.sum().item())
                    opacity_max = float(pred_opacity.max().item())
                    opacity_mean = float(pred_opacity.mean().item())
                else:
                    opacity_sum = opacity_max = opacity_mean = 0

                # PSNR-like metric
                if original_image is not None and pred_rgb_np is not None:
                    h1, w1 = original_image.shape[:2]
                    h2, w2 = pred_rgb_np.shape[:2]
                    if h1 != h2 or w1 != w2:
                        pred_rgb_np = cv2.resize(pred_rgb_np, (w1, h1))
                    mse = np.mean((original_image.astype(float) - pred_rgb_np.astype(float)) ** 2)
                    psnr = 10 * np.log10(255**2 / max(mse, 1e-10))
                else:
                    psnr = 0

                render_ok += 1
                opacity_sums.append(opacity_sum)
                rgb_psnrs.append(psnr)

                # Save visualizations (first 5 samples per camera)
                if f_idx < 5 and original_image is not None:
                    # Original
                    cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_original.png"), original_image)

                    # Rendered RGB
                    if pred_rgb_np is not None:
                        cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_rendered_rgb.png"), pred_rgb_np)

                    # RGB overlay (side by side)
                    if pred_rgb_np is not None:
                        overlay = np.hstack([original_image, pred_rgb_np])
                        cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_rgb_comparison.png"), overlay)

                    # Opacity heatmap
                    if opacity_colormap is not None:
                        cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_opacity_heatmap.png"), opacity_colormap)

                    # Opacity overlay on original
                    if opacity_colormap is not None and original_image is not None:
                        h1, w1 = original_image.shape[:2]
                        h2, w2 = opacity_colormap.shape[:2]
                        if h1 != h2 or w1 != w2:
                            opacity_colormap_resized = cv2.resize(opacity_colormap, (w1, h1))
                        else:
                            opacity_colormap_resized = opacity_colormap
                        blended = cv2.addWeighted(original_image, 0.5, opacity_colormap_resized, 0.5, 0)
                        cv2.imwrite(os.path.join(cam_dir, f"{sample_id}_opacity_overlay.png"), blended)

                record = {
                    'sample_id': sample_id,
                    'cam_id': cam_id,
                    'frame_id': frame_id,
                    'render_success': True,
                    'full_image_opacity_sum': opacity_sum,
                    'full_image_opacity_max': opacity_max,
                    'full_image_opacity_mean': opacity_mean,
                    'rgb_psnr': psnr,
                    'num_gaussians': num_gaussians,
                }
                per_sample_records.append(record)

            except Exception as e:
                render_fail += 1
                per_sample_records.append({
                    'sample_id': sample_id, 'cam_id': cam_id, 'frame_id': frame_id,
                    'render_success': False, 'error': str(e)[:100],
                })

        # Per-camera summary
        per_camera_stats[cam_id] = {
            'camera': cam_id,
            'num_frames': len(frames),
            'render_ok': render_ok,
            'render_fail': render_fail,
            'render_success_rate': render_ok / max(1, len(frames)),
            'mean_opacity_sum': float(np.mean(opacity_sums)) if opacity_sums else 0,
            'max_opacity_sum': float(np.max(opacity_sums)) if opacity_sums else 0,
            'mean_rgb_psnr': float(np.mean(rgb_psnrs)) if rgb_psnrs else 0,
            'all_opacity_positive': all(s > 0 for s in opacity_sums) if opacity_sums else False,
            'any_render_failure': render_fail > 0,
        }

        print(f"\n  {cam_id} Summary:")
        print(f"    Render success: {render_ok}/{len(frames)} ({render_ok/max(1,len(frames)):.1%})")
        if opacity_sums:
            print(f"    Mean opacity sum: {np.mean(opacity_sums):.2f}")
            print(f"    Max opacity sum: {np.max(opacity_sums):.2f}")
        if rgb_psnrs:
            print(f"    Mean RGB PSNR: {np.mean(rgb_psnrs):.2f} dB")

    # Save outputs
    with open(os.path.join(args.output_dir, 'render_sanity_summary.json'), 'w') as f:
        json.dump(per_camera_stats, f, indent=2, default=str)

    with open(os.path.join(args.output_dir, 'render_sanity_per_sample.jsonl'), 'w') as f:
        for record in per_sample_records:
            f.write(json.dumps(record, default=str) + '\n')

    # Generate Stage 1 report
    generate_stage1_report(per_camera_stats, num_gaussians, geo_ckpt_path, args.output_dir)

    print(f"\n{'=' * 80}")
    print(f"Stage 1 Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 80}")


def generate_stage1_report(per_camera_stats, num_gaussians, ckpt_path, output_dir):
    """Generate Stage 1 sanity report."""
    report = f"""# Stage 1: Short All-Camera Geometry Sanity Report

## 1. Real Geometry Loaded

- Checkpoint: {ckpt_path}
- Num Gaussians: {num_gaussians}
- Geometry keys loaded: positions, rotation, scale, density, features_albedo, features_specular

## 2. C1-C7 Render Sanity

| Camera | Frames Sampled | Render OK | Render Fail | Success Rate | Mean Opacity Sum | Mean RGB PSNR |
|--------|---------------|-----------|-------------|--------------|------------------|---------------|
"""
    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        stats = per_camera_stats.get(cam_id, {})
        report += (
            f"| {cam_id} | {stats.get('num_frames', 0)} | "
            f"{stats.get('render_ok', 0)} | {stats.get('render_fail', 0)} | "
            f"{stats.get('render_success_rate', 0):.1%} | "
            f"{stats.get('mean_opacity_sum', 0):.2f} | "
            f"{stats.get('mean_rgb_psnr', 0):.2f} |\n"
        )

    all_ok = all(s.get('render_success_rate', 0) > 0.9 for s in per_camera_stats.values())
    all_opacity_positive = all(s.get('all_opacity_positive', False) for s in per_camera_stats.values())

    report += f"""
## 3. Checks

| Check | Result |
|-------|--------|
| C1-C7 all render | {'✅ Yes' if all_ok else '⚠️ Some failures'} |
| full_image_opacity_sum non-zero | {'✅ Yes' if all_opacity_positive else '⚠️ Some zeros'} |
| RGB aligned with original | Check debug images in output dir |
| C2/C3/C5 render OK | {'✅ Yes' if all(per_camera_stats.get(c, {}).get('render_success_rate', 0) > 0.9 for c in ['C2', 'C3', 'C5']) else '⚠️ Issues'} |

## 4. Conclusion

"""
    if all_ok:
        report += "✅ **Stage 1 PASSED**: All cameras can render with real geometry. Proceed to Stage 2.\n"
    else:
        report += "⚠️ **Stage 1 PARTIAL**: Some cameras have render issues. Check debug images before proceeding.\n"

    with open(os.path.join(output_dir, 'stage1_sanity_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Short All-Camera Geometry Sanity Run')

    parser.add_argument('--output_dir', type=str, default='outputs/phase12_geometry_clean_rebuild_short')
    parser.add_argument('--geometry_checkpoint', type=str,
                        default='runs/phase11A_proto_infonce_opacity_lam005_lr1e4/step_20999.pth')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_stage1(args)


if __name__ == '__main__':
    main()
