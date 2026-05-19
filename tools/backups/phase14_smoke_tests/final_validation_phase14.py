#!/usr/bin/env python3
"""Phase 14 final validation: checkpoint stats, render sanity, reid init."""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.datasets import make as make_dataset
from threedgrut.model.model import MixtureOfGaussians

FINAL_CKPT = (
    "/data02/zhangrunxiang/3dgrut/outputs/"
    "phase14_clean_geometry/full_soft_reset_30k/"
    "Wildtrack-1505_180007/ckpt_last.pt"
)
CONFIG_PATH = "/data02/zhangrunxiang/3dgrut/configs/apps/wildtrack_full_3dgut.yaml"
DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
OUTPUT_DIR = (
    "/data02/zhangrunxiang/3dgrut/outputs/"
    "phase14_clean_geometry/full_soft_reset_30k/final_validation"
)
REID_INIT_DIR = (
    "/data02/zhangrunxiang/3dgrut/outputs/"
    "phase14_clean_geometry/full_soft_reset_30k/reid_init"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry_checkpoint", default=FINAL_CKPT)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--dataset_path", default=DATASET_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--reid_init_dir", default=REID_INIT_DIR)
    parser.add_argument("--samples_per_camera", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "render_sanity"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "layer0b"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "reid_preflight"), exist_ok=True)
    os.makedirs(args.reid_init_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load config
    conf = OmegaConf.load(args.config)

    # ===== TASK 2: Checkpoint Stats =====
    print("=" * 60)
    print("TASK 2: Checkpoint Stats")
    print("=" * 60)

    ckpt = torch.load(args.geometry_checkpoint, map_location="cpu", weights_only=False)
    print(f"Checkpoint type: {type(ckpt)}")
    print(f"Checkpoint keys: {list(ckpt.keys())[:10]}...")

    # Determine state dict location
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    print(f"State dict keys: {list(state.keys())[:20]}")

    # Extract shapes - checkpoint may or may not have prefix
    stats = {}
    prefix = "_mixture_of_gaussians"
    for key in ["positions", "density", "scale", "rotation", "features_albedo", "features_specular"]:
        full_key = f"{prefix}.{key}"
        if full_key in state:
            t = state[full_key]
            stats[key] = {"shape": list(t.shape), "dtype": str(t.dtype)}
            print(f"{key}: {t.shape} ({t.dtype})")
        elif key in state:
            t = state[key]
            stats[key] = {"shape": list(t.shape), "dtype": str(t.dtype)}
            print(f"{key}: {t.shape} ({t.dtype}) (no prefix)")
        else:
            stats[key] = None
            print(f"{key}: NOT FOUND")

    # Person feature
    has_person_feature = False
    for pf_key in [f"{prefix}._person_feature", f"{prefix}.person_feature", "person_feature", "_person_feature"]:
        if pf_key in state:
            has_person_feature = True
            t = state[pf_key]
            stats["_person_feature"] = {"shape": list(t.shape), "dtype": str(t.dtype)}
            print(f"_person_feature: {t.shape}")
            break

    if not has_person_feature:
        stats["_person_feature"] = None
        print("_person_feature: NOT FOUND")

    # Density / opacity stats
    density_key = f"{prefix}.density" if f"{prefix}.density" in state else "density"
    if stats["density"] is not None:
        density = state[density_key]
        density_vals = density.float()
        # Convert to opacity via sigmoid
        opacity = torch.sigmoid(density_vals)
        stats["density_stats"] = {
            "mean": float(density_vals.mean()),
            "std": float(density_vals.std()),
            "min": float(density_vals.min()),
            "max": float(density_vals.max()),
        }
        stats["opacity_stats"] = {
            "mean": float(opacity.mean()),
            "std": float(opacity.std()),
            "min": float(opacity.min()),
            "max": float(opacity.max()),
        }
        print(f"\nDensity stats: mean={stats['density_stats']['mean']:.4f}")
        print(f"Opacity stats: mean={stats['opacity_stats']['mean']:.4f}")

    # Scale stats
    scale_key = f"{prefix}.scale" if f"{prefix}.scale" in state else "scale"
    if stats["scale"] is not None:
        scale = state[scale_key]
        scale_vals = scale.float()
        stats["scale_stats"] = {
            "mean": float(scale_vals.mean()),
            "std": float(scale_vals.std()),
            "min": float(scale_vals.min()),
            "max": float(scale_vals.max()),
        }
        print(f"Scale stats: mean={stats['scale_stats']['mean']:.4f}")

    # Scene extent
    scene_extent = conf.get("scene_extent", conf.get("initialization", {}).get("scene_extent", "auto"))
    stats["scene_extent"] = scene_extent
    stats["final_gaussian_N"] = stats["positions"]["shape"][0] if stats["positions"] else 0
    stats["has_person_feature"] = has_person_feature

    # Save checkpoint_stats.json
    stats_path = os.path.join(args.output_dir, "checkpoint_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {stats_path}")

    # Save checkpoint_load_report.md
    report_lines = [
        "# Checkpoint Load Report",
        "",
        f"**Path**: `{args.geometry_checkpoint}`",
        f"**Size**: {os.path.getsize(args.geometry_checkpoint) / 1e6:.1f} MB",
        "",
        "## Checkpoint Structure",
        "",
        f"- Type: `{type(ckpt).__name__}`",
        f"- Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}",
        "",
        "## Gaussian Stats",
        "",
        f"- Final Gaussian N: **{stats['final_gaussian_N']}**",
        f"- Positions shape: `{stats['positions']['shape']}`",
        f"- Density shape: `{stats['density']['shape']}`",
        f"- Scale shape: `{stats['scale']['shape']}`",
        f"- Rotation shape: `{stats['rotation']['shape']}`",
        f"- Features albedo shape: `{stats['features_albedo']['shape']}`",
        f"- Features specular shape: `{stats['features_specular']['shape']}`",
        "",
        "## Person Feature",
        "",
        f"- Exists: **{has_person_feature}**",
    ]
    if has_person_feature:
        pf_shape = stats["_person_feature"]["shape"]
        report_lines.append(f"- Shape: `{pf_shape}`")
    else:
        report_lines.append("- Not present (expected for geometry-only training)")
    report_lines.append("")

    report_lines.extend([
        "## Density / Opacity Stats",
        "",
        f"- Density mean: {stats['density_stats']['mean']:.4f}",
        f"- Density std: {stats['density_stats']['std']:.4f}",
        f"- Density min: {stats['density_stats']['min']:.4f}",
        f"- Density max: {stats['density_stats']['max']:.4f}",
        "",
        f"- Opacity mean: {stats['opacity_stats']['mean']:.4f}",
        f"- Opacity std: {stats['opacity_stats']['std']:.4f}",
        f"- Opacity min: {stats['opacity_stats']['min']:.4f}",
        f"- Opacity max: {stats['opacity_stats']['max']:.4f}",
        "",
        "## Scale Stats",
        "",
        f"- Scale mean: {stats['scale_stats']['mean']:.4f}",
        f"- Scale std: {stats['scale_stats']['std']:.4f}",
        f"- Scale min: {stats['scale_stats']['min']:.4f}",
        f"- Scale max: {stats['scale_stats']['max']:.4f}",
        "",
        f"## Scene Extent: {scene_extent}",
        "",
        "## Verdict",
        "",
        "✅ Checkpoint loaded successfully. All expected tensors present.",
        "",
    ])

    report_path = os.path.join(args.output_dir, "checkpoint_load_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved: {report_path}")

    print("\n✅ TASK 2 COMPLETE")

    # ===== TASK 3: Render Sanity =====
    print("\n" + "=" * 60)
    print("TASK 3: Render Sanity")
    print("=" * 60)

    # Initialize model
    model = MixtureOfGaussians(conf, scene_extent=scene_extent)

    # Load checkpoint
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    # Load dataset
    train_dataset, val_dataset = make_dataset("wildtrack", conf, ray_jitter=None)

    # Build acceleration structure
    print("Building acceleration structure...")
    model.build_acc(rebuild=True)
    print("Done.")

    # Render sanity
    render_dir = os.path.join(args.output_dir, "render_sanity")
    os.makedirs(render_dir, exist_ok=True)
    rgb_dir = os.path.join(render_dir, "render_rgb_examples")
    opacity_dir = os.path.join(render_dir, "opacity_examples")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(opacity_dir, exist_ok=True)

    render_results = []
    cam_counts = defaultdict(int)
    samples_per_cam = 3  # Just 3 samples per camera for sanity

    print("Rendering samples...")
    with torch.no_grad():
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            cam_id = sample.get("camera_id", "UNKNOWN")
            frame_idx = sample.get("frame_idx", 0)

            if cam_counts[cam_id] >= samples_per_cam:
                continue

            try:
                gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(sample)
                render_out = model(gpu_batch, train=False, frame_id=frame_idx)

                pred_rgb = render_out.get("pred_rgb")
                pred_opacity = render_out.get("pred_opacity")

                if pred_rgb is None:
                    render_results.append({
                        "camera": cam_id, "frame_id": frame_idx,
                        "status": "no_pred_rgb",
                    })
                    continue

                # Handle tensor layouts
                if pred_rgb.dim() == 4:
                    pred_rgb = pred_rgb[0]
                if pred_rgb.dim() == 3 and pred_rgb.shape[-1] == 3:
                    pred_rgb = pred_rgb.permute(2, 0, 1)
                if pred_rgb.shape[0] == 3:
                    rgb_np = pred_rgb.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                else:
                    rgb_np = pred_rgb.clamp(0, 1).cpu().numpy()

                # Opacity
                if pred_opacity is not None:
                    if pred_opacity.dim() == 4:
                        pred_opacity = pred_opacity[0, ..., 0]
                    elif pred_opacity.dim() == 3:
                        if pred_opacity.shape[0] == 1:
                            pred_opacity = pred_opacity[0]
                        elif pred_opacity.shape[-1] == 1:
                            pred_opacity = pred_opacity[..., 0]
                    op_np = pred_opacity.clamp(0, 1).cpu().numpy()
                else:
                    op_np = np.zeros_like(rgb_np[:, :, 0])

                # Determine status
                rgb_min = float(rgb_np.min())
                rgb_max = float(rgb_np.max())
                rgb_mean = float(rgb_np.mean())
                op_min = float(op_np.min())
                op_max = float(op_np.max())
                op_mean = float(op_np.mean())

                if rgb_max < 0.01 and rgb_min < 0.01:
                    status = "too_dark"
                elif rgb_min > 0.99:
                    status = "too_bright"
                elif rgb_np.shape[0] == 0 or rgb_np.shape[1] == 0:
                    status = "empty"
                else:
                    status = "normal"

                H, W = rgb_np.shape[0], rgb_np.shape[1]

                render_results.append({
                    "camera": cam_id,
                    "frame_id": frame_idx,
                    "rgb_min": f"{rgb_min:.4f}",
                    "rgb_max": f"{rgb_max:.4f}",
                    "rgb_mean": f"{rgb_mean:.4f}",
                    "opacity_min": f"{op_min:.4f}",
                    "opacity_max": f"{op_max:.4f}",
                    "opacity_mean": f"{op_mean:.4f}",
                    "render_status": status,
                    "resolution": f"{W}x{H}",
                })

                # Save images
                cam_rgb_dir = os.path.join(rgb_dir, cam_id)
                cam_op_dir = os.path.join(opacity_dir, cam_id)
                os.makedirs(cam_rgb_dir, exist_ok=True)
                os.makedirs(cam_op_dir, exist_ok=True)

                rgb_img = (rgb_np * 255).astype(np.uint8)
                op_img = (op_np * 255).astype(np.uint8)
                Image.fromarray(rgb_img).save(os.path.join(cam_rgb_dir, f"frame_{frame_idx}.png"))
                Image.fromarray(op_img).save(os.path.join(cam_op_dir, f"frame_{frame_idx}.png"))

                cam_counts[cam_id] += 1
                print(f"  {cam_id} frame {frame_idx}: {status} ({W}x{H})")

            except Exception as e:
                render_results.append({
                    "camera": cam_id, "frame_id": frame_idx,
                    "status": f"error: {e}",
                })
                print(f"  {cam_id} frame {frame_idx}: ERROR - {e}")

    # Save render sanity report
    render_report = [
        "# Render Sanity Report",
        "",
        "## Results",
        "",
        "| Camera | Frame | RGB Min | RGB Max | RGB Mean | Op Min | Op Max | Op Mean | Status | Resolution |",
        "|--------|-------|---------|---------|----------|--------|--------|---------|--------|------------|",
    ]
    for r in render_results:
        render_report.append(
            f"| {r['camera']} | {r['frame_id']} | {r.get('rgb_min', 'N/A')} | {r.get('rgb_max', 'N/A')} | "
            f"{r.get('rgb_mean', 'N/A')} | {r.get('opacity_min', 'N/A')} | {r.get('opacity_max', 'N/A')} | "
            f"{r.get('opacity_mean', 'N/A')} | {r['status']} | {r.get('resolution', 'N/A')} |"
        )

    # Check all cameras
    all_ok = all(r["status"] == "normal" for r in render_results)
    render_report.extend([
        "",
        "## Verdict",
        "",
        f"- Total samples: {len(render_results)}",
        f"- Cameras covered: {len(set(r['camera'] for r in render_results))}",
        f"- All normal: **{all_ok}**",
        "",
        f"{'✅ PASS' if all_ok else '⚠️ ISSUES DETECTED'}",
    ])

    render_report_path = os.path.join(render_dir, "render_sanity_report.md")
    with open(render_report_path, "w") as f:
        f.write("\n".join(render_report))
    print(f"\nSaved: {render_report_path}")
    print("✅ TASK 3 COMPLETE")

    print("\n✅✅✅ TASKS 2-3 COMPLETE ✅✅✅")
    print("Checkpoint stats and render sanity done.")
    print("Next: Layer 0b validation (use phase13_layer0b_geometry_support_verify.py)")


if __name__ == "__main__":
    main()
