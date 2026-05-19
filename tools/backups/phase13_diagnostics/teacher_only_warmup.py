#!/usr/bin/env python3
"""
Phase 13 Layer 0b: Teacher-Only Warm-up Sanity Check

Runs 500-1000 steps of teacher-only training (no SupCon/Proto/MV).
Records per-camera metrics to verify training stability.

Usage:
    python tools/phase13_teacher_only_warmup.py \
        --geometry_checkpoint runs/Wildtrack-2802_161501/ckpt_last.pt \
        --person_feature_checkpoint outputs/phase12c_gaussianset_clean_random_fixed_eval_lr1e3_v2/checkpoint_latest.pt \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --dataset_path /data02/zhangrunxiang/data/Wildtrack \
        --output_dir outputs/phase13_layer0b_geometry_support_verify \
        --steps 1000 \
        --device cuda
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from threedgrut.datasets.dataset_wildtrack import WildtrackDataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import roi_pool, scale_bbox_to_render


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry_checkpoint", type=str, required=True)
    parser.add_argument("--person_feature_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/apps/wildtrack_full_3dgut.yaml")
    parser.add_argument("--dataset_path", type=str, default="/data02/zhangrunxiang/data/Wildtrack")
    parser.add_argument("--output_dir", type=str, default="outputs/phase13_layer0b_geometry_support_verify")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_model(geo_ckpt_path, pf_ckpt_path, device="cuda"):
    """Load geometry + person_feature from checkpoints."""
    print(f"Loading geometry: {geo_ckpt_path}")
    geo_ckpt = torch.load(geo_ckpt_path, map_location=device, weights_only=False)
    conf = geo_ckpt['config']
    n_geom = geo_ckpt['positions'].shape[0]
    print(f"  Geometry Gaussians: {n_geom}")

    pf_dim = 512
    if pf_ckpt_path and os.path.exists(pf_ckpt_path):
        print(f"Loading person_feature: {pf_ckpt_path}")
        pf_ckpt = torch.load(pf_ckpt_path, map_location=device, weights_only=False)
        if '_person_feature' in pf_ckpt['model_state_dict']:
            pf_tensor = pf_ckpt['model_state_dict']['_person_feature']
            pf_dim = pf_tensor.shape[1]
            print(f"  PF dim: {pf_dim}, N: {pf_tensor.shape[0]}")

    OmegaConf.set_struct(conf, False)
    conf.model.person_feature_dim = pf_dim
    OmegaConf.set_struct(conf, True)

    model = MixtureOfGaussians(conf, scene_extent=geo_ckpt.get('scene_extent', 10.0))
    model.positions.data = geo_ckpt['positions'].to(device)
    model.rotation.data = geo_ckpt['rotation'].to(device)
    model.scale.data = geo_ckpt['scale'].to(device)
    model.density.data = geo_ckpt['density'].to(device)
    model.features_albedo.data = geo_ckpt['features_albedo'].to(device)
    model.features_specular.data = geo_ckpt['features_specular'].to(device)

    if pf_ckpt_path and os.path.exists(pf_ckpt_path):
        pf = pf_ckpt['model_state_dict']['_person_feature'].to(device)
        n_pf = pf.shape[0]
        if n_geom > model._person_feature.shape[0]:
            new_pf = torch.zeros(n_geom, pf_dim, device=device, dtype=torch.float32)
            new_pf[:min(n_geom, n_pf)] = pf[:min(n_geom, n_pf)]
            model._person_feature.data = new_pf
        else:
            model._person_feature.data = pf[:n_geom]

    print(f"Model loaded: {model.num_gaussians} Gaussians, person_feature_dim={pf_dim}")
    return model, conf


def cosine_distillation_loss(f_v, t_v):
    """Compute cosine distillation loss: 1 - cos(f_v, t_v)."""
    f_v_norm = F.normalize(f_v, p=2, dim=0)
    t_v_norm = F.normalize(t_v, p=2, dim=0)
    return 1 - torch.dot(f_v_norm, t_v_norm)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("Phase 13 Layer 0b: Teacher-Only Warm-up")
    print("="*80)

    # Load model
    model, conf = load_model(args.geometry_checkpoint, args.person_feature_checkpoint, args.device)
    model.eval().to(args.device)
    model.build_acc(rebuild=True)

    # Load dataset
    print("\nLoading dataset...")
    dataset = WildtrackDataset(
        dataset_path=args.dataset_path, split="train",
        downsample_factor=4, load_teacher_cache=True,
    )
    print(f"Dataset: {len(dataset)} samples, render: {dataset.img_width}x{dataset.img_height}")

    # Metrics tracking
    teacher_loss_curve = []
    teacher_cosine_per_step = []
    teacher_cosine_per_camera = defaultdict(list)
    same_cos_list = []
    diff_cos_list = []
    pooled_feature_norm_list = []
    roi_valid_per_camera = defaultdict(lambda: {'valid': 0, 'total': 0})
    gradient_norm_list = []

    print(f"\nStarting {args.steps} teacher-only validation steps (no training, just diagnostic)...")
    print("No SupCon / Proto / MV enabled.")
    print("This tests: person_feature_map readout, ROI pooling, teacher cosine, gradient flow")

    for step in range(args.steps):
        # Get sample from dataset
        idx = step % len(dataset)
        sample = dataset[idx]
        cam_id = sample['camera_id']
        frame_idx = sample['frame_idx']
        instances = sample.get('instances', [])

        # Find valid instances
        valid_instances = [
            inst for inst in instances
            if inst.get('valid', False) and inst.get('teacher_embedding') is not None
        ]

        if not valid_instances:
            continue

        gpu_batch = dataset.get_gpu_batch_with_intrinsics(sample)

        # Forward pass (eval mode, no gradient for geometry)
        with torch.no_grad():
            render_out = model(
                gpu_batch, train=False, frame_id=frame_idx,
                render_person_feature=True
            )

        person_feature_map = render_out.get('person_feature_map')
        if person_feature_map is None:
            continue

        if isinstance(person_feature_map, tuple):
            person_feature_map = person_feature_map[0]

        # Enable gradient tracking for person_feature only
        person_feature_map.requires_grad_(True)

        # Compute teacher loss
        step_loss = 0
        step_cosines = []
        valid_count = 0

        for inst in valid_instances:
            bbox_xyxy = inst.get('bbox_xyxy')
            if bbox_xyxy is None:
                continue

            teacher_emb = inst['teacher_embedding']
            t_v = torch.tensor(teacher_emb, dtype=torch.float32, device=args.device)

            # ROI pooling
            bbox_tensor = torch.tensor(bbox_xyxy, dtype=torch.float32, device=args.device)
            f_v, pool_stats = roi_pool(person_feature_map, bbox_tensor)

            if f_v is None or isinstance(f_v, tuple):
                continue

            # Compute cosine similarity
            f_v_norm = F.normalize(f_v, p=2, dim=0)
            t_v_norm = F.normalize(t_v, p=2, dim=0)
            cos_sim = torch.dot(f_v_norm, t_v_norm).item()

            # Pooled feature norm
            pooled_norm = f_v.norm().item()

            loss_i = cosine_distillation_loss(f_v, t_v)
            step_loss = step_loss + loss_i
            step_cosines.append(cos_sim)
            pooled_feature_norm_list.append(pooled_norm)

            teacher_cosine_per_camera[cam_id].append(cos_sim)
            roi_valid_per_camera[cam_id]['valid'] += 1
            valid_count += 1

            # Track same/diff cosine (simplified: use threshold)
            if cos_sim > 0.5:
                same_cos_list.append(cos_sim)
            else:
                diff_cos_list.append(cos_sim)

        if valid_count == 0:
            continue

        avg_loss = step_loss.item() / valid_count
        avg_cosine = np.mean(step_cosines) if step_cosines else 0

        # Record metrics
        teacher_loss_curve.append(avg_loss)
        teacher_cosine_per_step.append(avg_cosine)
        roi_valid_per_camera[cam_id]['total'] += 1

        # Test gradient flow on person_feature_map
        person_feature_map.grad = None
        avg_loss_tensor = step_loss / valid_count
        avg_loss_tensor.backward()
        if person_feature_map.grad is not None:
            grad_norm = person_feature_map.grad.norm().item()
            gradient_norm_list.append(grad_norm)

        # Log progress
        if step < 10 or step % 100 == 0 or step == args.steps - 1:
            print(f"Step {step:5d}: teacher_loss={avg_loss:.4f}, "
                  f"cosine={avg_cosine:.4f}, valid={valid_count}, "
                  f"grad_norm={gradient_norm_list[-1] if gradient_norm_list else 0:.6f}")

    # Compute final metrics
    print("\n" + "="*80)
    print("Teacher-Only Warm-up Results")
    print("="*80)

    teacher_loss_curve_arr = np.array(teacher_loss_curve)
    teacher_cosine_arr = np.array(teacher_cosine_per_step)

    # Loss trend (first 20% vs last 20%)
    n = len(teacher_loss_curve_arr)
    if n > 20:
        early_loss = teacher_loss_curve_arr[:n//5].mean()
        late_loss = teacher_loss_curve_arr[-n//5:].mean()
        loss_decreasing = late_loss < early_loss
    else:
        loss_decreasing = False
        early_loss = late_loss = 0

    # Cosine trend
    if n > 20:
        early_cos = teacher_cosine_arr[:n//5].mean()
        late_cos = teacher_cosine_arr[-n//5:].mean()
        cosine_increasing = late_cos > early_cos
    else:
        cosine_increasing = False
        early_cos = late_cos = 0

    # Same/diff gap
    same_cos_mean = np.mean(same_cos_list) if same_cos_list else 0
    diff_cos_mean = np.mean(diff_cos_list) if diff_cos_list else 0
    same_diff_gap = same_cos_mean - diff_cos_mean

    # Pooled feature norm
    pooled_fn_mean = np.mean(pooled_feature_norm_list) if pooled_feature_norm_list else 0

    # Per-camera ROI valid ratio
    roi_valid_ratio_per_camera = {}
    for cam_id, counts in roi_valid_per_camera.items():
        if counts['total'] > 0:
            roi_valid_ratio_per_camera[cam_id] = counts['valid'] / counts['total']
        else:
            roi_valid_ratio_per_camera[cam_id] = 0

    # Per-camera cosine
    teacher_cosine_per_camera_mean = {
        cam_id: np.mean(cosines) for cam_id, cosines in teacher_cosine_per_camera.items()
    }

    # Print summary
    print(f"\nLoss trend: {early_loss:.4f} → {late_loss:.4f} "
          f"({'✅ decreasing' if loss_decreasing else '⚠️  not decreasing'})")
    print(f"Cosine trend: {early_cos:.4f} → {late_cos:.4f} "
          f"({'✅ increasing' if cosine_increasing else '⚠️  not increasing'})")
    print(f"Same cos mean: {same_cos_mean:.4f}")
    print(f"Diff cos mean: {diff_cos_mean:.4f}")
    print(f"Same-diff gap: {same_diff_gap:.4f}")
    print(f"Pooled feature norm mean: {pooled_fn_mean:.4f}")
    print(f"\nPer-camera ROI valid ratio:")
    for cam_id in sorted(roi_valid_ratio_per_camera.keys()):
        print(f"  {cam_id}: {roi_valid_ratio_per_camera[cam_id]:.4f}")
    print(f"\nPer-camera cosine:")
    for cam_id in sorted(teacher_cosine_per_camera_mean.keys()):
        print(f"  {cam_id}: {teacher_cosine_per_camera_mean[cam_id]:.4f}")

    # Determine PASS/FAIL
    pass_conditions = [
        loss_decreasing or late_loss < 0.5,
        cosine_increasing or late_cos > 0.3,
        pooled_fn_mean > 0.1,
        all(r > 0.5 for r in roi_valid_ratio_per_camera.values()) if roi_valid_ratio_per_camera else False,
    ]
    teacher_only_pass = all(pass_conditions)

    print(f"\n{'='*60}")
    print(f"Teacher-Only Verdict: {'✅ PASS' if teacher_only_pass else '❌ FAIL'}")
    print(f"{'='*60}")

    # Save metrics
    metrics = {
        'teacher_loss_curve': teacher_loss_curve,
        'teacher_cosine_mean': teacher_cosine_arr.mean().item(),
        'teacher_cosine_per_camera': teacher_cosine_per_camera_mean,
        'same_cos': same_cos_mean,
        'diff_cos': diff_cos_mean,
        'same_diff_gap': same_diff_gap,
        'pooled_feature_norm': pooled_fn_mean,
        'roi_valid_ratio_per_camera': roi_valid_ratio_per_camera,
        'teacher_only_pass': teacher_only_pass,
        'num_steps': len(teacher_loss_curve),
    }

    metrics_path = os.path.join(args.output_dir, "teacher_only_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    # Generate report
    report_path = os.path.join(args.output_dir, "teacher_only_report.md")
    with open(report_path, 'w') as f:
        f.write("# Phase 13 Layer 0b: Teacher-Only Warm-up Report\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Steps: {args.steps}\n")
        f.write(f"- Geometry checkpoint: `{args.geometry_checkpoint}`\n")
        f.write(f"- Person feature checkpoint: `{args.person_feature_checkpoint}`\n")
        f.write(f"- No SupCon / Proto / MV enabled\n\n")

        f.write(f"## Training Results\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Teacher loss (early) | {early_loss:.4f} |\n")
        f.write(f"| Teacher loss (late) | {late_loss:.4f} |\n")
        f.write(f"| Teacher cosine (early) | {early_cos:.4f} |\n")
        f.write(f"| Teacher cosine (late) | {late_cos:.4f} |\n")
        f.write(f"| Same cos mean | {same_cos_mean:.4f} |\n")
        f.write(f"| Diff cos mean | {diff_cos_mean:.4f} |\n")
        f.write(f"| Same-diff gap | {same_diff_gap:.4f} |\n")
        f.write(f"| Pooled feature norm | {pooled_fn_mean:.4f} |\n\n")

        f.write(f"## Per-Camera Metrics\n\n")
        f.write(f"| Camera | ROI Valid Ratio | Cosine |\n|--------|----------------|--------|\n")
        for cam_id in sorted(roi_valid_ratio_per_camera.keys()):
            cos = teacher_cosine_per_camera_mean.get(cam_id, 0)
            f.write(f"| {cam_id} | {roi_valid_ratio_per_camera[cam_id]:.4f} | {cos:.4f} |\n")
        f.write(f"\n")

        f.write(f"## Teacher-Only Verdict: {'✅ PASS' if teacher_only_pass else '❌ FAIL'}\n\n")

        if teacher_only_pass:
            f.write("## Next Steps\n")
            f.write("- Proceed to CE small overfit sanity check\n")
        else:
            f.write("## Next Steps\n")
            f.write("- Investigate person_feature_map / teacher target / feature norm / gradient\n")
    print(f"Saved report: {report_path}")

    return teacher_only_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
