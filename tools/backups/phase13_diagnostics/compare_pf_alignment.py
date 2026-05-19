#!/usr/bin/env python3
"""
Phase 13: Person Feature Alignment Comparison - Real Model Version

Compares two PF initialization schemes on fixed 52,493 geometry:
A. zero-pad old PF (50k old + 2,493 zeros)
B. reinit PF (random init [52493, 512])

Uses real geometry rendering, real dataset, real teacher embeddings, real ROI pooling.
"""

import os
import sys
import json
import csv
import time
import gc
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
from threedgrut.utils.roi_pooling import roi_pool

OUTPUT_DIR = "outputs/phase13_pf_alignment_compare"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry_checkpoint", type=str, required=True)
    p.add_argument("--person_feature_checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/apps/wildtrack_full_3dgut.yaml")
    p.add_argument("--dataset_path", type=str, default="/data02/zhangrunxiang/data/Wildtrack")
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--smoke_steps", type=int, default=50)
    p.add_argument("--train_steps", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_geometry_model(geo_ckpt_path, pf_init_fn, device="cuda"):
    """Load geometry from checkpoint, initialize person_feature with given function."""
    geo_ckpt = torch.load(geo_ckpt_path, map_location=device, weights_only=False)
    conf = geo_ckpt['config']
    n_geom = geo_ckpt['positions'].shape[0]
    print(f"  Geometry Gaussians: {n_geom}")

    OmegaConf.set_struct(conf, False)
    conf.model.person_feature_dim = 512
    OmegaConf.set_struct(conf, True)

    model = MixtureOfGaussians(conf, scene_extent=geo_ckpt.get('scene_extent', 10.0))
    model.positions.data = geo_ckpt['positions'].to(device)
    model.rotation.data = geo_ckpt['rotation'].to(device)
    model.scale.data = geo_ckpt['scale'].to(device)
    model.density.data = geo_ckpt['density'].to(device)
    model.features_albedo.data = geo_ckpt['features_albedo'].to(device)
    model.features_specular.data = geo_ckpt['features_specular'].to(device)

    # Initialize person_feature via provided function
    model._person_feature.data = pf_init_fn(n_geom, device)

    return model, n_geom


def zero_pad_old_pf(n_geom, device, pf_ckpt_path):
    """Scheme A: zero-pad old PF."""
    pf_ckpt = torch.load(pf_ckpt_path, map_location=device, weights_only=False)
    old_pf = pf_ckpt['model_state_dict']['_person_feature'].to(device)
    n_old = old_pf.shape[0]
    dim = old_pf.shape[1]

    new_pf = torch.zeros(n_geom, dim, device=device, dtype=torch.float32)
    new_pf[:n_old] = old_pf

    non_zero = (new_pf.norm(dim=1) > 0.01).sum().item()
    print(f"  Zero-pad old PF: old N={n_old}, new N={n_geom}, non_zero={non_zero}/{n_geom}")
    return new_pf


def reinit_pf(n_geom, device):
    """Scheme B: reinit PF with random values matching original init."""
    dim = 512
    new_pf = torch.randn(n_geom, dim, device=device, dtype=torch.float32) * 0.01
    pf_norm = new_pf.norm(dim=1)
    print(f"  Reinit PF: N={n_geom}, dim={dim}, norm_mean={pf_norm.mean().item():.4f}")
    return new_pf


def run_readout_diagnostic(model, dataset, device, scheme_name, output_dir):
    """Step 3: No-training teacher-readout diagnostic."""
    print(f"\n  Readout diagnostic: {scheme_name}...")

    model.eval().to(device)
    model.build_acc(rebuild=True)

    metrics = {
        'cosines': [],
        'pooled_norms': [],
        'teacher_norms': [],
        'valid_per_camera': defaultdict(lambda: {'valid': 0, 'total': 0}),
        'per_camera_cosines': defaultdict(list),
        'zero_count': 0,
        'total_count': 0,
    }

    sample_count = 0
    max_samples = 200  # Limit to 200 samples for speed

    for idx in range(len(dataset)):
        if sample_count >= max_samples:
            break

        sample = dataset[idx]
        cam_id = sample['camera_id']
        frame_idx = sample['frame_idx']
        instances = sample.get('instances', [])

        if not instances:
            continue

        gpu_batch = dataset.get_gpu_batch_with_intrinsics(sample)

        with torch.no_grad():
            render_out = model(gpu_batch, train=False, frame_id=frame_idx, render_person_feature=True)

        person_feature_map = render_out.get("person_feature_map")
        if person_feature_map is None:
            continue
        if isinstance(person_feature_map, tuple):
            person_feature_map = person_feature_map[0]

        for inst in instances:
            if sample_count >= max_samples:
                break
            if not inst.get('valid', False) or inst.get('teacher_embedding') is None:
                continue

            bbox_xyxy = inst.get('bbox_xyxy')
            if bbox_xyxy is None:
                continue

            teacher_emb = inst['teacher_embedding']
            t_v = torch.tensor(teacher_emb, dtype=torch.float32, device=device)
            bbox_tensor = torch.tensor(bbox_xyxy, dtype=torch.float32, device=device)

            f_v, pool_stats = roi_pool(person_feature_map, bbox_tensor)
            if f_v is None or isinstance(f_v, tuple):
                continue

            cos_sim = F.cosine_similarity(f_v, t_v, dim=0).item()
            pooled_norm = f_v.norm().item()
            teacher_norm = t_v.norm().item()

            metrics['cosines'].append(cos_sim)
            metrics['pooled_norms'].append(pooled_norm)
            metrics['teacher_norms'].append(teacher_norm)
            metrics['per_camera_cosines'][cam_id].append(cos_sim)
            metrics['valid_per_camera'][cam_id]['valid'] += 1
            metrics['valid_per_camera'][cam_id]['total'] += 1
            metrics['total_count'] += 1
            sample_count += 1

            if pooled_norm < 0.001:
                metrics['zero_count'] += 1

    # Summary
    all_cos = metrics['cosines']
    summary = {
        'scheme': scheme_name,
        'valid_sample_count': metrics['total_count'],
        'pooled_feature_norm_mean': float(np.mean(metrics['pooled_norms'])) if metrics['pooled_norms'] else 0,
        'pooled_feature_norm_median': float(np.median(metrics['pooled_norms'])) if metrics['pooled_norms'] else 0,
        'teacher_feature_norm_mean': float(np.mean(metrics['teacher_norms'])) if metrics['teacher_norms'] else 0,
        'teacher_feature_norm_median': float(np.median(metrics['teacher_norms'])) if metrics['teacher_norms'] else 0,
        'cosine_to_teacher_mean': float(np.mean(all_cos)) if all_cos else 0,
        'cosine_to_teacher_median': float(np.median(all_cos)) if all_cos else 0,
        'cosine_to_teacher_per_camera': {k: float(np.mean(v)) for k, v in metrics['per_camera_cosines'].items()},
        'per_camera_valid_ratio': {
            k: v['valid'] / max(v['total'], 1)
            for k, v in metrics['valid_per_camera'].items()
        },
        'zero_feature_ratio': metrics['zero_count'] / max(metrics['total_count'], 1),
    }

    print(f"    {scheme_name}: valid={summary['valid_sample_count']}, "
          f"cosine_mean={summary['cosine_to_teacher_mean']:.4f}, "
          f"pooled_norm_mean={summary['pooled_feature_norm_mean']:.4f}, "
          f"zero_ratio={summary['zero_feature_ratio']:.4f}")

    return summary


def run_training_test(model, dataset, device, steps, scheme_name, lr=1e-3, log_interval=10):
    """Steps 4-5: Teacher-only training test with frozen geometry."""
    print(f"\n  Training test: {scheme_name} ({steps} steps)...")

    model.train().to(device)
    model.build_acc(rebuild=True)

    # Freeze all geometry parameters
    for param_name in ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']:
        param = getattr(model, param_name)
        param.requires_grad = False

    # Only optimize _person_feature
    optimizer = torch.optim.Adam([model._person_feature], lr=lr)

    metrics_records = []
    step_start_time = time.time()
    hang_detected = False
    oom_detected = False

    max_samples = len(dataset)
    for step in range(steps):
        idx = step % max_samples
        sample = dataset[idx]
        cam_id = sample['camera_id']
        frame_idx = sample['frame_idx']
        instances = sample.get('instances', [])

        valid_instances = [
            inst for inst in instances
            if inst.get('valid', False) and inst.get('teacher_embedding') is not None
            and inst.get('bbox_xyxy') is not None
        ]

        if not valid_instances:
            continue

        gpu_batch = dataset.get_gpu_batch_with_intrinsics(sample)

        try:
            # Forward pass
            render_out = model(gpu_batch, train=False, frame_id=frame_idx, render_person_feature=True)
            person_feature_map = render_out.get("person_feature_map")
            if person_feature_map is None:
                continue
            if isinstance(person_feature_map, tuple):
                person_feature_map = person_feature_map[0]
        except RuntimeError as e:
            if "out of memory" in str(e):
                oom_detected = True
                print(f"    OOM at step {step}!")
                break
            raise

        # Compute teacher loss
        step_loss = 0
        step_cosines = []
        valid_count = 0

        for inst in valid_instances:
            bbox_xyxy = inst['bbox_xyxy']
            teacher_emb = inst['teacher_embedding']
            t_v = torch.tensor(teacher_emb, dtype=torch.float32, device=device)
            bbox_tensor = torch.tensor(bbox_xyxy, dtype=torch.float32, device=device)

            f_v, pool_stats = roi_pool(person_feature_map, bbox_tensor)
            if f_v is None or isinstance(f_v, tuple):
                continue

            cos_sim = F.cosine_similarity(f_v, t_v, dim=0).item()
            loss_i = 1 - F.cosine_similarity(f_v, t_v, dim=0)
            step_loss = step_loss + loss_i
            step_cosines.append(cos_sim)
            valid_count += 1

        if valid_count == 0:
            continue

        optimizer.zero_grad()
        avg_loss = step_loss / valid_count
        avg_loss.backward()

        grad_norm = model._person_feature.grad.norm().item()
        if not np.isfinite(grad_norm):
            print(f"    NaN/Inf grad at step {step}!")
            break

        optimizer.step()

        # Record metrics
        step_time = time.time() - step_start_time
        pooled_norm = model._person_feature.norm(dim=1).mean().item()
        avg_cosine = np.mean(step_cosines) if step_cosines else 0

        record = {
            'step': step,
            'scheme': scheme_name,
            'loss': avg_loss.item(),
            'teacher_cosine': avg_cosine,
            'grad_norm_person_feature': grad_norm,
            'pooled_feature_norm': pooled_norm,
            'valid_sample_count': valid_count,
            'camera': cam_id,
            'runtime_sec_per_step': step_time,
            'cuda_memory_mb': torch.cuda.max_memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0,
        }
        metrics_records.append(record)

        step_start_time = time.time()

        if step % log_interval == 0 or step == steps - 1:
            print(f"    Step {step:5d}: loss={avg_loss.item():.4f}, "
                  f"cosine={avg_cosine:.4f}, grad_norm={grad_norm:.6f}, "
                  f"valid={valid_count}")

        # Check for hang
        if step_time > 60:
            print(f"    Possible hang at step {step} (took {step_time:.1f}s)!")
            hang_detected = True
            break

    # Summary
    if metrics_records:
        summary = {
            'scheme': scheme_name,
            'steps_completed': len(metrics_records),
            'loss_start': metrics_records[0]['loss'],
            'loss_end': metrics_records[-1]['loss'],
            'cosine_start': metrics_records[0]['teacher_cosine'],
            'cosine_end': metrics_records[-1]['teacher_cosine'],
            'grad_norm_mean': float(np.mean([r['grad_norm_person_feature'] for r in metrics_records])),
            'runtime_mean': float(np.mean([r['runtime_sec_per_step'] for r in metrics_records])),
            'hang_detected': hang_detected,
            'oom_detected': oom_detected,
            'status': 'PASS' if not hang_detected and not oom_detected else 'FAIL',
        }

        if len(metrics_records) >= 10:
            n = len(metrics_records)
            early_loss = np.mean([r['loss'] for r in metrics_records[:n//5]])
            late_loss = np.mean([r['loss'] for r in metrics_records[-n//5:]])
            early_cos = np.mean([r['teacher_cosine'] for r in metrics_records[:n//5]])
            late_cos = np.mean([r['teacher_cosine'] for r in metrics_records[-n//5:]])
            summary['loss_decreasing'] = late_loss < early_loss
            summary['cosine_increasing'] = late_cos > early_cos

            print(f"    Loss trend: {early_loss:.4f} -> {late_loss:.4f} "
                  f"({'✅' if summary['loss_decreasing'] else '⚠️'})")
            print(f"    Cosine trend: {early_cos:.4f} -> {late_cos:.4f} "
                  f"({'✅' if summary['cosine_increasing'] else '⚠️'})")
    else:
        summary = {
            'scheme': scheme_name,
            'steps_completed': 0,
            'status': 'FAIL',
            'hang_detected': hang_detected,
            'oom_detected': oom_detected,
        }

    print(f"    {scheme_name} {steps}-step: {summary['status']} "
          f"(completed {summary['steps_completed']} steps)")

    return summary, metrics_records


def generate_final_report(readout_results, smoke50_results, train200_results, output_dir):
    """Step 6: Generate final comparison report."""
    report_path = os.path.join(output_dir, "final_report.md")

    with open(report_path, 'w') as f:
        f.write("# Phase 13: Person Feature Alignment Comparison Report\n\n")

        f.write("## 1. Configuration\n\n")
        f.write("- Geometry checkpoint: `runs/Wildtrack-2802_161501/ckpt_last.pt`\n")
        f.write("- Person feature checkpoint: `outputs/phase12c_gaussianset_clean_random_fixed_eval_lr1e3_v2/checkpoint_latest.pt`\n")
        f.write("- Config: `configs/apps/wildtrack_full_3dgut.yaml`\n")
        f.write("- Dataset: `/data02/zhangrunxiang/data/Wildtrack`\n")
        f.write("- Geometry N: 52,493\n")
        f.write("- Person feature dim: 512\n")
        f.write("- Learning rate: 1e-3\n")
        f.write("- Optimizer: Adam (person_feature only)\n")
        f.write("- Geometry frozen: positions, rotation, scale, density, features frozen\n\n")

        f.write("## 2. Initialization Schemes\n\n")
        f.write("| Scheme | Description | PF[0:50000] | PF[50000:52493] |\n")
        f.write("|--------|-------------|-------------|----------------|\n")
        f.write("| old_pf_zeropad | Zero-pad old PF | Phase12c old PF | Zeros |\n")
        f.write("| reinit_pf | Random init | Random * 0.01 | Random * 0.01 |\n\n")

        f.write("## 3. No-Training Readout Comparison\n\n")
        f.write("| Scheme | Valid Count | PF Norm Mean | Teacher Norm Mean | Cosine Mean | Cosine Median | Zero Feature Ratio |\n")
        f.write("|--------|-------------|--------------|-------------------|-------------|---------------|-------------------|\n")
        for s in readout_results.values():
            f.write(f"| {s['scheme']} | {s['valid_sample_count']} | "
                    f"{s['pooled_feature_norm_mean']:.4f} | "
                    f"{s['teacher_feature_norm_mean']:.4f} | "
                    f"{s['cosine_to_teacher_mean']:.4f} | "
                    f"{s['cosine_to_teacher_median']:.4f} | "
                    f"{s['zero_feature_ratio']:.4f} |\n")
        f.write("\n")

        f.write("### Per-Camera Cosine (No Training)\n\n")
        for s in readout_results.values():
            f.write(f"**{s['scheme']}**:\n")
            for cam_id, cos_val in sorted(s.get('cosine_to_teacher_per_camera', {}).items()):
                f.write(f"  - {cam_id}: {cos_val:.4f}\n")
            f.write("\n")

        f.write("## 4. 50-Step Smoke Test Comparison\n\n")
        f.write("| Scheme | Steps Completed | Loss Start | Loss End | Cosine Start | Cosine End | Grad Norm Mean | Runtime Mean | Status |\n")
        f.write("|--------|-----------------|------------|----------|--------------|------------|----------------|--------------|--------|\n")
        for s in smoke50_results.values():
            f.write(f"| {s['scheme']} | {s.get('steps_completed', 0)} | "
                    f"{s.get('loss_start', 0):.4f} | {s.get('loss_end', 0):.4f} | "
                    f"{s.get('cosine_start', 0):.4f} | {s.get('cosine_end', 0):.4f} | "
                    f"{s.get('grad_norm_mean', 0):.6f} | "
                    f"{s.get('runtime_mean', 0):.4f} | {s.get('status', 'N/A')} |\n")
        f.write("\n")

        f.write("## 5. 200-Step Results\n\n")
        if train200_results:
            f.write("| Scheme | Steps Completed | Loss Start | Loss End | Cosine Start | Cosine End | Grad Norm Mean | Status |\n")
            f.write("|--------|-----------------|------------|----------|--------------|------------|----------------|--------|\n")
            for s in train200_results.values():
                f.write(f"| {s['scheme']} | {s.get('steps_completed', 0)} | "
                        f"{s.get('loss_start', 0):.4f} | {s.get('loss_end', 0):.4f} | "
                        f"{s.get('cosine_start', 0):.4f} | {s.get('cosine_end', 0):.4f} | "
                        f"{s.get('grad_norm_mean', 0):.6f} | {s.get('status', 'N/A')} |\n")
        else:
            f.write("No 200-step results (no scheme passed 50-step).\n")
        f.write("\n")

        f.write("## 6. Hang/OOM Detection\n\n")
        hang_oom_found = False
        for results_dict in [smoke50_results, train200_results]:
            for s in results_dict.values():
                if s.get('hang_detected') or s.get('oom_detected'):
                    f.write(f"- **{s['scheme']}**: hang={s.get('hang_detected')}, oom={s.get('oom_detected')}\n")
                    hang_oom_found = True
        if not hang_oom_found:
            f.write("No hang or OOM detected.\n")
        f.write("\n")

        f.write("## 7. Analysis & Conclusion\n\n")

        # Compare results
        old_pf_readout = readout_results.get('old_pf_zeropad', {})
        reinit_readout = readout_results.get('reinit_pf', {})

        old_pf_cos = old_pf_readout.get('cosine_to_teacher_mean', 0)
        reinit_cos = reinit_readout.get('cosine_to_teacher_mean', 0)

        old_pf_50 = smoke50_results.get('old_pf_zeropad', {})
        reinit_50 = smoke50_results.get('reinit_pf', {})

        f.write("### No-Training Readout Comparison\n")
        f.write(f"- old_pf_zeropad cosine: {old_pf_cos:.4f}\n")
        f.write(f"- reinit_pf cosine: {reinit_cos:.4f}\n")
        diff_pct = abs(old_pf_cos - reinit_cos) / max(abs(reinit_cos), 0.001)
        if diff_pct < 0.1:
            f.write(f"- **Similar** (diff={diff_pct:.1%}) - old PF has no clear advantage over random init.\n\n")
        elif old_pf_cos > reinit_cos:
            f.write(f"- **old_pf_zeropad higher by {diff_pct:.1%}** - old PF may still have some semantic value.\n\n")
        else:
            f.write(f"- **reinit_pf higher by {diff_pct:.1%}** - old PF may be misaligned or causing negative transfer.\n\n")

        f.write("### 50-Step Smoke Test Comparison\n")
        f.write(f"- old_pf_zeropad: status={old_pf_50.get('status')}, "
                f"cosine_end={old_pf_50.get('cosine_end', 0):.4f}, "
                f"loss_decreasing={old_pf_50.get('loss_decreasing', 'N/A')}, "
                f"cosine_increasing={old_pf_50.get('cosine_increasing', 'N/A')}\n")
        f.write(f"- reinit_pf: status={reinit_50.get('status')}, "
                f"cosine_end={reinit_50.get('cosine_end', 0):.4f}, "
                f"loss_decreasing={reinit_50.get('loss_decreasing', 'N/A')}, "
                f"cosine_increasing={reinit_50.get('cosine_increasing', 'N/A')}\n\n")

        f.write("### 200-Step Results\n")
        if train200_results:
            for s in train200_results.values():
                f.write(f"- {s['scheme']}: status={s.get('status')}, "
                        f"cosine_end={s.get('cosine_end', 0):.4f}\n")
            f.write("\n")

        f.write("### Final Decision\n\n")

        # Determine decision
        if old_pf_50.get('status') != 'PASS' and reinit_50.get('status') != 'PASS':
            f.write("**Decision D**: Both schemes failed to train.\n")
            f.write("Problem is not PF initialization. Investigate person_feature_map rendering gradients, "
                    "ROI pooling differentiability, teacher targets, optimizer, label/batch construction.\n")
        elif old_pf_50.get('status') == 'PASS' and reinit_50.get('status') == 'PASS':
            old_end = old_pf_50.get('cosine_end', 0)
            reinit_end = reinit_50.get('cosine_end', 0)
            if abs(old_end - reinit_end) / max(abs(reinit_end), 0.001) < 0.05:
                f.write("**Decision B**: old_pf_zeropad similar to reinit_pf.\n")
                f.write("Old PF has no clear advantage. Use reinit as baseline, continue investigating "
                        "alignment. Do not use zero-pad as primary experiment.\n")
            elif old_end > reinit_end:
                f.write("**Decision A**: old_pf_zeropad outperforms reinit_pf.\n")
                f.write("Old PF still has semantic value. Consider old PF + better new point initialization "
                        "(KNN/parent-init), then run teacher-only 500 steps and CE small overfit.\n")
            else:
                f.write("**Decision C**: reinit_pf outperforms old_pf_zeropad.\n")
                f.write("Old PF may be index-misaligned or causing negative transfer. Use reinit as primary "
                        "scheme or find matching checkpoint.\n")
        else:
            f.write("**Decision D**: One or both schemes failed.\n")
            f.write(f"- old_pf_zeropad: {old_pf_50.get('status')}\n")
            f.write(f"- reinit_pf: {reinit_50.get('status')}\n")

        f.write("\n## 8. Next Steps\n\n")
        if old_pf_50.get('status') == 'PASS' and reinit_50.get('status') == 'PASS':
            if old_pf_50.get('cosine_end', 0) > reinit_50.get('cosine_end', 0):
                f.write("1. Run teacher-only 500-step with old_pf_zeropad\n")
                f.write("2. Run CE small overfit sanity\n")
                f.write("3. Consider KNN expansion for new 2,493 Gaussians\n")
            else:
                f.write("1. Use reinit_pf as baseline\n")
                f.write("2. Investigate why old PF doesn't help (index misalignment?)\n")
                f.write("3. Search for matching N=50,000 geometry checkpoint\n")
                f.write("4. Consider KNN expansion or parent-init for new Gaussians\n")
        else:
            f.write("1. Debug training issues\n")
            f.write("2. Check person_feature_map gradient flow\n")
            f.write("3. Verify ROI pooling differentiability\n")

    print(f"\nGenerated: {report_path}")

    # Generate summary CSV
    summary_csv_path = os.path.join(output_dir, "train_compare_summary.csv")
    with open(summary_csv_path, 'w', newline='') as f:
        fieldnames = ['scheme', 'init_cosine', 'final_cosine_50', 'loss_start', 'loss_50',
                      'loss_200', 'final_cosine_200', 'grad_norm_mean', 'runtime_mean', 'status', 'verdict']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for scheme_name in smoke50_results:
            smoke = smoke50_results[scheme_name]
            train = train200_results.get(scheme_name, {})
            readout = readout_results.get(scheme_name, {})
            writer.writerow({
                'scheme': scheme_name,
                'init_cosine': readout.get('cosine_to_teacher_mean', 0),
                'final_cosine_50': smoke.get('cosine_end', 0),
                'loss_start': smoke.get('loss_start', 0),
                'loss_50': smoke.get('loss_end', 0),
                'loss_200': train.get('loss_end', 0),
                'final_cosine_200': train.get('cosine_end', 0),
                'grad_norm_mean': smoke.get('grad_norm_mean', 0),
                'runtime_mean': smoke.get('runtime_mean', 0),
                'status': smoke.get('status', 'N/A'),
                'verdict': smoke.get('status', 'N/A'),
            })

    print(f"Generated: {summary_csv_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("Phase 13: Person Feature Alignment Comparison (Real Model)")
    print("="*80)

    # Load dataset first
    print("\nLoading dataset...")
    dataset = WildtrackDataset(
        dataset_path=args.dataset_path, split="train",
        downsample_factor=4, load_teacher_cache=True,
    )
    print(f"Dataset: {len(dataset)} samples, render: {dataset.img_width}x{dataset.img_height}")

    # ==========================================
    # Step 3: No-training readout diagnostic
    # ==========================================
    print("\n" + "="*60)
    print("Step 3: No-Training Readout Diagnostic")
    print("="*60)

    readout_results = {}

    # Scheme A: zero-pad old PF
    def zero_pad_init(n_geom, device):
        return zero_pad_old_pf(n_geom, device, args.person_feature_checkpoint)

    model_a, n_geom = load_geometry_model(args.geometry_checkpoint, zero_pad_init, args.device)
    readout_a = run_readout_diagnostic(model_a, dataset, args.device, "old_pf_zeropad", args.output_dir)
    readout_results["old_pf_zeropad"] = readout_a

    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    # Scheme B: reinit PF
    def reinit_init(n_geom, device):
        return reinit_pf(n_geom, device)

    model_b, n_geom = load_geometry_model(args.geometry_checkpoint, reinit_init, args.device)
    readout_b = run_readout_diagnostic(model_b, dataset, args.device, "reinit_pf", args.output_dir)
    readout_results["reinit_pf"] = readout_b

    del model_b
    gc.collect()
    torch.cuda.empty_cache()

    # Save readout results
    csv_path = os.path.join(args.output_dir, "teacher_readout_before_training.csv")
    json_path = os.path.join(args.output_dir, "teacher_readout_before_training.json")

    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['scheme', 'valid_sample_count', 'pooled_feature_norm_mean',
                      'pooled_feature_norm_median', 'teacher_feature_norm_mean',
                      'teacher_feature_norm_median', 'cosine_to_teacher_mean',
                      'cosine_to_teacher_median', 'zero_feature_ratio']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for s in readout_results.values():
            writer.writerow(s)

    with open(json_path, 'w') as f:
        json.dump(readout_results, f, indent=2)

    # ==========================================
    # Step 4: 50-step smoke test
    # ==========================================
    print("\n" + "="*60)
    print("Step 4: 50-Step Smoke Test")
    print("="*60)

    smoke50_results = {}
    smoke50_records = {}

    # Scheme A
    model_a, _ = load_geometry_model(args.geometry_checkpoint, zero_pad_init, args.device)
    smoke_a, records_a = run_training_test(model_a, dataset, args.device, args.smoke_steps, "old_pf_zeropad", lr=1e-3)
    smoke50_results["old_pf_zeropad"] = smoke_a
    smoke50_records["old_pf_zeropad"] = records_a

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "smoke50_old_pf_zeropad_metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for rec in records_a:
            f.write(json.dumps(rec) + '\n')

    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    # Scheme B
    model_b, _ = load_geometry_model(args.geometry_checkpoint, reinit_init, args.device)
    smoke_b, records_b = run_training_test(model_b, dataset, args.device, args.smoke_steps, "reinit_pf", lr=1e-3)
    smoke50_results["reinit_pf"] = smoke_b
    smoke50_records["reinit_pf"] = records_b

    metrics_path = os.path.join(args.output_dir, "smoke50_reinit_pf_metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for rec in records_b:
            f.write(json.dumps(rec) + '\n')

    del model_b
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # Step 5: 200-step for PASS schemes
    # ==========================================
    print("\n" + "="*60)
    print("Step 5: 200-Step Extended Training")
    print("="*60)

    train200_results = {}
    train200_records = {}

    for scheme_name in ["old_pf_zeropad", "reinit_pf"]:
        smoke = smoke50_results.get(scheme_name, {})
        if smoke.get('status') == 'PASS':
            print(f"\n  Running 200-step for {scheme_name}...")
            if scheme_name == "old_pf_zeropad":
                model, _ = load_geometry_model(args.geometry_checkpoint, zero_pad_init, args.device)
            else:
                model, _ = load_geometry_model(args.geometry_checkpoint, reinit_init, args.device)

            train, records = run_training_test(model, dataset, args.device, args.train_steps, scheme_name, lr=1e-3, log_interval=20)
            train200_results[scheme_name] = train
            train200_records[scheme_name] = records

            metrics_path = os.path.join(args.output_dir, f"train200_{scheme_name}_metrics.jsonl")
            with open(metrics_path, 'w') as f:
                for rec in records:
                    f.write(json.dumps(rec) + '\n')

            del model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"  Skipping 200-step for {scheme_name} (50-step failed)")

    # ==========================================
    # Step 6: Generate final report
    # ==========================================
    print("\n" + "="*60)
    print("Step 6: Final Report")
    print("="*60)

    generate_final_report(readout_results, smoke50_results, train200_results, args.output_dir)

    print("\n" + "="*80)
    print("Phase 13 PF Alignment Comparison Complete!")
    print(f"Results: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
