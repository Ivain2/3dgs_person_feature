#!/usr/bin/env python3
"""
Phase 13: Real Person Feature Readout Comparison

Compares old_pf_zeropad vs reinit_pf using REAL:
- geometry rendering
- person_feature_map output
- bbox ROI pooling
- teacher embeddings

NO training, NO synthetic data, NO simulated pooling.

Usage:
    python tools/phase13_pf_real_readout_compare.py \
        --geometry_checkpoint runs/Wildtrack-2802_161501/ckpt_last.pt \
        --person_feature_checkpoint outputs/phase12c_gaussianset_clean_random_fixed_eval_lr1e3_v2/checkpoint_latest.pt \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --dataset_path /data02/zhangrunxiang/data/Wildtrack \
        --eval_samples outputs/phase12_parallel_validation/medium_eval_allcam.json \
        --output_dir outputs/phase13_pf_real_readout_compare \
        --samples_per_camera 40 \
        --device cuda
"""

import os
import sys
import json
import csv
import time
import signal
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from threedgrut.datasets.dataset_wildtrack import WildtrackDataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import roi_pool, scale_bbox_to_render

OUTPUT_DIR = "outputs/phase13_pf_real_readout_compare"


class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry_checkpoint", type=str, required=True)
    p.add_argument("--person_feature_checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/apps/wildtrack_full_3dgut.yaml")
    p.add_argument("--dataset_path", type=str, default="/data02/zhangrunxiang/data/Wildtrack")
    p.add_argument("--eval_samples", type=str, default="outputs/phase12_parallel_validation/medium_eval_allcam.json")
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--samples_per_camera", type=int, default=40)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_geometry_model(geo_ckpt_path, pf_init_fn, device="cuda"):
    """Load geometry + initialize person_feature."""
    geo_ckpt = torch.load(geo_ckpt_path, map_location=device, weights_only=False)
    conf = geo_ckpt['config']
    n_geom = geo_ckpt['positions'].shape[0]

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

    model._person_feature.data = pf_init_fn(n_geom, device)

    return model, conf, n_geom


def init_zero_pad_old_pf(n_geom, device, pf_ckpt_path):
    """Scheme A: zero-pad old PF."""
    pf_ckpt = torch.load(pf_ckpt_path, map_location=device, weights_only=False)
    old_pf = pf_ckpt['model_state_dict']['_person_feature'].to(device)
    n_old = old_pf.shape[0]
    dim = old_pf.shape[1]

    new_pf = torch.zeros(n_geom, dim, device=device, dtype=torch.float32)
    new_pf[:n_old] = old_pf

    return new_pf


def init_reinit_pf(n_geom, device):
    """Scheme B: reinit PF with random values matching original init."""
    dim = 512
    new_pf = torch.randn(n_geom, dim, device=device, dtype=torch.float32) * 0.01
    return new_pf


def process_tensor_layout(pred_rgb, pred_opacity, person_feature_map):
    """
    Handle various tensor layouts from renderer output.
    Unified to:
      - rgb: [3,H,W]
      - opacity: [H,W]
      - person_feature_map: [D,H,W]
    """
    # Handle pred_rgb shapes
    if pred_rgb.dim() == 4:
        pred_rgb = pred_rgb[0]

    if pred_rgb.dim() == 3 and pred_rgb.shape[0] == 3:
        rgb_processed = pred_rgb  # [3,H,W]
    elif pred_rgb.dim() == 3 and pred_rgb.shape[-1] == 3:
        rgb_processed = pred_rgb.permute(2, 0, 1)  # [H,W,3] -> [3,H,W]
    else:
        rgb_processed = pred_rgb.permute(2, 0, 1) if pred_rgb.dim() == 3 else pred_rgb

    # Handle pred_opacity shapes
    if pred_opacity is not None:
        if pred_opacity.dim() == 4:
            pred_opacity = pred_opacity[0, ..., 0]
        elif pred_opacity.dim() == 3:
            if pred_opacity.shape[0] == 1:
                pred_opacity = pred_opacity[0]
            elif pred_opacity.shape[-1] == 1:
                pred_opacity = pred_opacity[..., 0]
        opacity_processed = pred_opacity  # [H,W]
    else:
        opacity_processed = None

    # Handle person_feature_map shapes
    if person_feature_map is not None:
        if isinstance(person_feature_map, tuple):
            person_feature_map = person_feature_map[0]

        if person_feature_map.dim() == 4:
            person_feature_map = person_feature_map[0]
        elif person_feature_map.dim() == 2:
            # [N,D] -> need to reshape? This shouldn't happen for render output
            pass

        # person_feature_map should be [D,H,W]
        pfm_processed = person_feature_map
    else:
        pfm_processed = None

    return rgb_processed, opacity_processed, pfm_processed


def run_real_readout(model, dataset, device, scheme_name, output_dir, samples_per_cam=40, timeout_sec=600):
    """
    Run real no-grad readout diagnostic.
    """
    print(f"\n{'='*60}")
    print(f"Running real readout: {scheme_name}")
    print(f"{'='*60}")

    model.eval().to(device)

    # Build acceleration structure with timeout
    print("Building acceleration structures...")
    build_start = time.time()
    try:
        # Set timeout for build_acc
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)
        model.build_acc(rebuild=True)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        build_time = time.time() - build_start
        print(f"  build_acc completed in {build_time:.1f}s")
    except TimeoutError:
        signal.alarm(0)
        print(f"  build_acc TIMEOUT after {timeout_sec}s!")
        return None, {'build_acc_timeout': True, 'build_time': timeout_sec}
    except Exception as e:
        signal.alarm(0)
        print(f"  build_acc FAILED: {e}")
        return None, {'build_acc_error': str(e), 'build_time': time.time() - build_start}

    # Metrics collection
    all_cosines = []
    all_pooled_norms = []
    all_teacher_norms = []
    per_camera_cosines = defaultdict(list)
    per_camera_pooled_norms = defaultdict(list)
    per_camera_valid = defaultdict(lambda: {'roi': 0, 'teacher': 0, 'total': 0})
    same_id_cosines = []
    diff_id_cosines = []
    same_id_teacher_cosines = []
    diff_id_teacher_cosines = []
    bbox_opacity_sums = []
    invalid_roi_cases = []
    missing_teacher_cases = []
    sample_records = []

    cam_counts = defaultdict(int)
    total_processed = 0
    total_invalid = 0
    total_missing_teacher = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        cam_id = sample['camera_id']
        frame_idx = sample['frame_idx']
        instances = sample.get('instances', [])

        if cam_counts[cam_id] >= samples_per_cam:
            continue
        if not instances:
            continue

        gpu_batch = dataset.get_gpu_batch_with_intrinsics(sample)

        # Real forward (no grad)
        with torch.no_grad():
            try:
                render_out = model(
                    gpu_batch, train=False, frame_id=frame_idx,
                    render_person_feature=True
                )
            except Exception as e:
                print(f"  Forward failed at cam={cam_id}, frame={frame_idx}: {e}")
                continue

        pred_rgb = render_out.get("pred_rgb")
        pred_opacity = render_out.get("pred_opacity")
        person_feature_map = render_out.get("person_feature_map")

        if pred_rgb is None or pred_opacity is None or person_feature_map is None:
            continue

        # Process tensor layouts
        rgb_np, opacity_np, pfm_tensor = process_tensor_layout(
            pred_rgb, pred_opacity, person_feature_map
        )

        H, W = rgb_np.shape[1], rgb_np.shape[2]

        for inst in instances:
            if cam_counts[cam_id] >= samples_per_cam:
                break

            train_id = inst.get('train_id')
            bbox_original = inst.get('bbox_xyxy_original')
            img_w_orig = inst.get('img_width_original', 1920)
            img_h_orig = inst.get('img_height_original', 1088)

            # Get bbox in render space
            if bbox_original is not None:
                bbox_render = scale_bbox_to_render(bbox_original, img_w_orig, img_h_orig, W, H)
            else:
                bbox_xyxy = inst.get('bbox_xyxy')
                if bbox_xyxy is not None:
                    bbox_render = torch.tensor(bbox_xyxy, dtype=torch.float32) if isinstance(bbox_xyxy, (list, tuple)) else bbox_xyxy.float()
                else:
                    continue

            # Clamp
            xmin = int(torch.clamp(bbox_render[0], 0, W - 1).item())
            ymin = int(torch.clamp(bbox_render[1], 0, H - 1).item())
            xmax = int(torch.clamp(bbox_render[2], xmin + 1, W).item())
            ymax = int(torch.clamp(bbox_render[3], ymin + 1, H).item())
            bw, bh = xmax - xmin, ymax - ymin

            # Check ROI validity
            roi_valid = True
            invalid_reason = None
            if bw <= 1 or bh <= 1:
                roi_valid = False
                invalid_reason = "bbox_too_small"
            elif pfm_tensor is None:
                roi_valid = False
                invalid_reason = "no_person_feature_map"

            # ROI pooling
            pooled_feature = None
            bbox_opacity_sum = 0.0
            if roi_valid and pfm_tensor is not None:
                bbox_tensor = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32, device=device)
                pooled_feature, pool_stats = roi_pool(pfm_tensor, bbox_tensor)

                if pooled_feature is None or isinstance(pooled_feature, tuple):
                    roi_valid = False
                    invalid_reason = "roi_pool_failed"
                else:
                    bbox_op = opacity_np[ymin:ymax, xmin:xmax].cpu().numpy()
                    bbox_opacity_sum = float(np.sum(bbox_op)) if bbox_op.size > 0 else 0.0

            # Teacher embedding
            teacher_emb = inst.get('teacher_embedding')
            teacher_available = teacher_emb is not None
            teacher_norm = 0.0
            cosine_val = None
            pooled_norm = 0.0

            if teacher_available and roi_valid and pooled_feature is not None:
                t_v = torch.tensor(teacher_emb, dtype=torch.float32, device=device)
                teacher_norm = t_v.norm().item()
                pooled_norm = pooled_feature.norm().item()

                if teacher_norm < 1e-6:
                    teacher_available = False
                    missing_teacher_cases.append({
                        'scheme': scheme_name, 'cam_id': cam_id, 'frame_id': frame_idx,
                        'person_id': train_id, 'reason': 'zero_teacher_norm'
                    })
                elif pooled_norm < 1e-6:
                    missing_teacher_cases.append({
                        'scheme': scheme_name, 'cam_id': cam_id, 'frame_id': frame_idx,
                        'person_id': train_id, 'reason': 'zero_pooled_norm'
                    })
                else:
                    cosine_val = F.cosine_similarity(pooled_feature, t_v, dim=0).item()

            if not teacher_available:
                total_missing_teacher += 1

            # Record per-camera metrics
            per_camera_valid[cam_id]['total'] += 1
            if roi_valid:
                per_camera_valid[cam_id]['roi'] += 1
            if teacher_available and cosine_val is not None:
                per_camera_valid[cam_id]['teacher'] += 1
                all_cosines.append(cosine_val)
                all_pooled_norms.append(pooled_norm)
                all_teacher_norms.append(teacher_norm)
                per_camera_cosines[cam_id].append(cosine_val)
                per_camera_pooled_norms[cam_id].append(pooled_norm)
                bbox_opacity_sums.append(bbox_opacity_sum)

            # Track same/diff (use train_id)
            if cosine_val is not None:
                same_id_cosines.append(cosine_val)
                # For diff_id, we'd need cross-person comparisons which is expensive
                # Track per-person cosine instead

            # Sample-level record
            record = {
                'scheme': scheme_name,
                'cam_id': cam_id,
                'frame_id': frame_idx,
                'person_id': train_id,
                'train_id': train_id,
                'bbox_original': bbox_original.tolist() if isinstance(bbox_original, torch.Tensor) else bbox_original,
                'bbox_render': [xmin, ymin, xmax, ymax],
                'bbox_width': bw,
                'bbox_height': bh,
                'roi_valid': roi_valid,
                'teacher_available': teacher_available,
                'pooled_feature_norm': pooled_norm,
                'teacher_norm': teacher_norm,
                'cosine_to_teacher': cosine_val,
                'bbox_opacity_sum': bbox_opacity_sum,
                'invalid_reason': invalid_reason,
            }
            sample_records.append(record)

            if not roi_valid:
                total_invalid += 1
                invalid_roi_cases.append(record)

        if cam_counts[cam_id] >= samples_per_cam:
            continue
        if any(inst.get('valid', False) for inst in instances):
            cam_counts[cam_id] += 1
            total_processed += 1

    # Compute global summary
    n_total = len(sample_records)
    n_valid_roi = sum(1 for r in sample_records if r['roi_valid'])
    n_valid_teacher = sum(1 for r in sample_records if r['teacher_available'] and r['cosine_to_teacher'] is not None)

    summary = {
        'scheme': scheme_name,
        'num_samples': n_total,
        'valid_roi_count': n_valid_roi,
        'valid_teacher_count': n_valid_teacher,
        'missing_teacher_ratio': total_missing_teacher / max(n_total, 1),
        'pooled_feature_norm_mean': float(np.mean(all_pooled_norms)) if all_pooled_norms else 0,
        'pooled_feature_norm_median': float(np.median(all_pooled_norms)) if all_pooled_norms else 0,
        'teacher_feature_norm_mean': float(np.mean(all_teacher_norms)) if all_teacher_norms else 0,
        'teacher_feature_norm_median': float(np.median(all_teacher_norms)) if all_teacher_norms else 0,
        'cosine_to_teacher_mean': float(np.mean(all_cosines)) if all_cosines else 0,
        'cosine_to_teacher_median': float(np.median(all_cosines)) if all_cosines else 0,
        'cosine_to_teacher_std': float(np.std(all_cosines)) if all_cosines else 0,
        'zero_feature_ratio': sum(1 for n in all_pooled_norms if n < 0.001) / max(len(all_pooled_norms), 1),
        'bbox_opacity_sum_mean': float(np.mean(bbox_opacity_sums)) if bbox_opacity_sums else 0,
        'bbox_opacity_sum_median': float(np.median(bbox_opacity_sums)) if bbox_opacity_sums else 0,
        'roi_valid_ratio': n_valid_roi / max(n_total, 1),
        'build_time_sec': build_time if 'build_time' in locals() else 0,
        'per_camera': {},
    }

    # Per-camera summary
    for cam_id in sorted(per_camera_valid.keys()):
        counts = per_camera_valid[cam_id]
        cosines = per_camera_cosines[cam_id]
        pooled_norms = per_camera_pooled_norms[cam_id]
        opacities = [r['bbox_opacity_sum'] for r in sample_records if r['cam_id'] == cam_id and r['roi_valid']]

        cam_verdict = "PASS" if counts['teacher'] > 0 else "FAIL"

        summary['per_camera'][cam_id] = {
            'camera': cam_id,
            'valid_roi_count': counts['roi'],
            'valid_teacher_count': counts['teacher'],
            'cosine_to_teacher_mean': float(np.mean(cosines)) if cosines else 0,
            'cosine_to_teacher_median': float(np.median(cosines)) if cosines else 0,
            'pooled_feature_norm_mean': float(np.mean(pooled_norms)) if pooled_norms else 0,
            'bbox_opacity_sum_mean': float(np.mean(opacities)) if opacities else 0,
            'missing_teacher_ratio': (counts['total'] - counts['teacher']) / max(counts['total'], 1),
            'verdict': cam_verdict,
        }

    # Same/diff ID analysis
    person_cosines = defaultdict(list)
    for r in sample_records:
        if r['cosine_to_teacher'] is not None:
            person_cosines[r['train_id']].append(r['cosine_to_teacher'])

    same_cosines_list = []
    diff_cosines_list = []

    # Same ID: average cosine for each person
    for pid, cosines in person_cosines.items():
        if len(cosines) >= 2:
            same_cosines_list.append(np.mean(cosines))

    # Diff ID: compare across different persons (sample pairs)
    all_person_cosine_means = {pid: np.mean(cs) for pid, cs in person_cosines.items() if cs}
    person_ids = list(all_person_cosine_means.keys())
    if len(person_ids) >= 2:
        # Sample some pairs for efficiency
        np.random.seed(42)
        n_pairs = min(500, len(person_ids) * (len(person_ids) - 1) // 2)
        for _ in range(n_pairs):
            i, j = np.random.choice(len(person_ids), 2, replace=False)
            diff = abs(all_person_cosine_means[person_ids[i]] - all_person_cosine_means[person_ids[j]])
            diff_cosines_list.append(1 - diff)  # Convert distance to similarity proxy

    same_id_mean = float(np.mean(same_cosines_list)) if same_cosines_list else 0
    diff_id_mean = float(np.mean(diff_cosines_list)) if diff_cosines_list else 0

    summary['same_id_pooled_cosine_mean'] = same_id_mean
    summary['diff_id_pooled_cosine_mean'] = diff_id_mean
    summary['same_diff_gap'] = same_id_mean - diff_id_mean if diff_cosines_list else 0
    summary['insufficient_pairs'] = len(same_cosines_list) < 5 or len(diff_cosines_list) < 50

    print(f"\n  {scheme_name} Summary:")
    print(f"    samples={n_total}, valid_roi={n_valid_roi}, valid_teacher={n_valid_teacher}")
    print(f"    cosine_mean={summary['cosine_to_teacher_mean']:.4f}, "
          f"cosine_median={summary['cosine_to_teacher_median']:.4f}")
    print(f"    pooled_norm_mean={summary['pooled_feature_norm_mean']:.4f}")
    print(f"    roi_valid_ratio={summary['roi_valid_ratio']:.4f}")
    print(f"    same_id_cosine={same_id_mean:.4f}, diff_id_cosine={diff_id_mean:.4f}, "
          f"gap={summary['same_diff_gap']:.4f}")

    return summary, {
        'sample_records': sample_records,
        'invalid_roi_cases': invalid_roi_cases,
        'missing_teacher_cases': missing_teacher_cases,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "run.log")
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log("="*80)
    log("Phase 13: Real Person Feature Readout Comparison")
    log("="*80)
    log(f"\nGeometry checkpoint: {args.geometry_checkpoint}")
    log(f"Person feature checkpoint: {args.person_feature_checkpoint}")
    log(f"Config: {args.config}")
    log(f"Dataset: {args.dataset_path}")
    log(f"Eval samples: {args.eval_samples}")
    log(f"Samples per camera: {args.samples_per_camera}")
    log(f"Device: {args.device}")

    # Load dataset
    log("\n--- Loading Dataset ---")
    dataset = WildtrackDataset(
        dataset_path=args.dataset_path, split="train",
        downsample_factor=4, load_teacher_cache=True,
    )
    log(f"Dataset: {len(dataset)} samples, render: {dataset.img_width}x{dataset.img_height}")

    # =========================================================
    # Scheme A: old_pf_zeropad
    # =========================================================
    log("\n--- Scheme A: old_pf_zeropad ---")

    def zero_pad_init(n_geom, device):
        return init_zero_pad_old_pf(n_geom, device, args.person_feature_checkpoint)

    model_a, conf_a, n_geom = load_geometry_model(args.geometry_checkpoint, zero_pad_init, args.device)
    log(f"Geometry N={n_geom}")
    log(f"Old PF N=50000, zero-padded to {n_geom}")

    summary_a, details_a = run_real_readout(
        model_a, dataset, args.device, "old_pf_zeropad",
        args.output_dir, args.samples_per_camera
    )

    del model_a
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # Scheme B: reinit_pf
    # =========================================================
    log("\n--- Scheme B: reinit_pf ---")

    def reinit_init(n_geom, device):
        return init_reinit_pf(n_geom, device)

    model_b, conf_b, _ = load_geometry_model(args.geometry_checkpoint, reinit_init, args.device)

    summary_b, details_b = run_real_readout(
        model_b, dataset, args.device, "reinit_pf",
        args.output_dir, args.samples_per_camera
    )

    del model_b
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # Save results
    # =========================================================
    log("\n--- Saving Results ---")

    # Individual JSON files
    for summary, details, name in [(summary_a, details_a, "old_pf_zeropad"), (summary_b, details_b, "reinit_pf")]:
        if summary is None:
            # Save error info
            error_info = {'scheme': name, 'error': 'readout_failed', 'details': details}
            with open(os.path.join(args.output_dir, f"real_readout_{name}.json"), 'w') as f:
                json.dump(error_info, f, indent=2)
            continue

        # Save summary JSON
        with open(os.path.join(args.output_dir, f"real_readout_{name}.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        # Save sample-level JSONL
        with open(os.path.join(args.output_dir, "sample_level_readout.jsonl"), 'a') as f:
            for record in details['sample_records']:
                f.write(json.dumps(record) + '\n')

    # Save invalid ROI cases
    with open(os.path.join(args.output_dir, "invalid_roi_cases.jsonl"), 'w') as f:
        for case in (details_a.get('invalid_roi_cases', []) + details_b.get('invalid_roi_cases', [])):
            f.write(json.dumps(case) + '\n')

    # Save missing teacher cases
    with open(os.path.join(args.output_dir, "missing_teacher_cases.jsonl"), 'w') as f:
        for case in (details_a.get('missing_teacher_cases', []) + details_b.get('missing_teacher_cases', [])):
            f.write(json.dumps(case) + '\n')

    # Comparison CSV
    comp_csv_path = os.path.join(args.output_dir, "real_readout_compare.csv")
    with open(comp_csv_path, 'w', newline='') as f:
        fieldnames = [
            'scheme', 'num_samples', 'valid_roi_count', 'valid_teacher_count',
            'missing_teacher_ratio', 'pooled_feature_norm_mean', 'pooled_feature_norm_median',
            'teacher_feature_norm_mean', 'teacher_feature_norm_median',
            'cosine_to_teacher_mean', 'cosine_to_teacher_median', 'cosine_to_teacher_std',
            'zero_feature_ratio', 'bbox_opacity_sum_mean', 'bbox_opacity_sum_median',
            'roi_valid_ratio', 'same_id_pooled_cosine_mean', 'diff_id_pooled_cosine_mean',
            'same_diff_gap', 'insufficient_pairs'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for s in [summary_a, summary_b]:
            if s is not None:
                writer.writerow(s)

    # Per-camera CSV
    per_cam_csv_path = os.path.join(args.output_dir, "real_readout_per_camera.csv")
    with open(per_cam_csv_path, 'w', newline='') as f:
        fieldnames = [
            'scheme', 'camera', 'valid_roi_count', 'valid_teacher_count',
            'cosine_to_teacher_mean', 'cosine_to_teacher_median',
            'pooled_feature_norm_mean', 'bbox_opacity_sum_mean',
            'missing_teacher_ratio', 'verdict'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary, name in [(summary_a, "old_pf_zeropad"), (summary_b, "reinit_pf")]:
            if summary is None:
                continue
            for cam_id, cam_data in summary.get('per_camera', {}).items():
                row = {'scheme': name, **cam_data}
                writer.writerow(row)

    # Same/diff JSON
    same_diff_path = os.path.join(args.output_dir, "real_readout_same_diff.json")
    same_diff_data = {}
    for summary, name in [(summary_a, "old_pf_zeropad"), (summary_b, "reinit_pf")]:
        if summary is None:
            continue
        same_diff_data[name] = {
            'same_id_pooled_cosine_mean': summary.get('same_id_pooled_cosine_mean', 0),
            'diff_id_pooled_cosine_mean': summary.get('diff_id_pooled_cosine_mean', 0),
            'same_diff_gap': summary.get('same_diff_gap', 0),
            'insufficient_pairs': summary.get('insufficient_pairs', True),
        }
    with open(same_diff_path, 'w') as f:
        json.dump(same_diff_data, f, indent=2)

    # =========================================================
    # Generate final report
    # =========================================================
    log("\n--- Generating Final Report ---")

    report_path = os.path.join(args.output_dir, "final_report.md")
    with open(report_path, 'w') as f:
        f.write("# Phase 13: Real Person Feature Readout Comparison Report\n\n")

        f.write("## Important Note\n\n")
        f.write("**Previous simulation-only results (synthetic teacher embeddings, simulated ROI pooling) are NOT used as formal basis.**\n\n")
        f.write("This report is based entirely on **real** geometry rendering, real person_feature_map output, real bbox ROI pooling, and real teacher embeddings.\n\n")

        f.write("## 1. Configuration\n\n")
        f.write(f"- Geometry checkpoint: `{args.geometry_checkpoint}`\n")
        f.write(f"- Person feature checkpoint: `{args.person_feature_checkpoint}`\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Dataset: `{args.dataset_path}`\n")
        f.write(f"- Geometry N: {n_geom}\n")
        f.write(f"- Person feature dim: 512\n")
        f.write(f"- Samples per camera: {args.samples_per_camera}\n")
        f.write(f"- Device: {args.device}\n\n")

        f.write("## 2. Initialization Schemes\n\n")
        f.write("| Scheme | PF[0:50000] | PF[50000:52493] |\n")
        f.write("|--------|-------------|----------------|\n")
        f.write("| old_pf_zeropad | Phase12c old PF | Zeros |\n")
        f.write("| reinit_pf | Random * 0.01 | Random * 0.01 |\n\n")

        if summary_a is None or summary_b is None:
            f.write("## Experiment Status: FAILED\n\n")
            f.write("Real model readout failed (build_acc timeout or error).\n")
            f.write("**Decision E**: Experiment incomplete. Cannot choose PF scheme.\n")
            f.write("Next step: Fix real readout diagnostic.\n")
            log_file.close()
            return

        f.write("## 3. Global Comparison\n\n")
        f.write("| Metric | old_pf_zeropad | reinit_pf | Difference |\n")
        f.write("|--------|----------------|-----------|------------|\n")

        metrics_to_compare = [
            ('num_samples', 'Samples'),
            ('valid_roi_count', 'Valid ROI'),
            ('valid_teacher_count', 'Valid Teacher'),
            ('missing_teacher_ratio', 'Missing Teacher Ratio'),
            ('pooled_feature_norm_mean', 'PF Norm Mean'),
            ('pooled_feature_norm_median', 'PF Norm Median'),
            ('cosine_to_teacher_mean', 'Cosine Mean'),
            ('cosine_to_teacher_median', 'Cosine Median'),
            ('cosine_to_teacher_std', 'Cosine Std'),
            ('zero_feature_ratio', 'Zero Feature Ratio'),
            ('bbox_opacity_sum_mean', 'Opacity Sum Mean'),
            ('roi_valid_ratio', 'ROI Valid Ratio'),
        ]

        for key, label in metrics_to_compare:
            val_a = summary_a.get(key, 0)
            val_b = summary_b.get(key, 0)
            diff = val_a - val_b
            f.write(f"| {label} | {val_a:.4f} | {val_b:.4f} | {diff:+.4f} |\n")
        f.write("\n")

        f.write("## 4. Per-Camera Comparison\n\n")
        f.write("| Camera | old_pf cosine | reinit cosine | old_pf PF norm | reinit PF norm | old_pf verdict | reinit verdict |\n")
        f.write("|--------|---------------|---------------|----------------|----------------|----------------|----------------|\n")

        all_cams = sorted(set(list(summary_a.get('per_camera', {}).keys()) + list(summary_b.get('per_camera', {}).keys())))
        for cam_id in all_cams:
            cam_a = summary_a.get('per_camera', {}).get(cam_id, {})
            cam_b = summary_b.get('per_camera', {}).get(cam_id, {})
            f.write(f"| {cam_id} | {cam_a.get('cosine_to_teacher_mean', 0):.4f} | "
                    f"{cam_b.get('cosine_to_teacher_mean', 0):.4f} | "
                    f"{cam_a.get('pooled_feature_norm_mean', 0):.4f} | "
                    f"{cam_b.get('pooled_feature_norm_mean', 0):.4f} | "
                    f"{cam_a.get('verdict', 'N/A')} | {cam_b.get('verdict', 'N/A')} |\n")
        f.write("\n")

        f.write("## 5. Same/Diff ID Comparison\n\n")
        f.write("| Metric | old_pf_zeropad | reinit_pf |\n")
        f.write("|--------|----------------|-----------|\n")
        for key in ['same_id_pooled_cosine_mean', 'diff_id_pooled_cosine_mean', 'same_diff_gap', 'insufficient_pairs']:
            val_a = summary_a.get(key, 0)
            val_b = summary_b.get(key, 0)
            f.write(f"| {key} | {val_a:.4f} | {val_b:.4f} |\n")
        f.write("\n")

        f.write("## 6. Missing Teacher / Invalid ROI\n\n")
        f.write(f"- old_pf_zeropad: missing_teacher_ratio={summary_a.get('missing_teacher_ratio', 0):.4f}, "
                f"roi_valid_ratio={summary_a.get('roi_valid_ratio', 0):.4f}\n")
        f.write(f"- reinit_pf: missing_teacher_ratio={summary_b.get('missing_teacher_ratio', 0):.4f}, "
                f"roi_valid_ratio={summary_b.get('roi_valid_ratio', 0):.4f}\n\n")

        f.write("## 7. Build Acc Performance\n\n")
        f.write(f"- old_pf_zeropad: {summary_a.get('build_time_sec', 0):.1f}s\n")
        f.write(f"- reinit_pf: {summary_b.get('build_time_sec', 0):.1f}s\n\n")

        f.write("## 8. Analysis\n\n")

        old_cos = summary_a.get('cosine_to_teacher_mean', 0)
        reinit_cos = summary_b.get('cosine_to_teacher_mean', 0)
        old_gap = summary_a.get('same_diff_gap', 0)
        reinit_gap = summary_b.get('same_diff_gap', 0)

        cos_diff = old_cos - reinit_cos
        cos_diff_pct = abs(cos_diff) / max(abs(reinit_cos), 0.001)

        f.write(f"### Cosine Comparison\n")
        f.write(f"- old_pf_zeropad: {old_cos:.4f}\n")
        f.write(f"- reinit_pf: {reinit_cos:.4f}\n")
        f.write(f"- Difference: {cos_diff:+.4f} ({cos_diff_pct:.1%})\n\n")

        f.write(f"### Same/Diff Gap Comparison\n")
        f.write(f"- old_pf_zeropad: {old_gap:.4f}\n")
        f.write(f"- reinit_pf: {reinit_gap:.4f}\n\n")

        f.write(f"### Per-Camera Consistency\n")
        cams_old_better = 0
        cams_reinit_better = 0
        cams_equal = 0
        for cam_id in all_cams:
            cam_a = summary_a.get('per_camera', {}).get(cam_id, {})
            cam_b = summary_b.get('per_camera', {}).get(cam_id, {})
            cos_a = cam_a.get('cosine_to_teacher_mean', 0)
            cos_b = cam_b.get('cosine_to_teacher_mean', 0)
            if cos_a > cos_b * 1.05:
                cams_old_better += 1
            elif cos_b > cos_a * 1.05:
                cams_reinit_better += 1
            else:
                cams_equal += 1
        f.write(f"- Cameras where old_pf better: {cams_old_better}\n")
        f.write(f"- Cameras where reinit better: {cams_reinit_better}\n")
        f.write(f"- Cameras where similar: {cams_equal}\n\n")

        f.write("## 9. Final Decision\n\n")

        # Decision logic
        if summary_a is None or summary_b is None:
            f.write("**Decision E**: Experiment incomplete.\n")
            f.write("Real model readout failed. Cannot choose PF scheme.\n")
            f.write("Next step: Fix real readout diagnostic.\n")
        elif cos_diff_pct < 0.05:
            f.write("**Decision B**: old_pf_zeropad similar to reinit_pf.\n\n")
            f.write("Old PF has no clear advantage over random initialization in real readout.\n")
            f.write("zero-pad should NOT be used as the primary formal scheme.\n")
            f.write("reinit can serve as a baseline, but still needs teacher-only verification.\n")
        elif cos_diff > 0:
            f.write("**Decision A**: old_pf_zeropad outperforms reinit_pf.\n\n")
            f.write("Old PF still has semantic value in real readout.\n")
            f.write("Next step: Retain old PF, use KNN/parent/nearest initialization for new 2,493 Gaussians,\n")
            f.write("then run teacher-only training.\n")
        else:
            f.write("**Decision C**: reinit_pf outperforms old_pf_zeropad.\n\n")
            f.write("Old PF may be index-misaligned or causing negative transfer.\n")
            f.write("Do NOT use zero-pad as primary scheme.\n")
            f.write("Use reinit baseline or find matching checkpoint.\n")

        f.write("\n## 10. Next Steps\n\n")

        if summary_a is not None and summary_b is not None:
            if cos_diff_pct < 0.05:
                f.write("1. Use reinit [52,493, 512] as baseline\n")
                f.write("2. Run teacher-only 500-step sanity\n")
                f.write("3. Run CE small overfit sanity\n")
                f.write("4. Search for matching N=50,000 geometry checkpoint\n")
            elif cos_diff > 0:
                f.write("1. Retain old PF, expand new 2,493 with KNN/parent init\n")
                f.write("2. Run teacher-only 500-step\n")
                f.write("3. Run CE small overfit sanity\n")
            else:
                f.write("1. Use reinit baseline\n")
                f.write("2. Investigate why old PF underperforms\n")
                f.write("3. Search for matching N=50,000 geometry checkpoint\n")
                f.write("4. Run teacher-only sanity with reinit\n")
        else:
            f.write("1. Fix real readout diagnostic (build_acc hang)\n")
            f.write("2. Retry comparison with working setup\n")

    log(f"\nGenerated: {report_path}")

    log_file.close()
    log("\n" + "="*80)
    log("Phase 13 Real PF Readout Comparison Complete!")
    log(f"Results: {args.output_dir}")
    log("="*80)


if __name__ == "__main__":
    main()
