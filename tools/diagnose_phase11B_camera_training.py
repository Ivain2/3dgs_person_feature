"""
Camera ROI Diagnostic - Focused analysis of camera bias.

This script diagnoses why certain cameras (C2/C3/C5) show 0% ROI validity
by sampling only from frames that exist in the dataset indices.
"""

import os
import sys
import json
import random
import numpy as np
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from threedgrut.trainer import Trainer3DGRUT
from threedgrut.datasets import make

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_feat(x):
    return x / (x.norm() + 1e-8)

def opacity_roi_pooling(feature_map, opacity_map, bbox_xyxy, denom_eps=1e-1, detach_opacity_weight=True):
    if opacity_map is None or feature_map is None:
        return None, {}
    xmin, ymin, xmax, ymax = int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2]), int(bbox_xyxy[3])
    H, W = feature_map.shape[1], feature_map.shape[2]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(W, max(xmin + 1, xmax))
    ymax = min(H, max(ymin + 1, ymax))
    alpha_region = opacity_map[ymin:ymax, xmin:xmax]
    alpha_sum = alpha_region.sum()
    denom = alpha_sum + denom_eps
    clamped = False
    if denom > 1e6:
        denom = 1e6
        clamped = True
    if detach_opacity_weight:
        weight = alpha_region.detach() / denom
    else:
        weight = alpha_region / denom
    feat_region = feature_map[:, ymin:ymax, xmin:xmax]
    pooled = (feat_region * weight.unsqueeze(0)).sum(dim=(1, 2))
    return pooled, {
        'alpha_sum': float(alpha_sum.item()),
        'denom': float(denom),
        'clamped': clamped,
        'pooled_norm': float(pooled.norm().item()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/apps/wildtrack_full_3dgut.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs/phase11B_v4_camera_roi_diagnostic')
    parser.add_argument('--samples_per_camera', type=int, default=50)
    parser.add_argument('--denom_eps', type=float, default=1e-1)
    parser.add_argument('--detach_opacity_weight', action='store_true', default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    config = OmegaConf.load(args.config)
    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=config_name)
    config.model.person_feature_dim = 512
    config.model.person_feature_lr = 1e-5
    config.loss.use_reid = True
    config.loss.lambda_reid = 0.0

    trainer = Trainer3DGRUT(config)
    print(f"Gaussians: {trainer.model.num_gaussians}")

    ckpt_path = 'runs/phase10C_topk_detach_lam005_lr1e4_stable/latest.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)

    for param in [trainer.model.positions, trainer.model.rotation, trainer.model.scale,
                  trainer.model.density, trainer.model.features_albedo, trainer.model.features_specular]:
        if param is not None:
            param.requires_grad_(False)
    trainer.model._person_feature.requires_grad_(True)
    trainer.model.optimizer = torch.optim.Adam(
        [{'params': trainer.model._person_feature, 'lr': 1e-5}], eps=1e-8, weight_decay=0,
    )

    train_dataset, _ = make('wildtrack', config, ray_jitter=None)
    print(f"Dataset indices: {len(train_dataset.indices)}")

    # Build cam_frame lookup from dataset indices
    cam_frame_set = set()
    cam_frame_to_idx = {}
    for idx, (cam_id, frame_idx) in enumerate(train_dataset.indices):
        cam_frame_set.add((cam_id, int(frame_idx)))
        cam_frame_to_idx[(cam_id, int(frame_idx))] = idx

    index_frames = sorted(set(fi for _, fi in train_dataset.indices))
    unique_cams = sorted(set(ci for ci, _ in train_dataset.indices))
    print(f"Unique cameras in dataset: {unique_cams}")
    print(f"Frames per camera: {len(index_frames)}")

    # Get the trainer's batch builder method
    def get_gpu_batch(cam_id, frame_idx):
        idx = cam_frame_to_idx.get((cam_id, int(frame_idx)))
        if idx is None:
            return None
        raw_batch = train_dataset[idx]
        return train_dataset.get_gpu_batch_with_intrinsics(raw_batch)

    # Sample from annotations, but ONLY for frames that exist in dataset
    camera_metrics = {cam: [] for cam in unique_cams}
    
    # Collect valid (cam, frame, person) tuples from annotations
    valid_annot_entries = []
    for frame_id, annots in train_dataset.annotations.items():
        if not isinstance(annots, list):
            continue
        fi = int(frame_id)
        if fi not in index_frames:
            continue  # Skip frames that don't exist in dataset
        for p in annots:
            pid = p.get('train_id') or p.get('new_id')
            if pid is None:
                continue
            annot_cam = p.get('camera_id')
            if annot_cam is None:
                continue
            cam_id = f"C{annot_cam + 1}"
            if (cam_id, fi) not in cam_frame_set:
                continue
            valid_annot_entries.append({
                'cam_id': cam_id, 'frame_idx': fi, 'person_id': pid, 'person_data': p
            })

    print(f"\nTotal valid annotation entries (within dataset frames): {len(valid_annot_entries)}")

    # Group by camera
    by_cam = defaultdict(list)
    for entry in valid_annot_entries:
        by_cam[entry['cam_id']].append(entry)

    random.seed(42)
    np.random.seed(42)

    for cam_id in unique_cams:
        cam_entries = by_cam.get(cam_id, [])
        n_samples = min(args.samples_per_camera, len(cam_entries))
        print(f"\n{'='*50}")
        print(f"Camera: {cam_id} - sampling {n_samples} from {len(cam_entries)} valid entries")
        print(f"{'='*50}")

        if n_samples == 0:
            print(f"  No valid entries for {cam_id}!")
            continue

        sampled = random.sample(cam_entries, n_samples)

        cam_invalid_reasons = defaultdict(int)
        render_success = 0

        for entry in sampled:
            frame_idx = entry['frame_idx']
            pid = entry['person_id']
            pdata = entry['person_data']

            gpu_batch = get_gpu_batch(cam_id, frame_idx)
            if gpu_batch is None:
                cam_invalid_reasons['no_batch'] += 1
                camera_metrics[cam_id].append({'valid': False, 'reason': 'no_batch'})
                continue

            try:
                render_out = trainer.model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
                person_feature_map = render_out['person_feature_map']
                person_opacity_map = render_out.get('person_opacity_map')
                render_ok = True
            except Exception as e:
                render_ok = False
                person_feature_map = None
                person_opacity_map = None

            if not render_ok:
                cam_invalid_reasons['render_failed'] += 1
                camera_metrics[cam_id].append({'valid': False, 'reason': 'render_failed'})
                continue

            render_success += 1

            D, H, W = person_feature_map.shape

            # Find instance in batch
            inst = None
            for i in gpu_batch.instances:
                if i.get('train_id') == pid:
                    inst = i
                    break

            if inst is None:
                cam_invalid_reasons['no_instance'] += 1
                camera_metrics[cam_id].append({'valid': False, 'reason': 'no_instance'})
                continue

            bbox = inst['bbox_xyxy']
            bbox_area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

            if person_opacity_map is not None:
                xmin, ymin = max(0, int(bbox[0])), max(0, int(bbox[1]))
                xmax, ymax = min(W, int(bbox[2])), min(H, int(bbox[3]))
                alpha_region = person_opacity_map[ymin:ymax, xmin:xmax]
                alpha_sum = float(alpha_region.sum().item())
                alpha_max = float(alpha_region.max().item()) if alpha_region.numel() > 0 else 0
            else:
                alpha_sum = 0
                alpha_max = 0

            bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)
            f_v, pool_stats = opacity_roi_pooling(
                person_feature_map, person_opacity_map, bbox_t,
                denom_eps=args.denom_eps, detach_opacity_weight=args.detach_opacity_weight,
            )

            teacher_emb = inst.get('teacher_embedding')
            if teacher_emb is not None:
                teacher_feat = normalize_feat(
                    torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                )
                if f_v is not None:
                    cos_sim = float(torch.dot(f_v, teacher_feat).item())
                else:
                    cos_sim = 0
            else:
                cos_sim = 0

            clamped = pool_stats.get('clamped', False) if pool_stats else True
            valid = (f_v is not None and alpha_sum > 1e-3 and not clamped)

            if f_v is None:
                cam_invalid_reasons['roi_pool_failed'] += 1
                reason = 'roi_pool_failed'
            elif alpha_sum < 1e-3:
                cam_invalid_reasons['alpha_too_small'] += 1
                reason = 'alpha_too_small'
            elif clamped:
                cam_invalid_reasons['denom_clamped'] += 1
                reason = 'denom_clamped'
            else:
                reason = ''

            camera_metrics[cam_id].append({
                'valid': valid,
                'reason': reason,
                'alpha_sum': alpha_sum,
                'alpha_max': alpha_max,
                'bbox_area': bbox_area,
                'cos_sim': cos_sim,
                'pooled_norm': pool_stats.get('pooled_norm', 0) if pool_stats else 0,
            })

        valid_count = sum(1 for m in camera_metrics[cam_id] if m['valid'])
        total = len(camera_metrics[cam_id])
        alphas = [m['alpha_sum'] for m in camera_metrics[cam_id] if m.get('alpha_sum', 0) > 0]
        print(f"  Valid: {valid_count}/{total} ({valid_count/max(total,1)*100:.1f}%)")
        print(f"  Render success: {render_success}")
        print(f"  Invalid reasons: {dict(cam_invalid_reasons)}")
        if alphas:
            print(f"  Alpha sum: mean={np.mean(alphas):.4f}, median={np.median(alphas):.4f}, min={min(alphas):.4f}, max={max(alphas):.4f}")
        cos_sims = [m.get('cos_sim', 0) for m in camera_metrics[cam_id] if m.get('cos_sim', 0) != 0]
        if cos_sims:
            print(f"  Cos similarity: mean={np.mean(cos_sims):.4f}, std={np.std(cos_sims):.4f}")

    # Save results
    metrics_path = os.path.join(args.output_dir, "camera_roi_metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for cam_id in unique_cams:
            for m in camera_metrics[cam_id]:
                f.write(json.dumps({**m, 'cam_id': cam_id}, default=str) + "\n")

    # Summary
    summary = {"per_camera": {}}
    for cam_id in unique_cams:
        ms = camera_metrics[cam_id]
        total = len(ms)
        valid = sum(1 for m in ms if m['valid'])
        alphas = [m.get('alpha_sum', 0) for m in ms if m.get('alpha_sum', 0) > 0]
        cos_sims = [m.get('cos_sim', 0) for m in ms if m.get('cos_sim', 0) != 0]
        reasons = defaultdict(int)
        for m in ms:
            if m.get('reason'):
                reasons[m['reason']] += 1

        summary["per_camera"][cam_id] = {
            "sampled": total,
            "valid": valid,
            "valid_ratio": valid / max(total, 1),
            "invalid_reasons": dict(reasons),
            "alpha_sum": {
                "mean": float(np.mean(alphas)) if alphas else 0,
                "median": float(np.median(alphas)) if alphas else 0,
                "min": float(min(alphas)) if alphas else 0,
                "max": float(max(alphas)) if alphas else 0,
            },
            "cos_similarity": {
                "mean": float(np.mean(cos_sims)) if cos_sims else 0,
                "std": float(np.std(cos_sims)) if cos_sims else 0,
            },
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("CAMERA ROI DIAGNOSTIC SUMMARY (within valid dataset frames only)")
    print(f"{'='*70}")
    print(f"{'Camera':<10} {'Valid':<8} {'Ratio':<10} {'AlphaMean':<12} {'CosMean':<10}")
    for cam_id in unique_cams:
        s = summary["per_camera"][cam_id]
        print(f"{cam_id:<10} {s['valid']:<8} {s['valid_ratio']*100:<10.1f} "
              f"{s['alpha_sum']['mean']:<12.4f} {s['cos_similarity']['mean']:<10.4f}")

    print(f"\nOutput: {summary_path}")


if __name__ == "__main__":
    main()
