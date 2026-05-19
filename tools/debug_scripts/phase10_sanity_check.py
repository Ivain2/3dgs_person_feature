#!/usr/bin/env python3
"""
Sanity check: compare mean / opacity / topk_opacity pooling on the same batch.

Usage:
    PYTHONPATH=. python tools/phase10_sanity_check.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --checkpoint runs/phase6_reid_main/latest.pth \
        --num_batches 10
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.roi_pooling import roi_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num_batches', type=int, default=10)
    args = parser.parse_args()

    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)

    conf.model.person_feature_dim = 512
    conf.loss.use_reid = True
    conf.loss.lambda_reid = 0.0

    print("Initializing trainer...")
    trainer = Trainer3DGRUT(conf)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)

    trainer.model.eval()

    results = {
        'mean': {'cos_tv': [], 'raw_norm': [], 'feat_nz': []},
        'opacity': {'cos_tv': [], 'raw_norm': [], 'feat_nz': [], 'alpha_sum': [], 'alpha_mean': [], 'alpha_max': [], 'skip': 0},
        'topk': {'cos_tv': [], 'raw_norm': [], 'feat_nz': [], 'alpha_sum': [], 'alpha_mean': [], 'alpha_max': [], 'skip': 0},
    }
    bbox_areas = []
    total_instances = 0

    train_iter = iter(trainer.train_dataloader)

    with torch.no_grad():
        for b in range(args.num_batches):
            try:
                batch_data = next(train_iter)
            except StopIteration:
                train_iter = iter(trainer.train_dataloader)
                batch_data = next(train_iter)

            gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)
            render_out = trainer.model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
            person_feature_map = render_out['person_feature_map']
            person_opacity_map = render_out.get('person_opacity_map')

            for inst in gpu_batch.instances:
                if not inst.get('valid', False):
                    continue
                teacher_emb = inst.get('teacher_embedding')
                if teacher_emb is None:
                    continue

                bbox = inst['bbox_xyxy']
                bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)
                t_v = torch.as_tensor(teacher_emb, dtype=torch.float32, device=trainer.device).squeeze()
                t_v = F.normalize(t_v, p=2, dim=0, eps=1e-6)

                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                bbox_areas.append(bbox_area)
                total_instances += 1

                # Mean pooling
                f_mean, _ = roi_pool(person_feature_map, bbox_t, pooling="mean")
                cos_mean = torch.dot(f_mean, t_v).item()
                raw_norm_mean = f_mean.norm(p=2).item()
                results['mean']['cos_tv'].append(cos_mean)
                results['mean']['raw_norm'].append(raw_norm_mean)
                D, H, W = person_feature_map.shape
                xmin = max(0, int(bbox_t[0].item()))
                ymin = max(0, int(bbox_t[1].item()))
                xmax = min(W, max(xmin + 1, int(bbox_t[2].item())))
                ymax = min(H, max(ymin + 1, int(bbox_t[3].item())))
                region = person_feature_map[:, ymin:ymax, xmin:xmax]
                results['mean']['feat_nz'].append((region.abs() > 1e-6).float().mean().item() * 100)

                # Opacity pooling
                f_opa, stats_opa = roi_pool(person_feature_map, bbox_t, opacity_map=person_opacity_map, pooling="opacity")
                if f_opa is not None:
                    cos_opa = torch.dot(f_opa, t_v).item()
                    results['opacity']['cos_tv'].append(cos_opa)
                    results['opacity']['raw_norm'].append(stats_opa['weighted_feature_norm_mean'])
                    results['opacity']['alpha_sum'].append(stats_opa['alpha_sum'])
                    results['opacity']['alpha_mean'].append(stats_opa['alpha_mean'])
                    results['opacity']['alpha_max'].append(stats_opa['alpha_max'])
                    results['opacity']['feat_nz'].append((region.abs() > 1e-6).float().mean().item() * 100)
                else:
                    results['opacity']['skip'] += 1

                # Top-k opacity pooling
                f_topk, stats_topk = roi_pool(person_feature_map, bbox_t, opacity_map=person_opacity_map, pooling="topk_opacity", topk_ratio=0.3)
                if f_topk is not None:
                    cos_topk = torch.dot(f_topk, t_v).item()
                    results['topk']['cos_tv'].append(cos_topk)
                    results['topk']['raw_norm'].append(stats_topk['weighted_feature_norm_mean'])
                    results['topk']['alpha_sum'].append(stats_topk['alpha_sum'])
                    results['topk']['alpha_mean'].append(stats_topk['alpha_mean'])
                    results['topk']['alpha_max'].append(stats_topk['alpha_max'])
                    results['topk']['feat_nz'].append((region.abs() > 1e-6).float().mean().item() * 100)
                else:
                    results['topk']['skip'] += 1

    print("\n" + "=" * 70)
    print(f"SANITY CHECK: {args.checkpoint}")
    print(f"Total instances: {total_instances}, bbox_area_mean: {np.mean(bbox_areas):.1f}")
    print("=" * 70)

    for mode in ['mean', 'opacity', 'topk']:
        r = results[mode]
        n = len(r['cos_tv'])
        if n == 0:
            print(f"\n  {mode}: NO valid instances (skipped={r.get('skip', 0)})")
            continue
        print(f"\n  {mode} pooling ({n} instances, skipped={r.get('skip', 0)}):")
        print(f"    cos(f_v, t_v): mean={np.mean(r['cos_tv']):.4f}  std={np.std(r['cos_tv']):.4f}  min={np.min(r['cos_tv']):.4f}  max={np.max(r['cos_tv']):.4f}")
        print(f"    raw_norm:      mean={np.mean(r['raw_norm']):.4f}")
        print(f"    feat_nz:       mean={np.mean(r['feat_nz']):.2f}%")
        if 'alpha_sum' in r and r['alpha_sum']:
            print(f"    alpha_sum:     mean={np.mean(r['alpha_sum']):.2f}  min={np.min(r['alpha_sum']):.2f}  max={np.max(r['alpha_sum']):.2f}")
            print(f"    alpha_mean:    mean={np.mean(r['alpha_mean']):.4f}  min={np.min(r['alpha_mean']):.4f}  max={np.max(r['alpha_max']):.4f}")

    if results['mean']['cos_tv'] and results['opacity']['cos_tv']:
        mean_cos = np.mean(results['mean']['cos_tv'])
        opa_cos = np.mean(results['opacity']['cos_tv'])
        topk_cos = np.mean(results['topk']['cos_tv']) if results['topk']['cos_tv'] else 0
        print(f"\n  >>> cos improvement: opacity vs mean = {opa_cos - mean_cos:+.4f}, topk vs mean = {topk_cos - mean_cos:+.4f}")

        mean_norm = np.mean(results['mean']['raw_norm'])
        opa_norm = np.mean(results['opacity']['raw_norm'])
        topk_norm = np.mean(results['topk']['raw_norm']) if results['topk']['raw_norm'] else 0
        print(f"  >>> norm improvement: opacity vs mean = {opa_norm / max(mean_norm, 1e-8):.1f}x, topk vs mean = {topk_norm / max(mean_norm, 1e-8):.1f}x")


if __name__ == "__main__":
    main()
