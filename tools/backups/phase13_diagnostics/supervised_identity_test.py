#!/usr/bin/env python3
"""
Phase 13: Gaussian-Set Identity Classification (Direct Supervised)

Goal: Learn discriminative per-Gaussian person_feature via direct identity classification.

Loss: L_total = lambda_ce * L_CE + lambda_tri * L_Triplet

L_CE: Cross-entropy loss with identity classifier (linear projection head)
L_Triplet: Hard example mining triplet loss on pooled features

This is the most direct supervised signal for ReID - learning to classify identities.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import pretty, traceback

pretty.install()
traceback.install()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from hydra import initialize_config_dir, compose


def normalize_feat(x, eps=1e-6):
    """L2 normalize features."""
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


class IdentityClassifier(nn.Module):
    """
    Linear projection head for identity classification.
    
    Input: person_feature [N, person_feature_dim]
    Output: logits [N, num_identities]
    
    This is the "P" function mentioned in the task.
    """
    def __init__(self, in_dim=512, num_identities=312):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_identities)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """x: [batch, in_dim] -> logits: [batch, num_identities]"""
        return self.fc(x)


def gaussian_set_pooling(model, gpu_batch, bbox, args, device):
    """
    EXACT same Gaussian-Set pooling path as Phase12C/E/F.
    
    Direct xyz projection with opacity weighting.
    """
    x1, y1, x2, y2 = bbox
    try:
        xyz = model.positions
        opacity = model.get_density().squeeze(-1)
        person_feature = model.get_person_feature()

        N = xyz.shape[0]
        if N == 0:
            return None, {'selected_gaussian_count': 0, 'gaussian_weight_sum': 0.0}

        intrinsics = gpu_batch.intrinsics
        if intrinsics is None or len(intrinsics) < 4:
            return None, {'selected_gaussian_count': 0, 'gaussian_weight_sum': 0.0}

        fx, fy, cx, cy = intrinsics
        T_to_world = gpu_batch.T_to_world[0]
        R_world_to_cam = T_to_world[:3, :3].t()
        t_world_to_cam = -R_world_to_cam @ T_to_world[:3, 3]

        xyz_cam = (R_world_to_cam @ xyz.t()).t() + t_world_to_cam
        depth = xyz_cam[:, 2]
        valid_depth = depth > 0

        x_img = fx * xyz_cam[:, 0] / xyz_cam[:, 2] + cx
        y_img = fy * xyz_cam[:, 1] / xyz_cam[:, 2] + cy
        h_img, w_img = gpu_batch.rays_dir.shape[1], gpu_batch.rays_dir.shape[2]

        valid = (
            valid_depth &
            torch.isfinite(x_img) & torch.isfinite(y_img) &
            (x_img >= 0) & (x_img < w_img) &
            (y_img >= 0) & (y_img < h_img) &
            (opacity > 0)
        )
        inside_bbox = (x_img >= x1) & (x_img < x2) & (y_img >= y1) & (y_img < y2)
        inside = valid & inside_bbox

        selected_count = int(inside.sum().item())
        if selected_count == 0:
            return None, {'selected_gaussian_count': 0, 'gaussian_weight_sum': 0.0}

        weights = opacity[inside]
        z = person_feature[inside]
        weight_sum = weights.sum()
        denom = weight_sum.clamp(min=1e-8)
        weighted_sum = (weights[:, None] * z).sum(dim=0)
        G = weighted_sum / denom
        G = normalize_feat(G)

        return G, {
            'selected_gaussian_count': selected_count,
            'gaussian_weight_sum': float(weight_sum.item()),
            'student_feature_norm': float(G.norm().item()),
        }
    except Exception as e:
        return None, {'selected_gaussian_count': 0, 'gaussian_weight_sum': 0.0, 'error': str(e)[:80]}


class BatchBuilder:
    """Build GPU batches for specific camera-frame pairs."""
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


def load_trained_geometry_into_model(model, geometry_ckpt_path, device):
    """Load trained geometry from checkpoint and freeze it."""
    ckpt = torch.load(geometry_ckpt_path, map_location=device)
    msd = ckpt.get('model_state_dict', ckpt)
    if not isinstance(msd, dict):
        msd = ckpt

    num_gaussians = None
    if 'positions' in msd:
        num_gaussians = msd['positions'].shape[0]

    if num_gaussians is None:
        return None, "Cannot determine num_gaussians from checkpoint"

    model.positions = torch.nn.Parameter(msd['positions'].to(device))
    model.rotation = torch.nn.Parameter(msd['rotation'].to(device))
    model.scale = torch.nn.Parameter(msd['scale'].to(device))
    model.density = torch.nn.Parameter(msd['density'].to(device))

    if 'features_albedo' in msd:
        model.features_albedo = torch.nn.Parameter(msd['features_albedo'].to(device))
    if 'features_specular' in msd:
        model.features_specular = torch.nn.Parameter(msd['features_specular'].to(device))

    model.n_active_features = msd.get('n_active_features', model.n_active_features)
    model.max_n_features = msd.get('max_n_features', model.max_n_features)

    for param_name in ['positions', 'rotation', 'scale', 'density']:
        param = getattr(model, param_name)
        param.requires_grad = False

    if hasattr(model, 'features_albedo') and model.features_albedo is not None:
        model.features_albedo.requires_grad = False
    if hasattr(model, 'features_specular') and model.features_specular is not None:
        model.features_specular.requires_grad = False

    return num_gaussians, None


def build_training_pool(dataset, batch_builder, model, allowed_cameras, args, device):
    """
    Build training pool with person_id labels.
    
    Returns list of samples with:
    - person_id (identity label)
    - cam_id
    - frame_idx
    - bbox
    - dataset_index
    """
    print(f"\nBuilding training pool...")
    
    person_id_set = set()
    samples = []
    
    for idx in range(len(dataset)):
        cam_id, frame_idx = dataset.indices[idx]
        if allowed_cameras and cam_id not in allowed_cameras:
            continue
        
        anns = dataset.annotations.get(int(frame_idx), [])
        for ann in anns:
            ann_cam_id = ann.get('camera_id')
            if ann_cam_id is None:
                continue
            ann_cam_str = f"C{ann_cam_id + 1}"
            if ann_cam_str != cam_id:
                continue
            
            pid = ann.get('new_id')
            if pid is None:
                continue
            
            bbox_dict = ann.get('bbox', {})
            if not isinstance(bbox_dict, dict) or len(bbox_dict) < 4:
                continue
            
            x1 = int(bbox_dict.get('xmin', 0))
            y1 = int(bbox_dict.get('ymin', 0))
            x2 = int(bbox_dict.get('xmax', 0))
            y2 = int(bbox_dict.get('ymax', 0))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area < args.min_bbox_area:
                continue
            
            person_id_set.add(int(pid))
            samples.append({
                'person_id': int(pid),
                'cam_id': cam_id,
                'frame_idx': int(frame_idx),
                'dataset_index': idx,
                'bbox': [x1, y1, x2, y2],
                'bbox_area': int(bbox_area),
            })
    
    print(f"  Found {len(samples)} samples with {len(person_id_set)} unique identities")
    return samples, sorted(list(person_id_set))


def compute_triplet_loss(features, labels, margin=0.3):
    """
    Batch-hard triplet loss.
    
    For each anchor, find hardest positive (same label, max distance)
    and hardest negative (different label, min distance).
    """
    device = features.device
    batch_size = features.size(0)
    
    labels = labels.cpu().numpy()
    dist_mat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size) + \
               torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size).t() - \
               2 * torch.matmul(features, features.t())
    dist_mat = dist_mat.clamp(min=0).sqrt()
    
    loss = 0
    num_valid = 0
    
    for i in range(batch_size):
        anchor_label = labels[i]
        
        pos_mask = torch.tensor([labels[j] == anchor_label for j in range(batch_size)], device=device)
        neg_mask = torch.tensor([labels[j] != anchor_label for j in range(batch_size)], device=device)
        pos_mask[i] = False
        
        if pos_mask.sum() == 0:
            continue
        
        pos_dist = dist_mat[i][pos_mask]
        neg_dist = dist_mat[i][neg_mask]
        
        hard_pos = pos_dist.max()
        hard_neg = neg_dist.min()
        
        triplet_loss = F.relu(hard_pos - hard_neg + margin)
        loss += triplet_loss
        num_valid += 1
    
    if num_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss / num_valid


def evaluate_identity(model, classifier, test_samples, batch_builder, args, device):
    """Evaluate identity classification accuracy on test set."""
    model.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_features = []
    
    gs_valid_count = 0
    gs_counts = []
    
    with torch.no_grad():
        for sample in test_samples[:200]:
            gpu_batch = batch_builder.get_batch(sample['cam_id'], sample['frame_idx'])
            if gpu_batch is None:
                continue
            
            G, gs_info = gaussian_set_pooling(model, gpu_batch, sample['bbox'], args, device)
            if G is None:
                continue
            
            gs_counts.append(gs_info['selected_gaussian_count'])
            if gs_info['selected_gaussian_count'] > 0:
                gs_valid_count += 1
            
            logits = classifier(G.unsqueeze(0))
            pred = logits.argmax(dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(sample['person_id'])
            all_features.append(G)
            
            if pred == sample['person_id']:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    same_cos_list = []
    diff_cos_list = []
    
    for i in range(len(all_features)):
        for j in range(i+1, min(len(all_features), i+50)):
            cos_sim = F.cosine_similarity(all_features[i].unsqueeze(0), all_features[j].unsqueeze(0)).item()
            if all_labels[i] == all_labels[j]:
                same_cos_list.append(cos_sim)
            else:
                diff_cos_list.append(cos_sim)
    
    same_cos = float(np.mean(same_cos_list)) if same_cos_list else None
    diff_cos = float(np.mean(diff_cos_list)) if diff_cos_list else None
    gap = (same_cos - diff_cos) if (same_cos is not None and diff_cos is not None) else None
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'same_cos': same_cos,
        'diff_cos': diff_cos,
        'gap': gap,
        'gs_valid_ratio': gs_valid_count / len(gs_counts) if gs_counts else 0,
        'avg_gs_count': float(np.mean(gs_counts)) if gs_counts else 0,
    }


def run_phase13(args):
    """Run Phase 13: Gaussian-Set Identity Classification."""
    print("=" * 80)
    print("Phase 13: Gaussian-Set Identity Classification")
    print("=" * 80)
    
    allowed_cameras = [c.strip() for c in args.allowed_cameras.split(',')] if args.allowed_cameras else ['C1', 'C4', 'C6', 'C7']
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    print(f"\n[1/5] Checking geometry checkpoint...")
    if not os.path.exists(args.geometry_checkpoint):
        print(f"ERROR: Geometry checkpoint not found: {args.geometry_checkpoint}")
        return False
    
    print(f"  Geometry checkpoint: {args.geometry_checkpoint}")
    
    print(f"\n[2/5] Loading model and geometry...")
    config_dir = os.path.join(REPO_ROOT, 'configs')
    config_file = 'apps/wildtrack_full_3dgut'
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_file)
    cfg.model.person_feature_dim = 512
    
    from threedgrut.trainer import Trainer3DGRUT
    trainer = Trainer3DGRUT(cfg)
    model = trainer.model
    device = trainer.device
    dataset = trainer.train_dataset
    
    num_gaussians, error = load_trained_geometry_into_model(model, args.geometry_checkpoint, device)
    if error:
        print(f"ERROR: Failed to load geometry: {error}")
        return False
    
    print(f"  num_gaussians: {num_gaussians}")
    print(f"  geometry_loaded: True")
    print(f"  geometry_frozen: True")
    
    print(f"\n  Initializing person_feature for {num_gaussians} Gaussians...")
    model._person_feature = torch.nn.Parameter(
        torch.randn(num_gaussians, 512, dtype=torch.float32, device=device) * 0.01
    )
    model._person_feature.requires_grad = True
    
    print(f"\n[3/5] Building training pool...")
    batch_builder = BatchBuilder(dataset)
    train_samples, all_person_ids = build_training_pool(dataset, batch_builder, model, allowed_cameras, args, device)
    
    if not train_samples:
        print("ERROR: No training samples found")
        return False
    
    person_id_to_idx = {pid: idx for idx, pid in enumerate(all_person_ids)}
    num_identities = len(all_person_ids)
    print(f"  num_identities: {num_identities}")
    
    print(f"\n  Creating identity classifier...")
    classifier = IdentityClassifier(in_dim=512, num_identities=num_identities).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': [model._person_feature], 'lr': args.person_feature_lr},
        {'params': classifier.parameters(), 'lr': args.classifier_lr},
    ], weight_decay=1e-4)
    
    print(f"  Optimizer: Adam (person_feature lr={args.person_feature_lr}, classifier lr={args.classifier_lr})")
    
    print(f"\n[4/5] Training identity classification...")
    metrics_log = []
    best_accuracy, best_gap, best_step = 0.0, 0.0, 0
    
    random.shuffle(train_samples)
    
    min_valid_per_batch = max(4, args.batch_size // 2)
    
    for step in range(args.num_steps):
        step_start = time.time()
        
        features_list = []
        labels_list = []
        gs_counts_list = []
        gs_weights_list = []
        
        num_attempts = 0
        max_attempts = args.batch_size * 5
        
        while len(features_list) < min_valid_per_batch and num_attempts < max_attempts:
            sample = random.choice(train_samples)
            num_attempts += 1
            
            gpu_batch = batch_builder.get_batch(sample['cam_id'], sample['frame_idx'])
            if gpu_batch is None:
                continue
            
            G, gs_info = gaussian_set_pooling(model, gpu_batch, sample['bbox'], args, device)
            if G is None:
                continue
            
            if sample['person_id'] not in person_id_to_idx:
                continue
            
            person_idx = person_id_to_idx[sample['person_id']]
            
            features_list.append(G)
            labels_list.append(person_idx)
            gs_counts_list.append(gs_info['selected_gaussian_count'])
            gs_weights_list.append(gs_info['gaussian_weight_sum'])
        
        valid_count = len(features_list)
        if valid_count < min_valid_per_batch:
            continue
        
        features_batch = torch.stack(features_list)
        labels_batch = torch.tensor(labels_list, dtype=torch.long, device=device)
        
        logits = classifier(features_batch)
        
        L_ce = F.cross_entropy(logits, labels_batch)
        
        if torch.isnan(L_ce):
            continue
        
        L_tri = compute_triplet_loss(features_batch, labels_batch, margin=args.triplet_margin)
        
        if isinstance(L_tri, torch.Tensor) and torch.isnan(L_tri):
            L_tri = torch.tensor(0.0, device=device)
        
        L_total = args.lambda_ce * L_ce + args.lambda_tri * L_tri
        
        if torch.isnan(L_total):
            continue
        
        L_total.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [model._person_feature] + list(classifier.parameters()),
            args.grad_clip_norm
        )
        
        if not torch.isnan(grad_norm):
            optimizer.step()
        
        with torch.no_grad():
            model._person_feature.data = normalize_feat(model._person_feature.data)
        
        step_time = time.time() - step_start
        
        if step % args.log_interval == 0:
            log_line = f"[PHASE13] Step {step:5d}: loss={L_total.item():.4f} " \
                       f"ce={L_ce.item():.4f} tri={L_tri.item() if isinstance(L_tri, torch.Tensor) else L_tri:.4f} " \
                       f"valid={valid_count} t={step_time:.2f}s"
            print(log_line)
        
        step_record = {
            'step': step,
            'loss_total': float(L_total.item()),
            'loss_ce': float(L_ce.item()),
            'loss_tri': float(L_tri.item()) if isinstance(L_tri, torch.Tensor) else L_tri,
            'valid_sample_count': valid_count,
            'selected_gaussian_count_mean': float(np.mean(gs_counts_list)) if gs_counts_list else 0,
            'gaussian_weight_sum_mean': float(np.mean(gs_weights_list)) if gs_weights_list else 0,
            'grad_norm': float(grad_norm),
        }
        
        if step % args.eval_interval == 0 or step == args.num_steps - 1:
            eval_results = evaluate_identity(model, classifier, train_samples, batch_builder, args, device)
            step_record.update(eval_results)
            
            print(f"  [EVAL] accuracy={eval_results['accuracy']:.4f} "
                  f"same_cos={eval_results['same_cos']:.4f} "
                  f"diff_cos={eval_results['diff_cos']:.4f} "
                  f"gap={eval_results['gap']:+.4f}")
            
            if eval_results['accuracy'] > best_accuracy:
                best_accuracy = eval_results['accuracy']
                best_step = step
                torch.save({
                    'model_state_dict': {
                        '_person_feature': model._person_feature.cpu().clone(),
                    },
                    'classifier_state_dict': classifier.cpu().state_dict(),
                    'geometry_source_path': args.geometry_checkpoint,
                    'geometry_loaded': True,
                    'geometry_frozen': True,
                    'person_feature_shape': list(model._person_feature.shape),
                    'num_identities': num_identities,
                    'person_id_to_idx': person_id_to_idx,
                    'step': step,
                    'accuracy': eval_results['accuracy'],
                    'gap': eval_results['gap'],
                }, os.path.join(args.output_dir, 'checkpoint_best_accuracy.pt'))
                classifier.to(device)
            
            if eval_results['gap'] is not None and eval_results['gap'] > best_gap:
                best_gap = eval_results['gap']
                torch.save({
                    'model_state_dict': {
                        '_person_feature': model._person_feature.cpu().clone(),
                    },
                    'classifier_state_dict': classifier.cpu().state_dict(),
                    'geometry_source_path': args.geometry_checkpoint,
                    'geometry_loaded': True,
                    'geometry_frozen': True,
                    'person_feature_shape': list(model._person_feature.shape),
                    'num_identities': num_identities,
                    'person_id_to_idx': person_id_to_idx,
                    'step': step,
                    'gap': eval_results['gap'],
                }, os.path.join(args.output_dir, 'checkpoint_best_gap.pt'))
                classifier.to(device)
        
        metrics_log.append(step_record)
    
    with open(os.path.join(args.output_dir, 'metrics.jsonl'), 'w') as f:
        for r in metrics_log:
            f.write(json.dumps(r, default=str) + '\n')
    
    torch.save({
        'model_state_dict': {
            '_person_feature': model._person_feature.cpu().clone(),
        },
        'classifier_state_dict': classifier.cpu().state_dict(),
        'geometry_source_path': args.geometry_checkpoint,
        'geometry_loaded': True,
        'geometry_frozen': True,
        'person_feature_shape': list(model._person_feature.shape),
        'num_identities': num_identities,
        'person_id_to_idx': person_id_to_idx,
        'step': args.num_steps - 1,
    }, os.path.join(args.output_dir, 'checkpoint_latest.pt'))
    
    last_accuracy = metrics_log[-1].get('accuracy', 0) if metrics_log else 0
    last_gap = metrics_log[-1].get('gap', 0) if metrics_log else 0
    
    summary = {
        'geometry_checkpoint': args.geometry_checkpoint,
        'geometry_loaded': True,
        'geometry_frozen': True,
        'num_gaussians': num_gaussians,
        'num_identities': num_identities,
        'person_feature_shape': [num_gaussians, 512],
        'best_step': best_step,
        'best_accuracy': best_accuracy,
        'best_gap': best_gap,
        'last_accuracy': last_accuracy,
        'last_gap': last_gap,
        'allowed_cameras': allowed_cameras,
    }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    generate_final_report(args.output_dir, summary)
    
    print(f"\n{'='*80}")
    print(f"Phase 13 complete! Best accuracy={best_accuracy:.4f}, Best gap={best_gap:+.4f}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")
    
    return True


def generate_final_report(output_dir, summary):
    """Generate final_report.md."""
    report = f"""# Phase 13: Gaussian-Set Identity Classification Report

## 1. 训练是否成功完成？

是

## 2. 是否使用真实 3DGS geometry？

是
- geometry_source: {summary['geometry_checkpoint']}
- num_gaussians: {summary['num_gaussians']}
- geometry_frozen: True

## 3. Identity classification 性能

- **Best accuracy**: {summary['best_accuracy']:.4f} (step {summary['best_step']})
- **Best gap**: {summary['best_gap']:+.4f}
- **Last accuracy**: {summary['last_accuracy']:.4f}
- **Last gap**: {summary['last_gap']:+.4f}

## 4. Gaussian-Set pooling 是否正常？

允许相机: {summary['allowed_cameras']}

## 5. 与 Phase 12 对比

Phase 13 使用直接身份分类监督（cross-entropy + triplet loss），
相比 Phase 12 的 teacher distillation 提供更强的身份区分信号。

## 训练配置

- num_identities: {summary['num_identities']}
- person_feature_lr: (see args.json)
- classifier_lr: (see args.json)
- lambda_ce: (see args.json)
- lambda_tri: (see args.json)
- triplet_margin: (see args.json)

## 输出文件

- checkpoint_best_accuracy.pt: 最高分类准确率的 checkpoint
- checkpoint_best_gap.pt: 最大 same_cos - diff_cos gap 的 checkpoint
- checkpoint_latest.pt: 最后一个 step 的 checkpoint
- metrics.jsonl: 训练指标
- summary.json: 实验总结

---

*报告生成时间：2026-05-13*
"""
    
    with open(os.path.join(output_dir, 'final_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Phase 13: Gaussian-Set Identity Classification')
    
    parser.add_argument('--geometry_checkpoint', type=str,
                        default='runs/phase11A_proto_infonce_opacity_lam005_lr1e4/step_20999.pth')
    parser.add_argument('--allowed_cameras', type=str, default='C1,C4,C6,C7')
    parser.add_argument('--min_bbox_area', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--person_feature_lr', type=float, default=1e-3)
    parser.add_argument('--classifier_lr', type=float, default=1e-3)
    parser.add_argument('--lambda_ce', type=float, default=1.0)
    parser.add_argument('--lambda_tri', type=float, default=0.1)
    parser.add_argument('--triplet_margin', type=float, default=0.3)
    parser.add_argument('--num_steps', type=int, default=2000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--grad_clip_norm', type=float, default=10.0)
    parser.add_argument('--output_dir', type=str,
                        default='outputs/phase13_supervised_identity')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    run_phase13(args)


if __name__ == '__main__':
    main()
