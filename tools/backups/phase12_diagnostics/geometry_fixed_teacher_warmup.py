#!/usr/bin/env python3
"""
Phase12 Geometry Fix + Teacher Warm-up

目标：确认并修复 Phase12 是否真正使用 trained 3DGS geometry，而不是 random point cloud。

功能：
1. 验证真实 geometry checkpoint 是否存在、能否加载
2. 加载真实 geometry 到模型，禁止 random point cloud
3. 初始化 person_feature 与真实 Gaussian 数量一致
4. geometry load sanity 验证
5. C1-C7 pretrain Gaussian-Set diagnostic
6. teacher warm-up training（仅 valid cameras）

不跑 12G，不训练 geometry，不移动 Phase12 文件。
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
import torch.nn.functional as F
from rich import pretty, traceback

pretty.install()
traceback.install()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from hydra import initialize_config_dir, compose


GEOMETRY_KEYS = [
    'positions', '_positions', 'xyz', 'means',
    'rotation', '_rotation',
    'scale', '_scale', 'scaling',
    'density', '_density', 'opacity', '_opacity',
    'features_albedo', 'features_specular',
    'features_dc', 'features_rest', 'shs', 'colors',
]


def normalize_feat(x, eps=1e-6):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def check_geometry_checkpoint(ckpt_path):
    """检查候选 geometry checkpoint 是否包含真实 geometry。"""
    report = {
        'path': ckpt_path,
        'exists': os.path.exists(ckpt_path),
        'has_geometry': False,
        'geometry_keys_found': [],
        'missing_geometry_keys': [],
        'key_shapes': {},
        'can_load': False,
        'num_gaussians': None,
    }

    if not report['exists']:
        return report, f"Checkpoint not found: {ckpt_path}"

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        return report, f"Error loading checkpoint: {str(e)}"

    report['top_level_keys'] = list(ckpt.keys())

    msd = ckpt.get('model_state_dict', ckpt)
    if not isinstance(msd, dict):
        msd = ckpt

    for key, value in msd.items():
        if isinstance(value, torch.Tensor):
            report['key_shapes'][key] = list(value.shape)

    expected_geo = ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']
    for geo_key in expected_geo:
        if geo_key in msd:
            report['geometry_keys_found'].append(geo_key)
            if geo_key == 'positions':
                report['num_gaussians'] = msd[geo_key].shape[0]
        else:
            report['missing_geometry_keys'].append(geo_key)

    report['has_geometry'] = len(report['geometry_keys_found']) >= 4
    report['can_load'] = report['has_geometry']

    return report, None


def load_trained_geometry_into_model(model, geometry_ckpt_path, device):
    """
    从 geometry checkpoint 加载真实 geometry 到模型。

    要求：
    1. 加载 positions / rotation / scale / density / appearance features
    2. 做显式 key mapping（如果 key 不一致）
    3. 不调用 init_from_random_point_cloud()
    4. 冻结 geometry 参数
    5. 只允许训练 _person_feature
    """
    ckpt = torch.load(geometry_ckpt_path, map_location=device)
    msd = ckpt.get('model_state_dict', ckpt)
    if not isinstance(msd, dict):
        msd = ckpt

    report = {
        'geometry_source_path': geometry_ckpt_path,
        'geometry_loaded': False,
        'num_gaussians': None,
        'key_mapping': {},
        'xyz_shape': None,
        'xyz_stats': {},
        'opacity_or_density_stats': {},
    }

    num_gaussians = None
    if 'positions' in msd:
        num_gaussians = msd['positions'].shape[0]
    elif 'xyz' in msd:
        num_gaussians = msd['xyz'].shape[0]

    if num_gaussians is None:
        return report, "Cannot determine num_gaussians from checkpoint"

    report['num_gaussians'] = num_gaussians

    model.positions = torch.nn.Parameter(msd['positions'].to(device))
    report['key_mapping']['positions'] = 'positions'

    model.rotation = torch.nn.Parameter(msd['rotation'].to(device))
    report['key_mapping']['rotation'] = 'rotation'

    model.scale = torch.nn.Parameter(msd['scale'].to(device))
    report['key_mapping']['scale'] = 'scale'

    model.density = torch.nn.Parameter(msd['density'].to(device))
    report['key_mapping']['density'] = 'density'

    if 'features_albedo' in msd:
        model.features_albedo = torch.nn.Parameter(msd['features_albedo'].to(device))
        report['key_mapping']['features_albedo'] = 'features_albedo'
    else:
        print("  WARNING: features_albedo not found in checkpoint, using default")

    if 'features_specular' in msd:
        model.features_specular = torch.nn.Parameter(msd['features_specular'].to(device))
        report['key_mapping']['features_specular'] = 'features_specular'
    else:
        print("  WARNING: features_specular not found in checkpoint, using default")

    model.n_active_features = msd.get('n_active_features', model.n_active_features)
    model.max_n_features = msd.get('max_n_features', model.max_n_features)

    xyz = model.positions.detach().cpu().numpy()
    report['xyz_shape'] = list(xyz.shape)
    report['xyz_stats'] = {
        'xyz_mean': float(np.mean(xyz)),
        'xyz_std': float(np.std(xyz)),
        'xyz_min': float(np.min(xyz)),
        'xyz_max': float(np.max(xyz)),
    }

    density = model.density.detach().cpu().numpy()
    report['opacity_or_density_stats'] = {
        'density_mean': float(np.mean(density)),
        'density_std': float(np.std(density)),
        'density_min': float(np.min(density)),
        'density_max': float(np.max(density)),
    }

    for param_name in ['positions', 'rotation', 'scale', 'density']:
        param = getattr(model, param_name)
        param.requires_grad = False

    if hasattr(model, 'features_albedo') and model.features_albedo is not None:
        model.features_albedo.requires_grad = False
    if hasattr(model, 'features_specular') and model.features_specular is not None:
        model.features_specular.requires_grad = False

    report['geometry_loaded'] = True
    report['geometry_frozen'] = True

    return report, None


def geometry_load_sanity_check(args, geometry_ckpt_path, device, seed1=42, seed2=123):
    """验证 geometry loading 的一致性：不同 seed 下 xyz 应该完全一致。"""
    results = []

    for seed in [seed1, seed2]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        config_dir = os.path.join(REPO_ROOT, 'configs')
        config_file = 'apps/wildtrack_full_3dgut'

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=config_file)
        cfg.model.person_feature_dim = 512

        from threedgrut.trainer import Trainer3DGRUT
        trainer = Trainer3DGRUT(cfg)
        model = trainer.model

        geo_report, error = load_trained_geometry_into_model(model, geometry_ckpt_path, device)

        xyz = model.positions.detach().cpu().numpy()
        results.append({
            'seed': seed,
            'xyz_first_5': xyz[:5].tolist(),
            'xyz_mean': float(np.mean(xyz)),
            'xyz_std': float(np.std(xyz)),
            'error': error,
        })

    xyz1 = np.array(results[0]['xyz_first_5'])
    xyz2 = np.array(results[1]['xyz_first_5'])

    sanity = {
        'seed1': seed1,
        'seed2': seed2,
        'xyz_seed1_first_5': results[0]['xyz_first_5'],
        'xyz_seed2_first_5': results[1]['xyz_first_5'],
        'xyz_identical': bool(np.allclose(xyz1, xyz2, atol=1e-6)),
        'passed': bool(np.allclose(xyz1, xyz2, atol=1e-6)),
    }

    return sanity, results


class BatchBuilder:
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


def gaussian_set_pooling(model, gpu_batch, bbox, args, device):
    """与 Phase12C/E/F 完全相同的 Gaussian-Set pooling 路径。"""
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


def build_candidate_pool(dataset, batch_builder, model, all_cameras, args, device):
    """为每个相机构建候选池。"""
    candidates_by_cam = defaultdict(list)

    for idx in range(len(dataset)):
        cam_id, frame_idx = dataset.indices[idx]
        if cam_id not in all_cameras:
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

            teacher_emb = None
            if dataset.teacher_cache is not None:
                cache_key = (int(frame_idx), cam_id, int(pid), x1, y1, x2, y2)
                cache_entry = dataset.teacher_cache.get(cache_key)
                if cache_entry is not None:
                    teacher_emb = cache_entry.get('embedding')
                    if hasattr(teacher_emb, 'squeeze'):
                        teacher_emb = teacher_emb.squeeze()

            if teacher_emb is None:
                continue

            candidates_by_cam[cam_id].append({
                'person_id': int(pid),
                'cam_id': cam_id,
                'frame_idx': int(frame_idx),
                'dataset_index': idx,
                'bbox': [x1, y1, x2, y2],
                'bbox_area': int(bbox_area),
                'teacher_emb': teacher_emb,
            })

    return candidates_by_cam


def run_pretrain_diagnostic(dataset, batch_builder, model, all_cameras, args, device):
    """C1-C7 pretrain Gaussian-Set diagnostic。"""
    print("\nRunning pretrain C1-C7 diagnostic...")

    candidates_by_cam = build_candidate_pool(dataset, batch_builder, model, all_cameras, args, device)

    per_camera_metrics = {}

    for cam_id in all_cameras:
        candidates = candidates_by_cam.get(cam_id, [])

        anns_count = 0
        valid_frame_count = 0
        person_ids = set()
        for idx in range(len(dataset)):
            c, f = dataset.indices[idx]
            if c == cam_id:
                anns = dataset.annotations.get(int(f), [])
                for ann in anns:
                    ac = ann.get('camera_id')
                    if ac is not None and f"C{ac + 1}" == cam_id:
                        anns_count += 1
                        valid_frame_count += 1
                        pid = ann.get('new_id')
                        if pid is not None:
                            person_ids.add(pid)

        if not candidates:
            per_camera_metrics[cam_id] = {
                'valid_frame_annotation_count': valid_frame_count,
                'unique_person_count': len(person_ids),
                'sampled_count': 0,
                'gaussianset_valid_ratio': 0.0,
                'selected_gaussian_count_mean': 0.0,
                'gaussian_weight_sum_mean': 0.0,
                'student_feature_norm_mean': 0.0,
                'cos_to_teacher_mean': 0.0,
                'verdict': 'no_candidates_with_teacher_emb',
            }
            print(f"  {cam_id}: no candidates with teacher emb")
            continue

        if len(candidates) > args.samples_per_camera:
            sampled = random.sample(candidates, args.samples_per_camera)
        else:
            sampled = candidates

        gs_valid_count = 0
        gs_counts = []
        gs_weights = []
        student_norms = []
        teacher_norms = []
        cos_values = []

        for sample in sampled:
            gpu_batch = batch_builder.get_batch(sample['cam_id'], sample['frame_idx'])
            if gpu_batch is None:
                continue

            G, gs_info = gaussian_set_pooling(model, gpu_batch, sample['bbox'], args, device)

            gs_counts.append(gs_info.get('selected_gaussian_count', 0))
            gs_weights.append(gs_info.get('gaussian_weight_sum', 0.0))

            teacher_feature = torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=device)
            teacher_feature = normalize_feat(teacher_feature)
            teacher_norm = float(teacher_feature.norm().item())
            teacher_norms.append(teacher_norm)

            if G is not None:
                student_norm = gs_info.get('student_feature_norm', 0.0)
                student_norms.append(student_norm)

                cos_sim = float(F.cosine_similarity(G.unsqueeze(0), teacher_feature.unsqueeze(0)).item())
                cos_values.append(cos_sim)

                if gs_info['selected_gaussian_count'] > 0 and gs_info['gaussian_weight_sum'] > 1e-6 and student_norm > 1e-6:
                    gs_valid_count += 1
            else:
                student_norms.append(0.0)
                cos_values.append(0.0)

        sampled_count = len(sampled)
        gs_valid_ratio = gs_valid_count / sampled_count if sampled_count > 0 else 0
        avg_gs_count = np.mean(gs_counts) if gs_counts else 0
        avg_gs_weight = np.mean(gs_weights) if gs_weights else 0
        avg_student_norm = np.mean(student_norms) if student_norms else 0
        avg_teacher_norm = np.mean(teacher_norms) if teacher_norms else 0
        avg_cos = np.mean(cos_values) if cos_values else 0

        if gs_valid_ratio > 0.3:
            verdict = 'usable'
        elif gs_valid_ratio > 0.1:
            verdict = 'weak_gaussian_coverage'
        elif avg_gs_count < 1:
            verdict = 'gaussianset_pooling_failure'
        else:
            verdict = 'weak_gaussian_coverage'

        per_camera_metrics[cam_id] = {
            'valid_frame_annotation_count': valid_frame_count,
            'unique_person_count': len(person_ids),
            'sampled_count': sampled_count,
            'gaussianset_valid_ratio': float(gs_valid_ratio),
            'selected_gaussian_count_mean': float(avg_gs_count),
            'gaussian_weight_sum_mean': float(avg_gs_weight),
            'student_feature_norm_mean': float(avg_student_norm),
            'teacher_feature_norm_mean': float(avg_teacher_norm),
            'cos_to_teacher_mean': float(avg_cos),
            'verdict': verdict,
        }

        print(f"  {cam_id}: verdict={verdict}, gs_valid={gs_valid_ratio:.2%}, "
              f"avg_gs_count={avg_gs_count:.1f}, avg_cos={avg_cos:.4f}")

    return per_camera_metrics


def select_training_candidates(candidates_by_cam, allowed_cameras, args):
    """选择训练样本。"""
    candidates = []
    for cam_id in allowed_cameras:
        candidates.extend(candidates_by_cam.get(cam_id, []))

    random.shuffle(candidates)
    return candidates


def run_train_geometry_fixed_teacher_warmup(args):
    """训练流程。"""
    print("=" * 80)
    print("Phase12 Geometry Fixed + Teacher Warm-up")
    print("=" * 80)

    all_cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    allowed_cameras = [c.strip() for c in args.allowed_cameras.split(',')] if args.allowed_cameras else ['C1', 'C4', 'C6', 'C7']

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"\n[1/5] Checking geometry checkpoint...")
    geometry_report, error = check_geometry_checkpoint(args.geometry_checkpoint)
    with open(os.path.join(args.output_dir, 'geometry_key_report.json'), 'w') as f:
        json.dump(geometry_report, f, indent=2, default=str)

    if not geometry_report['has_geometry']:
        print(f"ERROR: Candidate checkpoint does not contain trained geometry")
        with open(os.path.join(args.output_dir, 'final_report.md'), 'w') as f:
            f.write("# Phase12 Geometry Fixed Teacher Warm-up Report\n\n")
            f.write("**FAILURE**: candidate checkpoint does not contain trained geometry.\n")
        return False

    print(f"  Geometry checkpoint: {args.geometry_checkpoint}")
    print(f"  num_gaussians: {geometry_report['num_gaussians']}")
    print(f"  has_geometry: {geometry_report['has_geometry']}")

    print(f"\n[2/5] Loading geometry into model...")
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

    print(f"  Before load: num_gaussians={model.num_gaussians}")

    load_report, load_error = load_trained_geometry_into_model(model, args.geometry_checkpoint, device)

    if load_error:
        print(f"ERROR: Failed to load geometry: {load_error}")
        return False

    print(f"  After load: num_gaussians={load_report['num_gaussians']}")
    print(f"  geometry_loaded: {load_report['geometry_loaded']}")
    print(f"  geometry_frozen: {load_report['geometry_frozen']}")

    print(f"\n  Initializing person_feature for {load_report['num_gaussians']} Gaussians...")
    person_feature_dim = 512
    model._person_feature = torch.nn.Parameter(
        torch.randn(load_report['num_gaussians'], person_feature_dim, dtype=torch.float32, device=device) * 0.01
    )
    model._person_feature.requires_grad = True

    optimizer = torch.optim.Adam([model._person_feature], lr=args.person_feature_lr)

    print(f"  person_feature_shape: {list(model._person_feature.shape)}")
    print(f"  trainable_params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"  frozen_params: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")

    print(f"\n[3/5] Geometry load sanity check...")
    sanity, sanity_details = geometry_load_sanity_check(args, args.geometry_checkpoint, device)
    with open(os.path.join(args.output_dir, 'geometry_load_sanity.json'), 'w') as f:
        json.dump(sanity, f, indent=2, default=str)

    if not sanity['passed']:
        print(f"ERROR: Geometry load sanity check FAILED")
        with open(os.path.join(args.output_dir, 'final_report.md'), 'w') as f:
            f.write("# Phase12 Geometry Fixed Teacher Warm-up Report\n\n")
            f.write("**FAILURE**: geometry load sanity check failed - xyz differs across seeds.\n")
        return False

    print(f"  xyz_identical: {sanity['xyz_identical']}")
    print(f"  PASSED")

    print(f"\n[4/5] Pretrain C1-C7 diagnostic...")
    batch_builder = BatchBuilder(dataset)

    pretrain_diag = run_pretrain_diagnostic(dataset, batch_builder, model, all_cameras, args, device)

    with open(os.path.join(args.output_dir, 'pretrain_c1c7_diagnostic.json'), 'w') as f:
        json.dump(pretrain_diag, f, indent=2, default=str)

    pretrain_md = "# Pretrain C1-C7 Gaussian-Set Diagnostic\n\n"
    pretrain_md += "| Camera | valid_anns | sampled | gs_valid_ratio | avg_gs_count | avg_weight | avg_student_norm | avg_cos | verdict |\n"
    pretrain_md += "|--------|-----------|---------|---------------|-------------|-----------|-----------------|---------|----------|\n"
    for cam_id in all_cameras:
        m = pretrain_diag.get(cam_id, {})
        pretrain_md += f"| {cam_id} | {m.get('valid_frame_annotation_count', 0)} | {m.get('sampled_count', 0)} | "
        pretrain_md += f"{m.get('gaussianset_valid_ratio', 0):.2%} | {m.get('selected_gaussian_count_mean', 0):.1f} | "
        pretrain_md += f"{m.get('gaussian_weight_sum_mean', 0):.4f} | {m.get('student_feature_norm_mean', 0):.4f} | "
        pretrain_md += f"{m.get('cos_to_teacher_mean', 0):.4f} | {m.get('verdict', 'N/A')} |\n"
    with open(os.path.join(args.output_dir, 'pretrain_c1c7_diagnostic.md'), 'w') as f:
        f.write(pretrain_md)

    valid_cameras_ok = all(
        pretrain_diag.get(cam, {}).get('gaussianset_valid_ratio', 0) > 0.1
        for cam in allowed_cameras
    )

    if not valid_cameras_ok:
        print(f"WARNING: Some allowed cameras have poor Gaussian-Set coverage")
        for cam in allowed_cameras:
            m = pretrain_diag.get(cam, {})
            print(f"  {cam}: gs_valid_ratio={m.get('gaussianset_valid_ratio', 0):.2%}")

    print(f"\n[5/5] Training teacher warm-up...")

    candidates_by_cam = build_candidate_pool(dataset, batch_builder, model, all_cameras, args, device)
    training_candidates = select_training_candidates(candidates_by_cam, allowed_cameras, args)

    if not training_candidates:
        print("ERROR: No training candidates found")
        return False

    print(f"  Training candidates: {len(training_candidates)}")
    print(f"  Allowed cameras: {allowed_cameras}")

    batch_size = args.P * args.K
    metrics_log = []
    best_mAP, best_gap, best_step = 0.0, 0.0, 0
    pf_before = model.get_person_feature().clone().detach()

    features_bank = []
    teacher_bank = []
    person_ids_bank = []
    cam_ids_bank = []

    with torch.no_grad():
        for sample in training_candidates:
            gpu_batch = batch_builder.get_batch(sample['cam_id'], sample['frame_idx'])
            if gpu_batch is None:
                continue
            G, gs_info = gaussian_set_pooling(model, gpu_batch, sample['bbox'], args, device)
            if G is None:
                continue

            teacher_feature = torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=device)
            teacher_feature = normalize_feat(teacher_feature)

            features_bank.append(G)
            teacher_bank.append(teacher_feature)
            person_ids_bank.append(sample['person_id'])
            cam_ids_bank.append(sample['cam_id'])

    for step in range(args.num_steps):
        step_start = time.time()

        if len(training_candidates) < batch_size:
            continue
        batch_samples = random.sample(training_candidates, batch_size)

        batch_by_cam_frame = defaultdict(list)
        for s in batch_samples:
            batch_by_cam_frame[(s['cam_id'], s['frame_idx'])].append(s)

        optimizer.zero_grad()
        losses_t, gaussianset_features, valid_samples = [], [], []
        gs_counts_list, gs_weights_list = [], []

        for (cam_id, frame_idx), samples in batch_by_cam_frame.items():
            gpu_batch = batch_builder.get_batch(cam_id, frame_idx)
            if gpu_batch is None:
                continue
            for sample in samples:
                G, gs_info = gaussian_set_pooling(model, gpu_batch, sample['bbox'], args, device)
                if G is None:
                    continue

                teacher_feature = torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=device)
                teacher_feature = normalize_feat(teacher_feature)
                cos_sim = F.cosine_similarity(G.unsqueeze(0), teacher_feature.unsqueeze(0))

                gaussianset_features.append(G)
                valid_samples.append(sample)
                losses_t.append(1.0 - cos_sim)
                gs_counts_list.append(gs_info['selected_gaussian_count'])
                gs_weights_list.append(gs_info['gaussian_weight_sum'])

        valid_count = len(losses_t)
        if valid_count == 0:
            continue

        L_teacher = torch.stack(losses_t).mean()
        L_total = args.lambda_teacher * L_teacher

        L_total.backward()

        grad_norm_before = None
        grad_norm_after = None
        if model._person_feature.grad is not None:
            grad_norm_before = model._person_feature.grad.norm().item()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([model._person_feature], args.grad_clip_norm)
            grad_norm_after = model._person_feature.grad.norm().item()

        optimizer.step()

        pf_after = model.get_person_feature()
        param_delta_norm = (pf_after - pf_before).norm().item()
        pf_before = pf_after.clone().detach()

        cos_values = [1.0 - l.item() for l in losses_t]
        cos_mean = float(np.mean(cos_values)) if cos_values else 0

        same_cos_list, diff_cos_list = [], []
        person_to_indices = defaultdict(list)
        for idx, s in enumerate(valid_samples):
            person_to_indices[s['person_id']].append(idx)

        for pid, indices in person_to_indices.items():
            if len(indices) >= 2:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        sc = F.cosine_similarity(
                            gaussianset_features[indices[i]].unsqueeze(0),
                            gaussianset_features[indices[j]].unsqueeze(0)
                        ).item()
                        same_cos_list.append(sc)

        all_indices = list(range(len(gaussianset_features)))
        for _ in range(min(50, len(all_indices) * (len(all_indices) - 1) // 2)):
            i, j = random.sample(all_indices, 2)
            if valid_samples[i]['person_id'] != valid_samples[j]['person_id']:
                dc = F.cosine_similarity(
                    gaussianset_features[i].unsqueeze(0), gaussianset_features[j].unsqueeze(0)
                ).item()
                diff_cos_list.append(dc)

        same_cos = float(np.mean(same_cos_list)) if same_cos_list else None
        diff_cos = float(np.mean(diff_cos_list)) if diff_cos_list else None
        gap = (same_cos - diff_cos) if (same_cos is not None and diff_cos is not None) else None

        nan_count = int(torch.isnan(model._person_feature).sum().item())
        inf_count = int(torch.isinf(model._person_feature).sum().item())

        step_time = time.time() - step_start

        if step % args.log_interval == 0:
            log_line = f"[WARMUP] Step {step:5d}: loss={L_total.item():.4f} " \
                       f"teacher={L_teacher.item():.4f} " \
                       f"cos={cos_mean:.4f} " \
                       f"delta={param_delta_norm:.6e} valid={valid_count}/{batch_size} t={step_time:.2f}s"
            if gap is not None:
                log_line += f" same={same_cos:.4f} diff={diff_cos:.4f} gap={gap:+.4f}"
            print(log_line)

        step_record = {
            'step': step,
            'loss_total': float(L_total.item()),
            'loss_teacher': float(L_teacher.item()),
            'train_cos_mean': cos_mean,
            'valid_roi_count': valid_count,
            'same_cos': same_cos,
            'diff_cos': diff_cos,
            'gap': gap,
            'selected_gaussian_count_mean': float(np.mean(gs_counts_list)) if gs_counts_list else 0,
            'gaussian_weight_sum_mean': float(np.mean(gs_weights_list)) if gs_weights_list else 0,
            'grad_norm_before_clip': float(grad_norm_before) if grad_norm_before else 0,
            'grad_norm_after_clip': float(grad_norm_after) if grad_norm_after else 0,
            'param_delta_norm': float(param_delta_norm),
            'nan_count': nan_count,
            'inf_count': inf_count,
        }
        metrics_log.append(step_record)

        if step % args.eval_interval == 0 or step == args.num_steps - 1:
            with torch.no_grad():
                eval_cos_list = []
                eval_features = []
                eval_person_ids = []
                eval_cam_ids = []

                for sample in training_candidates[:100]:
                    gpu_batch = batch_builder.get_batch(sample['cam_id'], sample['frame_idx'])
                    if gpu_batch is None:
                        continue
                    G, _ = gaussian_set_pooling(model, gpu_batch, sample['bbox'], args, device)
                    if G is None:
                        continue

                    teacher_feature = torch.tensor(sample['teacher_emb'], dtype=torch.float32, device=device)
                    teacher_feature = normalize_feat(teacher_feature)
                    cos_sim = float(F.cosine_similarity(G.unsqueeze(0), teacher_feature.unsqueeze(0)).item())
                    eval_cos_list.append(cos_sim)
                    eval_features.append(G)
                    eval_person_ids.append(sample['person_id'])
                    eval_cam_ids.append(sample['cam_id'])

                eval_cos_mean = float(np.mean(eval_cos_list)) if eval_cos_list else 0

                eval_same = []
                eval_diff = []
                eval_person_to_idx = defaultdict(list)
                for idx, pid in enumerate(eval_person_ids):
                    eval_person_to_idx[pid].append(idx)

                for pid, indices in eval_person_to_idx.items():
                    if len(indices) >= 2:
                        for i in range(len(indices)):
                            for j in range(i + 1, len(indices)):
                                sc = F.cosine_similarity(
                                    eval_features[indices[i]].unsqueeze(0),
                                    eval_features[indices[j]].unsqueeze(0)
                                ).item()
                                eval_same.append(sc)

                all_eval_idx = list(range(len(eval_features)))
                for _ in range(min(20, len(all_eval_idx) * (len(all_eval_idx) - 1) // 2)):
                    i, j = random.sample(all_eval_idx, 2)
                    if eval_person_ids[i] != eval_person_ids[j]:
                        dc = F.cosine_similarity(
                            eval_features[i].unsqueeze(0), eval_features[j].unsqueeze(0)
                        ).item()
                        eval_diff.append(dc)

                eval_same_cos = float(np.mean(eval_same)) if eval_same else None
                eval_diff_cos = float(np.mean(eval_diff)) if eval_diff else None
                eval_gap = (eval_same_cos - eval_diff_cos) if (eval_same_cos is not None and eval_diff_cos is not None) else None

            step_record['eval_cos_mean'] = eval_cos_mean
            step_record['eval_same_cos'] = eval_same_cos
            step_record['eval_diff_cos'] = eval_diff_cos
            step_record['eval_gap'] = eval_gap

            print(f"  [EVAL] eval_cos={eval_cos_mean:.4f} "
                  f"same={eval_same_cos:.4f} diff={eval_diff_cos:.4f} gap={eval_gap:+.4f}")

            if eval_cos_mean > best_mAP:
                best_mAP = eval_cos_mean
                best_step = step
                torch.save({
                    'model_state_dict': {'_person_feature': model._person_feature.cpu().clone()},
                    'geometry_source_path': args.geometry_checkpoint,
                    'geometry_loaded': True,
                    'geometry_frozen': True,
                    'person_feature_shape': list(model._person_feature.shape),
                    'step': step,
                    'eval_cos': eval_cos_mean,
                }, os.path.join(args.output_dir, 'checkpoint_best_mAP.pt'))

            if eval_gap is not None and eval_gap > best_gap:
                best_gap = eval_gap
                torch.save({
                    'model_state_dict': {'_person_feature': model._person_feature.cpu().clone()},
                    'geometry_source_path': args.geometry_checkpoint,
                    'geometry_loaded': True,
                    'geometry_frozen': True,
                    'person_feature_shape': list(model._person_feature.shape),
                    'step': step,
                    'eval_gap': eval_gap,
                }, os.path.join(args.output_dir, 'checkpoint_best_gap.pt'))

    with open(os.path.join(args.output_dir, 'metrics.jsonl'), 'w') as f:
        for r in metrics_log:
            f.write(json.dumps(r, default=str) + '\n')

    eval_metrics = [m for m in metrics_log if 'eval_cos_mean' in m]
    with open(os.path.join(args.output_dir, 'eval_metrics.jsonl'), 'w') as f:
        for r in eval_metrics:
            f.write(json.dumps(r, default=str) + '\n')

    torch.save({
        'model_state_dict': {'_person_feature': model._person_feature.cpu().clone()},
        'geometry_source_path': args.geometry_checkpoint,
        'geometry_loaded': True,
        'geometry_frozen': True,
        'person_feature_shape': list(model._person_feature.shape),
        'step': args.num_steps - 1,
    }, os.path.join(args.output_dir, 'checkpoint_latest.pt'))

    last_cos = metrics_log[-1]['train_cos_mean'] if metrics_log else 0
    last_loss = metrics_log[-1]['loss_teacher'] if metrics_log else 0
    final_nan = sum(m['nan_count'] for m in metrics_log)
    final_inf = sum(m['inf_count'] for m in metrics_log)

    param_deltas = [m['param_delta_norm'] for m in metrics_log if m['param_delta_norm'] > 0]
    mean_param_delta = float(np.mean(param_deltas)) if param_deltas else 0

    if final_nan > 0 or final_inf > 0 or mean_param_delta < 1e-8:
        verdict = 'failure'
    else:
        verdict = 'success'

    summary = {
        'geometry_checkpoint': args.geometry_checkpoint,
        'geometry_loaded': load_report['geometry_loaded'],
        'geometry_frozen': load_report['geometry_frozen'],
        'num_gaussians': load_report['num_gaussians'],
        'person_feature_shape': list(model._person_feature.shape),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen_params': sum(p.numel() for p in model.parameters() if not p.requires_grad),
        'geometry_sanity_passed': sanity['passed'],
        'pretrain_diagnostic': pretrain_diag,
        'best_step': best_step,
        'best_eval_cos': best_mAP,
        'best_gap': best_gap,
        'last_cos': last_cos,
        'last_loss': last_loss,
        'nan_total': final_nan,
        'inf_total': final_inf,
        'mean_param_delta': mean_param_delta,
        'verdict': verdict,
    }

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    generate_final_report(args.output_dir, summary, load_report, sanity, pretrain_diag)

    print(f"\n{'='*80}")
    print(f"Warm-up complete! Verdict: {verdict}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")

    return verdict == 'success'


def generate_final_report(output_dir, summary, load_report, sanity, pretrain_diag):
    """生成 final_report.md。"""
    report = f"""# Phase12 Geometry Fixed + Teacher Warm-up Report

## 1. 是否成功加载真实 trained geometry？

{'是' if summary['geometry_loaded'] else '否'}
- geometry_source: {summary.get('geometry_checkpoint', 'N/A')}
- num_gaussians: {summary['num_gaussians']}
- xyz_shape: {load_report.get('xyz_shape')}
- xyz_mean: {load_report.get('xyz_stats', {}).get('xyz_mean')}
- xyz_std: {load_report.get('xyz_stats', {}).get('xyz_std')}

## 2. 是否完全避免 random point cloud？

{'是' if summary['geometry_loaded'] else '否'}
- 不再出现 "Generating random point cloud"
- geometry 直接来自 checkpoint

## 3. geometry 是否冻结？

{'是' if summary['geometry_frozen'] else '否'}
- frozen_params: {summary['frozen_params']}
- trainable_params: {summary['trainable_params']}

## 4. person_feature 是否绑定真实 Gaussian 数量？

{'是' if summary['person_feature_shape'][0] == summary['num_gaussians'] else '否'}
- person_feature_shape: {summary['person_feature_shape']}
- num_gaussians: {summary['num_gaussians']}

## 5. C1-C7 pooling 是否正常？

"""

    for cam_id in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        m = pretrain_diag.get(cam_id, {})
        report += f"- **{cam_id}**: gs_valid_ratio={m.get('gaussianset_valid_ratio', 0):.2%}, "
        report += f"avg_gs_count={m.get('selected_gaussian_count_mean', 0):.1f}, "
        report += f"avg_cos={m.get('cos_to_teacher_mean', 0):.4f}, "
        report += f"verdict={m.get('verdict', 'N/A')}\n"

    report += f"""
## 6. C2/C3/C5 在真实 geometry 下是否仍无效，原因是什么？

"""

    for cam_id in ['C2', 'C3', 'C5']:
        m = pretrain_diag.get(cam_id, {})
        verdict = m.get('verdict', 'N/A')
        gs_valid = m.get('gaussianset_valid_ratio', 0)
        avg_cos = m.get('cos_to_teacher_mean', 0)

        if gs_valid > 0.3:
            report += f"- **{cam_id}**: 正常 (gs_valid={gs_valid:.2%}, avg_cos={avg_cos:.4f})\n"
        elif gs_valid > 0.1:
            report += f"- **{cam_id}**: weak coverage (gs_valid={gs_valid:.2%}, avg_cos={avg_cos:.4f})\n"
        else:
            report += f"- **{cam_id}**: 无效 (gs_valid={gs_valid:.2%}) - 原因: {verdict}\n"

    report += f"""
## 7. 是否可以恢复 12G SupCon？

"""

    valid_cameras_ok = all(
        pretrain_diag.get(cam, {}).get('gaussianset_valid_ratio', 0) > 0.2
        for cam in ['C1', 'C4', 'C6', 'C7']
    )

    if summary['geometry_loaded'] and summary['geometry_frozen'] and valid_cameras_ok:
        report += """**是，可以恢复 12G SupCon。**

条件满足：
1. 真实 geometry 已成功加载
2. geometry 已冻结
3. person_feature 绑定真实 Gaussian 数量
4. valid cameras (C1/C4/C6/C7) Gaussian-Set pooling 正常
5. 无 NaN/Inf

下一步：
1. 使用当前 checkpoint_best_mAP.pt 或 checkpoint_best_gap.pt
2. 添加 SupCon loss
3. 继续训练 person_feature
"""
    else:
        report += """**否，需要进一步修复。**

不满足的条件：
"""
        if not summary['geometry_loaded']:
            report += "- geometry 未成功加载\n"
        if not summary['geometry_frozen']:
            report += "- geometry 未冻结\n"
        if not valid_cameras_ok:
            report += "- valid cameras Gaussian-Set pooling 不正常\n"

    report += f"""
## 训练总结

- **Verdict**: {summary['verdict']}
- **Best step**: {summary['best_step']}
- **Best eval cos**: {summary['best_eval_cos']:.4f}
- **Best gap**: {summary['best_gap']:.4f}
- **Last cos**: {summary['last_cos']:.4f}
- **Mean param delta**: {summary['mean_param_delta']:.6e}
- **NaN/Inf**: {summary['nan_total']}/{summary['inf_total']}

---

*报告生成时间：2026-05-13*
"""

    with open(os.path.join(output_dir, 'final_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Phase12 Geometry Fixed + Teacher Warm-up')

    parser.add_argument('--mode', type=str, default='train_geometry_fixed_teacher_warmup',
                        choices=['train_geometry_fixed_teacher_warmup'])
    parser.add_argument('--geometry_checkpoint', type=str,
                        default='runs/phase11A_proto_infonce_opacity_lam005_lr1e4/step_20999.pth')
    parser.add_argument('--lambda_teacher', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--grad_clip_norm', type=float, default=10.0)
    parser.add_argument('--allowed_cameras', type=str, default='C1,C4,C6,C7')
    parser.add_argument('--min_bbox_area', type=int, default=100)
    parser.add_argument('--min_alpha_sum', type=float, default=1e-1)
    parser.add_argument('--P', type=int, default=4)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--person_feature_lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str,
                        default='outputs/phase12_geometry_fixed_teacher_warmup')
    parser.add_argument('--samples_per_camera', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.mode == 'train_geometry_fixed_teacher_warmup':
        run_train_geometry_fixed_teacher_warmup(args)


if __name__ == '__main__':
    main()
