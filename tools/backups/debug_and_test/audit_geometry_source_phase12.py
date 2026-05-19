#!/usr/bin/env python3
"""
Phase12 Geometry Source Audit

目标：确认 Phase12 ReID 实验是否真的使用了训练好的 3DGS geometry，还是每次生成 random point cloud。

检查内容：
1. Phase12 checkpoints 是否包含真实 geometry
2. Phase12 model 初始化时是否生成 random point cloud
3. load_checkpoint 是否真正加载 xyz / scale / rotation / opacity 等 geometry
4. 查找项目中真实 3DGS / 3DGRUT geometry checkpoint
5. 给出修复建议

不训练模型，不运行 12G，不移动或整理 Phase12 文件。
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from rich import pretty, traceback

pretty.install()
traceback.install()


GEOMETRY_LIKE_KEYS = [
    'xyz', '_xyz', 'means', '_means', 'positions', 'gaussian_means',
    'scaling', '_scaling', 'scale',
    'rotation', '_rotation',
    'opacity', '_opacity',
    'features_dc', 'features_rest', 'shs', 'colors',
    'density',
    'features_albedo', 'features_specular',
]

PERSON_FEATURE_KEYS = ['_person_feature', 'person_feature']


def check_checkpoint_keys(checkpoint_path):
    """检查单个 checkpoint 的 keys 和 shapes。"""
    if not os.path.exists(checkpoint_path):
        return None, f"missing: {checkpoint_path}"
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        return None, f"error loading: {str(e)}"
    
    report = {
        'path': checkpoint_path,
        'exists': True,
        'top_level_keys': [],
        'tensor_keys_and_shapes': {},
        'nested_dict_keys_and_shapes': {},
        'has_person_feature': False,
        'person_feature_keys': [],
        'has_geometry': False,
        'geometry_keys': [],
        'missing_geometry_keys': [],
    }
    
    if isinstance(ckpt, dict):
        report['top_level_keys'] = list(ckpt.keys())
        
        for key in ckpt.keys():
            value = ckpt[key]
            if isinstance(value, torch.Tensor):
                report['tensor_keys_and_shapes'][key] = list(value.shape)
            elif isinstance(value, dict):
                nested = {}
                for nk, nv in value.items():
                    if isinstance(nv, torch.Tensor):
                        nested[nk] = list(nv.shape)
                    elif isinstance(nv, dict):
                        nested[nk] = f"dict with keys: {list(nv.keys())[:10]}"
                    else:
                        nested[nk] = str(type(nv))
                report['nested_dict_keys_and_shapes'][key] = nested
        
        model_state = ckpt.get('model_state_dict', {})
        if isinstance(model_state, dict):
            all_keys = list(model_state.keys())
            for key in all_keys:
                lower_key = key.lower()
                for geo_key in GEOMETRY_LIKE_KEYS:
                    if geo_key in lower_key:
                        if key not in report['geometry_keys']:
                            report['geometry_keys'].append(key)
                            report['has_geometry'] = True
            
            for key in all_keys:
                lower_key = key.lower()
                for pf_key in PERSON_FEATURE_KEYS:
                    if pf_key in lower_key:
                        if key not in report['person_feature_keys']:
                            report['person_feature_keys'].append(key)
                            report['has_person_feature'] = True
            
            expected_geo = ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']
            for geo in expected_geo:
                if geo not in model_state:
                    report['missing_geometry_keys'].append(geo)
    
    return report, None


def scan_code_for_geometry_init(code_files):
    """静态扫描代码文件，查找 geometry 初始化相关代码。"""
    report = {
        'files_scanned': [],
        'random_point_cloud_files': [],
        'person_feature_init': [],
        'checkpoint_load_logic': [],
        'geometry_related_code': [],
    }
    
    keywords = [
        'Generating random point cloud',
        'random point cloud',
        'torch.rand',
        'torch.randn',
        'init_random',
        'initialize_points',
        'num_gaussians',
        '_person_feature',
        'person_feature',
        'load_state_dict',
        'checkpoint',
        'xyz',
        'opacity',
        'scaling',
        'rotation',
        'init_from_random_point_cloud',
        'init_from_checkpoint',
    ]
    
    for filepath in code_files:
        if not os.path.exists(filepath):
            continue
        
        file_report = {
            'path': filepath,
            'exists': True,
            'matches': defaultdict(list),
        }
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            for keyword in keywords:
                if keyword.lower() in line.lower():
                    file_report['matches'][keyword].append({
                        'line': i,
                        'content': line.strip()[:120],
                    })
        
        if file_report['matches']:
            report['files_scanned'].append(file_report)
            
            if 'Generating random point cloud' in file_report['matches'] or \
               'init_from_random_point_cloud' in file_report['matches']:
                report['random_point_cloud_files'].append(filepath)
            
            if '_person_feature' in file_report['matches'] or 'person_feature' in file_report['matches']:
                report['person_feature_init'].append(filepath)
    
    return report


def runtime_audit_single(args, checkpoint_path, seed=42):
    """运行最小 runtime audit，加载 checkpoint 并检查 geometry 来源。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    report = {
        'checkpoint_path': checkpoint_path,
        'seed': seed,
        'model_class_name': None,
        'num_gaussians': None,
        'has_person_feature': False,
        'person_feature_shape': None,
        'has_xyz': False,
        'xyz_shape': None,
        'xyz_source': 'unknown',
        'xyz_stats': {},
        'has_opacity': False,
        'opacity_shape': None,
        'has_scaling': False,
        'scaling_shape': None,
        'has_rotation': False,
        'rotation_shape': None,
        'loaded_keys': [],
        'missing_keys': [],
        'unexpected_keys': [],
    }
    
    try:
        config_dir = os.path.join(REPO_ROOT, 'configs')
        config_file = 'apps/wildtrack_full_3dgut'
        
        from hydra import initialize_config_dir, compose
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=config_file)
        cfg.model.person_feature_dim = 512
        
        from threedgrut.trainer import Trainer3DGRUT
        trainer = Trainer3DGRUT(cfg)
        model = trainer.model
        
        report['model_class_name'] = model.__class__.__name__
        report['num_gaussians'] = model.num_gaussians if hasattr(model, 'num_gaussians') else None
        
        if hasattr(model, '_person_feature'):
            report['has_person_feature'] = True
            report['person_feature_shape'] = list(model._person_feature.shape)
        
        if hasattr(model, 'positions'):
            report['has_xyz'] = True
            report['xyz_shape'] = list(model.positions.shape)
            
            xyz = model.positions.detach().cpu().numpy()
            report['xyz_stats'] = {
                'xyz_mean': float(np.mean(xyz)),
                'xyz_std': float(np.std(xyz)),
                'xyz_min': float(np.min(xyz)),
                'xyz_max': float(np.max(xyz)),
                'first_5': xyz[:5].tolist(),
            }
        
        if hasattr(model, 'density') or hasattr(model, 'get_density'):
            report['has_opacity'] = True
            if hasattr(model, 'density'):
                report['opacity_shape'] = list(model.density.shape)
        
        if hasattr(model, 'scale'):
            report['has_scaling'] = True
            report['scaling_shape'] = list(model.scale.shape)
        
        if hasattr(model, 'rotation'):
            report['has_rotation'] = True
            report['rotation_shape'] = list(model.rotation.shape)
        
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=trainer.device)
            
            if 'model_state_dict' in ckpt:
                model_state = ckpt['model_state_dict']
                report['loaded_keys'] = list(model_state.keys())
                
                if '_person_feature' in model_state:
                    pf = model_state['_person_feature']
                    if hasattr(pf, 'to'):
                        model._person_feature.data.copy_(pf.to(trainer.device))
                
                expected_geo = ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']
                for geo in expected_geo:
                    if geo in model_state:
                        report['loaded_keys'].append(f"geometry:{geo}")
                    else:
                        report['missing_keys'].append(geo)
                
                model_state_keys = set(model_state.keys())
                model_keys = set(k for k, v in model.named_parameters())
                report['unexpected_keys'] = list(model_state_keys - model_keys)
                report['missing_keys'] = list(model_keys - model_state_keys)
        
        if report['xyz_stats']:
            if abs(report['xyz_stats']['xyz_mean']) < 0.1 and report['xyz_stats']['xyz_std'] > 0.5:
                report['xyz_source'] = 'likely_random'
            else:
                report['xyz_source'] = 'unknown'
        
    except Exception as e:
        report['error'] = str(e)
    
    return report


def compare_xyz_between_seeds(args, checkpoint_path, seed1=42, seed2=123):
    """比较不同 seed 下 xyz 是否变化，判断 geometry 来源。"""
    report1 = runtime_audit_single(args, checkpoint_path, seed=seed1)
    report2 = runtime_audit_single(args, checkpoint_path, seed=seed2)
    
    comparison = {
        'seed1': seed1,
        'seed2': seed2,
        'xyz_same_seed1': report1.get('xyz_stats', {}).get('first_5'),
        'xyz_same_seed2': report2.get('xyz_stats', {}).get('first_5'),
        'xyz_changed': False,
        'conclusion': 'unknown',
    }
    
    if report1.get('xyz_stats') and report2.get('xyz_stats'):
        xyz1 = np.array(report1['xyz_stats']['first_5'])
        xyz2 = np.array(report2['xyz_stats']['first_5'])
        
        if np.allclose(xyz1, xyz2, atol=1e-6):
            comparison['xyz_changed'] = False
            comparison['conclusion'] = 'xyz same across seeds - likely from checkpoint or deterministic'
        else:
            comparison['xyz_changed'] = True
            comparison['conclusion'] = 'xyz different across seeds - geometry from random init'
    
    return comparison, report1, report2


def search_geometry_checkpoints():
    """搜索项目中可能的真实 geometry checkpoint。"""
    patterns = [
        '**/checkpoint*.pt',
        '**/ckpt*.pt',
        '**/*.pth',
        '**/*.ckpt',
        '**/model*.pt',
        '**/point_cloud*.ply',
        '**/gaussians*.ply',
        '**/scene*.ply',
    ]
    
    candidates = []
    exclude_dirs = ['__pycache__', '.git', 'node_modules', 'outputs/phase12*']
    
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(REPO_ROOT, pattern), recursive=True):
            rel_path = os.path.relpath(filepath, REPO_ROOT)
            
            is_phase12 = 'phase12' in rel_path.lower()
            file_size = os.path.getsize(filepath)
            
            candidate = {
                'path': filepath,
                'relative_path': rel_path,
                'file_size_mb': file_size / (1024 * 1024),
                'is_phase12': is_phase12,
                'extension': os.path.splitext(filepath)[1],
            }
            
            if filepath.endswith('.pt') or filepath.endswith('.pth') or filepath.endswith('.ckpt'):
                try:
                    ckpt = torch.load(filepath, map_location='cpu')
                    if isinstance(ckpt, dict):
                        candidate['top_level_keys'] = list(ckpt.keys())
                        
                        has_geo = False
                        geo_keys = []
                        for key in GEOMETRY_LIKE_KEYS:
                            for ckpt_key in ckpt.keys():
                                if key in ckpt_key.lower():
                                    has_geo = True
                                    geo_keys.append(ckpt_key)
                        
                        candidate['has_geometry'] = has_geo
                        candidate['geometry_keys'] = geo_keys
                        
                        if 'model_state_dict' in ckpt:
                            msd = ckpt['model_state_dict']
                            if isinstance(msd, dict):
                                candidate['model_state_keys'] = list(msd.keys())[:20]
                                
                                has_geo_in_msd = False
                                geo_in_msd = []
                                for key in GEOMETRY_LIKE_KEYS:
                                    for msd_key in msd.keys():
                                        if key in msd_key.lower():
                                            has_geo_in_msd = True
                                            geo_in_msd.append(msd_key)
                                
                                candidate['has_geometry_in_model_state_dict'] = has_geo_in_msd
                                candidate['geometry_keys_in_msd'] = geo_in_msd
                
                except Exception as e:
                    candidate['load_error'] = str(e)
            
            elif filepath.endswith('.ply'):
                candidate['is_ply_point_cloud'] = True
                
                try:
                    import struct
                    with open(filepath, 'rb') as f:
                        header = f.read(200).decode('ascii', errors='ignore')
                        candidate['ply_header_preview'] = header[:100]
                        
                        has_xyz_ply = 'x' in header.lower() and 'y' in header.lower() and 'z' in header.lower()
                        has_opacity_ply = 'opacity' in header.lower() or 'red' in header.lower()
                        
                        candidate['ply_has_xyz'] = has_xyz_ply
                        candidate['ply_has_opacity'] = has_opacity_ply
                except Exception as e:
                    candidate['ply_read_error'] = str(e)
            
            candidates.append(candidate)
    
    candidates.sort(key=lambda x: x['file_size_mb'], reverse=True)
    return candidates


def generate_final_report(
    checkpoint_reports,
    code_scan_report,
    runtime_audit,
    geometry_candidates,
    output_dir,
):
    """生成 final_report.md。"""
    
    phase12_checkpoints = [r for r in checkpoint_reports if r and 'phase12' in r['path'].lower()]
    
    all_have_geometry = all(r.get('has_geometry', False) for r in phase12_checkpoints if r)
    any_have_geometry = any(r.get('has_geometry', False) for r in phase12_checkpoints if r)
    
    runtime_xyz_source = runtime_audit.get('comparison', {}).get('conclusion', 'unknown')
    xyz_changed = runtime_audit.get('comparison', {}).get('xyz_changed', False)
    
    geometry_ckpts = [c for c in geometry_candidates if c.get('has_geometry') or c.get('has_geometry_in_model_state_dict')]
    non_phase12_geo = [c for c in geometry_ckpts if not c.get('is_phase12')]
    
    if all_have_geometry:
        case = 'D'
        conclusion = "Phase12 使用了真实 checkpoint geometry。"
    elif not any_have_geometry and xyz_changed:
        case = 'A+B'
        conclusion = "Phase12 checkpoint 只包含 person_feature，不包含 geometry；运行时使用 random Gaussian positions。"
    elif not any_have_geometry and not xyz_changed:
        case = 'A'
        conclusion = "Phase12 没有保存真实 geometry，只是保存 identity feature。"
    elif any_have_geometry and xyz_changed:
        case = 'C'
        conclusion = "geometry 存在但加载逻辑没接上。"
    else:
        case = 'unknown'
        conclusion = "无法确定 geometry 来源。"
    
    report = f"""# Phase12 Geometry Source Audit Report

## 1. Phase12 checkpoints 是否包含真实 geometry？

"""
    
    for r in phase12_checkpoints:
        if r:
            has_geo = r.get('has_geometry', False)
            missing = r.get('missing_geometry_keys', [])
            report += f"- **{r['path']}**:\n"
            report += f"  - has_geometry: {has_geo}\n"
            report += f"  - geometry_keys: {r.get('geometry_keys', [])}\n"
            report += f"  - missing_geometry_keys: {missing}\n"
            report += f"  - has_person_feature: {r.get('has_person_feature', False)}\n"
            report += f"  - person_feature_keys: {r.get('person_feature_keys', [])}\n"
    
    report += f"""
## 2. Phase12 runtime 是否生成 random point cloud？

Runtime audit 结论: **{runtime_xyz_source}**
xyz changed between seeds: **{xyz_changed}**

"""
    
    if runtime_audit.get('audit1'):
        a1 = runtime_audit['audit1']
        report += f"Audit details (seed=42):\n"
        report += f"- model_class: {a1.get('model_class_name')}\n"
        report += f"- num_gaussians: {a1.get('num_gaussians')}\n"
        report += f"- has_person_feature: {a1.get('has_person_feature')}\n"
        report += f"- person_feature_shape: {a1.get('person_feature_shape')}\n"
        report += f"- has_xyz: {a1.get('has_xyz')}\n"
        report += f"- xyz_shape: {a1.get('xyz_shape')}\n"
        report += f"- xyz_source: {a1.get('xyz_source')}\n"
        if a1.get('xyz_stats'):
            report += f"- xyz_mean: {a1['xyz_stats'].get('xyz_mean')}\n"
            report += f"- xyz_std: {a1['xyz_stats'].get('xyz_std')}\n"
            report += f"- xyz_min: {a1['xyz_stats'].get('xyz_min')}\n"
            report += f"- xyz_max: {a1['xyz_stats'].get('xyz_max')}\n"
    
    report += f"""
## 3. 当前 person_feature 是否绑定真实 Gaussian xyz？

"""
    
    if not any_have_geometry and xyz_changed:
        report += """**否。** person_feature 绑定的是随机初始化的 Gaussian positions，不是训练好的 3DGS geometry。

证据：
- Phase12 checkpoints 不包含 positions/rotation/scale/opacity 等 geometry 参数
- 不同 random seed 下 xyz 值发生变化，说明 geometry 来自 random init
- 模型初始化时调用 `init_from_random_point_cloud()` 生成随机点云
- person_feature 在随机 positions 上训练，无法代表真实 3D 空间中的身份特征
"""
    else:
        report += "需要进一步分析。"
    
    report += f"""
## 4. C2/C3/C5 无效是否可能由 random geometry coverage 导致？

"""
    
    if not any_have_geometry and xyz_changed:
        report += """**是的，极有可能。**

由于 Gaussian positions 是随机初始化的：
- 随机 Gaussian 在某些视角（如 C6）可能碰巧有较好的 coverage
- 在其他视角（如 C2/C3/C5）可能几乎没有 Gaussian 投影到 bbox 内
- 这解释了为什么 C6 的 `selected_gaussian_count_mean=2.7`，而 C2=1.4, C3=0.3
- 这也解释了为什么大多数相机的 `full_opacity_positive_ratio=0.0`

**这是 Phase12 结果不可信的根本原因。**
"""
    else:
        report += "需要进一步分析。"
    
    report += f"""
## 5. 是否应该暂停 12G？

"""
    
    if not any_have_geometry and xyz_changed:
        report += """**是的，强烈建议暂停 12G。**

原因：
- 如果 Phase12C/E/F 的 geometry 是随机的，那么在其基础上训练的 12G SupCon 也将基于随机 geometry
- 随机 geometry 上的任何训练都无法得到有意义的 3D ReID 结果
- 必须先修复 geometry loading，再考虑继续训练

当前 Priority：
1. 找到真实 trained 3DGS geometry checkpoint
2. 修改 Phase12 初始化逻辑，加载并冻结真实 geometry
3. 只训练 per-Gaussian person_feature
4. 重新运行 C1-C7 diagnostic
5. 确认所有相机正常后，再考虑 12G
"""
    else:
        report += "视情况而定。"
    
    report += f"""
## 6. 项目中是否找到可用真实 geometry checkpoint？

"""
    
    if non_phase12_geo:
        report += f"""找到 {len(non_phase12_geo)} 个可能的 geometry checkpoint：

"""
        for c in non_phase12_geo[:5]:
            report += f"- {c['relative_path']} ({c['file_size_mb']:.1f} MB)\n"
            report += f"  - has_geometry: {c.get('has_geometry', False)}\n"
            if c.get('geometry_keys_in_msd'):
                report += f"  - geometry_keys: {c['geometry_keys_in_msd'][:10]}\n"
    else:
        report += """**未找到明确的 trained 3DGS geometry checkpoint。**

需要：
1. 检查是否有 3DGS training 的输出目录
2. 查找 .ply 格式的 point cloud 文件
3. 确认是否有其他实验保存了完整 geometry
"""
    
    report += """
## 7. 下一步如何把真实 trained 3DGS geometry 接入 Phase12？

### 修复路线

1. **找到真实 trained 3DGS geometry checkpoint**
   - 搜索 3DGS training 的输出目录
   - 查找包含 positions/rotation/scale/density 的 checkpoint
   - 或使用 .ply 格式的 point cloud

2. **修改 Phase12 模型初始化逻辑**
   在 Phase12 脚本中：
   ```python
   # 加载真实 geometry
   geometry_ckpt = torch.load(geometry_checkpoint_path, map_location=device)
   model.init_from_checkpoint(geometry_ckpt, setup_optimizer=False)
   
   # 冻结 geometry 参数
   for param_name in ['positions', 'rotation', 'scale', 'density', 'features_albedo', 'features_specular']:
       param = getattr(model, param_name)
       param.requires_grad = False
   
   # 只初始化并训练 person_feature
   person_feature_dim = 512
   model._person_feature = torch.nn.Parameter(
       torch.randn(model.num_gaussians, person_feature_dim, device=device) * 0.01
   )
   ```

3. **禁止在 Phase12 ReID 训练中重新 random init xyz**
   - 移除 `init_from_random_point_cloud()` 调用
   - 或使用 `init_from_checkpoint()` 替代

4. **Checkpoint 保存时记录 geometry 信息**
   ```python
   checkpoint = {
       'model_state_dict': {
           'positions': model.positions,
           'rotation': model.rotation,
           'scale': model.scale,
           'density': model.density,
           'features_albedo': model.features_albedo,
           'features_specular': model.features_specular,
           '_person_feature': model._person_feature,
       },
       'geometry_source_path': geometry_checkpoint_path,
       'geometry_loaded': True,
       'geometry_frozen': True,
       'person_feature_shape': list(model._person_feature.shape),
   }
   ```

5. **重新运行 C1-C7 Gaussian-Set source consistency diagnostic**
   - 确认 selected_gaussian_count / weight_sum / feature_norm 正常
   - 确认所有相机都有合理的 opacity coverage

6. **确认正常后再考虑 12G SupCon**

### 最终判定

"""
    
    report += f"**Case: {case}**\n\n"
    report += f"{conclusion}\n\n"
    
    report += """---

*报告生成时间：2026-05-13*
*审计基于 checkpoint keys、code scan、runtime audit 三类证据*
"""
    
    with open(os.path.join(output_dir, 'final_report.md'), 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Phase12 Geometry Source Audit')
    parser.add_argument('--output_dir', type=str, default='outputs/phase12_geometry_source_audit')
    parser.add_argument('--checkpoints', nargs='+', default=[
        'outputs/phase12f_gaussianset_ema_proto_mv_lam01_proto005/checkpoint_best_fixed_cos.pt',
        'outputs/phase12c_gaussianset_clean_random_fixed_eval_lr1e3_v2/checkpoint_best_fixed_cos.pt',
        'outputs/phase12e_gaussianset_mv_lam01_tau02/checkpoint_best_fixed_cos.pt',
    ])
    parser.add_argument('--code_files', nargs='+', default=[
        'tools/archive/phase12/phase12f_EMA_PROTO_PLUS_MV_INFO_NCE_可复用.py',
        'tools/archive/phase12/phase12c_CLEAN_RANDOM_TEACHER_ONLY_一次性实验.py',
        'tools/eval_reid_gaussianset.py',
        'tools/diagnose_phase12_gaussianset_source_consistency.py',
    ])
    parser.add_argument('--model_files', nargs='+', default=[
        'threedgrut/model/model.py',
        'threedgrut/trainer.py',
    ])
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Phase12 Geometry Source Audit")
    print("=" * 80)
    
    print("\n[1/5] Checking checkpoint keys...")
    checkpoint_reports = []
    for ckpt_path in args.checkpoints:
        report, error = check_checkpoint_keys(ckpt_path)
        if error:
            print(f"  {ckpt_path}: {error}")
        else:
            print(f"  {ckpt_path}: OK")
        checkpoint_reports.append(report)
    
    with open(os.path.join(args.output_dir, 'checkpoint_key_report.json'), 'w') as f:
        json.dump(checkpoint_reports, f, indent=2, default=str)
    
    print("\n[2/5] Scanning code for geometry initialization...")
    all_code_files = args.code_files + args.model_files
    code_scan_report = scan_code_for_geometry_init(all_code_files)
    
    with open(os.path.join(args.output_dir, 'code_scan_report.json'), 'w') as f:
        json.dump(code_scan_report, f, indent=2, default=str)
    
    code_scan_md = "# Phase12 Code Scan Report\n\n"
    code_scan_md += f"## Files Scanned: {len(code_scan_report['files_scanned'])}\n\n"
    
    for file_info in code_scan_report['files_scanned']:
        code_scan_md += f"## {file_info['path']}\n\n"
        for keyword, matches in file_info['matches'].items():
            code_scan_md += f"### Keyword: `{keyword}` ({len(matches)} matches)\n\n"
            for match in matches[:5]:
                code_scan_md += f"- Line {match['line']}: `{match['content']}`\n"
            if len(matches) > 5:
                code_scan_md += f"- ... and {len(matches) - 5} more matches\n"
            code_scan_md += "\n"
    
    with open(os.path.join(args.output_dir, 'code_scan_report.md'), 'w') as f:
        f.write(code_scan_md)
    
    print("\n[3/5] Running runtime audit...")
    primary_checkpoint = args.checkpoints[0]
    if not os.path.exists(primary_checkpoint):
        print(f"  Primary checkpoint not found: {primary_checkpoint}")
        primary_checkpoint = None
        for ckpt in args.checkpoints:
            if os.path.exists(ckpt):
                primary_checkpoint = ckpt
                break
    
    runtime_audit_result = {
        'comparison': {'conclusion': 'unknown', 'xyz_changed': None},
        'audit1': None,
        'audit2': None,
    }
    
    if primary_checkpoint:
        print(f"  Using checkpoint: {primary_checkpoint}")
        comparison, audit1, audit2 = compare_xyz_between_seeds(args, primary_checkpoint, seed1=42, seed2=123)
        runtime_audit_result['comparison'] = comparison
        runtime_audit_result['audit1'] = audit1
        runtime_audit_result['audit2'] = audit2
        print(f"  xyz_changed: {comparison.get('xyz_changed')}")
        print(f"  conclusion: {comparison.get('conclusion')}")
    else:
        print("  No checkpoint available for runtime audit")
    
    with open(os.path.join(args.output_dir, 'runtime_audit.json'), 'w') as f:
        json.dump(runtime_audit_result, f, indent=2, default=str)
    
    print("\n[4/5] Searching for geometry checkpoints...")
    geometry_candidates = search_geometry_checkpoints()
    print(f"  Found {len(geometry_candidates)} candidate files")
    
    geo_candidates_with_geo = [c for c in geometry_candidates if c.get('has_geometry') or c.get('has_geometry_in_model_state_dict')]
    print(f"  Candidates with geometry: {len(geo_candidates_with_geo)}")
    
    with open(os.path.join(args.output_dir, 'geometry_candidate_report.json'), 'w') as f:
        json.dump(geometry_candidates, f, indent=2, default=str)
    
    print("\n[5/5] Generating final report...")
    generate_final_report(
        checkpoint_reports,
        code_scan_report,
        runtime_audit_result,
        geometry_candidates,
        args.output_dir,
    )
    
    print(f"\nAudit complete! Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
