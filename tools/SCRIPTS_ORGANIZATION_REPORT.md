# Tools 目录脚本整理报告

## 整理日期: 2026-05-19

## 概述

对 `/data02/zhangrunxiang/3dgrut/tools/` 下的 **63 个 Python 脚本** 进行了全面整理，按用途分为两类：
- **可复用脚本**（20 个）：重命名为更精确的名称，保留在 `tools/` 根目录
- **一次性诊断/实验脚本**（43 个）：重命名后移至 `tools/backups/` 目录备份

整理后 `tools/` 根目录从 63 个脚本减少到 **20 个**，大幅提升了可读性和可维护性。

---

## 目录结构

```
tools/
├── [可复用脚本 - 20 个]
└── backups/
    ├── debug_and_test/              (9 个) 调试和测试脚本
    ├── phase12_diagnostics/         (7 个) Phase12 几何诊断脚本
    ├── phase13_diagnostics/         (8 个) Phase13 身份/几何诊断脚本
    ├── phase14_smoke_tests/         (7 个) Phase14 冒烟测试脚本
    ├── phase15_experiments/         (3 个) Phase15 ReID 训练实验
    ├── phase16_diagnostics/         (2 个) Phase16 CE 诊断脚本
    └── train_experiments/          (9 个) 训练变体实验脚本
```

---

## 一、可复用脚本（保留在 tools/ 根目录）

### 1.1 评估脚本 (6 个)

| 原文件名 | 新文件名 | 用途说明 |
|---------|---------|---------|
| `evaluate_3d_aggregated_reid.py` | `evaluate_3d_reid_aggregated_features.py` | 评估 3D 聚合后的 ReID 特征（加载 aggregated_features.pt，render + ROI pooling + V2.1 评估协议） |
| `evaluate_pure_registration_render2d.py` | `evaluate_2d_rendered_registration_features.py` | 评估纯注册特征的 2D 渲染表现（经过 renderer + ROI pooling，测试渲染器是否破坏判别能力） |
| `evaluate_pure_registration_direct3d.py` | `evaluate_3d_direct_registered_features.py` | 评估纯注册特征的 3D 直接计算（绕过 renderer，直接投影+池化，测试特征本身是否有判别力） |
| `evaluate_2d_reid_baseline.py` | `evaluate_2d_reid_baseline.py` | 2D ReID 基线评估（不经过 3D，纯 2D 特征的 mAP/R1/ROC-AUC 等指标） |
| `evaluate_identity_features.py` | `evaluate_identity_features.py` | 评估 identity 特征的表现（原型匹配、身份分离度等） |
| `evaluate_teacher_student_models.py` | `evaluate_teacher_student_models.py` | 评估 teacher-student 模型性能（对比 teacher embedding 和 student 预测） |

### 1.2 诊断脚本 (3 个)

| 原文件名 | 新文件名 | 用途说明 |
|---------|---------|---------|
| `diagnose_reid_feature_clamp_range.py` | `diagnose_reid_feature_clamp_audit.py` | 审计 ReID 特征是否会触发 CUDA clamp（x = 0.5 + SH_C0 * f，统计 clamp ratio） |
| `diagnose_feature_renderer_linearity.py` | `diagnose_feature_renderer_linearity.py` | 诊断特征渲染器线性度（T1 zero, T2 scale, T3 additivity, T4 sign, T5 background 五个测试） |
| `diagnose_phase11B_camera_training.py` | `diagnose_phase11B_camera_training.py` | 诊断 Phase11B camera-specific 训练效果（各相机视角的训练/验证指标） |

### 1.3 特征构建/聚合脚本 (2 个)

| 原文件名 | 新文件名 | 用途说明 |
|---------|---------|---------|
| `build_pure_registered_gaussian_features.py` | `build_registered_gaussian_features.py` | 构建纯注册的 Gaussian 特征（将 V2 2D ReID 特征注册到 3D Gaussian，输出 registered_features.pt） |
| `aggregate_2d_reid_to_3d.py` | `aggregate_2d_features_to_3d_gaussians.py` | 将 2D ReID 特征聚合到 3D Gaussian（多视角聚合，输出 aggregated_features.pt） |

### 1.4 原型构建脚本 (1 个)

| 原文件名 | 新文件名 | 用途说明 |
|---------|---------|---------|
| `build_teacher_prototypes.py` | `build_reid_teacher_prototypes.py` | 构建 teacher ReID 原型（从 2D teacher 特征计算每个 identity 的原型向量） |

### 1.5 训练脚本 (5 个)

| 原文件名 | 新文件名 | 用途说明 |
|---------|---------|---------|
| `train_reid_feature_distillation.py` | `train_reid_distillation.py` | ReID 特征蒸馏训练（使用 teacher embedding 监督 _person_feature，支持 foreground-aware pooling） |
| `train_reid_baseline.py` | `train_reid_baseline_model.py` | ReID 基线模型训练（基础的 ReID 训练，用于对比） |
| `train_contrastive_reid.py` | `train_contrastive_reid_model.py` | 对比学习 ReID 训练（使用 contrastive loss 优化特征） |
| `train_prototype_reid.py` | `train_prototype_reid_model.py` | 基于原型的 ReID 训练（使用 prototype-level 损失） |
| `train_phase11C_mv_ema.py` | `train_reid_phase11C_mv_ema.py` | Phase11C 多视角 EMA 训练（multi-view 指数移动平均） |

### 1.6 Phase 相关脚本 (3 个)

| 原文件名 | 新文件名 | 用途说明 |
|---------|---------|---------|
| `phase5_short_train_reid.py` | `train_reid_phase5_short.py` | Phase5 短期 ReID 训练（快速验证训练流程） |
| `phase5_overfit_reid.py` | `overfit_reid_phase5.py` | Phase5 过拟合 ReID 测试（验证模型是否有足够容量拟合数据） |
| `phase4_reid_training_verify.py` | `verify_reid_training_phase4.py` | Phase4 ReID 训练验证（验证训练数据加载和损失计算正确性） |

---

## 二、一次性诊断/实验脚本（移至 backups/）

### 2.1 debug_and_test/ (9 个)

| 原文件名 | 备份文件名 | 用途说明 |
|---------|-----------|---------|
| `test_v4_multi_render.py` | `test_multi_render.py` | V4 多渲染测试（测试多次渲染的一致性） |
| `test_v4_grad_chain2.py` | `test_grad_chain_v2.py` | V4 梯度链测试 v2（验证梯度反向传播） |
| `test_v4_grad_chain.py` | `test_grad_chain.py` | V4 梯度链测试 v1（验证梯度反向传播） |
| `diagnose_person_gaussian_support.py` | `diagnose_person_gaussian_support.py` | 诊断行人 Gaussian 支撑情况（多少 Gaussian 对应行人） |
| `diagnose_c1c7_camera_quality.py` | `diagnose_camera_quality_c1c7.py` | 诊断 C1-C7 相机图像质量 |
| `diagnose_bbox_roi_consistency.py` | `diagnose_bbox_roi_consistency.py` | 诊断 bbox 与 ROI 一致性 |
| `diagnose_phase12_gaussianset_source_consistency.py` | `diagnose_gaussian_set_source_consistency.py` | 诊断 Gaussian set 来源一致性 |
| `audit_phase12_geometry_source.py` | `audit_geometry_source_phase12.py` | 审计 Phase12 几何来源 |
| `eval_reid_gaussianset.py` | `eval_reid_gaussian_set.py` | 评估 ReID Gaussian set 性能 |

### 2.2 phase12_diagnostics/ (7 个)

| 原文件名 | 备份文件名 | 用途说明 |
|---------|-----------|---------|
| `phase12_stage_b_spatial_distribution.py` | `diagnose_gaussian_spatial_distribution.py` | 诊断 Gaussian 空间分布 |
| `phase12_stage2_geometry_validation.py` | `validate_stage2_geometry.py` | 验证 Stage2 几何参数 |
| `phase12_stage1_geometry_sanity.py` | `sanity_check_stage1_geometry.py` |  Sanity check Stage1 几何参数 |
| `phase12_projection_consistency_check.py` | `check_projection_consistency.py` | 检查投影一致性 |
| `phase12_geometry_pooling_watershed_diagnostic.py` | `diagnose_geometry_pooling_watershed.py` | 诊断几何池化 watershed 问题 |
| `phase12_parallel_validation.py` | `parallel_geometry_validation.py` | 并行几何验证 |
| `phase12_geometry_fixed_teacher_warmup.py` | `geometry_fixed_teacher_warmup.py` | 固定几何的 teacher warmup |

### 2.3 phase13_diagnostics/ (8 个)

| 原文件名 | 备份文件名 | 用途说明 |
|---------|-----------|---------|
| `phase13_verify_bbox_scale.py` | `verify_bbox_scale.py` | 验证 bbox 缩放参数 |
| `phase13_teacher_only_warmup.py` | `teacher_only_warmup.py` | Teacher-only warmup 测试 |
| `phase13_supervised_identity.py` | `supervised_identity_test.py` | 有监督 identity 测试 |
| `phase13_pf_real_readout_compare.py` | `compare_pf_real_readout.py` | 比较 PF real readout |
| `phase13_pf_alignment_compare.py` | `compare_pf_alignment.py` | 比较 PF alignment |
| `phase13_layer0b_geometry_support_verify.py` | `verify_layer0b_geometry_support.py` | 验证 Layer0b 几何支撑 |
| `phase13_gaussian_count_audit.py` | `audit_gaussian_count.py` | 审计 Gaussian 数量 |
| `phase13_bbox_scale_diagnostic.py` | `diagnose_bbox_scale.py` | 诊断 bbox 缩放问题 |

### 2.4 phase14_smoke_tests/ (7 个)

| 原文件名 | 备份文件名 | 用途说明 |
|---------|-----------|---------|
| `phase14_smoke_run.py` | `smoke_test_phase14.py` | Phase14 冒烟测试 |
| `phase14_preflight_check.py` | `preflight_check_phase14.py` | Phase14 飞行前检查 |
| `phase14_geometry_smoke.py` | `smoke_test_geometry.py` | 几何冒烟测试 |
| `phase14_final_validation.py` | `final_validation_phase14.py` | Phase14 最终验证 |
| `phase14_debug_checkpoint.py` | `debug_checkpoint_phase14.py` | Phase14 调试 checkpoint |
| `phase14_collect_smoke_metrics.py` | `collect_smoke_metrics_phase14.py` | 收集 Phase14 冒烟指标 |
| `phase14_checkpoint_cleanup.py` | `checkpoint_cleanup_phase14.py` | Phase14 checkpoint 清理 |

### 2.5 phase15_experiments/ (3 个)

| 原文件名 | 备份文件名 | 用途说明 |
|---------|-----------|---------|
| `phase15_teacher_only_smoke.py` | `smoke_train_teacher_only_reid.py` | Teacher-only ReID 冒烟训练 |
| `phase15_medium_1000.py` | `train_teacher_only_reid_1000iters.py` | Teacher-only ReID 1000 步训练 |
| `phase15_identity_diagnostic.py` | `diagnose_identity_feature_collapse.py` | 诊断 identity 特征 collapse |

### 2.6 phase16_diagnostics/ (2 个)

| 原文件名 | 备份文件名 | 用途说明 |
|---------|-----------|---------|
| `phase16_ce_diagnostics.py` | `diagnose_ce_training_failure.py` | 诊断 CE 训练失败根因 |
| `phase16_ce_small_overfit.py` | `ce_small_overfit_test.py` | CE 小规模过拟合测试 |

### 2.7 train_experiments/ (9 个)

| 原文件名 | 备份文件名 | 用途说明 |
|---------|-----------|---------|
| `train_phase11B_v4_gradient_stabilized_frozen_mv_identity.py` | `train_gradient_stabilized_frozen_mv_identity.py` | 梯度稳定的冻结多视角身份训练 |
| `train_phase11B_v3_frozen_mv_identity.py` | `train_frozen_mv_identity_v3.py` | 冻结多视角身份训练 v3 |
| `train_phase11B_mv_supcon.py` | `train_mv_supcon_reid.py` | 多视角 SupCon ReID 训练 |
| `train_phase11B_mv_supcon.py.bak_20260506_212828` | `train_mv_supcon_reid_backup.py` | 多视角 SupCon ReID 训练备份 |
| `train_phase11A_proto_infonce.py` | `train_proto_infonce.py` | 原型 InfoNCE 训练 |
| `train_reid_feature_distillation.phase11B_partial_backup.py` | `train_reid_distillation_partial_backup.py` | ReID 蒸馏训练部分备份 |
| `debug_phase11B_sampler_supcon.py` | `debug_sampler_supcon.py` | 调试 Phase11B 采样器 SupCon |
| `ablate_lambda_reid.py` | `ablate_lambda_reid_weights.py` | 消融 lambda_reid 权重 |
| `phase17_teacher_ce_train.py` | `train_teacher_ce_reid.py` | Teacher + CE ReID 训练 |

---

## 三、整理原则

### 保留在 tools/ 根目录的条件
- 会被 **多次运行**（训练、评估、诊断）
- 是 **标准 workflow** 的一部分
- 名称能 **准确描述功能**

### 移至 backups/ 的条件
- 只用于 **一次性调试/验证/冒烟测试**
- 是某个 phase 的 **临时实验**
- 已有 **更新版本替代**（如 v3/v4 替代 v1/v2）
- 是 **备份文件**（.bak, .partial_backup）

---

## 四、统计

| 类别 | 数量 |
|------|------|
| tools/ 根目录（可复用） | **20** |
| backups/debug_and_test/ | 9 |
| backups/phase12_diagnostics/ | 7 |
| backups/phase13_diagnostics/ | 8 |
| backups/phase14_smoke_tests/ | 7 |
| backups/phase15_experiments/ | 3 |
| backups/phase16_diagnostics/ | 2 |
| backups/train_experiments/ | 9 |
| **总计备份** | **45** |
| **总计** | **65** |

---

## 五、可复用脚本使用指南

### 评估流程

```bash
# 1. 评估 2D ReID 基线
python tools/evaluate_2d_reid_baseline.py

# 2. 将 2D 特征聚合到 3D
python tools/aggregate_2d_features_to_3d_gaussians.py

# 3. 构建注册的 Gaussian 特征
python tools/build_registered_gaussian_features.py

# 4. 评估 3D 聚合特征
python tools/evaluate_3d_reid_aggregated_features.py

# 5. 评估 2D 渲染注册特征
python tools/evaluate_2d_rendered_registration_features.py

# 6. 评估 3D 直接注册特征
python tools/evaluate_3d_direct_registered_features.py
```

### 诊断流程

```bash
# 1. 诊断渲染器线性度
python tools/diagnose_feature_renderer_linearity.py

# 2. 审计 ReID 特征 clamp 范围
python tools/diagnose_reid_feature_clamp_audit.py
```

### 训练流程

```bash
# 1. 构建 teacher prototypes
python tools/build_reid_teacher_prototypes.py

# 2. ReID 特征蒸馏训练
python tools/train_reid_distillation.py
```
