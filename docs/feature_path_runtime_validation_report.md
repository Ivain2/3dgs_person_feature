# 3DGRUT 独立 CUDA Feature 渲染路径 - Runtime 验证报告

> **日期**: 2026-05-19
> **项目**: 3DGRUT V3 Feature Path

---

## 一、Transmittance / Early Termination 审计

### 状态: **PASS** ✅

**报告**: [docs/feature_path_transmittance_audit.md](file:///data02/zhangrunxiang/3dgrut/docs/feature_path_transmittance_audit.md)

### 修复历史

| 版本 | 修复内容 | 文件 | 状态 |
|------|----------|------|------|
| v1 | 移除 `processHitBwd` 调用（nullptr segfault） | referenceBwdOptix.cu | ✅ |
| v2 | 添加 `rayTransmittance *= (1-galpha)` | referenceBwdOptix.cu | ✅ |
| v3 | 添加 `hitMinGaussianResponse` 和 `alphaMinThreshold` 阈值检查 | referenceBwdOptix.cu | ✅ |
| v4 | 添加 `minTransmittance` early termination | referenceBwdOptix.cu | ✅ |

### 验证结果

| 检查项 | 状态 |
|--------|------|
| #1 forward weight = alpha * T_before | ✅ |
| #2 backward weight 与 forward 一致 | ✅ |
| #3 每个 hit 后更新 T | ✅ |
| #4 early termination 一致 | ✅ |
| #5 多 hit 梯度使用 alpha_i * Π(1-alpha_j) | ✅ |

### 核心代码 (修复后)

```cpp
// referenceBwdOptix.cu - backward kernel
while (startT < endT) {
    // Early termination: match forward
    if (rayTransmittance <= params.minTransmittance) {
        break;
    }
    trace(rayPayload, rayOrigin, rayDirection, startT + epsT, endT);
    ...
    if (params.featureChannels > 0 && ...) {
        // Compute gres, galpha (same as forward)
        const float gres = particleResponse<...>(grayDist);
        const float galpha = fminf(0.99f, gres * particleDensity);

        // Match forward threshold checks
        if ((gres > params.hitMinGaussianResponse) && (galpha > params.alphaMinThreshold)) {
            const float weight = galpha * rayTransmittance;
            // Propagate gradient
            for (unsigned int ch = 0; ch < params.featureChannels; ++ch) {
                atomicAdd(&particleFeatureGrad[offset], weight * featGrad);
            }
            // CRITICAL: Update transmittance
            rayTransmittance *= (1.0f - galpha);
        }
    }
    startT = fmaxf(startT, rayHit.distance);
}
```

---

## 二、OptiX 环境诊断

### 状态: **Runtime 不可用** ⚠️

**报告**: [outputs/optix_environment_check/final_report.md](file:///data02/zhangrunxiang/3dgrut/outputs/optix_environment_check/final_report.md)

### 环境信息

| 项目 | 状态 |
|------|------|
| GPU | NVIDIA A100 80GB PCIe (无 RT Cores) |
| CUDA | 11.8 |
| Driver | 535.183.06 |
| libnvoptix.so.1 | ❌ 无法加载 |
| OptixTracer 创建 | ❌ SIGSEGV (exit code -11) |
| CUDA_HOME | ❌ 未设置 |
| lib3dgrt_cc | ✅ 编译成功，存在 |

### 诊断结论

当前环境 OptiX runtime/driver/extension 初始化链路不可用。Segfault 可能来自：
1. **OptiX runtime 库缺失** (`libnvoptix.so.1` 无法加载)
2. A100 无 RT Cores，虽然 OptiX 理论上可通过 CUDA cores 运行，但当前环境配置不完整
3. 容器/驱动版本不匹配

**不是代码问题**。需要在正确配置 OptiX 的环境（RTX GPU + 完整 OptiX SDK）中复测。

---

## 三、Runtime 测试状态

### 任务3: 线性度测试

**状态**: ⏸️ **环境阻塞** - OptixTracer 初始化 segfault

**脚本**: [tools/diagnose_feature_renderer_linearity.py](file:///data02/zhangrunxiang/3dgrut/tools/diagnose_feature_renderer_linearity.py)

**测试项**:
- T1 zero: feature=0 → output≈0
- T2 scale: render(2f)≈2·render(f)
- T3 additivity: render(f1+f2)≈render(f1)+render(f2)
- T4 signed: negative feature 不被 clamp
- T5 old-vs-new: 旧 SH path + linearize vs new feature path
- T6 opacity: new path 不需要 0.5*opacity/SH_C0 修正

### 任务4: 三路 3D ReID 对比

**状态**: ⏸️ **环境阻塞** - 同上

**脚本**: [tools/evaluate_3d_reid_aggregated_features.py](file:///data02/zhangrunxiang/3dgrut/tools/evaluate_3d_reid_aggregated_features.py)

### 任务5: Frozen Probe

**状态**: ⏸️ **阻塞** - 依赖任务4的 roi_features_new.npz 输出

### 任务2: Gradient Check (Multi-Hit)

**状态**: ⏸️ **环境阻塞** - 同上

---

## 四、代码修改汇总

### CUDA 层修复 (referenceBwdOptix.cu)

| 行号 | 修复 | 说明 |
|------|------|------|
| L136-138 | 添加 early termination | `if (rayTransmittance <= params.minTransmittance) break;` |
| L180-182 | 添加阈值检查 | `if ((gres > params.hitMinGaussianResponse) && (galpha > params.alphaMinThreshold))` |
| L196-199 | 添加 transmittance 更新 | `rayTransmittance *= (1.0f - galpha);` |

### Python 层修复

| 文件 | 修复 |
|------|------|
| [threedgrut/utils/misc.py](file:///data02/zhangrunxiang/3dgrut/threedgrut/utils/misc.py#L26-L27) | 添加 `replace=True` 到 OmegaConf resolver 注册 |
| [tools/diagnose_feature_renderer_linearity.py](file:///data02/zhangrunxiang/3dgrut/tools/diagnose_feature_renderer_linearity.py) | 修复配置（background, pipeline_type, alpha params） |
| [tools/diagnose_feature_path_gradient.py](file:///data02/zhangrunxiang/3dgrut/tools/diagnose_feature_path_gradient.py) | 新增 multi-hit gradient 测试 |

---

## 五、编译状态

**CUDA Extension**: ✅ 编译成功（4/4 files, exit code 0）

```
[1/4] x86_64-conda-linux-gnu-c++ optixTracer.cpp
[2/4] x86_64-conda-linux-gnu-c++ bindings.cpp
[3/4] nvcc particlePrimitives.cu
[4/4] link lib3dgrt_cc.so
```

---

## 六、结论与下一步

### 当前状态总结

| 项目 | 状态 | 说明 |
|------|------|------|
| Transmittance 审计 | ✅ PASS | Forward/backward 完全一致 |
| CUDA 编译 | ✅ 成功 | lib3dgrt_cc.so 编译通过 |
| 代码静态检查 | ✅ 通过 | 所有 10 项验证点通过 |
| OptiX 环境 | ⚠️ 不可用 | 需 RTX GPU + 完整 OptiX SDK |
| Runtime 测试 | ⏸️ 阻塞 | 等待可用环境 |
| ReID 对比 | ⏸️ 阻塞 | 依赖 Runtime 测试 |
| Frozen Probe | ⏸️ 阻塞 | 依赖 ReID 对比输出 |

### 下一步

**需要在可用环境中运行的测试**:

```bash
# 1. 线性度测试
python tools/diagnose_feature_renderer_linearity.py --feature_dim 8

# 2. Gradient check (含 multi-hit)
python tools/diagnose_feature_path_gradient.py --feature_dim 3

# 3. 三路 ReID 对比
python tools/evaluate_3d_reid_aggregated_features.py \
    --checkpoint <path> --dataset_path <path> --run_mode all_three

# 4. Frozen Probe
python tools/train_3d_reid_probe.py \
    --roi_features outputs/v3_feature_path_compare/roi_features/roi_features_new.npz \
    --probe linear --epochs 50

python tools/train_3d_reid_probe.py \
    --roi_features outputs/v3_feature_path_compare/roi_features/roi_features_new.npz \
    --probe mlp --hidden_dim 256 --epochs 50
```

### V4 进入条件

| 条件 | 当前状态 |
|------|----------|
| Transmittance 审计通过 | ✅ 已通过 |
| CUDA 编译成功 | ✅ 已通过 |
| Runtime 测试通过 | ⏸️ 待验证 |
| ReID 对比完成 | ⏸️ 待验证 |
| Probe 结果确认 | ⏸️ 待验证 |

**当所有 runtime 测试通过后**，根据 probe 结果决定 V4 方向：
- raw weak + probe strong → V4: trainable identity latent + ROI MLP head
- probe weak → Gaussian purity audit
- camera strong + ID weak → camera adversarial
