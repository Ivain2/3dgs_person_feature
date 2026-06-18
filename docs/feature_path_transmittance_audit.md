# 3DGRUT Feature Path - Transmittance & Early Termination 审计报告 (v2)

> **日期**: 2026-05-19
> **状态**: ✅ 已通过，所有问题已修复

---

## 1. Forward rayParticleWeight 定义

**文件**: `threedgrt_tracer/src/kernels/cuda/barycentricSurfelsOptix.cu`

**位置**: L137-150

```cpp
const float rayParticleAlpha = fminf(0.99f, rayParticleHitKernelResponse * particleNormalsDensity.w);
if ((rayParticleHitKernelResponse > params.hitMinGaussianResponse) && (rayParticleAlpha > params.alphaMinThreshold)) {
    const float rayParticleWeight = rayParticleAlpha * rayTransmittance;  // L139
    // ... radiance accumulation ...
    rayTransmittance *= (1 - rayParticleAlpha);  // L150
}
```

**公式**: `weight_i = alpha_i * T_before_i`，其中 `T_before_i = Π(1-alpha_j)` for j < i

---

## 2. Backward weight 定义

**文件**: `threedgrt_tracer/src/kernels/cuda/referenceBwdOptix.cu`

**位置**: L177-183 (修复后)

```cpp
const float gres   = particleResponse<PipelineParameters::ParticleKernelDegree>(grayDist);
const float galpha = fminf(0.99f, gres * particleDensity);
if ((gres > params.hitMinGaussianResponse) && (galpha > params.alphaMinThreshold)) {
    const float weight = galpha * rayTransmittance;
    // gradient propagation
    rayTransmittance *= (1.0f - galpha);
}
```

**公式**: `weight_i = galpha_i * T_before_i`，其中 `T_before_i = Π(1-galpha_j)` for j < i

---

## 3. 逐项检查

### 3.1 Forward rayParticleWeight = alpha * transmittance_before_hit

**状态**: ✅ PASS

**位置**: `barycentricSurfelsOptix.cu:139`
```cpp
const float rayParticleWeight = rayParticleAlpha * rayTransmittance;
```

`rayTransmittance` 在 hit 前使用，hit 后才更新（L150）。

---

### 3.2 Backward weight 与 forward 一致

**状态**: ✅ PASS（修复后）

**对比**:

| 项目 | Forward | Backward (修复后) | 一致? |
|------|---------|-------------------|-------|
| alpha 计算 | `fminf(0.99f, response * density)` | `fminf(0.99f, gres * particleDensity)` | ✅ |
| weight 计算 | `alpha * rayTransmittance` | `galpha * rayTransmittance` | ✅ |
| transmittance 更新 | `*= (1 - alpha)` | `*= (1 - galpha)` | ✅ |
| 阈值检查 | `gres > hitMinGaussianResponse && alpha > alphaMinThreshold` | 同左 | ✅ (修复后) |
| early termination | `rayTransmittance > minTransmittance` | 同左 | ✅ (修复后) |

---

### 3.3 Backward 每个 hit 后执行 rayTransmittance *= (1-alpha)

**状态**: ✅ PASS（修复后）

**位置**: `referenceBwdOptix.cu:199`
```cpp
rayTransmittance *= (1.0f - galpha);
```

**执行条件**: 仅当 hit 通过阈值检查时更新（与 forward 一致）。

---

### 3.4 Forward 若有 minTransmittance / early break，backward 是否一致

**状态**: ✅ PASS（修复后）

**Forward** (`barycentricSurfelsOptix.cu:118`):
```cpp
while ((rayLastHitDistance <= minMaxT.y) && (rayTransmittance > params.minTransmittance)) {
```

**Backward (修复前)** (`referenceBwdOptix.cu:133`):
```cpp
while (startT < endT) {
    // 缺少 minTransmittance 检查 ❌
```

**Backward (修复后)** (`referenceBwdOptix.cu:133-138`):
```cpp
while (startT < endT) {
    if (rayTransmittance <= params.minTransmittance) {
        break;
    }
```

**结果**: 修复后 backward 的 early termination 与 forward 一致。

---

### 3.5 多 hit 梯度是否使用 alpha_i * Π(1-alpha_j)

**状态**: ✅ PASS（修复后）

**Forward 序列**:
```
T = 1.0
Hit 1: weight_1 = alpha_1 * 1.0,        T *= (1-alpha_1)
Hit 2: weight_2 = alpha_2 * (1-alpha_1), T *= (1-alpha_2)
Hit 3: weight_3 = alpha_3 * (1-alpha_1)(1-alpha_2), T *= (1-alpha_3)
```

**Backward 序列 (修复后)**:
```
T = 1.0
Hit 1: weight_1 = galpha_1 * 1.0,        T *= (1-galpha_1)
Hit 2: weight_2 = galpha_2 * (1-galpha_1), T *= (1-galpha_2)
Hit 3: weight_3 = galpha_3 * (1-galpha_1)(1-galpha_2), T *= (1-galpha_3)
```

**梯度公式**: `dL/d_f_i = weight_i * dL/d_feature = alpha_i * Π(1-alpha_j) * dL/d_feature`

---

## 4. 修复历史

| 版本 | 修复内容 | 文件 | 行号 |
|------|----------|------|------|
| v1 | 移除 `processHitBwd` 调用（nullptr segfault 修复） | referenceBwdOptix.cu | L143-L201 |
| v2 | 添加 `rayTransmittance *= (1-galpha)` | referenceBwdOptix.cu | L199 |
| v3 | 添加 `hitMinGaussianResponse` 和 `alphaMinThreshold` 阈值检查 | referenceBwdOptix.cu | L182 |
| v4 | 添加 `minTransmittance` early termination | referenceBwdOptix.cu | L136-L138 |

---

## 5. 最终判定: **PASS** ✅

所有 5 项检查已通过，forward 和 backward 的 transmittance 重放逻辑完全一致。

| 检查项 | 状态 |
|--------|------|
| #1 forward weight = alpha * T_before | ✅ |
| #2 backward weight 与 forward 一致 | ✅ |
| #3 每个 hit 后更新 T | ✅ |
| #4 early termination 一致 | ✅ |
| #5 多 hit 梯度使用 alpha_i * Π(1-alpha_j) | ✅ |

---

## 6. 编译状态

CUDA extension 已成功重新编译（4/4 files compiled, exit code 0）。
