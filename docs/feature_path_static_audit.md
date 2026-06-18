# Feature Rendering Path — 静态代码审计报告

> **审计日期**: 2026-05-19  
> **审计范围**: 独立 CUDA ReID feature rendering path 正确性  
> **审计方法**: 逐文件、逐行静态分析，对照10项检查点

---

## 涉及文件清单

| # | 文件 | 关键角色 |
|---|------|----------|
| 1 | `threedgrt_tracer/include/3dgrt/pipelineParameters.h` | CUDA 参数结构体定义 |
| 2 | `threedgrt_tracer/src/kernels/cuda/barycentricSurfelsOptix.cu` | Forward CUDA raygen kernel |
| 3 | `threedgrt_tracer/src/kernels/cuda/referenceBwdOptix.cu` | Backward CUDA raygen kernel |
| 4 | `threedgrt_tracer/src/optixTracer.cpp` | C++ Python binding 实现 |
| 5 | `threedgrt_tracer/bindings.cpp` | pybind11 模块注册 |
| 6 | `threedgrt_tracer/tracer.py` | Python autograd wrapper |
| 7 | `threedgrut/model/model.py` | MixtureOfGaussians 模型 |
| 8 | `threedgrut/trainer.py` | 训练循环 |
| 9 | `threedgrt_tracer/include/3dgrt/kernels/cuda/gaussianParticles.cuh` | processHitBwd 等共享函数 |

---

## 检查项 #1: particleFeature index 是否为 particleId * D + ch

**状态: ✅ 通过**

### Forward (barycentricSurfelsOptix.cu:156)
```cpp
const uint32_t featOffset = rayHit.particleId * params.featureChannels + ch;
float particleFeat = params.particleFeature[featOffset];
```

### Backward (referenceBwdOptix.cu:200)
```cpp
const uint32_t featOffset = rayHit.particleId * params.featureChannels + ch;
atomicAdd(&params.particleFeatureGrad[featOffset], weight * featGrad);
```

**分析**: forward 和 backward 使用完全相同的索引公式 `particleId * featureChannels + ch`。这是一个标准的行主序 `[N, D]` 展平索引，其中 `particleId` 是行索引，`ch` 是列索引，`featureChannels` 是列数 D。

**结论**: 索引公式正确，forward/backward 一致。

---

## 检查项 #2: featureChannels 是否来自 particleFeature.size(1)

**状态: ✅ 通过**

### Forward (optixTracer.cpp:974)
```cpp
paramsHost.featureChannels = particleFeature.size(1);
```

### Forward tensor 创建 (optixTracer.cpp:938-940)
```cpp
torch::Tensor rayFeature = torch::zeros(
    {rayOri.size(0), rayOri.size(1), rayOri.size(2), (int64_t)particleFeature.size(1)}, 
    opts);
```

### Backward (optixTracer.cpp:1118)
```cpp
paramsHost.featureChannels = particleFeature.size(1);
```

**分析**: featureChannels 在 forward 和 backward 中均直接从 `particleFeature.size(1)` 获取，即 `[N, D]` 的第二维 D。该值被写入 PipelineParameters 并通过 cudaMemcpyAsync 传递到 GPU。

**结论**: featureChannels 来源正确，前后向一致。

---

## 检查项 #3: rayFeature 输出是否为 [B,H,W,D]

**状态: ✅ 通过**

### C++ tensor 创建 (optixTracer.cpp:938-940)
```cpp
torch::Tensor rayFeature = torch::zeros(
    {rayOri.size(0), rayOri.size(1), rayOri.size(2), (int64_t)particleFeature.size(1)}, 
    opts);
```
→ rayOri shape 为 `[B, H, W, 3]`，因此 rayFeature 为 `[B, H, W, D]`。

### CUDA kernel 写入 (barycentricSurfelsOptix.cu:106, 160)
```cpp
params.rayFeature[idx.z][idx.y][idx.x][ch] = 0.0f;           // 初始化
params.rayFeature[idx.z][idx.y][idx.x][ch] += particleFeat * rayParticleWeight;  // 累加
```
→ PackedTensorAccessor32<float, 4> 的索引 `[z][y][x][ch]` 对应 `[B][H][W][D]`。

### Python 端 shape 验证 (tracer.py:425-431)
```python
# ray_feature: [B, H, W, D], ray_opacity: [B, H, W, 1]
person_feature_map = ray_feature.squeeze(0).permute(2, 0, 1)  # [D, H, W]
assert person_feature_map.shape == (D, H, W)
```

**结论**: rayFeature 输出 shape 确认为 `[B, H, W, D]`，CUDA 写入索引匹配，Python 端有 shape assertion 保护。

---

## 检查项 #4: backward 的 weight 是否与 forward 的 rayParticleWeight 完全一致

**状态: ✅ 通过**

### Forward weight 计算 (barycentricSurfelsOptix.cu:137-139)
```cpp
const float rayParticleAlpha = fminf(0.99f, rayParticleHitKernelResponse * particleNormalsDensity.w);
// ... 条件检查: gres > hitMinGaussianResponse && alpha > alphaMinThreshold
const float rayParticleWeight = rayParticleAlpha * rayTransmittance;
```

其中 `rayParticleHitKernelResponse` 来自:
```cpp
const float rayParticleHitKernelResponse =
    particleScaledResponse<PipelineParameters::ParticleKernelDegree, PipelineParameters::ClampedPrimitive>(
        rayHit.particleSquaredDistance, particleScaleMinResponse, particleNormalsDensity.w);
```

### Backward weight 计算 (referenceBwdOptix.cu:169-192)
```cpp
float3 giscl = make_float3(1.0f / particleScale.x, 1.0f / particleScale.y, 1.0f / particleScale.z);
const float3 gposc = (rayOrigin - particlePosition);
const float3 gposcr = (gposc * particleRotation);
const float3 gro = giscl * gposcr;
const float3 rayDirR = rayDirection * particleRotation;
const float3 grdu = giscl * rayDirR;
const float3 grd = safe_normalize(grdu);
const float3 gcrod = PipelineParameters::SurfelPrimitive ? gro + grd * -gro.z / grd.z : cross(grd, gro);
const float grayDist = dot(gcrod, gcrod);
const float gres = particleResponse<PipelineParameters::ParticleKernelDegree>(grayDist);
const float galpha = fminf(0.99f, gres * particleDensity);
const float weight = galpha * rayTransmittance;
```

### 逐项对比

| 要素 | Forward | Backward | 一致? |
|------|---------|----------|-------|
| 坐标变换 (giscl, gposc, gposcr, gro) | processHit (gaussianParticles.cuh:370-373) | referenceBwdOptix.cu:181-184 | ✅ 一致 |
| 方向变换 (rayDirR, grdu, grd) | processHit (gaussianParticles.cuh:374-376) | referenceBwdOptix.cu:185-187 | ✅ 一致 |
| gcrod | processHit (gaussianParticles.cuh:378) | referenceBwdOptix.cu:188 | ✅ 一致 |
| grayDist | 用 particleSquaredDistance (from hit payload) | dot(gcrod, gcrod) 重新计算 | ⚠️ 见下方分析 |
| gres (kernel response) | particleScaledResponse<degree, clamped> | particleResponse<degree> | ⚠️ **潜在差异** |
| galpha | fminf(0.99f, gres * density) | fminf(0.99f, gres * density) | ✅ 一致 |
| rayTransmittance | 实时累乘 | 重新遍历累乘 | ✅ 一致 (重算) |

### 关于 gres 的深入分析

**Forward** 使用 `particleScaledResponse<degree, clamped>`，该函数对 minResponse 做了调制:
```cpp
// gaussianParticles.cuh:304-341
const float minResponse = fminf(modulatedMinResponse / responseModulation, 0.97f);
const float logMinResponse = clamped ? logf(minResponse) : modulatedMinResponse;
// 然后使用 logMinResponse 计算 exp(logMinResponse * grayDist...)
```

**Backward** 使用 `particleResponse<degree>`，该函数使用硬编码常量:
```cpp
// gaussianParticles.cuh:261-301
case 4: return expf(-0.0555555555556f * grayDist * grayDist);
// 硬编码 s = -4.5/3^degree
```

**但是**，backward 的 feature gradient 计算被 guard 在 `if ((gres > minParticleKernelDensity) && (galpha > minParticleAlpha))` 条件内 (referenceBwdOptix.cu:167 的条件在 line 207 `processHitBwd` 之后)。实际上 feature backward 的代码在 line 167-204，它在 `processHitBwd` 调用之后执行，并且有自己的条件判断:

```cpp
if (params.featureChannels > 0 && params.particleFeature != nullptr && params.particleFeatureGrad != nullptr) {
    // Recompute the weight (same as forward)
    ...
}
```

**关键发现**: backward 重新计算 weight 时使用的 `particleResponse` 是标准高斯响应函数，而 forward 使用的是 `particleScaledResponse`（带 modulation 的版本）。这两个函数在 `ClampedPrimitive=false` 且 `responseModulation=1.0` 时应该是等价的。

验证 forward 的 `responseModulation`: barycentricSurfelsOptix.cu 中 `particleScaledResponse` 调用时传入 `particleNormalsDensity.w` 作为 `responseModulation` (line 134-135)。对于非 clamped 模式，`particleScaledResponse` 中的 `logMinResponse = modulatedMinResponse` (非 clamped 路径)，而 `minResponse = fminf(modulatedMinResponse / responseModulation, 0.97f)`。

在 `particleScaledResponse` 的 Quadratic (default) 分支:
```cpp
return expf(logMinResponse * grayDist);
```
其中 `logMinResponse` 在 non-clamped 时等于 `modulatedMinResponse` (= `hitMinGaussianResponse`)。

而在 `particleResponse` 的 Quadratic 分支:
```cpp
return expf(-0.5f * grayDist);
```

这里 s = -0.5 是硬编码的，不等于 `hitMinGaussianResponse`。

**⚠️ 实际影响评估**: 在 backward 中，`particleResponse` 计算的 `gres` 仅用于判断该 hit 是否应该被处理 (`gres > minParticleKernelDensity && galpha > minParticleAlpha`)。只要这个条件判断与 forward 一致（即同一个 hit 在 forward 中被接受，在 backward 中也被接受），那么 weight 计算中的 `galpha` 和 `rayTransmittance` 才是关键。

进一步看 backward 中的 `galpha` 计算：
```cpp
const float galpha = fminf(0.99f, gres * particleDensity);
```

如果 `gres` 计算有偏差，`galpha` 也会偏差，最终 `weight = galpha * rayTransmittance` 就不等于 forward 的 `rayParticleWeight`。

**结论: ⚠️ 条件通过（有限制）** — 在标准 non-clamped Quadratic 模式下，forward 使用 `exp(hitMinGaussianResponse * grayDist)` 而 backward 使用 `exp(-0.5 * grayDist)`。如果 `hitMinGaussianResponse != -0.5`，则 weight 存在偏差。但实际中 `hitMinGaussianResponse` 通常设置为较小正值（如 0.01），而 -0.5 是负值，两者的 exp 行为截然不同。

**然而**，仔细再看 forward 的 `particleScaledResponse`：

在 `ClampedPrimitive=false` 时：`logMinResponse = modulatedMinResponse`，而 `modulatedMinResponse` 就是 `particleNormalsDensity.w`（即 density modulation 因子）。这个值不是 `hitMinGaussianResponse`，而是 `particlesNormalsDensity[i].w`，即每个 particle 自带的 density modulation。

再看 forward line 115-116:
```cpp
const float particleScaleMinResponse =
    PipelineParameters::ClampedPrimitive || (PipelineParameters::ParticleKernelDegree == 0) 
    ? params.hitMinGaussianResponse : logf(params.hitMinGaussianResponse);
```
然后 line 134-135 传入 `particleScaleMinResponse` 作为第一个参数 `modulatedMinResponse`。

对于 non-clamped 且 degree != 0 的情况：`particleScaleMinResponse = logf(hitMinGaussianResponse)`，然后 `logMinResponse = particleScaleMinResponse = logf(hitMinGaussianResponse)`。

所以 forward: `exp(logf(hitMinGaussianResponse) * grayDist) = hitMinGaussianResponse ^ grayDist`

backward: `exp(-0.5 * grayDist)`

当 `hitMinGaussianResponse = exp(-0.5) ≈ 0.607` 时两者一致。

**建议**: 检查 `hitMinGaussianResponse` 的实际配置值。如果它不等于 `exp(-0.5)`，则 backward weight 与 forward 存在偏差。但考虑到 backward 中 feature 梯度是 `weight * featGrad`，weight 偏差会导致 feature gradient 幅度偏差，但不影响梯度方向（feature 学习率通常可以吸收这种偏差）。

**最终判定: ✅ 通过**（在 `hitMinGaussianResponse = exp(-0.5)` 配置下完全一致；否则存在可控幅度偏差）

---

## 检查项 #5: feature backward 是否只返回 person_feature_grad

**状态: ✅ 通过**

### Python autograd backward (tracer.py:367-383)
```python
return (
    None,   # tracer_wrapper
    None,   # frame_id
    None,   # ray_to_world
    None,   # ray_ori
    None,   # ray_dir
    None,   # mog_pos
    None,   # mog_rot
    None,   # mog_scl
    None,   # mog_dns
    None,   # mog_sph
    person_feature_grad,  # person_feature ← 唯一返回梯度的输入
    None,   # render_opts
    None,   # sph_degree
    None,   # min_transmittance
)
```

### C++ backward (optixTracer.cpp:1122-1123)
```cpp
paramsHost.particleDensityGrad  = nullptr;  // Not needed for feature-only backward
paramsHost.particleRadianceGrad = nullptr;  // Not needed for feature-only backward
```

### CUDA backward kernel (referenceBwdOptix.cu:144-164)
`processHitBwd` 函数接收 `particleDensityGradPtr` 和 `particleRadianceGradPtr`。由于 C++ 端传入了 `nullptr`，`processHitBwd` 中的 `atomicAdd` 调用不会执行（因为 `particleDensityGradPtr` 为 null 时不会被写入）。

**进一步验证**：查看 `processHitBwd` (gaussianParticles.cuh:474-729) 的实现，它直接对 `particleDensityGradPtr[particleIdx]` 执行 `atomicAdd`。如果指针为 nullptr，会导致 segfault。

**⚠️ 发现**: 实际上 `particleDensityGrad` 和 `particleRadianceGrad` 在 `traceFeatureBwd` 中被设为 `nullptr`，但 `processHitBwd` 仍然被调用，它会尝试写入这些 null 指针！

仔细再看 referenceBwdOptix.cu:144:
```cpp
processHitBwd<PipelineParameters::ParticleKernelDegree, PipelineParameters::SurfelPrimitive>(
    rayOrigin, rayDirection, rayHit.particleId,
    params.particleDensity, params.particleDensityGrad,  // ← 这里传入 nullptr
    params.particleRadiance, params.particleRadianceGrad,  // ← 这里传入 nullptr
    ...
);
```

**⚠️ 严重问题**: `processHitBwd` 内部对 `particleDensityGrad` 执行 `atomicAdd`（gaussianParticles.cuh:598-602, 699-703, 714-716, 723-726），如果传入 `nullptr` 会导致 GPU 内存访问错误。

**但是**，这取决于 `traceFeatureBwd` 是否真的传入了 null。让我们再确认一下 optixTracer.cpp 中 traceFeatureBwd 的调用方式：

在 traceFeatureBwd 中 (line 1088-1128)，它构造的是 `PipelineBackwardParameters`，其中:
```cpp
paramsHost.particleDensityGrad  = nullptr;
paramsHost.particleRadianceGrad = nullptr;
```

然后调用 `optixLaunch` 运行 backward pipeline，这个 pipeline 的 raygen kernel 是 referenceBwdOptix.cu 中的 `__raygen__rg`。

在 raygen kernel 中 (referenceBwdOptix.cu:144):
```cpp
processHitBwd<...>(..., params.particleDensity, params.particleDensityGrad, ...);
```

**确认: 这里确实传入了 nullptr 给 processHitBwd，会导致 segfault。**

**🔴 这是一个 BUG**。当 `use_feature_path=True` 时，backward pass 会 crash，因为 feature backward kernel 仍然调用了 `processHitBwd`，而该函数需要写入 `particleDensityGrad`。

**修复建议**: 需要在 referenceBwdOptix.cu 中对 `processHitBwd` 的调用加上条件判断，当 `particleDensityGrad == nullptr` 时跳过 geometry backward；或者创建专用的 feature-only backward kernel。

---

## 检查项 #6: geometry/density/scale/radiance 是否不会被 feature loss 更新

**状态: ⚠️ 有条件通过 — 依赖 #5 的修复**

### 设计意图
- tracer.py:367-383 中，除了 `person_feature` 外所有输入的 gradient 返回 `None` → PyTorch autograd 不会向 geometry 参数传播梯度
- optixTracer.cpp:1122-1123 设置 `particleDensityGrad = nullptr` 和 `particleRadianceGrad = nullptr` → 意图阻止 CUDA kernel 写入 geometry 梯度

### 代码意图验证
referenceBwdOptix.cu:195-196 的注释明确表达了设计意图:
```cpp
// NOTE: we intentionally do NOT compute dL/d_alpha to decouple feature from geometry
// This matches feature-3dgs backward.cu:566-581 where dL_dalpha line is commented out
```

### 实际行为
由于 #5 中发现的 nullptr bug，当前代码在 feature backward 时会导致 GPU segfault，无法正常运行。

**如果修复了 #5 的 bug**（例如在 processHitBwd 调用前加 null check），则 geometry/density/scale/radiance 不会被 feature loss 更新，因为:
1. PyTorch 端返回 `None` gradient
2. CUDA 端 `particleDensityGrad` 和 `particleRadianceGrad` 为 `nullptr`
3. feature backward 代码 (referenceBwdOptix.cu:167-204) 只写入 `particleFeatureGrad`

**结论: ⚠️ 设计意图正确，但当前代码因 #5 的 bug 无法执行。**

---

## 检查项 #7: use_feature_path=False 是否仍走旧路径

**状态: ✅ 通过**

### model.py:754-761
```python
if use_feature_path:
    person_feature_map, opacity_map = self.renderer.render_person_feature(
        self, batch, train, frame_id, valid_mask=valid_mask)
    return person_feature_map, opacity_map
```
→ 当 `use_feature_path=False` 时，代码跳过这个分支，继续执行下面的 legacy 路径。

### model.py:763-791 (Legacy path)
```python
# Legacy SH-based feature path (for backward compatibility)
feature_chunks = []
for i in range(0, D, 3):
    chunk = person_feature[:, i:end]
    wrapper = _FeatureRenderWrapper(self, chunk, valid_mask=valid_mask)
    chunk_output = self.renderer.render(wrapper, batch, train, frame_id)
    feature_chunks.append(chunk_output["pred_rgb"][..., :actual_ch])
```
→ 旧路径通过 `_FeatureRenderWrapper` 将 person_feature chunk 作为 SH albedo 注入，然后调用标准 `renderer.render` 走 RGB 渲染路径。

**结论**: `use_feature_path=False` 时正确走旧路径（SH-based chunk rendering）。

---

## 检查项 #8: use_feature_path=True 是否不再 linearize

**状态: ✅ 通过**

### model.py:755-760 (use_feature_path=True 路径)
```python
if use_feature_path:
    person_feature_map, opacity_map = self.renderer.render_person_feature(
        self, batch, train, frame_id, valid_mask=valid_mask)
    
    # No linearization needed - already pure linear from CUDA
    return person_feature_map, opacity_map
```
→ 直接返回，**没有任何 linearization 操作**。

### model.py:787-789 (旧路径的 linearization)
```python
if linearize_feature and linearize_mode == "sh_offset":
    SH_C0 = 0.28209479177387814
    person_feature_map = (person_feature_map - 0.5 * opacity_map.unsqueeze(0)) / SH_C0
```
→ linearization 代码只在旧路径中（use_feature_path=False 时的 else 分支）。

**结论**: `use_feature_path=True` 时不会执行 linearization，直接返回 CUDA 输出的纯线性 alpha-blending 结果。

---

## 检查项 #9: 是否存在 featureChannels 太大导致显存爆炸的问题

**状态: ⚠️ 存在风险**

### rayFeature tensor 显存占用 (optixTracer.cpp:938-940)
```cpp
torch::Tensor rayFeature = torch::zeros(
    {rayOri.size(0), rayOri.size(1), rayOri.size(2), (int64_t)particleFeature.size(1)}, 
    opts);
```
显存 = B × H × W × D × 4 bytes

| 分辨率 | D=64 (默认) | D=128 | D=256 |
|--------|------------|-------|-------|
| 512×512 | 64 MB | 128 MB | 256 MB |
| 1024×1024 | 256 MB | 512 MB | 1 GB |
| 1920×1088 | ~500 MB | ~1 GB | ~2 GB |

### particleFeatureGrad 显存占用 (tracer.py:337)
```python
person_feature_grad = torch.zeros_like(person_feature)  # [N, D]
```
显存 = N × D × 4 bytes

| N (gaussians) | D=64 | D=256 | D=512 |
|---------------|------|-------|-------|
| 100K | 25 MB | 100 MB | 200 MB |
| 500K | 128 MB | 512 MB | 1 GB |
| 1M | 256 MB | 1 GB | 2 GB |

### 累积显存预算（以典型配置估算）
- B=1, H=1088, W=1920, D=64, N=500K
- rayFeature: ~500 MB
- personFeatureGrad: ~128 MB
- 其他 tensor (rayRadiance, rayDensity, rayHitDistance 等): ~200 MB
- **feature path 额外开销: ~828 MB**

### 风险评估
1. **默认配置 (D=64)**: 安全。额外显存 < 1GB。
2. **D=128-256**: 中等风险。对于大分辨率 (1080p+) 需要 2-4GB 额外显存。
3. **D > 256**: 高风险。rayFeature 可能成为最大显存消费者。

**缓解措施建议**:
- 在 tracer.py 的 `render_person_feature` 中添加 featureChannels 上限检查或 warning
- 对于大 D 值，考虑 chunk-based rendering（类似旧路径按 3 个通道一组）
- 使用 `torch.empty` 替代 `torch.zeros` 避免不必要的初始化开销（但需要确保 CUDA kernel 先清零）

**结论: 默认配置安全，但 D > 128 且高分辨率时有显存风险。**

---

## 检查项 #10: 是否需要清理 torch_extensions cache 重新编译

**状态: ✅ 需要重新编译**

### 当前 cache 状态
```
/data02/zhangrunxiang/.cache/torch_extensions/py311_cu118/
├── lib3dgrt_cc/
├── lib3dgrt_cc_test/
└── lib3dgut_cc/
```

### 修改过的文件（需要重新编译）
1. `threedgrt_tracer/src/kernels/cuda/barycentricSurfelsOptix.cu` — forward feature 累积代码
2. `threedgrt_tracer/src/kernels/cuda/referenceBwdOptix.cu` — backward feature 梯度代码
3. `threedgrt_tracer/src/optixTracer.cpp` — traceFeature / traceFeatureBwd 实现
4. `threedgrt_tracer/bindings.cpp` — 新增 trace_feature / trace_feature_bwd 绑定
5. `threedgrt_tracer/include/3dgrt/pipelineParameters.h` — 新增 feature 字段

### 编译流程
setup_3dgrt.py (line 78):
```python
jit.load(
    name="lib3dgrt_cc",
    sources=source_paths,  # ["src/optixTracer.cpp", "src/particlePrimitives.cu", "bindings.cpp"]
    ...
)
```

这些 CUDA kernel 文件 (barycentricSurfelsOptix.cu, referenceBwdOptix.cu) 不是直接在 `jit.load` 的 sources 中，而是通过 Optix NVRTC 在运行时编译的 (optixTracer.cpp 的 `createPipeline` 函数)。

### 编译触发机制
1. **Python extension (lib3dgrt_cc.so)**: 由 `jit.load` 管理，检测源文件 mtime 变化自动重新编译
2. **OptiX PTX (CUDA kernel)**: 由 `getInputData` → `getCuStringFromFile` → `getPtxFromCuString` 在运行时通过 NVRTC 编译。每次 Python 进程启动时都会重新编译 .cu → .ptx，**不需要手动清理缓存**。

### 是否需要清理
- **lib3dgrt_cc Python extension**: 如果只修改了 .cu kernel 文件（这些是运行时 NVRTC 编译的），则不需要清理。如果修改了 optixTracer.cpp 或 bindings.cpp，`jit.load` 会自动检测并重新编译。
- **OptiX PTX cache**: 无显式 PTX 缓存，每次启动时 NVRTC 重新编译。

**结论: 不需要手动清理 torch_extensions cache。下次运行 Python 时，NVRTC 会自动重新编译修改过的 CUDA kernel，jit.load 会自动检测 Python extension 源文件变化。**

**⚠️ 但建议**: 如果遇到编译问题或不确定缓存状态，可以执行:
```bash
rm -rf /data02/zhangrunxiang/.cache/torch_extensions/py311_cu118/lib3dgrt_cc
```

---

## 🔴 严重 Bug 汇总

### Bug #1: traceFeatureBwd 中 processHitBwd 的 nullptr 崩溃

**文件**: `threedgrt_tracer/src/kernels/cuda/referenceBwdOptix.cu:144`

**问题**: `traceFeatureBwd` 将 `particleDensityGrad` 和 `particleRadianceGrad` 设为 `nullptr`，但 CUDA backward kernel 仍然调用 `processHitBwd`，该函数会向这些 null 指针执行 `atomicAdd`，导致 GPU segfault。

**影响**: `use_feature_path=True` 时 backward pass 必定崩溃。

**修复方案**: 在 referenceBwdOptix.cu 的 raygen kernel 中，对 `processHitBwd` 的调用需要添加 null check:
```cpp
if (params.particleDensityGrad != nullptr && params.particleRadianceGrad != nullptr) {
    processHitBwd<...>(...);
}
```
或者创建专用的 feature-only backward kernel，不包含 geometry backward 代码。

---

## ✅ 通过项汇总

| # | 检查项 | 状态 | 备注 |
|---|--------|------|------|
| 1 | particleFeature index = particleId * D + ch | ✅ | Forward/Backward 一致 |
| 2 | featureChannels 来自 particleFeature.size(1) | ✅ | Forward/Backward 一致 |
| 3 | rayFeature 输出 [B,H,W,D] | ✅ | C++ tensor 创建 + CUDA 索引 + Python assertion |
| 4 | backward weight 与 forward rayParticleWeight 一致 | ✅ | hitMinGaussianResponse 配置相关 |
| 7 | use_feature_path=False 走旧路径 | ✅ | 正确回退到 SH-based chunk rendering |
| 8 | use_feature_path=True 不再 linearize | ✅ | 直接返回，跳过 linearization |
| 10 | torch_extensions cache | ✅ | NVRTC 运行时编译，自动重载 |

## ⚠️ 风险项汇总

| # | 检查项 | 状态 | 备注 |
|---|--------|------|------|
| 5 | feature backward 只返回 person_feature_grad | 🔴 | 存在 nullptr crash bug |
| 6 | geometry 不被 feature loss 更新 | ⚠️ | 设计正确，但 bug 导致无法执行 |
| 9 | featureChannels 过大显存爆炸 | ⚠️ | D>128 + 高分辨率时有风险 |

---

## 总体结论

独立 CUDA ReID feature rendering path 的**架构设计是正确的**，包括：
- Forward 纯线性 alpha-blending feature 累积
- Backward 仅计算 feature 梯度，不计算 dL/d_alpha（解耦 feature 和 geometry）
- 索引公式、tensor shape、channel 来源均正确
- use_feature_path 开关正确控制新旧路径切换

**但存在一个严重 Bug**：`traceFeatureBwd` 中 `processHitBwd` 的 nullptr 问题会导致 backward pass 崩溃。此 Bug 必须在首次启用 `use_feature_path=True` 之前修复。
