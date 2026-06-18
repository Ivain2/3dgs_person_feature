# OptiX 环境问题诊断报告

## 问题的真相

### 1. 之前能跑，现在报错的原因

**之前能跑**：是因为当时 `setup_3dgrt.py` 运行了 `jit.load("lib3dgrt_cc", ...)`，PyTorch 会在 `~/.cache/torch_extensions/py311_cu118/lib3dgrt_cc/` 下编译并缓存扩展模块。import 时直接从缓存加载，不重新编译。

**现在报错**：有两个问题叠加：
1. **CUDA 代码修改后重新编译** → 触发了 rebuild → 编译成功 → 但加载时 OptixTracer 初始化时 segfault
2. **诊断脚本的 import 问题**：`threedgrt_tracer` 模块不是通过 pip/conda 安装的，而是通过 `setup_3dgrt.py` 的 `jit.load` 动态编译并注册到 `torch_extensions`。单独 import 时需要先调用 `setup_3dgrt(conf)` 才能成功。

### 2. 当前真实状态

| 项目 | 状态 | 说明 |
|------|------|------|
| CUDA 代码编译 | ✅ 成功 | lib3dgrt_cc.so 编译通过 |
| 代码逻辑修复 | ✅ 完成 | transmittance replay, early termination, threshold check |
| `import threedgrt_tracer` | ❌ 需要先 setup | 这不是 bug，是设计如此 |
| OptixTracer 初始化 | ❌ SIGSEGV | **真正的问题在这里** |

### 3. OptixTracer SIGSEGV 的根本原因

**系统中不存在 `libnvoptix.so.1`**：
- `find / -name "libnvoptix*"` → NOT FOUND
- `ldconfig -p | grep nvoptix` → 空
- `ctypes.CDLL('libnvoptix.so.1')` → FAIL

**这意味着**：这台服务器从未安装过 NVIDIA OptiX SDK runtime 库。

**那之前是怎么跑的？**

两种可能：
1. **之前没真正跑过 OptiX 渲染** — 只做了代码编译和静态检查，没有创建 OptixTracer 实例
2. **之前的环境有 OptiX** — 如果之前在其他机器/容器/环境中运行过，那台机器装了 OptiX SDK

---

## 如何安装 OptiX

### 方案 1：安装 OptiX SDK（推荐）

```bash
# 下载 OptiX 7.5.0 SDK（与项目中 header 版本一致）
# 需要 NVIDIA 开发者账号
# https://developer.nvidia.com/designworks/optix/downloads/legacy

# 假设下载到 /tmp
cd /tmp
# 解压后会有 runfile
chmod +x NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64.run
sudo ./NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64.run --target /opt/optix-7.5.0

# 设置环境变量
export OPTIX_INSTALL_DIR=/opt/optix-7.5.0
export LD_LIBRARY_PATH=/opt/optix-7.5.0/lib64:$LD_LIBRARY_PATH

# 验证
ls /opt/optix-7.5.0/lib64/libnvoptix.so.1
```

### 方案 2：通过 conda 安装（如果可用）

```bash
# 某些 conda channel 可能有 optix
conda install -c nvidia optix  # 不一定有，需要验证
```

### 方案 3：联系服务器管理员

```bash
# 询问管理员是否有 OptiX SDK 已安装在系统其他地方
find / -name "libnvoptix*" 2>/dev/null
# 如果管理员可以安装，请安装 NVIDIA OptiX SDK 7.5+
```

### 方案 4：在有 OptiX 的 GPU 节点上运行

如果实验室/公司有 RTX 系列 GPU 的节点（带 RT Cores 和 OptiX SDK），把代码和数据同步到那里运行诊断脚本。

---

## 当前可以做什么

### 不需要 OptiX 的：
1. ✅ **代码审计** — 已完成，transmittance/early-termination 全部修复
2. ✅ **CUDA 编译** — 已完成
3. ✅ **2D→3D 聚合** — 之前已完成
4. ✅ **旧 SH path 的评估** — 之前已跑过（V3.0.2/V3.0.3 的 render2d 结果）

### 需要 OptiX 的：
1. ❌ **新 feature path 的线性度测试**
2. ❌ **Gradient check**
3. ❌ **3D ReID 三路对比（含 use_feature_path=True）**
4. ❌ **Frozen Probe**

### 可以做的变通：

**旧 SH path 的评估可以跑**（因为之前已经跑过 V3.0.2/V3.0.3 的 render2d），但 `use_feature_path=True` 需要 OptiX。

---

## 建议

1. **确认之前是否真正跑过 OptiX 渲染**：
   ```bash
   # 查看 outputs 中是否有包含 OptixTracer 运行的实验结果
   ls outputs/v3*render2d*/
   ```

2. **如果有 RTX GPU 节点**：把 `/data02/zhangrunxiang/3dgrut` 同步过去，在那边跑诊断

3. **如果只能在这台机器**：联系管理员安装 OptiX SDK，或者尝试从 NVIDIA 官网下载

4. **如果短期无法解决**：可以先在旧 SH path 上做 V4 实验（使用 `use_feature_path=False` + linearize），验证 ReID 训练流程是否正确，等 OptiX 就绪后再切换到新 feature path 对比
