# OptiX 环境问题诊断与解决方案

## 问题描述

之前 3DGRUT 训练正常（Phase 12-16 产出了 checkpoint），但现在 `OptixTracer` 创建时 segfault (exit code -11)。

## 问题根因

经过彻底排查：

1. **`libnvoptix.so.1` 在系统中完全不存在**（全系统搜索无结果）
2. **driver 535.216.01 应该自带此库**，但当前环境没有
3. **之前能跑的原因**：当时的 `.so` 缓存文件还在，没有重新编译
4. **现在不行的原因**：我们清除了缓存 `rm -rf ~/.cache/torch_extensions/py311_cu118/lib3dgrt_cc`，触发了重新编译，但新编译的 `.so` 在 runtime 加载 `libnvoptix.so.1` 时失败

### 可能的原因（按可能性排序）

| # | 原因 | 可能性 |
|---|------|--------|
| 1 | 系统管理员更新了 driver，但新安装不完整 | 高 |
| 2 | 容器环境下 driver 挂载不完整 | 中 |
| 3 | conda 环境重建导致 | 低 |

## 解决方案

### 方案 1：联系服务器管理员（推荐）

请求管理员：
```bash
# 重新安装/修复 nvidia driver 535.216.01
sudo apt install --reinstall nvidia-driver-535
# 或者
sudo ldconfig  # 更新动态链接库缓存
# 检查
ldconfig -p | grep nvoptix
```

### 方案 2：从 NVIDIA 下载 OptiX SDK（需要 NVIDIA 开发者账号）

```bash
# 下载 OptiX SDK 7.7（与 driver 535 兼容）
# https://developer.nvidia.com/designworks/optix/downloads/legacy
# 选择 Linux 版本

# 安装到用户目录（无需 root）
cd /tmp
# 假设下载到 /tmp/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh
chmod +x NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh
./NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh --target $HOME/optix-7.7.0

# 设置环境变量
echo 'export OPTIX_INSTALL_DIR=$HOME/optix-7.7.0' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/optix-7.7.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
ls $HOME/optix-7.7.0/lib64/libnvoptix.so.1
ldconfig -p | grep nvoptix
```

### 方案 3：从 NVIDIA driver runfile 中提取（最快速）

```bash
# 下载与当前 driver 版本一致的 runfile
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.216.01/NVIDIA-Linux-x86_64-535.216.01.run -O /tmp/nvidia-driver.run

# 提取而不安装
cd /tmp
sh NVIDIA-Linux-x86_64-535.216.01.run --extract-only /tmp/nvidia-extract

# 找到 libnvoptix.so
find /tmp/nvidia-extract -name "libnvoptix*"
# 应该在 libnvidia-optixal.so.535.216.01

# 复制到用户目录并创建符号链接
mkdir -p $HOME/.local/lib
cp /tmp/nvidia-extract/libnvidia-optixal.so.535.216.01 $HOME/.local/lib/libnvoptix.so.1
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
python3 -c "import ctypes; ctypes.CDLL('libnvoptix.so.1'); print('SUCCESS')"
```

### 方案 4：暂时跳过 OptiX 验证，继续实验

如果短期无法解决 OptiX 环境问题，可以：
1. **代码静态审计已通过**（transmittance/early-termination 全部修复）
2. **跳过 runtime 验证**
3. **直接用旧 SH path + linearize 做 V4 实验**（`use_feature_path=False`）
4. 等 OptiX 环境修复后再切换到新 feature path 对比

## 诊断脚本

| 脚本 | 状态 |
|------|------|
| `tools/diagnose_optix_environment.py` | ✅ 就绪 |
| `tools/diagnose_feature_renderer_linearity.py` | ✅ 就绪 |
| `tools/diagnose_feature_path_gradient.py` | ✅ 就绪 |
| `tools/evaluate_3d_reid_aggregated_features.py` | ✅ 就绪 |
| `tools/train_3d_reid_probe.py` | ✅ 就绪 |

所有脚本都包含 graceful fallback，OptiX 不可用时会自动跳过并说明。

## 当前代码状态

| 项目 | 状态 |
|------|------|
| Transmittance replay 修复 | ✅ 完成 |
| Early termination 修复 | ✅ 完成 |
| Threshold check 修复 | ✅ 完成 |
| CUDA 编译 | ✅ 成功 |
| 代码静态审计 | ✅ 通过 |
| Runtime 验证 | ⏸️ 阻塞 |
