# CUDA 与换数据集相关说明

## 0. 修改 CUDA 文件的约定

- **不好定位问题的文件不要随意改**：CUDA/内核代码一旦出问题很难排查，修改前请确认必要性。
- **若必须修改，请先备份**：例如 `cp cameraProjections.cuh cameraProjections.cuh.bak` 或保留 `.orig` 副本，便于出问题时对比和回溯。
- 当前已在 `threedgut_tracer/include/3dgut/kernels/cuda/sensors/` 下保留 `cameraProjections.cuh.orig` 作为该文件的原始备份，供日后对照使用。

---

## 1. WildTrack 与推荐数据集的差异（为何更容易出问题）

**是的**：WildTrack 相比项目本身推荐的数据集（NeRF Synthetic、MipNeRF360、ScanNet++ 等）**更复杂**，因此**更容易暴露出**代码里已有的边界情况或设计问题。

| 方面 | 推荐数据集（NeRF/MipNeRF360/ScanNet++） | WildTrack |
|------|----------------------------------------|-----------|
| 相机数量 | 单场景、相机数量相对固定 | **多相机、多视角**，同一场景下多路图像 |
| 数据规模与内存 | 相对可控 | **显存与 CPU 内存压力更大**，batch/多视角切换更频繁 |
| 动态与格式 | 多为静态场景、格式统一 | 可能含**动态/时序**，数据组织方式不同 |

在这种“更复杂”的使用场景下，以下问题更容易被触发：

- **Python 与 C++ 双重 `.contiguous()`**：  
  `tracer.py` 里 backward 对 `torch.split` 结果调用了 `.contiguous()`，而 `splatRaster.cpp` 的 `voidDataPtr` 又对传入张量再次调用 `.contiguous()`。  
  在简单、每步单视角且分辨率较小的场景下可能表现正常；在 **WildTrack 这种多相机数据、每步单视角但分辨率大（1920×1088）、内存操作更频繁** 的场景下，更容易出现**内存布局不一致或多余拷贝**，表现为崩溃、梯度异常等，且难以定位。  
  项目内 `analyze_contiguous_issue.py` 也把 WildTrack 作为“更容易暴露出该问题”的典型场景做了说明。

因此：**不是 WildTrack 本身“错了”，而是它比推荐数据集更复杂，更容易把现有实现里潜在的问题（例如双重 contiguous）暴露出来。**

---

## 2. 其他已知问题与换数据集注意点（不修改 CUDA 的说明）

### 2.1 分辨率与 rolling shutter（仅说明，未改代码）

- `cameraProjections.cuh` 中 `relativeShutterTime` 使用 `(resolution.x - 1.f)` / `(resolution.y - 1.f)` 作分母，当分辨率为 1 时存在除零风险。
- 若使用**极小分辨率**或**极大 downsample_factor** 导致宽/高为 1，可能触发。  
  建议：换数据集时保证分辨率至少为 2；若确需修改 CUDA，请按第 0 节先备份再改。

### 2.2 configs/dataset 下各数据集对比（谁算“简单”、谁用广角）

| 数据集 | 类型 | 相机模型 | 说明 |
|--------|------|----------|------|
| **NeRF** (`nerf.yaml`) | 推荐、简单 | 纯针孔、无畸变 | 合成数据，单场景、固定分辨率，最容易跑通。 |
| **Colmap** (`colmap.yaml`) | 推荐、中等 | 针孔或 **OpenCV 鱼眼** | 来自 COLMAP 重建；支持 `PINHOLE` / `SIMPLE_PINHOLE` / `OPENCV_FISHEYE`，若标定是鱼眼则用广角模型。 |
| **ScanNet++** (`scannetpp.yaml`) | 推荐、鱼眼 | **OpenCV 鱼眼** | 广角/鱼眼数据，需按 [FisheyeGS](https://github.com/zmliao/Fisheye-GS) 预处理。 |
| **WildTrack** (`wildtrack.yaml`) | 非推荐、复杂 | **当前实现为 OpenCV 针孔** | 7 路相机、多视角；实现里用 `OpenCVPinholeCameraModelParameters`，从 `calibrations/intrinsic_original` 读 `camera_matrix` + `distortion_coefficients`（按针孔+径向/切向解析）。 |

因此：**“简单数据集”主要指 NeRF（以及单场景、每步单视角的 Colmap/ScanNet++ 使用方式）；WildTrack 是多相机数据、每步仍可单视角，但分辨率大、数据管线更复杂，且当前代码里是针孔模型。**

### 2.3 WildTrack 与 OpenCV 广角/鱼眼（为何你想做但没调通）

- **项目本身支持 OpenCV 广角/鱼眼**：Colmap 支持 `OPENCV_FISHEYE`，ScanNet++ 用鱼眼；CUDA 侧 `cameraProjections.cuh` 等也支持 `OpenCVFisheyeModel`。
- **WildTrack 在当前代码里是针孔**：`dataset_wildtrack.py` 只使用 `OpenCVPinholeCameraModelParameters`，`get_gpu_batch_with_intrinsics` 里传的是 `intrinsics_OpenCVPinholeCameraModelParameters`，`intrinsics_OpenCVFisheyeCameraModelParameters=None`。标定从 `intr_C1.xml` 等读 `camera_matrix` + `distortion_coefficients`，并按针孔模型的 k1,k2,k3,p1,p2（及可选的 thin prism）解析。
- **若 WildTrack 实际是广角/鱼眼**：  
  - 若官方或你使用的标定是 **OpenCV 鱼眼模型**（4 个径向系数、θ 多项式），当前按针孔解析会**重投影错误**，表现为训练不收敛、渲染错位或模糊。  
  - 要支持 WildTrack 广角，需要：用 `OpenCVFisheyeCameraModelParameters`、从标定文件读鱼眼参数（或先做一次标定格式转换），并在 `get_gpu_batch_with_intrinsics` 里传 `intrinsics_OpenCVFisheyeCameraModelParameters`，同时保证 tracer 侧按鱼眼路径调用（与 ScanNet++/Colmap fisheye 一致）。
- **没调通的其他常见原因**：  
  - 多相机导致的 **contiguous/显存/梯度** 问题（见第 1 节）。  
  - **数据路径与目录结构**：代码期望 `Image_subsets`、`calibrations/intrinsic_original`（`intr_C1.xml` 等）、`calibrations/extrinsic`（`extr_C1.xml` 等）；若目录名或文件名不一致会直接报错或读错。  
  - **图像尺寸与 padding**：实现里会把图像 pad 到 16 的倍数（如 1920×1088）；若标定里的 `image_size` 与当前读图尺寸不一致，需要和 downsample/pad 逻辑一致。

### 2.4 相机模型与数据格式（通用）

- 需提供与 `threedgrut/datasets/camera_models.py` 一致的 OpenCV 针孔或鱼眼参数。
- ScanNet++ 需按 [FisheyeGS](https://github.com/zmliao/Fisheye-GS) 的流程做数据预处理。

### 2.5 高斯数量与显存

- buffer 按 `numParticles` 动态分配，无硬编码上限；换到更大/更密场景可能 OOM，需适当降低分辨率或初始点数。

---

## 3. 小结

| 项目 | 说明 |
|------|------|
| **修改 CUDA** | 不好定位的不要随意改；若改，先备份（如 `.cuh.bak` / `.cuh.orig`），便于回溯。 |
| **WildTrack 更易出问题** | 相比推荐数据集更复杂（多相机、多视角、内存压力大），更容易暴露出如**双重 contiguous** 等问题；分析含义即如此。 |
| **contiguous 问题** | 建议在 Python 或 C++ 一侧统一做 contiguous，避免两侧重复调用；出问题时优先检查多相机/大 batch 路径。 |
| **简单 vs 复杂数据集** | NeRF 最简单；Colmap/ScanNet++ 为推荐、中等或鱼眼；WildTrack 为多相机、当前实现为针孔，若实际是广角需改用鱼眼模型。 |
| **WildTrack 没调通** | 可能原因：多相机/contiguous、数据路径与目录结构、相机模型不匹配（若实际是鱼眼而代码用针孔）、图像尺寸与标定不一致。 |
