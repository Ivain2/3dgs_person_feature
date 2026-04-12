# WildTrack 小样本跑通检查清单

针对数据路径：`/data02/zhangrunxiang/data/Wildtrack_small_sample`。

## 1. 你的数据里有什么（与代码期望对照）

| 代码期望 | 你的样本 | 状态 |
|----------|----------|------|
| `Image_subsets/C1` … `C7`，每目录下 `.png` | `Image_subsets/C1`…`C7`，每目录 10 张 `00000000.png` … `00000045.png` | ✅ 一致 |
| `calibrations/intrinsic_original/intr_C1.xml` … `intr_C7.xml` | 存在，且为 OpenCV `camera_matrix` + `distortion_coefficients` + `image_size` | ✅ 一致 |
| `calibrations/extrinsic/extr_C1.xml` … `extr_C7.xml` | 存在，且含 `R`、`T` | ✅ 一致 |
| `annotations_positions/*.json`（可选） | 存在，用于 scene bbox | ✅ 有 |

结论：**目录结构和文件名与 `dataset_wildtrack.py` 的期望一致，不缺目录或命名。**

---

## 2. 标定格式（intr_C1.xml 等）

- **camera_matrix**：3×3，已含 fx, fy, cx, cy。
- **distortion_coefficients**：你的是 **5 个参数**（OpenCV 常见顺序为 `k1, k2, p1, p2, k3`）。
- **image_size**：1920×1080。

当前 `dataset_wildtrack.py` 的解析方式为：

- `radial_coeffs[:3] = dist_coeffs[:3]`（前 3 个当径向）
- `tangential_coeffs[:] = dist_coeffs[3:5]`（第 4、5 个当切向）

即代码假定畸变顺序为 **`[k1, k2, k3, p1, p2]`**。  
若你标定实际顺序是 OpenCV 常用的 **`[k1, k2, p1, p2, k3]`**，则径向/切向会错位，可能导致训练/渲染异常。若跑通后重投影明显不对，再考虑改这里或确认标定参数顺序。

---

## 3. Config 缺什么 / 怎么才能跑通

### 3.1 数据集路径（必须）

- WildTrack 用的是 **`config.dataset.data_dir`**，不是顶层的 `path`（见 `threedgrut/datasets/__init__.py` 里 `case "wildtrack"`）。
- **`configs/dataset/wildtrack.yaml`** 里**没有** `data_dir` 默认值，所以必须在某一层 config 里提供 `dataset.data_dir`。

你当前 **`configs/apps/wildtrack_3dgut.yaml`** 里已经写了：

```yaml
dataset:
  data_dir: /data02/zhangrunxiang/data/Wildtrack_small_sample
  downsample_factor: 1
  test_split_interval: 5
```

因此**用这个 app config 时，数据路径是齐的**，不需要再在 dataset 下补别的路径。

### 3.2 顶层 `path`（必须）

- `base_gs` 里 `path: ???` 是必填。
- 你的 app 里写了 `path: outputs/wildtrack_3dgut`，会被用作：
  - 日志/实验名（如 `object_name = Path(conf.path).stem`）
  - 输出目录等

所以**不打算改输出目录的话，不用动 `path`**；若想用别的输出目录，可命令行覆盖，例如：  
`path=outputs/my_wildtrack out_dir=runs`。

### 3.3 初始化

- WildTrack 使用 **random** 初始化即可（不依赖 COLMAP 的 `conf.path`）。
- 你的 app 已包含 `defaults: - /initialization: random`，**不需要**再为 WildTrack 单独配初始化。

### 3.4 小结：缺什么配置？

- **数据目录**：已在 `configs/apps/wildtrack_3dgut.yaml` 里通过 `dataset.data_dir` 指定，**不缺**。
- **顶层 path**：已在同一文件里设为 `outputs/wildtrack_3dgut`，**不缺**。
- 若**不用**该 app 文件、而是直接用 `--config-name dataset/wildtrack` 之类，则必须在命令行传 `dataset.data_dir`，例如：  
  `dataset.data_dir=/data02/zhangrunxiang/data/Wildtrack_small_sample`。

---

## 4. 推荐运行方式（验证跑通）

在项目根目录（含 `train.py` 的目录）下：

```bash
# 使用当前 app config，数据路径已在 yaml 里
python train.py --config-name apps/wildtrack_3dgut out_dir=runs experiment_name=wildtrack_small
```

如需覆盖数据路径或输出目录：

```bash
python train.py --config-name apps/wildtrack_3dgut \
  dataset.data_dir=/data02/zhangrunxiang/data/Wildtrack_small_sample \
  out_dir=runs experiment_name=wildtrack_small
```

若未使用 `apps/wildtrack_3dgut.yaml`（例如只用了 `dataset: wildtrack`），则**必须**显式传：

```bash
dataset.data_dir=/data02/zhangrunxiang/data/Wildtrack_small_sample
```

---

## 5. 仍可能出问题的地方（非“缺配置”）

1. **多相机 / contiguous**：WildTrack 多相机、多视角，容易触发之前文档里写的 Python/C++ 双重 `.contiguous()` 或显存问题；若报错在 backward/显存，可先对照 `docs/CUDA_AND_DATASET_NOTES.md`。
2. **畸变参数顺序**：如上，若标定是 `[k1,k2,p1,p2,k3]` 而代码按 `[k1,k2,k3,p1,p2]` 用，重投影会错；先跑通再根据可视化决定是否改 `dataset_wildtrack.py` 的解析。
3. **图像尺寸与 padding**：代码会把读到的图 pad 到 16 的倍数；标定里是 1920×1080，若某相机实际分辨率不一致，需要和 `_detect_image_dimensions` / downsample 逻辑一致。
4. **App 里无效键**：`configs/apps/wildtrack_3dgut.yaml` 里的 `model.num_points`、`train.batch_size`、`render.num_rays` 等若在代码里未被使用，不会影响跑通，只是多余；若 Hydra 报错 “unknown config key”，可删掉对应项。

按上面检查后，**数据与路径配置足以跑 WildTrack**；若仍报错，把报错信息与使用的完整命令贴出来即可进一步对一下。
