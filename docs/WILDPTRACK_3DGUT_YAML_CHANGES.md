# wildtrack_3dgut.yaml 修改原因与理由

下面逐项说明：**原来写的是什么**、**为什么那样写无效或错误**、**改成了什么**、**依据的代码或配置**。

---

## 1. `model.background_color` → `model.background.color`

**原来：**
```yaml
model:
  background_color: [0.0, 0.0, 0.0]
```

**为什么错：**
- 代码里**没有** `config.model.background_color` 这个键。
- 实际用的是 **`config.model.background.color`**，且值必须是**字符串** `"black"`、`"white"` 或 `"random"`，不是 RGB 列表。

**依据：**
- `threedgrut/render.py` 第 50–55 行：`conf.model.background.color == "black"` / `"white"`。
- `threedgrut/datasets/__init__.py`：`config.model.background.color` 传给 `bg_color`。
- `threedgrut/model/background.py` 第 64–76 行：`self.config.color` 且断言为 `"black"` / `"white"` / `"random"`。

**改成：**
```yaml
model:
  background:
    color: black
```
这样才和 base_gs 以及代码一致，背景会按“黑色”正确生效。

---

## 2. `model.num_points` → 删除，改用 `initialization.num_gaussians`

**原来：**
```yaml
model:
  num_points: 50000
```

**为什么错：**
- 整个项目里**没有任何地方**读取 `config.model.num_points`。
- 随机初始化时，点数来自 **`config.initialization.num_gaussians`**。

**依据：**
- `threedgrut/trainer.py` 第 249–251 行：`model.init_from_random_point_cloud(num_gaussians=conf.initialization.num_gaussians, ...)`。
- `configs/initialization/random.yaml` 里只有 `num_gaussians: 100_000`，没有 `num_points`。

**改成：**
- 删掉 `model.num_points`。
- 在 app 里写 `initialization.num_gaussians: 50000`，这样“初始 5 万个点”的意图才会被代码用到。

---

## 3. `train.batch_size` → 删除

**原来：**
```yaml
train:
  batch_size: 1
```

**为什么无效：**
- 训练时 DataLoader 的 `batch_size` 是**在代码里写死的 1**，没有从 config 里读。
- 所以写 `config.train.batch_size` 也不会被用到。

**依据：**
- `threedgrut/trainer.py` 第 144、154 行：创建 DataLoader 时直接写 `"batch_size": 1`，没有用 `conf.train.batch_size`。

**改成：**
- 删掉 `train.batch_size`，避免让人误以为可以在这里改 batch 大小。

---

## 4. `render.num_rays` → 删除

**原来：**
```yaml
render:
  num_rays: 1024
```

**为什么无效：**
- 代码里**没有任何** `conf.render.num_rays` 或 `config.render.num_rays` 的引用。
- 渲染/训练用的光线数由别处逻辑决定，不是这个键。

**改成：**
- 删掉 `render.num_rays`，因为写了也不会起作用。

---

## 5. `optimizer.lr_density` → 删除；学习率只保留 `optimizer.lr`

**原来：**
```yaml
optimizer:
  lr: 0.001
  lr_density: 0.01
```

**为什么错：**
- 项目里**没有** `config.optimizer.lr_density` 这个键。
- 每个参数的学习率在 **`config.optimizer.params`** 下，例如 density 的学习率是 **`optimizer.params.density.lr`**（在 base_gs 里是 0.05）。

**依据：**
- `threedgrut/model/model.py` 第 536、553 行：遍历 `conf.optimizer.params.items()`，每个 param group 的 lr 来自 `args`（即 `optimizer.params.positions`、`optimizer.params.density` 等），没有 `lr_density`。

**改成：**
- 删掉 `optimizer.lr_density`。
- 保留 `optimizer.lr: 0.001`，这是 Adam 构造时的默认 lr，代码里会用（`conf.optimizer.lr`）。若以后要改 density 的学习率，应在 base 或 app 里写 `optimizer.params.density.lr`。

---

## 6. `eval_frequency` → 删除

**原来：**
```yaml
eval_frequency: 20
```

**为什么无效：**
- 代码里**没有** `eval_frequency`，只有 **`val_frequency`**（验证频率）。
- base_gs 里也只有 `val_frequency: 5000`，没有 `eval_frequency`。

**依据：**
- `threedgrut/trainer.py` 第 121、743 行：只用 `conf.val_frequency` 决定何时做验证。

**改成：**
- 删掉 `eval_frequency`。需要控制验证频率时，用已有的 `val_frequency: 10` 即可。

---

## 7. 其他保留项（为什么保留）

| 项 | 原因 |
|----|------|
| `path` | 顶层必填，用于日志/实验名、Renderer 等（base_gs 里 `path: ???`）。 |
| `out_dir` | 控制输出目录，base_gs 默认 `./runs`，这里显式写便于统一。 |
| `val_frequency: 10` | 代码用 `conf.val_frequency` 控制每多少步做一次验证。 |
| `n_iterations: 100` | 代码用 `conf.n_iterations` 控制总训练步数。 |
| `dataset.data_dir` 等 | WildTrack 必须用 `dataset.data_dir` 指定数据根目录，downsample、test_split_interval 数据集代码都会读。 |
| `optimizer.lr: 0.001` | 代码用 `conf.optimizer.lr` 作为 Adam 的默认学习率。 |

---

## 总结

- **改动的目的**：只保留**代码里真实用到的配置键**，且**键名和取值格式**与 base_gs 及代码一致。
- **删掉的**：要么键名不存在（如 `background_color`、`lr_density`、`eval_frequency`），要么键存在但代码从不读（如 `num_points`、`train.batch_size`、`render.num_rays`）。
- **改写的**：把“想表达的意思”放到**正确的键**下（如背景 → `model.background.color`，初始点数 → `initialization.num_gaussians`），这样配置才会真正生效。

这样改完之后，yaml 里写的每一项都有据可查、会被用到，也不会因为写错键而导致行为不符合预期或 Hydra 合并出奇怪结果。
