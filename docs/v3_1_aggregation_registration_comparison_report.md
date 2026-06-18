# V3.1 聚合与注册对比报告

> 生成日期：2026-05-21
> 对比模式：**recall** / **pure07** / **balanced**
> 场景：3DGS + 2D ReID 特征聚合，评估不同注册策略对身份特征质量的影响

---

## 一、总览指标汇总表

### 1.1 注册统计（Registration Stats）

| 指标 | Recall | Pure07 | Balanced |
|------|--------|--------|----------|
| nonzero_count | **634** | 103 | 190 |
| nonzero_ratio | **1.00%** | 0.16% | 0.30% |
| zero_feature_ratio | 99.00% | 99.84% | 99.70% |
| dominant_ratio (mean) | 0.355 | **0.899** | 0.266 |
| id_entropy (mean) | 1.888 | **0.295** | 2.222 |
| camera_entropy (mean) | 0.044 | **0.008** | 0.109 |
| pairwise_cosine_mean | 0.989 | 0.976 | **0.214** |
| per_id_support (mean) | **2.039** | 0.331 | 0.611 |
| per_id_support (max) | **69** | 33 | 5 |
| feature_mode | weighted | weighted | clean-weighted |

### 1.2 C 路径 ReID 检索指标

| 指标 | Recall | Pure07 | Balanced |
|------|--------|--------|----------|
| mAP | 0.0226 | 0.0203 | **0.0233** |
| Rank-1 | 0.0170 | **0.0199** | 0.0172 |
| Rank-5 | 0.0636 | **0.0858** | 0.0585 |
| Rank-10 | 0.0939 | **0.1124** | 0.0909 |

### 1.3 C 路径 ReID 配对指标

| 指标 | Recall | Pure07 | Balanced |
|------|--------|--------|----------|
| pos_cosine_mean | 0.673 | 0.036 | 0.316 |
| neg_cosine_mean | 0.560 | 0.022 | 0.206 |
| pos_neg_gap | **0.113** | 0.014 | 0.110 |
| ROC-AUC | **0.592** | 0.507 | 0.585 |
| PR-AUC | **0.408** | 0.341 | 0.402 |
| EER | 0.419 | **0.023** | 0.428 |

### 1.4 Probe 指标

| 指标 | Recall-Linear | Recall-MLP | Pure07-Linear | Pure07-MLP | Balanced-Linear | Balanced-MLP |
|------|--------------|------------|---------------|------------|-----------------|--------------|
| id_accuracy | 0.0588 | 0.0611 | 0.0537 | 0.0601 | **0.0938** | **0.1717** |
| camera_probe_accuracy | 0.5107 | 0.5107 | **0.3468** | **0.3468** | 0.7799 | 0.7799 |
| probe mAP | 0.0225 | 0.0254 | 0.0203 | 0.0205 | 0.0234 | **0.0266** |
| probe Rank-1 | 0.0176 | 0.0216 | **0.0196** | 0.0109 | 0.0175 | 0.0159 |
| probe pos_neg_gap | **0.1111** | 0.0731 | 0.0190 | 0.0029 | 0.1023 | 0.0667 |
| probe ROC-AUC | 0.5848 | 0.5912 | 0.5098 | 0.4956 | 0.5796 | 0.5788 |

> 随机基线：id_accuracy = 1/90 ≈ 0.0111；camera_probe = 1/7 ≈ 0.1429

---

## 二、十维度对比分析

### 2.1 哪个版本 nonzero Gaussian 最多？

**Recall（634）>> Balanced（190）> Pure07（103）**

Recall 模式下 nonzero Gaussian 数量是 Pure07 的 6.2 倍、Balanced 的 3.3 倍。Recall 的宽松注册策略（低阈值、不要求纯度）让更多 Gaussian 获得非零特征，但这也意味着更多"脏"Gaussian 被纳入。Balanced 介于两者之间，约为 Recall 的 30%。

### 2.2 哪个版本 ROI support ratio 最高？

**Recall（2.039）>> Balanced（0.611）> Pure07（0.331）**

Recall 的 per_id_support_mean 远超另外两个模式，平均每个 ID 有约 2 个 Gaussian 支撑，最大值达到 69。Pure07 平均仅 0.331，大量 ID 完全没有 Gaussian 支撑（min=0）。Balanced 的 support 为 0.611，约为 Pure07 的 1.8 倍，但远不及 Recall。

### 2.3 哪个版本 dominant_ratio / id_entropy 最好？

**Pure07 的纯度指标最优**

- **dominant_ratio**：Pure07（0.899）>> Recall（0.355）> Balanced（0.266）
  - Pure07 几乎每个 Gaussian 都被单一 ID 主导（≥0.7 阈值生效）
  - Balanced 的 dominant_ratio 最低，说明其 Gaussian 的 ID 贡献最为分散

- **id_entropy**：Pure07（0.295）<< Recall（1.888）< Balanced（2.222）
  - Pure07 的 id_entropy 极低，身份信息高度集中
  - Balanced 的 id_entropy 最高，身份信息最为混杂

**结论**：Pure07 在纯度维度上一骑绝尘，但代价是覆盖面极窄。

### 2.4 哪个版本 feature cosine 最低？

**Balanced（0.214）<< Pure07（0.976）< Recall（0.989）**

Balanced 的 pairwise_cosine_mean 仅为 0.214，远低于 Recall（0.989）和 Pure07（0.976）。这表明 Balanced 的 clean-weighted 特征模式确实打破了"所有特征几乎相同"的退化现象。Recall 和 Pure07 的特征高度相似（cosine > 0.97），说明加权聚合后的特征几乎坍缩到同一方向。

**但低 cosine 不等于好的区分性**——Balanced 的 pos_neg_gap（0.110）并不优于 Recall（0.113），说明特征虽然不再坍缩，但正负对之间的区分度并未提升。

### 2.5 哪个版本 C 路径 ReID 最好？

**结果分裂，无全面赢家**

| 维度 | 最优 | 说明 |
|------|------|------|
| mAP | Balanced（0.0233） | 微弱领先 |
| Rank-1/5/10 | Pure07 | 排名指标全面领先 |
| pos_neg_gap | Recall（0.113） | 区分度最高 |
| ROC-AUC | Recall（0.592） | 分类能力最强 |

- **Pure07** 在 Rank 指标上最好，可能因为少量高纯度 Gaussian 提供了更干净的查询特征
- **Recall** 在 gap/AUC 上最好，说明更多 Gaussian 虽然有噪声，但整体信号更强
- **Balanced** mAP 微弱领先但 Rank 指标最差，说明其检索结果排序不够好

**三者 ReID 整体水平极低**（mAP < 2.5%，Rank-1 < 2%），均远未达到可用水平。

### 2.6 哪个版本 Linear/MLP Probe 最好？

**Balanced 在 ID probe 上显著领先**

- **id_accuracy**：Balanced-MLP（0.1717）是 Recall-MLP（0.0611）的 2.8 倍，是 Pure07-MLP（0.0601）的 2.9 倍
- **probe mAP**：Balanced-MLP（0.0266）> Recall-MLP（0.0254）> Pure07-MLP（0.0205）
- **MLP 一致优于 Linear**：三个模式下 MLP probe 的 id_accuracy 均高于 Linear

但需注意：即使 Balanced-MLP 的 id_accuracy 达到 0.172，在 90 类分类中仍远低于可用水平（随机基线 0.011），说明特征中的身份信号依然极弱。

### 2.7 Camera probe 是否强于 ID probe？

**是，且差距悬殊**

| 模式 | camera_probe_accuracy | id_accuracy (Linear) | id_accuracy (MLP) | camera/id 倍数 |
|------|----------------------|---------------------|-------------------|---------------|
| Recall | 0.5107 | 0.0588 | 0.0611 | 8.3x / 8.4x |
| Pure07 | 0.3468 | 0.0537 | 0.0601 | 6.5x / 5.8x |
| Balanced | 0.7799 | 0.0938 | 0.1717 | 8.3x / 4.5x |

- **所有模式下 camera probe 准确率远高于 ID probe**，说明 3D 聚合特征中相机信息远强于身份信息
- **Balanced 的 camera probe 最高（0.78）**，远超随机基线（0.143），说明 clean-weighted 模式虽然提升了身份可分性，但同时也大幅增强了相机编码
- Pure07 的 camera probe 最低（0.35），但仍远高于随机基线，且高于其 ID probe

**关键发现**：2D→3D 聚合过程中，相机视角信息被大量保留，而身份信息被严重稀释。

### 2.8 是否出现 recall 增加 support 但引入 camera/background bias？

**是**

- Recall 的 per_id_support_mean（2.039）最高，提供了最广的覆盖
- 但 Recall 的 camera_entropy（0.044）虽非最高，camera_probe_accuracy（0.5107）已远超随机（0.143），说明特征中混入了显著的相机偏差
- 更严重的是 **Balanced**：其 camera_entropy（0.109）是 Recall 的 2.5 倍，camera_probe_accuracy（0.7799）是 Recall 的 1.5 倍
- Balanced 的 clean-weighted 模式在试图去除背景噪声的同时，反而让相机视角信号变得更加突出

**结论**：Recall 增加的 support 确实带来了 camera/background 污染；Balanced 的去噪策略进一步放大了相机偏差。

### 2.9 是否出现 pure 提高 purity 但 support 太低？

**是**

- Pure07 的 dominant_ratio（0.899）和 id_entropy（0.295）表明纯度极高
- 但 nonzero_count 仅 103（Recall 的 16%），per_id_support_mean 仅 0.331
- 大量 ID 完全没有 Gaussian 支撑（per_id_support min=0），意味着这些 ID 在 3D 空间中"不可见"
- Pure07 的 C 路径 ReID Rank 指标虽好，但 mAP 最低（0.0203），说明高纯度但低覆盖导致很多查询完全无法匹配

**结论**：Pure07 的纯度策略有效，但覆盖面严重不足，大量 ID 被完全丢弃。

### 2.10 Balanced 是否同时改善 support 和 identity gap？

**部分改善，但引入严重 camera bias**

| 维度 | Recall | Pure07 | Balanced | 判定 |
|------|--------|--------|----------|------|
| per_id_support | 2.039 | 0.331 | 0.611 | 介于两者之间，约为 Pure07 的 1.8x |
| pos_neg_gap | 0.113 | 0.014 | 0.110 | 接近 Recall，远优于 Pure07 |
| id_accuracy (MLP) | 0.061 | 0.060 | **0.172** | 显著最优 |
| mAP | 0.0226 | 0.0203 | **0.0233** | 微弱最优 |
| camera_probe | 0.511 | 0.347 | **0.780** | 严重最差 |

- Balanced 在 identity 维度（id_accuracy、mAP）上确实是最优的
- support 相比 Pure07 有所提升（0.611 vs 0.331），但远不及 Recall（2.039）
- **致命问题**：camera_probe_accuracy 高达 0.78，意味着特征中约 78% 的信息可被用于预测相机，身份信号被相机信号严重压制
- Balanced 的 per_id_support max 仅 5（Recall 为 69），说明其覆盖仍然很薄

**结论**：Balanced 在身份维度上是最优候选，但 camera bias 问题使其不能直接作为后续方案。

---

## 三、决策判定

### 3.1 按决策规则逐条判定

| 规则 | 条件 | 判定 |
|------|------|------|
| recall support 高但 camera probe 更强 | ✅ Recall camera_probe=0.511 >> id_accuracy=0.061 | **coverage 有用，但污染严重** |
| pure gap 好但 support 太低 | ✅ Pure07 dominant_ratio=0.899, support=0.331 | **purity 有用，但 coverage 不够** |
| balanced 同时提升 ROI support、pos-neg gap、probe | ⚠️ 部分成立，但 camera_probe=0.78 | **身份维度最优，但 camera bias 严重，不能直接作为后续候选** |
| 三者 probe 都弱 | ✅ 最高 id_accuracy 仅 0.172（90 类） | **teacher prototype / 2D→3D aggregation 仍然失败** |

### 3.2 核心诊断

**当前 2D→3D 特征聚合的根本问题不是覆盖面或纯度的取舍，而是相机偏差对身份信号的系统性压制。**

证据链：
1. 所有模式下 camera_probe >> id_probe（倍数 4.5x~8.4x）
2. Balanced 的 clean-weighted 去噪反而放大了 camera 编码（0.78）
3. 即使是最优的 Balanced-MLP id_accuracy（0.172），在 90 类中也仅略高于随机
4. Recall 和 Pure07 的 pairwise_cosine > 0.97，特征几乎完全坍缩

### 3.3 下一步建议

**优先级 1（推荐）：common-component removal / foreground-aware aggregation**

- 当前 bbox 内 opacity 本身并非瓶颈——Recall 有 634 个 nonzero Gaussian，说明几何覆盖是够的
- 问题在于加权聚合时，相机共有的背景/环境分量被放大，而身份特有的前景分量被稀释
- 建议尝试：
  - **Common-component removal**：在聚合前对 2D 特征做 PCA/CCA，去除跨相机共有的主成分
  - **Foreground-aware aggregation**：仅使用人体分割 mask 内的像素特征，而非整个 bbox
  - **Camera-invariant projection**：在 2D→3D 投影时加入相机不变性约束

**优先级 2（备选）：teacher prototype 改进**

- 当前 teacher 特征可能本身就编码了较强的视角信息
- 可考虑使用视角不变的 teacher（如经 camera-contrastive 训练的模型）

**优先级 3（暂不推荐）：person-aware densification / 几何重训**

- 仅当证明 bbox 内 opacity 本身不足以覆盖人体区域时才考虑
- 当前 Recall 模式已有 634 个 nonzero Gaussian，几何覆盖并非首要瓶颈
- 几何重训代价高且不解决特征污染问题

---

## 四、聚合审计补充数据

来自 `aggregation_summary.json`：

| 指标 | 值 |
|------|-----|
| 总 Gaussian 数 | 63,379 |
| 总 ID 数 | 311 |
| beta_valid_count | 634 |
| beta_valid_ratio | 1.00% |
| beta_mean | 1,723.5 |
| beta_median | 3.66 |
| beta P90 | 1,302.1 |
| beta P99 | 41,297.7 |
| beta_mass_total | 1,092,670 |
| beta_mass_kept_by_purity07 | 6,112.4（0.56%） |
| dominant_ratio_beta_weighted_mean | 0.172 |
| purity07_valid_count | 103 |
| id_entropy_mean | 1.888 |
| camera_entropy_mean | 0.044 |
| num_hit_ids_mean | 14.64 |
| num_hit_cameras_mean | 1.22 |
| total_views | 345 |
| total_dets_used | 4,008 |

**关键观察**：
- beta 分布极度偏斜（median=3.66 vs mean=1,723），少量 Gaussian 占据绝大部分 opacity mass
- purity07 mask 仅保留 0.56% 的 beta mass，说明高纯度 Gaussian 几乎都是低 opacity 的边缘点
- 每个 Gaussian 平均被 14.6 个 ID 和 1.2 个相机看到，进一步印证了 ID 混杂问题

---

## 五、结论

| 模式 | 优势 | 劣势 | 综合评价 |
|------|------|------|----------|
| **Recall** | 覆盖最广、gap/AUC 最高 | 特征坍缩（cosine=0.989）、camera bias 显著 | 覆盖有用但污染严重 |
| **Pure07** | 纯度最高、Rank 指标最好 | 覆盖极低（103 Gaussian）、大量 ID 不可见 | 纯度有用但覆盖不够 |
| **Balanced** | ID probe 最强、mAP 最高 | camera bias 最严重（0.78）、覆盖仍薄 | 身份维度最优但 camera 污染使其不可直接使用 |

**最终判定**：三者均未解决 2D→3D 聚合的根本问题——相机偏差对身份信号的压制。下一步应优先尝试 **common-component removal** 或 **foreground-aware aggregation**，而非在 recall/pure/balanced 之间继续调参。仅当证明 opacity 覆盖本身不足时，才考虑 person-aware densification 或几何重训。
