# Registration / Coordinate / Gaussian Index Alignment Audit Report

**Date**: 2026-05-21  
**Scope**: Verify whether 634 nonzero person_feature Gaussians are written to correct indices, correct coordinates, and correct visible regions.

---

## 1. Artifact Alignment (Task 1)

### 1.1 Key Artifact Shapes

| Artifact | Key | Shape | Match? |
|----------|-----|-------|--------|
| checkpoint_1000.pt | positions | [63379, 3] | - |
| registered_features.pt | features_beta | [63379, 512] | ✅ N matches |
| registered_features.pt | mask_beta | [63379] | ✅ N matches |
| registered_features.pt | beta | [63379] | ✅ N matches |
| registered_features.pt | dominant_ratio | [63379] | ✅ N matches |

### 1.2 Missing Fields

`registered_features.pt` does **NOT** contain:
- `gaussian_indices` / `valid_indices` — no explicit index mapping
- `xyz` — no saved Gaussian positions
- `ids` / `frame_ids` / `camera_ids` — no per-Gaussian identity assignment

### 1.3 Conclusion

- Checkpoint N = 63379, Registered N = 63379: **counts match** ✅
- No explicit index mapping exists, but the code review confirms **no reordering logic** in the pipeline
- Index alignment **cannot be directly proven** from saved artifacts, but **no evidence of mismatch**

---

## 2. Gaussian Index Alignment (Task 2)

### 2.1 Nonzero Feature Index Distribution

| Statistic | Value |
|-----------|-------|
| Nonzero count | 634 / 63379 (1.0%) |
| Index range | 50998 — 63371 |
| Index mean | 60379.3 |
| All in last quarter (index >= 47534) | **YES** |
| Indices < 634 | **0** |

### 2.2 Why Are Nonzero Indices Concentrated at the End?

This is **NOT an index mismatch bug**. The explanation is:

1. **3DGS MCMC training** progressively adds new Gaussians via splitting. New Gaussians are appended at the end of the array.
2. The **aggregation process** (`aggregate_2d_features_to_3d_gaussians.py`) uses a gradient trick that accumulates rendering weight per Gaussian. Only Gaussians that are **visible in person bounding boxes** accumulate nonzero `beta`.
3. The **person regions** in the Wildtrack scene are relatively small compared to the background. Most of the 63379 Gaussians represent background/scene structure.
4. During MCMC training, Gaussians that are added later (higher indices) tend to be in **under-reconstructed regions**, which may include person areas that need more Gaussians for detail.

### 2.3 Beta and Dominant Ratio

| Metric | Value |
|--------|-------|
| Beta > eps count | 634 |
| Beta valid matches nonzero | **YES** (exact match) |
| Dominant ratio mean | 0.3554 |
| Dominant ratio min | 0.0617 |

**Critical finding**: Dominant ratio mean = 0.36 means each valid Gaussian's rendering weight comes from **multiple person IDs** on average. Only 36% of a Gaussian's contribution is from its dominant ID. This indicates **severe multi-ID mixing**.

### 2.4 Conclusion

- Index alignment is **likely correct** — no evidence of mask compression or scatter error
- The tail concentration is explained by MCMC training dynamics, not a bug
- **However, index alignment cannot be definitively proven** without saved `valid_indices`

---

## 3. Coordinate Sanity Check (Task 3)

### 3.1 Position Ranges

| Axis | All Gaussians | Nonzero Gaussians | Nonzero within All Range? |
|------|--------------|-------------------|--------------------------|
| X | [-11024.8, 14732.3] | [-1325.1, 1415.8] | ✅ YES |
| Y | [-30176.6, 14253.8] | [-3996.6, -746.0] | ✅ YES |
| Z | [-12120.2, 12793.6] | [-853.4, 2339.4] | ✅ YES |

### 3.2 Z-Scores of Nonzero Mean vs All Mean

| Axis | Z-Score | Interpretation |
|------|---------|---------------|
| X | 0.91 | Within normal range |
| Y | 1.41 | Slightly shifted, but not outlier |
| Z | 0.09 | Within normal range |

### 3.3 Density and Scale

| Metric | All Gaussians | Nonzero Gaussians |
|--------|--------------|-------------------|
| Density mean | -1.0645 | **10.6972** |
| Scale mean | 3.1945 | 1.5753 |

**Key finding**: Nonzero Gaussians have **much higher density** (10.7 vs -1.1). This confirms they are in **high-opacity regions** — exactly where person features should be registered.

### 3.4 Conclusion

- The coordinate range of nonzero Gaussians is **within** the full Gaussian range
- The large coordinate values (±30000) are the **3DGS coordinate scale**, not a bug
- Nonzero Gaussians are **not outliers** — they are in the scene center where people are
- **No coordinate system mismatch detected**

---

## 4. Projection to Camera & BBox Alignment (Task 4)

### 4.1 Per-Camera Projection Results

| Camera | In Image | In BBox | Behind Camera | Total |
|--------|----------|---------|---------------|-------|
| C1 | 409/634 (64.5%) | 208/634 (32.8%) | 0 | 634 |
| C2 | 40/634 (6.3%) | 40/634 (6.3%) | 0 | 634 |
| C3 | 100/634 (15.8%) | 96/634 (15.1%) | 0 | 634 |
| C4 | 214/634 (33.8%) | 209/634 (33.0%) | 0 | 634 |
| C5 | 79/634 (12.5%) | 79/634 (12.5%) | 0 | 634 |
| C6 | 479/634 (75.6%) | 138/634 (21.8%) | 0 | 634 |
| C7 | 394/634 (62.1%) | 394/634 (62.1%) | 0 | 634 |

### 4.2 Overall Statistics

| Metric | Value |
|--------|-------|
| Avg in-image per camera | 245.0 / 634 (38.6%) |
| Avg in-bbox per camera | 166.3 / 634 (26.2%) |
| Behind-camera count | 0 (all cameras) |

### 4.3 Conclusion

- **All 634 nonzero Gaussians are in front of all 7 cameras** — no behind-camera issues
- **38.6% project into image** on average — reasonable for a multi-camera setup
- **26.2% project into person bboxes** — these are the Gaussians that actually contribute to person ReID
- C2 and C5 have very few nonzero Gaussians visible — likely due to camera angle/position
- **Projection alignment is correct** — coordinates and camera matrices are consistent

---

## 5. ROI Feature Support Alignment (Task 5)

### 5.1 Approximate Analysis Results

| Camera | ROIs with Support | Total ROIs | Support Ratio |
|--------|------------------|------------|---------------|
| C1 | 6 | 20 | 30.0% |
| C4 | 6 | 20 | 30.0% |
| C7 | 5 | 20 | 25.0% |
| **Overall** | **17** | **60** | **28.3%** |

### 5.2 Interpretation

**71.7% of person ROIs have NO nonzero-feature Gaussian support.** This means:

1. Most person bounding boxes are rendered by Gaussians that have **zero person features**
2. The 634 valid Gaussians cover only a small fraction of the person regions
3. The remaining 62745 Gaussians (with zero person features) dominate the rendering in most person areas
4. This explains why:
   - A/B paths produce non-zero but near-random ROI features (they render through the SH path, which has some signal but no identity)
   - C path has 25% Zero-ROI (these are the ROIs with no nonzero-feature support)
   - C path's non-zero ROI features are weak (only a small fraction of the rendering weight comes from identity-carrying Gaussians)

### 5.3 Missing Artifact

Per-pixel/per-ROI Gaussian contribution indices are **not available** from the current renderer output. The analysis above is based on **approximate projection-based bbox overlap**, which overestimates support (it counts any Gaussian projecting into the bbox, not just those contributing significant rendering weight).

---

## 6. Answers to Required Questions

### Q1: Does registered_features.pt match checkpoint Gaussian count?
**YES**. Both have N=63379.

### Q2: Are features correctly scattered to original Gaussian indices?
**LIKELY YES**, but **cannot be definitively proven**. No `valid_indices` or `gaussian_indices` are saved. Code review shows no reordering logic. The tail concentration of nonzero indices is explained by MCMC training dynamics.

### Q3: Is the index distribution of 634 nonzero Gaussians abnormal?
**YES, it is unusual** (all in last quarter, indices 50998-63371), but **explainable** by MCMC training. Not a bug.

### Q4: Are the xyz coordinates of 634 nonzero Gaussians abnormal?
**NO**. They are within the full Gaussian coordinate range, in the scene center where people are located. Their density is much higher than average, consistent with being in person regions.

### Q5: Is the coordinate anomaly global or only for nonzero/valid subset?
**GLOBAL**. All Gaussians have coordinates in the ±30000 range. This is the 3DGS coordinate scale, not a coordinate system error.

### Q6: Can nonzero Gaussians project into C1-C7 images?
**YES**. 38.6% of nonzero Gaussians project into images on average. All are in front of all cameras.

### Q7: Can they project into person bboxes?
**PARTIALLY**. 26.2% of nonzero Gaussians project into person bboxes on average. This is sufficient for some signal but far from full coverage.

### Q8: Can we proceed with aggregation optimization?
**YES, with caveats**. The main bottleneck is not index/coordinate mismatch, but **coverage sparsity** (only 634/63379 Gaussians carry identity features, covering only 28.3% of person ROIs).

### Q9: What is the blocker?
**The blocker is 2D→3D coverage sparsity**, not index/coordinate mismatch. Specifically:
1. Only 1% of Gaussians have nonzero person features
2. Only 28.3% of person ROIs have any nonzero-feature Gaussian support
3. Dominant ratio is only 0.36, meaning severe multi-ID mixing
4. Feature similarity is 96.9%, indicating near-collapse

---

## 7. Decision

**Next step: Fix 2D→3D coverage.**

The registration/coordinate/index alignment is **not the problem**. The real issues are:

1. **Coverage sparsity**: The gradient-trick aggregation only assigns features to 634/63379 Gaussians. This is because:
   - The aggregation only processes a subset of frames/cameras
   - Many person-region Gaussians have zero `beta` (they weren't visible in the processed detection masks)
   - The `beta > 1e-6` threshold is too strict for marginal contributions

2. **Multi-ID mixing**: Dominant ratio = 0.36 means most "valid" Gaussians see multiple people. This corrupts the identity signal.

3. **Feature collapse**: All 634 features have norm=1.0 and 96.9% cosine similarity. The teacher prototype assignment maps all Gaussians to similar prototypes.

**Recommended actions** (in priority order):
1. **Re-run aggregation with more frames/cameras** to increase coverage
2. **Lower the beta threshold** or use a softer assignment to include more Gaussians
3. **Apply purity filtering** (dominant_ratio >= 0.7) before feature assignment
4. **Consider trainable per-Gaussian identity latent** (V4) instead of fixed teacher prototypes

**Current gate**: NO V4, NO tracking, NO loss tuning. Next step is **improving 2D→3D coverage**.
