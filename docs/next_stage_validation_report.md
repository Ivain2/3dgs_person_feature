# Next Stage Validation Report

**Date**: 2026-05-21
**Scope**: Validate eval protocol, 3DGUT feature path runtime, and whether 3D ROI features retain identity signal.
**Non-goals**: No V4 training, no tracking integration, no new experiment design.

---

## 1. Scope and Non-goals

- **Goal**: Confirm that (a) the cross-camera evaluation protocol is correct, (b) the new 3DGUT independent CUDA feature rendering path is mathematically and runtime-correct, and (c) the current 3D ROI features still carry discriminative identity signal.
- **Non-goals**:
  - No V4 large-scale training (trainable per-Gaussian identity latent).
  - No tracking system integration.
  - No new experiment design or protocol changes.
- **Renderer scope**: Only 3DGUT (tile-based rasterization) is validated. 3DGRT / OptiX path is noted separately.

---

## 2. Eval Protocol Status

### 2.1 Camera ID Handling

**Status: PASS**

Audit output: [outputs/eval_protocol_check/camera_id_audit.json](file:///data02/zhangrunxiang/3dgrut/outputs/eval_protocol_check/camera_id_audit.json)

- **num_samples**: 42,275
- **num_ids**: 311
- **num_cameras**: 7
- **camera_id_unique**: C1, C2, C3, C4, C5, C6, C7
- **unknown_camera_count**: 0
- **empty_camera_count**: 0
- **Format**: All camera IDs use the "C{N}" format consistently across V2 detections, 3D ReID evaluation, and probe training.
- Per-camera sample counts range from 2,231 (C4) to 9,336 (C6).

### 2.2 Cross-Camera Exclusion

**Status: PASS**

Audit output: [outputs/eval_protocol_check/pair_construction_audit.json](file:///data02/zhangrunxiang/3dgrut/outputs/eval_protocol_check/pair_construction_audit.json)

- **same_camera_excluded**: true -- gallery indices are filtered to exclude same-camera matches in all evaluation scripts.
- **cross_camera_positive_pairs**: 60,115
- **cross_camera_negative_pairs**: 120,230
- **same_camera pairs**: excluded (both positive and negative).
- **per_camera_pair_nonempty_count**: 21 unique undirected camera pairs (7 choose 2).
- All per-camera-pair metrics are non-empty.

### 2.3 Required Audit JSON Status

| File | Path | Status |
|------|------|--------|
| camera_id_audit.json | [outputs/eval_protocol_check/](file:///data02/zhangrunxiang/3dgrut/outputs/eval_protocol_check/camera_id_audit.json) | Generated in this session |
| pair_construction_audit.json | [outputs/eval_protocol_check/](file:///data02/zhangrunxiang/3dgrut/outputs/eval_protocol_check/pair_construction_audit.json) | Generated in this session |
| v2_reproduction_metrics.json | [outputs/eval_protocol_check/](file:///data02/zhangrunxiang/3dgrut/outputs/eval_protocol_check/v2_reproduction_metrics.json) | Marked "not_rerun" (see below) |
| final_report.md | [outputs/eval_protocol_check/](file:///data02/zhangrunxiang/3dgrut/outputs/eval_protocol_check/final_report.md) | Generated below |

### 2.4 V2 Baseline Reproduction

**Status: not_rerun**

Audit output: [outputs/eval_protocol_check/v2_reproduction_metrics.json](file:///data02/zhangrunxiang/3dgrut/outputs/eval_protocol_check/v2_reproduction_metrics.json)

V2 baseline is previously established, but fresh reproduction metrics should be stored separately.

| Metric | Value |
|--------|-------|
| **mAP** | 0.4595 |
| **Rank-1** | 0.7087 |
| **ROC-AUC** | 0.895 |

These are reference values from the prior V2.1 run (`outputs/v2_1_2d_reid_full/`). `evaluate_2d_reid_baseline.py` was not rerun in this validation session.

---

## 3. Feature Path Runtime Status

### 3.1 3DGUT Feature Path Linearity

**Status: PASS**

Test output: [outputs/feature_path_linearity_check/](file:///data02/zhangrunxiang/3dgrut/outputs/feature_path_linearity_check/final_report.md)

Test configuration: feature_dim=8, 200 Gaussians, 64x64 image, render_method=3dgut, use_feature_path=True.

| Test | Description | Result | Key Metric |
|------|-------------|--------|------------|
| **T1 zero** | render(0) ≈ 0 | **PASS** | max_abs=0.00e+00 |
| **T2 scale** | render(2f) ≈ 2*render(f) | **PASS** | rel_err=0.00e+00 |
| **T3 additivity** | render(f1+f2) ≈ render(f1)+render(f2) | **PASS** | rel_err=4.25e-07 |
| **T4 signed** | Negative feature values preserved (no clamp) | **PASS** | neg_ratio=0.5149 |

T5 (old-vs-new) and T6 (opacity relation) are FAIL, which is **expected**: the old SH path includes non-linear operations (SH offset, opacity clamp, view-dependency) that differ from the pure linear alpha blending of the new feature path.

### 3.2 3DGUT Feature Path Gradient

**Status: PASS**

Test output: [outputs/feature_path_gradient_check/](file:///data02/zhangrunxiang/3dgrut/outputs/feature_path_gradient_check/final_report.md)

Test configuration: feature_dim=3, 10 Gaussians, 8x8 image, loss=sum.

| Check | Description | Result | Key Metric |
|-------|-------------|--------|------------|
| **Check 1** | Autograd gradient non-zero | **PASS** | gradient norm=3.81e+01 |
| **Check 2** | Finite-difference vs autograd | **PASS** | rel_error_max=**6.08e-05** (threshold: 1e-3) |
| **Check 3** | Geometry/radiance gradient decoupling | **PASS** | max_geo_grad=0.00e+00 |
| **Check 4** | Train/eval forward consistency | **PASS** | feat_diff=0.00e+00 |

Check 2's relative error (6.08e-05) is well below the 1e-3 threshold, confirming the CUDA backward kernel implementation is numerically correct. Check 3 confirms that feature gradients do not leak into geometry/density/radiance parameters.

### 3.3 3DGRT / OptiX-Specific Test

**Status: BLOCKED_BY_ENVIRONMENT**

3DGRT / OptiX tracer-specific validation is blocked by the current OptiX runtime / driver / container / extension initialization path. This should not be interpreted as a general conclusion that A100 cannot run OptiX code.

This validation session only targets the 3DGUT tile-based feature path, which remains fully validated per Sections 3.1 and 3.2.

---

## 4. Three-Way ReID Comparison

### 4.1 Configuration

| Path | Flag | Description |
|------|------|-------------|
| **A** | use_feature_path=False, linearize_feature=False | Old raw SH path |
| **B** | use_feature_path=False, linearize_feature=True | Old SH path + linearize |
| **C** | use_feature_path=True, linearize_feature=False | New CUDA feature path |

Model: `phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt`
Gaussians: 63,379 total, 634 valid, feature_dim=512

Output: [outputs/v3_feature_path_compare/](file:///data02/zhangrunxiang/3dgrut/outputs/v3_feature_path_compare/)

### 4.2 Results

| Metric | A (old raw SH) | B (old SH + linearize) | C (new feature path) |
|--------|----------------|------------------------|---------------------|
| Features extracted | 6,239 | 6,239 | 6,239 |
| **Zero-ROI** | 0 (0%) | 0 (0%) | **6,239 (100%)** |
| **mAP** | 0.0330 | 0.0338 | 0.0201 |
| **Rank-1** | 0.0370 | 0.0473 | 0.0196 |
| **Rank-5** | 0.0960 | 0.0939 | 0.0835 |
| **Pos-Neg Gap** | -0.0000 | -0.0000 | 0.0000 |
| **ROC-AUC** | 0.4999 | 0.5006 | 0.5000 |
| Best-F1 | 0.0903 | 0.0906 | 0.1320 |

Cross-camera positive pairs: 60,115. Cross-camera negative pairs: 120,230.

---

## 5. Interpretation

### 5.1 A/B Paths: Non-Zero but Near-Random

Paths A and B successfully extract non-zero ROI features (6,239 features, 0% Zero-ROI). However, the ReID metrics are near-random:

- mAP ≈ 0.033 (random baseline for 311 IDs would be ~1/311 ≈ 0.003, but a meaningful system should achieve much higher)
- ROC-AUC ≈ 0.50 (random coin flip)
- Pos-Neg Gap ≈ 0.0000 (positive and negative pairs are indistinguishable)

**Conclusion**: The naive 2D→3D→2D ReID feature pipeline produces features that are non-zero but lack discriminative identity signal. The 3D identity feature is weak/collapsed. This is a **ReID quality problem**, not a rendering pipeline problem.

### 5.2 C Path: 100% Zero-ROI

Path C (new feature path) produces **6,239 out of 6,239 features as all-zero vectors (100% Zero-ROI)**. This is fundamentally different from the A/B near-random result.

**C path 100% Zero-ROI is a real-model integration failure of the new feature path until proven otherwise.** Since A/B paths produce non-zero ROI features while C produces all-zero ROI features, the immediate blocker is likely in:

- Feature buffer transfer (`dptrFeatureParameters` → `particlesPrecomputedFeatures`)
- Tensor layout / stride mismatch between the 512-dim person feature tensor and the renderer's expected input
- Feature dimension handling (the kernel was instantiated with `kMaxFeatureDim=128`, later fixed to 1024)
- Valid-mask / index mapping (only 634 of 63,379 Gaussians are marked valid)
- Real-model feature-path integration in the render_person_feature_map call chain

**Do not conflate C's zero output with ordinary ReID collapse.** C producing zeros is a separate integration blocker that prevents any ReID evaluation from being meaningful for the new path.

### 5.3 Key Distinction

| Issue | Symptom | Implication |
|-------|---------|-------------|
| A/B near-random | Non-zero features, mAP≈0.03, ROC-AUC≈0.50 | Identity signal is weak/collapsed in 3D aggregation; ReID quality problem |
| C 100% zero | All features are exactly zero vectors | New feature path has an integration blocker; rendering pipeline problem |

These are **two distinct problems** that require different debugging approaches.

---

## 6. V4 Readiness

**Decision: NO -- Do not proceed to V4.**

Current blockers:

1. **C new feature path produces 100% Zero-ROI in real-model evaluation.** The path works correctly in synthetic unit tests (T1-T4, gradient checks) but returns all zeros when rendering from the trained 3DGS model with 512-dim person features.
2. **Frozen Probe has no diagnostic value on all-zero features.** Running linear/MLP probes on zero vectors would only confirm the collapse and provide no signal about identity retention.
3. **The real-model integration issue must be resolved first.** Before any V4 training (trainable per-Gaussian identity latent + ROI MLP head), we must confirm that the new feature path produces non-zero, non-degenerate features on real models.
4. **Tracking integration is not a current-stage goal.** Per the scope definition, tracking system integration is explicitly excluded from this validation phase.

---

## 7. Immediate Blocker

**The new feature path synthetic tests pass, but real-model integration fails with 100% Zero-ROI.**

This is the single gating issue. Until the new feature path produces non-zero ROI features from a real trained 3DGS model, no downstream evaluation (ReID metrics, frozen probe, V4 training) can be meaningfully performed.

The likely investigation areas (in priority order):

1. Verify `particlesPrecomputedFeatures` buffer contents at the point of feature kernel launch (is it zeroed or does it contain the 512-dim person features?)
2. Compare the feature buffer flow between old path (which produces non-zero output) and new path
3. Check the valid-mask / Gaussian index mapping in the feature rendering kernel
4. Audit the person feature norm distribution across the 634 "valid" Gaussians

---

## 8. Summary

| Validation | Status | Notes |
|------------|--------|-------|
| Camera ID / cross-camera eval | **PASS** | C1-C7, 0 unknown, 21 camera pairs non-empty |
| V2 baseline reproduction | **not_rerun** | Reference: mAP=0.4595, R1=0.7087, ROC-AUC=0.895 |
| 3DGUT linearity (T1-T4) | **PASS** | Synthetic feature path is mathematically linear |
| 3DGUT gradient (Check 1-4) | **PASS** | rel_error_max=6.08e-05; geometry decoupled |
| 3DGRT / OptiX-specific test | **BLOCKED_BY_ENVIRONMENT** | OptiX runtime/driver/extension init path blocked |
| A: old raw SH | **Non-zero, near-random** | mAP=0.0330, ROC-AUC=0.4999, Zero-ROI=0% |
| B: old SH + linearize | **Non-zero, near-random** | mAP=0.0338, ROC-AUC=0.5006, Zero-ROI=0% |
| C: new feature path | **100% Zero-ROI** | All 6,239 features are zero vectors |
| Frozen Probe | **SKIPPED** | No signal on zero features |
| V4 readiness | **NO** | Blocked by real-model Zero-ROI |

---

## 9. Files Modified in This Validation

| File | Change |
|------|--------|
| `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh` | Added feature forward/backward kernels; fixed `kMaxFeatureDim` 128→1024 |
| `threedgut_tracer/src/gutRenderer.cu` | Modified `renderFeatureForward/Backward` to use independent kernels; added person feature buffer re-copy after projectOnTiles |
| `tools/diagnose_feature_path_gradient.py` | Added `intrinsics` to synthetic batch; fixed in-place tensor modification in finite-difference loop |
| `tools/diagnose_feature_renderer_linearity.py` | Added 3DGUT render method configuration support |
| `tools/evaluate_3d_reid_aggregated_features.py` | Added `selected_frames` fallback inference from V2 detections when not in aggregated features |

## 10. Audit Files Generated

| File | Path |
|------|------|
| camera_id_audit.json | [outputs/eval_protocol_check/camera_id_audit.json](file:///data02/zhangrunxiang/3dgrut/outputs/eval_protocol_check/camera_id_audit.json) |
| pair_construction_audit.json | [outputs/eval_protocol_check/pair_construction_audit.json](file:///data02/zhangrunxiang/3dgrut/outputs/eval_protocol_check/pair_construction_audit.json) |
| v2_reproduction_metrics.json | [outputs/eval_protocol_check/v2_reproduction_metrics.json](file:///data02/zhangrunxiang/3dgrut/outputs/eval_protocol_check/v2_reproduction_metrics.json) |
