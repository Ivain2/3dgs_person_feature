# tools/ Directory Structure Index

## Directory Structure

```
tools/
├── # === Reusable Scripts (Production-Ready) ===
├── train_reid_feature_distillation.py   # Main training script (Phase 10) - ReID feature distillation with ROI pooling
├── train_contrastive_reid.py            # Contrastive learning training (Phase 9)
├── train_prototype_reid.py              # Prototype-based training (Phase 8)
├── train_reid_baseline.py               # Baseline ReID training (Phase 6)
├── eval_teacher_student_model.py        # Evaluation script for teacher vs student comparison
├── eval_identity_features.py            # Identity feature evaluation
├── ablate_lambda_reid.py                # Lambda parameter ablation study
├── build_teacher_prototypes.py          # Build teacher prototype embeddings
│
├── # === Legacy Phase Scripts (Keep for Reference) ===
├── phase4_reid_training_verify.py       # Phase 4 verification script (legacy)
├── phase5_short_train_reid.py           # Phase 5 short training (legacy)
├── phase5_overfit_reid.py               # Phase 5 overfit training (legacy)
├── phase10_train.py                     # Original Phase 10 training (backup)
├── phase10_sanity_check.py              # Sanity check for Phase 10 (can be moved to debug if needed)
├── phase9_train.py                      # Original Phase 9 training (backup)
├── phase8_train_with_prototype.py       # Original Phase 8 training (backup)
├── phase7_teacher_vs_student_eval.py    # Original Phase 7 evaluation (backup)
├── phase6_train_reid.py                 # Original Phase 6 training (backup)
├── phase6_eval_identity_features.py     # Original Phase 6 evaluation (backup)
├── phase6_ablate_lambda.py              # Original Phase 6 ablation (backup)
├── build_reid_teacher_prototypes.py     # Original prototype builder (backup)
│
├── # === Experiment Logs (Backup Only) ===
└── experiment_logs/
    ├── phase4_5/    # Phase 4-5 training logs
    ├── phase6/      # Phase 6 training logs
    ├── phase7/      # Phase 7 teacher-student evaluation logs
    ├── phase8/      # Phase 8 prototype training logs
    ├── phase9/      # Phase 9 contrastive training logs
    └── phase10/     # Phase 10 feature distillation logs
        ├── phase10A_*    # Mean pooling experiments
        ├── phase10B_*    # Opacity pooling experiments
        └── phase10C_*    # TopK/Opacity pooling with various configurations
│
└── debug_scripts/   # One-time debug and sanity check scripts
    ├── sanity_check_*.py          # Various sanity checks
    ├── test_*.py                  # Gradient and render tests
    ├── verify_*.py                # Cache alignment verification
    ├── debug_*.py                 # Gradient flow debugging
    ├── diagnose_*.py              # Zero gradient diagnosis
    ├── warmup_*.py                # Warmup testing scripts
    └── build_reid_teacher_cache.py # Teacher cache builder
```

## File Categories

### Production Scripts (Use these)
| Script | Purpose |
|--------|---------|
| `train_reid_feature_distillation.py` | Main training script with ROI pooling options (mean, opacity, topk) |
| `eval_teacher_student_model.py` | Comprehensive evaluation with teacher-student comparison |
| `build_teacher_prototypes.py` | Generate prototype embeddings from teacher model |
| `train_prototype_reid.py` | Train with prototype-based contrastive loss |
| `train_contrastive_reid.py` | Train with standard contrastive loss |
| `train_reid_baseline.py` | Baseline ReID training without prototypes |
| `eval_identity_features.py` | Evaluate identity feature quality |
| `ablate_lambda_reid.py` | Study the effect of lambda_reid parameter |

### Debug Scripts (Reference Only)
These are one-time debugging scripts that were used to diagnose specific issues. Keep for reference but don't reuse without modification.

### Experiment Logs (Historical Data)
Organized by phase. Each phase contains:
- `*_log.json` - Training logs with metrics per iteration
- `*_eval.json` - Evaluation results with top1, top5, mAP, gap
- `*_eval.csv` - Per-instance evaluation details

## Usage Examples

### Train with default configuration
```bash
python tools/train_reid_feature_distillation.py
```

### Train with specific pooling method
```bash
python tools/train_reid_feature_distillation.py \
    --pooling topk_opacity \
    --topk_ratio 0.3 \
    --lambda_reid 0.05 \
    --person_feature_lr 1e-4 \
    --detach_opacity_weight
```

### Evaluate teacher vs student
```bash
python tools/eval_teacher_student_model.py \
    --checkpoint runs/phase10C_topk_detach_lam005_lr1e4_stable/latest.pth \
    --pooling topk_opacity \
    --topk_ratio 0.3
```

### Build teacher prototypes
```bash
python tools/build_teacher_prototypes.py \
    --cache_path /path/to/teacher_cache.pt \
    --output_path /path/to/prototypes.pt
```

## Key Parameters

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pooling` | `topk_opacity` | ROI pooling method (mean, opacity, topk_opacity) |
| `--topk_ratio` | `0.3` | Ratio of top-k opacity pixels to use |
| `--lambda_reid` | `0.05` | Weight for ReID loss |
| `--person_feature_lr` | `1e-4` | Learning rate for person features |
| `--detach_opacity_weight` | `True` | Detach opacity gradients to prevent geometry corruption |
| `--warmup_iters` | `1000` | Number of warmup iterations |
| `--train_iters` | `10000` | Number of training iterations |

### Stable Configuration (Recommended)
Based on experiments, the stable configuration is:
- `--pooling topk_opacity`
- `--topk_ratio 0.3`
- `--lambda_reid 0.05` (not 0.1 - higher values cause feature collapse)
- `--person_feature_lr 1e-4` (not 5e-4 - higher values cause feature collapse)
- `--detach_opacity_weight` (prevents geometry parameter corruption)

## Notes

1. **Feature Collapse Warning**: High lambda_reid (0.1) combined with high learning rate (5e-4) causes complete feature collapse (gap ≈ 0)
2. **Always use detach_opacity_weight**: This prevents ReID gradients from corrupting geometry parameters
3. **Best Result So Far**: Phase 10B with opacity pooling achieved student top1=0.205 (vs teacher 0.671)
4. **Next Step**: Consider implementing Phase 10D with Prototype-level InfoNCE for better feature separation
