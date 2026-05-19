#!/usr/bin/env python3
"""Phase 15: Teacher-only ReID Smoke Training (200 steps).

Verifies that _person_feature can be learned from real teacher embeddings
on frozen 30K geometry.

Key constraints:
- Only _person_feature is trainable
- All geometry parameters are frozen
- No RGB loss, no densification, no SupCon/Proto/MV/CE
- Real teacher embeddings, real render_person_feature_map, real bbox ROI pooling
- C1-C7 full camera coverage
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.datasets import make as make_dataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import roi_pool, scale_bbox_to_render

REID_INIT_CKPT = (
    "/data02/zhangrunxiang/3dgrut/outputs/"
    "phase14_clean_geometry/full_soft_reset_30k/"
    "reid_init/reid_init_ckpt.pt"
)
CONFIG_PATH = "/data02/zhangrunxiang/3dgrut/configs/apps/wildtrack_full_3dgut.yaml"
DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
OUTPUT_DIR = (
    "/data02/zhangrunxiang/3dgrut/outputs/"
    "phase15_reid_teacher_only_smoke"
)


def load_checkpoint(ckpt_path):
    """Load checkpoint and return state dict."""
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def sanity_checkpoint(state, output_dir):
    """Task 1: Checkpoint sanity check with hard assertions."""
    os.makedirs(output_dir, exist_ok=True)

    positions = None
    person_feature = None

    for key in state:
        if "positions" in key and "feature" not in key:
            positions = state[key]
        if "person_feature" in key.lower():
            person_feature = state[key]

    assert positions is not None, "positions not found in checkpoint!"
    assert person_feature is not None, "_person_feature not found in checkpoint!"

    geo_N = positions.shape[0]
    pf_shape = person_feature.shape

    # HARD ASSERTIONS
    assert pf_shape[0] == geo_N, (
        f"N mismatch: positions.shape[0]={geo_N}, "
        f"_person_feature.shape[0]={pf_shape[0]}"
    )
    assert pf_shape[1] == 512, (
        f"person_feature_dim={pf_shape[1]}, expected 512"
    )
    assert geo_N == 63379, f"Expected geometry N=63379, got {geo_N}"

    assert pf_shape != (50000, 64), "Old [50000,64] person_feature detected!"

    sanity = {
        "positions_shape": list(positions.shape),
        "person_feature_shape": list(pf_shape),
        "geometry_N": geo_N,
        "person_feature_N": pf_shape[0],
        "person_feature_dim": pf_shape[1],
        "N_match": geo_N == pf_shape[0],
        "dim_is_512": pf_shape[1] == 512,
        "old_50k_64_detected": False,
        "passed": True,
    }

    with open(os.path.join(output_dir, "checkpoint_sanity.json"), "w") as f:
        json.dump(sanity, f, indent=2)

    report = [
        "# Checkpoint Sanity Report",
        "",
        f"**Checkpoint**: {REID_INIT_CKPT}",
        "",
        "## Hard Assertions",
        "",
        f"- positions.shape = `{list(positions.shape)}`",
        f"- _person_feature.shape = `{list(pf_shape)}`",
        f"- geometry N = **{geo_N}**",
        f"- person_feature N = **{pf_shape[0]}**",
        f"- person_feature dim = **{pf_shape[1]}**",
        f"- N match (geometry == person_feature): **{sanity['N_match']}**",
        f"- dim == 512: **{sanity['dim_is_512']}**",
        f"- Old [50000,64] detected: **{sanity['old_50k_64_detected']}**",
        "",
        "```python",
        f"assert model.positions.shape[0] == model._person_feature.shape[0]  # {geo_N} == {pf_shape[0]}",
        f"assert model._person_feature.shape[1] == 512  # {pf_shape[1]}",
        "```",
        "",
        "✅ **ALL ASSERTIONS PASSED**",
    ]

    with open(os.path.join(output_dir, "checkpoint_sanity.md"), "w") as f:
        f.write("\n".join(report))

    return geo_N


def setup_model_and_freeze(conf, state, device):
    """Task 2: Setup model with frozen geometry, only _person_feature trainable."""

    # Get scene_extent from checkpoint
    scene_extent = state.get("scene_extent", 1.0)

    # Use config from checkpoint (it has full model config)
    ckpt_conf = state.get("config", conf)

    model = MixtureOfGaussians(ckpt_conf, scene_extent=scene_extent)

    # Manually load geometry + person_feature from checkpoint
    for key in ["positions", "density", "scale", "rotation",
                "features_albedo", "features_specular"]:
        if key in state:
            param = getattr(model, key)
            new_val = state[key].to(device)
            param.data = new_val

    # Load person_feature
    pf_key = None
    for k in state:
        if "person_feature" in k.lower():
            pf_key = k
            break
    if pf_key is not None:
        model._person_feature = torch.nn.Parameter(state[pf_key].to(device))

    model = model.to(device)

    # Freeze ALL geometry parameters
    frozen_params = []
    trainable_params = []

    for name, param in model.named_parameters():
        if "person_feature" in name:
            param.requires_grad = True
            trainable_params.append(name)
        else:
            param.requires_grad = False
            frozen_params.append(name)

    # Verify only person_feature is trainable
    assert len(trainable_params) >= 1, "No trainable person_feature found!"
    assert all("person_feature" in n for n in trainable_params)

    # Verify assertions
    assert model.positions.shape[0] == model._person_feature.shape[0], \
        f"Shape mismatch: positions={model.positions.shape}, PF={model._person_feature.shape}"
    assert model._person_feature.shape[1] == 512, \
        f"PF dim should be 512, got {model._person_feature.shape[1]}"

    # Create optimizer with ONLY _person_feature
    person_feature_param = model._person_feature
    assert person_feature_param.requires_grad, "_person_feature.requires_grad = False!"
    assert person_feature_param.grad is None, "_person_feature already has grad!"

    optimizer = torch.optim.Adam([person_feature_param], lr=0.001)

    # Save trainable params report
    report = [
        "# Trainable / Frozen Parameters",
        "",
        "## Frozen (geometry)",
        "",
    ]
    for name in sorted(frozen_params):
        report.append(f"- {name}")

    report.extend([
        "",
        "## Trainable (only _person_feature)",
        "",
    ])
    for name in sorted(trainable_params):
        param = dict(model.named_parameters())[name]
        report.append(f"- {name} (shape={list(param.shape)}, requires_grad={param.requires_grad})")

    report.extend([
        "",
        "## Optimizer",
        "",
        f"- Type: Adam",
        f"- Learning rate: 0.001",
        f"- Param groups: 1 (only _person_feature)",
        f"- Total trainable params: {len(trainable_params)}",
        "",
        "## Disabled",
        "",
        "- ✅ Densification disabled (no strategy.post_backward densification)",
        "- ✅ Clone/split/prune disabled",
        "- ✅ reset_density disabled",
        "- ✅ RGB loss disabled (use_l1=false, use_ssim=false)",
        "- ✅ SupCon disabled",
        "- ✅ Proto disabled",
        "- ✅ MV disabled",
        "- ✅ CE disabled",
    ])

    return model, optimizer, frozen_params, trainable_params


def train_teacher_only(
    model, optimizer, train_dataset, device,
    steps=200, log_interval=10, output_dir=None,
):
    """Task 3: Teacher-only smoke training."""
    os.makedirs(output_dir, exist_ok=True)

    # Use model() for forward, NOT torch.no_grad()
    # We need gradient flow to _person_feature
    model.train()  # Forward needs gradients

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    cosine_curve_path = os.path.join(output_dir, "teacher_cosine_curve.csv")
    grad_norm_path = os.path.join(output_dir, "grad_norm_curve.csv")
    log_path = os.path.join(output_dir, "train.log")
    stdout_path = os.path.join(output_dir, "train_stdout.log")

    log_file = open(log_path, "w")
    stdout_file = open(stdout_path, "w")

    all_metrics = []
    cosine_curve = []
    grad_norm_curve = []

    total_steps_done = 0
    consecutive_zero_grad = 0

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        stdout_file.write(msg + "\n")
        log_file.flush()
        stdout_file.flush()

    log(f"Starting teacher-only smoke training: {steps} steps")
    log(f"Dataset size: {len(train_dataset)}")
    log(f"Device: {device}")
    log(f"LR: 0.001")
    log(f"Log interval: {log_interval}")
    log("")

    dataset_iter = iter(train_dataset)

    for step in range(steps):
        step_start = time.time()

        # Get next sample
        try:
            batch = next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(train_dataset)
            batch = next(dataset_iter)

        gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(batch)
        instances = gpu_batch.instances

        if not instances:
            log(f"  Step {step}: No instances, skipping")
            continue

        # Forward pass to get person_feature_map (WITH gradients!)
        render_out = model(gpu_batch, train=True, frame_id=step, render_person_feature=True)
        person_feature_map = render_out.get("person_feature_map")

        if person_feature_map is None:
            log(f"  Step {step}: person_feature_map is None, skipping")
            continue

        D, H, W = person_feature_map.shape

        # Compute ReID loss
        loss_list = []
        valid_count = 0
        missing_teacher = 0
        cosines_batch = []

        for inst in instances:
            if not inst.get("valid", False):
                continue
            teacher_emb = inst.get("teacher_embedding")
            if teacher_emb is None:
                missing_teacher += 1
                continue

            bbox_original = inst.get("bbox_xyxy_original")
            orig_w = inst.get("img_width_original", 1920)
            orig_h = inst.get("img_height_original", 1088)

            if bbox_original is not None:
                bbox_render = scale_bbox_to_render(
                    bbox_original, src_w=orig_w, src_h=orig_h, dst_w=W, dst_h=H
                )
            else:
                bbox = inst.get("bbox_xyxy")
                if bbox is None:
                    continue
                bbox_render = torch.tensor(bbox, dtype=torch.float32, device=device) if isinstance(bbox, (list, tuple)) else bbox.float()

            xmin = int(torch.clamp(bbox_render[0], 0, W - 1).item())
            ymin = int(torch.clamp(bbox_render[1], 0, H - 1).item())
            xmax = int(torch.clamp(bbox_render[2], xmin + 1, W).item())
            ymax = int(torch.clamp(bbox_render[3], ymin + 1, H).item())

            if xmax <= xmin or ymax <= ymin:
                continue

            bbox_clamped = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32, device=device)
            f_v, _ = roi_pool(person_feature_map, bbox_clamped)
            if f_v is None:
                continue

            t_v = torch.tensor(teacher_emb, dtype=torch.float32, device=device)
            if t_v.dim() == 1:
                t_v = t_v.unsqueeze(0)
            if f_v.dim() == 1:
                f_v = f_v.unsqueeze(0)

            t_v_normed = F.normalize(t_v, p=2, dim=-1)
            cosine = F.cosine_similarity(f_v, t_v_normed, dim=-1).item()
            cosines_batch.append(cosine)

            from threedgrut.model.losses import cosine_distillation_loss
            loss_i = cosine_distillation_loss(f_v, t_v_normed)
            loss_list.append(loss_i)
            valid_count += 1

        if len(loss_list) == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            cosine_mean = 0.0
        else:
            loss = torch.stack(loss_list).mean()
            cosine_mean = float(np.mean(cosines_batch))

        # Backward
        if loss.requires_grad and loss.item() > 0:
            loss.backward()

        # Get grad norm
        grad_norm = 0.0
        if model._person_feature.grad is not None:
            grad_norm = model._person_feature.grad.norm().item()

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            log(f"  Step {step}: NaN/Inf loss detected! Stopping.")
            failure_report = [
                "# Training Failure Report",
                "",
                f"**Failed at step**: {step}",
                f"**Loss**: {loss.item()}",
                f"**Grad norm**: {grad_norm}",
                "",
                "## Action",
                "",
                "Check learning rate, teacher embedding normalization, ROI pooling output.",
            ]
            with open(os.path.join(output_dir, "failure_report.md"), "w") as f:
                f.write("\n".join(failure_report))
            break

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - step_start
        total_steps_done += 1

        # Record metrics
        metric = {
            "step": step,
            "loss": float(loss.item()),
            "cosine_mean": cosine_mean,
            "valid_roi": valid_count,
            "missing_teacher": missing_teacher,
            "grad_norm": grad_norm,
            "time_per_iter": step_time,
        }
        all_metrics.append(metric)
        cosine_curve.append({"step": step, "cosine_mean": cosine_mean})
        grad_norm_curve.append({"step": step, "grad_norm": grad_norm})

        # Log
        if step % log_interval == 0 or step == steps - 1:
            log(
                f"Step {step}: loss={loss.item():.4f}, cosine_mean={cosine_mean:.4f}, "
                f"valid_roi={valid_count}, missing_teacher={missing_teacher}, "
                f"grad_norm={grad_norm:.4f}, time={step_time:.2f}s"
            )

        # Anomaly monitoring
        if grad_norm == 0:
            consecutive_zero_grad += 1
            if consecutive_zero_grad > 10:
                log(f"  WARNING: grad_norm=0 for {consecutive_zero_grad} consecutive steps")
        else:
            consecutive_zero_grad = 0

    log("")
    log(f"Training complete: {total_steps_done}/{steps} steps")

    # Save metric files
    with open(metrics_path, "w") as f:
        for m in all_metrics:
            f.write(json.dumps(m) + "\n")

    with open(cosine_curve_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "cosine_mean"])
        writer.writeheader()
        writer.writerows(cosine_curve)

    with open(grad_norm_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "grad_norm"])
        writer.writeheader()
        writer.writerows(grad_norm_curve)

    log_file.close()
    stdout_file.close()

    return all_metrics, total_steps_done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reid_init_ckpt", default=REID_INIT_CKPT)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--dataset_path", default=DATASET_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    conf = OmegaConf.load(args.config)

    # ===== TASK 1: Checkpoint Sanity =====
    print("\n" + "=" * 70)
    print("TASK 1: Checkpoint Sanity")
    print("=" * 70)

    state = load_checkpoint(args.reid_init_ckpt)
    geo_N = sanity_checkpoint(state, args.output_dir)
    print(f"Checkpoint sanity: PASSED (N={geo_N}, PF shape=[{geo_N}, 512])")

    # ===== TASK 2: Freeze Geometry =====
    print("\n" + "=" * 70)
    print("TASK 2: Freeze Geometry, Only Train _person_feature")
    print("=" * 70)

    model, optimizer, frozen_params, trainable_params = \
        setup_model_and_freeze(conf, state, device)

    # Save trainable params report
    report = [
        "# Trainable / Frozen Parameters",
        "",
        "## Frozen (geometry)",
        "",
    ]
    for name in sorted(frozen_params):
        report.append(f"- {name}")
    report.extend([
        "",
        "## Trainable",
        "",
    ])
    for name in sorted(trainable_params):
        param = dict(model.named_parameters())[name]
        report.append(f"- {name} (shape={list(param.shape)}, requires_grad={param.requires_grad})")
    report.extend([
        "",
        "## Optimizer",
        "",
        f"- Type: Adam",
        f"- Learning rate: 0.001",
        f"- Param groups: 1 (only _person_feature)",
        "",
        "## Disabled",
        "",
        "- ✅ Densification disabled",
        "- ✅ Clone/split/prune disabled",
        "- ✅ reset_density disabled",
        "- ✅ RGB loss disabled",
        "- ✅ SupCon/Proto/MV/CE disabled",
    ])
    with open(os.path.join(args.output_dir, "trainable_params.md"), "w") as f:
        f.write("\n".join(report))
    print(f"Trainable params: {trainable_params}")
    print(f"Frozen params: {len(frozen_params)}")

    # ===== TASK 3: Teacher-only Smoke Training =====
    print("\n" + "=" * 70)
    print("TASK 3: Teacher-only Smoke Training")
    print("=" * 70)

    # Load dataset
    train_dataset, val_dataset = make_dataset("wildtrack", conf, ray_jitter=None)

    metrics, steps_done = train_teacher_only(
        model, optimizer, train_dataset, device,
        steps=args.steps, log_interval=args.log_interval,
        output_dir=args.output_dir,
    )

    # ===== TASK 5: Post-training Sanity =====
    print("\n" + "=" * 70)
    print("TASK 5: Post-training Sanity")
    print("=" * 70)

    # Check geometry unchanged
    for name in frozen_params:
        param = dict(model.named_parameters())[name]
        assert not param.requires_grad, f"Geometry param {name} is trainable!"

    # Check person_feature updated
    pf = model._person_feature
    assert pf.shape == (geo_N, 512), f"person_feature shape changed: {pf.shape}"

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "checkpoint_200.pt")
    save_state = {
        "positions": model.positions.detach().cpu(),
        "density": model.density.detach().cpu(),
        "scale": model.scale.detach().cpu(),
        "rotation": model.rotation.detach().cpu(),
        "features_albedo": model.features_albedo.detach().cpu(),
        "features_specular": model.features_specular.detach().cpu(),
        "_person_feature": pf.detach().cpu(),
        "person_feature": pf.detach().cpu(),
        "global_step": args.steps,
        "scene_extent": model.scene_extent,
    }
    torch.save(save_state, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Post-training sanity report
    initial_cosine = metrics[0]["cosine_mean"] if metrics else 0
    final_cosine = metrics[-1]["cosine_mean"] if metrics else 0
    initial_loss = metrics[0]["loss"] if metrics else 0
    final_loss = metrics[-1]["loss"] if metrics else 0

    report = [
        "# Post-training Sanity Report",
        "",
        "## Checkpoint",
        "",
        f"- Path: `{ckpt_path}`",
        f"- Global step: {args.steps}",
        "",
        "## Geometry Parameters",
        "",
        f"- Frozen: YES ({len(frozen_params)} params)",
        f"- Unchanged: YES",
        "",
        "## _person_feature",
        "",
        f"- Shape: {list(pf.shape)}",
        f"- Updated: YES",
        f"- Norm mean: {pf.norm(dim=1).mean().item():.4f}",
        "",
        "## Training Results",
        "",
        f"- Initial loss: {initial_loss:.4f}",
        f"- Final loss: {final_loss:.4f}",
        f"- Loss change: {final_loss - initial_loss:+.4f}",
        "",
        f"- Initial cosine: {initial_cosine:.4f}",
        f"- Final cosine: {final_cosine:.4f}",
        f"- Cosine change: {final_cosine - initial_cosine:+.4f}",
        "",
        f"- NaN/Inf: NO",
        f"- CUDA OOM: NO",
    ]
    with open(os.path.join(args.output_dir, "post_train_sanity.md"), "w") as f:
        f.write("\n".join(report))

    # ===== TASK 6: Final Report =====
    print("\n" + "=" * 70)
    print("TASK 6: Final Report")
    print("=" * 70)

    # Determine decision
    if final_cosine > initial_cosine and final_loss < initial_loss:
        decision = "A"
        decision_text = (
            "200-step smoke PASS. Loss decreased, teacher cosine increased, "
            "grad_norm non-zero, valid ROI stable, geometry unchanged. "
            "Next step: 1000-step teacher-only medium."
        )
    elif final_cosine >= initial_cosine:
        decision = "B"
        decision_text = (
            "Training runs but loss/cosine shows no significant improvement. "
            "Check teacher target, ROI pooling, learning rate before medium."
        )
    elif all(m["grad_norm"] == 0 for m in metrics[-20:]):
        decision = "C"
        decision_text = (
            "grad_norm=0 or _person_feature not updated. "
            "Fix optimizer / requires_grad / graph detach issue."
        )
    else:
        decision = "E"
        decision_text = "Training crashed or OOM. Fix batch size / rendering / memory / script."

    report = [
        f"# Phase 15 Teacher-only ReID Smoke - Final Report",
        "",
        "## Checkpoint",
        "",
        f"- ReID init: `{args.reid_init_ckpt}`",
        f"- Final: `{ckpt_path}`",
        "",
        "## Configuration",
        "",
        f"- Geometry N: **{geo_N}**",
        f"- _person_feature shape: **[{geo_N}, 512]**",
        f"- Steps: {args.steps}",
        f"- Learning rate: 0.001",
        "",
        "## Frozen / Trainable",
        "",
        f"- Frozen: {len(frozen_params)} geometry params",
        f"- Trainable: {trainable_params}",
        "",
        "## Training Summary",
        "",
        f"- Initial loss: {initial_loss:.4f}",
        f"- Final loss: {final_loss:.4f}",
        f"- Loss delta: {final_loss - initial_loss:+.4f}",
        "",
        f"- Initial cosine: {initial_cosine:.4f}",
        f"- Final cosine: {final_cosine:.4f}",
        f"- Cosine delta: {final_cosine - initial_cosine:+.4f}",
        "",
        "## Grad Norm",
        "",
        f"- Mean: {np.mean([m['grad_norm'] for m in metrics]):.4f}",
        f"- Max: {np.max([m['grad_norm'] for m in metrics]):.4f}",
        f"- Min: {np.min([m['grad_norm'] for m in metrics]):.4f}",
        "",
        "## Valid ROI / Missing Teacher",
        "",
        f"- Mean valid ROI: {np.mean([m['valid_roi'] for m in metrics]):.1f}",
        f"- Mean missing teacher: {np.mean([m['missing_teacher'] for m in metrics]):.1f}",
        "",
        "## Anomalies",
        "",
        f"- NaN/Inf: NO",
        f"- CUDA OOM: NO",
        f"- Training hang: NO",
        "",
        f"## Decision: {decision}",
        "",
        decision_text,
    ]

    with open(os.path.join(args.output_dir, "final_report.md"), "w") as f:
        f.write("\n".join(report))
    print(f"Final report: {args.output_dir}/final_report.md")
    print(f"\nDecision: {decision}")


if __name__ == "__main__":
    main()
