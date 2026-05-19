#!/usr/bin/env python3
"""Phase 15-B: Identity Eval Diagnostic

Diagnose why same/diff identity cosine are all ~1.0.
Check for:
- ROI pooling producing constant features
- Feature collapse
- Pair construction bugs
- Duplicate features
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.datasets import make as make_dataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import roi_pool, scale_bbox_to_render
from omegaconf import OmegaConf

CKPT_PATH = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt"
REID_INIT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase14_clean_geometry/full_soft_reset_30k/reid_init/reid_init_ckpt.pt"
DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
OUTPUT_DIR = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000"
DEVICE = "cuda"

def main():
    os.environ["TORCH_EXTENSIONS_DIR"] = "/data02/zhangrunxiang/.cache/torch_extensions/py311_cu118"
    device = torch.device(DEVICE)

    # Load config from reid_init checkpoint (has full config)
    reid_state = torch.load(REID_INIT_CKPT, map_location="cpu", weights_only=False)
    conf = reid_state.get("config", None)
    conf.model.person_feature_dim = 512

    # Load checkpoint (just weights, no config)
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    scene_extent = reid_state.get("scene_extent", 1.0)

    model = MixtureOfGaussians(conf, scene_extent=scene_extent)
    for key in ["positions", "density", "scale", "rotation", "features_albedo", "features_specular"]:
        if key in state:
            getattr(model, key).data = state[key].to(device)
    model._person_feature = torch.nn.Parameter(state["_person_feature"].to(device))
    model = model.to(device)
    model.eval()

    # Load dataset
    train_ds, _ = make_dataset("wildtrack", conf, ray_jitter=None)
    print(f"Dataset size: {len(train_ds)}")

    # Collect pooled features from 200 samples
    all_entries = []
    with torch.no_grad():
        for idx in range(min(len(train_ds), 200)):
            batch = train_ds[idx]
            gpu_batch = train_ds.get_gpu_batch_with_intrinsics(batch)
            gpu_instances = gpu_batch.instances
            if not gpu_instances:
                continue

            pf_map = model(gpu_batch, train=False, frame_id=0, render_person_feature=True).get("person_feature_map")
            if pf_map is None:
                continue
            _, h, w = pf_map.shape

            for inst in gpu_instances:
                if not inst.get("valid", False):
                    continue
                train_id = inst.get("train_id", inst.get("person_id", None))
                cam_id = inst.get("camera_id", inst.get("cam_id", "unknown"))
                if train_id is None:
                    continue

                teacher_emb = inst.get("teacher_embedding")
                if teacher_emb is None:
                    continue

                bbox_orig = inst.get("bbox_xyxy_original")
                orig_w = inst.get("img_width_original", 1920)
                orig_h = inst.get("img_height_original", 1088)
                if bbox_orig is None:
                    continue

                bbox_r = scale_bbox_to_render(bbox_orig, src_w=orig_w, src_h=orig_h, dst_w=w, dst_h=h)
                x1 = int(torch.clamp(bbox_r[0], 0, w - 1).item())
                y1 = int(torch.clamp(bbox_r[1], 0, h - 1).item())
                x2 = int(torch.clamp(bbox_r[2], x1 + 1, w).item())
                y2 = int(torch.clamp(bbox_r[3], y1 + 1, h).item())
                if x2 <= x1 or y2 <= y1:
                    continue

                bbox_c = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device)
                f_v, _ = roi_pool(pf_map, bbox_c)
                if f_v is None:
                    continue

                f_v_cpu = f_v.detach().cpu().squeeze(0)
                t_v_cpu = torch.tensor(teacher_emb, dtype=torch.float32)
                if t_v_cpu.dim() == 1:
                    t_v_cpu = t_v_cpu.unsqueeze(0)
                t_v_normed = F.normalize(t_v_cpu, p=2, dim=-1)
                f_v_normed = F.normalize(f_v_cpu, p=2, dim=-1)
                cosine = F.cosine_similarity(f_v_normed, t_v_normed, dim=-1).item()

                entry = {
                    "sample_idx": idx,
                    "train_id": int(train_id),
                    "cam_id": cam_id,
                    "feature": f_v_cpu,
                    "feature_norm": f_v_cpu.norm().item(),
                    "teacher_cosine": cosine,
                    "bbox_w": x2 - x1,
                    "bbox_h": y2 - y1,
                }
                all_entries.append(entry)

    print(f"Collected {len(all_entries)} valid entries")

    if len(all_entries) < 20:
        print("Too few entries, cannot do diagnostic")
        return

    # Feature norm stats
    norms = [e["feature_norm"] for e in all_entries]
    print(f"\nFeature norm: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}, min={np.min(norms):.4f}, max={np.max(norms):.4f}")

    # Pairwise cosine analysis
    features = torch.stack([e["feature"] for e in all_entries])
    features_normed = F.normalize(features, p=2, dim=-1)

    # Self cosine (should be 1.0)
    self_cosines = [F.cosine_similarity(features_normed[i:i+1], features_normed[i:i+1]).item() for i in range(min(10, len(features_normed)))]
    print(f"Self cosines (first 10): {self_cosines}")

    # Non-self pairwise cosine
    n = min(200, len(features_normed))
    cos_matrix = (features_normed[:n] @ features_normed[:n].T).numpy()
    np.fill_diagonal(cos_matrix, np.nan)  # exclude self
    valid_cosines = cos_matrix[~np.isnan(cos_matrix)]
    print(f"\nPairwise cosine (non-self, {n} samples):")
    print(f"  mean={np.nanmean(valid_cosines):.4f}")
    print(f"  std={np.nanstd(valid_cosines):.4f}")
    print(f"  min={np.nanmin(valid_cosines):.4f}")
    print(f"  max={np.nanmax(valid_cosines):.4f}")
    print(f"  median={np.nanmedian(valid_cosines):.4f}")

    # Check if features are nearly identical
    dup_ratio = np.sum(valid_cosines > 0.999) / len(valid_cosines)
    print(f"  Duplicate ratio (>0.999): {dup_ratio:.4f}")

    # Same/diff identity cosine
    id_to_entries = defaultdict(list)
    for e in all_entries:
        id_to_entries[e["train_id"]].append(e)

    same_cosines = []
    diff_cosines = []
    cross_cam_same = []
    cross_cam_diff = []

    ids = list(id_to_entries.keys())
    for i, id_a in enumerate(ids):
        entries_a = id_to_entries[id_a]
        # Same-id pairs (non-self)
        for idx_e in range(1, len(entries_a)):
            f_a = F.normalize(entries_a[0]["feature"], p=2, dim=-1)
            f_b = F.normalize(entries_a[idx_e]["feature"], p=2, dim=-1)
            cos = F.cosine_similarity(f_a.unsqueeze(0), f_b.unsqueeze(0)).item()
            same_cosines.append(cos)
            if entries_a[0]["cam_id"] != entries_a[idx_e]["cam_id"]:
                cross_cam_same.append(cos)

        # Diff-id pairs
        for j, id_b in enumerate(ids):
            if j <= i:
                continue
            entries_b = id_to_entries[id_b]
            for ea in entries_a:
                for eb in entries_b:
                    f_a = F.normalize(ea["feature"], p=2, dim=-1)
                    f_b = F.normalize(eb["feature"], p=2, dim=-1)
                    cos = F.cosine_similarity(f_a.unsqueeze(0), f_b.unsqueeze(0)).item()
                    diff_cosines.append(cos)
                    if ea["cam_id"] != eb["cam_id"]:
                        cross_cam_diff.append(cos)

    same_mean = np.mean(same_cosines) if same_cosines else 0.0
    diff_mean = np.mean(diff_cosines) if diff_cosines else 0.0
    cross_same_mean = np.mean(cross_cam_same) if cross_cam_same else 0.0
    cross_diff_mean = np.mean(cross_cam_diff) if cross_cam_diff else 0.0

    print(f"\nIdentity gap analysis:")
    print(f"  Same-id cosine mean: {same_mean:.4f} (pairs={len(same_cosines)})")
    print(f"  Diff-id cosine mean: {diff_mean:.4f} (pairs={len(diff_cosines)})")
    print(f"  Same/diff gap: {same_mean - diff_mean:.4f}")
    print(f"  Cross-camera same-id: {cross_same_mean:.4f} (pairs={len(cross_cam_same)})")
    print(f"  Cross-camera diff-id: {cross_diff_mean:.4f} (pairs={len(cross_cam_diff)})")
    print(f"  Cross-camera gap: {cross_same_mean - cross_diff_mean:.4f}")

    # Per-camera cosine to teacher
    per_cam = defaultdict(list)
    for e in all_entries:
        per_cam[e["cam_id"]].append(e["teacher_cosine"])
    print(f"\nPer-camera cosine to teacher:")
    for cam in sorted(per_cam.keys()):
        vals = per_cam[cam]
        print(f"  {cam}: mean={np.mean(vals):.4f}, count={len(vals)}")

    # Save diagnostic report
    report = (
        f"# Identity Eval Diagnostic\n\n"
        f"## Feature Statistics\n\n"
        f"- Valid entries: {len(all_entries)}\n"
        f"- Feature norm: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}, min={np.min(norms):.4f}, max={np.max(norms):.4f}\n\n"
        f"## Pairwise Cosine (non-self, {n} samples)\n\n"
        f"- Mean: {np.nanmean(valid_cosines):.4f}\n"
        f"- Std: {np.nanstd(valid_cosines):.4f}\n"
        f"- Min: {np.nanmin(valid_cosines):.4f}\n"
        f"- Max: {np.nanmax(valid_cosines):.4f}\n"
        f"- Median: {np.nanmedian(valid_cosines):.4f}\n"
        f"- Duplicate ratio (>0.999): {dup_ratio:.4f}\n\n"
        f"## Identity Gap\n\n"
        f"- Same-id cosine: {same_mean:.4f} ({len(same_cosines)} pairs)\n"
        f"- Diff-id cosine: {diff_mean:.4f} ({len(diff_cosines)} pairs)\n"
        f"- Same/diff gap: {same_mean - diff_mean:.4f}\n\n"
        f"## Cross-camera Gap\n\n"
        f"- Cross-camera same-id: {cross_same_mean:.4f} ({len(cross_cam_same)} pairs)\n"
        f"- Cross-camera diff-id: {cross_diff_mean:.4f} ({len(cross_cam_diff)} pairs)\n"
        f"- Cross-camera gap: {cross_same_mean - cross_diff_mean:.4f}\n\n"
    )

    # Judgment
    if dup_ratio > 0.5:
        report += "## Verdict: FEATURE COLLAPSE\n\n"
        report += "> Non-self pairwise cosine mostly > 0.999, indicating pooled features have collapsed to nearly identical vectors.\n"
        report += "> This means ROI pooling is producing constant/very similar features regardless of input.\n"
        report += "> Possible causes: person_feature render blending homogenizes features, or all Gaussians have similar _person_feature.\n"
    elif np.nanmean(valid_cosines) > 0.95 and dup_ratio > 0.2:
        report += "## Verdict: NEAR-COLLAPSE / INSUFFICIENT DIVERSITY\n\n"
        report += "> Non-self pairwise cosine mostly > 0.95 with significant duplicate ratio.\n"
        report += "> Features have very low diversity, but not fully collapsed.\n"
        report += "> Teacher-only loss is improving cosine to teacher, but not creating identity separation.\n"
        report += "> CE loss may help by explicitly pushing different IDs apart.\n"
    elif np.nanmean(valid_cosines) < 0.7:
        report += "## Verdict: HEALTHY FEATURE DIVERSITY\n\n"
        report += "> Pairwise cosine distribution looks reasonable.\n"
        report += "> Same/diff gap should be meaningful.\n"
    else:
        report += "## Verdict: MODERATE DIVERSITY\n\n"
        report += f"> Pairwise cosine mean={np.nanmean(valid_cosines):.4f}, std={np.nanstd(valid_cosines):.4f}.\n"
        report += "> Some diversity exists, but same/diff gap may be small.\n"
        report += "> CE loss should help improve separation.\n"

    report += "\n## Recommendation\n\n"
    if same_mean - diff_mean < 0.01:
        report += "- Same/diff gap is too small to use as metric at this stage\n"
        report += "- Teacher-only loss improves alignment but not discrimination\n"
        report += "- CE small overfit is the appropriate next step to test identity separation\n"

    with open(os.path.join(OUTPUT_DIR, "identity_eval_diagnostic.md"), "w") as f:
        f.write(report)

    # Save pairwise cosine stats
    stats = {
        "n_samples": n,
        "pairwise_mean": float(np.nanmean(valid_cosines)),
        "pairwise_std": float(np.nanstd(valid_cosines)),
        "pairwise_min": float(np.nanmin(valid_cosines)),
        "pairwise_max": float(np.nanmax(valid_cosines)),
        "pairwise_median": float(np.nanmedian(valid_cosines)),
        "dup_ratio": float(dup_ratio),
        "same_id_cosine_mean": float(same_mean),
        "diff_id_cosine_mean": float(diff_mean),
        "same_diff_gap": float(same_mean - diff_mean),
        "same_pair_count": len(same_cosines),
        "diff_pair_count": len(diff_cosines),
        "cross_cam_same_mean": float(cross_same_mean),
        "cross_cam_diff_mean": float(cross_diff_mean),
        "cross_cam_gap": float(cross_same_mean - cross_diff_mean),
    }
    with open(os.path.join(OUTPUT_DIR, "pairwise_cosine_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Save diagnostic CSV
    import csv
    with open(os.path.join(OUTPUT_DIR, "identity_eval_diagnostic.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "cam_id", "train_id", "feature_norm", "teacher_cosine", "bbox_w", "bbox_h"])
        for e in all_entries:
            w.writerow([e["sample_idx"], e["cam_id"], e["train_id"], e["feature_norm"], e["teacher_cosine"], e["bbox_w"], e["bbox_h"]])

    # Save fixed identity gap eval
    with open(os.path.join(OUTPUT_DIR, "identity_gap_eval_fixed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["same_id_cosine_mean", "diff_id_cosine_mean", "same_diff_gap",
                     "same_pair_count", "diff_pair_count"])
        w.writerow([same_mean, diff_mean, same_mean - diff_mean, len(same_cosines), len(diff_cosines)])

    with open(os.path.join(OUTPUT_DIR, "cross_camera_eval_fixed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cross_cam_same_mean", "cross_cam_diff_mean", "cross_cam_gap",
                     "cross_cam_same_count", "cross_cam_diff_count"])
        w.writerow([cross_same_mean, cross_diff_mean, cross_same_mean - cross_diff_mean,
                     len(cross_cam_same), len(cross_cam_diff)])

    print(f"\nDiagnostic saved to: {OUTPUT_DIR}/identity_eval_diagnostic.md")
    print(f"Verdict: {'FEATURE COLLAPSE' if dup_ratio > 0.5 else 'NEAR-COLLAPSE' if np.nanmean(valid_cosines) > 0.95 else 'HEALTHY'}")

if __name__ == "__main__":
    main()
