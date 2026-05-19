#!/usr/bin/env python3
"""V3.0.1: Learning-free 2D->3D ReID Feature Aggregation (LUDVIG-style).

Uses gradient trick to backproject 2D teacher embeddings into 3D Gaussian features.
V3.0.1 adds id_contrib / dominant_ratio for feature collapse diagnosis.
NO TRAINING. NO GEOMETRY CHANGES. NO TRACKER/BBOX SCALE CHANGES.

Coordinate protocol (frozen from V0):
  - annotation bbox: 1920x1080, padding 1080->1088, downsample=4
  - render size: WxH=480x272, scale_bbox_to_render: sx=sy=0.25
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.datasets.dataset_wildtrack import WildtrackDataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.roi_pooling import scale_bbox_to_render, _clamp_bbox

DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
REID_INIT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase14_clean_geometry/full_soft_reset_30k/reid_init/reid_init_ckpt.pt"
DEFAULT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt"
FEAT_DIM = 512
CHUNK_SIZE = 512


def setup_model(ckpt_path, device):
    reid_state = torch.load(REID_INIT_CKPT, map_location="cpu", weights_only=False)
    conf = reid_state.get("config")
    conf.model.person_feature_dim = FEAT_DIM
    scene_extent = reid_state.get("scene_extent", 1.0)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ["positions", "density", "scale", "rotation", "features_albedo", "features_specular"]:
        if key in state:
            getattr(model, key).data = state[key].to(device)
    pf_key = "_person_feature" if "_person_feature" in state else "person_feature"
    if pf_key in state:
        model._person_feature = torch.nn.Parameter(state[pf_key].to(device))
    else:
        model._person_feature = torch.nn.Parameter(
            torch.randn(model.positions.shape[0], FEAT_DIM, device=device) * 0.01)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, conf


def load_v2_data(detections_csv, features_npz):
    dets = []
    with open(detections_csv, "r") as f:
        for row in csv.DictReader(f):
            row["frame_id"] = int(row["frame_id"])
            row["person_id"] = int(row["person_id"])
            row["train_id"] = int(row["train_id"])
            row["border_flag"] = row["border_flag"] == "True"
            row["small_bbox"] = row["small_bbox"] == "True"
            bbox_str = row["bbox_xyxy_original"]
            row["bbox_xyxy_original"] = [int(x.strip()) for x in bbox_str.strip("[]").split(",")]
            dets.append(row)
    features = np.load(features_npz)["features"]
    for i, d in enumerate(dets):
        d["embedding"] = features[i].astype(np.float32)
    return dets


def build_frame_cam_index(dataset):
    mapping = {}
    for idx in range(len(dataset)):
        batch = dataset[idx]
        fid = batch.get("frame_idx", -1)
        cid = batch.get("camera_id", "unknown")
        mapping[(fid, cid)] = idx
    return mapping


def create_bbox_mask(bbox_render, H, W, shrink_ratio=0.0):
    xmin, ymin, xmax, ymax = [int(v) for v in bbox_render]
    if shrink_ratio > 0:
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        bw, bh = xmax - xmin, ymax - ymin
        sw, sh = bw * (1 - shrink_ratio), bh * (1 - shrink_ratio)
        xmin = max(0, int(cx - sw / 2))
        ymin = max(0, int(cy - sh / 2))
        xmax = min(W, int(cx + sw / 2))
        ymax = min(H, int(cy + sh / 2))
    mask = torch.zeros(H, W, dtype=torch.float32)
    if xmax > xmin and ymax > ymin:
        mask[max(0, ymin):min(H, ymax), max(0, xmin):min(W, xmax)] = 1.0
    return mask


def handle_overlaps(masks, H, W):
    overlap = torch.zeros(H, W, dtype=torch.float32)
    for m in masks:
        overlap = overlap + m
    overlap_mask = overlap > 1
    cleaned = []
    for m in masks:
        m = m.clone()
        m[overlap_mask] = 0.0
        cleaned.append(m)
    return cleaned


def build_id_mapping(v2_dets):
    train_ids = sorted(set(d["train_id"] for d in v2_dets))
    id_to_col = {tid: i for i, tid in enumerate(train_ids)}
    col_to_id = {i: tid for tid, i in id_to_col.items()}
    return train_ids, id_to_col, col_to_id


def aggregate_view_chunk(model, gpu_batch, masks_chunk, embeddings_chunk, train_ids_chunk, num_ids, id_to_col, device):
    K = len(masks_chunk)
    if K == 0:
        return None, None, None, 0

    M = torch.stack(masks_chunk).to(device)
    E = torch.tensor(np.stack(embeddings_chunk), device=device, dtype=torch.float32)
    train_id_cols = torch.tensor([id_to_col[tid] for tid in train_ids_chunk], device=device, dtype=torch.long)

    N = model.positions.shape[0]
    orig_pf = model._person_feature

    dummy = torch.nn.Parameter(torch.zeros(N, FEAT_DIM, device=device, dtype=torch.float32, requires_grad=True))
    model._person_feature = dummy

    try:
        with torch.enable_grad():
            pf_map, _ = model.render_person_feature_map(gpu_batch, train=False, frame_id=0)

        rendered_k = pf_map[:K]
        loss = (rendered_k * M).sum()
        loss.backward()

        A = dummy.grad[:, :K].clone()
        num_update = A @ E
        den_update = A.sum(dim=1)

        one_hot = torch.zeros(K, num_ids, device=device, dtype=torch.float32)
        one_hot.scatter_(1, train_id_cols.unsqueeze(1), 1.0)
        id_update = A @ one_hot
    except RuntimeError as e:
        print(f"  [WARN] Gradient trick failed: {e}")
        num_update = None
        den_update = None
        id_update = None
        K = 0
    finally:
        model._person_feature = orig_pf
        if dummy.grad is not None:
            dummy.grad = None
        del dummy
        torch.cuda.empty_cache()

    return num_update, den_update, id_update, K


def aggregate_view(model, gpu_batch, dets, H, W, device, mask_policy, shrink_ratio, overlap_policy, num_ids, id_to_col):
    K = len(dets)
    if K == 0:
        return None, None, None, 0

    masks = []
    embeddings = []
    train_ids = []
    for d in dets:
        bbox_orig = d["bbox_xyxy_original"]
        bbox_render = scale_bbox_to_render(bbox_orig, src_w=1920, src_h=1088, dst_w=W, dst_h=H)
        xmin, ymin, xmax, ymax = _clamp_bbox(bbox_render, H, W)
        shrink = shrink_ratio if mask_policy == "shrink" else 0.0
        mask = create_bbox_mask([xmin, ymin, xmax, ymax], H, W, shrink_ratio=shrink)
        masks.append(mask)
        embeddings.append(d["embedding"])
        train_ids.append(d["train_id"])

    if overlap_policy == "ignore":
        masks = handle_overlaps(masks, H, W)

    if K <= CHUNK_SIZE:
        return aggregate_view_chunk(model, gpu_batch, masks, embeddings, train_ids, num_ids, id_to_col, device)

    num_total = None
    den_total = None
    id_total = None
    total_used = 0

    for chunk_start in range(0, K, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, K)
        masks_c = masks[chunk_start:chunk_end]
        embeddings_c = embeddings[chunk_start:chunk_end]
        train_ids_c = train_ids[chunk_start:chunk_end]

        num_update, den_update, id_update, k_used = aggregate_view_chunk(
            model, gpu_batch, masks_c, embeddings_c, train_ids_c, num_ids, id_to_col, device)

        if num_update is not None:
            if num_total is None:
                num_total = num_update
                den_total = den_update
                id_total = id_update
            else:
                num_total += num_update
                den_total += den_update
                id_total += id_update
            total_used += k_used

    return num_total, den_total, id_total, total_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--v2_detections", default="outputs/v2_1_2d_reid_full/detections.csv")
    parser.add_argument("--v2_features", default="outputs/v2_1_2d_reid_full/features.npz")
    parser.add_argument("--mask_policy", default="full", choices=["full", "shrink"])
    parser.add_argument("--shrink_ratio", type=float, default=0.1)
    parser.add_argument("--overlap_policy", default="ignore", choices=["ignore", "keep"])
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--sample_mode", default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="outputs/v3_0_1_small_full")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print(f"V3.0.1-small: 2D->3D Aggregation (mask={args.mask_policy}, overlap={args.overlap_policy})")
    print("=" * 60)

    print("\n[1/5] Loading model...")
    model, conf = setup_model(args.ckpt, device)
    N = model.positions.shape[0]
    print(f"  N_gaussians={N}, _person_feature={model._person_feature.shape}")

    print("\n[2/5] Loading dataset and V2 data...")
    ds = WildtrackDataset(dataset_path=DATASET_PATH, downsample_factor=4, load_teacher_cache=False)
    v2_dets = load_v2_data(args.v2_detections, args.v2_features)
    print(f"  Dataset size: {len(ds)}, V2 detections: {len(v2_dets)}")

    print("\n[3/5] Building frame-camera index and ID mapping...")
    fc_index = build_frame_cam_index(ds)
    print(f"  Indexed {len(fc_index)} (frame,cam) pairs")

    train_ids, id_to_col, col_to_id = build_id_mapping(v2_dets)
    num_ids = len(train_ids)
    print(f"  Unique train_ids: {num_ids}")

    view_dets = defaultdict(list)
    for d in v2_dets:
        view_dets[(d["frame_id"], d["camera_id"])].append(d)

    all_frames = sorted(set(f for f, c in view_dets.keys()))
    if args.sample_mode == "random":
        rng = np.random.RandomState(args.seed)
        rng.shuffle(all_frames)
    selected_frames = all_frames[:args.max_frames]
    print(f"  Selected {len(selected_frames)} frames")

    print("\n[4/5] Aggregating features + id_contrib...")
    num = torch.zeros(N, FEAT_DIM, device=device, dtype=torch.float32)
    den = torch.zeros(N, device=device, dtype=torch.float32)
    id_contrib = torch.zeros(N, num_ids, device=device, dtype=torch.float32)
    view_stats = []
    cam_stats = defaultdict(lambda: {"views": 0, "dets": 0, "used": 0})
    total_views = 0
    total_dets_used = 0
    t0 = time.time()

    for frame_id in selected_frames:
        for cam_id in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            key = (frame_id, cam_id)
            if key not in view_dets or key not in fc_index:
                continue

            dets = view_dets[key]
            if len(dets) == 0:
                continue

            batch = ds[fc_index[key]]
            gpu_batch = ds.get_gpu_batch_with_intrinsics(batch)
            H, W = gpu_batch.rays_ori.shape[1], gpu_batch.rays_ori.shape[2]

            num_update, den_update, id_update, K_used = aggregate_view(
                model, gpu_batch, dets, H, W, device,
                args.mask_policy, args.shrink_ratio, args.overlap_policy, num_ids, id_to_col)

            if num_update is not None:
                num += num_update
                den += den_update
                id_contrib += id_update
                total_dets_used += K_used

            total_views += 1
            cam_stats[cam_id]["views"] += 1
            cam_stats[cam_id]["dets"] += len(dets)
            cam_stats[cam_id]["used"] += K_used
            view_stats.append({
                "frame_id": frame_id, "camera_id": cam_id,
                "num_detections": len(dets), "used_detections": K_used,
            })

            if total_views % 20 == 0:
                valid = (den > 0).sum().item()
                elapsed = time.time() - t0
                print(f"  Views: {total_views}, Dets: {total_dets_used}, "
                      f"Valid: {valid}/{N}, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Done: {total_views} views, {total_dets_used} dets, {elapsed:.0f}s")

    eps = 1e-6
    valid_mask = den > eps
    g_raw = torch.zeros(N, FEAT_DIM, dtype=torch.float32)
    g_raw[valid_mask] = (num[valid_mask] / den[valid_mask].unsqueeze(1)).cpu()
    g_l2 = F.normalize(g_raw, p=2, dim=1)

    dominant_weight = id_contrib.max(dim=1).values
    dominant_id_col = id_contrib.argmax(dim=1)
    dominant_ratio = dominant_weight / (den + eps)

    id_contrib_sum = id_contrib.sum(dim=1)
    consistency_err = (id_contrib_sum - den).abs()
    mean_err = consistency_err.mean().item()
    max_err = consistency_err.max().item()
    rel_err = (consistency_err / (den + eps)).mean().item()

    print(f"\n  [Sanity] id_contrib sum vs beta: mean_err={mean_err:.6f}, max_err={max_err:.6f}, rel_err={rel_err:.6f}")

    torch.save({
        "gaussian_features_raw": g_raw,
        "gaussian_features_l2": g_l2,
        "beta": den.cpu(),
        "valid_gaussian_mask": valid_mask.cpu(),
        "id_contrib": id_contrib.cpu(),
        "dominant_ratio": dominant_ratio.cpu(),
        "dominant_id_col": dominant_id_col.cpu(),
        "id_to_col": id_to_col,
        "col_to_id": col_to_id,
        "selected_frames": selected_frames,
        "config": {
            "mask_policy": args.mask_policy,
            "shrink_ratio": args.shrink_ratio,
            "overlap_policy": args.overlap_policy,
            "max_frames": args.max_frames,
            "feat_dim": FEAT_DIM,
            "ckpt": args.ckpt,
            "num_ids": num_ids,
        },
    }, os.path.join(args.out_dir, "aggregated_features.pt"))

    beta_np = den.cpu().numpy()
    valid_np = valid_mask.cpu().numpy()
    norm_raw = g_raw.norm(dim=1).numpy()
    dom_ratio_np = dominant_ratio.cpu().numpy()
    dom_ratio_valid = dom_ratio_np[valid_np] if valid_np.any() else np.array([0])

    purity07_mask = (den > eps) & (dominant_ratio >= 0.7)
    beta_mass_total = den.sum().item()
    beta_mass_kept_by_beta = den[valid_mask].sum().item()
    beta_mass_kept_by_purity = den[purity07_mask].sum().item()

    summary = {
        "num_gaussians": N,
        "num_ids": num_ids,
        "beta_valid_count": int(valid_mask.sum()),
        "beta_valid_ratio": float(valid_mask.float().mean()),
        "zero_beta_ratio": float((den == 0).float().mean()),
        "beta_mean": float(beta_np[valid_np].mean()) if valid_np.any() else 0,
        "beta_median": float(np.median(beta_np[valid_np])) if valid_np.any() else 0,
        "beta_p90": float(np.percentile(beta_np[valid_np], 90)) if valid_np.any() else 0,
        "beta_p99": float(np.percentile(beta_np[valid_np], 99)) if valid_np.any() else 0,
        "feature_norm_raw_mean": float(norm_raw[valid_np].mean()) if valid_np.any() else 0,
        "feature_norm_raw_median": float(np.median(norm_raw[valid_np])) if valid_np.any() else 0,
        "dominant_ratio_mean": float(dom_ratio_np.mean()),
        "dominant_ratio_median": float(np.median(dom_ratio_np)),
        "dominant_ratio_p10": float(np.percentile(dom_ratio_np, 10)),
        "dominant_ratio_p90": float(np.percentile(dom_ratio_np, 90)),
        "dominant_ratio_beta_weighted_mean": float((dominant_ratio * den).sum() / (den.sum() + eps)),
        "purity07_valid_count": int(purity07_mask.sum()),
        "purity07_valid_ratio": float(purity07_mask.float().mean()),
        "beta_mass_total": beta_mass_total,
        "beta_mass_kept_by_beta_mask": beta_mass_kept_by_beta,
        "beta_mass_kept_by_purity07_mask": beta_mass_kept_by_purity,
        "id_contrib_beta_consistency_error": {"mean": mean_err, "max": max_err, "relative_mean": rel_err},
        "total_views": total_views,
        "total_dets_used": total_dets_used,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(args.out_dir, "aggregation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "aggregation_gaussian_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gaussian_idx", "beta", "valid", "feature_norm_raw",
                     "dominant_ratio", "dominant_id_col"])
        for i in range(min(N, 10000)):
            w.writerow([i, f"{beta_np[i]:.6f}", bool(valid_np[i]), f"{norm_raw[i]:.6f}",
                        f"{dom_ratio_np[i]:.6f}", int(dominant_id_col[i].item())])

    with open(os.path.join(args.out_dir, "aggregation_view_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "camera_id", "num_detections", "used_detections"])
        w.writeheader()
        w.writerows(view_stats)

    consistency_flag = "**PASS**" if rel_err < 1e-3 else "**FAIL (high error)**"
    r = "# V3.0.1 Aggregation Report\n\n"
    r += f"## Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
    r += "## Configuration\n\n"
    r += f"- ckpt: {args.ckpt}\n"
    r += f"- mask_policy: {args.mask_policy}\n"
    r += f"- shrink_ratio: {args.shrink_ratio}\n"
    r += f"- overlap_policy: {args.overlap_policy}\n"
    r += f"- max_frames: {args.max_frames}\n"
    r += f"- feat_dim: {FEAT_DIM}\n"
    r += f"- num_ids: {num_ids}\n\n"
    r += "## Results\n\n"
    r += "| Metric | Value |\n|--------|-------|\n"
    for k, v in summary.items():
        if isinstance(v, float):
            r += f"| {k} | {v:.4f} |\n"
        elif isinstance(v, dict):
            for kk, vv in v.items():
                r += f"| {k}.{kk} | {vv:.6f} |\n"
        else:
            r += f"| {k} | {v} |\n"
    r += f"\n## ID-Contrib Consistency\n\n"
    r += f"- Error check: {consistency_flag}\n"
    r += f"- Mean abs error: {mean_err:.6f}\n"
    r += f"- Max abs error: {max_err:.6f}\n"
    r += f"- Mean relative error: {rel_err:.6f}\n"
    r += "\n## Per-Camera Stats\n\n"
    r += "| Camera | Views | Dets | Used |\n|--------|-------|------|------|\n"
    for cam in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
        cs = cam_stats.get(cam, {"views": 0, "dets": 0, "used": 0})
        r += f"| {cam} | {cs['views']} | {cs['dets']} | {cs['used']} |\n"
    r += f"\n## Decision\n\n"
    if summary["beta_valid_ratio"] > 0.1 and summary["beta_mean"] > 0:
        r += "PASS: Aggregation produced valid features.\n"
    else:
        r += "FAIL: Too few valid Gaussians or zero beta.\n"

    with open(os.path.join(args.out_dir, "final_report.md"), "w") as f:
        f.write(r)

    print(f"\nAggregation complete. Saved to {args.out_dir}/")
    print(f"  Valid Gaussians: {summary['beta_valid_count']}/{N} ({summary['beta_valid_ratio']:.2%})")
    print(f"  Purity>=0.7: {summary['purity07_valid_count']}/{N} ({summary['purity07_valid_ratio']:.2%})")
    print(f"  Dominant ratio mean: {summary['dominant_ratio_mean']:.4f}")
    print(f"  ID-contrib consistency: mean_err={mean_err:.6f}, rel_err={rel_err:.6f}")


if __name__ == "__main__":
    main()
