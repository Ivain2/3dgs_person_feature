#!/usr/bin/env python3
"""V3.0.2 Task A: Renderer Linearity Diagnosis.

Tests whether person_feature rendering is linear (or has bias/nonlinearity).
5 tests: T1 zero, T2 scale, T3 additivity, T4 sign, T5 background.

NO TRAINING. NO GEOMETRY CHANGES.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from threedgrut.datasets.dataset_wildtrack import WildtrackDataset
from threedgrut.model.model import MixtureOfGaussians

DATASET_PATH = "/data02/zhangrunxiang/data/Wildtrack"
REID_INIT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase14_clean_geometry/full_soft_reset_30k/reid_init/reid_init_ckpt.pt"
DEFAULT_CKPT = "/data02/zhangrunxiang/3dgrut/outputs/phase15_reid_teacher_only_medium_1000/checkpoint_1000.pt"
FEAT_DIM = 512


def setup_model(ckpt_path, device, feature_dim):
    reid_state = torch.load(REID_INIT_CKPT, map_location="cpu", weights_only=False)
    conf = reid_state.get("config")
    conf.model.person_feature_dim = feature_dim
    scene_extent = reid_state.get("scene_extent", 1.0)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ["positions", "density", "scale", "rotation", "features_albedo", "features_specular"]:
        if key in state:
            getattr(model, key).data = state[key].to(device)
    model._person_feature = torch.nn.Parameter(
        torch.zeros(model.positions.shape[0], feature_dim, device=device, dtype=torch.float32))
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, conf


@torch.no_grad()
def test_zero(model, gpu_batch, linearize_feature=False, linearize_mode="sh_offset"):
    f = torch.zeros_like(model._person_feature)
    model._person_feature = torch.nn.Parameter(f)
    pf_map, opacity_map = model.render_person_feature_map(
        gpu_batch, train=False, frame_id=0,
        linearize_feature=linearize_feature, linearize_mode=linearize_mode)
    max_abs = pf_map.abs().max().item()
    mean_abs = pf_map.abs().mean().item()
    return {"max_abs": max_abs, "mean_abs": mean_abs, "pass": max_abs < 1e-5}


@torch.no_grad()
def test_scale(model, gpu_batch, feature_dim, generator, linearize_feature=False, linearize_mode="sh_offset"):
    f = torch.randn(model.positions.shape[0], feature_dim, generator=generator, dtype=torch.float32).to(gpu_batch.rays_ori.device)
    model._person_feature = torch.nn.Parameter(f)
    out_f, _ = model.render_person_feature_map(
        gpu_batch, train=False, frame_id=0,
        linearize_feature=linearize_feature, linearize_mode=linearize_mode)

    model._person_feature = torch.nn.Parameter(f * 2.0)
    out_2f, _ = model.render_person_feature_map(
        gpu_batch, train=False, frame_id=0,
        linearize_feature=linearize_feature, linearize_mode=linearize_mode)

    abs_err = (out_2f - 2 * out_f).abs()
    abs_err_val = abs_err.max().item()
    rel_denom = out_f.abs().max() + 1e-8
    rel_err_val = (abs_err / rel_denom).max().item()

    return {"abs_error_max": abs_err_val, "rel_error_max": rel_err_val, "pass": rel_err_val < 0.01}


@torch.no_grad()
def test_additivity(model, gpu_batch, feature_dim, generator, linearize_feature=False, linearize_mode="sh_offset"):
    f1 = torch.randn(model.positions.shape[0], feature_dim, generator=generator, dtype=torch.float32).to(gpu_batch.rays_ori.device)
    f2 = torch.randn(model.positions.shape[0], feature_dim, generator=generator, dtype=torch.float32).to(gpu_batch.rays_ori.device)

    model._person_feature = torch.nn.Parameter(f1)
    out_f1, _ = model.render_person_feature_map(
        gpu_batch, train=False, frame_id=0,
        linearize_feature=linearize_feature, linearize_mode=linearize_mode)

    model._person_feature = torch.nn.Parameter(f2)
    out_f2, _ = model.render_person_feature_map(
        gpu_batch, train=False, frame_id=0,
        linearize_feature=linearize_feature, linearize_mode=linearize_mode)

    model._person_feature = torch.nn.Parameter(f1 + f2)
    out_f12, _ = model.render_person_feature_map(
        gpu_batch, train=False, frame_id=0,
        linearize_feature=linearize_feature, linearize_mode=linearize_mode)

    expected = out_f1 + out_f2
    abs_err = (out_f12 - expected).abs()
    abs_err_val = abs_err.max().item()
    rel_denom = expected.abs().max() + 1e-8
    rel_err_val = (abs_err / rel_denom).max().item()

    return {"abs_error_max": abs_err_val, "rel_error_max": rel_err_val, "pass": rel_err_val < 0.01}


@torch.no_grad()
def test_sign(model, gpu_batch, feature_dim, generator, linearize_feature=False, linearize_mode="sh_offset"):
    f = torch.randn(model.positions.shape[0], feature_dim, generator=generator, dtype=torch.float32).to(gpu_batch.rays_ori.device)
    model._person_feature = torch.nn.Parameter(f)
    pf_map, opacity_map = model.render_person_feature_map(
        gpu_batch, train=False, frame_id=0,
        linearize_feature=linearize_feature, linearize_mode=linearize_mode)

    neg_count = (pf_map < -1e-6).sum().item()
    total_count = pf_map.numel()
    neg_ratio = neg_count / max(1, total_count)
    min_val = pf_map.min().item()

    return {"min_value": min_val, "negative_ratio": neg_ratio, "pass": neg_ratio > 0.001}


@torch.no_grad()
def test_background(model, gpu_batch, feature_dim, generator, linearize_feature=False, linearize_mode="sh_offset"):
    f = torch.randn(model.positions.shape[0], feature_dim, generator=generator, dtype=torch.float32).to(gpu_batch.rays_ori.device)
    model._person_feature = torch.nn.Parameter(f)
    pf_map, opacity_map = model.render_person_feature_map(
        gpu_batch, train=False, frame_id=0,
        linearize_feature=linearize_feature, linearize_mode=linearize_mode)

    bg_mask = opacity_map < 0.01
    if bg_mask.sum() > 0:
        bg_features = pf_map[:, bg_mask]
        bg_mean = bg_features.abs().mean().item()
        bg_max = bg_features.abs().max().item()
    else:
        bg_mean = 0.0
        bg_max = 0.0

    fg_mask = opacity_map >= 0.1
    if fg_mask.sum() > 0:
        fg_features = pf_map[:, fg_mask]
        fg_mean = fg_features.abs().mean().item()
    else:
        fg_mean = 0.0

    bg_fg_ratio = bg_mean / (fg_mean + 1e-8)

    return {"bg_feature_mean_abs": bg_mean, "fg_feature_mean_abs": fg_mean,
            "bg_fg_ratio": bg_fg_ratio, "pass": bg_fg_ratio < 0.5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--max_views", type=int, default=5)
    parser.add_argument("--feature_dim", type=int, default=FEAT_DIM)
    parser.add_argument("--linearize_feature", action="store_true")
    parser.add_argument("--linearize_mode", default="sh_offset", choices=["sh_offset"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="outputs/v3_0_2_renderer_linearity")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    seed = 42
    generator = torch.Generator(device="cpu").manual_seed(seed)

    print("=" * 60)
    lin_str = f" (linearize={args.linearize_feature}, mode={args.linearize_mode})" if args.linearize_feature else ""
    print(f"V3.0.2/3.0.3: Renderer Linearity Diagnosis (max_views={args.max_views}){lin_str}")
    print("=" * 60)

    print("\n[1/3] Loading model...")
    model, conf = setup_model(args.ckpt, device, args.feature_dim)
    N = model.positions.shape[0]
    print(f"  N_gaussians={N}, person_feature_dim={args.feature_dim}")

    print("\n[2/3] Loading dataset...")
    ds = WildtrackDataset(dataset_path=DATASET_PATH, downsample_factor=4, load_teacher_cache=False)
    print(f"  Dataset size: {len(ds)}")

    fc_mapping = {}
    for idx in range(len(ds)):
        batch = ds[idx]
        fid = batch.get("frame_idx", -1)
        cid = batch.get("camera_id", "unknown")
        fc_mapping[(fid, cid)] = idx

    view_indices = list(fc_mapping.values())[:args.max_views]
    print(f"  Selected {len(view_indices)} views")

    print("\n[3/3] Running linearity tests...")
    per_view_results = []
    all_results = {"T1_zero": [], "T2_scale": [], "T3_additivity": [], "T4_sign": [], "T5_background": []}

    t0 = time.time()
    for vi, idx in enumerate(view_indices):
        batch = ds[idx]
        gpu_batch = ds.get_gpu_batch_with_intrinsics(batch)
        fid = batch.get("frame_idx", -1)
        cid = batch.get("camera_id", "unknown")
        print(f"\n  View {vi+1}/{len(view_indices)}: frame={fid}, cam={cid}")

        r_zero = test_zero(model, gpu_batch, args.linearize_feature, args.linearize_mode)
        r_scale = test_scale(model, gpu_batch, args.feature_dim, generator, args.linearize_feature, args.linearize_mode)
        r_add = test_additivity(model, gpu_batch, args.feature_dim, generator, args.linearize_feature, args.linearize_mode)
        r_sign = test_sign(model, gpu_batch, args.feature_dim, generator, args.linearize_feature, args.linearize_mode)
        r_bg = test_background(model, gpu_batch, args.feature_dim, generator, args.linearize_feature, args.linearize_mode)

        view_result = {
            "frame_id": fid, "camera_id": cid, "view_index": idx,
            "T1_zero": r_zero, "T2_scale": r_scale, "T3_additivity": r_add,
            "T4_sign": r_sign, "T5_background": r_bg,
        }
        per_view_results.append(view_result)
        for test_name, result in [("T1_zero", r_zero), ("T2_scale", r_scale),
                                   ("T3_additivity", r_add), ("T4_sign", r_sign),
                                   ("T5_background", r_bg)]:
            all_results[test_name].append(result)

        print(f"    T1 zero: max_abs={r_zero['max_abs']:.6e}, mean_abs={r_zero['mean_abs']:.6e}, PASS={r_zero['pass']}")
        print(f"    T2 scale: abs_err={r_scale['abs_error_max']:.6e}, rel_err={r_scale['rel_error_max']:.6e}, PASS={r_scale['pass']}")
        print(f"    T3 add: abs_err={r_add['abs_error_max']:.6e}, rel_err={r_add['rel_error_max']:.6e}, PASS={r_add['pass']}")
        print(f"    T4 sign: min={r_sign['min_value']:.6e}, neg_ratio={r_sign['negative_ratio']:.6f}, PASS={r_sign['pass']}")
        print(f"    T5 bg: bg_mean={r_bg['bg_feature_mean_abs']:.6e}, fg_mean={r_bg['fg_feature_mean_abs']:.6e}, ratio={r_bg['bg_fg_ratio']:.6f}, PASS={r_bg['pass']}")

    elapsed = time.time() - t0

    summary = {}
    for test_name, results in all_results.items():
        summary[test_name] = {"count": len(results)}
        if test_name == "T1_zero":
            summary[test_name]["max_abs_mean"] = float(np.mean([r["max_abs"] for r in results]))
            summary[test_name]["mean_abs_mean"] = float(np.mean([r["mean_abs"] for r in results]))
            summary[test_name]["pass_ratio"] = float(np.mean([r["pass"] for r in results]))
            summary[test_name]["all_pass"] = all(r["pass"] for r in results)
        elif test_name in ("T2_scale", "T3_additivity"):
            summary[test_name]["abs_error_max"] = float(max(r["abs_error_max"] for r in results))
            summary[test_name]["rel_error_max"] = float(max(r["rel_error_max"] for r in results))
            summary[test_name]["pass_ratio"] = float(np.mean([r["pass"] for r in results]))
            summary[test_name]["all_pass"] = all(r["pass"] for r in results)
        elif test_name == "T4_sign":
            summary[test_name]["min_value_mean"] = float(np.mean([r["min_value"] for r in results]))
            summary[test_name]["negative_ratio_mean"] = float(np.mean([r["negative_ratio"] for r in results]))
            summary[test_name]["pass_ratio"] = float(np.mean([r["pass"] for r in results]))
            summary[test_name]["all_pass"] = all(r["pass"] for r in results)
        elif test_name == "T5_background":
            summary[test_name]["bg_fg_ratio_mean"] = float(np.mean([r["bg_fg_ratio"] for r in results]))
            summary[test_name]["pass_ratio"] = float(np.mean([r["pass"] for r in results]))
            summary[test_name]["all_pass"] = all(r["pass"] for r in results)

    summary["total_views"] = len(view_indices)
    summary["feature_dim"] = args.feature_dim
    summary["elapsed_seconds"] = elapsed

    with open(os.path.join(args.out_dir, "renderer_linearity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "renderer_linearity_per_view.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "camera_id",
                     "T1_max_abs", "T1_mean_abs", "T1_pass",
                     "T2_abs_err", "T2_rel_err", "T2_pass",
                     "T3_abs_err", "T3_rel_err", "T3_pass",
                     "T4_min", "T4_neg_ratio", "T4_pass",
                     "T5_bg_mean", "T5_fg_mean", "T5_ratio", "T5_pass"])
        for vr in per_view_results:
            w.writerow([vr["frame_id"], vr["camera_id"],
                        f"{vr['T1_zero']['max_abs']:.6e}", f"{vr['T1_zero']['mean_abs']:.6e}", vr['T1_zero']['pass'],
                        f"{vr['T2_scale']['abs_error_max']:.6e}", f"{vr['T2_scale']['rel_error_max']:.6e}", vr['T2_scale']['pass'],
                        f"{vr['T3_additivity']['abs_error_max']:.6e}", f"{vr['T3_additivity']['rel_error_max']:.6e}", vr['T3_additivity']['pass'],
                        f"{vr['T4_sign']['min_value']:.6e}", f"{vr['T4_sign']['negative_ratio']:.6f}", vr['T4_sign']['pass'],
                        f"{vr['T5_background']['bg_feature_mean_abs']:.6e}", f"{vr['T5_background']['fg_feature_mean_abs']:.6e}",
                        f"{vr['T5_background']['bg_fg_ratio']:.6f}", vr['T5_background']['pass']])

    flags = {}
    flags["zero_offset"] = "FAIL (renderer has constant offset)" if not summary["T1_zero"]["all_pass"] else "PASS"
    flags["scale_nonlinear"] = "FAIL (render(2f) != 2*render(f))" if not summary["T2_scale"]["all_pass"] else "PASS"
    flags["additivity_nonlinear"] = "FAIL (render(f1+f2) != render(f1)+render(f2))" if not summary["T3_additivity"]["all_pass"] else "PASS"
    flags["sign_clamp"] = "FAIL (negative values clamped/sigmoided)" if not summary["T4_sign"]["all_pass"] else "PASS"
    flags["bg_leakage"] = "FAIL (background has non-zero features)" if not summary["T5_background"]["all_pass"] else "PASS"

    r = "# V3.0.3 Renderer Linearity Report (SH-Offset Linearized)\n\n" if args.linearize_feature else "# V3.0.2 Renderer Linearity Report\n\n"
    r += f"## Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
    r += "## Configuration\n\n"
    r += f"- ckpt: {args.ckpt}\n- max_views: {args.max_views}\n- feature_dim: {args.feature_dim}\n"
    if args.linearize_feature:
        r += f"- linearize_feature: True\n- linearize_mode: {args.linearize_mode}\n"
    r += "\n"
    r += "## Results Summary\n\n"
    r += "| Test | Status | Details |\n|------|--------|--------|\n"
    r += f"| T1 Zero | {flags['zero_offset']} | max_abs_mean={summary['T1_zero']['max_abs_mean']:.6e} |\n"
    r += f"| T2 Scale | {flags['scale_nonlinear']} | rel_err_max={summary['T2_scale']['rel_error_max']:.6e} |\n"
    r += f"| T3 Additivity | {flags['additivity_nonlinear']} | rel_err_max={summary['T3_additivity']['rel_error_max']:.6e} |\n"
    r += f"| T4 Sign | {flags['sign_clamp']} | neg_ratio_mean={summary['T4_sign']['negative_ratio_mean']:.6f} |\n"
    r += f"| T5 Background | {flags['bg_leakage']} | bg_fg_ratio_mean={summary['T5_background']['bg_fg_ratio_mean']:.6f} |\n"
    r += f"\n## Diagnosis\n\n"
    r += f"- **Zero test**: {'Renderer produces zero output for zero input ✓' if summary['T1_zero']['all_pass'] else '⚠ Renderer has bias/offset — check background handling'}\n"
    r += f"- **Scale test**: {'Linear scaling preserved ✓' if summary['T2_scale']['all_pass'] else '⚠ Nonlinearity detected — render(2f)≠2×render(f)'}\n"
    r += f"- **Additivity test**: {'Linear additivity preserved ✓' if summary['T3_additivity']['all_pass'] else '⚠ Nonlinearity detected — render(f1+f2)≠render(f1)+render(f2)'}\n"
    r += f"- **Sign test**: {'Negative values preserved ✓' if summary['T4_sign']['all_pass'] else '⚠ Sign clamped — check for clamp/sigmoid/ReLU in renderer'}\n"
    r += f"- **Background test**: {'Background features near zero ✓' if summary['T5_background']['all_pass'] else '⚠ Background leakage — low-opacity regions have non-zero features'}\n"

    all_pass = all(summary[t]["all_pass"] for t in all_results)
    r += f"\n## Overall: {'PASS — renderer is linear' if all_pass else 'FAIL — renderer has nonlinearity/bias'}\n"

    with open(os.path.join(args.out_dir, "final_report.md"), "w") as f:
        f.write(r)

    print(f"\n{'='*60}")
    print(f"V3.0.2 Renderer Linearity Diagnosis Complete")
    for tname, result in summary.items():
        if isinstance(result, dict) and "all_pass" in result:
            status = "PASS" if result["all_pass"] else "FAIL"
            print(f"  {tname}: {status}")
    print(f"Report: {args.out_dir}/final_report.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
