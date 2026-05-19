#!/usr/bin/env python3
"""V3.0.2 Task B: Build Pure Registered Gaussian Features.

Bypasses the 2D renderer entirely. For each Gaussian, directly assign the
teacher prototype of its dominant ID. This tests whether the registration
itself (beta/dominant_ratio logic) has discriminative power for ReID.

R1 beta: valid = beta > eps, feature_i = prototype[dominant_id_i]
R2 purity07: valid = (beta > eps) & (dominant_ratio >= 0.7), feature_i = prototype[dominant_id_i]
invalid Gaussian feature = 0.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_teacher_prototypes(path):
    """Load teacher prototypes, handling dict/list/tensor formats."""
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, torch.Tensor):
        if data.ndim == 2:
            prototypes = data
            id_to_row = {i: i for i in range(data.shape[0])}
            fmt = "tensor [N, D]"
        else:
            raise ValueError(f"Unexpected tensor shape: {data.shape}")
    elif isinstance(data, dict):
        if "prototypes" in data and isinstance(data["prototypes"], torch.Tensor):
            prototypes = data["prototypes"]
            fmt = "dict['prototypes']"
            id_to_row = {}
            if "id_mapping" in data:
                id_mapping = data["id_mapping"]
                if isinstance(id_mapping, dict):
                    for k, v in id_mapping.items():
                        id_to_row[int(k)] = int(v)
                elif isinstance(id_mapping, list):
                    id_to_row = {int(tid): i for i, tid in enumerate(id_mapping)}
            else:
                id_to_row = {i: i for i in range(prototypes.shape[0])}
        else:
            first_tensor_val = None
            int_keys = sorted([k for k in data.keys() if isinstance(k, (int,)) or (isinstance(k, str) and k.isdigit())],
                              key=lambda x: int(x))
            if int_keys:
                prototypes_list = [data[k] if isinstance(data[k], torch.Tensor) else torch.tensor(data[k]) for k in int_keys]
                prototypes = torch.stack(prototypes_list, dim=0)
                id_to_row = {int(k): i for i, k in enumerate(int_keys)}
                fmt = f"dict[int_keys]"
            else:
                raise ValueError(f"Cannot parse dict keys: {list(data.keys())[:10]}")
    elif isinstance(data, (list, tuple)):
        prototypes = torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in data], dim=0)
        id_to_row = {i: i for i in range(prototypes.shape[0])}
        fmt = "list"
    else:
        raise ValueError(f"Unknown format: {type(data)}")

    print(f"  Teacher prototypes format: {fmt}")
    print(f"  Shape: {prototypes.shape}, dtype: {prototypes.dtype}")
    print(f"  ID mapping entries: {len(id_to_row)}")
    return prototypes, id_to_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregated_features", required=True)
    parser.add_argument("--teacher_prototypes",
                        default="/data02/zhangrunxiang/data/Wildtrack/reid_teacher_prototypes.pt")
    parser.add_argument("--beta_eps", type=float, default=1e-6)
    parser.add_argument("--purity_thr", type=float, default=0.7)
    parser.add_argument("--out_dir", default="outputs/v3_0_2_pure_registration")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print(f"V3.0.2: Build Pure Registered Gaussian Features")
    print("=" * 60)

    print("\n[1/3] Loading aggregated features...")
    agg = torch.load(args.aggregated_features, map_location="cpu", weights_only=False)
    beta = agg["beta"]
    dominant_ratio = agg["dominant_ratio"]
    dominant_id_col = agg["dominant_id_col"]
    id_to_col = agg["id_to_col"]
    col_to_id = agg["col_to_id"]
    N = beta.shape[0]
    num_ids = len(id_to_col)
    print(f"  N_gaussians: {N}, num_ids: {num_ids}")
    print(f"  beta valid (>eps): {(beta > args.beta_eps).sum().item()}")
    print(f"  dominant_ratio mean: {dominant_ratio.mean().item():.4f}")

    print("\n[2/3] Loading teacher prototypes...")
    prototypes, proto_id_to_row = load_teacher_prototypes(args.teacher_prototypes)
    feat_dim = prototypes.shape[1]

    col_to_proto_row = {}
    missing_count = 0
    for col, train_id in col_to_id.items():
        if train_id in proto_id_to_row:
            col_to_proto_row[col] = proto_id_to_row[train_id]
        else:
            missing_count += 1
    if missing_count > 0:
        print(f"  [WARN] {missing_count} train_ids not found in teacher prototypes")

    print("\n[3/3] Building registered features...")
    t0 = time.time()

    proto_row_tensor = torch.zeros(len(col_to_proto_row), feat_dim, dtype=torch.float32)
    for col, row_idx in sorted(col_to_proto_row.items()):
        if col < len(proto_row_tensor):
            proto_row_tensor[col] = prototypes[row_idx]

    dominant_col_np = dominant_id_col.numpy()
    feature_matrix = proto_row_tensor[torch.tensor(dominant_col_np, dtype=torch.long)]

    mask_beta = beta > args.beta_eps
    mask_purity07 = (beta > args.beta_eps) & (dominant_ratio >= args.purity_thr)

    features_beta = torch.zeros(N, feat_dim, dtype=torch.float32)
    features_beta[mask_beta] = feature_matrix[mask_beta]

    features_purity07 = torch.zeros(N, feat_dim, dtype=torch.float32)
    features_purity07[mask_purity07] = feature_matrix[mask_purity07]

    total_beta = beta.sum().item()
    kept_beta_beta = beta[mask_beta].sum().item()
    kept_beta_purity = beta[mask_purity07].sum().item()

    feat_norm_beta = features_beta[mask_beta].norm(dim=1) if mask_beta.any() else torch.tensor([0.0])
    feat_norm_purity = features_purity07[mask_purity07].norm(dim=1) if mask_purity07.any() else torch.tensor([0.0])

    dr_np = dominant_ratio.numpy()
    beta_np = beta.numpy()

    elapsed = time.time() - t0

    report = {
        "num_gaussians": N,
        "num_ids": num_ids,
        "feat_dim": feat_dim,
        "beta_eps": args.beta_eps,
        "purity_thr": args.purity_thr,
        "beta": {
            "valid_count": int(mask_beta.sum()),
            "valid_ratio": float(mask_beta.float().mean()),
            "kept_beta_mass_ratio": kept_beta_beta / (total_beta + 1e-6),
            "feature_norm_mean": float(feat_norm_beta.mean()) if mask_beta.any() else 0,
            "feature_norm_median": float(feat_norm_beta.median()) if mask_beta.any() else 0,
        },
        "purity07": {
            "valid_count": int(mask_purity07.sum()),
            "valid_ratio": float(mask_purity07.float().mean()),
            "kept_beta_mass_ratio": kept_beta_purity / (total_beta + 1e-6),
            "feature_norm_mean": float(feat_norm_purity.mean()) if mask_purity07.any() else 0,
            "feature_norm_median": float(feat_norm_purity.median()) if mask_purity07.any() else 0,
        },
        "dominant_ratio_stats": {
            "mean": float(dr_np.mean()),
            "median": float(np.median(dr_np)),
            "p10": float(np.percentile(dr_np, 10)),
            "p90": float(np.percentile(dr_np, 90)),
        },
        "elapsed_seconds": elapsed,
    }

    with open(os.path.join(args.out_dir, "registration_summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    torch.save({
        "features_beta": features_beta,
        "mask_beta": mask_beta,
        "features_purity07": features_purity07,
        "mask_purity07": mask_purity07,
        "beta": beta,
        "dominant_ratio": dominant_ratio,
        "id_to_col": id_to_col,
        "col_to_id": col_to_id,
        "col_to_proto_row": col_to_proto_row,
        "config": {
            "aggregated_features": args.aggregated_features,
            "teacher_prototypes": args.teacher_prototypes,
            "beta_eps": args.beta_eps,
            "purity_thr": args.purity_thr,
        },
    }, os.path.join(args.out_dir, "registered_features.pt"))

    r = "# V3.0.2 Pure Registration Report\n\n"
    r += f"## Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
    r += "## Configuration\n\n"
    r += f"- beta_eps: {args.beta_eps}\n- purity_thr: {args.purity_thr}\n\n"
    r += "## Results\n\n"
    r += f"- Total Gaussians: {N}\n"
    r += f"- Feature dim: {feat_dim}\n\n"
    r += "### Beta Mask\n\n"
    r += f"- Valid count: {report['beta']['valid_count']} ({report['beta']['valid_ratio']:.4f})\n"
    r += f"- Kept beta mass: {report['beta']['kept_beta_mass_ratio']:.4f}\n"
    r += f"- Feature norm mean: {report['beta']['feature_norm_mean']:.4f}\n\n"
    r += "### Purity07 Mask\n\n"
    r += f"- Valid count: {report['purity07']['valid_count']} ({report['purity07']['valid_ratio']:.4f})\n"
    r += f"- Kept beta mass: {report['purity07']['kept_beta_mass_ratio']:.4f}\n"
    r += f"- Feature norm mean: {report['purity07']['feature_norm_mean']:.4f}\n\n"
    r += "### Dominant Ratio\n\n"
    r += f"- Mean: {report['dominant_ratio_stats']['mean']:.4f}\n"
    r += f"- Median: {report['dominant_ratio_stats']['median']:.4f}\n"
    r += f"- P10: {report['dominant_ratio_stats']['p10']:.4f}\n"
    r += f"- P90: {report['dominant_ratio_stats']['p90']:.4f}\n"

    with open(os.path.join(args.out_dir, "final_report.md"), "w") as f:
        f.write(r)

    print(f"\nRegistration complete. Saved to {args.out_dir}/")
    print(f"  Beta valid: {report['beta']['valid_count']}/{N} ({report['beta']['valid_ratio']:.2%})")
    print(f"  Purity07 valid: {report['purity07']['valid_count']}/{N} ({report['purity07']['valid_ratio']:.2%})")
    print(f"  Report: {args.out_dir}/final_report.md")


if __name__ == "__main__":
    main()
