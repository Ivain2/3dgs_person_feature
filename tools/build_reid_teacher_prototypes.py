#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Build teacher prototypes from the existing ReID teacher cache.

For each train_id, compute:
    P_i = normalize(mean({t_v | train_id = i}))

Usage:
    cd /data02/zhangrunxiang/3dgrut
    conda activate 3dgrut
    PYTHONPATH=/data02/zhangrunxiang/3dgrut:$PYTHONPATH python tools/build_reid_teacher_prototypes.py \
        --config configs/apps/wildtrack_full_3dgut.yaml \
        --output /data02/zhangrunxiang/data/Wildtrack/reid_teacher_prototypes.pt \
        --min_count 2
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from threedgrut.datasets.cache_key import parse_cache_filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str,
                        default='/data02/zhangrunxiang/data/Wildtrack/reid_teacher_prototypes.pt')
    parser.add_argument('--min_count', type=int, default=2)
    args = parser.parse_args()

    print("=" * 70)
    print("Build Teacher Prototypes")
    print("=" * 70)
    print(f"  config    = {args.config}")
    print(f"  output    = {args.output}")
    print(f"  min_count = {args.min_count}")

    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = args.config.replace("configs/", "").replace(".yaml", "")

    from hydra import initialize_config_dir, compose
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        conf = compose(config_name=config_name)

    dataset_path = conf.dataset.get('dataset_path', '/data02/zhangrunxiang/data/Wildtrack')
    cache_dir = os.path.join(dataset_path, "reid_teacher_cache")

    if not os.path.exists(cache_dir):
        print(f"ERROR: Cache directory not found: {cache_dir}")
        return False

    print(f"\nScanning cache: {cache_dir}")

    embedding_by_id = {}
    total_entries = 0
    skipped = 0
    nan_count = 0

    for filename in sorted(os.listdir(cache_dir)):
        if not filename.endswith(".pt"):
            continue

        total_entries += 1

        try:
            cache_key = parse_cache_filename(filename)
        except (ValueError, IndexError):
            skipped += 1
            continue

        train_id = cache_key[2]

        filepath = os.path.join(cache_dir, filename)
        try:
            data = torch.load(filepath, map_location="cpu", weights_only=False)
        except Exception:
            skipped += 1
            continue

        embedding = data.get('embedding')
        if embedding is None:
            skipped += 1
            continue

        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)
        embedding = embedding.float().squeeze()

        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            nan_count += 1
            skipped += 1
            continue

        embedding = F.normalize(embedding, p=2, dim=0, eps=1e-6)

        if train_id not in embedding_by_id:
            embedding_by_id[train_id] = []
        embedding_by_id[train_id].append(embedding)

        if total_entries % 5000 == 0:
            print(f"  scanned {total_entries} entries, {len(embedding_by_id)} unique IDs")

    print(f"\nScan complete:")
    print(f"  total entries scanned: {total_entries}")
    print(f"  skipped: {skipped}")
    print(f"  NaN/Inf: {nan_count}")
    print(f"  unique train_ids: {len(embedding_by_id)}")

    if not embedding_by_id:
        print("ERROR: No valid embeddings found!")
        return False

    all_ids = sorted(embedding_by_id.keys())
    max_id = max(all_ids)
    embedding_dim = embedding_by_id[all_ids[0]][0].shape[0]
    print(f"  max train_id: {max_id}")
    print(f"  embedding_dim: {embedding_dim}")

    prototypes = torch.zeros(max_id + 1, embedding_dim)
    counts = torch.zeros(max_id + 1, dtype=torch.long)
    valid_mask = torch.zeros(max_id + 1, dtype=torch.bool)

    for tid in all_ids:
        embs = embedding_by_id[tid]
        count = len(embs)
        counts[tid] = count

        if count < args.min_count:
            continue

        stacked = torch.stack(embs)
        mean_emb = stacked.mean(dim=0)
        prototype = F.normalize(mean_emb, p=2, dim=0, eps=1e-6)
        prototypes[tid] = prototype
        valid_mask[tid] = True

    num_valid = valid_mask.sum().item()
    print(f"\n  valid IDs (count >= {args.min_count}): {num_valid}")
    print(f"  invalid IDs (count < {args.min_count}): {len(all_ids) - num_valid}")

    valid_counts = counts[valid_mask]
    if valid_counts.numel() > 0:
        print(f"\n  count stats (valid IDs):")
        print(f"    min  = {valid_counts.min().item()}")
        print(f"    mean = {valid_counts.float().mean().item():.1f}")
        print(f"    max  = {valid_counts.max().item()}")

    valid_protos = prototypes[valid_mask]
    if valid_protos.numel() > 0:
        proto_norms = valid_protos.norm(p=2, dim=1)
        print(f"\n  prototype norm stats:")
        print(f"    min  = {proto_norms.min().item():.6f}")
        print(f"    mean = {proto_norms.mean().item():.6f}")
        print(f"    max  = {proto_norms.max().item():.6f}")

    has_nan = torch.isnan(prototypes).any().item()
    has_inf = torch.isinf(prototypes).any().item()
    print(f"\n  NaN in prototypes: {has_nan}")
    print(f"  Inf in prototypes: {has_inf}")

    print(f"\n  First few train_ids:")
    for tid in all_ids[:10]:
        v = "valid" if valid_mask[tid] else "INVALID"
        print(f"    train_id={tid}: count={counts[tid].item()}, {v}")

    save_data = {
        'prototypes': prototypes,
        'counts': counts,
        'valid_mask': valid_mask,
        'train_ids': all_ids,
        'embedding_dim': embedding_dim,
        'min_count': args.min_count,
        'num_valid_ids': num_valid,
        'total_cache_entries': total_entries,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(save_data, args.output)
    print(f"\nPrototypes saved: {args.output}")
    print(f"  file size: {os.path.getsize(args.output) / 1024:.1f} KB")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
