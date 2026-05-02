#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset / Cache Reconciliation Check.

Verifies that teacher feature cache is consistent with annotations_remapped.
Must pass before training starts.

Usage:
    python tools/verify_cache_alignment.py \
        --dataset_path /data02/zhangrunxiang/data/Wildtrack \
        --cache_dir /data02/zhangrunxiang/data/Wildtrack/reid_teacher_cache

Reports:
    - Total annotation instances
    - Cache hit count
    - Cache hit rate
    - train_id mismatch count
    - bbox mismatch count
    - PASS / FAIL verdict
"""

import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from threedgrut.datasets.cache_key import make_cache_key
from threedgrut.datasets.reid_teacher_cache import ReidTeacherCache


def load_annotations(dataset_path):
    annot_dir = os.path.join(dataset_path, "annotations_remapped")
    if not os.path.exists(annot_dir):
        print(f"[ERROR] annotations_remapped not found at {annot_dir}")
        sys.exit(1)

    annotations = {}
    for annot_file in sorted(os.listdir(annot_dir)):
        if annot_file.endswith(".json"):
            frame_id = int(re.search(r'([0-9]+)\.json', annot_file).group(1))
            with open(os.path.join(annot_dir, annot_file), 'r') as f:
                annotations[frame_id] = json.load(f)

    return annotations


def verify_alignment(dataset_path, cache_dir, bbox_tol=5):
    print(f"=== Cache Alignment Verification ===")
    print(f"  dataset_path: {dataset_path}")
    print(f"  cache_dir:    {cache_dir}")
    print()

    cache = ReidTeacherCache(dataset_path, cache_dir=cache_dir)
    annotations = load_annotations(dataset_path)

    total_instances = 0
    cache_hits = 0
    train_id_mismatches = 0
    bbox_mismatches = 0
    embedding_norm_failures = 0

    for frame_id, frame_annots in sorted(annotations.items()):
        for inst_idx, person in enumerate(frame_annots):
            raw_id = person.get('raw_id', 0)
            train_id = person.get('new_id') or person.get('train_id')
            camera_id_int = person.get('camera_id')
            if raw_id == 0 or train_id is None or camera_id_int is None:
                continue

            camera_id = f"C{camera_id_int + 1}"

            bbox_dict = person.get('bbox', {})
            bbox_xyxy = [
                bbox_dict.get('xmin', 0),
                bbox_dict.get('ymin', 0),
                bbox_dict.get('xmax', 0),
                bbox_dict.get('ymax', 0),
            ]

            if bbox_xyxy[2] <= bbox_xyxy[0] or bbox_xyxy[3] <= bbox_xyxy[1]:
                continue

            total_instances += 1

            cache_key = make_cache_key(frame_id, camera_id, train_id, bbox_xyxy)
            cache_entry = cache.get(cache_key)

            if cache_entry is None:
                continue

            cache_hits += 1

            if cache_entry.get('train_id') != train_id:
                train_id_mismatches += 1
                continue

            cache_bbox = cache_entry.get('bbox_xyxy', [])
            if cache_bbox:
                for c, v in zip(cache_bbox, bbox_xyxy):
                    if abs(c - v) > bbox_tol:
                        bbox_mismatches += 1
                        break

            import numpy as np
            emb = cache_entry.get('embedding')
            if emb is not None:
                norm = np.linalg.norm(emb)
                if not (0.99 < norm < 1.01):
                    embedding_norm_failures += 1

    hit_rate = cache_hits / total_instances if total_instances > 0 else 0.0

    print(f"--- Results ---")
    print(f"  Total annotation instances: {total_instances}")
    print(f"  Cache hits:                 {cache_hits}")
    print(f"  Cache hit rate:             {hit_rate:.2%}")
    print(f"  train_id mismatches:        {train_id_mismatches}")
    print(f"  bbox mismatches:            {bbox_mismatches}")
    print(f"  embedding norm failures:    {embedding_norm_failures}")
    print()

    passed = (
        train_id_mismatches == 0
        and bbox_mismatches == 0
        and embedding_norm_failures == 0
        and hit_rate > 0.5
    )

    if passed:
        print(f"  VERDICT: PASS (hit_rate={hit_rate:.2%})")
    else:
        reasons = []
        if train_id_mismatches > 0:
            reasons.append(f"train_id_mismatch={train_id_mismatches}")
        if bbox_mismatches > 0:
            reasons.append(f"bbox_mismatch={bbox_mismatches}")
        if embedding_norm_failures > 0:
            reasons.append(f"norm_fail={embedding_norm_failures}")
        if hit_rate <= 0.5:
            reasons.append(f"low_hit_rate={hit_rate:.2%}")
        print(f"  VERDICT: FAIL ({', '.join(reasons)})")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Verify cache alignment")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--bbox_tol", type=int, default=5)
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.path.join(args.dataset_path, "reid_teacher_cache")
    passed = verify_alignment(args.dataset_path, cache_dir, args.bbox_tol)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
