#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Build ReID Teacher Feature Cache.

Extracts retrieval embeddings from a retrained ClipReID teacher model
for every person instance in annotations_remapped, and saves them to disk.

Usage (real teacher):
    python tools/build_reid_teacher_cache.py \
        --dataset_path /data02/zhangrunxiang/data/Wildtrack \
        --clipreid_checkpoint /data02/zhangrunxiang/CLIP-ReID-master/results/wildtrack_vit_clipreid/ViT-B-16_30.pth \
        --clip_model ViT-B-16 \
        --embedding_dim 512

Usage (random, smoke test only):
    python tools/build_reid_teacher_cache.py \
        --dataset_path /data02/zhangrunxiang/data/Wildtrack \
        --use_random \
        --embedding_dim 512

Cache output:
    {dataset_path}/reid_teacher_cache/          (real)
    {dataset_path}/reid_teacher_cache_random/   (random, smoke test only)

Each .pt file contains:
    {
        "frame_id": int,
        "camera_id": str,           # "C1" ~ "C7"
        "inst_idx": int,
        "train_id": int,            # 0-based continuous
        "raw_id": int,
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "embedding": np.ndarray,    # [D] float32, L2-normalized
        "embedding_dim": int,
        "model_name": str,
        "checkpoint": str,
    }
"""

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from threedgrut.datasets.cache_key import make_cache_key, make_cache_filename


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

    print(f"Loaded {len(annotations)} annotation files from {annot_dir}")
    return annotations


def load_image_crop(dataset_path, camera_id, frame_id, bbox_xyxy):
    """Load and crop a person image from the Wildtrack dataset.

    Args:
        dataset_path: root of Wildtrack dataset
        camera_id: "C1"~"C7"
        frame_id: frame number
        bbox_xyxy: [xmin, ymin, xmax, ymax]

    Returns:
        [H, W, 3] RGB numpy array (uint8), or None if failed
    """
    img_dir = os.path.join(dataset_path, "Image_subsets", camera_id)
    img_path = os.path.join(img_dir, f"{frame_id:08d}.png")

    if not os.path.exists(img_path):
        candidates = sorted(os.listdir(img_dir)) if os.path.exists(img_dir) else []
        for c in candidates:
            if c.endswith(".png"):
                num = int(re.search(r'([0-9]+)', c).group(1))
                if num == frame_id:
                    img_path = os.path.join(img_dir, c)
                    break

    if not os.path.exists(img_path):
        return None

    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    xmin, ymin, xmax, ymax = [int(v) for v in bbox_xyxy]
    h, w = img_rgb.shape[:2]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, max(xmin + 1, xmax))
    ymax = min(h, max(ymin + 1, ymax))

    if xmax <= xmin or ymax <= ymin:
        return None

    crop = img_rgb[ymin:ymax, xmin:xmax]
    return crop


def build_cache_real(
    dataset_path,
    output_dir,
    clipreid_checkpoint,
    clip_model="ViT-B-16",
    embedding_dim=512,
    batch_size=32,
    device="cuda",
):
    """Build cache using retrained ClipReID teacher."""
    from threedgrut.datasets.clipreid_wrapper import ClipReIDWrapper

    print(f"[build_cache] Loading ClipReID teacher from {clipreid_checkpoint} ...")
    wrapper = ClipReIDWrapper(
        checkpoint_path=clipreid_checkpoint,
        model_name=clip_model,
        device=device,
    )
    print(f"[build_cache] Teacher ready. embedding_dim={wrapper.embedding_dim}")

    annotations = load_annotations(dataset_path)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    skip_crop_fail = 0
    skip_zero_norm = 0

    all_items = []
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

            all_items.append({
                "frame_id": frame_id,
                "camera_id": camera_id,
                "inst_idx": inst_idx,
                "train_id": train_id,
                "raw_id": raw_id,
                "bbox_xyxy": bbox_xyxy,
            })

    print(f"[build_cache] Total instances to process: {len(all_items)}")

    for i in range(0, len(all_items), batch_size):
        batch_items = all_items[i:i + batch_size]
        batch_crops = []
        batch_valid = []

        for item in batch_items:
            crop = load_image_crop(
                dataset_path, item["camera_id"], item["frame_id"], item["bbox_xyxy"]
            )
            if crop is None:
                skip_crop_fail += 1
                batch_valid.append(False)
                batch_crops.append(np.zeros((64, 32, 3), dtype=np.uint8))
            else:
                batch_valid.append(True)
                batch_crops.append(crop)

        try:
            embeddings = wrapper.extract_batch(batch_crops)
        except Exception as e:
            print(f"[WARN] Batch inference failed at batch {i}: {e}, falling back to single")
            embeddings = []
            for crop in batch_crops:
                try:
                    emb = wrapper.extract(crop)
                    embeddings.append(emb)
                except Exception:
                    embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            embeddings = np.stack(embeddings, axis=0)

        for j, item in enumerate(batch_items):
            if not batch_valid[j]:
                continue

            embedding = embeddings[j].astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm < 1e-6:
                skip_zero_norm += 1
                continue

            embedding = embedding / norm
            saved_norm = np.linalg.norm(embedding)
            assert 0.99 < saved_norm < 1.01, (
                f"[Cache normalize check] norm={saved_norm:.4f} at frame={item['frame_id']}"
            )

            cache_key = make_cache_key(
                item["frame_id"], item["camera_id"], item["train_id"], item["bbox_xyxy"]
            )
            filename = make_cache_filename(cache_key)

            cache_data = {
                "frame_id": item["frame_id"],
                "camera_id": item["camera_id"],
                "inst_idx": item["inst_idx"],
                "train_id": item["train_id"],
                "raw_id": item["raw_id"],
                "bbox_xyxy": item["bbox_xyxy"],
                "embedding": embedding,
                "embedding_dim": embedding_dim,
                "model_name": clip_model,
                "checkpoint": os.path.basename(clipreid_checkpoint),
            }

            filepath = os.path.join(output_dir, filename)
            torch.save(cache_data, filepath)
            count += 1

        if (i + batch_size) % 1000 < batch_size:
            print(f"[build_cache] Progress: {min(i + batch_size, len(all_items))}/{len(all_items)}")

    print(f"[build_cache] Done. Saved {count} entries to {output_dir}")
    print(f"[build_cache] Skipped: crop_fail={skip_crop_fail}, zero_norm={skip_zero_norm}")


def build_cache_random(dataset_path, output_dir, embedding_dim=512):
    """Build cache with random embeddings (smoke test only)."""
    annotations = load_annotations(dataset_path)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
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

            embedding = np.random.randn(embedding_dim).astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm < 1e-6:
                continue
            embedding = embedding / norm

            cache_key = make_cache_key(frame_id, camera_id, train_id, bbox_xyxy)
            filename = make_cache_filename(cache_key)

            cache_data = {
                "frame_id": frame_id,
                "camera_id": camera_id,
                "inst_idx": inst_idx,
                "train_id": train_id,
                "raw_id": raw_id,
                "bbox_xyxy": bbox_xyxy,
                "embedding": embedding,
                "embedding_dim": embedding_dim,
                "model_name": "random",
                "checkpoint": "random",
            }

            filepath = os.path.join(output_dir, filename)
            torch.save(cache_data, filepath)
            count += 1

    print(f"[build_cache_random] Done. Saved {count} entries to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build ReID Teacher Feature Cache")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to Wildtrack dataset root")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto)")
    parser.add_argument("--clipreid_checkpoint", type=str, default=None,
                        help="Path to retrained ClipReID checkpoint (.pth)")
    parser.add_argument("--clip_model", type=str, default="ViT-B-16",
                        help="CLIP backbone name (default: ViT-B-16)")
    parser.add_argument("--embedding_dim", type=int, default=512,
                        help="Dimension of teacher embedding")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (cuda or cpu)")
    parser.add_argument("--use_random", action="store_true",
                        help="Use random embeddings (smoke test only)")
    args = parser.parse_args()

    if args.use_random:
        output_dir = args.output_dir or os.path.join(args.dataset_path, "reid_teacher_cache_random")
        build_cache_random(args.dataset_path, output_dir, args.embedding_dim)
    else:
        if args.clipreid_checkpoint is None:
            print("[ERROR] --clipreid_checkpoint is required for real cache building")
            print("  Use --use_random for smoke test only")
            sys.exit(1)

        if not os.path.isfile(args.clipreid_checkpoint):
            print(f"[ERROR] Checkpoint not found: {args.clipreid_checkpoint}")
            sys.exit(1)

        output_dir = args.output_dir or os.path.join(args.dataset_path, "reid_teacher_cache")
        build_cache_real(
            args.dataset_path,
            output_dir,
            args.clipreid_checkpoint,
            args.clip_model,
            args.embedding_dim,
            args.batch_size,
            args.device,
        )


if __name__ == "__main__":
    main()
