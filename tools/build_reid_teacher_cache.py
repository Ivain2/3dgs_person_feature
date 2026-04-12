# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Build ReID Teacher Feature Cache.

This tool extracts ReID features from images using a frozen teacher model (e.g., CLIP-ReID)
and saves them to a cache for fast training.

Usage:
    python tools/build_reid_teacher_cache.py --dataset_path /path/to/Wildtrack --feature_dim 128

Cache Output Structure:
    {dataset_path}/reid_teacher_cache/
        {frame_id:08d}_{camera_id}_{inst_idx}.pt

Each .pt file contains:
    {
        "frame_id": int,
        "camera_id": str,           # "C0" ~ "C6"
        "inst_idx": int,            # instance index in frame
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "raw_id": int,             # original person ID
        "train_id": int,           # remapped identity ID (0-based)
        "embedding": np.ndarray     # [D] ReID feature vector
    }
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from threedgrut.datasets.dataset_wildtrack import WildtrackDataset
from threedgrut.datasets.reid_teacher_cache import ReidTeacherCache


class ReidTeacherExtractor:
    """Teacher ReID feature extractor (placeholder for actual CLIP-ReID model).

    In the current implementation, this is a placeholder that generates random features.
    To use actual CLIP-ReID, replace the extract_feature method with your teacher model.
    """

    def __init__(self, feature_dim: int = 128, device: str = "cuda"):
        self.feature_dim = feature_dim
        self.device = device
        print(f"🔖 Initialized ReID Teacher Extractor (placeholder) with feature_dim={feature_dim}")

    def extract_feature(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract ReID feature from image crop.

        Args:
            image: Full image [H, W, 3] in RGB format
            bbox: Bounding box [xmin, ymin, xmax, ymax]

        Returns:
            Feature vector [D] (L2 normalized)
        """
        # Crop image using bbox
        xmin, ymin, xmax, ymax = [int(v) for v in bbox]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image.shape[1], xmax)
        ymax = min(image.shape[0], ymax)

        if xmax <= xmin or ymax <= ymin:
            return np.zeros(self.feature_dim, dtype=np.float32)

        crop = image[ymin:ymax, xmin:xmax]

        if crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Resize crop to fixed size for consistent feature extraction
        crop_resized = cv2.resize(crop, (128, 256))  # Person ReID standard size

        # Convert to tensor
        crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
        crop_tensor = crop_tensor.unsqueeze(0).to(self.device)  # [1, 3, 256, 128]

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        crop_tensor = (crop_tensor - mean) / std

        # TODO: Replace this with actual CLIP-ReID model inference
        # Example:
        # with torch.no_grad():
        #     features = self.clip_reid_model(crop_tensor)
        # features = features.squeeze().cpu().numpy()

        # Placeholder: generate random feature (for testing only)
        features = np.random.randn(self.feature_dim).astype(np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)

        return features


def build_cache(
    dataset_path: str,
    output_dir: str = None,
    feature_dim: int = 128,
    device: str = "cuda",
    overwrite: bool = False,
):
    """Build teacher feature cache from dataset annotations.

    Args:
        dataset_path: Path to Wildtrack dataset root
        output_dir: Output directory for cache. If None, uses {dataset_path}/reid_teacher_cache
        feature_dim: Dimension of ReID features
        device: Device to run extraction on
        overwrite: Whether to overwrite existing cache entries
    """
    if output_dir is None:
        output_dir = os.path.join(dataset_path, "reid_teacher_cache")

    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Cache output directory: {output_dir}")

    dataset = WildtrackDataset(
        dataset_path=dataset_path,
        split="train",
        load_teacher_cache=False,  # Don't load cache during cache building
    )

    id_map = dataset.id_map
    raw_to_new = id_map.get("raw_to_new", {})

    extractor = ReidTeacherExtractor(feature_dim=feature_dim, device=device)

    print(f"🔨 Building teacher feature cache for {len(dataset)} samples...")

    cache_count = 0
    skip_count = 0

    for idx in tqdm(range(len(dataset)), desc="Extracting features"):
        batch = dataset[idx]
        frame_idx = batch["frame_idx"]
        cam_id = batch["camera_id"]
        image = (batch["rgb"] * 255).astype(np.uint8)  # Convert back to [0, 255] RGB

        instances = batch.get("instances", [])
        if not instances:
            continue

        for inst_idx, instance in enumerate(instances):
            cache_filename = f"{frame_idx:08d}_{cam_id}_{inst_idx}.pt"
            cache_path = os.path.join(output_dir, cache_filename)

            if os.path.exists(cache_path) and not overwrite:
                skip_count += 1
                continue

            raw_id = instance.get("raw_id", 0)
            train_id = instance.get("train_id", -1)
            bbox = instance.get("bbox_xyxy")

            if bbox is None:
                continue

            embedding = extractor.extract_feature(image, bbox)

            cache_entry = {
                "frame_id": frame_idx,
                "camera_id": cam_id,
                "inst_idx": inst_idx,
                "bbox_xyxy": bbox,
                "raw_id": raw_id,
                "train_id": train_id,
                "embedding": embedding,
            }

            torch.save(cache_entry, cache_path)
            cache_count += 1

    print(f"\n✅ Cache building complete!")
    print(f"   New entries: {cache_count}")
    print(f"   Skipped (existing): {skip_count}")
    print(f"   Total cache size: {cache_count + skip_count}")

    return output_dir


def validate_cache(dataset_path: str, cache_dir: str = None):
    """Validate that cache entries align with annotations.

    Args:
        dataset_path: Path to Wildtrack dataset root
        cache_dir: Cache directory. If None, uses {dataset_path}/reid_teacher_cache

    Returns:
        True if validation passes, False otherwise
    """
    if cache_dir is None:
        cache_dir = os.path.join(dataset_path, "reid_teacher_cache")

    cache = ReidTeacherCache(dataset_path, cache_dir=cache_dir)

    annotations_dir = os.path.join(dataset_path, "annotations_remapped")
    if not os.path.exists(annotations_dir):
        print(f"❌ annotations_remapped not found at {annotations_dir}")
        return False

    result = cache.validate_alignment(annotations_dir)

    if result:
        print(f"✅ Cache validation passed!")
    else:
        print(f"❌ Cache validation FAILED!")

    return result


def main():
    parser = argparse.ArgumentParser(description="Build ReID teacher feature cache")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to Wildtrack dataset root",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=128,
        help="Dimension of ReID features (default: 128)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for cache (default: {dataset_path}/reid_teacher_cache)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run extraction on (default: cuda)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache entries",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate existing cache, don't rebuild",
    )

    args = parser.parse_args()

    if args.validate_only:
        validate_cache(args.dataset_path, args.output_dir)
        return

    output_dir = build_cache(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        feature_dim=args.feature_dim,
        device=args.device,
        overwrite=args.overwrite,
    )

    validate_cache(args.dataset_path, output_dir)


if __name__ == "__main__":
    main()
