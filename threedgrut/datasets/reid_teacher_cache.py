# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ReID Teacher Feature Cache Loader.

This module provides a cache loader for pre-extracted ReID teacher features.
The cache is generated offline (see tools/build_reid_teacher_cache.py).

Cache Structure:
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

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple


class ReidTeacherCache:
    """Loader for pre-extracted ReID teacher feature cache."""

    def __init__(self, dataset_path: str, cache_dir: str = None):
        """Initialize the teacher cache loader.

        Args:
            dataset_path: Path to the dataset root (contains annotations_remapped, etc.)
            cache_dir: Optional custom cache directory. If None, uses {dataset_path}/reid_teacher_cache/
        """
        self.dataset_path = dataset_path
        self.cache_dir = cache_dir or os.path.join(dataset_path, "reid_teacher_cache")
        self._cache_index: Dict[Tuple, str] = {}  # (frame_id, camera_id, inst_idx) -> file_path
        self._load_cache_index()

    def _load_cache_index(self) -> None:
        """Build index of available cache files."""
        if not os.path.exists(self.cache_dir):
            print(f"⚠️  Warning: Teacher cache directory not found: {self.cache_dir}")
            return

        for filename in os.listdir(self.cache_dir):
            if not filename.endswith(".pt"):
                continue

            # Parse filename: {frame_id:08d}_{camera_id}_{inst_idx}.pt
            parts = filename.replace(".pt", "").split("_")
            if len(parts) != 3:
                continue

            try:
                frame_id = int(parts[0])
                camera_id = parts[1]
                inst_idx = int(parts[2])
            except ValueError:
                continue

            cache_key = (frame_id, camera_id, inst_idx)
            self._cache_index[cache_key] = os.path.join(self.cache_dir, filename)

        print(f"✅ Loaded teacher cache: {len(self._cache_index)} entries from {self.cache_dir}")

    def get(self, cache_key: Tuple[int, str, int]) -> Optional[Dict]:
        """Get teacher embedding for a specific (frame_id, camera_id, inst_idx).

        Args:
            cache_key: Tuple of (frame_id, camera_id, inst_idx)

        Returns:
            Dict containing frame_id, camera_id, inst_idx, bbox_xyxy, raw_id, train_id, embedding
            or None if not found
        """
        if cache_key not in self._cache_index:
            return None

        import torch

        file_path = self._cache_index[cache_key]
        try:
            data = torch.load(file_path, map_location="cpu", weights_only=False)
            return data
        except Exception as e:
            print(f"⚠️  Warning: Failed to load cache {file_path}: {e}")
            return None

    def __contains__(self, cache_key: Tuple[int, str, int]) -> bool:
        """Check if cache entry exists."""
        return cache_key in self._cache_index

    def __len__(self) -> int:
        """Return number of cache entries."""
        return len(self._cache_index)

    def get_all_for_frame(self, frame_id: int) -> List[Dict]:
        """Get all teacher embeddings for a specific frame.

        Args:
            frame_id: Frame index

        Returns:
            List of cache entries for all cameras and instances in the frame
        """
        results = []
        for cache_key, file_path in self._cache_index.items():
            if cache_key[0] == frame_id:
                data = self.get(cache_key)
                if data is not None:
                    results.append(data)
        return results

    def validate_alignment(self, annotations_remapped_dir: str) -> bool:
        """Validate that cache entries align with annotations_remapped.

        Args:
            annotations_remapped_dir: Path to annotations_remapped directory

        Returns:
            True if alignment is valid, False otherwise
        """
        import json

        if not os.path.exists(annotations_remapped_dir):
            print(f"⚠️  Warning: annotations_remapped not found at {annotations_remapped_dir}")
            return False

        errors = []
        for annot_file in os.listdir(annotations_remapped_dir):
            if not annot_file.endswith(".json"):
                continue

            frame_id = int(annot_file.replace(".json", ""))
            annot_path = os.path.join(annotations_remapped_dir, annot_file)

            with open(annot_path, "r") as f:
                annotations = json.load(f)

            for inst_idx, person in enumerate(annotations):
                raw_id = person.get("raw_id") or person.get("personID", 0)
                train_id = person.get("train_id")

                # Skip invalid IDs
                if raw_id == 0 or train_id is None:
                    continue

                # Check each camera
                views = person.get("views", [])
                for view in views:
                    camera_id = f"C{view.get('viewNum', 0)}"
                    if view.get("xmin") == -1:  # Invisible in this view
                        continue

                    cache_key = (frame_id, camera_id, inst_idx)
                    if cache_key not in self._cache_index:
                        errors.append(f"Missing cache for {cache_key}")
                    else:
                        # Validate train_id matches
                        data = self.get(cache_key)
                        if data and data.get("train_id") != train_id:
                            errors.append(
                                f"Train ID mismatch for {cache_key}: "
                                f"expected {train_id}, got {data.get('train_id')}"
                            )

        if errors:
            print(f"⚠️  Cache alignment validation failed with {len(errors)} errors:")
            for error in errors[:10]:
                print(f"  - {error}")
            return False

        print("✅ Cache alignment validation passed")
        return True
