# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared Cache Key Helper for Teacher Feature Cache.

Stable key = (frame_id, camera_id, train_id, x1_px, y1_px, x2_px, y2_px)

Why not (frame_id, camera_id, inst_idx):
  - inst_idx depends on annotation JSON traversal order, which is unstable
  - annotation updates (add/remove person) shift all subsequent indices
  - train_id + bbox uniquely identifies a person projection in a frame+camera

Both build_reid_teacher_cache.py and dataset_wildtrack.py MUST use this
helper to ensure key consistency.
"""

from __future__ import annotations

from typing import Tuple


def make_cache_key(
    frame_id: int,
    camera_id: str,
    train_id: int,
    bbox_xyxy: list,
) -> Tuple[int, str, int, int, int, int, int]:
    """Build a stable cache key from annotation fields.

    Args:
        frame_id: frame number
        camera_id: "C1"~"C7"
        train_id: 0-based continuous identity
        bbox_xyxy: [xmin, ymin, xmax, ymax] in pixel coords

    Returns:
        Tuple of (frame_id, camera_id, train_id, x1, y1, x2, y2)
    """
    x1 = int(bbox_xyxy[0])
    y1 = int(bbox_xyxy[1])
    x2 = int(bbox_xyxy[2])
    y2 = int(bbox_xyxy[3])
    return (frame_id, camera_id, train_id, x1, y1, x2, y2)


def make_cache_filename(key: tuple) -> str:
    """Convert cache key to filename.

    Args:
        key: output of make_cache_key()

    Returns:
        filename string like "00000123_C1_0042_0100_0200_0300_0400.pt"
    """
    frame_id, camera_id, train_id, x1, y1, x2, y2 = key
    return f"{frame_id:08d}_{camera_id}_{train_id:04d}_{x1:04d}_{y1:04d}_{x2:04d}_{y2:04d}.pt"


def parse_cache_filename(filename: str) -> Tuple[int, str, int, int, int, int, int]:
    """Parse cache filename back to key tuple.

    Args:
        filename: e.g. "00000123_C1_0042_0100_0200_0300_0400.pt"

    Returns:
        cache key tuple
    """
    name = filename.replace(".pt", "")
    parts = name.split("_")
    if len(parts) != 7:
        raise ValueError(f"Invalid cache filename: {filename}")
    frame_id = int(parts[0])
    camera_id = parts[1]
    train_id = int(parts[2])
    x1 = int(parts[3])
    y1 = int(parts[4])
    x2 = int(parts[5])
    y2 = int(parts[6])
    return (frame_id, camera_id, train_id, x1, y1, x2, y2)
