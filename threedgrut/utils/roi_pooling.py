# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ROI Pooling Utility for ReID Feature Extraction.

This module implements Region-of-Interest (ROI) pooling for extracting instance-level
embeddings from dense feature maps, which is essential for ReID distillation.

Core Concept:
    ROI pooling bridges the gap between:
    - Dense feature maps (rendered from 3D Gaussians): [D, H, W]
    - Instance embeddings (needed for ReID loss): [D]

The process:
    1. Given a feature map [D, H, W] and a bounding box [xmin, ymin, xmax, ymax]
    2. Extract the region within the bbox from the feature map
    3. Apply average pooling over the region
    4. L2 normalize the result to get a unit vector

This is used in Version A (render feature map + ROI pooling) as the baseline approach.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def roi_pool(
    feature_map: torch.Tensor,
    bbox: torch.Tensor,
    output_size: tuple = (1, 1),
    spatial_scale: float = 1.0,
) -> torch.Tensor:
    """Extract instance embedding from feature map using ROI pooling.

    Args:
        feature_map: Dense feature map, shape [D, H, W] or [B, D, H, W]
        bbox: Bounding box in format [xmin, ymin, xmax, ymax], shape [4] or [B, 4]
        output_size: Output size for ROI pooling, default (1, 1) gives a single vector
        spatial_scale: Scale factor for bbox coordinates

    Returns:
        Instance embedding tensor, shape [D] or [B, D], L2 normalized

    Example:
        >>> feature_map = torch.randn(128, 1088, 1920)  # [D, H, W]
        >>> bbox = torch.tensor([100, 50, 300, 400])    # [xmin, ymin, xmax, ymax]
        >>> embedding = roi_pool(feature_map, bbox)      # [128]
    """
    if feature_map.ndim == 3:
        # [D, H, W] -> [1, D, H, W]
        feature_map = feature_map.unsqueeze(0)
        squeeze_output = True
    else:
        # [B, D, H, W]
        squeeze_output = False

    B, D, H, W = feature_map.shape

    if bbox.ndim == 1:
        # [4] -> [B, 4]
        bbox = bbox.unsqueeze(0).expand(B, -1)

    # Clamp bbox to valid range
    bbox = bbox.clone()
    bbox[:, 0] = torch.clamp(bbox[:, 0], 0, W - 1)  # xmin
    bbox[:, 1] = torch.clamp(bbox[:, 1], 0, H - 1)  # ymin
    bbox[:, 2] = torch.clamp(bbox[:, 2], 1, W)      # xmax
    bbox[:, 3] = torch.clamp(bbox[:, 3], 1, H)      # ymax

    # Convert bbox to ROI format for PyTorch's roi_pool
    # Format: [batch_idx, xmin, ymin, xmax, ymax]
    batch_indices = torch.arange(B, device=bbox.device).float().unsqueeze(1)
    rois = torch.cat([batch_indices, bbox.float()], dim=1)  # [B, 5]

    # Apply ROI pooling
    # PyTorch's roi_pool expects [N, C, H, W] input and rois in [x1, y1, x2, y2] format
    # Output shape: [N, C, output_size[0], output_size[1]]
    pooled = torch.nn.functional.roi_pool(
        feature_map,
        rois,
        output_size=output_size,
        spatial_scale=spatial_scale,
    )  # [B, D, 1, 1]

    # Squeeze to get [B, D]
    pooled = pooled.squeeze(-1).squeeze(-1)

    # L2 normalize
    normalized = F.normalize(pooled, p=2, dim=-1)

    if squeeze_output:
        # [1, D] -> [D]
        return normalized.squeeze(0)
    return normalized


def roi_pool_simple(
    feature_map: torch.Tensor,
    bbox: torch.Tensor,
) -> torch.Tensor:
    """Simple ROI pooling implementation without PyTorch's roi_pool dependency.

    This is a fallback implementation that uses direct tensor slicing and average pooling.

    Args:
        feature_map: Dense feature map, shape [D, H, W] or [B, D, H, W]
        bbox: Bounding box in format [xmin, ymin, xmax, ymax]

    Returns:
        Instance embedding tensor, L2 normalized
    """
    if feature_map.ndim == 3:
        feature_map = feature_map.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, D, H, W = feature_map.shape

    # Ensure bbox is [B, 4]
    if bbox.ndim == 1:
        bbox = bbox.unsqueeze(0).expand(B, -1)

    # Clamp bbox
    bbox = bbox.clone()
    bbox[:, 0] = torch.clamp(bbox[:, 0], 0, W - 1)
    bbox[:, 1] = torch.clamp(bbox[:, 1], 0, H - 1)
    bbox[:, 2] = torch.clamp(bbox[:, 2], bbox[:, 0] + 1, W)
    bbox[:, 3] = torch.clamp(bbox[:, 3], bbox[:, 1] + 1, H)

    # Extract and pool each region
    embeddings = []
    for b in range(B):
        xmin, ymin, xmax, ymax = bbox[b].long()
        region = feature_map[b, :, ymin:ymax, xmin:xmax]  # [D, region_h, region_w]
        pooled = region.mean(dim=(1, 2))  # [D]
        normalized = F.normalize(pooled.unsqueeze(0), p=2, dim=-1)  # [1, D]
        embeddings.append(normalized)

    result = torch.cat(embeddings, dim=0)  # [B, D]

    if squeeze_output:
        return result.squeeze(0)  # [D]
    return result


def batch_roi_pool(
    feature_maps: torch.Tensor,
    bboxes: torch.Tensor,
) -> torch.Tensor:
    """Batch ROI pooling for multiple instances.

    Args:
        feature_maps: List of feature maps [D, H, W] or single tensor [B, D, H, W]
        bboxes: List of bboxes or tensor [N, 4] where N is total number of instances

    Returns:
        Tensor of embeddings [N, D]
    """
    if isinstance(feature_maps, (list, tuple)):
        # Assume all feature maps have same spatial dimensions
        D = feature_maps[0].shape[0]
        H, W = feature_maps[0].shape[1:]
        B = len(feature_maps)

        # Stack feature maps
        feature_maps = torch.stack(feature_maps, dim=0)  # [B, D, H, W]
    else:
        B, D, H, W = feature_maps.shape

    N = bboxes.shape[0]

    # For simplicity, process one at a time
    embeddings = []
    for n in range(N):
        bbox = bboxes[n]
        # Assume batch_idx = n % B for simplicity (round-robin)
        batch_idx = n % B
        embedding = roi_pool_simple(
            feature_maps[batch_idx:batch_idx+1],
            bbox.unsqueeze(0),
        )
        embeddings.append(embedding)

    return torch.cat(embeddings, dim=0)  # [N, D]
