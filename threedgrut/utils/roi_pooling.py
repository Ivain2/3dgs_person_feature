# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ROI Pooling Utility for ReID Feature Extraction.

ROI pooling = pixel -> instance aggregation, used to align teacher embedding.
Given a dense feature map [D, H, W] rendered from 3D Gaussians and a 2D bbox,
extract an instance-level embedding [D] by average pooling over the bbox region.

This bridges the gap between:
  - Dense 2D feature maps (rendered via alpha compositing from 3D Gaussians)
  - Instance-level embeddings (needed for ReID distillation loss)

Uses pure PyTorch tensor slicing (no torchvision dependency).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def roi_pool(feature_map: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    """Extract instance embedding from feature map using ROI average pooling.

    Args:
        feature_map: [D, H, W] dense feature map rendered from 3D Gaussians
        bbox: [4] bounding box in format [xmin, ymin, xmax, ymax]

    Returns:
        [D] L2-normalized instance embedding, differentiable w.r.t. feature_map
    """
    D, H, W = feature_map.shape

    xmin = max(0, int(bbox[0].item()))
    ymin = max(0, int(bbox[1].item()))
    xmax = min(W, max(xmin + 1, int(bbox[2].item())))
    ymax = min(H, max(ymin + 1, int(bbox[3].item())))

    region = feature_map[:, ymin:ymax, xmin:xmax]  # [D, region_h, region_w]
    pooled = region.mean(dim=(1, 2))  # [D]
    normalized = F.normalize(pooled, p=2, dim=0)  # [D]

    return normalized
