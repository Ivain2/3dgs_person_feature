# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ROI Pooling Utility for ReID Feature Extraction.

Supports multiple pooling modes:
  - mean: simple average over bbox region
  - opacity: opacity-weighted average (foreground-aware)
  - topk_opacity: top-k opacity pixels only

When detach_opacity_weight=True, the opacity map is detached before
being used as a pooling weight, preventing ReID loss gradients from
flowing back through opacity/geometry parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def scale_bbox_to_render(bbox_xyxy, src_w, src_h, dst_w, dst_h):
    """Scale bbox from source image resolution to render resolution.

    Args:
        bbox_xyxy: [x1, y1, x2, y2] tensor or list in source image coordinates
        src_w: source image width (e.g., 1920)
        src_h: source image height (e.g., 1080)
        dst_w: render width (e.g., 480)
        dst_h: render height (e.g., 272)

    Returns:
        torch.Tensor: [x1, y1, x2, y2] in render coordinates
    """
    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)

    if isinstance(bbox_xyxy, (list, tuple)):
        x1, y1, x2, y2 = bbox_xyxy
    else:
        x1 = bbox_xyxy[0].item()
        y1 = bbox_xyxy[1].item()
        x2 = bbox_xyxy[2].item()
        y2 = bbox_xyxy[3].item()

    x1_scaled = x1 * scale_x
    y1_scaled = y1 * scale_y
    x2_scaled = x2 * scale_x
    y2_scaled = y2 * scale_y

    return torch.tensor([x1_scaled, y1_scaled, x2_scaled, y2_scaled], dtype=torch.float32)


def _clamp_bbox(bbox, H, W):
    xmin = max(0, int(bbox[0].item()))
    ymin = max(0, int(bbox[1].item()))
    xmax = min(W, max(xmin + 1, int(bbox[2].item())))
    ymax = min(H, max(ymin + 1, int(bbox[3].item())))
    return xmin, ymin, xmax, ymax


def roi_pool_mean(feature_map, bbox):
    D, H, W = feature_map.shape
    xmin, ymin, xmax, ymax = _clamp_bbox(bbox, H, W)
    region = feature_map[:, ymin:ymax, xmin:xmax]
    pooled = region.mean(dim=(1, 2))
    return F.normalize(pooled, p=2, dim=0)


def roi_pool_opacity(feature_map, opacity_map, bbox, eps=1e-4, min_alpha_sum=0.01, detach_opacity_weight=False):
    D, H, W = feature_map.shape
    xmin, ymin, xmax, ymax = _clamp_bbox(bbox, H, W)

    region = feature_map[:, ymin:ymax, xmin:xmax]        # [D, rh, rw]
    alpha_region = opacity_map[ymin:ymax, xmin:xmax]      # [rh, rw]

    if detach_opacity_weight:
        alpha_region = alpha_region.detach()

    alpha_sum = alpha_region.sum()
    if alpha_sum.item() < min_alpha_sum:
        return None, {"alpha_sum": alpha_sum.item(), "skipped": True}

    weighted = (region * alpha_region.unsqueeze(0)).sum(dim=(1, 2))  # [D]
    pooled = weighted / (alpha_sum + eps)

    alpha_mean_val = alpha_region.mean().item()
    alpha_max_val = alpha_region.max().item()
    raw_norm = region.norm(p=2, dim=0).mean().item()
    weighted_norm = pooled.norm(p=2).item()

    normalized = F.normalize(pooled, p=2, dim=0)

    stats = {
        "alpha_sum": alpha_sum.item(),
        "alpha_mean": alpha_mean_val,
        "alpha_max": alpha_max_val,
        "raw_feature_norm_mean": raw_norm,
        "weighted_feature_norm_mean": weighted_norm,
        "skipped": False,
    }
    return normalized, stats


def roi_pool_topk_opacity(feature_map, opacity_map, bbox, topk_ratio=0.3, eps=1e-4, min_alpha_sum=0.01, detach_opacity_weight=False):
    D, H, W = feature_map.shape
    xmin, ymin, xmax, ymax = _clamp_bbox(bbox, H, W)

    region = feature_map[:, ymin:ymax, xmin:xmax]        # [D, rh, rw]
    alpha_region = opacity_map[ymin:ymax, xmin:xmax]      # [rh, rw]

    if detach_opacity_weight:
        alpha_region = alpha_region.detach()

    rh, rw = alpha_region.shape
    n_pixels = rh * rw
    k = max(1, int(n_pixels * topk_ratio))

    alpha_flat = alpha_region.reshape(-1)                  # [N]
    _, topk_idx = alpha_flat.topk(k)
    mask = torch.zeros_like(alpha_flat)
    mask[topk_idx] = 1.0
    mask = mask.reshape(rh, rw)                            # [rh, rw]

    masked_alpha = alpha_region * mask
    alpha_sum = masked_alpha.sum()

    if alpha_sum.item() < min_alpha_sum:
        return None, {"alpha_sum": alpha_sum.item(), "skipped": True, "topk_k": k}

    weighted = (region * masked_alpha.unsqueeze(0)).sum(dim=(1, 2))
    pooled = weighted / (alpha_sum + eps)

    alpha_mean_val = masked_alpha[mask.bool()].mean().item() if mask.sum() > 0 else 0.0
    alpha_max_val = alpha_region.max().item()
    raw_norm = region.norm(p=2, dim=0).mean().item()
    weighted_norm = pooled.norm(p=2).item()

    normalized = F.normalize(pooled, p=2, dim=0)

    stats = {
        "alpha_sum": alpha_sum.item(),
        "alpha_mean": alpha_mean_val,
        "alpha_max": alpha_max_val,
        "raw_feature_norm_mean": raw_norm,
        "weighted_feature_norm_mean": weighted_norm,
        "topk_k": k,
        "topk_ratio": topk_ratio,
        "skipped": False,
    }
    return normalized, stats


def roi_pool(feature_map, bbox, opacity_map=None, pooling="mean", topk_ratio=0.3,
             eps=1e-4, min_alpha_sum=0.01, detach_opacity_weight=False):
    if pooling == "mean" or opacity_map is None:
        result = roi_pool_mean(feature_map, bbox)
        return result, {"skipped": False, "pooling": "mean"}
    elif pooling == "opacity":
        return roi_pool_opacity(feature_map, opacity_map, bbox, eps, min_alpha_sum, detach_opacity_weight)
    elif pooling == "topk_opacity":
        return roi_pool_topk_opacity(feature_map, opacity_map, bbox, topk_ratio, eps, min_alpha_sum, detach_opacity_weight)
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")
