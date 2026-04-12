# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from fused_ssim import fused_ssim


@torch.cuda.nvtx.range("l1_loss")
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


@torch.cuda.nvtx.range("l2_loss")
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


@torch.cuda.nvtx.range("ssim")
def ssim(img1, img2, window_size=11, size_average=True):
    # predicted_image, gt_image: [BS, CH, H, W], predicted_image is differentiable
    return fused_ssim(img1, img2, padding="valid")


def cosine_distillation_loss(query: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Cosine distillation loss: 1 - cosine_similarity.

    This loss measures the alignment between student (query) and teacher (target) embeddings.
    Lower value means better alignment (1 = opposite, 0 = identical).

    Args:
        query: Student embedding tensor (will be normalized internally)
        target: Teacher embedding tensor (will be normalized internally)
        dim: Dimension along which to compute cosine similarity

    Returns:
        Mean cosine distillation loss (scalar tensor)
    """
    # Normalize both embeddings to unit vectors
    query_norm = torch.nn.functional.normalize(query, p=2, dim=dim)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=dim)

    # Cosine similarity: ranges from -1 (opposite) to 1 (identical)
    cosine_sim = (query_norm * target_norm).sum(dim=dim)

    # Distillation loss: 1 - cosine_similarity
    # Ranges from 0 (identical) to 2 (opposite)
    loss = 1.0 - cosine_sim

    return loss.mean()
