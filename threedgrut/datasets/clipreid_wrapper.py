# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ClipReID Teacher Wrapper.

Loads a retrained ClipReID checkpoint and extracts retrieval embeddings
from cropped person images. Used by build_reid_teacher_cache.py to
pre-extract teacher features offline.

Key design decisions:
  - Uses ClipReID's modified CLIP visual encoder (with resolution params)
  - Loads image_encoder weights from retrained ClipReID checkpoint
  - Output is `image_features_proj[:,0]` for ViT-B/16, shape [B, 512]
  - This is the CLIP projected visual output, NOT logits, NOT BN features
  - L2 normalization is done inside extract(), so cache always stores unit vectors
  - Model is frozen (eval mode, no grad)
  - Supports both CPU and GPU inference
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

_CLIPREID_REPO = "/data02/zhangrunxiang/CLIP-ReID-master"


class ClipReIDWrapper:
    """Wrapper for ClipReID teacher model inference.

    Loads the CLIP visual encoder (modified by ClipReID with resolution params)
    with weights from a retrained ClipReID checkpoint. Only the image_encoder
    (visual) weights are loaded; the text encoder and prompt learner are not
    needed for embedding extraction.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "ViT-B-16",
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name

        if model_name == "ViT-B-16":
            self.embedding_dim = 512
        elif model_name == "RN50":
            self.embedding_dim = 1024
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model = self._load_model(checkpoint_path, model_name)
        self.transform = self._build_transform()

    def _load_model(self, checkpoint_path, model_name):
        sys.path.insert(0, _CLIPREID_REPO)
        from model.clip.clip import _download, _MODELS
        from model.clip.model import build_model

        url = _MODELS[model_name]
        model_path = _download(url)

        try:
            model_jit = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        if state_dict is None:
            state_dict = model_jit.state_dict()

        h_resolution = int((256 - 16) // 16 + 1)
        w_resolution = int((128 - 16) // 16 + 1)
        vision_stride_size = 16

        clip_model = build_model(
            state_dict, h_resolution, w_resolution, vision_stride_size
        )

        if os.path.isfile(checkpoint_path):
            ckpt_state = torch.load(checkpoint_path, map_location="cpu")
            image_encoder_state = {}
            for k, v in ckpt_state.items():
                clean_k = k.replace("module.", "")
                if clean_k.startswith("image_encoder."):
                    new_key = clean_k[len("image_encoder."):]
                    image_encoder_state[new_key] = v

            if image_encoder_state:
                missing, unexpected = clip_model.visual.load_state_dict(
                    image_encoder_state, strict=False
                )
                print(f"[ClipReIDWrapper] Loaded image_encoder from {checkpoint_path}")
                if missing:
                    print(f"[ClipReIDWrapper] Missing keys ({len(missing)}): {missing[:5]}...")
                if unexpected:
                    print(f"[ClipReIDWrapper] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            else:
                print(f"[ClipReIDWrapper] WARNING: No image_encoder keys found in checkpoint")
                print(f"[ClipReIDWrapper] Using original CLIP weights (no retrained encoder)")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        clip_model = clip_model.to(self.device)
        if self.device == "cpu":
            clip_model.float()

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def _build_transform(self):
        return Compose([
            Resize((256, 128), interpolation=Image.BICUBIC),
            CenterCrop((256, 128)),
            lambda img: img.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def extract(self, image_crop_rgb: np.ndarray) -> np.ndarray:
        """Extract retrieval embedding from a single cropped person image.

        Args:
            image_crop_rgb: [H, W, 3] RGB numpy array (uint8 or float [0,1])

        Returns:
            [D] float32 L2-normalized embedding (unit vector)
        """
        if image_crop_rgb.dtype != np.uint8:
            if image_crop_rgb.max() <= 1.0:
                image_crop_rgb = (image_crop_rgb * 255).astype(np.uint8)
            else:
                image_crop_rgb = image_crop_rgb.astype(np.uint8)

        img = Image.fromarray(image_crop_rgb)
        img_t = self.transform(img).unsqueeze(0).to(self.device)

        emb = self.model.encode_image(img_t)
        if isinstance(emb, (tuple, list)):
            emb = emb[2][:, 0]
        emb = F.normalize(emb, p=2, dim=-1)

        return emb.squeeze(0).float().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract retrieval embeddings from a batch of cropped person images.

        Args:
            crops: list of [H, W, 3] RGB numpy arrays

        Returns:
            [N, D] float32 L2-normalized embeddings
        """
        tensors = []
        for crop in crops:
            if crop.dtype != np.uint8:
                if crop.max() <= 1.0:
                    crop = (crop * 255).astype(np.uint8)
                else:
                    crop = crop.astype(np.uint8)
            img = Image.fromarray(crop)
            tensors.append(self.transform(img))

        batch_t = torch.stack(tensors, dim=0).to(self.device)
        emb = self.model.encode_image(batch_t)
        if isinstance(emb, (tuple, list)):
            emb = emb[2][:, 0]
        emb = F.normalize(emb, p=2, dim=-1)

        return emb.float().cpu().numpy().astype(np.float32)
