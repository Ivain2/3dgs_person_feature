#!/usr/bin/env python3
"""Test: verify person_feature gradient path"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from threedgrut.trainer import Trainer3DGRUT

conf_dir = os.path.join(REPO_ROOT, "configs")
with initialize_config_dir(config_dir=conf_dir, version_base=None):
    conf = compose(config_name="apps/wildtrack_full_3dgut")

conf.model.person_feature_dim = 512
conf.model.person_feature_lr = 1e-6
conf.loss.use_reid = True
conf.loss.lambda_reid = 0.0

trainer = Trainer3DGRUT(conf)
pf = trainer.model.get_person_feature()

# Test: render + direct feature_map sum
dataset = trainer.train_dataset
idx = 0
raw_batch = dataset[idx]
gpu_batch = dataset.get_gpu_batch_with_intrinsics(raw_batch)

# First render
render_out = trainer.model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
fmap1 = render_out['person_feature_map']
print(f"Render 1: feature_map.requires_grad={fmap1.requires_grad}, grad_fn={fmap1.grad_fn}")

trainer.model.zero_grad()
loss1 = fmap1.sum()
print(f"Loss 1: {loss1.item():.4f}")
loss1.backward()
print(f"After backward 1: pf.grad norm = {pf.grad.norm().item():.6e}")

# Second render (fresh)
trainer.model.zero_grad()
gpu_batch2 = dataset.get_gpu_batch_with_intrinsics(dataset[idx])
render_out2 = trainer.model(gpu_batch2, train=False, frame_id=0, render_person_feature=True)
fmap2 = render_out2['person_feature_map']
print(f"\nRender 2: feature_map.requires_grad={fmap2.requires_grad}, grad_fn={fmap2.grad_fn}")

loss2 = fmap2[:, 100:110, 100:110].sum()
print(f"Loss 2: {loss2.item():.4f}")
loss2.backward()
print(f"After backward 2: pf.grad norm = {pf.grad.norm().item():.6e}")

# Check person_feature value
print(f"\nperson_feature stats: mean={pf.mean().item():.6f}, std={pf.std().item():.6f}, requires_grad={pf.requires_grad}")
print(f"person_feature shape: {pf.shape}")
