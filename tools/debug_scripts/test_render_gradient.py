#!/usr/bin/env python3
"""Test gradient flow through rendering"""

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
import os
from threedgrut.trainer import Trainer3DGRUT

config_dir = os.path.abspath('configs')
config_name = 'apps/wildtrack_full_3dgut'

with initialize_config_dir(config_dir=config_dir, version_base=None):
    conf = compose(config_name=config_name)

conf.model.person_feature_dim = 512
conf.loss.use_reid = True

trainer = Trainer3DGRUT(conf)

# Get a batch
batch_data = next(iter(trainer.train_dataloader))
gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)

print(f"Instances: {len(gpu_batch.instances)}")

# Forward
trainer.model.train()
trainer.model.zero_grad()

person_feature = trainer.model.get_person_feature()
print(f"\nBefore forward:")
print(f"  person_feature.requires_grad: {person_feature.requires_grad}")

# Forward with person_feature rendering
render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
person_feature_map = render_out.get('person_feature_map')

print(f"\nAfter forward:")
print(f"  person_feature_map.shape: {person_feature_map.shape}")
print(f"  person_feature_map.requires_grad: {person_feature_map.requires_grad}")
print(f"  person_feature_map.grad_fn: {person_feature_map.grad_fn}")

# Check if person_feature_map has any connection to person_feature
# by checking if modifying person_feature changes person_feature_map
print(f"\nChecking gradient connection...")

# Create a simple loss on person_feature_map
loss_map = person_feature_map.abs().mean()
print(f"  loss_map: {loss_map.item():.6f}")

# Backward
loss_map.backward()

print(f"\nAfter backward (with person_feature_map loss):")
print(f"  person_feature.grad is None: {person_feature.grad is None}")
if person_feature.grad is not None:
    print(f"  person_feature.grad.abs().mean(): {person_feature.grad.abs().mean().item():.6f}")
    print(f"  person_feature.grad.abs().max(): {person_feature.grad.abs().max().item():.6f}")
    
    # Check if gradient is sparse (only some Gaussians contributed)
    nonzero_grad = (person_feature.grad.abs() > 1e-8).sum().item()
    total_elements = person_feature.grad.numel()
    print(f"  Nonzero gradient elements: {nonzero_grad} / {total_elements} ({100*nonzero_grad/total_elements:.2f}%)")
