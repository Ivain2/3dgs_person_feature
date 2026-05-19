#!/usr/bin/env python3
"""Test: single vs multiple renders gradient comparison"""
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
trainer.model.load_state_dict(torch.load('runs/phase10C_topk_detach_lam005_lr1e4_stable/latest.pth', map_location=trainer.device)['model_state_dict'], strict=False)

pf = trainer.model.get_person_feature()
dataset = trainer.train_dataset

# Test 1: Single render, direct sum
print("=" * 60)
print("Test 1: Single render, direct sum")
print("=" * 60)
trainer.model.zero_grad()
gpu_batch = dataset.get_gpu_batch_with_intrinsics(dataset[0])
render_out = trainer.model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
fmap = render_out['person_feature_map']
print(f"feature_map.requires_grad: {fmap.requires_grad}")
print(f"feature_map.mean: {fmap.mean().item():.6f}")
loss = fmap.sum()
loss.backward()
print(f"person_feature.grad_norm: {pf.grad.norm().item():.6e}")

# Test 2: Single render, then second render, backward second
print("\n" + "=" * 60)
print("Test 2: Two renders, backward second")
print("=" * 60)
trainer.model.zero_grad()
gpu_batch1 = dataset.get_gpu_batch_with_intrinsics(dataset[0])
render_out1 = trainer.model(gpu_batch1, train=False, frame_id=0, render_person_feature=True)
fmap1 = render_out1['person_feature_map']
print(f"Render 1 done")

trainer.model.zero_grad()
gpu_batch2 = dataset.get_gpu_batch_with_intrinsics(dataset[0])
render_out2 = trainer.model(gpu_batch2, train=False, frame_id=0, render_person_feature=True)
fmap2 = render_out2['person_feature_map']
print(f"Render 2 done, fmap.requires_grad: {fmap2.requires_grad}")

loss2 = fmap2.sum()
loss2.backward()
print(f"person_feature.grad_norm after 2nd render backward: {pf.grad.norm().item():.6e}")

# Test 3: Two renders, sum both, backward
print("\n" + "=" * 60)
print("Test 3: Two renders, sum both, backward")
print("=" * 60)
trainer.model.zero_grad()
gpu_batch3a = dataset.get_gpu_batch_with_intrinsics(dataset[0])
render_out3a = trainer.model(gpu_batch3a, train=False, frame_id=0, render_person_feature=True)
fmap3a = render_out3a['person_feature_map']

trainer.model.zero_grad()
gpu_batch3b = dataset.get_gpu_batch_with_intrinsics(dataset[1])
render_out3b = trainer.model(gpu_batch3b, train=False, frame_id=0, render_person_feature=True)
fmap3b = render_out3b['person_feature_map']

loss3 = fmap3a.sum() + fmap3b.sum()
loss3.backward()
print(f"person_feature.grad_norm after both renders backward: {pf.grad.norm().item():.6e}")

# Test 4: Two renders, use first one for backward
print("\n" + "=" * 60)
print("Test 4: Two renders, use FIRST for backward")
print("=" * 60)
trainer.model.zero_grad()
gpu_batch4a = dataset.get_gpu_batch_with_intrinsics(dataset[0])
render_out4a = trainer.model(gpu_batch4a, train=False, frame_id=0, render_person_feature=True)
fmap4a = render_out4a['person_feature_map']
saved_for_backward = fmap4a.sum()

trainer.model.zero_grad()
gpu_batch4b = dataset.get_gpu_batch_with_intrinsics(dataset[1])
render_out4b = trainer.model(gpu_batch4b, train=False, frame_id=0, render_person_feature=True)
fmap4b = render_out4b['person_feature_map']

print(f"Saved fmap4a.sum() before second render")
print(f"Second render done")

saved_for_backward.backward()
print(f"person_feature.grad_norm after backward of FIRST render: {pf.grad.norm().item():.6e}")
