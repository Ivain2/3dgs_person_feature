#!/usr/bin/env python3
"""Quick test: run only test1 (pooled_feature_sum) to check gradient"""
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

# Quick test: direct sum
trainer.model.zero_grad()
test_val = pf.sum() * 0.0001
test_val.backward()
print(f"Direct person_feature.sum() grad_norm: {pf.grad.norm().item():.6e}")

# Test: render + ROI pooling
# Sample one view manually
dataset = trainer.train_dataset
cam_id, frame_idx = dataset.indices[0]
idx = 0
raw_batch = dataset[idx]
gpu_batch = dataset.get_gpu_batch_with_intrinsics(raw_batch)

print(f"\nRendering person_feature...")
render_out = trainer.model(gpu_batch, train=False, frame_id=0, render_person_feature=True)
fmap = render_out['person_feature_map']
amap = render_out.get('person_opacity_map')

print(f"feature_map.shape: {fmap.shape}")
print(f"feature_map.requires_grad: {fmap.requires_grad}")
print(f"opacity_map.requires_grad: {amap.requires_grad if amap is not None else 'N/A'}")

# Test direct region sum
trainer.model.zero_grad()
region = fmap[:, 100:150, 100:150]
loss_direct = region.sum()
loss_direct.backward()
print(f"\nDirect region sum grad_norm: {pf.grad.norm().item():.6e}")

# Test ROI pooling from same render
trainer.model.zero_grad()
from threedgrut.utils.roi_pooling import roi_pool

# Find a person in this frame
if hasattr(gpu_batch, 'instances') and gpu_batch.instances:
    inst = gpu_batch.instances[0]
    bbox = inst.get('bbox_xyxy', [50, 50, 200, 200])
    bbox_t = torch.tensor(bbox, dtype=torch.float32, device=trainer.device)
    
    print(f"\nROI pooling with bbox: {bbox}")
    f_v, stats = roi_pool(fmap, bbox_t, opacity_map=amap, pooling='opacity', detach_opacity_weight=True)
    print(f"pooled_feature: requires_grad={f_v.requires_grad}, grad_fn={f_v.grad_fn}")
    print(f"pooled_norm: {f_v.norm().item():.4f}")
    print(f"alpha_sum: {stats.get('alpha_sum', 'N/A')}")
    
    trainer.model.zero_grad()
    loss_roi = f_v.sum()
    loss_roi.backward()
    print(f"ROI pooled sum grad_norm: {pf.grad.norm().item():.6e}")
else:
    print("No instances in this batch")

# Test: what about person_feature_map directly?
trainer.model.zero_grad()
fmap2 = render_out['person_feature_map']
loss_fmap = fmap2.sum()
loss_fmap.backward()
print(f"\nfeature_map.sum() grad_norm: {pf.grad.norm().item():.6e}")

# Test: re-render and direct sum
trainer.model.zero_grad()
gpu_batch2 = dataset.get_gpu_batch_with_intrinsics(dataset[idx])
render_out2 = trainer.model(gpu_batch2, train=False, frame_id=0, render_person_feature=True)
fmap3 = render_out2['person_feature_map']
loss_fmap3 = fmap3.sum()
loss_fmap3.backward()
print(f"Re-rendered feature_map.sum() grad_norm: {pf.grad.norm().item():.6e}")
