#!/usr/bin/env python3
"""Quick Phase 4 test - minimal warmup"""

import os
import sys
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.roi_pooling import roi_pool

config_dir = os.path.abspath('configs')
config_name = 'apps/wildtrack_full_3dgut'

with initialize_config_dir(config_dir=config_dir, version_base=None):
    conf = compose(config_name=config_name)

conf.model.person_feature_dim = 512
conf.loss.use_reid = True
conf.loss.lambda_reid = 0.05

print("Initializing trainer...")
trainer = Trainer3DGRUT(conf)

print(f"Gaussians: {trainer.model.num_gaussians}")
print(f"person_feature_dim: {trainer.model.get_person_feature().shape[1]}")

# Warmup: 20 iterations
print("\nWarmup training: 20 iterations...")
trainer.model.train()
train_iter = iter(trainer.train_dataloader)

for step in range(20):
    try:
        batch_data = next(train_iter)
    except StopIteration:
        train_iter = iter(trainer.train_dataloader)
        batch_data = next(train_iter)
    
    gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)
    
    if len(gpu_batch.instances) == 0:
        continue
    
    trainer.model.zero_grad()
    render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
    
    pred_rgb = render_out['pred_rgb']
    person_feature_map = render_out['person_feature_map']
    
    L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)
    
    # Simple L_reid
    L_reid = torch.zeros(1, device=trainer.device)
    valid_count = 0
    for inst in gpu_batch.instances:
        if inst.get('valid', False) and inst.get('teacher_embedding') is not None:
            bbox = inst['bbox_xyxy']
            f_v = roi_pool(person_feature_map, torch.tensor(bbox, dtype=torch.float32, device=trainer.device))
            t_v = torch.tensor(inst['teacher_embedding'], dtype=torch.float32, device=trainer.device)
            f_v_norm = F.normalize(f_v, p=2, dim=0)
            t_v_norm = F.normalize(t_v, p=2, dim=0)
            L_reid = L_reid + (1 - torch.dot(f_v_norm, t_v_norm))
            valid_count += 1
    
    if valid_count > 0:
        L_reid = L_reid / valid_count
    
    L_total = L_rgb + conf.loss.lambda_reid * L_reid
    L_total.backward()
    trainer.model.optimizer.step()
    
    if step % 5 == 0:
        print(f"  Step {step}: L_rgb={L_rgb.item():.4f}, L_reid={L_reid.item():.4f}, " +
              f"person_feature_map.abs().mean()={person_feature_map.abs().mean().item():.6f}")

# Check gradient
print("\nChecking gradient after warmup...")
batch_data = next(iter(trainer.train_dataloader))
gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch_data)

if len(gpu_batch.instances) > 0:
    trainer.model.zero_grad()
    render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
    person_feature_map = render_out['person_feature_map']
    
    print(f"person_feature_map stats:")
    print(f"  mean: {person_feature_map.mean().item():.6f}")
    print(f"  std: {person_feature_map.std().item():.6f}")
    print(f"  min: {person_feature_map.min().item():.6f}")
    print(f"  max: {person_feature_map.max().item():.6f}")
    
    nonzero_pixels = (person_feature_map.abs() > 1e-6).sum().item()
    total_pixels = person_feature_map.numel()
    print(f"  Nonzero pixels: {nonzero_pixels} / {total_pixels} ({100*nonzero_pixels/total_pixels:.4f}%)")
    
    # Backward from person_feature_map
    loss = person_feature_map.abs().mean()
    loss.backward()
    
    person_feature = trainer.model.get_person_feature()
    print(f"\nperson_feature.grad:")
    print(f"  is None: {person_feature.grad is None}")
    if person_feature.grad is not None:
        print(f"  abs().mean(): {person_feature.grad.abs().mean().item():.6f}")
        print(f"  abs().max(): {person_feature.grad.abs().max().item():.6f}")
        nonzero_grad = (person_feature.grad.abs() > 1e-8).sum().item()
        total = person_feature.grad.numel()
        print(f"  Nonzero elements: {nonzero_grad} / {total} ({100*nonzero_grad/total:.4f}%)")
        
        if nonzero_grad > 0:
            print("\n✅ SUCCESS: Gradient flows from person_feature_map to person_feature!")
        else:
            print("\n⚠️  Gradient is zero - may need more training")
