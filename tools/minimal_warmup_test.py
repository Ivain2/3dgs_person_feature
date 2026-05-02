#!/usr/bin/env python3
"""Minimal warmup test - 30 iters RGB only"""

import os, sys, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from threedgrut.trainer import Trainer3DGRUT

config_dir = os.path.abspath('configs')
config_name = 'apps/wildtrack_full_3dgut'

with initialize_config_dir(config_dir=config_dir, version_base=None):
    conf = compose(config_name=config_name)

conf.model.person_feature_dim = 512

print("Initializing...")
trainer = Trainer3DGRUT(conf)
print(f"Gaussians: {trainer.model.num_gaussians}")

train_iter = iter(trainer.train_dataloader)

print("\nWarmup: 30 iterations with RGB loss only...")
trainer.model.train()

for step in range(30):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(trainer.train_dataloader)
        batch = next(train_iter)
    
    gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch)
    if len(gpu_batch.instances) == 0:
        continue
    
    trainer.model.zero_grad()
    render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
    
    pred_rgb = render_out['pred_rgb']
    person_feature_map = render_out['person_feature_map']
    
    L_rgb = F.l1_loss(pred_rgb, gpu_batch.rgb_gt)
    L_rgb.backward()
    trainer.model.optimizer.step()
    
    if step % 5 == 0:
        with torch.no_grad():
            rgb_nz = (pred_rgb.abs() > 1e-6).sum().item() / pred_rgb.numel() * 100
            feat_nz = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
        print(f"Step {step:2d}: L_rgb={L_rgb.item():.4f}, rgb_nz={rgb_nz:.3f}%, feat_nz={feat_nz:.3f}%")

# Final check
print("\nFinal check:")
batch = next(iter(trainer.train_dataloader))
gpu_batch = trainer.train_dataset.get_gpu_batch_with_intrinsics(batch)

trainer.model.zero_grad()
render_out = trainer.model(gpu_batch, train=True, frame_id=0, render_person_feature=True)
person_feature_map = render_out['person_feature_map']

with torch.no_grad():
    rgb_nz = (render_out['pred_rgb'].abs() > 1e-6).sum().item() / render_out['pred_rgb'].numel() * 100
    feat_nz = (person_feature_map.abs() > 1e-6).sum().item() / person_feature_map.numel() * 100
    print(f"pred_rgb nonzero: {rgb_nz:.3f}%")
    print(f"person_feature_map nonzero: {feat_nz:.3f}%")

person_feature_map.abs().mean().backward()
pf = trainer.model.get_person_feature()

if pf.grad is not None:
    g_mean = pf.grad.abs().mean().item()
    g_max = pf.grad.abs().max().item()
    nz = (pf.grad.abs() > 1e-8).sum().item() / pf.grad.numel() * 100
    print(f"\nperson_feature.grad:")
    print(f"  abs().mean(): {g_mean:.6f}")
    print(f"  abs().max(): {g_max:.6f}")
    print(f"  nonzero: {nz:.4f}%")
    
    if g_mean > 1e-8:
        print("\n✅ SUCCESS: Gradient flows after warmup!")
    else:
        print("\n⚠️  Gradient still zero - need more training")
else:
    print("\n❌ person_feature.grad is None")
