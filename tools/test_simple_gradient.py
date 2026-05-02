#!/usr/bin/env python3
"""Simple gradient test without rendering"""

import torch
import torch.nn.functional as F

# Test 1: Simple gradient flow through cat
print("Test 1: Gradient through torch.cat with zeros")
feature_chunk = torch.randn(100, 3, requires_grad=True)
zeros = torch.zeros(100, 5)
combined = torch.cat([feature_chunk, zeros], dim=1)
loss = combined.sum()
loss.backward()
print(f"  feature_chunk.grad is None: {feature_chunk.grad is None}")
if feature_chunk.grad is not None:
    print(f"  feature_chunk.grad.abs().mean(): {feature_chunk.grad.abs().mean().item():.6f}")
print()

# Test 2: Gradient flow through slicing
print("Test 2: Gradient through slicing")
full_feature = torch.randn(100, 8, requires_grad=True)
chunk = full_feature[:, :3]
loss = chunk.sum()
loss.backward()
print(f"  full_feature.grad is None: {full_feature.grad is None}")
if full_feature.grad is not None:
    print(f"  full_feature.grad.abs().mean(): {full_feature.grad.abs().mean().item():.6f}")
    print(f"  full_feature.grad[:, :3].abs().mean(): {full_feature.grad[:, :3].abs().mean().item():.6f}")
    print(f"  full_feature.grad[:, 3:].abs().mean(): {full_feature.grad[:, 3:].abs().mean().item():.6f}")
print()

# Test 3: Check if person_feature is properly connected
print("Test 3: Simulating person_feature rendering")
person_feature = torch.randn(50000, 512, requires_grad=True)

# Simulate chunking
chunk = person_feature[:, :3]
zeros = torch.zeros(50000, 5)
combined = torch.cat([chunk, zeros], dim=1)

# Simulate rendering (just a simple transformation for testing)
rendered = combined * 0.5
loss = rendered.sum()
loss.backward()

print(f"  person_feature.grad is None: {person_feature.grad is None}")
if person_feature.grad is not None:
    print(f"  person_feature.grad.abs().mean(): {person_feature.grad.abs().mean().item():.6f}")
    print(f"  person_feature.grad[:, :3].abs().mean(): {person_feature.grad[:, :3].abs().mean().item():.6f}")
    print(f"  person_feature.grad[:, 3:].abs().mean(): {person_feature.grad[:, 3:].abs().mean().item():.6f}")
