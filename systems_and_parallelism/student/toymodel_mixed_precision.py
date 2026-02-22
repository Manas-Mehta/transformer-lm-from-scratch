#!/usr/bin/env python3
"""
Section 1.1.5(a) — benchmarking_mixed_precision: ToyModel dtype inspection

Inspect data types at each stage of a forward/backward pass under
torch.autocast with FP16. Shows which layers get downcast and which stay FP32.

Usage: uv run python student/toymodel_mixed_precision.py
"""
import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


model = ToyModel(20, 5).cuda()
x = torch.randn(4, 20, device="cuda")  # FP32 input
target = torch.randint(0, 5, (4,), device="cuda")

print(f"Model parameters dtype: {next(model.parameters()).dtype}")
print()

with torch.autocast(device_type="cuda", dtype=torch.float16):
    # Parameters stay FP32 in memory — autocast creates temporary FP16 copies
    print(f"Parameters inside autocast: {model.fc1.weight.dtype}")

    # Forward pass — track dtypes at each stage
    fc1_out = model.relu(model.fc1(x))
    print(f"After fc1 + relu (output dtype): {fc1_out.dtype}")

    ln_out = model.ln(fc1_out)
    print(f"After LayerNorm (output dtype):  {ln_out.dtype}")

    logits = model.fc2(ln_out)
    print(f"Logits dtype:                    {logits.dtype}")

    loss = nn.functional.cross_entropy(logits, target)
    print(f"Loss dtype:                      {loss.dtype}")

loss.backward()
print(f"Gradient dtype (fc1.weight.grad): {model.fc1.weight.grad.dtype}")
