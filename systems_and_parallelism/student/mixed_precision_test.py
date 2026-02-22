#!/usr/bin/env python3
"""
Section 1.1.5 — mixed_precision_accumulation (5 pts)

Run 4 code snippets that demonstrate precision loss during accumulation.
Each adds 0.01 a thousand times. The expected result is 10.0.

Usage: uv run python student/mixed_precision_test.py
"""
import torch

# Snippet 1: FP32 accumulator + FP32 values → accurate
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(f"Snippet 1 — FP32 + FP32:       {s}")

# Snippet 2: FP16 accumulator + FP16 values → precision loss
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"Snippet 2 — FP16 + FP16:       {s}")

# Snippet 3: FP32 accumulator + FP16 values → auto-upcast, better than snippet 2
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"Snippet 3 — FP32 + FP16:       {s}")

# Snippet 4: FP32 accumulator + explicit FP16→FP32 cast → same as snippet 3
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(f"Snippet 4 — FP32 + FP16→FP32:  {s}")
