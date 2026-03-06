# Assignment 2 — Deliverables Only

GPU: NVIDIA A100-SXM4-40GB

---

## 1.1.1 Setup

> No graded deliverable. Setup verification step.

---

## 1.1.2 Model Sizing

> No separate graded deliverable. Model configs used throughout 1.1.

---

## 1.1.3(a) Benchmarking Script

> The benchmarking script (`student/benchmark.py`) supports all five model sizes from Table 1, configurable context lengths, forward-only and forward+backward modes, and uses `torch.cuda.synchronize()` before and after each timed step to ensure accurate wall-clock GPU measurements. It includes warmup steps to eliminate one-time CUDA initialization costs, and supports mixed-precision (BF16), NVTX profiling annotations, and memory profiling via command-line flags.

## 1.1.3(b) Benchmarking Results

> On an A100-SXM4-40GB, forward-only latency ranges from 38.32 ms (small) to 219.24 ms (2.7B), scaling roughly 5.7x across configurations. Forward+backward latency ranges from 78.33 ms to 672.30 ms, with the backward pass costing approximately 2-3x the forward pass due to gradient computation through every layer. Standard deviations are consistently below 1% of the mean, indicating stable measurements. The results were obtained with batch_size=4, context_length=128, 5 warmup steps, and 10 measurement steps.

## 1.1.3(c) Warmup Effect

> With zero warmup steps, the first measurement includes ~1550 ms of CUDA initialization overhead, inflating the mean to 226 ms with a standard deviation of 441 ms. A single warmup step eliminates this, bringing the mean to 77.6 ms with std of 2.0 ms. Beyond 1-2 warmup steps, there is negligible improvement (warmup=5 gives 76.3 ms mean). We use 5 warmup steps throughout the assignment to be safe, but even 1-2 would suffice for stable measurements.

---

## 1.1.4(a) nsys vs timeit

> The nsys `:measurement` NVTX range matches our timeit results almost exactly (within 0.1 ms), e.g., small ctx=128 gives 48.32 ms from timeit vs 48.34 ms from nsys. This is expected because both measure the same wall-clock interval using `torch.cuda.synchronize()`.

## 1.1.4(b) Dominant GPU Kernel

> The CUDA kernel with the most cumulative GPU time is `ampere_sgemm_128x64_tn` (matrix multiplication), e.g., 50.4% of GPU time with 222 invocations for small ctx=128, and all `ampere_sgemm_*` variants together account for 70-94% across all configs. The same GEMM kernel type dominates in forward+backward, though the backward pass adds additional sgemm/cutlass variants for transposed gradient computations.

## 1.1.4(c) Other Non-Trivial Kernels

> Besides matrix multiplications, the main non-trivial kernels are: elementwise kernels for activation functions and residual additions (~6-10%), reduction kernels for LayerNorm mean/variance and softmax max (~3%), and sigmoid/exp kernels for SiLU activation and softmax (~1%). These correspond to the non-matmul transformer operations and together account for ~15-22% of forward pass GPU time, shrinking as model size grows.

## 1.1.4(d) Full Training Step

> In a full training step, the matmul fraction drops from ~78% (forward-only) to ~64% of GPU kernel time (small ctx=128). The backward pass adds elementwise gradient kernels and transposed matmul variants, while the AdamW optimizer adds `multi_tensor_apply_kernel` entries (~17% of GPU time) for fused momentum/variance/weight updates, though the optimizer wall-clock time is very cheap (~3 ms, <3% of step time).

## 1.1.4(e) Softmax vs Matmul in Attention

> From CUDA kernel profiling, softmax-related kernels (`reduce_kernel MaxOps` + `exp_kernel`) take ~1-2% of GPU time while all `ampere_sgemm_*` matmul kernels take 79-85%, consistent with the ~64x FLOPs difference (matmul does O(S^2 * d_k) vs softmax's O(S^2) with d_k=64). Matmul also better utilizes the GPU's tensor cores (compute-bound) while softmax is memory-bandwidth-bound, further widening the runtime gap.

---

## 1.1.5 Mixed Precision

### mixed_precision_accumulation

> FP32+FP32 gives 10.0001 (nearly perfect), while FP16+FP16 gives 9.953 -- off by ~0.05 due to the accumulator stalling when the running sum grows too large for FP16 to represent small increments. Using an FP32 accumulator with FP16 values (Snippets 3/4) gives 10.002, much closer to correct, because the FP32 accumulator prevents stalling -- the remaining small error comes from 0.01 being slightly imprecise in FP16 representation (stored as ~0.010002). This demonstrates why mixed precision keeps accumulation operations (LayerNorm, loss) in FP32.

### 1.1.5(a) ToyModel dtype inspection

> Under FP16 autocast, model parameters remain stored in FP32 -- autocast does not modify the weights in memory, it creates temporary FP16 copies for computation. Linear layer outputs (matmuls) are downcast to FP16 for speed on tensor cores. LayerNorm output is kept in FP32 because its mean/variance accumulation is precision-sensitive (as demonstrated in mixed_precision_accumulation). The loss (cross_entropy) is computed in FP32, and all gradients are stored in FP32. This confirms autocast is truly "mixed" -- it selectively downcasts only the compute-heavy, precision-tolerant operations (matmuls) while preserving FP32 for accumulation-sensitive operations.

### 1.1.5(b) LayerNorm and precision

> The precision-sensitive parts of LayerNorm are the mean and variance computations, which involve summing many activation values -- exactly the accumulation pattern that loses precision in low-precision formats (as demonstrated in mixed_precision_accumulation). With BF16 instead of FP16, we still need to keep LayerNorm in FP32 because BF16 has even fewer mantissa bits (7 vs 10), making accumulation errors worse despite its better dynamic range. PyTorch's autocast correctly keeps LayerNorm in FP32 for both FP16 and BF16 modes.

### 1.1.5(c) Benchmarking mixed precision [5 pts]

Results:

| Model | FP32 Fwd+Bwd (ms) | BF16 Fwd+Bwd (ms) | Speedup |
|-------|--------------------|--------------------|---------|
| small | 77.17 | 87.79 | 0.88x (slower!) |
| medium | 153.67 | 174.10 | 0.88x (slower!) |
| large | 269.73 | 265.16 | 1.02x (breakeven) |
| xl | 463.06 | 361.23 | 1.28x |
| 2.7B | 674.57 | 242.04 | 2.79x |

> On an A100-SXM4-40GB with context_length=128, BF16 mixed precision provides significant speedups only for larger models: 2.79x for 2.7B and 1.28x for XL, while small and medium models are actually 12% slower due to autocast overhead exceeding the matmul speedup. The crossover point is around the large model (1.02x breakeven). This confirms that mixed precision benefits scale with the fraction of runtime spent on matmuls -- small models at short context lengths are overhead-dominated, not compute-bound, so faster tensor core matmuls don't translate to overall speedup.

---

## 1.1.6 Memory Profiling

### 1.1.6(a) Memory timeline [2-3 sentences]

> The forward-only memory timeline for the 2.7B model at ctx=128 shows a steady climb to 23,845 MB as activations accumulate across the 32 transformer layers, then flattens -- there are no gradients or optimizer states. The full training step (forward + backward + optimizer) could not complete because the 2.7B model requires ~40 GB just for parameters (10 GB) + gradients (10 GB) + AdamW states (20 GB), leaving no room for activations on the 40 GB A100. This demonstrates that the 2.7B model cannot be trained on a single A100 without techniques like gradient checkpointing, model parallelism, or reduced precision.

### 1.1.6(b) Peak memory by context length

| Context Length | Forward Peak (MB) | Full Training Peak (MB) |
|---------------|-------------------|------------------------|
| 128 | 23,845.1 | OOM |
| 256 | 36,607.4 | OOM |
| 512 | OOM | OOM |

> Peak forward-only memory scales steeply with context length: 23,845 MB at ctx=128, 36,607 MB at ctx=256, and OOM at ctx=512. All full training runs (forward + backward + optimizer) OOM'd because the 2.7B model's parameter + gradient + AdamW state memory (~40 GB) fills the entire A100 before any activations can be allocated. This illustrates that activation memory scales with `batch_size x context_length x d_model x num_layers`, and at longer contexts, even inference alone can exceed GPU capacity.

### 1.1.6(c) Mixed-precision memory [2-3 sentences]

| Mode | Precision | Peak Memory (MB) |
|------|-----------|-----------------|
| Forward, ctx=128 | FP32 | 23,845.1 |
| Forward, ctx=128 | BF16 | 32,826.2 |
| Full training, ctx=128 | FP32 | OOM |
| Full training, ctx=128 | BF16 | OOM |

> Counterintuitively, `torch.autocast` with BF16 **increases** peak memory from 23,845 MB to 32,826 MB for the forward pass (+37%). This is because autocast keeps model parameters in FP32 and creates temporary BF16 copies for matmul operations -- both versions coexist in GPU memory. For true memory savings, you would need to convert the model weights themselves to BF16 (e.g., `model.bfloat16()`), rather than relying on autocast which is designed for speed, not memory reduction.

### 1.1.6(d) Activation tensor size [1-2 sentences]

> A single activation tensor in the 2.7B model's residual stream has shape (4, context_length, 2560) in FP32. At context_length=128 this is 4 x 128 x 2560 x 4 bytes = 5.0 MB; at context_length=1024, it is 40.0 MB. Each transformer layer produces (and must save for backward) at least one such tensor.

### 1.1.6(e) Largest allocations [1-2 sentences]

> When reducing the Detail slider to ~13% (371 of 2867 entries), the largest allocations form a dominant blue staircase pattern climbing from ~5 GiB to ~22 GiB. These are the residual stream activation tensors at each transformer layer -- each ~5 MB (batch=4 x ctx=128 x d_model=2560 x 4 bytes), and there are 32 layers' worth stacking up. The stack traces point to the forward pass of each `TransformerBlock`, specifically the output tensors from the attention and feed-forward sublayers.

---

## 1.2 Problem (pytorch_attention): 5 points

Results:

| d_model | seq_len | Fwd (ms) | Bwd (ms) | Mem after fwd (MB) |
|---------|---------|----------|----------|-------------------|
| 16 | 256 | 0.40 | 2.67 | 20.7 |
| 16 | 1024 | 0.73 | 2.20 | 212.2 |
| 16 | 4096 | 7.59 | 26.00 | 3,116.0 |
| 16 | 8192 | 28.27 | 96.65 | 12,397.8 |
| 16 | 16384 | OOM | OOM | OOM |
| 32 | 256 | 0.39 | 1.27 | 53.4 |
| 32 | 1024 | 0.68 | 2.15 | 214.9 |
| 32 | 4096 | 7.92 | 26.45 | 3,127.0 |
| 32 | 8192 | 29.15 | 98.48 | 12,425.8 |
| 32 | 16384 | OOM | OOM | OOM |
| 64 | 256 | 0.38 | 1.27 | 78.4 |
| 64 | 1024 | 0.70 | 2.22 | 220.4 |
| 64 | 4096 | 8.37 | 27.38 | 3,149.0 |
| 64 | 8192 | 30.97 | 102.12 | 12,481.8 |
| 64 | 16384 | OOM | OOM | OOM |
| 128 | 256 | 0.38 | 1.26 | 128.4 |
| 128 | 1024 | 0.76 | 2.34 | 231.4 |
| 128 | 4096 | 9.21 | 29.14 | 3,193.0 |
| 128 | 8192 | 34.41 | 109.12 | 12,593.8 |
| 128 | 16384 | OOM | OOM | OOM |

> **Table**: See results above. All configurations OOM at seq_len=16384 regardless of d_model, because the attention score matrix requires `8 x 16384^2 x 4 = 32 GB` in FP32 -- leaving no room on the 40 GB A100 for the softmax output, gradients, and other tensors. Memory after the forward pass is dominated by the O(n^2) attention scores: ~78 MB at seq=256 growing to ~12.5 GB at seq=8192 for d=64, with d_model contributing only a small additive term for Q/K/V storage.
>
> The backward pass saves the full attention matrix P for gradient computation, so backward memory also scales as O(n^2). Forward time scales quadratically (~0.4ms at seq=256 to ~31ms at seq=8192), and backward is consistently ~3x forward. To eliminate this O(n^2) memory bottleneck, FlashAttention computes attention in tiles using online softmax, never materializing the full attention matrix -- reducing memory from O(n^2) to O(n) and often improving speed by keeping data in fast GPU SRAM.

---

## 1.3 Problem (torch_compile): 5 points

### 1.3(a) Compiled Attention Results

| d_model | seq_len | Fwd (ms) | Bwd (ms) | Mem after fwd (MB) |
|---------|---------|----------|----------|-------------------|
| 16 | 256 | 0.28 | 15.45 | 16.9 |
| 16 | 1024 | 0.59 | 4.70 | 148.8 |
| 16 | 4096 | 4.01 | 14.63 | 2,094.3 |
| 16 | 8192 | 16.71 | 54.58 | 8,306.3 |
| 16 | 16384 | 56.82 | 539.32 | 33,108.3 |
| 32 | 256 | 0.37 | 2.96 | 73.6 |
| 32 | 1024 | 0.51 | 1.57 | 152.0 |
| 32 | 4096 | 5.07 | 16.30 | 2,107.3 |
| 32 | 8192 | 18.19 | 58.33 | 8,338.3 |
| 32 | 16384 | 62.27 | 479.21 | 33,172.3 |
| 64 | 256 | 0.37 | 1.00 | 122.9 |
| 64 | 1024 | 0.63 | 1.63 | 158.5 |
| 64 | 4096 | 4.96 | 16.44 | 2,133.3 |
| 64 | 8192 | 16.99 | 57.89 | 8,402.3 |
| 64 | 16384 | 69.40 | 233.90 | 33,300.3 |
| 128 | 256 | 0.37 | 1.03 | 221.4 |
| 128 | 1024 | 0.69 | 1.74 | 171.5 |
| 128 | 4096 | 5.82 | 18.19 | 2,185.3 |
| 128 | 8192 | 20.44 | 64.84 | 8,530.3 |
| 128 | 16384 | 83.18 | 261.44 | 33,556.3 |

### 1.3(b) Compiled Full Model Results

| Model Size | Mode | Forward (ms) | Full Step (ms) |
|-----------|------|-------------|---------------|
| small | vanilla | 38.78 | 87.33 |
| small | compiled | 10.24 | 40.10 |
| medium | vanilla | 79.22 | 183.19 |
| medium | compiled | 30.87 | 123.84 |
| large | vanilla | 116.21 | 336.14 |
| large | compiled | 63.61 | 261.63 |

> **(a)** Compiled attention is ~1.5-1.9x faster in the forward pass and ~1.7x faster in the backward pass at practical sizes (seq >= 1024). Notably, `torch.compile` eliminates OOM at seq_len=16384 by fusing operations and avoiding some intermediate tensor materializations -- though memory remains O(n^2) at ~33 GB. At very small configs (d=16, seq=256), compilation overhead can make the backward pass slower.
>
> **(b)** Compiled full model shows 1.8-3.8x forward speedup and 1.3-2.2x full-step speedup. The small model benefits most because it has many small fusible operations. Larger models see diminishing returns as compute-bound matmuls dominate. Neither compiled attention nor compiled full model eliminates the fundamental O(n^2) memory scaling -- that requires FlashAttention.

---

## 1.3.2 Problem (flash_forward): 25 points

### Test Results (all 6/6 passed)

| Test | Status |
|------|--------|
| `test_flash_forward_pass_pytorch` | PASSED |
| `test_flash_forward_pass_triton[False]` (non-causal) | PASSED |
| `test_flash_forward_pass_triton[True]` (causal) | PASSED |
| `test_flash_backward_pytorch` | PASSED |
| `test_flash_backward_triton[False]` (non-causal) | PASSED |
| `test_flash_backward_triton[True]` (causal) | PASSED |

> (a) Pure PyTorch `torch.autograd.Function` implementing FlashAttention-2 forward pass. Passes `test_flash_forward_pass_pytorch`.
>
> (b) Triton kernel implementing FlashAttention-2 forward pass, wrapped in `torch.autograd.Function`. Passes `test_flash_forward_pass_triton` for both causal and non-causal.
>
> (c) Causal masking implemented in both PyTorch and Triton versions. Both pass with `is_causal=True` and `is_causal=False`.

---

## 1.3.2 Problem (flash_backward): 10 points

> Backward pass implemented using PyTorch + `torch.compile`, following Equations 13-19 (recomputation of P from L instead of saving O(n^2) attention matrix). Passes `test_flash_backward_pytorch` and `test_flash_backward_triton` for both causal modes.

---

## 1.3.2 Problem (flash_benchmarking): 15 points

### BFloat16 Results (selected)

| d_head | seq_len | FA2 fwd | FA2 bwd | FA2 e2e | Van fwd | Van bwd | Van e2e | Fwd spdup | E2E spdup |
|--------|---------|---------|---------|---------|---------|---------|---------|-----------|-----------|
| 16 | 128 | 0.009ms | 0.710ms | 0.719ms | 0.246ms | 0.868ms | 1.114ms | 27.51x | 1.55x |
| 16 | 4096 | 0.051ms | 0.756ms | 0.808ms | 0.508ms | 1.102ms | 1.610ms | 9.92x | 1.99x |
| 16 | 32768 | 1.191ms | 26.714ms | 27.905ms | 27.758ms | 60.671ms | 88.429ms | 23.30x | 3.17x |
| 64 | 4096 | 0.114ms | 0.754ms | 0.869ms | 0.516ms | 1.087ms | 1.603ms | 4.52x | 1.85x |
| 64 | 32768 | 2.523ms | 40.099ms | 42.622ms | 28.214ms | 60.515ms | 88.729ms | 11.18x | 2.08x |
| 128 | 4096 | 0.202ms | 1.187ms | 1.389ms | 0.536ms | 1.125ms | 1.661ms | 2.65x | 1.20x |
| 128 | 32768 | 7.610ms | 62.593ms | 70.202ms | 28.938ms | 61.297ms | 90.236ms | 3.80x | 1.29x |

All d_head values OOM at seq_len=65536.

### Float32 Results (selected)

| d_head | seq_len | FA2 fwd | FA2 bwd | FA2 e2e | Van fwd | Van bwd | Van e2e | Fwd spdup | E2E spdup |
|--------|---------|---------|---------|---------|---------|---------|---------|-----------|-----------|
| 16 | 4096 | 0.071ms | 0.660ms | 0.731ms | 0.808ms | 1.824ms | 2.632ms | 11.45x | 3.60x |
| 16 | 32768 | 1.767ms | 31.495ms | 33.262ms | 45.560ms | 104.104ms | 149.664ms | 25.78x | 4.50x |
| 32 | 32768 | 3.005ms | 34.797ms | 37.802ms | 46.886ms | 105.912ms | 152.799ms | 15.60x | 4.04x |
| 64 | 32768 | 9.306ms | 52.282ms | 61.587ms | 53.808ms | 120.031ms | 173.838ms | 5.78x | 2.82x |

fp32 d_head=128: All sequence lengths hit Triton shared memory error (`OutOfResources: Required 181248 > Hardware limit 166912`). With 64x64 tiles and D=128 in fp32, each tile block needs ~181KB, exceeding the A100's 166KB/block limit. bf16 d=128 works because bf16 is 2 bytes (halving shared memory usage).

All d_head values OOM at seq_len=65536 for fp32.

> The Triton FA2 forward kernel achieves 2.5-27x speedup over vanilla PyTorch attention, with the largest gains at small d_head and long sequences where vanilla attention's O(n^2) memory traffic dominates. End-to-end (forward + backward) speedup is more modest at 1.2-4.5x because the backward pass uses `torch.compile` rather than a tiled Triton kernel, so it still benefits from some fusion but not the full SRAM tiling advantage.
>
> Key trends: (1) Speedup increases with sequence length -- FA2's tiled approach scales better than O(n^2). (2) Speedup decreases with d_head -- larger heads make matmuls compute-bound, reducing FA2's memory-access advantage. (3) fp32 benefits more than bf16 (4.5x vs 3.2x E2E at seq=32768) because vanilla fp32 pays 2x the memory bandwidth cost while FA2's forward accumulates in fp32 regardless. (4) Both implementations OOM at seq=65536 -- FA2's forward would survive, but the backward (which materializes O(n^2) attention) causes the OOM. A full Triton backward would fix this.
