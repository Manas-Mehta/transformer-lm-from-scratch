# Section 1.1 -- Profiling and Benchmarking

This section is about understanding how fast (and how memory-hungry) transformer models are on real GPU hardware. We will build a benchmarking script, profile it with NVIDIA tools, experiment with mixed-precision training, and profile memory usage. Everything runs on an A100 40GB GPU on the NYU HPC cluster.

---

## 1.1.1 Setup -- Importing your Basics Transformer Model

### What this section is about

Before we can benchmark anything, we need to confirm that our environment works: that we can import the staff-provided transformer model and run it on a GPU. This is a sanity check step.

### Key concepts

- **Staff model package (`a1_basics`)**: The assignment provides a pre-built transformer language model as a Python package called `a1_basics`. We import it with `from a1_basics.model import BasicsTransformerLM`. This is the model we will benchmark throughout Section 1.1.
- **`uv` package manager**: Instead of `pip`, this project uses `uv` to manage dependencies. We run scripts with `uv run python script.py` so that all packages are resolved correctly.
- **Singularity overlay on HPC**: Our code and packages live inside a filesystem overlay (`/scratch/mm14444/overlay-25GB-500K.ext3`) mounted inside a Singularity container (`/scratch/mm14444/ubuntu-20.04.3.sif`). SLURM jobs use `singularity exec --overlay ... --nv` to get GPU access.

### How to verify

Start an interactive GPU session:

```bash
srun --account=csci_ga_3033_131-2026sp --partition=c12m85-a100-1 \
     --gres=gpu:1 --time=00:30:00 --pty /bin/bash
```

Then inside the node:

```bash
singularity exec --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif /bin/bash

cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism
uv run python -c "from a1_basics.model import BasicsTransformerLM; print('Import OK')"
```

If you see `Import OK`, the setup is working.

### Deliverable

> No graded deliverable for 1.1.1. This is a setup verification step.

---

## 1.1.2 Model Sizing

### What this section is about

We need to understand how many parameters each model configuration has. The assignment gives us 5 model sizes (small through 2.7B), and we need to be able to instantiate them and count their parameters. This grounds our intuition: bigger models = more parameters = more compute = more memory.

### Key concepts

- **Model configurations (Table 1)**: Each model size is defined by four hyperparameters:

| Size | d_model | d_ff | num_layers | num_heads |
|------|---------|------|------------|-----------|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 1600 | 6400 | 48 | 25 |
| 2.7B | 2560 | 10240 | 32 | 32 |

All use `vocab_size=10000` and `batch_size=4`.

- **d_model**: The hidden dimension size. Every token is represented as a vector of this length.
- **d_ff**: The feed-forward inner dimension. Each transformer layer has a 2-layer MLP that expands to `d_ff` then contracts back to `d_model`.
- **num_layers**: How many transformer blocks are stacked. More layers = deeper model.
- **num_heads**: How many attention heads in multi-head attention. Each head looks at `d_model / num_heads` dimensions.

### Counting parameters

A rough formula for a transformer LM's parameter count:

```
Embedding:      vocab_size * d_model
Per layer:      4 * d_model^2  (Q, K, V, O projections)
              + 2 * d_model * d_ff  (FFN up + down)
              + layer norm params (small)
Output head:    d_model * vocab_size (often tied with embedding)
```

You can count exactly in PyTorch:

```python
model = BasicsTransformerLM(vocab_size=10000, context_length=128, ...)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
```

### Deliverable

> No separate graded deliverable for 1.1.2. The model configs are used throughout the rest of 1.1.

---

## 1.1.3 End-to-End Benchmarking

This is where the real work begins. We build a benchmarking script, time all model sizes, and study the effect of warmup.

---

### (a) benchmarking_script [10 pts]

### What this section is about

We need to write a Python script (`benchmark.py`) that measures how long forward and backward passes take for our transformer model on GPU. This is harder than it sounds because GPU operations are *asynchronous* -- we need special care to get accurate timings.

### Key concepts

#### Why GPU timing is tricky: CUDA is asynchronous

When you call `model(input_ids)` in PyTorch on a GPU, Python does NOT wait for the GPU to finish. It just *queues* the work and moves on. This is called **asynchronous execution**.

Think of it like a restaurant: you (Python/CPU) place an order (launch a kernel), but the kitchen (GPU) hasn't finished cooking yet. If you start your timer, place the order, and stop the timer immediately, you've only measured how long it took to *place the order*, not how long the food took to cook.

**Solution**: `torch.cuda.synchronize()` -- this forces Python to wait until ALL queued GPU operations are done. We call it:
1. **Before** starting the timer (to make sure no previous GPU work is still running)
2. **After** the operations we want to time (to make sure they're actually done)

#### Why we need warmup steps

The very first time you run a model on GPU, PyTorch does a lot of one-time setup:
- CUDA context initialization (loading the GPU driver)
- cuDNN autotuning (finding the fastest algorithm for each operation)
- Memory allocation (first-time allocations are slower)
- JIT compilation of certain kernels

These one-time costs inflate the first measurement. **Warmup steps** run the model a few times *without timing* so that all this setup is done before we start measuring.

#### Forward vs. backward pass

- **Forward pass**: Run the model to get predictions. `output = model(input_ids)`
- **Backward pass**: Compute gradients for training. `loss.backward()`
- **Both**: Forward + backward together, which is what happens during training.

### What we need to build

A script that:
1. Parses command-line arguments (model size, context length, mode, etc.)
2. Creates the appropriate model and moves it to GPU
3. Generates random input data (we don't need real text for timing)
4. Runs warmup steps (not timed)
5. Runs measurement steps with proper GPU synchronization
6. Reports mean and standard deviation of timing

### Pseudocode / structure

```
parse_args()
    -> model_size, context_length, batch_size, vocab_size
    -> warmup_steps, measure_steps
    -> mode (forward / backward / both)
    -> mixed_precision flag, profile_memory flag, nvtx_attention flag

create_model(model_size, vocab_size, context_length)
    -> look up config dict
    -> instantiate BasicsTransformerLM with those params
    -> move to cuda
    -> return model

generate_random_batch(batch_size, context_length, vocab_size)
    -> torch.randint to create random token IDs on GPU

benchmark(model, input_ids, mode, warmup_steps, measure_steps, amp_context)
    -> WARMUP LOOP (warmup_steps times):
        if backward mode: zero gradients
        run forward pass (inside amp_context for mixed precision)
        if backward mode: compute loss, run backward
        synchronize GPU
    -> MEASUREMENT LOOP (measure_steps times):
        if backward mode: zero gradients
        synchronize GPU          <-- drain any pending work
        start timer
        run forward pass
        if backward mode: loss.backward()
        synchronize GPU          <-- wait for GPU to finish
        stop timer
        record elapsed time
    -> return list of times

main()
    -> parse args, print config
    -> create model and data
    -> set up amp_context (autocast or nullcontext)
    -> call benchmark()
    -> compute and print mean/std
```

### Code walkthrough

Let me explain the trickiest parts:

**`torch.cuda.synchronize()`** -- This is the critical call. Without it, we'd be timing how fast Python can *launch* GPU kernels, not how fast the GPU actually runs them. We synchronize:
- Before `start = timeit.default_timer()` -- to ensure no leftover GPU work from previous iteration
- After the forward/backward -- to ensure the GPU is done before we stop the timer

**`model.zero_grad(set_to_none=True)`** -- Before each backward pass, we must clear old gradients. `set_to_none=True` is faster than setting to zero because it deallocates the gradient tensors entirely instead of filling them with zeros.

**`nullcontext()`** -- When NOT using mixed precision, we need a "do nothing" context manager. `nullcontext()` serves this purpose. It's a stand-in for `torch.autocast(...)` that doesn't change anything.

**`output.sum()` as loss** -- We need a scalar loss to call `.backward()`. Since we're just timing (not training for real), `output.sum()` is the simplest scalar we can compute from the model output.

**`timeit.default_timer()`** -- This uses the highest-resolution clock available on your platform. Combined with `torch.cuda.synchronize()`, it gives accurate wall-clock GPU timing.

**NVTX ranges** -- These are annotations for the NVIDIA profiler (covered in 1.1.4). `nvtx.range("name")` marks a region in the profiler timeline so we can see which part of the code corresponds to which GPU activity. They have negligible overhead.

### Final code: `student/benchmark.py`

```python
'''
Parse command-line arguments (model size, context length, warmup steps, etc.)
Initialize a model from the given hyperparameters
Generate random input data
Run warm-up steps (not timed)
Time n measurement steps for forward and/or backward passes
Call torch.cuda.synchronize() after each step
Report mean and standard deviation
'''

import argparse
import timeit
import math
import torch
import torch.cuda.nvtx as nvtx
import numpy as np
from contextlib import nullcontext
import a1_basics.model
from a1_basics.model import BasicsTransformerLM
from a1_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Transformer LM")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=MODEL_CONFIGS.keys(),
                        help="Model size from Table 1")
    parser.add_argument("--context_length", type=int, default=128,
                        help="Sequence length (128, 256, 512, 1024)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (fixed at 4 per assignment)")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Vocabulary size (fixed at 10000)")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Number of warm-up steps before timing")
    parser.add_argument("--measure_steps", type=int, default=10,
                        help="Number of steps to time")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["forward", "backward", "both"],
                        help="What to benchmark: forward only, backward only, or both")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use BF16 mixed precision (for Section 1.1.5)")
    parser.add_argument("--profile_memory", action="store_true",
                        help="Enable memory profiling (for Section 1.1.6)")
    parser.add_argument("--nvtx_attention", action="store_true",
                        help="Enable NVTX-annotated attention for 1.1.4(e)")
    return parser.parse_args()

def create_model(model_size, vocab_size, context_length):
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,
    )
    model = model.to("cuda")
    return model

def generate_random_batch(batch_size, context_length, vocab_size):
    return torch.randint(0, vocab_size, (batch_size, context_length), device="cuda")

def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """NVTX-annotated attention for profiling (Section 1.1.4e)."""
    d_k = K.shape[-1]
    with nvtx.range("attention_matmul_QK"):
        attention_scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("attention_softmax"):
        attention_weights = torch.softmax(attention_scores, dim=-1)
    with nvtx.range("attention_matmul_V"):
        output = torch.einsum("...qk,...kd->...qd", attention_weights, V)
    return output

def benchmark(model, input_ids, mode, warmup_steps, measure_steps, amp_context):
    """Core benchmarking loop with proper GPU synchronization."""
    # --- Warmup phase (not timed) ---
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            if mode in ("backward", "both"):
                model.zero_grad(set_to_none=True)
            with amp_context:
                output = model(input_ids)
            if mode in ("backward", "both"):
                loss = output.sum()
                loss.backward()
            torch.cuda.synchronize()

    # --- Measurement phase ---
    times = []
    with nvtx.range("measurement"):
        for i in range(measure_steps):
            if mode in ("backward", "both"):
                model.zero_grad(set_to_none=True)

            torch.cuda.synchronize()            # drain pending GPU work
            start = timeit.default_timer()

            with nvtx.range(f"step_{i}"):
                with nvtx.range("forward"):
                    with amp_context:
                        output = model(input_ids)
                if mode in ("backward", "both"):
                    with nvtx.range("backward"):
                        loss = output.sum()
                        loss.backward()

            torch.cuda.synchronize()            # wait for GPU to finish
            end = timeit.default_timer()
            times.append(end - start)

    return times

def profile_memory(model, input_ids, mode, warmup_steps, amp_context, args):
    """Memory profiling with PyTorch memory snapshot (Section 1.1.6)."""
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Warmup with optimizer
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        with amp_context:
            output = model(input_ids)
        if mode in ("backward", "both"):
            loss = output.sum()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

    # Reset peak stats and start recording
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # One measured step
    optimizer.zero_grad()
    with amp_context:
        output = model(input_ids)
    if mode in ("backward", "both"):
        loss = output.sum()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Save snapshot
    mp_tag = "_bf16" if args.mixed_precision else "_fp32"
    fname = f"memory_{args.model_size}_{args.context_length}_{args.mode}{mp_tag}.pickle"
    torch.cuda.memory._dump_snapshot(fname)
    torch.cuda.memory._record_memory_history(enabled=None)

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Memory snapshot saved to: {fname}")
    print(f"Peak memory allocated: {peak_mem_mb:.1f} MB")

def main():
    args = parse_args()

    print(f"=== Benchmarking {args.model_size} model ===")
    print(f"Context length: {args.context_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mode: {args.mode}")
    print(f"Warmup: {args.warmup_steps}, Measure: {args.measure_steps}")
    print(f"Mixed precision: {args.mixed_precision}")
    print()

    # Optionally monkey-patch attention for NVTX annotation
    if args.nvtx_attention:
        a1_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        print("NVTX attention annotations: ENABLED")

    model = create_model(args.model_size, args.vocab_size, args.context_length)
    input_ids = generate_random_batch(args.batch_size, args.context_length, args.vocab_size)

    # Set up mixed precision context
    if args.mixed_precision:
        amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()

    # Branch: memory profiling vs. timing benchmark
    if args.profile_memory:
        profile_memory(model, input_ids, args.mode, args.warmup_steps, amp_context, args)
        return

    times = benchmark(model, input_ids, args.mode,
                      args.warmup_steps, args.measure_steps, amp_context)

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Results over {args.measure_steps} steps:")
    print(f"  Mean: {mean_time * 1000:.2f} ms")
    print(f"  Std:  {std_time * 1000:.2f} ms")
    print(f"  Times (ms): {[f'{t*1000:.2f}' for t in times]}")

if __name__ == "__main__":
    main()
```

### How to run

**Interactive (for quick testing)**:

```bash
# Get an interactive GPU session
srun --account=csci_ga_3033_131-2026sp --partition=c12m85-a100-1 \
     --gres=gpu:1 --time=00:30:00 --pty /bin/bash

# Inside the node, start the container
singularity exec --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  --nv /scratch/mm14444/ubuntu-20.04.3.sif /bin/bash

cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

# Quick test with the small model
uv run python student/benchmark.py --model_size small --context_length 128 --mode forward
```

**Via SLURM**: See the sbatch scripts in 1.1.3(b) below.

### Deliverable answer for 1.1.3(a)

> The benchmarking script (`student/benchmark.py`) supports all five model sizes from Table 1, configurable context lengths, forward-only and forward+backward modes, and uses `torch.cuda.synchronize()` before and after each timed step to ensure accurate wall-clock GPU measurements. It includes warmup steps to eliminate one-time CUDA initialization costs, and supports mixed-precision (BF16), NVTX profiling annotations, and memory profiling via command-line flags.

---

### (b) Time all model sizes

### What this section is about

Now we actually run the benchmark across all 5 model sizes to see how latency scales with model size. We measure both forward-only and forward+backward (training step) at context_length=128 with batch_size=4.

### How to run

Submit the SLURM job:

```bash
cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism
sbatch student/run_1_1_3b.sbatch
```

The SLURM script (`student/run_1_1_3b.sbatch`) loops over all 5 model sizes and runs both `--mode forward` and `--mode both` for each.

### Results

**GPU: NVIDIA A100-SXM4-40GB** | context_length=128 | batch_size=4 | warmup=5 | measure=10

| Model | Fwd Mean (ms) | Fwd Std (ms) | Fwd+Bwd Mean (ms) | Fwd+Bwd Std (ms) |
|-------|---------------|--------------|--------------------|--------------------|
| small | 38.32 | 0.36 | 78.33 | 0.46 |
| medium | 76.59 | 0.71 | 154.60 | 0.31 |
| large | 114.59 | 0.56 | 273.11 | 3.71 |
| xl | 155.08 | 1.06 | 457.88 | 1.36 |
| 2.7B | 219.24 | 0.10 | 672.30 | 0.23 |

### Analysis

Several important patterns:

1. **Forward pass scales roughly linearly with model depth and width.** Going from small (12 layers, 768 hidden) to 2.7B (32 layers, 2560 hidden) increases forward time by ~5.7x. This makes sense because the compute is dominated by matrix multiplications whose cost scales with `num_layers * d_model^2`.

2. **Backward pass is roughly 2-3x the cost of forward.** For the small model, fwd+bwd (78.33 ms) is about 2.04x the forward-only time (38.32 ms). For 2.7B, the ratio is 672.30 / 219.24 = 3.07x. The backward pass is more expensive because it needs to compute gradients for every parameter, which requires roughly 2x the FLOPs of the forward pass (chain rule through every layer).

3. **Low standard deviations** (generally <1% of mean) confirm that our warmup is effective and measurements are stable.

4. **The "2.7B" model is not actually 2.7 billion parameters** at vocab_size=10000 (it would be at larger vocab sizes). But it has the same architecture dimensions as a real 2.7B model.

### Deliverable answer for 1.1.3(b)

> On an A100-SXM4-40GB, forward-only latency ranges from 38.32 ms (small) to 219.24 ms (2.7B), scaling roughly 5.7x across configurations. Forward+backward latency ranges from 78.33 ms to 672.30 ms, with the backward pass costing approximately 2-3x the forward pass due to gradient computation through every layer. Standard deviations are consistently below 1% of the mean, indicating stable measurements. The results were obtained with batch_size=4, context_length=128, 5 warmup steps, and 10 measurement steps.

---

### (c) Warmup effect

### What this section is about

We test what happens when you vary the number of warmup steps (0, 1, 2, 5) to demonstrate why warmup is necessary for accurate benchmarking.

### Key concepts

The first time PyTorch runs a model on GPU, several things happen that inflate the timing:
- **CUDA context initialization**: The GPU driver is loaded and set up (~0.5-2 seconds on first call)
- **cuDNN autotuning**: PyTorch tests different convolution/GEMM algorithms to find the fastest one
- **Memory allocator warmup**: First allocations trigger the caching allocator to reserve GPU memory pools
- **Kernel JIT compilation**: Some operations are compiled on first use

After these one-time costs are paid, subsequent iterations are much faster and more consistent.

### How to run

```bash
sbatch student/run_1_1_3c.sbatch
```

This script runs the small model with `--mode both` and `--warmup_steps` set to 0, 1, 2, and 5.

### Results

**Model: small** | context_length=128 | mode=both | measure_steps=10

| Warmup Steps | Mean (ms) | Std (ms) | 1st Step (ms) | 2nd Step (ms) |
|-------------|-----------|----------|----------------|----------------|
| 0 | 226.39 | 441.17 | 1549.87 | 84.33 |
| 1 | 77.61 | 2.02 | 78.77 | 76.84 |
| 2 | 76.42 | 2.58 | 74.87 | 75.54 |
| 5 | 76.29 | 2.13 | 75.38 | 76.07 |

### Analysis

1. **Zero warmup is catastrophic.** The first measured step takes 1549.87 ms (nearly 20x the steady-state time) because it includes all the one-time CUDA initialization overhead. This single outlier inflates the mean to 226.39 ms and the standard deviation to 441.17 ms -- making the measurement useless.

2. **One warmup step fixes most of the problem.** With just 1 warmup step, the mean drops to 77.61 ms with std of only 2.02 ms. The CUDA initialization is now paid during warmup.

3. **Two or more warmup steps are essentially the same.** The difference between warmup=2 (76.42 ms) and warmup=5 (76.29 ms) is negligible. This tells us that 1-2 warmup steps are sufficient for this model.

4. **The standard deviation is the key indicator.** With 0 warmup, std is 441 ms (194% of mean). With 1+ warmup, std drops to ~2 ms (~2.6% of mean). Low std means reproducible measurements.

### Deliverable answer for 1.1.3(c)

> With zero warmup steps, the first measurement includes ~1550 ms of CUDA initialization overhead, inflating the mean to 226 ms with a standard deviation of 441 ms. A single warmup step eliminates this, bringing the mean to 77.6 ms with std of 2.0 ms. Beyond 1-2 warmup steps, there is negligible improvement (warmup=5 gives 76.3 ms mean). We use 5 warmup steps throughout the assignment to be safe, but even 1-2 would suffice for stable measurements.

---

## 1.1.4 Nsight Systems Profiler [10 pts]

### What this section is about

Nsight Systems (`nsys`) is NVIDIA's system-wide profiler. It records a timeline of everything happening on GPU (kernel launches, memory copies, CUDA API calls) and lets us visualize where time is actually spent. To make the profiler output useful, we'll add **NVTX annotations** to our benchmark.py so we can label regions like "warmup", "forward", "backward" in the profiler timeline.

### Key concepts

#### What is `nsys`?

`nsys` is a command-line profiler from NVIDIA. You wrap your normal Python command with `nsys profile ...` and it records a trace file (`.nsys-rep`) that you can open in Nsight Systems GUI. Think of it as a "flight recorder" for your GPU.

#### What is NVTX?

**NVTX** (NVIDIA Tools Extension) lets you add *named markers* to your code. When the profiler records a trace, these markers show up as labeled regions on the timeline. Without NVTX, you'd just see a wall of GPU kernels with cryptic names. With NVTX, you can see "this is the forward pass", "this is the backward pass", "this is the QK matmul", etc.

In PyTorch:
```python
import torch.cuda.nvtx as nvtx

with nvtx.range("forward"):
    output = model(input_ids)  # this region is now labeled "forward" in the profiler
```

#### What is monkey-patching?

Monkey-patching means replacing a function at runtime. The staff's model calls `a1_basics.model.scaled_dot_product_attention(...)` inside its attention layers. We can replace this function with our own annotated version, and the model will call our version instead — without modifying the staff's code:

```python
import a1_basics.model
a1_basics.model.scaled_dot_product_attention = our_annotated_version
```

### What we need to change in benchmark.py

We need to make **4 changes** to benchmark.py:

1. **Add imports** for NVTX, math, and the a1_basics module
2. **Add NVTX ranges to `benchmark()`** — wrap warmup, measurement, forward, backward
3. **Add `annotated_scaled_dot_product_attention()`** — a new function for 1.1.4(e)
4. **Add `--nvtx_attention` flag** — to optionally monkey-patch the attention function
5. **Add the monkey-patch logic in `main()`**

### Step-by-step: building the code

#### Step 1: Add new imports

At the top of `benchmark.py`, add these imports:

```python
import math
import torch.cuda.nvtx as nvtx   # NVTX annotations for nsys profiling
import a1_basics.model            # needed for monkey-patching attention
```

You already have `from a1_basics.model import BasicsTransformerLM` — keep that. The new `import a1_basics.model` gives you access to the module-level function `a1_basics.model.scaled_dot_product_attention` which we'll replace later.

#### Step 2: Add `--nvtx_attention` flag to `parse_args()`

In your `parse_args()` function, add this argument:

```python
parser.add_argument("--nvtx_attention", action="store_true",
                    help="Enable NVTX-annotated attention for 1.1.4(e)")
```

This is a boolean flag — when you pass `--nvtx_attention` on the command line, `args.nvtx_attention` is `True`.

#### Step 3: Wrap `benchmark()` with NVTX ranges

Currently your `benchmark()` function looks like this (simplified):

```python
def benchmark(model, input_ids, mode, warmup_steps, measure_steps):
    # warmup
    for _ in range(warmup_steps):
        ...
    # measurement
    times = []
    for i in range(measure_steps):
        ...
    return times
```

Wrap each section with `nvtx.range(...)`:

```python
def benchmark(model, input_ids, mode, warmup_steps, measure_steps):
    # ─── Phase 1: Warm-up ───
    with nvtx.range("warmup"):                         # <-- NEW
        for _ in range(warmup_steps):
            if mode in ("backward", "both"):
                model.zero_grad(set_to_none=True)
            output = model(input_ids)
            if mode in ("backward", "both"):
                loss = output.sum()
                loss.backward()
            torch.cuda.synchronize()

    # ─── Phase 2: Measurement ───
    times = []
    with nvtx.range("measurement"):                    # <-- NEW
        for i in range(measure_steps):
            if mode in ("backward", "both"):
                model.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            start = timeit.default_timer()

            with nvtx.range(f"step_{i}"):               # <-- NEW
                with nvtx.range("forward"):              # <-- NEW
                    output = model(input_ids)

                if mode in ("backward", "both"):
                    with nvtx.range("backward"):         # <-- NEW
                        loss = output.sum()
                        loss.backward()

            torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append(end - start)

    return times
```

The NVTX ranges have **zero overhead** on your actual timing (they only affect the profiler). In nsys, you'll now be able to filter by these ranges to isolate warmup vs. measurement, forward vs. backward.

#### Step 4: Add the annotated attention function

Add this function after `generate_random_batch()` and before `benchmark()`:

```python
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """NVTX-annotated version of the staff's attention function.
    Used for 1.1.4(e) to compare softmax vs matmul time."""
    d_k = K.shape[-1]

    with nvtx.range("attention_matmul_QK"):
        attention_scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("attention_softmax"):
        attention_weights = torch.softmax(attention_scores, dim=-1)

    with nvtx.range("attention_matmul_V"):
        output = torch.einsum("...qk,...kd->...qd", attention_weights, V)

    return output
```

This is essentially the same logic as the staff's `scaled_dot_product_attention`, but with NVTX ranges around each sub-operation. The `math.sqrt(d_k)` is why we added `import math` earlier.

#### Step 5: Add monkey-patch logic in `main()`

In `main()`, right after parsing args and printing the config (before `create_model`), add:

```python
# Monkey-patch attention with NVTX annotations if requested (for 1.1.4e)
if args.nvtx_attention:
    a1_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    print("NVTX attention annotations: ENABLED")
```

That's it! When you run with `--nvtx_attention`, the staff model will call your annotated version instead of its own, and nsys will show separate timing for QK matmul, softmax, and V matmul.

### How to run

The SLURM script `student/run_1_1_4.sbatch` runs `nsys profile` for various model sizes, context lengths, and modes:

```bash
sbatch student/run_1_1_4.sbatch
```

An individual `nsys` command looks like:

```bash
uv run nsys profile -o profiles/fwd_small_128 --force-overwrite true \
  python student/benchmark.py --model_size small --context_length 128 \
  --mode forward --warmup_steps 5 --measure_steps 1
```

Key `nsys` flags:
- `-o <name>` -- output file name (produces `<name>.nsys-rep`)
- `--force-overwrite true` -- overwrite existing trace files

To get text stats: `nsys stats profiles/fwd_small_128.nsys-rep`

To view traces visually, download the `.nsys-rep` file to your local machine and open it in the Nsight Systems GUI app.

---

### (a) Total time on forward pass — does it match timeit?

**Question**: What is the total time spent on your forward pass? Does it match what we measured before with the Python standard library?

Profile the forward pass and use `nsys stats` to get the total GPU time:

```bash
# Profile
uv run nsys profile -o profiles/fwd_small_128 --force-overwrite true \
  python student/benchmark.py --model_size small --context_length 128 \
  --mode forward --warmup_steps 5 --measure_steps 1

# Get stats
nsys stats profiles/fwd_small_128.nsys-rep
```

Look at the "CUDA GPU Kernel Summary" — the sum of all kernel times should roughly match your `timeit` measurement from 1.1.3(b) (e.g., small forward ≈ 38.32 ms).

**RESULTS** (from `sbatch student/run_1_1_4.sbatch` and `run_1_1_4_kernels.sbatch`):

| Model | ctx | timeit (ms) | nsys measurement (ms) | OOM? |
|-------|-----|------------|----------------------|------|
| small | 128 | 36.83 | 36.85 | |
| small | 256 | 37.36 | 37.39 | |
| small | 512 | 46.27 | 46.31 | |
| small | 1024 | 103.91 | 103.95 | |
| medium | 128 | 75.35 | 75.38 | |
| medium | 256 | 74.67 | 74.69 | |
| medium | 512 | 137.19 | 137.23 | |
| medium | 1024 | — | — | OOM |
| large | 128 | 112.03 | 112.06 | |
| large | 256 | 141.75 | 141.78 | |
| large | 512 | 284.59 | 284.64 | |
| large | 1024 | — | — | OOM |
| xl | 128 | 152.31 | 152.34 | |
| xl | 256 | 280.63 | 280.69 | |
| xl | 512 | — | — | OOM |
| xl | 1024 | — | — | OOM |
| 2.7B | 128 | 219.37 | 219.42 | |
| 2.7B | 256 | 416.30 | 416.35 | |
| 2.7B | 512 | — | — | OOM |
| 2.7B | 1024 | — | — | OOM |

### Deliverable answer for 1.1.4(a) [1-2 sentences]

> The total forward pass time reported by nsys matches our timeit measurements almost exactly (within <1 ms), e.g., small model ctx=128 gives 36.83 ms from timeit vs 36.85 ms from nsys. This is expected because both methods use `torch.cuda.synchronize()` to ensure all GPU work completes before measuring, so they capture the same wall-clock interval.

---

### (b) Which CUDA kernel takes the most cumulative GPU time?

**Question**: What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is it invoked? Is it the same kernel that takes the most time in forward+backward?

Look at "CUDA GPU Kernel Summary" under "Stats Systems View", sorted by total time. The top kernel will likely be a GEMM (matrix multiplication) kernel.

**RESULTS** — CUDA GPU Kernel Summary (forward pass):

| Model | ctx | Top kernel | % of GPU time | Instances | Total sgemm % |
|-------|-----|-----------|--------------|-----------|---------------|
| small | 128 | `ampere_sgemm_128x64_tn` | 50.4% | 148 | ~72.6% |
| small | 256 | `ampere_sgemm_32x128_tn` | 36.5% | 240 | ~85.3% |
| small | 512 | `ampere_sgemm_128x32_tn` | 31.2% | 96 | ~75.9% |
| medium | 128 | `ampere_sgemm_32x128_tn` | 59.9% | 576 | ~83.4% |
| medium | 256 | `ampere_sgemm_128x64_tn` | 77.2% | 676 | ~83.1% |

The specific GEMM variant changes based on matrix dimensions, but all are matrix multiplication kernels. The same kernel type dominates during forward+backward — the backward pass adds more GEMM invocations for gradient computation.

### Deliverable answer for 1.1.4(b) [1-2 sentences]

> The CUDA kernel with the most cumulative GPU time is `ampere_sgemm_*` (matrix multiplication), accounting for 70-85% of forward pass time across all model sizes. For example, in the medium model at ctx=128, `ampere_sgemm_32x128_tn` alone takes 59.9% with 576 invocations. The same GEMM kernel type dominates in forward+backward mode, as the backward pass also performs matrix multiplications for gradient computation.

---

### (c) What other kernels account for non-trivial CUDA runtime?

**Question**: Although the vast majority of FLOPs are in matrix multiplications, what other kernels account for non-trivial CUDA runtime in the forward pass?

Look beyond the top GEMM entry in the kernel summary table.

**RESULTS** — Non-matmul kernels from GPU Kernel Summary (small model, ctx=512 as example):

| Kernel | What it does | % of GPU time |
|--------|-------------|--------------|
| `elementwise_kernel` (various) | Activations (GELU/SiLU), residual adds | ~4-6% |
| `reduce_kernel` (MeanOps) | LayerNorm mean computation | ~0.4% |
| `reduce_kernel` (MaxOps) | Softmax max for numerical stability | ~1.6% |
| `reduce_kernel` (func_wrapper) | LayerNorm variance computation | ~1.3% |
| `BinaryFunctor` / `BUnaryFunctor` | Element-wise mul/add (scaling, bias) | ~2.7% |
| `sigmoid_kernel_cuda` | SiLU activation (sigmoid component) | ~0.8% |
| `exp_kernel_cuda` | Softmax exponential | ~1.7% |
| `index_elementwise_kernel` | Embedding table lookup | ~0.4% |

### Deliverable answer for 1.1.4(c) [1-2 sentences]

> Besides matrix multiplications, the main non-trivial kernels are: elementwise kernels for activation functions and residual additions (~4-6%), reduction kernels for LayerNorm (mean/variance) and softmax (~3-4%), and the sigmoid/exp kernels for SiLU activation and softmax (~2%). These correspond to the non-matmul transformer operations: LayerNorm, activation functions, attention softmax, and embedding lookup.

---

### (d) Profile a full training step (forward + backward + optimizer)

**Question**: How does the fraction of time spent on matrix multiplication change when running a full training step (forward + backward + optimizer) vs. forward-only?

The assignment asks for a **complete training step**: forward → loss → backward → optimizer.step().
We added `--optimizer` flag to benchmark.py that creates an AdamW optimizer and calls `optimizer.step()` after each backward pass.

#### What we changed in benchmark.py for (d)

1. **Added `--optimizer` flag** in `parse_args()`
2. **Created AdamW optimizer** in `main()` when `--optimizer` is set
3. **Added `optimizer.step()` with NVTX range** in `benchmark()` after backward pass

Run with `--mode both --optimizer`:
```bash
nsys profile --trace cuda,nvtx --stats=true \
  --output profiles/train_small_128 --force-overwrite=true \
  uv run python student/benchmark.py --model_size small --context_length 128 \
  --mode both --optimizer --warmup_steps 3 --measure_steps 1
```

**Previous RESULTS (without optimizer)** — Forward vs Backward NVTX ranges:

| Model | ctx | Forward (ms) | Backward (ms) | Step total (ms) | Backward/Forward |
|-------|-----|-------------|--------------|-----------------|-----------------|
| small | 128 | 36.2 | 39.5 | 75.7 | 1.09x |
| small | 256 | 35.7 | 39.6 | 75.3 | 1.11x |
| small | 512 | 36.5 | 40.1 | 76.6 | 1.10x |
| small | 1024 | 36.7 | 121.6 | 158.3 | 3.31x |
| medium | 128 | 72.7 | 77.8 | 150.5 | 1.07x |
| medium | 256 | 71.1 | 91.6 | 162.8 | 1.29x |
| medium | 512 | 73.5 | 245.4 | 318.9 | 3.34x |
| medium | 1024 | 92.8 | 621.5 | 714.4 | 6.70x |
| large | 128 | 106.8 | 126.5 | 233.3 | 1.18x |
| large | 256 | 107.7 | 253.5 | 361.3 | 2.35x |
| large | 512 | 151.5 | 588.0 | 739.6 | 3.88x |
| xl | 128 | 144.9 | 255.4 | 400.4 | 1.76x |
| xl | 256 | 176.3 | 592.2 | 768.6 | 3.36x |
| xl | 512 | 357.6 | 1170.2 | 1527.8 | 3.27x |

**RESULTS WITH OPTIMIZER** (from `sbatch student/run_1_1_4d.sbatch`):

| Model | ctx | Forward (ms) | Backward (ms) | Optimizer (ms) | Step total (ms) | Optim % of step |
|-------|-----|-------------|--------------|----------------|-----------------|-----------------|
| small | 128 | 45.8 | 57.8 | 3.0 | 106.6 | 2.8% |
| small | 256 | 45.0 | 59.2 | 3.1 | 107.4 | 2.9% |
| small | 512 | 47.2 | 58.8 | 3.0 | 109.1 | 2.8% |

**CUDA kernel breakdown — matmul fraction in full training step (forward+backward+optimizer):**

| Model | ctx | sgemm % (full step) | sgemm % (forward-only) | Difference |
|-------|-----|--------------------|-----------------------|------------|
| small | 128 | ~63.5% | ~70-75% | -7 to -12 pp |
| small | 256 | ~70.1% | ~80-85% | -10 to -15 pp |
| small | 512 | ~69.6% | ~82% | -12 pp |

The optimizer adds `multi_tensor_apply_kernel` entries (~15% of GPU kernel time) — these are fused elementwise kernels for AdamW's momentum update, variance update, weight decay, and parameter step. The backward pass also adds elementwise gradient accumulation kernels that further reduce the matmul fraction.

### Deliverable answer for 1.1.4(d) [1-2 sentences]

> In a full training step (forward + backward + AdamW optimizer), the matmul fraction drops from ~70-85% (forward-only) to ~63-70% of GPU kernel time. The backward pass adds elementwise gradient kernels, and the AdamW optimizer adds fused multi-tensor kernels for momentum/variance/weight updates (~15% of GPU time), though the optimizer wall-clock time is very cheap (~3ms, <3% of step time) since these are simple elementwise operations.

---

### (e) Softmax vs. matmul time within attention

**Question**: Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

Run with `--nvtx_attention` to enable the annotated attention function:

```bash
uv run nsys profile -o profiles/attn_small_128 --force-overwrite true \
  python student/benchmark.py --model_size small --context_length 128 \
  --mode forward --warmup_steps 5 --measure_steps 1 --nvtx_attention
```

In the nsys output, filter by NVTX ranges to see:
- `attention_matmul_QK` — the Q×K^T matrix multiplication
- `attention_softmax` — the softmax operation
- `attention_matmul_V` — the attention_weights × V multiplication

**RESULTS** — Attention kernel breakdown with `--nvtx_attention` (GPU kernel times):

| Model | ctx | softmax kernel (ms) | softmax % | All sgemm (ms) | sgemm % |
|-------|-----|-------------------|-----------|----------------|---------|
| small | 128 | 0.37 | 0.7% | ~37.9 | ~74.6% |
| small | 256 | 0.97 | 0.9% | ~85.7 | ~85.5% |
| small | 512 | 3.85 | 1.9% | ~169.6 | ~82.1% |
| small | 1024 | 14.52 | 3.6% | ~314.1 | ~78.7% |
| medium | 128 | 0.95 | 0.6% | ~141.6 | ~84.8% |

The softmax kernel (`softmax_warp_forward`) is tiny compared to GEMM. As context length increases from 128→1024, softmax grows from 0.7% to 3.6% of GPU time.

**FLOPs comparison** (for attention with seq_len S, head_dim d_k):
- QK^T matmul: O(S² × d_k) FLOPs
- Softmax: O(S²) FLOPs
- V matmul: O(S² × d_k) FLOPs

With d_k = 64 (small model: 768/12 heads), matmul has ~64x more FLOPs than softmax per attention head. The runtime ratio (~80% vs ~1-4%) reflects this FLOPs gap, though softmax's share grows with context length because it becomes more memory-bandwidth-bound while matmul remains compute-bound.

### Deliverable answer for 1.1.4(e) [1-2 sentences]

> Softmax takes a tiny fraction of attention runtime compared to the matrix multiplications: ~0.7% at ctx=128 growing to ~3.6% at ctx=1024 for the small model, while the two matmul operations (QK^T and V) together account for ~75-85%. This matches the FLOPs difference — matmul computes O(S²·d_k) FLOPs with head_dim d_k=64, roughly 64x more than softmax's O(S²), and matmul is also more compute-bound (better utilizing tensor cores) while softmax is memory-bandwidth-bound.

---

## 1.1.5 Mixed Precision

### What this section is about (big picture)

In 1.1.4, we saw that transformers spend 70-85% of their GPU time on matrix multiplication.
**Mixed precision** is a technique that makes those matmuls ~2x faster by using smaller numbers.

Here's the simple idea:
- Normal training uses **FP32** (32-bit floating point) for all math.
- Mixed precision uses **BF16** (16-bit) for the heavy matmuls, but keeps **FP32** for the delicate stuff (normalization, loss).
- Since BF16 numbers are half the size, the GPU can crunch through them roughly twice as fast.

It's called "mixed" because you mix two precisions — fast BF16 where it's safe, accurate FP32 where it matters.

### Key concepts

#### What is a floating point number?

A number like `3.14` is stored in the computer as bits split into three parts:
- **Sign bit**: positive or negative (1 bit)
- **Exponent bits**: how big/small the number is (like scientific notation: 3.14 × 10^2)
- **Mantissa bits**: how precise the number is (how many decimal places)

| Type | Total Bits | Exponent | Mantissa | What it means |
|------|-----------|----------|----------|---------------|
| FP32 | 32 | 8 | 23 | Very precise, large range, but slow and uses lots of memory |
| FP16 | 16 | 5 | 10 | Fast, but tiny range — numbers can "overflow" and become infinity |
| BF16 | 16 | 8 | 7 | Fast, same range as FP32, but less precise (fewer decimal places) |

**Why BF16 is preferred over FP16 for training:**
- FP16 has only 5 exponent bits → numbers can overflow (become infinity) during training
- BF16 has 8 exponent bits (same as FP32) → same range, so no overflow risk
- BF16 has fewer mantissa bits (7 vs 10) → slightly less precise, but that's fine for matmuls
- The A100 GPU has special hardware ("tensor cores") that process BF16 matmuls at 312 TFLOPS vs only 19.5 TFLOPS for FP32 — that's **16x more throughput**!

#### What is `torch.autocast`?

`torch.autocast` is PyTorch's tool for automatic mixed precision. You wrap your forward pass in it:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(input_ids)  # matmuls inside here run in BF16
```

What happens inside:
- **Matrix multiplications** (the expensive 70-85% of GPU time) → automatically run in BF16
- **LayerNorm, softmax, loss** (delicate operations) → stay in FP32 for accuracy
- **Model weights stay in FP32** in memory — autocast creates temporary BF16 copies just for the computation
- **Gradients are computed in FP32** — backward pass is full precision

You don't have to manually change any dtypes. PyTorch figures out which ops to downcast.

#### What is `nullcontext()`?

A "do nothing" context manager. We use it so our code doesn't need if/else branches:

```python
from contextlib import nullcontext

# Without mixed precision: amp_context does nothing
# With mixed precision: amp_context activates BF16
amp_context = torch.autocast(...) if args.mixed_precision else nullcontext()

with amp_context:
    output = model(input_ids)  # works either way!
```

### What we changed in benchmark.py

We made **4 small changes** to support mixed precision:

1. **Added import**: `from contextlib import nullcontext`
2. **Added flag**: `--mixed_precision` in `parse_args()`
3. **In `main()`**: Create `amp_context` (either `torch.autocast` or `nullcontext`) and pass it to `benchmark()`
4. **In `benchmark()`**: Wrap `output = model(input_ids)` with `with amp_context:`

That's it. Now `--mixed_precision` tells PyTorch to run matmuls in BF16 on tensor cores.

---

### mixed_precision_accumulation [5 pts]

### What this sub-problem is about

Before we start using lower precision, this exercise shows **why low precision is dangerous for accumulation** (adding lots of small numbers together). This is the core reason mixed precision exists — you can't just do everything in FP16/BF16.

### The intuition (why precision matters for adding)

Imagine you have $8.00 in your bank account and someone deposits $0.01.
- In FP32 (23 mantissa bits): Your balance becomes $8.01 — it can represent both 8 and 0.01 at the same time.
- In FP16 (10 mantissa bits): The closest number to 8.01 it can represent is 8.0. So your deposit vanishes! The accumulator "stalls" because it can't represent the small difference.

This is exactly what happens in LayerNorm (computing mean/variance) and loss functions — they sum many values, and precision loss accumulates.

### The code: `student/mixed_precision_test.py`

We add 0.01 one thousand times. The correct answer is 10.0.

```python
# Snippet 1: FP32 accumulator + FP32 values → best possible
# Snippet 2: FP16 accumulator + FP16 values → precision loss (stalling)
# Snippet 3: FP32 accumulator + FP16 values → better, but 0.01 is already imprecise in FP16
# Snippet 4: FP32 accumulator + FP16→FP32 cast → same as 3 (damage already done at creation)
```

### How to run

```bash
uv run python student/mixed_precision_test.py
```

### Actual results

```
Snippet 1 — FP32 + FP32:       10.000133514404297
Snippet 2 — FP16 + FP16:       9.953125
Snippet 3 — FP32 + FP16:       10.00213623046875
Snippet 4 — FP32 + FP16→FP32:  10.00213623046875
```

### Why each snippet gives that result

- **Snippet 1** (FP32+FP32): 10.0001 — very close. FP32 has 23 mantissa bits — plenty of precision. The tiny 0.0001 error is normal floating-point rounding that accumulates over 1000 additions.
- **Snippet 2** (FP16+FP16): **9.953 — visibly wrong** (off by ~0.05). The accumulator `s` is FP16. Once `s` grows past ~8.0, adding 0.01 gets rounded away because FP16 can't represent the difference between 8.0 and 8.01. The sum "stalls" — this is the key danger of low-precision accumulation.
- **Snippet 3** (FP32+FP16): 10.002 — very close. The FP32 accumulator prevents stalling. The small overshoot comes from 0.01 being stored in FP16 as ~0.01000213... (slightly over), which adds up over 1000 iterations.
- **Snippet 4** (FP32+cast): Identical to Snippet 3 (10.002). Explicitly casting FP16→FP32 before adding doesn't change anything — the imprecision was already baked in when 0.01 was created as FP16.

**Key takeaway**: The FP32 accumulator (Snippets 3/4) prevents the stalling problem, making them much closer to 10.0 than pure FP16 (Snippet 2). This is why `torch.autocast` keeps accumulation-heavy operations (LayerNorm mean/variance, loss functions) in FP32.

### Deliverable answer for mixed_precision_accumulation

> FP32+FP32 gives 10.0001 (nearly perfect), while FP16+FP16 gives 9.953 — off by ~0.05 due to the accumulator stalling when the running sum grows too large for FP16 to represent small increments. Using an FP32 accumulator with FP16 values (Snippets 3/4) gives 10.002, much closer to correct, because the FP32 accumulator prevents stalling — the remaining small error comes from 0.01 being slightly imprecise in FP16 representation (stored as ~0.010002). This demonstrates why mixed precision keeps accumulation operations (LayerNorm, loss) in FP32.

---

### (a) ToyModel dtype inspection

### What this sub-problem is about

The PDF gives us a small `ToyModel` with: `Linear → ReLU → LayerNorm → Linear`. We run it inside `torch.autocast(dtype=torch.float16)` and print the dtype at each stage to see which operations autocast downcasts and which it keeps in FP32.

This proves autocast is truly **"mixed"** — it's not just "run everything in FP16."

### The ToyModel data flow (what autocast does to each layer)

```
Input (FP32)
    ↓
fc1 = Linear (matmul)  → autocast runs this in FP16 → output is FP16
    ↓
relu                    → stays FP16 (simple elementwise, no precision concern)
    ↓
LayerNorm               → autocast KEEPS this in FP32 (accumulation-sensitive!)
    ↓
fc2 = Linear (matmul)  → autocast runs this in FP16 → output is FP16
    ↓
cross_entropy loss      → computed in FP32 (loss needs precision)
    ↓
backward()              → gradients stored in FP32
```

**Why this pattern?**
- Matmuls (Linear layers) are the expensive 70-85% → run in FP16 for speed
- LayerNorm computes mean/variance (accumulation!) → must stay FP32 (remember mixed_precision_test.py)
- Loss function → must stay FP32 (small differences in loss guide training)
- Gradients → FP32 (used to update weights, need to be accurate)
- **Weights themselves never change dtype** — they stay FP32 in memory. Autocast just creates temporary FP16 copies for the computation.

### How to run

```bash
uv run python student/toymodel_mixed_precision.py
```

### Expected output

```
Model parameters dtype: torch.float32           ← weights are always FP32
Parameters inside autocast: torch.float32       ← autocast doesn't change stored weights
After fc1 + relu (output dtype): torch.float16  ← matmul ran in FP16
After LayerNorm (output dtype):  torch.float32  ← LayerNorm stays FP32!
Logits dtype:                    torch.float16  ← another matmul → FP16
Loss dtype:                      torch.float32  ← cross_entropy stays FP32
Gradient dtype (fc1.weight.grad): torch.float32 ← gradients are FP32
```

### Deliverable answer for 1.1.5(a)

> PENDING — fill in actual output after running on HPC. The expected pattern is: model parameters remain FP32 even inside autocast. Linear layer (matmul) outputs are FP16. LayerNorm output is FP32 (accumulation-sensitive). Loss is FP32. Gradients are FP32.

---

### (b) LayerNorm and precision

### What the question asks

The PDF asks: "FP16 autocast treats LayerNorm differently than Linear layers. What parts of LayerNorm are sensitive to precision? If we use BF16 instead of FP16, do we still need to keep LayerNorm in FP32?"

### Why LayerNorm needs FP32 (connecting to mixed_precision_accumulation)

LayerNorm computes:
```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

Step by step, here's what's precision-sensitive:
1. **mean(x)** = sum all values / count → this is **accumulation** (adding many numbers together)
2. **var(x)** = sum of (x - mean)² / count → also **accumulation**

This is exactly the problem from mixed_precision_test.py! When you sum many values in FP16, the accumulator stalls and you get the wrong answer. If the mean is even slightly off, the normalization is wrong, which cascades through the whole network.

### Do we still need FP32 with BF16?

**Yes.** BF16 has even fewer mantissa bits (7) than FP16 (10), so accumulation precision is actually **worse** with BF16. BF16's advantage is its larger range (8 exponent bits, no overflow), but that doesn't help with the accumulation problem. The mean/variance computation still needs FP32.

This is why PyTorch's autocast keeps LayerNorm in FP32 regardless of whether you use FP16 or BF16.

### Deliverable answer for 1.1.5(b) [2-3 sentences]

> The precision-sensitive parts of LayerNorm are the mean and variance computations, which involve summing many activation values — exactly the accumulation pattern that loses precision in low-precision formats (as demonstrated in mixed_precision_accumulation). With BF16 instead of FP16, we still need to keep LayerNorm in FP32 because BF16 has even fewer mantissa bits (7 vs 10), making accumulation errors worse despite its better dynamic range. PyTorch's autocast correctly keeps LayerNorm in FP32 for both FP16 and BF16 modes.

---

### (c) benchmarking_mixed_precision [5 pts]

### What this sub-problem is about

Now we actually measure how much faster BF16 is. We run each model size (small → 2.7B) with and without `--mixed_precision` and compare the forward+backward times.

### How it works in the code

In `benchmark.py`, we added:
```python
# In main():
if args.mixed_precision:
    amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    amp_context = nullcontext()   # does nothing — FP32 as usual

# In benchmark():
with amp_context:
    output = model(input_ids)   # matmuls inside run in BF16 if enabled
```

The backward pass does NOT need to be wrapped — PyTorch automatically computes gradients at the right precision based on what happened in the forward pass.

### How to run

```bash
sbatch student/run_1_1_5c.sbatch
```

This sbatch runs all three parts:
1. `mixed_precision_test.py` — the accumulation exercise
2. `toymodel_mixed_precision.py` — the dtype inspection for (a)
3. FP32 vs BF16 benchmark for all 5 model sizes at ctx=128, mode=both

### Results

**RESULTS PENDING — run `sbatch student/run_1_1_5c.sbatch` on HPC**

| Model | FP32 Fwd+Bwd (ms) | BF16 Fwd+Bwd (ms) | Speedup |
|-------|--------------------|--------------------|---------|
| small | ? | ? | ? |
| medium | ? | ? | ? |
| large | ? | ? | ? |
| xl | ? | ? | ? |
| 2.7B | ? | ? | ? |

### What to expect

- **BF16 should be 1.3-2x faster than FP32**, especially for larger models
- **Why?** The A100 tensor cores process BF16 matmuls at 312 TFLOPS vs 19.5 TFLOPS for FP32 — 16x more raw throughput. But the overall speedup is less than 16x because:
  - Not everything is matmul (LayerNorm, softmax stay FP32)
  - Memory bandwidth is also a bottleneck (not just compute)
  - Kernel launch overhead is the same regardless of precision
- **Larger models benefit more** — they spend a bigger fraction of time in matmuls (the part that gets faster), so the overall speedup is larger
- **Smaller models benefit less** — overhead is a larger fraction of their total time

### Deliverable answer for 1.1.5(c) [2-3 sentences]

> PENDING — fill in after running on HPC with actual numbers.

---

## 1.1.6 Profiling Memory [8 pts]

### What this section is about

Beyond timing, we need to understand how much GPU memory our models use. Memory is often the bottleneck -- you can't train a model that doesn't fit in GPU RAM. PyTorch provides tools to record detailed memory snapshots showing exactly where memory is allocated.

### Key concepts

#### GPU memory breakdown

During training, GPU memory is consumed by:

1. **Model parameters**: The weights themselves. A model with N parameters in FP32 uses N * 4 bytes.
2. **Gradients**: Same size as parameters (one gradient per parameter).
3. **Optimizer states**: Adam/AdamW stores 2 extra values per parameter (momentum + variance). That's 2 * N * 4 bytes = 8 bytes per parameter.
4. **Activations**: Intermediate outputs saved during the forward pass for use in the backward pass. This scales with batch_size * context_length * d_model * num_layers.
5. **Temporary buffers**: Workspace for matmul operations, etc.

For a model with P parameters in FP32:
- Parameters: P * 4 bytes
- Gradients: P * 4 bytes
- Optimizer states (AdamW): P * 8 bytes
- **Total for parameters alone: P * 16 bytes** (4x the model size!)
- Plus activations, which depend on batch size and sequence length

#### PyTorch memory profiling tools

```python
# Reset peak memory tracking
torch.cuda.reset_peak_memory_stats()

# Start recording allocation history
torch.cuda.memory._record_memory_history(max_entries=1000000)

# ... run your code ...

# Save snapshot to file (viewable in PyTorch memory visualizer)
torch.cuda.memory._dump_snapshot("snapshot.pickle")

# Stop recording
torch.cuda.memory._record_memory_history(enabled=None)

# Get peak memory
peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
```

The `.pickle` snapshot can be uploaded to https://pytorch.org/memory_viz to get an interactive visualization showing every allocation and deallocation over time.

#### The profile_memory() function in benchmark.py

```python
def profile_memory(model, input_ids, mode, warmup_steps, amp_context, args):
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Warmup with optimizer (so optimizer states are allocated)
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        with amp_context:
            output = model(input_ids)
        if mode in ("backward", "both"):
            loss = output.sum()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

    # Reset and start recording
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # One measured step with optimizer
    optimizer.zero_grad()
    with amp_context:
        output = model(input_ids)
    if mode in ("backward", "both"):
        loss = output.sum()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Save and report
    mp_tag = "_bf16" if args.mixed_precision else "_fp32"
    fname = f"memory_{args.model_size}_{args.context_length}_{args.mode}{mp_tag}.pickle"
    torch.cuda.memory._dump_snapshot(fname)
    torch.cuda.memory._record_memory_history(enabled=None)

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Memory snapshot saved to: {fname}")
    print(f"Peak memory allocated: {peak_mem_mb:.1f} MB")
```

Key differences from the timing benchmark:
- **Includes an optimizer** (`AdamW`) -- because in real training, optimizer states are a major memory consumer.
- **Uses `reset_peak_memory_stats()`** -- to only count memory from the measured step, not warmup.
- **Records memory history** -- produces a detailed trace of every allocation/deallocation.
- **Only one measured step** -- we only need one step to see the memory pattern.

### How to run

```bash
# Single example: 2.7B model, context_length=128, forward+backward, FP32
uv run python student/benchmark.py \
  --model_size 2.7B --context_length 128 --mode both --profile_memory

# With BF16
uv run python student/benchmark.py \
  --model_size 2.7B --context_length 128 --mode both --profile_memory --mixed_precision

# Full sweep via SLURM
sbatch student/run_1_1_6.sbatch
```

The SLURM script runs the 2.7B model at context lengths 128, 256, 512, in both forward-only and both modes, in both FP32 and BF16.

---

### (a) Memory timeline images — forward vs. full training step

**Question**: Run your script to get a memory profile of the 2.7B model when doing inference only (just forward pass) and a full training step. How do your memory timelines look? Can you tell which stage is running based on the peaks you see?

**Deliverable**: Two images of the "Active memory timeline" from the `memory_viz` tool: one for forward pass, one for full training step (forward + backward + optimizer). Plus a 2-3 sentence response.

```bash
# Forward only
uv run python student/benchmark.py \
  --model_size 2.7B --context_length 128 --mode forward --profile_memory --warmup_steps 5

# Full training step
uv run python student/benchmark.py \
  --model_size 2.7B --context_length 128 --mode both --profile_memory --warmup_steps 5
```

This produces two `.pickle` files. Upload them to https://pytorch.org/memory_viz to visualize.

**RESULTS PENDING — run `sbatch student/run_1_1_6.sbatch`**

**How to interpret the memory timeline**:
- **Forward-only**: You should see memory climb as each layer's activations are computed, then flatten. No gradient or optimizer memory.
- **Full training step**: Memory climbs during forward (activations saved), peaks during backward (gradients allocated), then drops as activations are freed. A final bump appears for the optimizer step (AdamW momentum + variance updates).

### Deliverable answer for 1.1.6(a) [2-3 sentences]

> PENDING — fill in after running. Expected: The forward-only timeline shows a steady climb as activations accumulate across layers. The full training step shows three distinct phases: (1) forward pass where memory grows as activations are saved, (2) backward pass where memory peaks as gradients are computed and activations are freed, and (3) optimizer step where a brief allocation occurs for AdamW state updates.

---

### (b) Peak memory table by context length

**Question**: What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?

**Deliverable**: A table with two numbers per context length.

```bash
for CTX in 128 256 512; do
  uv run python student/benchmark.py \
    --model_size 2.7B --context_length $CTX --mode forward --profile_memory --warmup_steps 5
  uv run python student/benchmark.py \
    --model_size 2.7B --context_length $CTX --mode both --profile_memory --warmup_steps 5
done
```

**RESULTS PENDING — run `sbatch student/run_1_1_6.sbatch`**

Expected format:

| Context Length | Forward Peak (MB) | Full Training Peak (MB) |
|---------------|-------------------|------------------------|
| 128 | ? | ? |
| 256 | ? | ? |
| 512 | ? | ? |

### Deliverable answer for 1.1.6(b)

> PENDING — fill in after running. Expected: Peak memory increases with context length because activation memory scales with `batch_size × context_length × d_model × num_layers`. Full training uses significantly more than forward-only because it stores activations for the backward pass, plus gradients and optimizer states.

---

### (c) Mixed-precision memory — does it help?

**Question**: Find the peak memory usage of the 2.7B model when using mixed-precision, for both a forward pass and a full optimizer step. Does mixed-precision significantly affect memory usage?

**Deliverable**: A 2-3 sentence response.

```bash
# Forward + BF16
uv run python student/benchmark.py \
  --model_size 2.7B --context_length 128 --mode forward --profile_memory --mixed_precision --warmup_steps 5

# Full training + BF16
uv run python student/benchmark.py \
  --model_size 2.7B --context_length 128 --mode both --profile_memory --mixed_precision --warmup_steps 5
```

**RESULTS PENDING — run `sbatch student/run_1_1_6.sbatch`**

### Deliverable answer for 1.1.6(c) [2-3 sentences]

> PENDING — fill in after running. Expected: BF16 reduces memory somewhat because activations are stored in half precision (2 bytes vs 4 bytes). However, model parameters, gradients, and optimizer states remain in FP32 under autocast, so total savings are less than 2x. The savings are most visible at longer context lengths where activation memory dominates.

---

### (d) Activation tensor size calculation

**Question**: Consider the 2.7B model. At our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MB (i.e., divide the number of bytes by 1024^2).

**Deliverable**: A 1-2 sentence response with your derivation.

This is a **math question** — no code to run.

**Calculation**:
- The 2.7B model has `d_model = 2560`
- A residual stream activation has shape: `(batch_size, context_length, d_model)` = `(4, 128, 2560)` at our default settings
- In FP32, each value is 4 bytes
- Total bytes: `4 × 128 × 2560 × 4` = `5,242,880` bytes
- In MB: `5,242,880 / (1024^2)` = **5.0 MB**

At context_length=1024: `4 × 1024 × 2560 × 4 = 41,943,040 bytes = 40.0 MB`

### Deliverable answer for 1.1.6(d) [1-2 sentences]

> A single activation tensor in the 2.7B model's residual stream has shape (4, context_length, 2560) in FP32. At context_length=128 this is 4 × 128 × 2560 × 4 bytes = 5.0 MB; at context_length=1024, it is 40.0 MB. Each transformer layer produces (and must save for backward) at least one such tensor.

---

### (e) Largest allocations in memory_viz

**Question**: Now look closely at the "Active Memory Timeline" from `pytorch.org/memory_viz` of a memory snapshot of the 2.7B model doing a forward pass. When you reduce the "Detail" level, what is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?

**Deliverable**: A 1-2 sentence response.

Upload the forward-only `.pickle` file to https://pytorch.org/memory_viz. Use the "Detail" slider to filter — at ~10%, only the 10% largest allocations are shown.

**RESULTS PENDING — run `sbatch student/run_1_1_6.sbatch`, then visualize the pickle file**

### Deliverable answer for 1.1.6(e) [1-2 sentences]

> PENDING — fill in after running. Expected: The largest allocations should be ~5 MB each (matching our calculation in (d)) and come from the residual stream activations at each transformer layer. The stack traces should point to the forward pass of each `TransformerBlock`, specifically the output of attention and feed-forward sublayers.

---

## Summary of all SLURM scripts

| Script | Section | What it does |
|--------|---------|-------------|
| `student/run_1_1_3b.sbatch` | 1.1.3(b) | Timing sweep: all 5 model sizes, forward + both modes |
| `student/run_1_1_3c.sbatch` | 1.1.3(c) | Warmup effect: warmup=0,1,2,5 for small model |
| `student/run_1_1_4.sbatch` | 1.1.4 | Nsight Systems profiling: various sizes, ctx lengths, NVTX |
| `student/run_1_1_5c.sbatch` | 1.1.5 | Mixed precision: accumulation test, toymodel, FP32 vs BF16 sweep |
| `student/run_1_1_6.sbatch` | 1.1.6 | Memory profiling: 2.7B model, various ctx lengths, FP32 + BF16 |

All SLURM scripts use the `:ro` (read-only) overlay so they can run in parallel without filesystem conflicts.

To submit all at once:

```bash
cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism
sbatch student/run_1_1_3b.sbatch
sbatch student/run_1_1_3c.sbatch
sbatch student/run_1_1_4.sbatch
sbatch student/run_1_1_5c.sbatch
sbatch student/run_1_1_6.sbatch
```

## Quick reference: key PyTorch APIs used

| API | Purpose | Section |
|-----|---------|---------|
| `torch.cuda.synchronize()` | Wait for GPU to finish all pending work | 1.1.3 |
| `timeit.default_timer()` | High-resolution wall-clock timer | 1.1.3 |
| `torch.cuda.nvtx.range()` | Label code regions for NVIDIA profiler | 1.1.4 |
| `torch.autocast()` | Automatic mixed precision context | 1.1.5 |
| `contextlib.nullcontext()` | No-op context manager (FP32 fallback) | 1.1.5 |
| `torch.cuda.reset_peak_memory_stats()` | Reset peak memory counter | 1.1.6 |
| `torch.cuda.memory._record_memory_history()` | Start/stop memory allocation recording | 1.1.6 |
| `torch.cuda.memory._dump_snapshot()` | Save memory snapshot to file | 1.1.6 |
| `torch.cuda.max_memory_allocated()` | Query peak memory usage | 1.1.6 |

---

# Section 1.2 -- Optimizing Attention with FlashAttention-2

## 1.2.1 Benchmarking PyTorch Attention

### What this section is about

Before optimizing attention, we need to benchmark the current (naive) attention implementation to understand its performance and memory characteristics. The naive attention computes the full `seq_len x seq_len` attention score matrix, which grows quadratically with sequence length and causes out-of-memory errors at long sequences.

The attention formula is:

```
Attention(Q, K, V) = softmax(mask(Q x K^T / sqrt(d_k))) x V
```

The problem: storing the full attention matrix of shape `(batch_size, seq_len, seq_len)` requires `O(n^2)` memory, which becomes enormous at long sequence lengths.

### Key concepts

#### Why attention is the bottleneck

For a single attention head:
- **QK matmul**: `(batch, seq_len, d) x (batch, d, seq_len)` -> `(batch, seq_len, seq_len)` -- O(n^2 d) FLOPs
- **Softmax**: over `(batch, seq_len, seq_len)` -- O(n^2) operations
- **V matmul**: `(batch, seq_len, seq_len) x (batch, seq_len, d)` -> `(batch, seq_len, d)` -- O(n^2 d) FLOPs
- **Memory for attention scores**: `batch x seq_len x seq_len x 4 bytes` (FP32)

At seq_len=16384 with batch=8: the attention matrix alone is `8 x 16384 x 16384 x 4 = 32 GB` -- larger than most GPUs!

### Problem (pytorch_attention): 5 points

**(a)** Write a benchmarking script that:
1. Fix batch_size=8, no multihead attention (remove head dimension)
2. Iterate through the cartesian product of:
   - `d_model` in [16, 32, 64, 128]
   - `seq_len` in [256, 1024, 4096, 8192, 16384]
3. Create random Q, K, V inputs of the appropriate size
4. Time 100 forward passes through the staff's attention function
5. Measure memory in use before backward starts, then time 100 backward passes
6. Warm up first! Call `torch.cuda.synchronize()` after each pass

**What we need to build**: A new script `student/benchmark_attention.py`

#### Pseudocode

```
import itertools
from a1_basics.model import scaled_dot_product_attention

batch_size = 8
d_models = [16, 32, 64, 128]
seq_lens = [256, 1024, 4096, 8192, 16384]

for d_model, seq_len in itertools.product(d_models, seq_lens):
    try:
        # Create random Q, K, V on GPU
        Q = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)

        # Build causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda")).bool()

        # Warmup
        for _ in range(5):
            out = scaled_dot_product_attention(Q, K, V, mask)
            torch.cuda.synchronize()

        # Time 100 forward passes
        torch.cuda.synchronize()
        start = timeit.default_timer()
        for _ in range(100):
            out = scaled_dot_product_attention(Q, K, V, mask)
            torch.cuda.synchronize()
        fwd_time = (timeit.default_timer() - start) / 100

        # Measure memory before backward
        mem_before_bwd = torch.cuda.memory_allocated()

        # Time 100 backward passes
        torch.cuda.synchronize()
        start = timeit.default_timer()
        for _ in range(100):
            out = scaled_dot_product_attention(Q, K, V, mask)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
        bwd_time = (timeit.default_timer() - start) / 100

        print(f"d={d_model}, seq={seq_len}: fwd={fwd_time*1000:.2f}ms, bwd={bwd_time*1000:.2f}ms, mem={mem_before_bwd/1e6:.1f}MB")

    except torch.cuda.OutOfMemoryError:
        print(f"d={d_model}, seq={seq_len}: OOM")
        torch.cuda.empty_cache()
```

#### Important details

- **No multihead**: Use `scaled_dot_product_attention(Q, K, V, mask)` directly -- Q, K, V have shape `(batch, seq_len, d)`, no head dimension.
- **Mask**: The staff's attention expects a boolean mask. Build a lower-triangular causal mask.
- **OOM handling**: Wrap each config in try/except for `torch.cuda.OutOfMemoryError`. Call `torch.cuda.empty_cache()` after OOM.
- **Memory measurement**: Call `torch.cuda.memory_allocated()` after the forward pass but before backward to see how much memory the attention scores consume.

**File**: `student/benchmark_attention.py`

**How to run**:
```bash
uv run python student/benchmark_attention.py
```

**RESULTS PENDING -- run on HPC**

Expected results format:

| d_model | seq_len | Fwd (ms) | Bwd (ms) | Memory (MB) | Notes |
|---------|---------|----------|----------|-------------|-------|
| 16 | 256 | ? | ? | ? | |
| ... | ... | ... | ... | ... | |
| 128 | 16384 | ? | ? | ? | OOM? |

### Deliverable answer for pytorch_attention [table + 1-2 paragraph response]

> PENDING -- fill in after running. Report the timings table, note which configs OOM, do the memory accounting for one config that OOMs (batch x seq^2 x 4 bytes for FP32 attention scores), explain how backward memory scales with seq_len (saved attention matrix for gradient computation), and mention FlashAttention as the solution to eliminate the O(n^2) memory.

---

## 1.2.2 torch.compile (Problem: torch_compile, 5 points)

### What this section is about

`torch.compile` is PyTorch's JIT compiler (since PyTorch 2.0). It analyzes your computation graph and tries to fuse operations into optimized Triton kernels automatically. We'll test whether it speeds up our attention and full model.

### Key concepts

#### How torch.compile works

```python
# Original
output = model(input)

# Compiled -- first call is slow (compilation), subsequent calls are fast
compiled_model = torch.compile(model)
output = compiled_model(input)
```

`torch.compile` traces the computation graph on the first call, generates optimized GPU kernels, and caches them. The first call is slow (compilation overhead), but subsequent calls can be significantly faster because:
- Operations get **fused** (multiple elementwise ops become one kernel)
- Memory access patterns get **optimized**
- Unnecessary intermediate tensors get **eliminated**

### Problem (torch_compile): 5 points

**(a)** Extend your attention benchmarking script to include a compiled version of the PyTorch attention, and compare its performance to the uncompiled version with the same configurations as `pytorch_attention`.

**How to compile the attention function**:
```python
from a1_basics.model import scaled_dot_product_attention

# Compile the attention function
compiled_attention = torch.compile(scaled_dot_product_attention)

# Use it exactly like the original
out = compiled_attention(Q, K, V, mask)
```

**Important**: The first call triggers compilation and is very slow (30-60s). Make sure your warmup accounts for this -- use extra warmup steps for the compiled version.

**Deliverable**: A table comparing compiled vs. uncompiled forward and backward pass timings.

**(b)** Compile your entire Transformer model and compare performance.

```python
model = create_model(args.model_size, args.vocab_size, args.context_length)
compiled_model = torch.compile(model)

# Use compiled_model in your benchmark instead of model
```

**Deliverable**: A table comparing vanilla vs. compiled Transformer model for forward, backward, and optimizer steps.

**RESULTS PENDING -- build the script and run on HPC**

### Deliverable answers for torch_compile

> PENDING -- fill in after running. Expected: torch.compile should provide modest speedups (1.1-1.5x) for attention by fusing elementwise operations. For the full model, speedups may be more significant because there are more fusion opportunities across layers. The compiled version should NOT fix the O(n^2) memory problem -- that requires FlashAttention.

---

## 1.3 FlashAttention-2

### What this section is about

This is the big implementation section. We'll implement FlashAttention-2, which computes attention **without ever materializing the full `seq_len x seq_len` attention score matrix**. Instead, it processes attention in small tiles, keeping only tile-sized chunks in fast GPU SRAM at a time.

This has two huge benefits:
1. **Memory**: O(n) instead of O(n^2) -- can handle much longer sequences
2. **Speed**: Fewer reads/writes to slow GPU HBM memory -- often faster despite doing some recomputation

### Key concepts

#### The memory problem with naive attention

Naive attention computes:
```
S = Q x K^T / sqrt(d)     # shape: (batch, seq_len, seq_len) -- O(n^2) memory!
P = softmax(S)             # same shape
O = P x V                  # shape: (batch, seq_len, d)
```

The S and P matrices are `(seq_len x seq_len)` -- at seq_len=16384, that's 268M entries x 4 bytes = 1GB **per batch element**. This is why long sequences cause OOM.

#### FlashAttention's solution: tiling + online softmax

FlashAttention splits Q into tiles of size `B_q` and K,V into tiles of size `B_k`. For each Q tile, it iterates over all K,V tiles and:
1. Computes a small `B_q x B_k` attention score tile (fits in SRAM!)
2. Updates a running softmax using the **online softmax trick** (tracks running max `m` and running sum `l`)
3. Accumulates the output `O` tile incrementally

The maximum memory for attention scores is now `B_q x B_k` (e.g., 128 x 128 = 16K entries) instead of `seq_len x seq_len`.

#### Online softmax trick

Normal softmax requires seeing ALL values to compute the denominator. Online softmax maintains running statistics:
- `m` = running row-wise maximum (for numerical stability)
- `l` = running softmax denominator (sum of exp)
- `O` = running output accumulator (rescaled as m and l update)

As each new K tile arrives, we update m, rescale the previous O, and accumulate the new contribution.

#### Logsumexp (L) -- saved for backward

FlashAttention saves `L = m + log(l)` (the logsumexp of attention scores) instead of the full attention matrix. This single vector per query row is enough to recompute attention scores during the backward pass.

#### Backward pass with recomputation

Instead of saving the giant `P` matrix for backward, FlashAttention:
1. **Recomputes** P from Q, K, and the saved L during backward
2. This trades compute for memory -- we do the QK matmul twice, but never store O(n^2) activations

The backward pass formulas (from the PDF):
```
dV = P^T x dO
dP = dO x V^T
dS = P * (dP - D)          where D = rowsum(O * dO)
dQ = dS x K / sqrt(d)
dK = dS^T x Q / sqrt(d)
```

### 1.3.1 Weighted Sum Example (provided by assignment)

The assignment walks through a complete Triton kernel example (weighted sum) to teach you:
- **Triton basics**: `@triton.jit`, `tl.program_id`, `tl.make_block_ptr`, `tl.load`, `tl.store`
- **Tiling**: Processing data in blocks of `ROWS_TILE_SIZE x D_TILE_SIZE`
- **Block pointers**: How to set up and advance pointers through memory
- **Autograd integration**: `torch.autograd.Function` with `forward()` and `backward()`

Study the weighted sum code in the PDF carefully -- it's the template for your FlashAttention kernel.

Key Triton concepts:
- `tl.program_id(0)` -- which thread block am I? (like the loop index)
- `tl.make_block_ptr(...)` -- create a pointer to a tile of data
- `tl.load(block_ptr, boundary_check=..., padding_option="zero")` -- load a tile from memory
- `tl.store(block_ptr, data, boundary_check=...)` -- write a tile to memory
- `block_ptr.advance((row_delta, col_delta))` -- move the pointer to the next tile

---

### 1.3.2 Problem (flash_forward): 25 points

**(a)** Write a **pure PyTorch** (no Triton) `autograd.Function` that implements the FlashAttention-2 forward pass.

#### What you need to build

A class that inherits from `torch.autograd.Function` with:
- `forward(ctx, Q, K, V, is_causal=False)` -> returns O (the attention output)
- Saves L, Q, K, V, O for the backward pass using `ctx.save_for_backward`
- `backward(...)` -> raises `NotImplementedError` for now

#### Interface

```python
class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Q: (batch, N_q, d)
        # K: (batch, N_k, d)
        # V: (batch, N_k, d)
        # Returns: O of shape (batch, N_q, d)
        ...

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
```

#### Algorithm (from PDF Algorithm 1)

```
Choose tile sizes B_q, B_k (e.g., 16 or 32 -- must be at least 16)

Split Q into T_q tiles of size B_q
Split K, V into T_k tiles of size B_k

for i in range(T_q):      # for each query tile
    Load Q_i from Q[:, i*B_q:(i+1)*B_q, :]

    Initialize:
        O_i = zeros(batch, B_q, d)
        l_i = zeros(batch, B_q)          # running softmax denominator
        m_i = -inf * ones(batch, B_q)    # running max

    for j in range(T_k):  # for each key tile
        Load K_j, V_j

        # Compute attention scores for this tile
        S_ij = Q_i @ K_j^T / sqrt(d)     # (batch, B_q, B_k)

        # Update running max
        m_new = max(m_i, rowmax(S_ij))    # (batch, B_q)

        # Compute unnormalized softmax
        P_ij = exp(S_ij - m_new[:, :, None])  # (batch, B_q, B_k)

        # Update running denominator
        l_i = exp(m_i - m_new) * l_i + rowsum(P_ij)

        # Rescale previous O and add new contribution
        O_i = diag(exp(m_i - m_new)) @ O_i + P_ij @ V_j

        m_i = m_new

    # Final normalization
    O_i = diag(1/l_i) @ O_i
    L_i = m_i + log(l_i)   # logsumexp for backward

    Write O_i, L_i to output
```

#### Key implementation notes

- **Batch dimension**: All operations should be batched. Use `[:, tile_start:tile_end, :]` to index tiles.
- **Tile sizes**: Choose B_q and B_k that are powers of 2, at least 16. The tests use dimensions that are multiples of these.
- **`diag(x) @ M`** means element-wise multiply each row of M by the corresponding element of x. In code: `M * x.unsqueeze(-1)` or `M * x[:, :, None]`.
- **Save for backward**: `ctx.save_for_backward(L, Q, K, V, O)` -- the test checks that exactly one saved tensor has shape `(batch, N_q)` which is L.
- **is_causal**: For now you can ignore this (set `is_causal=False`).

#### File and testing

**File**: `student/flash_attention.py` (create this new file)

**Adapter**: In `tests/adapters.py`, implement:
```python
def get_flashattention_autograd_function_pytorch():
    from student.flash_attention import FlashAttentionPyTorch
    return FlashAttentionPyTorch
```

**Test**:
```bash
uv run pytest -k test_flash_forward_pass_pytorch
```

The test creates Q, K, V of shape `(4, 128, 64)`, runs your implementation, checks that O and L match the reference (within rtol=1e-2, atol=1e-2).

### Deliverable for flash_forward (a)

> A `torch.autograd.Function` subclass implementing the FlashAttention-2 forward pass in pure PyTorch. Must pass `test_flash_forward_pass_pytorch`.

---

**(b)** Write a **Triton kernel** for the FlashAttention-2 forward pass.

This is the same algorithm as (a), but implemented as a Triton GPU kernel for much better performance.

#### What you need to build

1. A `@triton.jit` kernel function `flash_fwd_kernel(...)`
2. A `torch.autograd.Function` subclass that launches the kernel

#### Kernel structure

The assignment provides the function declaration:

```python
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    ...
```

Key points:
- **Launch grid**: `(T_q, batch_size)` -- each program instance handles one query tile in one batch element
- **Use `tl.make_block_ptr`** to set up pointers for Q, K, V, O, L tiles
- **Accumulate in FP32**: Use `tl.float32` for O, l, m buffers (even if inputs are FP16)
- **Cast P to V's dtype** before the `tl.dot(P, V)` multiplication
- **Cast O back** to the input dtype before writing to global memory
- **Use `acc=` argument** in `tl.dot` for accumulation: `acc = tl.dot(P, V_tile, acc=acc)`

#### Precision tips from the PDF

- On-chip buffers (O, l, m) should be `tl.float32`
- Cast `P` to `V`'s dtype before `tl.dot`
- Cast final `O` to the output pointer's dtype before `tl.store`
- Get dtypes with `tensor.dtype` or `block_ptr.type.element_ty`
- Cast with `tensor.to(dtype)`

#### Autograd wrapper

```python
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, N_q, d = Q.shape
        N_k = K.shape[1]

        O = torch.empty_like(Q)
        L = torch.empty(batch, N_q, device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE = ...  # e.g., 64
        K_TILE_SIZE = ...  # e.g., 64
        scale = 1.0 / (d ** 0.5)

        grid = (triton.cdiv(N_q, Q_TILE_SIZE), batch)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
```

#### File and testing

**File**: `student/flash_attention.py` (same file, add the Triton class)

**Adapter**: In `tests/adapters.py`:
```python
def get_flashattention_autograd_function_triton():
    from student.flash_attention import FlashAttentionTriton
    return FlashAttentionTriton
```

**Test**:
```bash
uv run pytest -k test_flash_forward_pass_triton
```

This test runs on GPU and checks both `is_causal=False` and `is_causal=True`.

### Deliverable for flash_forward (b)

> A Triton kernel implementing FlashAttention-2 forward pass, wrapped in a `torch.autograd.Function`. Must pass `test_flash_forward_pass_triton`.

**RESULTS PENDING -- implement and test**

---

## Summary of all files for Section 1.2

| File | What it contains |
|------|-----------------|
| `student/benchmark_attention.py` | Benchmarks naive attention (1.2.1) and torch.compile (1.2.2) |
| `student/flash_attention.py` | FlashAttention-2 implementations -- PyTorch (1.3.2a) and Triton (1.3.2b) |
| `tests/adapters.py` | Hook up your implementations to the test suite |
