'''
Parse command-line arguments (model size, context length, warmup steps, etc.)
Initialize a model from the given hyperparameters
Generate random input data
Run warm-up steps (not timed)
Time n measurement steps for forward and/or backward passes
Call torch.cuda.synchronize() after each step
Report mean and standard deviation
'''


# STRUCTURE

'''
benchmark.py
│
├── Imports
├── 1. MODEL_CONFIGS dictionary         ← so we can pick a model by name
├── 2. parse_args()                     ← accept command-line arguments
├── 3. create_model()                   ← build a model from a config
├── 4. generate_random_batch()          ← create fake input data
├── 5. annotated_attention()            ← NVTX-annotated attention for 1.1.4(e)
├── 6. benchmark()                      ← the core: warmup + timed runs (with NVTX)
├── 7. profile_memory()                 ← memory profiling for 1.1.6
└── 8. main()                           ← glue everything together
'''


import argparse          # For command-line argument parsing
import timeit            # For high-resolution timing
import math
import torch
import torch.cuda.nvtx as nvtx   # NVTX annotations for nsys profiling (Section 1.1.4)
import numpy as np
from contextlib import nullcontext  # No-op context manager (for mixed precision toggle)
# see notes for more info on nullcontext

import a1_basics.model
from a1_basics.model import BasicsTransformerLM
from a1_basics.optimizer import AdamW



# ───----      1. Model Configurations (from Table 1 in the assignment) ───



MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}



# -------        2. parse_args()                     -----------------------



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



# -------        3. create_model()                     -----------------------
'''
function create_model(model_size, vocab_size, context_length):
    1. Look up the config dict for this model_size
    2. Create a BasicsTransformerLM with those hyperparameters
    3. Move it to GPU
    4. Return it
'''


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
    model = model.to("cuda")  # Move model to GPU
    return model




 #------        4. generate_random_batch()             -----------------------

'''
Why random data at all? We're measuring hardware speed, not model quality.
The GPU does the exact same matrix multiplications whether the input is "The cat sat"
or random integers. Random data lets us skip the entire data loading/tokenization pipeline.

function generate_random_batch(batch_size, context_length, vocab_size):
    return random integers of shape (batch_size, context_length), range [0, vocab_size)
'''

def generate_random_batch(batch_size, context_length, vocab_size):
    """Generate a random batch of token IDs on GPU."""
    return torch.randint(0, vocab_size, (batch_size, context_length), device="cuda")

# Why device="cuda"? Creating tensors directly on the GPU avoids a CPU→GPU transfer,
# which would add noise to your timings. Since we're using random data anyway,
# there's no reason to create it on CPU first.


#-------        5. annotated_attention() for 1.1.4(e)  -----------------------

'''
For Section 1.1.4(e): Compare softmax vs matmul time within attention.
We monkey-patch the staff's scaled_dot_product_attention with NVTX ranges
so nsys can show us exactly how long each sub-operation takes.

The staff's original function (a1_basics.model.scaled_dot_product_attention) uses
einsum for matmul and softmax. We wrap each part with nvtx.range().
'''

def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """NVTX-annotated version of the staff's attention function."""
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


#-------        6. benchmark()                        -----------------------


# Goal: Run the model N times and measure how long each run takes.
'''
IMP: see notes for step by step build up of the benchmark() function
base + async + warm up + backward pass + mixed precision + memory profiling

function benchmark(model, input_ids, num_steps):
    times = []
    for each step:
        start timer
        run model(input_ids)
        stop timer
        record (stop - start)
    return times
'''

def benchmark(model, input_ids, mode, warmup_steps, measure_steps, amp_context):
    """
    Run warmup, then timed measurement steps.
    Returns list of times in seconds.
    NVTX ranges added so nsys can distinguish warmup/measurement/forward/backward.
    """
    # ─── Phase 1: Warm-up (not timed, wrapped in NVTX so we can filter it out in nsys) ───
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

    # ─── Phase 2: Measurement ───
    times = []
    with nvtx.range("measurement"):
        for i in range(measure_steps):
            if mode in ("backward", "both"):
                model.zero_grad(set_to_none=True)

            torch.cuda.synchronize()           # GPU idle before timing
            start = timeit.default_timer()

            with nvtx.range(f"step_{i}"):
                with nvtx.range("forward"):
                    with amp_context:
                        output = model(input_ids)

                if mode in ("backward", "both"):
                    with nvtx.range("backward"):
                        loss = output.sum()
                        loss.backward()

            torch.cuda.synchronize()           # GPU finished before stopping timer
            end = timeit.default_timer()

            times.append(end - start)

    return times


#-------        7. profile_memory() for 1.1.6         -----------------------

'''
For Section 1.1.6: Memory profiling using PyTorch's memory recorder.

How it works:
1. Warm up (so CUDA context + memory pools are initialized)
2. Reset peak memory stats
3. Start recording memory history
4. Run exactly ONE step (forward, backward, optimizer)
5. Save snapshot to .pickle file
6. Stop recording
7. Report peak memory

The .pickle file can be visualized at https://pytorch.org/memory_viz
'''

def profile_memory(model, input_ids, mode, warmup_steps, amp_context, args):
    """Run memory profiling for one step after warm-up."""
    # We use torch.optim.AdamW here (not the staff's AdamW) because the assignment
    # asks for a "full training step" which includes the optimizer.
    # AdamW stores 2 extra tensors per parameter (momentum + variance).
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Warm-up (not recorded) — same reason as in benchmark()
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        with amp_context:
            output = model(input_ids)
        if mode in ("backward", "both"):
            loss = output.sum()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

    # Reset peak memory stats so we measure just the next step
    torch.cuda.reset_peak_memory_stats()

    # Start recording
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # One step (forward + optional backward + optional optimizer)
    optimizer.zero_grad()
    with amp_context:
        output = model(input_ids)
    if mode in ("backward", "both"):
        loss = output.sum()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Save and stop
    mp_tag = "_bf16" if args.mixed_precision else "_fp32"
    fname = f"memory_{args.model_size}_{args.context_length}_{args.mode}{mp_tag}.pickle"
    torch.cuda.memory._dump_snapshot(fname)
    torch.cuda.memory._record_memory_history(enabled=None)

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Memory snapshot saved to: {fname}")
    print(f"Peak memory allocated: {peak_mem_mb:.1f} MB")


#-------        8. main()                             -----------------------


def main():
    args = parse_args()

    # Print config so you can verify what's running
    print(f"=== Benchmarking {args.model_size} model ===")
    print(f"Context length: {args.context_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mode: {args.mode}")
    print(f"Warmup: {args.warmup_steps}, Measure: {args.measure_steps}")
    print(f"Mixed precision: {args.mixed_precision}")
    print()

    # Monkey-patch attention with NVTX annotations if requested (for 1.1.4e)
    if args.nvtx_attention:
        a1_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        print("NVTX attention annotations: ENABLED")

    # Create model and data
    model = create_model(args.model_size, args.vocab_size, args.context_length)
    input_ids = generate_random_batch(args.batch_size, args.context_length, args.vocab_size)

    # Mixed precision context
    if args.mixed_precision:
        amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()

    # Memory profiling mode (1.1.6) — runs one step then exits
    if args.profile_memory:
        profile_memory(model, input_ids, args.mode, args.warmup_steps, amp_context, args)
        return

    # Run benchmark
    times = benchmark(model, input_ids, args.mode,
                      args.warmup_steps, args.measure_steps, amp_context)

    # Report results
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Results over {args.measure_steps} steps:")
    print(f"  Mean: {mean_time * 1000:.2f} ms")
    print(f"  Std:  {std_time * 1000:.2f} ms")
    print(f"  Times (ms): {[f'{t*1000:.2f}' for t in times]}")


if __name__ == "__main__":
    main()
