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
├── 5. benchmark()                      ← the core: warmup + timed runs
└── 6. main()                           ← glue everything together
'''


import argparse          # For command-line argument parsing
import timeit            # For high-resolution timing
import torch
import numpy as np
from a1_basics.model import BasicsTransformerLM
import math
import torch.cuda.nvtx as nvtx   # NVTX annotations for nsys profiling
import a1_basics.model            # needed for monkey-patching attention



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
    parser.add_argument("--nvtx_attention", action="store_true",
                    help="Enable NVTX-annotated attention for 1.1.4(e)")
    parser.add_argument("--optimizer", action="store_true",
                    help="Include AdamW optimizer step for 1.1.4(d)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["forward", "backward", "both"],
                        help="What to benchmark: forward only, backward only, or both")
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


#-------        5. benchmark()                        -----------------------


# Goal: Run the model N times and measure how long each run takes.
'''
IMP: see notes for step by step build up of the benchmark() function
base + async + warm up + backward pass

+ NVTX wrapping

function benchmark(model, input_ids, num_steps):
    times = []
    for each step:
        start timer
        run model(input_ids)
        stop timer
        record (stop - start)
    return times
'''

def benchmark(model, input_ids, mode, warmup_steps, measure_steps, optimizer=None):
    """
    Run warmup, then timed measurement steps.
    Returns list of times in seconds.
    """
    # ─── Phase 1: Warm-up (not timed) ───
    with nvtx.range("warmup"):                         # <-- NEW
        for _ in range(warmup_steps):
            if mode in ("backward", "both"):
                model.zero_grad(set_to_none=True)
            output = model(input_ids)
            if mode in ("backward", "both"):
                loss = output.sum()
                loss.backward()
                if optimizer is not None:
                    optimizer.step()
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

                    if optimizer is not None:
                        with nvtx.range("optimizer_step"):
                            optimizer.step()

            torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append(end - start)

    return times


#-------        6. main()                             -----------------------


def main():
    args = parse_args()

    # Print config so you can verify what's running
    print(f"=== Benchmarking {args.model_size} model ===")
    print(f"Context length: {args.context_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mode: {args.mode}")
    print(f"Warmup: {args.warmup_steps}, Measure: {args.measure_steps}")
    print()
    
    # Monkey-patch attention with NVTX annotations if requested (for 1.1.4e)
    if args.nvtx_attention:
        a1_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        print("NVTX attention annotations: ENABLED")

        
    # Create model and data
    model = create_model(args.model_size, args.vocab_size, args.context_length)
    input_ids = generate_random_batch(args.batch_size, args.context_length, args.vocab_size)

    # Create optimizer if requested (for 1.1.4d — full training step)
    optimizer = None
    if args.optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        print("AdamW optimizer: ENABLED")

    # Run benchmark
    times = benchmark(model, input_ids, args.mode,
                      args.warmup_steps, args.measure_steps, optimizer=optimizer)

    # Report results
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Results over {args.measure_steps} steps:")
    print(f"  Mean: {mean_time * 1000:.2f} ms")
    print(f"  Std:  {std_time * 1000:.2f} ms")
    print(f"  Times (ms): {[f'{t*1000:.2f}' for t in times]}")


if __name__ == "__main__":
    main()
