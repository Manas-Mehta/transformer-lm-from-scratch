"""
Section 1.2 — pytorch_attention (5 pts): Benchmark PyTorch attention at different scales
Section 1.3 — torch_compile (5 pts): Benchmark compiled attention and compiled full model

Usage:
    uv run python student/benchmark_attention.py                    # 1.2(a): vanilla attention
    uv run python student/benchmark_attention.py --compiled         # 1.3(a): + torch.compile attention
    uv run python student/benchmark_attention.py --full_model       # 1.3(b): compiled full model
"""

import argparse
import itertools
import timeit
import torch
import numpy as np
from a1_basics.model import scaled_dot_product_attention, BasicsTransformerLM


def benchmark_attention():
    """1.2(a): Benchmark vanilla PyTorch attention at various scales."""
    args = parse_args()

    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    warmup_steps = 10
    measure_steps = 100

    print(f"=== Benchmarking PyTorch Attention ===")
    print(f"Batch size: {batch_size}, Warmup: {warmup_steps}, Measure: {measure_steps}")
    if args.compiled:
        print(f"torch.compile: ENABLED")
    print()

    # Choose attention function
    attn_fn = scaled_dot_product_attention
    if args.compiled:
        attn_fn = torch.compile(scaled_dot_product_attention)

    print(f"{'d_model':>7} | {'seq_len':>7} | {'Fwd (ms)':>10} | {'Bwd (ms)':>10} | {'Mem after fwd (MB)':>18} | Notes")
    print("-" * 80)

    for d_model, seq_len in itertools.product(d_models, seq_lens):
        try:
            Q = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device="cuda", requires_grad=True)

            # Causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda")).bool()

            # Warmup (extra for compiled to handle JIT compilation)
            n_warmup = warmup_steps * 3 if args.compiled else warmup_steps
            for _ in range(n_warmup):
                out = attn_fn(Q, K, V, mask)
                torch.cuda.synchronize()

            # Time forward passes
            torch.cuda.synchronize()
            start = timeit.default_timer()
            for _ in range(measure_steps):
                out = attn_fn(Q, K, V, mask)
                torch.cuda.synchronize()
            fwd_time = (timeit.default_timer() - start) / measure_steps

            # Measure memory after forward (before backward)
            torch.cuda.reset_peak_memory_stats()
            out = attn_fn(Q, K, V, mask)
            torch.cuda.synchronize()
            mem_after_fwd = torch.cuda.max_memory_allocated() / (1024 ** 2)

            # Time backward passes
            torch.cuda.synchronize()
            start = timeit.default_timer()
            for _ in range(measure_steps):
                Q.grad = K.grad = V.grad = None
                out = attn_fn(Q, K, V, mask)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()
            bwd_time = (timeit.default_timer() - start) / measure_steps

            print(f"{d_model:>7} | {seq_len:>7} | {fwd_time*1000:>10.2f} | {bwd_time*1000:>10.2f} | {mem_after_fwd:>18.1f} | ")

            # Free memory
            del Q, K, V, out, mask
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{d_model:>7} | {seq_len:>7} | {'OOM':>10} | {'OOM':>10} | {'OOM':>18} | Out of memory")
            torch.cuda.empty_cache()


def benchmark_full_model():
    """1.3(b): Benchmark compiled vs vanilla full Transformer model."""
    from student.benchmark import MODEL_CONFIGS, create_model, generate_random_batch

    args = parse_args()
    warmup_steps = 10
    measure_steps = 10

    print("=== Benchmarking Full Model: Vanilla vs Compiled ===")
    print()

    for model_size in ["small", "medium", "large"]:
        for compiled in [False, True]:
            try:
                model = create_model(model_size, 10000, 128)
                if compiled:
                    model = torch.compile(model)

                input_ids = generate_random_batch(4, 128, 10000)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                tag = "compiled" if compiled else "vanilla"

                # Extra warmup for compiled (JIT)
                n_warmup = warmup_steps * 5 if compiled else warmup_steps
                for _ in range(n_warmup):
                    optimizer.zero_grad()
                    out = model(input_ids)
                    loss = out.sum()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()

                # Time forward
                times_fwd = []
                for _ in range(measure_steps):
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    out = model(input_ids)
                    torch.cuda.synchronize()
                    times_fwd.append(timeit.default_timer() - start)

                # Time forward + backward + optimizer
                times_full = []
                for _ in range(measure_steps):
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    out = model(input_ids)
                    loss = out.sum()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    times_full.append(timeit.default_timer() - start)

                fwd_mean = np.mean(times_fwd) * 1000
                full_mean = np.mean(times_full) * 1000
                print(f"{model_size:>8} {tag:>10}: fwd={fwd_mean:.2f}ms, full={full_mean:.2f}ms")

                del model, input_ids, optimizer
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"{model_size:>8} {tag:>10}: OOM")
                torch.cuda.empty_cache()

    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark attention")
    parser.add_argument("--compiled", action="store_true",
                        help="Use torch.compile for attention (1.3a)")
    parser.add_argument("--full_model", action="store_true",
                        help="Benchmark compiled full model (1.3b)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.full_model:
        benchmark_full_model()
    else:
        benchmark_attention()
