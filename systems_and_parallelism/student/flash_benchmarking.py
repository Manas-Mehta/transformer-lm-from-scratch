"""
1.3.2 Problem (flash_benchmarking): 15 points

(a) Benchmark Triton FlashAttention-2 vs vanilla PyTorch attention.
    Uses triton.testing.do_bench for timing.

Sweep:
    - seq_len: powers of 2 from 128 to 65536
    - d_head: powers of 2 from 16 to 128
    - precision: bfloat16, float32
    - batch_size=1, causal=True

Usage:
    uv run python student/flash_benchmarking.py
"""

import itertools
import torch
import triton
from a1_basics.model import scaled_dot_product_attention
from student.flash_attention import FlashAttentionTriton


def vanilla_attention_fwd(Q, K, V, mask):
    """Forward pass using staff's vanilla PyTorch attention."""
    return scaled_dot_product_attention(Q, K, V, mask)


def vanilla_attention_fwd_bwd(Q, K, V, mask):
    """Forward + backward using vanilla PyTorch attention."""
    out = scaled_dot_product_attention(Q, K, V, mask)
    loss = out.sum()
    loss.backward()


def flash_fwd(Q, K, V):
    """Forward pass using Triton FlashAttention-2."""
    return FlashAttentionTriton.apply(Q, K, V, True)


def flash_fwd_bwd(Q, K, V):
    """Forward + backward using Triton FlashAttention-2."""
    out = FlashAttentionTriton.apply(Q, K, V, True)
    loss = out.sum()
    loss.backward()


def benchmark_config(seq_len, d_head, dtype, batch_size=1, warmup=100, rep=1000):
    """Benchmark one configuration, return dict of timings or None on OOM."""
    try:
        Q = torch.randn(batch_size, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)

        # Causal mask for vanilla attention (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device="cuda")).bool()

        # --- Triton FA2 ---
        # Forward
        flash_fwd_ms = triton.testing.do_bench(
            lambda: FlashAttentionTriton.apply(Q, K, V, True),
            warmup=warmup, rep=rep,
        )

        # End-to-end (forward + backward)
        def _flash_e2e():
            Q.grad = K.grad = V.grad = None
            out = FlashAttentionTriton.apply(Q, K, V, True)
            out.sum().backward()

        flash_e2e_ms = triton.testing.do_bench(_flash_e2e, warmup=warmup, rep=rep)
        flash_bwd_ms = flash_e2e_ms - flash_fwd_ms

        # --- Vanilla PyTorch ---
        # Forward
        vanilla_fwd_ms = triton.testing.do_bench(
            lambda: scaled_dot_product_attention(Q, K, V, mask),
            warmup=warmup, rep=rep,
        )

        # End-to-end
        def _vanilla_e2e():
            Q.grad = K.grad = V.grad = None
            out = scaled_dot_product_attention(Q, K, V, mask)
            out.sum().backward()

        vanilla_e2e_ms = triton.testing.do_bench(_vanilla_e2e, warmup=warmup, rep=rep)
        vanilla_bwd_ms = vanilla_e2e_ms - vanilla_fwd_ms

        del Q, K, V, mask
        torch.cuda.empty_cache()

        return {
            "flash_fwd": flash_fwd_ms,
            "flash_bwd": flash_bwd_ms,
            "flash_e2e": flash_e2e_ms,
            "vanilla_fwd": vanilla_fwd_ms,
            "vanilla_bwd": vanilla_bwd_ms,
            "vanilla_e2e": vanilla_e2e_ms,
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def main():
    seq_lens = [2**i for i in range(7, 17)]  # 128 to 65536
    d_heads = [2**i for i in range(4, 8)]    # 16 to 128
    dtypes = [torch.bfloat16, torch.float32]
    batch_size = 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {batch_size}, Causal: True")
    print()

    # Header
    print(f"{'dtype':>8} | {'d_head':>6} | {'seq_len':>7} | "
          f"{'FA2 fwd':>9} | {'FA2 bwd':>9} | {'FA2 e2e':>9} | "
          f"{'Van fwd':>9} | {'Van bwd':>9} | {'Van e2e':>9} | "
          f"{'Fwd spdup':>9} | {'E2E spdup':>9}")
    print("-" * 130)

    for dtype in dtypes:
        dtype_name = "bf16" if dtype == torch.bfloat16 else "fp32"
        for d_head in d_heads:
            for seq_len in seq_lens:
                result = benchmark_config(seq_len, d_head, dtype, batch_size)

                if result is None:
                    print(f"{dtype_name:>8} | {d_head:>6} | {seq_len:>7} | "
                          f"{'OOM':>9} | {'OOM':>9} | {'OOM':>9} | "
                          f"{'OOM':>9} | {'OOM':>9} | {'OOM':>9} | "
                          f"{'---':>9} | {'---':>9}")
                else:
                    fwd_speedup = result["vanilla_fwd"] / result["flash_fwd"] if result["flash_fwd"] > 0 else float('inf')
                    e2e_speedup = result["vanilla_e2e"] / result["flash_e2e"] if result["flash_e2e"] > 0 else float('inf')

                    print(f"{dtype_name:>8} | {d_head:>6} | {seq_len:>7} | "
                          f"{result['flash_fwd']:>8.3f}ms | {result['flash_bwd']:>8.3f}ms | {result['flash_e2e']:>8.3f}ms | "
                          f"{result['vanilla_fwd']:>8.3f}ms | {result['vanilla_bwd']:>8.3f}ms | {result['vanilla_e2e']:>8.3f}ms | "
                          f"{fwd_speedup:>8.2f}x | {e2e_speedup:>8.2f}x")

        print()  # blank line between dtypes


if __name__ == "__main__":
    main()
