#!/bin/bash
#SBATCH --job-name=fp32_d128
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_fp32_d128.out
#SBATCH --error=logs/%j_fp32_d128.err
#SBATCH --requeue

# Temp job: rerun only the missing fp32 d_head=128 sweep from flash_benchmarking

singularity exec --bind /scratch --nv \
  --overlay /scratch/mm14444/overlay-25GB-500K.ext3:ro \
  /scratch/mm14444/ubuntu-20.04.3.sif \
  /bin/bash -c '
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:$PATH
    cd /scratch/mm14444/transformer-lm-from-scratch/systems_and_parallelism

    uv run python -c "
import torch
import triton
from a1_basics.model import scaled_dot_product_attention
from student.flash_attention import FlashAttentionTriton

d_head = 128
dtype = torch.float32
batch_size = 1
warmup = 100
rep = 1000

print(f\"GPU: {torch.cuda.get_device_name(0)}\")
print(f\"fp32 d_head=128 only | batch={batch_size} | causal=True\")
print()
print(f\"   dtype | d_head | seq_len |   FA2 fwd |   FA2 bwd |   FA2 e2e |   Van fwd |   Van bwd |   Van e2e | Fwd spdup | E2E spdup\")
print(\"-\" * 130)

for exp in range(7, 17):  # 128 to 65536
    seq_len = 2**exp
    try:
        Q = torch.randn(batch_size, seq_len, d_head, device=\"cuda\", dtype=dtype, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_head, device=\"cuda\", dtype=dtype, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_head, device=\"cuda\", dtype=dtype, requires_grad=True)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=\"cuda\")).bool()

        flash_fwd_ms = triton.testing.do_bench(
            lambda: FlashAttentionTriton.apply(Q, K, V, True), warmup=warmup, rep=rep)

        def _flash_e2e():
            Q.grad = K.grad = V.grad = None
            out = FlashAttentionTriton.apply(Q, K, V, True)
            out.sum().backward()
        flash_e2e_ms = triton.testing.do_bench(_flash_e2e, warmup=warmup, rep=rep)
        flash_bwd_ms = flash_e2e_ms - flash_fwd_ms

        vanilla_fwd_ms = triton.testing.do_bench(
            lambda: scaled_dot_product_attention(Q, K, V, mask), warmup=warmup, rep=rep)

        def _vanilla_e2e():
            Q.grad = K.grad = V.grad = None
            out = scaled_dot_product_attention(Q, K, V, mask)
            out.sum().backward()
        vanilla_e2e_ms = triton.testing.do_bench(_vanilla_e2e, warmup=warmup, rep=rep)
        vanilla_bwd_ms = vanilla_e2e_ms - vanilla_fwd_ms

        fwd_sp = vanilla_fwd_ms / flash_fwd_ms if flash_fwd_ms > 0 else float(\"inf\")
        e2e_sp = vanilla_e2e_ms / flash_e2e_ms if flash_e2e_ms > 0 else float(\"inf\")

        print(f\"    fp32 |    {d_head} | {seq_len:>7} | {flash_fwd_ms:>8.3f}ms | {flash_bwd_ms:>8.3f}ms | {flash_e2e_ms:>8.3f}ms | {vanilla_fwd_ms:>8.3f}ms | {vanilla_bwd_ms:>8.3f}ms | {vanilla_e2e_ms:>8.3f}ms | {fwd_sp:>8.2f}x | {e2e_sp:>8.2f}x\")

        del Q, K, V, mask
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f\"    fp32 |    {d_head} | {seq_len:>7} |       OOM |       OOM |       OOM |       OOM |       OOM |       OOM |       --- |       ---\")
    except Exception as e:
        torch.cuda.empty_cache()
        print(f\"    fp32 |    {d_head} | {seq_len:>7} | ERROR: {type(e).__name__}: {e}\")
"
'
