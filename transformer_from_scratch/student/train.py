"""
Training script for Transformer Language Model on TinyStories

Usage (from nyu-llm-reasoners-a1/ directory):

    Apple Silicon (M-series):
        uv run student/train.py --device mps --batch_size 32 --total_steps 5000

    NVIDIA GPU:
        uv run student/train.py --device cuda --batch_size 64 --total_steps 5000

    Resume from checkpoint:
        uv run student/train.py --device mps --resume_from checkpoints/checkpoint_step_1000.pt
"""

import argparse
import csv
import os
import time
import numpy as np
import torch

from student.model import (
    TransformerLM,
    AdamW,
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer LM on TinyStories")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (32 for M-series, 64-128 for H100)")
    parser.add_argument("--total_steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--max_learning_rate", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=6e-5, help="Minimum learning rate (10% of max)")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max L2 norm")

    # Data paths
    parser.add_argument("--train_data", type=str, default="data/train_tokens.npy", help="Path to training data")
    parser.add_argument("--valid_data", type=str, default="data/valid_tokens.npy", help="Path to validation data")

    # Logging and checkpointing
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")

    # Device
    parser.add_argument("--device", type=str, default="mps", help="Device: 'cpu', 'cuda', or 'mps'")

    # Ablation flags (Section 7.3)
    parser.add_argument("--no_rmsnorm", action="store_true", help="Remove RMSNorm (layer_norm_ablation)")
    parser.add_argument("--post_norm", action="store_true", help="Use post-norm instead of pre-norm (pre_norm_ablation)")
    parser.add_argument("--no_rope", action="store_true", help="Remove RoPE (no_pos_emb)")
    parser.add_argument("--use_silu_ffn", action="store_true", help="Use SiLU FFN instead of SwiGLU (swiglu_ablation)")

    args = parser.parse_args()

    # Identify active ablations
    ablations = []
    if args.no_rmsnorm:
        ablations.append("NO_RMSNORM")
    if args.post_norm:
        ablations.append("POST_NORM")
    if args.no_rope:
        ablations.append("NO_ROPE")
    if args.use_silu_ffn:
        ablations.append("SILU_FFN")

    print("="*70)
    print("TRANSFORMER LM TRAINING - TINYSTORIES")
    print("="*70)
    print(f"\nModel Configuration:")
    print(f"  vocab_size:      {args.vocab_size}")
    print(f"  context_length:  {args.context_length}")
    print(f"  d_model:         {args.d_model}")
    print(f"  num_layers:      {args.num_layers}")
    print(f"  num_heads:       {args.num_heads}")
    print(f"  d_ff:            {args.d_ff}")
    print(f"  rope_theta:      {args.rope_theta}")
    if ablations:
        print(f"  ablations:       {', '.join(ablations)}")
    else:
        print(f"  ablations:       None (base model)")

    print(f"\nTraining Configuration:")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  total_steps:     {args.total_steps}")
    print(f"  max_lr:          {args.max_learning_rate}")
    print(f"  min_lr:          {args.min_learning_rate}")
    print(f"  warmup_iters:    {args.warmup_iters}")
    print(f"  weight_decay:    {args.weight_decay}")
    print(f"  grad_clip:       {args.grad_clip}")
    print(f"  device:          {args.device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        pre_norm=not args.post_norm,
        use_rope=not args.no_rope,
        use_swiglu=not args.use_silu_ffn,
        device=args.device,
    )

    model = model.to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Trainable parameters: {num_params_trainable:,}")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,  # Will be overridden by scheduler
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Load data with memory mapping (CRITICAL for large datasets!)
    print("\n" + "="*70)
    print("LOADING DATA (MEMORY-MAPPED)")
    print("="*70)

    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data not found: {args.train_data}")
    if not os.path.exists(args.valid_data):
        raise FileNotFoundError(f"Validation data not found: {args.valid_data}")

    print(f"\nLoading training data: {args.train_data}")
    train_dataset = np.load(args.train_data, mmap_mode='r')
    print(f"  Train tokens: {len(train_dataset):,}")

    print(f"\nLoading validation data: {args.valid_data}")
    valid_dataset = np.load(args.valid_data, mmap_mode='r')
    print(f"  Valid tokens: {len(valid_dataset):,}")

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from is not None:
        print("\n" + "="*70)
        print("RESUMING FROM CHECKPOINT")
        print("="*70)
        print(f"\nLoading checkpoint: {args.resume_from}")
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from step {start_step}")

    # Cosine cycle iterations (typically same as total_steps)
    cosine_cycle_iters = args.total_steps

    # CSV logging for learning curves (needed for Section 7 experiments)
    log_path = os.path.join(args.checkpoint_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "train_loss", "val_loss", "val_perplexity", "lr", "wallclock_seconds", "tokens_per_sec"])
    training_start_time = time.time()

    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print()

    model.train()

    for step in range(start_step, args.total_steps):
        step_start_time = time.time()

        # Get learning rate for this step
        lr = get_lr_cosine_schedule(
            step,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            cosine_cycle_iters
        )

        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Sample a batch
        inputs, targets = get_batch(train_dataset, args.batch_size, args.context_length, args.device)

        # Forward pass
        logits = model(inputs)  # (batch, seq, vocab_size)

        # Reshape for cross_entropy: (batch*seq, vocab_size) and (batch*seq,)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        targets_flat = targets.view(batch_size * seq_len)

        loss = cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

        step_time = time.time() - step_start_time

        # Logging
        if step % args.log_every == 0:
            tokens_per_sec = args.batch_size * args.context_length / step_time
            elapsed = time.time() - training_start_time
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | LR: {lr:.6f} | {tokens_per_sec:.0f} tok/s | {elapsed:.0f}s")
            # Write train-only row to CSV
            log_writer.writerow([step, f"{loss.item():.6f}", "", "", f"{lr:.8f}", f"{elapsed:.1f}", f"{tokens_per_sec:.0f}"])
            log_file.flush()

        # Validation
        if step % args.eval_every == 0 and step > 0:
            model.eval()

            # Run validation on multiple batches for more stable estimate
            num_val_batches = 10
            val_losses = []

            with torch.no_grad():
                for _ in range(num_val_batches):
                    val_inputs, val_targets = get_batch(valid_dataset, args.batch_size, args.context_length, args.device)
                    val_logits = model(val_inputs)

                    # Reshape for cross_entropy
                    val_logits_flat = val_logits.view(-1, val_logits.size(-1))
                    val_targets_flat = val_targets.view(-1)

                    val_loss = cross_entropy(val_logits_flat, val_targets_flat)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_perplexity = np.exp(avg_val_loss)
            elapsed = time.time() - training_start_time

            print(f"  >> VAL | Loss: {avg_val_loss:.4f} | Perplexity: {val_perplexity:.2f} | {elapsed:.0f}s")

            # Write validation row to CSV
            log_writer.writerow([step, f"{loss.item():.6f}", f"{avg_val_loss:.6f}", f"{val_perplexity:.4f}", f"{lr:.8f}", f"{elapsed:.1f}", ""])
            log_file.flush()

            model.train()

        # Checkpointing
        if step % args.save_every == 0 and step > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{step}.pt")
            save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"  ▶ Saved checkpoint: {checkpoint_path}")

    # Final checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.total_steps, final_checkpoint_path)
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nFinal checkpoint saved: {final_checkpoint_path}")

    # Final validation
    print("\nRunning final validation...")
    model.eval()

    num_val_batches = 20
    val_losses = []

    with torch.no_grad():
        for _ in range(num_val_batches):
            val_inputs, val_targets = get_batch(valid_dataset, args.batch_size, args.context_length, args.device)
            val_logits = model(val_inputs)

            val_logits_flat = val_logits.view(-1, val_logits.size(-1))
            val_targets_flat = val_targets.view(-1)

            val_loss = cross_entropy(val_logits_flat, val_targets_flat)
            val_losses.append(val_loss.item())

    final_val_loss = sum(val_losses) / len(val_losses)
    final_perplexity = np.exp(final_val_loss)
    total_elapsed = time.time() - training_start_time

    print(f"\nFinal Validation Results:")
    print(f"  Loss:       {final_val_loss:.4f}")
    print(f"  Perplexity: {final_perplexity:.2f}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"\nTarget for TinyStories:")
    print(f"  M-series:   val_loss <= 2.00")
    print(f"  H100:       val_loss <= 1.45")

    # Write final validation to CSV and close
    log_writer.writerow([args.total_steps, "", f"{final_val_loss:.6f}", f"{final_perplexity:.4f}", "", f"{total_elapsed:.1f}", ""])
    log_file.close()
    print(f"\nTraining log saved: {log_path}")


if __name__ == "__main__":
    main()
