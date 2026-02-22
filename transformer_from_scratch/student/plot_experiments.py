"""
Plot learning curves from experiment CSVs for Section 7 writeup.

Usage:
    uv run student/plot_experiments.py

Requires matplotlib:
    uv pip install matplotlib

Outputs PNG files in experiments/ directory.
"""

import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (works without display)
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed. Run:")
    print("  uv pip install matplotlib")
    sys.exit(1)


EXPERIMENTS_DIR = "experiments"
PLOTS_DIR = os.path.join(EXPERIMENTS_DIR, "plots")


def read_csv(path):
    """Read a training_log.csv and return dict of lists."""
    data = {"step": [], "train_loss": [], "val_loss": [], "val_perplexity": [],
            "lr": [], "wallclock_seconds": [], "tokens_per_sec": []}
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                val = row.get(key, "")
                if val == "":
                    data[key].append(None)
                else:
                    data[key].append(float(val))
    return data


def get_val_points(data):
    """Extract (step, val_loss) pairs where val_loss is not None."""
    steps, losses = [], []
    if data is None:
        return steps, losses
    for s, v in zip(data["step"], data["val_loss"]):
        if v is not None and s is not None:
            steps.append(s)
            losses.append(v)
    return steps, losses


def get_train_points(data):
    """Extract (step, train_loss) pairs where train_loss is not None."""
    steps, losses = [], []
    if data is None:
        return steps, losses
    for s, v in zip(data["step"], data["train_loss"]):
        if v is not None and s is not None:
            steps.append(s)
            losses.append(v)
    return steps, losses


def plot_lr_sweep():
    """Plot 1: Learning rate sweep — val_loss vs step for each LR."""
    print("Plotting LR sweep...")
    fig, ax = plt.subplots(figsize=(10, 6))

    lrs = ["1e-4", "3e-4", "6e-4", "1e-3", "3e-3"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for lr, color in zip(lrs, colors):
        csv_path = os.path.join(EXPERIMENTS_DIR, f"lr_{lr}", "training_log.csv")
        data = read_csv(csv_path)
        steps, losses = get_val_points(data)
        if steps:
            ax.plot(steps, losses, label=f"lr={lr}", color=color, linewidth=1.5)
            # Mark final point
            ax.scatter([steps[-1]], [losses[-1]], color=color, s=40, zorder=5)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Learning Rate Sweep — Validation Loss", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    out_path = os.path.join(PLOTS_DIR, "lr_sweep.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_batch_size():
    """Plot 2: Batch size experiment — val_loss vs tokens processed."""
    print("Plotting batch size experiment...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    configs = [
        ("bs_16", 16, 256, "#1f77b4"),
        ("bs_32", 32, 256, "#ff7f0e"),
        ("bs_64", 64, 256, "#2ca02c"),
        ("bs_128", 128, 256, "#d62728"),
    ]

    # Plot (a): val_loss vs step
    for name, bs, ctx, color in configs:
        csv_path = os.path.join(EXPERIMENTS_DIR, name, "training_log.csv")
        data = read_csv(csv_path)
        steps, losses = get_val_points(data)
        if steps:
            ax1.plot(steps, losses, label=f"bs={bs}", color=color, linewidth=1.5)

    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Validation Loss", fontsize=12)
    ax1.set_title("Batch Size — Val Loss vs Step", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot (b): val_loss vs tokens processed
    for name, bs, ctx, color in configs:
        csv_path = os.path.join(EXPERIMENTS_DIR, name, "training_log.csv")
        data = read_csv(csv_path)
        steps, losses = get_val_points(data)
        if steps:
            tokens = [s * bs * ctx for s in steps]
            ax2.plot(tokens, losses, label=f"bs={bs}", color=color, linewidth=1.5)

    ax2.set_xlabel("Tokens Processed", fontsize=12)
    ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.set_title("Batch Size — Val Loss vs Tokens", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    out_path = os.path.join(PLOTS_DIR, "batch_size.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_ablations():
    """Plot 3: All ablations vs base model."""
    print("Plotting ablations...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Base model data (for comparison in each subplot)
    base_csv = os.path.join(EXPERIMENTS_DIR, "lr_6e-4", "training_log.csv")
    base_data = read_csv(base_csv)
    base_steps, base_losses = get_val_points(base_data)

    # --- Subplot 1: No RMSNorm (layer_norm_ablation) ---
    ax = axes[0, 0]
    if base_steps:
        ax.plot(base_steps, base_losses, label="Base (pre-norm + RMSNorm)", color="#2ca02c", linewidth=1.5)

    for name, label, color in [
        ("ablation_no_rmsnorm", "No RMSNorm (lr=6e-4)", "#d62728"),
        ("ablation_no_rmsnorm_lowlr", "No RMSNorm (lr=1e-4)", "#ff7f0e"),
    ]:
        csv_path = os.path.join(EXPERIMENTS_DIR, name, "training_log.csv")
        data = read_csv(csv_path)
        steps, losses = get_val_points(data)
        if steps:
            ax.plot(steps, losses, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Ablation: Remove RMSNorm")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Post-norm (pre_norm_ablation) ---
    ax = axes[0, 1]
    if base_steps:
        ax.plot(base_steps, base_losses, label="Pre-norm (default)", color="#2ca02c", linewidth=1.5)

    csv_path = os.path.join(EXPERIMENTS_DIR, "ablation_post_norm", "training_log.csv")
    data = read_csv(csv_path)
    steps, losses = get_val_points(data)
    if steps:
        ax.plot(steps, losses, label="Post-norm", color="#d62728", linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Ablation: Pre-norm vs Post-norm")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: No RoPE (no_pos_emb) ---
    ax = axes[1, 0]
    if base_steps:
        ax.plot(base_steps, base_losses, label="With RoPE (default)", color="#2ca02c", linewidth=1.5)

    csv_path = os.path.join(EXPERIMENTS_DIR, "ablation_no_rope", "training_log.csv")
    data = read_csv(csv_path)
    steps, losses = get_val_points(data)
    if steps:
        ax.plot(steps, losses, label="No Position Embedding", color="#d62728", linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Ablation: RoPE vs NoPE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 4: SwiGLU vs SiLU (swiglu_ablation) ---
    ax = axes[1, 1]
    if base_steps:
        ax.plot(base_steps, base_losses, label="SwiGLU (d_ff=1344)", color="#2ca02c", linewidth=1.5)

    csv_path = os.path.join(EXPERIMENTS_DIR, "ablation_silu_ffn", "training_log.csv")
    data = read_csv(csv_path)
    steps, losses = get_val_points(data)
    if steps:
        ax.plot(steps, losses, label="SiLU FFN (d_ff=2048)", color="#d62728", linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Ablation: SwiGLU vs SiLU FFN")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Architecture Ablations — Validation Loss Curves", fontsize=16, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "ablations.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary():
    """Print a summary table of all experiment results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Final Val Loss':>15} {'Status':>10}")
    print("-" * 55)

    experiments = [
        "lr_1e-4", "lr_3e-4", "lr_6e-4", "lr_1e-3", "lr_3e-3",
        "bs_16", "bs_32", "bs_64", "bs_128",
        "ablation_no_rmsnorm", "ablation_no_rmsnorm_lowlr",
        "ablation_post_norm", "ablation_no_rope", "ablation_silu_ffn",
    ]

    for name in experiments:
        csv_path = os.path.join(EXPERIMENTS_DIR, name, "training_log.csv")
        data = read_csv(csv_path)
        steps, losses = get_val_points(data)
        if losses:
            final = losses[-1]
            status = "OK" if final < 10 else "DIVERGED"
            print(f"  {name:<28} {final:>15.4f} {status:>10}")
        else:
            print(f"  {name:<28} {'N/A':>15} {'MISSING':>10}")

    print("=" * 55)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 70)
    print("GENERATING LEARNING CURVE PLOTS FOR WRITEUP")
    print("=" * 70)

    plot_lr_sweep()
    plot_batch_size()
    plot_ablations()
    print_summary()

    print(f"\nAll plots saved to: {PLOTS_DIR}/")
    print("Copy these into your writeup.pdf!")


if __name__ == "__main__":
    main()
