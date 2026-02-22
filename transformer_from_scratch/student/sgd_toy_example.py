"""
SGD Toy Example - Section 4.2 Learning Rate Sensitivity

This script demonstrates how learning rate affects SGD optimization.
We minimize a simple quadratic loss: L = mean(weights^2)

The optimal solution is weights=0, which gives loss=0.
"""

import torch
from torch.optim import SGD


def run_sgd_experiment(lr, num_steps=10):
    """
    Run SGD with a given learning rate.

    Args:
        lr: Learning rate to test
        num_steps: Number of optimization steps (default: 10)

    Returns:
        List of loss values at each step
    """
    # Initialize weights: 10x10 matrix with values ~ N(0, 5^2)
    weights = torch.nn.Parameter(5 * torch.randn(10, 10))

    # Create SGD optimizer
    opt = SGD([weights], lr=lr)

    # Track loss over time
    losses = []

    print(f"\n{'='*60}")
    print(f"Running SGD with lr={lr}")
    print(f"{'='*60}")
    print(f"{'Step':<6} {'Loss':<15} {'Max Weight':<15} {'Min Weight':<15}")
    print(f"{'-'*60}")

    for t in range(num_steps):
        # Zero gradients
        opt.zero_grad()

        # Compute loss: mean of squared weights
        loss = (weights ** 2).mean()

        # Backward pass
        loss.backward()

        # Update weights
        opt.step()

        # Record loss
        losses.append(loss.item())

        # Print statistics
        print(f"{t:<6} {loss.item():<15.6f} {weights.max().item():<15.6f} {weights.min().item():<15.6f}")

    print(f"{'='*60}\n")

    return losses


def main():
    """Run the toy experiment with lr=10, 100, 1000"""

    print("\n" + "="*70)
    print("SGD TOY EXPERIMENT - LEARNING RATE SENSITIVITY")
    print("="*70)
    print("\nObjective: Minimize L = mean(weights^2)")
    print("Optimal solution: weights = 0, loss = 0")
    print("Initial weights: 10x10 matrix sampled from N(0, 5^2)")
    print("\n")

    # Test different learning rates
    learning_rates = [10, 100, 1000]
    results = {}

    for lr in learning_rates:
        losses = run_sgd_experiment(lr=lr, num_steps=10)
        results[lr] = losses

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for lr, losses in results.items():
        final_loss = losses[-1]
        initial_loss = losses[0]

        if final_loss < initial_loss * 0.1:
            behavior = "✅ Converged smoothly"
        elif final_loss < initial_loss:
            behavior = "⚠️  Decreased but unstable"
        elif final_loss > 1e10:
            behavior = "❌ EXPLODED (diverged)"
        else:
            behavior = "❌ Failed to converge"

        print(f"\nlr={lr:>4}: {behavior}")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss:   {final_loss:.6f}")
        print(f"  Change:       {final_loss - initial_loss:+.6f}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    main()
