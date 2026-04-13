"""Generate GRPO experiment figures for the writeup (Part 7 deliverable)."""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load data from HPC run ──────────────────────────────────────────────────

EVAL_JSON = os.path.join(os.path.dirname(__file__), "logs", "grpo-lr-_eval.json")
ROLLOUTS_JSON = os.path.join(os.path.dirname(__file__), "logs", "grpo-lr-_rollouts.json")

with open(EVAL_JSON) as f:
    eval_log = json.load(f)

with open(ROLLOUTS_JSON) as f:
    rollouts = json.load(f)

steps = [e["grpo_step"] for e in eval_log]
rewards = [e["reward"] for e in eval_log]
format_rewards = [e["format_reward"] for e in eval_log]
answer_rewards = [e["answer_reward"] for e in eval_log]

# ── Figure 1: Validation Reward Curve (main deliverable) ────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(steps, rewards, marker="o", linewidth=2.2, markersize=6,
        color="#2ecc71", label="Reward (answer correct)")
ax.plot(steps, format_rewards, marker="s", linewidth=2.0, markersize=5,
        color="#3498db", label="Format reward (has <answer> tags)", linestyle="--")

ax.axhline(y=0.30, color="gray", linestyle=":", linewidth=1.2,
           label="30% threshold (PDF reference)")

ax.set_xlabel("GRPO Step", fontsize=13)
ax.set_ylabel("Validation Reward", fontsize=13)
ax.set_title("GRPO Validation Reward vs. Training Steps\n"
             "(200-example dev set, lr=1e-5, reinforce_with_baseline)",
             fontsize=14)
ax.set_xlim(0, 210)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "grpo_reward_curve.png"), dpi=150)
plt.close(fig)
print("Saved: grpo_reward_curve.png")


# ── Figure 2: Example Rollouts Over Time ────────────────────────────────────

fig, axes = plt.subplots(1, len(rollouts), figsize=(4.5 * len(rollouts), 6))
if len(rollouts) == 1:
    axes = [axes]

for idx, entry in enumerate(rollouts):
    ax = axes[idx]
    step = entry["step"]
    target = entry["prompt_question"]
    r = entry["rewards"]
    n_correct = sum(1 for x in r if x == 1.0)

    # Show first response (truncated)
    resp = entry["responses"][0]
    # Clean up for display
    resp_display = resp.replace("\n", "\n")
    if len(resp_display) > 300:
        resp_display = resp_display[:300] + "..."

    wrapped = textwrap.fill(resp_display, width=45)

    ax.text(0.05, 0.95, f"Target: {target}", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
    ax.text(0.05, 0.88, f"Correct: {n_correct}/{len(r)}",
            transform=ax.transAxes, fontsize=11, va="top",
            color="#2ecc71" if n_correct > len(r) // 2 else "#e74c3c")
    ax.text(0.05, 0.80, wrapped, transform=ax.transAxes,
            fontsize=7.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))

    ax.set_title(f"Step {step}", fontsize=13, fontweight="bold")
    ax.axis("off")

fig.suptitle("Example GRPO Rollouts Over Training",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "grpo_example_rollouts.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: grpo_example_rollouts.png")


# ── Figure 3: Reward Breakdown (bar chart per eval step) ────────────────────

fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(steps))
width = 0.35

bars1 = ax.bar(x - width/2, [r * 200 for r in rewards], width,
               label="Correct answers", color="#2ecc71")
bars2 = ax.bar(x + width/2, [(f - r) * 200 for f, r in zip(format_rewards, rewards)],
               width, label="Correct format, wrong answer", color="#e67e22")

ax.set_xlabel("GRPO Step", fontsize=13)
ax.set_ylabel("Count (out of 200 eval examples)", fontsize=13)
ax.set_title("GRPO Eval Breakdown: Answer Correctness vs. Format Compliance",
             fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in steps], rotation=45)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "grpo_eval_breakdown.png"), dpi=150)
plt.close(fig)
print("Saved: grpo_eval_breakdown.png")


# ── Print summary ───────────────────────────────────────────────────────────

print(f"\nGRPO Baseline Results Summary:")
print(f"  Initial eval (step 10):  reward={rewards[0]:.1%}")
print(f"  Final eval (step {steps[-1]}): reward={rewards[-1]:.1%}")
print(f"  Peak reward:             {max(rewards):.1%} at step {steps[rewards.index(max(rewards))]}")
print(f"  Final format reward:     {format_rewards[-1]:.1%}")
print(f"\nAll figures saved to: {FIGURES_DIR}")
