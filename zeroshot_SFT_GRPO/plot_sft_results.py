"""Generate all SFT experiment figures for the writeup."""

import matplotlib.pyplot as plt
import numpy as np
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Data from HPC runs ───────────────────────────────────────────────────────

steps = [50, 100, 150, 200]

accuracy = {
    "n=128":  [0.610, 0.590, 0.595, 0.550],
    "n=256":  [0.495, 0.605, 0.540, 0.575],
    "n=512":  [0.475, 0.585, 0.575, 0.590],
    "n=1024": [0.450, 0.565, 0.560, 0.580],
    "n=full (10k)": [0.555, 0.560, 0.595, 0.535],
}

# Eval breakdown: (f1a1, f1a0, f0a0) per step, per run
eval_breakdown = {
    "n=128": {
        50:  (122, 25, 53),
        100: (118, 30, 52),
        150: (119, 26, 55),
        200: (110, 17, 73),
    },
    "n=256": {
        50:  (99, 28, 73),
        100: (121, 23, 56),
        150: (108, 17, 75),
        200: (115, 22, 63),
    },
    "n=512": {
        50:  (95, 15, 90),
        100: (117, 22, 61),
        150: (115, 16, 69),
        200: (118, 18, 64),
    },
    "n=1024": {
        50:  (90, 19, 91),
        100: (113, 11, 76),
        150: (112, 10, 78),
        200: (116, 17, 67),
    },
    "n=full (10k)": {
        50:  (111, 25, 64),
        100: (112, 22, 66),
        150: (119, 22, 59),
        200: (107, 13, 80),
    },
}

# Best model (sft-model-512) full test set eval
best_model_eval = {
    "Intellect test": {"f1a1": 299, "f1a0": 47, "f0a0": 154, "n": 500},
    "MATH test":      {"f1a1": 278, "f1a0": 46, "f0a0": 176, "n": 500},
}

ZERO_SHOT_BASELINE = 0.608

colors = {
    "n=128":  "#e74c3c",
    "n=256":  "#e67e22",
    "n=512":  "#2ecc71",
    "n=1024": "#3498db",
    "n=full (10k)": "#9b59b6",
}

# ── Figure 1: Validation Accuracy Curves ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

for label, accs in accuracy.items():
    ax.plot(steps, accs, marker="o", linewidth=2.2, markersize=7,
            label=label, color=colors[label])

ax.axhline(y=ZERO_SHOT_BASELINE, color="black", linestyle="--",
           linewidth=1.5, label=f"Zero-shot baseline ({ZERO_SHOT_BASELINE:.1%})")

ax.set_xlabel("Training Step", fontsize=13)
ax.set_ylabel("Validation Accuracy", fontsize=13)
ax.set_title("SFT Validation Accuracy vs. Training Steps\n(200-example MATH eval, different dataset sizes)", fontsize=14)
ax.set_xticks(steps)
ax.set_ylim(0.40, 0.68)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "sft_accuracy_curves.png"), dpi=150)
plt.close(fig)
print("Saved: sft_accuracy_curves.png")


# ── Figure 2: Best Accuracy per Dataset Size (bar chart) ─────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

labels_bar = list(accuracy.keys())
best_accs = [max(v) for v in accuracy.values()]
best_steps_list = [steps[np.argmax(v)] for v in accuracy.values()]
bar_colors = [colors[l] for l in labels_bar]

bars = ax.bar(labels_bar, best_accs, color=bar_colors, edgecolor="white", linewidth=1.2)
ax.axhline(y=ZERO_SHOT_BASELINE, color="black", linestyle="--",
           linewidth=1.5, label=f"Zero-shot baseline ({ZERO_SHOT_BASELINE:.1%})")

for bar, step, acc in zip(bars, best_steps_list, best_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{acc:.1%}\n(step {step})", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Best Validation Accuracy", fontsize=13)
ax.set_title("Peak SFT Accuracy by Dataset Size", fontsize=14)
ax.set_ylim(0.40, 0.68)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "sft_best_accuracy_bar.png"), dpi=150)
plt.close(fig)
print("Saved: sft_best_accuracy_bar.png")


# ── Figure 3: Eval Category Breakdown (stacked bars over steps, per run) ─────

fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

for idx, (run_label, breakdown) in enumerate(eval_breakdown.items()):
    ax = axes[idx]
    f1a1 = [breakdown[s][0] for s in steps]
    f1a0 = [breakdown[s][1] for s in steps]
    f0a0 = [breakdown[s][2] for s in steps]

    x = np.arange(len(steps))
    width = 0.6
    ax.bar(x, f1a1, width, label="Correct", color="#2ecc71")
    ax.bar(x, f1a0, width, bottom=f1a1, label="Wrong answer", color="#e67e22")
    ax.bar(x, f0a0, width, bottom=[a + b for a, b in zip(f1a1, f1a0)],
           label="No \\boxed{}", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("Step")
    ax.set_title(run_label, fontsize=12, color=colors[run_label], fontweight="bold")
    if idx == 0:
        ax.set_ylabel("Count (out of 200)")

axes[-1].legend(loc="upper right", fontsize=9)
fig.suptitle("Eval Category Breakdown by Training Step (200-example validation)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "sft_eval_breakdown.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: sft_eval_breakdown.png")


# ── Figure 4: Best Model Test Set Results ────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for idx, (test_name, data) in enumerate(best_model_eval.items()):
    ax = axes[idx]
    sizes = [data["f1a1"], data["f1a0"], data["f0a0"]]
    labels_pie = [
        f'Correct\n{data["f1a1"]}/{data["n"]} ({data["f1a1"]/data["n"]:.1%})',
        f'Wrong answer\n{data["f1a0"]}/{data["n"]} ({data["f1a0"]/data["n"]:.1%})',
        f'No \\boxed{{}}\n{data["f0a0"]}/{data["n"]} ({data["f0a0"]/data["n"]:.1%})',
    ]
    pie_colors = ["#2ecc71", "#e67e22", "#e74c3c"]
    ax.pie(sizes, labels=labels_pie, colors=pie_colors,
           autopct="", startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"{test_name}\nAccuracy: {data['f1a1']/data['n']:.1%}", fontsize=13)

fig.suptitle("Best SFT Model (sft-model-512) — Full Test Set Evaluation (n=500 each)",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "sft_best_model_test_eval.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: sft_best_model_test_eval.png")


# ── Figure 5: Format compliance (f0a0 rate) across steps ─────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

for run_label, breakdown in eval_breakdown.items():
    f0a0_rates = [breakdown[s][2] / 200 for s in steps]
    ax.plot(steps, f0a0_rates, marker="s", linewidth=2.2, markersize=7,
            label=run_label, color=colors[run_label])

ax.set_xlabel("Training Step", fontsize=13)
ax.set_ylabel("No \\boxed{} Rate (f0a0 / 200)", fontsize=13)
ax.set_title("Format Compliance Degradation During SFT\n(fraction of outputs missing \\boxed{} formatting)", fontsize=14)
ax.set_xticks(steps)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(loc="upper left", fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "sft_format_compliance.png"), dpi=150)
plt.close(fig)
print("Saved: sft_format_compliance.png")

print("\nAll figures saved to:", FIGURES_DIR)
