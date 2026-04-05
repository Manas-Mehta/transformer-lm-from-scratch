"""Plot SFT validation accuracy curves from eval JSON logs.

Usage (run locally after copying logs from HPC):
    uv run python -m student.plot_sft --log-dir logs --output plots/sft_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--output", default="plots/sft_curves.png")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Find all eval JSON files produced by sft_experiment.py
    json_files = sorted(log_dir.glob("sft-n*_eval.json"))
    if not json_files:
        print(f"No eval JSON files found in {log_dir}/")
        print("Expected filenames like: sft-n128_eval.json, sft-nfull_eval.json")
        return

    baseline_acc = 0.608  # Part 3 zero-shot result

    fig, ax = plt.subplots(figsize=(8, 5))

    for json_file in json_files:
        data = json.loads(json_file.read_text())
        if not data:
            continue
        steps = [d["train_step"] for d in data]
        accs = [d["accuracy"] for d in data]
        # Extract label from filename: sft-n128_eval.json → n=128
        label = json_file.stem.replace("_eval", "")  # e.g. sft-n128
        label = label.replace("sft-n", "n=").replace("nfull", "n=full (10k)")
        ax.plot(steps, accs, marker="o", label=label)

    # Zero-shot baseline as a dashed horizontal line
    ax.axhline(y=baseline_acc, color="gray", linestyle="--", label=f"Zero-shot baseline ({baseline_acc:.1%})")

    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("MATH Accuracy")
    ax.set_title("SFT Validation Accuracy vs. Training Steps\n(Qwen 2.5 Math 1.5B, varied dataset size)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
