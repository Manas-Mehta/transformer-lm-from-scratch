"""Evaluation script for MATH and Intellect test sets with category breakdown."""

from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from student.drgrpo_grader import question_only_reward_fn


def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def evaluate(llm, prompts, ground_truths, verbose=False, n_examples=10):
    """Run evaluation and return accuracy with format/answer category breakdown.

    Returns:
        accuracy: float
        counts: dict with keys f1a1, f1a0, f0a0
        examples: dict with lists of failure examples for each category
    """
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)

    counts = {"f1a1": 0, "f1a0": 0, "f0a0": 0}
    examples = {"f1a0": [], "f0a0": []}

    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        reward = question_only_reward_fn(text, ground_truths[i])
        fmt = int(reward["format_reward"])
        ans = int(reward["answer_reward"])

        if fmt == 1 and ans == 1:
            counts["f1a1"] += 1
        elif fmt == 1 and ans == 0:
            counts["f1a0"] += 1
            if len(examples["f1a0"]) < n_examples:
                examples["f1a0"].append({
                    "prompt_tail": prompts[i][-300:],
                    "output": text,
                    "ground_truth": ground_truths[i],
                })
        else:  # fmt == 0, ans == 0
            counts["f0a0"] += 1
            if len(examples["f0a0"]) < n_examples:
                examples["f0a0"].append({
                    "prompt_tail": prompts[i][-300:],
                    "output": text,
                    "ground_truth": ground_truths[i],
                })

    n = len(outputs)
    accuracy = counts["f1a1"] / n

    print(f"\n=== Category Breakdown (n={n}) ===")
    print(f"  Format=1, Answer=1 (correct):     {counts['f1a1']:4d} / {n}  ({counts['f1a1']/n:.1%})")
    print(f"  Format=1, Answer=0 (wrong answer): {counts['f1a0']:4d} / {n}  ({counts['f1a0']/n:.1%})")
    print(f"  Format=0, Answer=0 (no \\boxed{{}}):  {counts['f0a0']:4d} / {n}  ({counts['f0a0']/n:.1%})")
    print(f"  Overall accuracy:                  {accuracy:.4f}")

    if verbose:
        _print_examples(examples)

    return accuracy, counts, examples


def _print_examples(examples):
    """Print logged failure examples for writeup analysis."""
    sep = "-" * 60

    print(f"\n{sep}")
    print("FORMAT=0 EXAMPLES (no \\boxed{} found in output)")
    print(sep)
    for j, ex in enumerate(examples["f0a0"]):
        print(f"\n[Example {j+1}]")
        print(f"  Ground truth: {ex['ground_truth']}")
        print(f"  Output (last 400 chars):\n  ...{ex['output'][-400:]!r}")

    print(f"\n{sep}")
    print("FORMAT=1 BUT WRONG ANSWER EXAMPLES")
    print(sep)
    for j, ex in enumerate(examples["f1a0"]):
        print(f"\n[Example {j+1}]")
        print(f"  Ground truth: {ex['ground_truth']}")
        print(f"  Output (last 400 chars):\n  ...{ex['output'][-400:]!r}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--intellect-path", default="data/intellect_math_train_dev_test/test")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--verbose", action="store_true",
                        help="Print example outputs for each failure category")
    args = parser.parse_args()

    prompt_template = load_prompt("intellect")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Evaluate on Intellect test
    print(f"\n=== Intellect Test ({args.intellect_path}) ===")
    dataset = load_from_disk(args.intellect_path)
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    prompts, gts = [], []
    for ex in dataset:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))

    print(f"[Sample prompt tail] ...{prompts[0][-200:]}")
    acc, counts, examples = evaluate(llm, prompts, gts, verbose=args.verbose)
    print(f"Intellect Accuracy: {acc:.4f}")

    # Evaluate on MATH
    print("\n=== MATH Test (hiyouga/math12k) ===")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]

    print(f"[Sample prompt tail] ...{prompts[0][-200:]}")
    acc, counts, examples = evaluate(llm, prompts, gts, verbose=args.verbose)
    print(f"MATH Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
