"""Part 4 SFT Experiment: finetune Qwen2.5-Math-1.5B on the Prime Intellect MATH dataset.

Run with:
    uv run python -m student.sft_experiment --n-examples 512 --n-steps 200

On HPC with 2 GPUs: policy model on cuda:0, vLLM evaluator on cuda:1.
"""

import argparse
import json
import os
import random
from pathlib import Path
from unittest.mock import patch

import torch
import wandb
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from student.evaluate import evaluate, load_prompt
from student.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


# ──────────────────────────────────────────────────────────────────────────────
# vLLM helpers (from PDF §4.3 starter code)
# ──────────────────────────────────────────────────────────────────────────────

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Start a vLLM engine on a specific GPU for evaluation rollouts.

    Uses patches from TRL (github.com/huggingface/trl) to:
      1. Fix world_size so vLLM doesn't try to span multiple GPUs.
      2. Skip the memory footprint profiling assertion that doesn't apply here.
    """
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm(policy: torch.nn.Module, llm) -> None:
    """Copy the policy's current weights into the vLLM model runner in-place.

    Adapted from TRL grpo_trainer.py (github.com/huggingface/trl).
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_intellect_math(data_path: str, n_examples: int | None, seed: int):
    """Load Prime Intellect MATH train split and return (prompts, outputs)."""
    ds = load_from_disk(data_path)
    if n_examples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n_examples, len(ds))))

    prompts, outputs = [], []
    for ex in ds:
        msgs = ex["messages"]
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        prompt = (sys_msg + "\n\n" + user_msg) if sys_msg else user_msg
        prompts.append(prompt)
        outputs.append(asst_msg)

    return prompts, outputs


def infinite_batch_iter(prompts: list, outputs: list, batch_size: int):
    """Yield (prompt_batch, output_batch) indefinitely, shuffling each epoch."""
    idx = list(range(len(prompts)))
    while True:
        random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            b = idx[start : start + batch_size]
            yield [prompts[i] for i in b], [outputs[i] for i in b]


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT experiment for Part 4")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--data-path", default="data-distrib/intellect_math/train")
    parser.add_argument("--n-examples", type=int, default=None,
                        help="Number of train examples (None = use all)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Microbatch size (per optimizer sub-step)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Number of optimizer steps (each = batch_size * grad_accum examples)")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate on MATH every this many optimizer steps")
    parser.add_argument("--max-eval-examples", type=int, default=200,
                        help="Max MATH test examples for validation (keep ≤ 500)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="vLLM memory fraction on the eval GPU (cuda:1)")
    parser.add_argument("--output-dir", default="/scratch/mm14444/sft-model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="nyu-llm-reasoners-a3")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    policy_device = "cuda:0"
    vllm_device = "cuda:1"

    # ── WandB ─────────────────────────────────────────────────────────────────
    run_name = args.wandb_name or f"sft-n{args.n_examples or 'full'}-lr{args.lr}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        mode=args.wandb_mode,
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading training data from {args.data_path} (n_examples={args.n_examples})")
    train_prompts, train_outputs = load_intellect_math(
        args.data_path, args.n_examples, args.seed
    )
    print(f"  {len(train_prompts)} training examples loaded.")

    # ── Load model + tokenizer ────────────────────────────────────────────────
    print(f"Loading policy model {args.model} on {policy_device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # saves memory on long sequences (PDF §4.1)
    ).to(policy_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.train()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    # ── vLLM for evaluation ───────────────────────────────────────────────────
    print(f"Initializing vLLM on {vllm_device}...")
    llm = init_vllm(args.model, vllm_device, args.seed, args.gpu_memory_utilization)

    # ── MATH eval set ─────────────────────────────────────────────────────────
    print("Loading MATH evaluation set...")
    prompt_template = load_prompt("intellect")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_eval_examples:
        math_ds = math_ds.select(range(min(args.max_eval_examples, len(math_ds))))
    eval_prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    eval_gts = [ex["answer"] for ex in math_ds]
    print(f"  {len(eval_prompts)} eval examples.")

    # ── Training loop ─────────────────────────────────────────────────────────
    data_gen = infinite_batch_iter(train_prompts, train_outputs, args.batch_size)
    train_step = 0
    eval_step = 0
    eval_log = []  # list of {"train_step": int, "accuracy": float, "f1a1": int, "f1a0": int, "f0a0": int}
    total_microsteps = args.n_steps * args.gradient_accumulation_steps

    print(f"\nStarting SFT: {args.n_steps} optimizer steps "
          f"(microbatch={args.batch_size}, GA={args.gradient_accumulation_steps}, "
          f"effective_batch={args.batch_size * args.gradient_accumulation_steps})")

    for micro_idx in range(total_microsteps):
        prompt_batch, output_batch = next(data_gen)

        # Tokenize and move to GPU
        batch = tokenize_prompt_and_output(prompt_batch, output_batch, tokenizer)
        input_ids = batch["input_ids"].to(policy_device)
        labels = batch["labels"].to(policy_device)
        response_mask = batch["response_mask"].to(policy_device)

        # Forward pass (with gradient tracking)
        result = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
        log_probs = result["log_probs"]

        # SFT microbatch step: computes loss and calls loss.backward()
        loss, _ = sft_microbatch_train_step(
            log_probs, response_mask, args.gradient_accumulation_steps
        )

        # Optimizer step every gradient_accumulation_steps microbatches
        if (micro_idx + 1) % args.gradient_accumulation_steps == 0:
            train_step += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({"train/loss": loss.item(), "train_step": train_step})
            if train_step % 10 == 0 or train_step == 1:
                print(f"  step {train_step}/{args.n_steps}  loss={loss.item():.4f}")

            # Periodic MATH evaluation
            if train_step % args.eval_every == 0 or train_step == args.n_steps:
                print(f"\n[Eval @ step {train_step}] Syncing weights into vLLM...")
                model.eval()
                load_policy_into_vllm(model, llm)
                acc, counts, _ = evaluate(llm, eval_prompts, eval_gts, verbose=False)
                eval_step += 1
                wandb.log({
                    "eval/accuracy": acc,
                    "eval/f1a1": counts["f1a1"],
                    "eval/f1a0": counts["f1a0"],
                    "eval/f0a0": counts["f0a0"],
                    "eval_step": eval_step,
                })
                print(f"[Eval @ step {train_step}] MATH accuracy: {acc:.4f}  "
                      f"(correct={counts['f1a1']}, wrong={counts['f1a0']}, "
                      f"no_box={counts['f0a0']})\n")
                eval_log.append({
                    "train_step": train_step,
                    "accuracy": acc,
                    "f1a1": counts["f1a1"],
                    "f1a0": counts["f1a0"],
                    "f0a0": counts["f0a0"],
                })
                log_path = Path("logs") / f"{run_name}_eval.json"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(json.dumps(eval_log, indent=2))
                model.train()

    # ── Save model ────────────────────────────────────────────────────────────
    print(f"\nSaving model to {args.output_dir}...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

    wandb.finish()


if __name__ == "__main__":
    main()
