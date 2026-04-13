"""Part 7 GRPO Experiment: train Qwen2.5-Math-1.5B-Instruct on Countdown with GRPO.

Run with:
    uv run python -m student.grpo_experiment --n-grpo-steps 200

On HPC with 2 GPUs: policy model on cuda:0, vLLM on cuda:1.

Follows Algorithm 2 from the PDF (§7.1):
  for step = 1..n_grpo_steps:
      sample questions → generate G rollouts → compute rewards → compute advantages
      for each inner training step:
          tokenize rollouts → get policy log-probs → GRPO microbatch step → optimizer.step()
"""

import argparse
import json
import random
import re
from pathlib import Path
from unittest.mock import patch

import torch
import wandb
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from student.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from student.sft import (
    get_response_log_probs,
    tokenize_prompt_and_output,
)


# ──────────────────────────────────────────────────────────────────────────────
# vLLM helpers (same as sft_experiment.py, from PDF §4.3)
# ──────────────────────────────────────────────────────────────────────────────

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.8):
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


def load_policy_into_vllm(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ──────────────────────────────────────────────────────────────────────────────
# Countdown reward function
# ──────────────────────────────────────────────────────────────────────────────

def countdown_reward_fn(response: str, ground_truth: str) -> dict:
    """Reward for Countdown: extract equation from <answer> tags and check if it equals target.

    Returns dict with format_reward, answer_reward, reward (all 0.0 or 1.0).
    """
    target = int(ground_truth)

    # Check format: must have <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not match:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    answer_text = match.group(1).strip()
    if not answer_text:
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}

    # Try to evaluate: look for the last "= number" pattern
    equals_matches = re.findall(r"=\s*([-\d.]+)", answer_text)
    if equals_matches:
        try:
            result = float(equals_matches[-1])
            if abs(result - target) < 1e-6:
                return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
        except ValueError:
            pass

    # Try evaluating each line as an expression (last to first)
    lines = [l.strip() for l in answer_text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        # Strip "Step N:" prefix
        cleaned = re.sub(r"^Step\s*\d+\s*:", "", line).strip()
        # Try left side of "expr = result" or the whole thing
        parts = cleaned.split("=") if "=" in cleaned else [cleaned]
        for part in parts:
            part = part.strip()
            try:
                result = eval(part, {"__builtins__": {}}, {})  # noqa: S307
                if isinstance(result, (int, float)) and abs(result - target) < 1e-6:
                    return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
            except Exception:
                continue

    return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_prompt_template() -> str:
    path = Path(__file__).parent / "prompts" / "countdown.prompt"
    return path.read_text()


def format_countdown_prompt(template: str, nums: list, target: int) -> str:
    question = (
        f"Using the numbers in the list {nums}, "
        f"create an equation that equals {target}."
    )
    return template.replace("{question}", question)


def load_countdown_split(data_path: str, split: str):
    """Load a Countdown dataset split, return list of (prompt, ground_truth) pairs."""
    ds = load_from_disk(data_path)
    template = load_prompt_template()

    prompts, ground_truths = [], []
    for ex in ds[split]:
        prompts.append(format_countdown_prompt(template, ex["nums"], ex["target"]))
        ground_truths.append(str(ex["target"]))

    return prompts, ground_truths


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_countdown(llm, prompts, ground_truths, n_examples=None):
    """Generate greedy responses and compute mean reward on Countdown."""
    from vllm import SamplingParams

    if n_examples and n_examples < len(prompts):
        prompts = prompts[:n_examples]
        ground_truths = ground_truths[:n_examples]

    params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, params)

    total_reward, total_format, total_answer = 0.0, 0.0, 0.0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        r = countdown_reward_fn(text, ground_truths[i])
        total_reward += r["reward"]
        total_format += r["format_reward"]
        total_answer += r["answer_reward"]

    n = len(outputs)
    return {
        "reward": total_reward / n,
        "format_reward": total_format / n,
        "answer_reward": total_answer / n,
        "n": n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop — Algorithm 2 (PDF §7.1)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO experiment on Countdown (Part 7)")
    # Model
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--data-path", default="data-distrib/countdown/dataset")
    # GRPO hyperparameters (PDF §7.2 p.24)
    parser.add_argument("--n-grpo-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--rollout-batch-size", type=int, default=16,
                        help="Total rollout responses per step (n_prompts * group_size)")
    parser.add_argument("--group-size", type=int, default=8,
                        help="Number of responses per question")
    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1,
                        help="1 = on-policy; >1 = off-policy")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--cliprange", type=float, default=0.2,
                        help="Clip parameter epsilon for grpo_clip loss")
    parser.add_argument("--loss-type", default="reinforce_with_baseline",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    parser.add_argument("--use-std-normalization", type=bool, default=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    # Eval
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--n-eval-examples", type=int, default=200)
    # Infrastructure
    parser.add_argument("--output-dir", default="/scratch/mm14444/grpo-model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="nyu-llm-reasoners-a3")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── Derived hyperparameters ──────────────────────────────────────────────
    assert args.rollout_batch_size % args.group_size == 0, \
        "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size

    # For on-policy: train_batch_size = rollout_batch_size
    train_batch_size = args.rollout_batch_size * args.epochs_per_rollout_batch
    assert train_batch_size % args.gradient_accumulation_steps == 0, \
        f"train_batch_size ({train_batch_size}) must be divisible by gradient_accumulation_steps ({args.gradient_accumulation_steps})"
    micro_train_batch_size = train_batch_size // args.gradient_accumulation_steps
    assert micro_train_batch_size >= 1, \
        f"micro_train_batch_size ({micro_train_batch_size}) must be >= 1"

    policy_device = "cuda:0"
    vllm_device = "cuda:1"

    # ── WandB ────────────────────────────────────────────────────────────────
    run_name = args.wandb_name or f"grpo-lr{args.lr}-{args.loss_type}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        mode=args.wandb_mode,
    )
    wandb.define_metric("grpo_step")
    wandb.define_metric("train/*", step_metric="grpo_step")
    wandb.define_metric("eval/*", step_metric="grpo_step")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading Countdown data from {args.data_path}...")
    train_prompts, train_gts = load_countdown_split(args.data_path, "train")
    dev_prompts, dev_gts = load_countdown_split(args.data_path, "dev")
    print(f"  Train: {len(train_prompts)}, Dev: {len(dev_prompts)}")

    # ── Load model + tokenizer ───────────────────────────────────────────────
    print(f"Loading policy model {args.model} on {policy_device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(policy_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.config.use_cache = False

    # ── Optimizer (PDF §7.2 p.24) ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=0.0, betas=(0.9, 0.95),
    )
    optimizer.zero_grad()

    # ── vLLM for rollout generation ──────────────────────────────────────────
    print(f"Initializing vLLM on {vllm_device}...")
    llm = init_vllm(args.model, vllm_device, args.seed, args.gpu_memory_utilization)

    # ── Sampling params ──────────────────────────────────────────────────────
    from vllm import SamplingParams

    rollout_params = SamplingParams(
        temperature=args.sampling_temperature,
        min_tokens=args.sampling_min_tokens,
        max_tokens=args.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    eval_log = []
    example_rollouts = []  # save a few for writeup
    train_indices = list(range(len(train_prompts)))

    print(f"\nStarting GRPO: {args.n_grpo_steps} steps, "
          f"{n_prompts_per_rollout_batch} prompts/step × {args.group_size} rollouts/prompt, "
          f"lr={args.lr}, loss={args.loss_type}")
    print(f"  micro_batch={micro_train_batch_size}, ga_steps={args.gradient_accumulation_steps}")

    for grpo_step in range(1, args.n_grpo_steps + 1):

        # ── 1. Sample questions ──────────────────────────────────────────────
        batch_indices = random.sample(train_indices, n_prompts_per_rollout_batch)
        batch_prompts = [train_prompts[i] for i in batch_indices]
        batch_gts = [train_gts[i] for i in batch_indices]

        # ── 2. Sync policy → vLLM ───────────────────────────────────────────
        model.eval()
        load_policy_into_vllm(model, llm)

        # ── 3. Generate G rollouts per question ─────────────────────────────
        # Repeat each prompt group_size times for batched generation
        repeated_prompts = [p for p in batch_prompts for _ in range(args.group_size)]
        repeated_gts = [g for g in batch_gts for _ in range(args.group_size)]

        vllm_outputs = llm.generate(repeated_prompts, rollout_params)
        rollout_responses = [out.outputs[0].text for out in vllm_outputs]

        # ── 4. Compute rewards and advantages ────────────────────────────────
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=countdown_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )

        # Log train reward stats
        wandb.log({
            "train/mean_reward": reward_meta["mean_reward"],
            "train/mean_format_reward": sum(
                1.0 for r in rollout_responses
                if re.search(r"<answer>.*?</answer>", r, re.DOTALL)
            ) / len(rollout_responses),
            "grpo_step": grpo_step,
        })

        # Save example rollouts periodically (for writeup)
        if grpo_step in (1, 10, 50, 100, 200) or grpo_step == args.n_grpo_steps:
            example_rollouts.append({
                "step": grpo_step,
                "prompt_question": batch_gts[0],  # target
                "responses": rollout_responses[:args.group_size],
                "rewards": raw_rewards[:args.group_size].tolist(),
            })

        # ── 5. Training phase ────────────────────────────────────────────────
        model.train()

        # For off-policy (grpo_clip): compute old log-probs before updating
        old_log_probs_all = None
        if args.loss_type == "grpo_clip":
            model.eval()
            with torch.no_grad():
                old_lps = []
                for mb_start in range(0, len(rollout_responses), micro_train_batch_size):
                    mb_end = mb_start + micro_train_batch_size
                    mb_prompts = repeated_prompts[mb_start:mb_end]
                    mb_responses = rollout_responses[mb_start:mb_end]
                    batch = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                    result = get_response_log_probs(
                        model,
                        batch["input_ids"].to(policy_device),
                        batch["labels"].to(policy_device),
                    )
                    old_lps.append(result["log_probs"].cpu())
                old_log_probs_all = torch.cat(old_lps, dim=0)
            model.train()

        # Iterate over rollout batch (possibly multiple epochs for off-policy)
        all_indices = list(range(len(rollout_responses)))
        total_loss = 0.0
        n_microbatches = 0

        for _epoch in range(args.epochs_per_rollout_batch):
            random.shuffle(all_indices)

            for mb_start in range(0, len(all_indices), micro_train_batch_size):
                mb_idx = all_indices[mb_start : mb_start + micro_train_batch_size]
                if len(mb_idx) < micro_train_batch_size:
                    continue  # skip incomplete microbatch

                mb_prompts = [repeated_prompts[i] for i in mb_idx]
                mb_responses = [rollout_responses[i] for i in mb_idx]
                mb_advantages = advantages[mb_idx].unsqueeze(1).to(policy_device)
                mb_raw_rewards = raw_rewards[mb_idx].unsqueeze(1).to(policy_device)

                # Tokenize
                batch = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                input_ids = batch["input_ids"].to(policy_device)
                labels = batch["labels"].to(policy_device)
                response_mask = batch["response_mask"].to(policy_device)

                # Forward pass: get policy log-probs
                result = get_response_log_probs(model, input_ids, labels)
                policy_log_probs = result["log_probs"]

                # Old log-probs for grpo_clip
                mb_old_lps = None
                if old_log_probs_all is not None:
                    mb_old_lps = old_log_probs_all[mb_idx].to(policy_device)
                    # Pad/truncate to match current sequence length
                    seq_len = policy_log_probs.shape[1]
                    if mb_old_lps.shape[1] < seq_len:
                        pad = torch.zeros(
                            mb_old_lps.shape[0], seq_len - mb_old_lps.shape[1],
                            device=policy_device,
                        )
                        mb_old_lps = torch.cat([mb_old_lps, pad], dim=1)
                    elif mb_old_lps.shape[1] > seq_len:
                        mb_old_lps = mb_old_lps[:, :seq_len]

                # GRPO microbatch train step
                loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    loss_type=args.loss_type,
                    raw_rewards=mb_raw_rewards if args.loss_type == "no_baseline" else None,
                    advantages=mb_advantages if args.loss_type != "no_baseline" else None,
                    old_log_probs=mb_old_lps,
                    cliprange=args.cliprange if args.loss_type == "grpo_clip" else None,
                )

                total_loss += loss.item()
                n_microbatches += 1

                # Optimizer step after gradient_accumulation_steps microbatches
                if n_microbatches % args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    wandb.log({
                        "train/loss": total_loss / args.gradient_accumulation_steps,
                        "train/grad_norm": grad_norm.item(),
                        "grpo_step": grpo_step,
                    })

        # Handle leftover gradients (shouldn't happen with correct config)
        if n_microbatches % args.gradient_accumulation_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # ── Print progress ───────────────────────────────────────────────────
        if grpo_step % 5 == 0 or grpo_step == 1:
            print(f"  step {grpo_step}/{args.n_grpo_steps}  "
                  f"mean_reward={reward_meta['mean_reward']:.3f}  "
                  f"loss={total_loss / max(n_microbatches, 1):.4f}")

        # ── 6. Periodic validation ───────────────────────────────────────────
        if grpo_step % args.eval_every == 0 or grpo_step == args.n_grpo_steps:
            print(f"\n[Eval @ step {grpo_step}] Syncing weights into vLLM...")
            model.eval()
            load_policy_into_vllm(model, llm)
            eval_results = evaluate_countdown(
                llm, dev_prompts, dev_gts, n_examples=args.n_eval_examples,
            )
            wandb.log({
                "eval/reward": eval_results["reward"],
                "eval/format_reward": eval_results["format_reward"],
                "eval/answer_reward": eval_results["answer_reward"],
                "grpo_step": grpo_step,
            })
            print(f"[Eval @ step {grpo_step}] "
                  f"reward={eval_results['reward']:.3f}  "
                  f"format={eval_results['format_reward']:.3f}  "
                  f"answer={eval_results['answer_reward']:.3f}  "
                  f"(n={eval_results['n']})\n")
            eval_log.append({"grpo_step": grpo_step, **eval_results})

            # Save eval log
            log_path = Path("logs") / f"{run_name}_eval.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(eval_log, indent=2))
            model.train()

    # ── Save model ───────────────────────────────────────────────────────────
    print(f"\nSaving model to {args.output_dir}...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save example rollouts for writeup
    rollout_path = Path("logs") / f"{run_name}_rollouts.json"
    rollout_path.write_text(json.dumps(example_rollouts, indent=2))
    print(f"Saved example rollouts to {rollout_path}")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
