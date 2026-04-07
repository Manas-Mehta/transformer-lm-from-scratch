"""GRPO helper methods for Part 7.

All functions are derived directly from the assignment PDF (a3-6.pdf):
  - compute_group_normalized_rewards : §7.1 Eq. 27 / Eq. 30
  - masked_mean                      : §7.2 (aggregate per-token losses over response tokens)
  - compute_naive_policy_gradient_loss: §7.2 Eq. 31
  - compute_grpo_clip_loss            : §7.2 Eq. 32
  - compute_policy_gradient_loss      : §7.2 (dispatcher)
  - grpo_microbatch_train_step        : §7.2 (train loop microbatch step)
"""

from typing import Literal

import torch
from torch import Tensor


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and group-normalize them to produce advantages.

    For each group of `group_size` responses to the same question:
      - If normalize_by_std=True  (PDF §7.1 Eq. 27):
            A^(i) = (r^(i) - mean(group)) / (std(group) + advantage_eps)
      - If normalize_by_std=False (PDF §7.1 Eq. 30):
            A^(i) = r^(i) - mean(group)

    Args:
        reward_fn: Callable[[str, str], dict] — returns {"reward": float, ...}.
        rollout_responses: flat list of all rollout response strings.
            Length = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: ground truth for each response.
            Same length as rollout_responses; each GT is repeated group_size times.
        group_size: number of responses per question.
        advantage_eps: small constant to prevent division by zero (Eq. 27).
        normalize_by_std: whether to divide by the group std (Eq. 27 vs Eq. 30).

    Returns:
        advantages:  (rollout_batch_size,) group-normalized rewards.
        raw_rewards: (rollout_batch_size,) unnormalized rewards.
        metadata:    dict of summary statistics for logging.
    """
    # Step 1: compute raw reward for every response (PDF §7.1: r^(i) = R(q, o^(i)))
    raw_rewards = torch.tensor(
        [
            reward_fn(r, g)["reward"]
            for r, g in zip(rollout_responses, repeated_ground_truths)
        ],
        dtype=torch.float32,
    )

    n = len(rollout_responses)
    advantages = torch.zeros_like(raw_rewards)

    # Step 2: normalize within each group of group_size responses
    for i in range(n // group_size):
        start = i * group_size
        end = start + group_size
        group = raw_rewards[start:end]
        group_mean = group.mean()

        if normalize_by_std:
            # PDF §7.1 Eq. 27
            advantages[start:end] = (group - group_mean) / (group.std() + advantage_eps)
        else:
            # PDF §7.1 Eq. 30
            advantages[start:end] = group - group_mean

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }
    return advantages, raw_rewards, metadata


def masked_mean(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
) -> Tensor:
    """Mean of tensor elements where mask==1, optionally along a dimension.

    Used in the GRPO microbatch step to average the per-token loss over
    response tokens only (PDF §7.2).

    Args:
        tensor: tensor of any shape.
        mask:   same shape as tensor; 1 (or True) where elements are included.
        dim:    dimension to average along. If None, average over all elements.

    Returns:
        Tensor: mean of the masked elements (scalar if dim=None).
    """
    mask = mask.float()
    if dim is None:
        return (tensor * mask).sum() / mask.sum()
    else:
        return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    """Per-token naive REINFORCE policy gradient loss (PDF §7.2 Eq. 31).

    Loss = -A_t * log π_θ(o_t | q, o_{<t})

    The advantage A is the same scalar for every token in a response, so
    raw_rewards_or_advantages (batch_size, 1) is broadcast over seq_length.

    Args:
        raw_rewards_or_advantages: (batch_size, 1) — raw reward or advantage per response.
        policy_log_probs:          (batch_size, seq_length) — per-token log-probs.

    Returns:
        (batch_size, seq_length) per-token loss.
    """
    # PDF §7.2 Eq. 31: -A_t * log π_θ  (broadcast advantage over seq_length)
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-token GRPO-Clip loss (PDF §7.2 Eq. 32).

    Loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    where ratio = π_θ(o_t) / π_θ_old(o_t) = exp(log_π_θ - log_π_θ_old).

    Clipping prevents the policy from straying too far from π_θ_old in a
    single update step (PDF §7.1, GRPO-Clip objective Eq. 28).

    Args:
        advantages:       (batch_size, 1) — per-response advantage A.
        policy_log_probs: (batch_size, seq_length) — log-probs of the current policy.
        old_log_probs:    (batch_size, seq_length) — log-probs of the old (rollout) policy.
        cliprange:        ε — clip parameter (e.g. 0.2).

    Returns:
        loss:     (batch_size, seq_length) per-token loss.
        metadata: dict with clip_fraction statistic.
    """
    # ratio = π_θ / π_θ_old  (PDF §7.1 Eq. 28)
    log_ratio = policy_log_probs - old_log_probs          # (B, L)
    ratio = torch.exp(log_ratio)                          # (B, L)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)  # (B, L)

    # PDF §7.2 Eq. 32: -min(ratio * A, clipped_ratio * A)
    # advantages is (B, 1) and broadcasts over seq_length
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)  # (B, L)

    clip_fraction = ((ratio < 1 - cliprange) | (ratio > 1 + cliprange)).float().mean()
    return loss, {"clip_fraction": clip_fraction}


def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Dispatcher: select the correct policy gradient loss (PDF §7.2).

    Three modes (PDF §7.2):
      "no_baseline":            naive PG with raw rewards (no baseline subtraction)
      "reinforce_with_baseline": naive PG with group-normalized advantages
      "grpo_clip":              GRPO-Clip loss (Eq. 32)

    Args:
        policy_log_probs: (batch_size, seq_length).
        loss_type:        which loss variant to use.
        raw_rewards:      (batch_size, 1) unnormalized rewards.
        advantages:       (batch_size, 1) group-normalized advantages.
        old_log_probs:    (batch_size, seq_length) log-probs of old policy.
        cliprange:        clip parameter ε for grpo_clip.

    Returns:
        (per_token_loss tensor, metadata dict)
    """
    if loss_type == "no_baseline":
        # REINFORCE with raw rewards — no baseline subtraction (PDF §6.4 Eq. 20 / §7.2 Eq. 31)
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        # REINFORCE with group mean baseline (PDF §6.5 Eq. 22 / §7.2 Eq. 31)
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        # GRPO-Clip (PDF §7.2 Eq. 32)
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Single GRPO microbatch forward-backward pass (PDF §7.2).

    Mirrors the SFT microbatch step (PDF §4.1) but uses a policy gradient loss
    instead of cross-entropy, and uses masked_mean to aggregate per-token losses.

    Steps:
      1. Compute per-token PG loss via compute_policy_gradient_loss.
      2. Average over response tokens with masked_mean (scalar).
      3. Divide by gradient_accumulation_steps (PDF §4.1 gradient accumulation pattern).
      4. Call loss.backward().

    Args:
        policy_log_probs:          (batch_size, seq_length).
        response_mask:             (batch_size, seq_length) — 1 for response tokens.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        loss_type:                 "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards:               (batch_size, 1) needed for "no_baseline".
        advantages:                (batch_size, 1) needed for "reinforce_with_baseline"/"grpo_clip".
        old_log_probs:             (batch_size, seq_length) needed for "grpo_clip".
        cliprange:                 ε needed for "grpo_clip".

    Returns:
        (loss_scalar, metadata_dict)
    """
    # Step 1: per-token loss  (B, L)
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Step 2+3: average over response tokens, scale for gradient accumulation
    loss = masked_mean(per_token_loss, response_mask) / gradient_accumulation_steps

    # Step 4: backward (PDF §4.1 gradient accumulation pattern)
    loss.backward()

    return loss, metadata
