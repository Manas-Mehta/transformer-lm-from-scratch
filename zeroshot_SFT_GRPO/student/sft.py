"""SFT helper methods for Part 4: tokenization, entropy, log-probs, masked ops, microbatch step."""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize prompt+output pairs and build a response_mask.

    Concatenates prompt and output tokens for each example, pads to the
    maximum combined length across the batch, then constructs:
      - input_ids: padded sequence with the final token removed (for next-token prediction input)
      - labels:    padded sequence with the first token removed (the prediction targets)
      - response_mask: boolean mask on `labels`, True only for output tokens

    Args:
        prompt_strs: list of prompt strings.
        output_strs: list of output strings.
        tokenizer: HuggingFace tokenizer.

    Returns:
        dict with keys:
            "input_ids":      (batch, max_len-1)
            "labels":         (batch, max_len-1)
            "response_mask":  (batch, max_len-1) bool
    """
    all_ids: list[list[int]] = []
    prompt_lens: list[int] = []
    output_lens: list[int] = []

    for prompt, output in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        o_ids = tokenizer.encode(output, add_special_tokens=False)
        all_ids.append(p_ids + o_ids)
        prompt_lens.append(len(p_ids))
        output_lens.append(len(o_ids))

    max_len = max(len(ids) for ids in all_ids)
    pad_id = tokenizer.pad_token_id

    # Pad right to max_len
    padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]
    padded_t = torch.tensor(padded, dtype=torch.long)  # (B, max_len)

    input_ids = padded_t[:, :-1]  # (B, max_len-1)
    labels = padded_t[:, 1:]       # (B, max_len-1)

    B, seq_len = input_ids.shape
    response_mask = torch.zeros(B, seq_len, dtype=torch.bool)
    for i, (p_len, o_len) in enumerate(zip(prompt_lens, output_lens)):
        # labels[j] = concat[j+1]; it's an output token when j+1 is in [p_len, p_len+o_len)
        # i.e. j in [p_len-1, p_len+o_len-1)
        start = p_len - 1
        end = p_len + o_len - 1
        response_mask[i, start:end] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: Tensor) -> Tensor:
    """Per-token entropy H(p) = -sum_x p(x) log p(x) over the vocabulary dimension.

    Args:
        logits: (batch, seq_len, vocab_size) unnormalized logits.

    Returns:
        (batch, seq_len) per-position entropy values.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Per-token conditional log-probabilities from a causal LM.

    Computes log p_theta(labels[t] | input_ids[0..t]) for each position t.
    Does NOT apply a response_mask — masking is handled by the caller.

    Args:
        model: HuggingFace causal LM (placed on the correct device).
        input_ids: (batch, seq_len) token IDs.
        labels: (batch, seq_len) shifted targets (i.e. input_ids[:, 1:] with padding).
        return_token_entropy: if True, also return per-token entropy.

    Returns:
        dict with:
            "log_probs":     (batch, seq_len) log p(label | prefix)
            "token_entropy": (batch, seq_len) optional, present only when return_token_entropy=True
    """
    logits = model(input_ids).logits  # (B, L, V)
    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, L, V)
    log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L)

    result: dict[str, Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    """Sum tensor elements (where mask==1) along a dimension, divide by normalize_constant.

    Args:
        tensor: tensor to sum.
        mask: same shape as tensor; only elements where mask==1 contribute.
        dim: dimension to sum along. If None, sum over all dimensions.
        normalize_constant: divisor.

    Returns:
        Tensor with the summed (and normalized) values.
    """
    masked = tensor * mask
    if dim is None:
        total = masked.sum()
    else:
        total = masked.sum(dim=dim)
    return total / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Single SFT microbatch forward-backward pass.

    Loss formula:
        loss = -mean_over_batch( sum_over_seq(log_probs * mask) / normalize_constant )
                / gradient_accumulation_steps

    This is equivalent to:
        loss = -sum(log_probs * mask) / (batch_size * normalize_constant * GA)

    Calls loss.backward() so gradients accumulate in model parameters.

    Args:
        policy_log_probs:         (batch, seq_len) per-token log-probs from SFT model.
        response_mask:            (batch, seq_len) 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        normalize_constant:       divisor for the per-sample sum (default 1.0).

    Returns:
        (loss_scalar, metadata_dict)
    """
    # Sum over seq per sample, normalize, average over batch, scale for GA
    per_sample = masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant)
    loss = -per_sample.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {"loss": loss.detach()}
