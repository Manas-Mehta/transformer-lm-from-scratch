from einops import rearrange
import torch 
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        
        # Weight shape: (out_features, in_features) -- same layout as nn.Linear
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Initialize with truncated normal
        sigma = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x):
        # y = Wx in math = x @ W^T in code
        return x @ self.weight.T
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        
        # Embedding matrix: (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # Initialize with truncated normal (different from Linear!)
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        # Fancy indexing: grab rows corresponding to each token ID
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps


        # Gain parameter: one per dimension, initialized to 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model) or any (..., d_model)

        in_dtype = x.dtype
        x = x.to(torch.float32)  # Upcast for numerical stability

        # RMS = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and apply gain
        result = (x / rms) * self.weight

        return result.to(in_dtype)  # Cast back to original dtype
    

def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """SwiGLU FFN: Gated Linear Unit with SiLU activation"""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        # Three linear projections (no bias)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # Gate
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # Down
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # Value

    def forward(self, x):
        # x shape: (..., d_model)

        # CRITICAL: Compute w1(x) ONCE and reuse to avoid redundant computation
        gate_input = self.w1(x)
        gate = gate_input * torch.sigmoid(gate_input)  # SiLU(W1*x) - inline for efficiency

        content = self.w3(x)  # W3*x

        return self.w2(gate * content)  # W2(SiLU(W1*x) * W3*x)


class SiLUFFN(nn.Module):
    """Simple SiLU feed-forward network (no gating). For swiglu_ablation.
    FFN(x) = W2 * SiLU(W1 * x)
    Use d_ff = 4 * d_model to approximately match SwiGLU parameter count.
    """
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()

        # Precompute the angles for all positions and all dimension pairs
        # theta_k = 1 / (Theta^(2k/d)) for k = 0, 1, ..., d/2-1
        k = torch.arange(0, d_k, 2, device=device).float()  # [0, 2, 4, ..., d_k-2]
        freqs = 1.0 / (theta ** (k / d_k))                   # shape: (d_k/2,)

        # positions: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device).float()

        # angles: outer product of positions and frequencies
        angles = torch.outer(positions, freqs)  # shape: (max_seq_len, d_k/2)

        # Precompute cos and sin (these are NOT learnable parameters)
        # Use register_buffer so they move to GPU with the model but aren't trained
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)

        # Slice cos/sin for the requested positions
        cos = self.cos[token_positions]  # type: ignore # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]  # type: ignore # (..., seq_len, d_k/2)

        # Split x into pairs of dimensions
        x1 = x[..., 0::2]   # even indices: x0, x2, x4, ...
        x2 = x[..., 1::2]   # odd indices:  x1, x3, x5, ...

        # Apply 2D rotation to each pair
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Interleave back: [r1_0, r2_0, r1_1, r2_1, ...]
        out = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return out.flatten(-2)  # merge the last two dims
    

def softmax(x, dim):
    # x: arbitrary tensor
    # dim: which dimension to normalize over

    # Step 1: subtract max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max

    # Step 2: exponentiate
    exp_x = torch.exp(x_shifted)

    # Step 3: normalize
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q: (batch_size, ..., seq_len, d_k)
    # K: (batch_size, ..., seq_len, d_k)
    # V: (batch_size, ..., seq_len, d_v)
    # mask: (seq_len, seq_len) boolean -- True = ATTEND, False = BLOCK


    d_k = Q.size(-1)

    # Step 1: compute attention scores
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    # scores shape: (..., seq_len, seq_len) 

    # Step 2: apply mask (set blocked positions to -inf)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Step 3: softmax over the key dimension
    attn_weights = softmax(scores, dim=-1)
    # attn_weights shape: (..., seq_len, seq_len)

    # Step 4: weighted sum of values
    return attn_weights @ V
    # output shape: (..., seq_len, d_v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Four projection matrices (all use our custom Linear, no bias)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE for positional encoding (optional - only created if parameters provided)
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device
            )
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        batch, seq_len, d_model = x.shape

        # Step 1: Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Step 2: Reshape to separate heads
        # (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
        Q = rearrange(Q, "b s (h dk) -> b h s dk", h=self.num_heads)
        K = rearrange(K, "b s (h dk) -> b h s dk", h=self.num_heads)
        V = rearrange(V, "b s (h dk) -> b h s dk", h=self.num_heads)

        # Step 3: Apply RoPE to Q and K (NOT V) - only if RoPE is enabled
        if self.rope is not None:
            if token_positions is None:
                # Default: sequential positions [0, 1, 2, ...]
                positions = torch.arange(seq_len, device=x.device)
            else:
                positions = token_positions
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)

        # Step 4: Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # Step 5: Compute attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        # attn_output: (batch, num_heads, seq, d_k)

        # Step 6: Merge heads back
        attn_output = rearrange(attn_output, "b h s dk -> b s (h dk)")

        # Step 7: Final projection
        return self.output_proj(attn_output)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=None,
                 use_rmsnorm=True, pre_norm=True, use_swiglu=True,
                 device=None, dtype=None):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.pre_norm = pre_norm

        # Always create norms (so state_dict keys are consistent for load_state_dict)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        if use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SiLUFFN(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        if not self.use_rmsnorm:
            # No normalization (layer_norm_ablation)
            x = x + self.attn(x)
            x = x + self.ffn(x)
        elif self.pre_norm:
            # Pre-norm (default): Norm -> Sublayer -> Add
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            # Post-norm: Sublayer -> Add -> Norm
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff,
                 rope_theta=10000.0, use_rmsnorm=True, pre_norm=True, use_rope=True, use_swiglu=True,
                 device=None, dtype=None):
        super().__init__()
        self.context_length = context_length
        # Final norm only needed for pre-norm with RMSNorm
        # Post-norm blocks already normalize output; no-norm has no norms at all
        self._use_final_norm = use_rmsnorm and pre_norm

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # If use_rope is False, pass theta=None to disable RoPE in attention
        block_theta = rope_theta if use_rope else None
        block_max_seq_len = context_length if use_rope else None

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff,
                           max_seq_len=block_max_seq_len, theta=block_theta,
                           use_rmsnorm=use_rmsnorm, pre_norm=pre_norm, use_swiglu=use_swiglu,
                           device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        # Always create ln_final so state_dict keys are consistent
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len) -- LongTensor

        x = self.token_embeddings(token_ids)   # (batch, seq, d_model)

        for layer in self.layers:
            x = layer(x)

        if self._use_final_norm:
            x = self.ln_final(x)               # final normalization (needed for pre-norm!)

        logits = self.lm_head(x)               # (batch, seq, vocab_size)

        return logits
    

def cross_entropy(inputs, targets):
    # inputs: (batch_size, vocab_size) for adapter tests
    #         Generally can be (..., vocab_size) with arbitrary batch dims
    # targets: (batch_size,) or (...) -- integer indices

    # Step 1: Extract the input logit for the correct class
    # gather picks inputs[..., targets[...]] for each position
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # shape: (batch_size,) or (...)

    # Step 2: Compute log-sum-exp with the max trick
    max_logits = inputs.max(dim=-1, keepdim=True).values  # (..., 1)
    shifted = inputs - max_logits                          # (..., vocab_size)
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.exp(shifted).sum(dim=-1))
    # shape: (batch_size,) or (...)

    # Step 3: Loss per position
    loss_per_position = -target_logits + log_sum_exp       # (batch_size,) or (...)

    # Step 4: Average over ALL positions (the assignment says "average across the batch")
    return loss_per_position.mean()



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data    # the gradient tensor

                # Get or initialize state for this parameter
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0                            # step counter
                    state["m"] = torch.zeros_like(p.data)     # first moment
                    state["v"] = torch.zeros_like(p.data)     # second moment

                # Increment step
                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                # Update moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)       # m = beta1*m + (1-beta1)*g
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v = beta2*v + (1-beta2)*g^2

                # Bias-corrected learning rate
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # Update parameters
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)
                # p = p - alpha_t * m / (sqrt(v) + eps)

                # Weight decay (decoupled -- applied to theta directly, NOT through gradient)
                p.data.add_(p.data, alpha=-lr * weight_decay)
                # p = p - lr * lambda * p

        return loss
    


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    """
    it: current iteration/step (integer)
    max_learning_rate: maximum (peak) learning rate (alpha_max in theory)
    min_learning_rate: minimum (final) learning rate (alpha_min in theory)
    warmup_iters: number of warmup iterations (T_w in theory)
    cosine_cycle_iters: total iterations where cosine schedule applies (T_c in theory)

    Returns: learning rate at iteration it
    """
    if it < warmup_iters:
        # Linear warmup: 0 -> max_learning_rate over warmup_iters steps
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine annealing: max_learning_rate -> min_learning_rate
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)   # 0 at warmup_iters, 1 at cosine_cycle_iters
        cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))  # 1 -> 0
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)
    else:
        # Post-annealing: constant min_learning_rate
        return min_learning_rate
    

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    """
    parameters: list of nn.Parameter (or iterable)
    max_l2_norm: maximum allowed L2 norm
    eps: small value for numerical stability (default 1e-6 per assignment)

    Modifies gradients IN PLACE.
    """
    # Step 1: Compute total L2 norm across ALL parameters
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm_sq += p.grad.data.pow(2).sum().item()
    total_norm = total_norm_sq ** 0.5

    # Step 2: If norm exceeds max, scale all gradients down
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)


import numpy as np

def get_batch(dataset, batch_size, context_length, device):
    """
    dataset: numpy array of token IDs (1D, dtype uint16) - memory-mapped!
    batch_size: number of sequences per batch
    context_length: length of each sequence
    device: 'cpu', 'cuda:0', or 'mps'

    Returns: (inputs, targets) both shape (batch_size, context_length)
    """
    n = len(dataset)

    # Random starting indices
    # Max start index: n - context_length - 1 (need context_length + 1 tokens)
    max_start = n - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    # Build input and target arrays
    inputs = np.stack([dataset[s : s + context_length] for s in starts])
    targets = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])

    # Convert to PyTorch tensors on the right device
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets


def save_checkpoint(model, optimizer, iteration, out):
    """
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: file path or file-like object
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    """
    src: file path or file-like object
    model: torch.nn.Module (to load weights into)
    optimizer: torch.optim.Optimizer (to load state into)

    Returns: iteration number
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def decode(model, tokenizer, prompt, max_tokens=256, temperature=0.7, top_p=0.9, device="mps"):
    """
    Generate text from the model given a prompt.

    Args:
        model: TransformerLM model (already loaded, in eval mode)
        tokenizer: Tokenizer instance (for encode/decode)
        prompt: Text prompt to start generation from
        max_tokens: Maximum number of NEW tokens to generate
        temperature: Temperature for softmax scaling (lower = more deterministic)
        top_p: Top-p (nucleus) sampling threshold (1.0 = no filtering)
        device: Device string ('mps', 'cuda', 'cpu')

    Returns:
        Generated text string (prompt + completion)
    """
    model.eval()

    # Encode the prompt into token IDs
    token_ids = tokenizer.encode(prompt)

    # Convert to tensor with batch dimension: shape (1, seq_len)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Get <|endoftext|> token ID for stopping
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    # Autoregressive generation loop
    with torch.no_grad():
        for _ in range(max_tokens):
            # Truncate to context_length if sequence is too long (sliding window)
            if x.size(1) > model.context_length:
                x_input = x[:, -model.context_length:]
            else:
                x_input = x

            # Forward pass: logits shape (1, seq_len, vocab_size)
            logits = model(x_input)

            # Get logits for the LAST position: shape (vocab_size,)
            next_token_logits = logits[0, -1, :]

            # Apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Convert to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                # Sort probabilities descending
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                # Cumulative sum
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Mask tokens where cumsum (excluding current) exceeds top_p
                sorted_mask = cumulative_probs - sorted_probs > top_p

                # Zero out tail tokens
                sorted_probs[sorted_mask] = 0.0

                # Renormalize
                sorted_probs = sorted_probs / sorted_probs.sum()

                # Scatter back to original order
                probs = torch.zeros_like(probs)
                probs.scatter_(0, sorted_indices, sorted_probs)

            # Sample next token
            next_token = torch.multinomial(probs.unsqueeze(0), num_samples=1)

            # Append to sequence
            x = torch.cat([x, next_token], dim=1)

            # Stop if EOS token
            if next_token.item() == eos_token_id:
                break

    # Decode full sequence back to text
    generated_ids = x[0].tolist()
    return tokenizer.decode(generated_ids)