"""
FlashAttention-2 Implementations

Contains:
  - FlashAttentionPyTorch: Pure PyTorch implementation (Section 1.3.2a)
  - FlashAttentionTriton: Triton kernel implementation (Section 1.3.2b)
  - flash_backward: Backward pass using PyTorch + torch.compile (Section 1.3.3)
"""

import math
import torch
from einops import einsum


# ──────────────────────────────────────────────────────────────────────────
#  (a) Pure PyTorch FlashAttention-2 Forward Pass
# ──────────────────────────────────────────────────────────────────────────

class FlashAttentionPyTorch(torch.autograd.Function):
    """FlashAttention-2 forward pass implemented in pure PyTorch (no Triton).

    Uses tiled computation with online softmax to avoid materializing the
    full seq_len x seq_len attention matrix.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Args:
            Q: (batch, N_q, d)
            K: (batch, N_k, d)
            V: (batch, N_k, d)
            is_causal: bool — apply causal masking
        Returns:
            O: (batch, N_q, d)
        """
        batch, N_q, d = Q.shape
        N_k = K.shape[1]
        scale = 1.0 / math.sqrt(d)

        # Choose tile sizes (powers of 2, at least 16)
        B_q = min(64, N_q)
        B_k = min(64, N_k)

        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)

        # Output buffers
        O = torch.zeros_like(Q)
        L = torch.empty(batch, N_q, device=Q.device, dtype=Q.dtype)

        for i in range(T_q):
            q_start = i * B_q
            q_end = min(q_start + B_q, N_q)
            Q_i = Q[:, q_start:q_end, :]  # (batch, B_q_actual, d)

            B_q_actual = q_end - q_start

            # Running accumulators
            O_i = torch.zeros(batch, B_q_actual, d, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(batch, B_q_actual, device=Q.device, dtype=Q.dtype)
            m_i = torch.full((batch, B_q_actual), float('-inf'), device=Q.device, dtype=Q.dtype)

            for j in range(T_k):
                k_start = j * B_k
                k_end = min(k_start + B_k, N_k)
                K_j = K[:, k_start:k_end, :]  # (batch, B_k_actual, d)
                V_j = V[:, k_start:k_end, :]

                # Compute attention scores: (batch, B_q_actual, B_k_actual)
                S_ij = torch.bmm(Q_i, K_j.transpose(-2, -1)) * scale

                # Apply causal mask if needed
                if is_causal:
                    # Query indices: q_start..q_end, Key indices: k_start..k_end
                    q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                    k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                    causal_mask = q_idx >= k_idx  # (B_q_actual, B_k_actual)
                    S_ij = S_ij.masked_fill(~causal_mask.unsqueeze(0), -1e6)

                # Update running max: (batch, B_q_actual)
                m_new = torch.max(m_i, S_ij.max(dim=-1).values)

                # Compute unnormalized attention weights
                P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))  # (batch, B_q_actual, B_k_actual)

                # Rescale factor for previous accumulations
                alpha = torch.exp(m_i - m_new)  # (batch, B_q_actual)

                # Update running denominator
                l_i = alpha * l_i + P_ij.sum(dim=-1)

                # Rescale previous O and add new contribution
                O_i = alpha.unsqueeze(-1) * O_i + torch.bmm(P_ij, V_j)

                m_i = m_new

            # Final normalization
            O_i = O_i / l_i.unsqueeze(-1)

            # Logsumexp for backward
            L_i = m_i + torch.log(l_i)

            # Write to output
            O[:, q_start:q_end, :] = O_i
            L[:, q_start:q_end] = L_i

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = _flash_backward(Q, K, V, O, L, dO, is_causal)
        return dQ, dK, dV, None  # None for is_causal


# ──────────────────────────────────────────────────────────────────────────
#  Backward pass (Section 1.3.3 — flash_backward)
#  Uses PyTorch + torch.compile, following Equations 13-19 from the PDF.
# ──────────────────────────────────────────────────────────────────────────

def _flash_backward_impl(Q, K, V, O, L, dO, is_causal):
    """Compute dQ, dK, dV using recomputation (Equations 13-19).

    This avoids storing the full attention matrix P by recomputing it from Q, K, L.
    """
    batch, N_q, d = Q.shape
    N_k = K.shape[1]
    scale = 1.0 / math.sqrt(d)

    # Eq 13: S = QK^T / sqrt(d)
    S = torch.bmm(Q, K.transpose(-2, -1)) * scale  # (batch, N_q, N_k)

    # Apply causal mask
    if is_causal:
        q_idx = torch.arange(N_q, device=Q.device).unsqueeze(1)
        k_idx = torch.arange(N_k, device=Q.device).unsqueeze(0)
        causal_mask = q_idx >= k_idx
        S = S.masked_fill(~causal_mask.unsqueeze(0), -1e6)

    # Eq 14: P = exp(S - L)
    P = torch.exp(S - L.unsqueeze(-1))  # (batch, N_q, N_k)

    # Eq 15: dV = P^T @ dO
    dV = torch.bmm(P.transpose(-2, -1), dO)  # (batch, N_k, d)

    # Eq 16: dP = dO @ V^T
    dP = torch.bmm(dO, V.transpose(-2, -1))  # (batch, N_q, N_k)

    # Eq 17: D = rowsum(O * dO)
    D = (O * dO).sum(dim=-1)  # (batch, N_q)

    # Eq 17: dS = P * (dP - D)
    dS = P * (dP - D.unsqueeze(-1))  # (batch, N_q, N_k)

    # Eq 18: dQ = dS @ K / sqrt(d)
    dQ = torch.bmm(dS, K) * scale  # (batch, N_q, d)

    # Eq 19: dK = dS^T @ Q / sqrt(d)
    dK = torch.bmm(dS.transpose(-2, -1), Q) * scale  # (batch, N_k, d)

    return dQ, dK, dV


# Compile the backward for speed
_flash_backward = torch.compile(_flash_backward_impl)


# ──────────────────────────────────────────────────────────────────────────
#  (b) Triton FlashAttention-2 Forward Pass
# ──────────────────────────────────────────────────────────────────────────

try:
    import triton
    import triton.language as tl

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        is_causal: tl.constexpr,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Offset pointers by batch
        Q_ptr = Q_ptr + batch_index * stride_qb
        K_ptr = K_ptr + batch_index * stride_kb
        V_ptr = V_ptr + batch_index * stride_vb
        O_ptr = O_ptr + batch_index * stride_ob
        L_ptr = L_ptr + batch_index * stride_lb

        # Q block pointer for this query tile
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        # Load Q tile
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)

        # Initialize accumulators in FP32
        O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l_acc = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m_acc = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)

        # K, V block pointers (start at first key tile)
        K_block_ptr = tl.make_block_ptr(
            K_ptr,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            V_ptr,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        # Iterate over key tiles
        n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
        for j in range(n_key_tiles):
            K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)
            V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)

            # S = Q @ K^T * scale — (Q_TILE_SIZE, K_TILE_SIZE)
            S_tile = tl.dot(Q_tile, tl.trans(K_tile)) * scale

            # Apply causal mask
            if is_causal:
                q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                causal_mask = q_offsets[:, None] >= k_offsets[None, :]
                S_tile = tl.where(causal_mask, S_tile, float('-inf'))

            # Update running max
            m_new = tl.maximum(m_acc, tl.max(S_tile, axis=1))

            # Compute P = exp(S - m_new)
            P_tile = tl.exp(S_tile - m_new[:, None])

            # Rescale factor
            alpha = tl.exp(m_acc - m_new)

            # Update running denominator
            l_acc = alpha * l_acc + tl.sum(P_tile, axis=1)

            # Rescale previous O and add new contribution
            O_acc = alpha[:, None] * O_acc + tl.dot(P_tile.to(V_tile.dtype), V_tile)

            m_acc = m_new

            # Advance K, V pointers
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # Final normalization
        O_acc = O_acc / l_acc[:, None]
        L_vals = m_acc + tl.log(l_acc)

        # Write O (cast back to input dtype)
        O_block_ptr = tl.make_block_ptr(
            O_ptr,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        tl.store(O_block_ptr, O_acc.to(Q_tile.dtype), boundary_check=(0, 1))

        # Write L
        L_block_ptr = tl.make_block_ptr(
            L_ptr,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        tl.store(L_block_ptr, L_vals, boundary_check=(0,))


    class FlashAttentionTriton(torch.autograd.Function):
        """FlashAttention-2 using Triton kernel for forward, torch.compile for backward."""

        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            batch, N_q, d = Q.shape
            N_k = K.shape[1]

            O = torch.empty_like(Q)
            L = torch.empty(batch, N_q, device=Q.device, dtype=torch.float32)

            # Tile sizes — tune these as needed
            Q_TILE_SIZE = min(64, N_q)
            K_TILE_SIZE = min(64, N_k)
            scale = 1.0 / math.sqrt(d)

            grid = (triton.cdiv(N_q, Q_TILE_SIZE), batch)

            flash_fwd_kernel[grid](
                Q, K, V, O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                N_q, N_k,
                scale,
                is_causal=is_causal,
                D=d,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
            )

            ctx.save_for_backward(L, Q, K, V, O)
            ctx.is_causal = is_causal
            return O

        @staticmethod
        def backward(ctx, dO):
            L, Q, K, V, O = ctx.saved_tensors
            is_causal = ctx.is_causal
            dQ, dK, dV = _flash_backward(Q, K, V, O, L, dO, is_causal)
            return dQ, dK, dV, None

    TRITON_AVAILABLE = True

except ImportError:
    TRITON_AVAILABLE = False

    class FlashAttentionTriton(torch.autograd.Function):
        """Placeholder — Triton not available on this platform."""
        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            raise RuntimeError("Triton is not available. Run on Linux with a GPU.")
        @staticmethod
        def backward(ctx, dO):
            raise RuntimeError("Triton is not available.")
